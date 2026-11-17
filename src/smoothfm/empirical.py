"""The empirical velocity v_N and the empirical path rho_N of a training set (main.tex
eq:empirical, Sec. 2).

With x_t = (1 - t) z + t y, z ~ gamma (density K, the truncated latent), y ~ pi_N (the
empirical measure of the training points x_1..x_N), sigma = 1 - t and
K_sigma(u) = sigma^{-D} K(u / sigma):

    rho_N(x, t) = (1/N) sum_i K_sigma(x - t x_i),
    w_i(x, t)   = K_sigma(x - t x_i) / sum_j K_sigma(x - t x_j),
    m_N = sum_i w_i x_i,   v_N = (m_N - x) / sigma.

v_N is the minimizer of the empirical loss eq:efm over all fields, and its flow memorizes
the training data. rho_N(., t) is a mixture of box-truncated Gaussians, so it vanishes
outside the union of the boxes Omega_i(t) = t x_i + sigma Omega, where the weights are
undefined (the paper leaves v_N undefined there). The code then uses the untruncated
Gaussian weights, which the sampler of the exact v_N needs when a trajectory
leaves the boxes; ``n_outside_support`` counts those points.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .latent import as_points

__all__ = ["EmpiricalField", "EmpiricalFlow", "StoppedLaw"]

_LOG_SQRT2PI = 0.5 * np.log(2.0 * np.pi)


class EmpiricalField:
    """v_N, m_N, w_i and rho_N of the training points ``samples`` (N, D) under the truncated
    latent; evaluation points (M, D) are processed in blocks of ``block`` rows."""

    def __init__(self, samples, latent, block: int = 4096):
        self.latent = latent
        self.samples = as_points(samples, latent.dim).reshape(-1, latent.dim)
        self.block = int(block)
        self.n_outside_support = 0

    @property
    def dim(self):
        return self.samples.shape[1]

    @property
    def n(self):
        return self.samples.shape[0]

    def _check(self, x, t):
        t = float(t)
        if not 0.0 <= t < 1.0:
            raise ValueError("t must lie in [0, 1)")
        return as_points(x, self.dim).reshape(-1, self.dim), t

    def log_kernels(self, x, t):
        """log K_sigma(x_p - t x_i) (M, N), -inf outside the component box."""
        x, t = self._check(x, t)
        s = 1.0 - t
        u = (x[:, None, :] - t * self.samples[None]) / s
        return self.latent.logkernel(u).sum(-1) - self.dim * np.log(s)

    def weights(self, x, t):
        """w_i(x, t) (M, N), rows sum to 1; untruncated Gaussian weights where rho_N = 0."""
        x, t = self._check(x, t)
        L = self.log_kernels(x, t)
        lse = logsumexp(L, axis=1, keepdims=True)
        out = ~np.isfinite(lse[:, 0])
        if out.any():
            self.n_outside_support += int(out.sum())
            s = 1.0 - t
            u = (x[out][:, None, :] - t * self.samples[None]) / s
            Lg = -0.5 * np.sum(u * u, -1)
            L[out] = Lg
            lse[out] = logsumexp(Lg, axis=1, keepdims=True)
        return np.exp(L - lse)

    def _blocks(self, x, t, fn):
        x, t = self._check(x, t)
        out = [fn(x[k:k + self.block], t) for k in range(0, x.shape[0], self.block)]
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim))

    def posterior_mean(self, x, t):
        """m_N(x, t) (M, D)."""
        return self._blocks(x, t, lambda xb, tb: self.weights(xb, tb) @ self.samples)

    def velocity(self, x, t):
        """v_N(x, t) = (m_N - x) / (1 - t) (eq:empirical)."""
        xx, t = self._check(x, t)
        return (self.posterior_mean(xx, t) - xx) / (1.0 - t)

    def log_density(self, x, t):
        """log rho_N(x, t) (M,), -inf outside the support."""
        def fn(xb, tb):
            return (logsumexp(self.log_kernels(xb, tb), axis=1) - np.log(self.n))[:, None]
        return self._blocks(x, t, fn)[:, 0]


class EmpiricalFlow:
    """The exact v_N as a sampler velocity (``v(x, t)``, ``v.div(x, t)``): v_N and its
    classical divergence off the jump set,

        div v_N = ((t / sigma^2) tr Cov_w - D) / sigma

    (the delta terms on the faces of the component boxes, where v_N jumps, are omitted;
    every exact flow logs max |log f - log rho_N(x_end, t_end)|, the measured effect). v and div share the weights, so the last (x, t) is cached; ``nfev`` counts
    weight evaluations, ``n_outside_support`` the points with rho_N = 0 (fallback weights)."""

    def __init__(self, samples, latent, block: int = 4096):
        self.field = EmpiricalField(samples, latent, block)
        self.latent = latent
        self.dim = latent.dim
        self.nfev = 0
        self._key = self._val = None

    @property
    def n_outside_support(self):
        return self.field.n_outside_support

    def _eval(self, x, t):
        x, t = self.field._check(x, t)
        key = (t, x.shape, hash(x.tobytes()))
        if key != self._key:
            self.nfev += 1
            Y = self.field.samples
            D = self.dim
            sig = 1.0 - t
            m = np.empty_like(x)
            tr = np.empty(x.shape[0])
            for k in range(0, x.shape[0], self.field.block):
                w = self.field.weights(x[k:k + self.field.block], t)
                mk = w @ Y
                m[k:k + w.shape[0]] = mk
                # two-pass trace: E_w y^2 - m^2 has error ~ eps |y|^2, amplified by t/sigma^2
                d2 = np.sum((Y[None] - mk[:, None, :]) ** 2, axis=-1)
                tr[k:k + w.shape[0]] = np.sum(w * d2, axis=1)
            self._key, self._val = key, ((m - x) / sig, (t / sig**2 * tr - D) / sig)
        return self._val

    def __call__(self, x, t):
        return self._eval(x, t)[0]

    def div(self, x, t):
        return self._eval(x, t)[1]


class StoppedLaw:
    """rho_N(., t), the law of x_t = t y + sigma z with y ~ pi_N, z ~ gamma: the equal-weight
    mixture of the Gaussians N(t x_i, sigma^2 I) truncated to the boxes t x_i + sigma Omega.
    At t = t_end it is the law of the stopped exact flow of v_N (the ``tv_law`` rows), and
    x_t / t is distributed as the KDE of the training points with kernel K and bandwidth
    (1 - t)/t (Sec. 2)."""

    def __init__(self, samples, t, latent):
        self.latent = latent
        self.t = float(t)
        self.sigma = 1.0 - self.t
        self.samples = as_points(samples, latent.dim).reshape(-1, latent.dim)
        self.centres = self.t * self.samples
        self.dim = latent.dim
        self._field = EmpiricalField(self.samples, latent)

    def logpdf(self, x):
        return self._field.log_density(x, self.t)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def sample(self, n, rng=None):
        """Component index rng.integers(0, N, n) first, then the latent draws."""
        rng = np.random.default_rng(rng)
        k = rng.integers(0, self.samples.shape[0], int(n))
        return self.centres[k] + self.sigma * self.latent.sample(int(n), rng)

    def cell_mass(self, edges):
        """Mass of rho_N in the cells of a tensor grid: 1D edges (K + 1,) -> (K,);
        2D a pair of edge arrays -> (Kx, Ky), from the latent CDF per axis."""
        if self.dim == 1:
            e = np.asarray(edges, float).ravel()
            F = self.latent.cdf((e[:, None] - self.centres[None, :, 0]) / self.sigma)
            return np.diff(F, axis=0).mean(1)
        ex, ey = (np.asarray(e, float).ravel() for e in edges)
        Fx = np.diff(self.latent.cdf((ex[:, None] - self.centres[None, :, 0]) / self.sigma), axis=0)
        Fy = np.diff(self.latent.cdf((ey[:, None] - self.centres[None, :, 1]) / self.sigma), axis=0)
        return Fx @ Fy.T / self.samples.shape[0]

    def edges(self):
        """1D component box ends t x_i -+ sigma L."""
        c = self.centres[:, 0]
        return np.concatenate([c - self.sigma * self.latent.L, c + self.sigma * self.latent.L])

    def cdf(self, x):
        """1D: F(x) = (1/N) sum_i F_L((x - t x_i) / sigma)."""
        x = np.asarray(x, float).ravel()
        return self.latent.cdf((x[:, None] - self.centres[None, :, 0]) / self.sigma).mean(1)

    def tv_to(self, other, n=None, rng=None):
        """TV(rho, other) for another StoppedLaw (same latent and t).

        1D, exact up to rounding: g = rho - rho' is continuous between the component box ends;
        on cells of width <= sigma/4 between them the sign changes of g are located by
        bisection, and TV = sum over the pieces with g > 0 of the differences of the exact
        CDFs, (F - F')(b) - (F - F')(a). Returns (tv, 0.0).
        2D: Monte Carlo, n draws of rho (default 4096, rng 0), mean (1 - rho'/rho)_+ and its standard error; (tv, se)."""
        if self.dim == 1:
            sig = self.sigma
            br = np.unique(np.concatenate([self.edges(), other.edges()]))
            a, b = br[:-1], br[1:]
            g = lambda x: self.pdf(x[:, None]) - other.pdf(x[:, None])
            mid = 0.5 * (a + b)
            live = (self.pdf(mid[:, None]) > 0) | (other.pdf(mid[:, None]) > 0)
            a, b = a[live], b[live]
            k = np.maximum(1, np.ceil((b - a) / (0.25 * sig)).astype(int))
            seg = np.repeat(np.arange(a.size), k)
            j = np.arange(seg.size) - np.repeat(np.cumsum(k) - k, k)
            lo = a[seg] + (b - a)[seg] * j / k[seg]
            hi = a[seg] + (b - a)[seg] * (j + 1) / k[seg]
            # one-sided values at the cell ends (g jumps only at box ends, the subinterval ends)
            nudge = 1e-9 * (hi - lo)
            glo, ghi = g(lo + nudge), g(hi - nudge)
            ch = np.flatnonzero(np.sign(glo) * np.sign(ghi) < 0)
            l, r, gl = lo[ch].copy(), hi[ch].copy(), glo[ch]
            for _ in range(60):
                mm = 0.5 * (l + r)
                gm = g(mm)
                left = np.sign(gm) == np.sign(gl)
                l, r = np.where(left, mm, l), np.where(left, r, mm)
            cuts = np.unique(np.concatenate([lo, hi, 0.5 * (l + r)]))
            ca, cb = cuts[:-1], cuts[1:]
            pos = g(0.5 * (ca + cb)) > 0
            D = self.cdf(cuts) - other.cdf(cuts)
            return float(np.sum((D[1:] - D[:-1])[pos])), 0.0
        x = self.sample(4096 if n is None else int(n), 0 if rng is None else rng)
        with np.errstate(over="ignore", invalid="ignore"):
            r = np.maximum(1.0 - np.exp(other.logpdf(x) - self.logpdf(x)), 0.0)
        return float(r.mean()), float(r.std(ddof=1) / np.sqrt(r.size))
