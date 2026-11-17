"""The targets pi of the experiments and their population quantities (main.tex sec:data,
eq:velocity, eq:oracle).

Convention: x_t = (1 - t) z + t y with z ~ gamma (density K, the latent N(0, I)
conditioned on Omega = [-L, L]^D) and y ~ pi independent; sigma = 1 - t. The population
velocity of eq:velocity is v_pi(x, t) = (m(x, t) - x) / sigma with the posterior mean
m = E[y | x_t = x]. With the truncated latent the posterior is the untruncated one
restricted to the box

    B = prod_d [l_d, u_d],   l_d = (x_d - sigma L) / t,   u_d = (x_d + sigma L) / t,

    q(y | x) ∝ pi(y) phi_D((x - t y) / sigma) 1_B(y),
    rho(x, t) = Z_L^{-D} int_B pi(y) sigma^{-D} phi_D((x - t y) / sigma) dy,

with rho(., t) the population path (the code writes p_t and log p_t for rho(., t) and its
logarithm). Where rho(x, t) = 0 the posterior is undefined: NaN, and a RuntimeWarning with
the count (``Target.n_undefined`` accumulates it). At t = 0, m = E[y] and C = Cov[y] for x in
Omega.

Jacobian trace of the 1D targets (Leibniz on the moving faces of B,
d l_d/d x_d = d u_d/d x_d = 1/t):

    d m_d / d x_d = (t / sigma^2) C_dd + (1/t) [(u_d - m_d) q_d(u_d) + (m_d - l_d) q_d(l_d)],

q_d the marginal posterior density of y_d; div v_pi = (sum_d d m_d / d x_d - D) / sigma.

Classes: BetaMixture1D (TrimodalBeta, Inhomogeneous), NoisyCircle, Checkerboard,
EightGaussians, TwoSpirals (the six targets of sec:data); PopulationVelocity (v_pi with the sampler interface, the oracle's reference and the
time-integration control of sec:setup).
Paper: PopulationVelocity is the reference of the oracle rule, of the velocity error in
Figure 1 (fig:hero) and of the population TV floors of sec:setup (experiments/compute/floors.py).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import betainc, betaln, i0e, i1e, logsumexp

from .latent import TruncatedGaussian, as_points, truncnorm_moments

__all__ = ["Target", "TrimodalBeta", "Inhomogeneous", "NoisyCircle", "Checkerboard",
           "EightGaussians", "TwoSpirals", "PopulationVelocity"]

_GL8, _WGL8 = leggauss(8)
_LOG2PI = np.log(2.0 * np.pi)
_LOG_SQRT2PI = 0.5 * _LOG2PI


def _log_ZL(L):
    return TruncatedGaussian(1, L)._logZ


class Target:
    """Interface of a target pi on R^dim: ``sample``, ``logpdf``, ``mean``, ``cov``, and the
    posterior quantities of the truncated-latent interpolant: ``posterior(x, t, L)`` -> (m, C,
    log rho(x, t)) and, for the 1D targets, ``jacobian_trace(x, t, L)`` (tr grad_x m, face
    terms included)."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n_undefined = 0

    def sample(self, n, rng=None):
        raise NotImplementedError

    def logpdf(self, x):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def cov(self):
        raise NotImplementedError

    def posterior(self, x, t, L):
        """(m (n, D), C (n, D, D), log rho(x, t) (n,)): posterior mean and covariance of y given
        x_t = x, and the log density of the population path (module docstring)."""
        raise NotImplementedError

    def jacobian_trace(self, x, t, L):
        """tr grad_x m(x, t) (n,), with the face terms of the truncated posterior."""
        raise NotImplementedError(f"{type(self).__name__} has no Jacobian trace")

    # -- shared pieces ---------------------------------------------------------------
    def _at_zero(self, x, L):
        """t = 0: m = E[y], C = Cov[y], log rho(x, 0) = log K(x) for x in Omega (NaN outside)."""
        n, D = x.shape
        m = np.broadcast_to(self.mean, (n, D)).copy()
        C = np.broadcast_to(self.cov, (n, D, D)).copy()
        logp = TruncatedGaussian(D, L).logpdf(x)
        return self._undefined(m, C, logp, ~np.isfinite(logp), 0.0)

    def _undefined(self, m, C, logp, bad, t):
        """NaN where the posterior is undefined (rho(x, t) = 0), with a counted RuntimeWarning."""
        k = int(np.count_nonzero(bad))
        if k:
            m[bad] = np.nan
            C[bad] = np.nan
            logp[bad] = -np.inf
            self.n_undefined += k
            warnings.warn(f"{type(self).__name__}: p_t(x) = 0 at {k} of {bad.size} points "
                          f"(t = {t:.6g}); posterior set to NaN", RuntimeWarning, stacklevel=3)
        return m, C, logp

    @staticmethod
    def _box(x, t, L):
        sig = 1.0 - t
        return (x - sig * L) / t, (x + sig * L) / t


def _faces_trace(t, C, m, l, u, ql, qu):
    """sum_d (t / sigma^2) C_dd + (1/t) [(u_d - m_d) q_d(u_d) + (m_d - l_d) q_d(l_d)]."""
    sig = 1.0 - t
    diag = np.diagonal(C, axis1=1, axis2=2)
    face = np.where(qu > 0, (u - m) * qu, 0.0) + np.where(ql > 0, (m - l) * ql, 0.0)
    return np.sum(t / sig**2 * diag + face / t, axis=1)


# ---------------------------------------------------------------- 1D Beta mixtures

class BetaMixture1D(Target):
    """pi(y) = sum_j w_j Beta(a_j, b_j) mapped affinely from [0, 1] to [-1, 1] (sec:data).

    Posterior: composite 8-point Gauss-Legendre in y on [-1, 1] ∩ [l, u], panels
    at the y-grid (``n_panels`` uniform panels of [-1, 1]) and at the images
    y = (x - sigma z) / t of the latent breaks linspace(-L, L, 49), so l and u are panel edges.
    Sampling inverts the exact CDF by bisection (60 steps): ``sample(n, rng)`` consumes
    rng.uniform(size=n).
    """

    chunk = 4000

    def __init__(self, w, a, b, n_panels: int = 64):
        super().__init__(1)
        self.w = np.asarray(w, float) / np.sum(w)
        self.a = np.asarray(a, float)
        self.b = np.asarray(b, float)
        self.logc = -betaln(self.a, self.b) - np.log(2.0)
        self.support = (-1.0, 1.0)
        self.n_panels = int(n_panels)

    def _pdf1(self, y):
        y = np.asarray(y, float)
        u = np.clip((y + 1) / 2, 0.0, 1.0)
        out = np.zeros_like(y)
        with np.errstate(divide="ignore"):
            lu, l1u = np.log(u), np.log1p(-u)
        for w, a, b, c in zip(self.w, self.a, self.b, self.logc):
            out += w * np.exp(c + (a - 1) * lu + (b - 1) * l1u)
        return np.where((y > -1) & (y < 1), out, 0.0)

    def pdf(self, x):
        return self._pdf1(as_points(x, 1)[..., 0])

    def logpdf(self, x):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x))

    def cdf(self, y):
        """P(Y <= y) = sum_j w_j I_u(a_j, b_j), u = (y + 1)/2."""
        u = np.clip((np.asarray(y, float) + 1) / 2, 0.0, 1.0)
        return sum(w * betainc(a, b, u) for w, a, b in zip(self.w, self.a, self.b))

    def ppf(self, q, iters: int = 60):
        """Quantile by bisection on the CDF (interval width 2^{1 - iters})."""
        q = np.asarray(q, float)
        lo, hi = -np.ones_like(q), np.ones_like(q)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < q
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        return self.ppf(rng.uniform(size=int(n)))[:, None]

    @property
    def mean(self):
        """E[y] = sum_j w_j (2 a_j / (a_j + b_j) - 1)."""
        mu = self.a / (self.a + self.b)
        return np.array([np.sum(self.w * (2 * mu - 1))])

    @property
    def cov(self):
        """Var[y] = sum_j w_j (4 Var_j + (2 mu_j - 1)^2) - E[y]^2."""
        a, b = self.a, self.b
        mu = a / (a + b)
        v = a * b / ((a + b) ** 2 * (a + b + 1))
        m2 = np.sum(self.w * (4 * v + (2 * mu - 1) ** 2))
        return np.array([[m2 - self.mean[0] ** 2]])

    def _rule(self, xd, t, L, lat):
        """Nodes Y (n, Q) and log weights log(GL weight k_L(z) / sigma), z = (x - t Y)/sigma."""
        sig = 1.0 - t
        yb = np.linspace(-1.0, 1.0, self.n_panels + 1)
        zb = np.linspace(-L, L, 49)
        xc = xd[:, None]
        lo = np.maximum(-1.0, (xc - sig * zb[-1:]) / t)
        hi = np.minimum(1.0, (xc - sig * zb[:1]) / t)
        br = np.concatenate([np.broadcast_to(yb, (xc.shape[0], yb.size)), (xc - sig * zb) / t],
                            axis=1)
        # lo > hi (empty interval) clips every edge to hi: zero-width panels, weight 0
        br = np.sort(np.minimum(np.maximum(br, lo), hi), axis=1)
        a, b = br[:, :-1], br[:, 1:]
        half = 0.5 * (b - a)
        Y = (a[..., None] + half[..., None] * (_GL8 + 1)).reshape(xc.shape[0], -1)
        W = (half[..., None] * _WGL8).reshape(xc.shape[0], -1)
        with np.errstate(divide="ignore"):
            logW = np.log(W) + lat.logkernel((xc - t * Y) / sig) - np.log(sig)
        return Y, logW

    def posterior(self, x, t, L):
        x = as_points(x, 1).reshape(-1, 1)
        t = float(t)
        if t <= 0:
            return self._at_zero(x, L)
        lat = TruncatedGaussian(1, L)
        n = x.shape[0]
        m, C, logp = np.empty((n, 1)), np.empty((n, 1, 1)), np.empty(n)
        for s0 in range(0, n, self.chunk):
            sl = slice(s0, s0 + self.chunk)
            with np.errstate(divide="ignore", invalid="ignore"):
                Y, logW = self._rule(x[sl, 0], t, L, lat)
                logf = logW + self.logpdf(Y[..., None])
            logf = np.where(np.isnan(logf), -np.inf, logf)
            mx = logf.max(axis=1)
            ok = np.isfinite(mx)
            f = np.exp(logf - np.where(ok, mx, 0.0)[:, None])
            Z = f.sum(1)
            Zs = np.where(ok, Z, 1.0)
            Yp = Y[..., None]
            mb = np.einsum("nq,nqd->nd", f, Yp) / Zs[:, None]
            # centred second moment: E[y^2] - m^2 loses the relative accuracy of
            # C ~ (sigma / t)^2 as t -> 1, which the divergence divides by sigma^3
            d = Yp - mb[:, None, :]
            m[sl] = mb
            C[sl] = np.einsum("nq,nqd,nqe->nde", f, d, d) / Zs[:, None, None]
            with np.errstate(divide="ignore"):
                logp[sl] = np.where(ok, np.log(Zs) + np.where(ok, mx, 0.0), -np.inf)
        return self._undefined(m, C, logp, ~np.isfinite(logp), t)

    def jacobian_trace(self, x, t, L):
        """(t / sigma^2) C + (1/t) [(u - m) q(u) + (m - l) q(l)],
        q(u) = pi(u) phi(L) / (sigma Z_L p_t(x)) (zero where pi vanishes at the face)."""
        x = as_points(x, 1).reshape(-1, 1)
        t = float(t)
        if t <= 0:
            return np.zeros(x.shape[0])
        m, C, logp = self.posterior(x, t, L)
        sig = 1.0 - t
        l, u = self._box(x, t, L)
        lq = -0.5 * L * L - _LOG_SQRT2PI - np.log(sig) - _log_ZL(L) - logp[:, None]
        with np.errstate(over="ignore"):
            qu = self.pdf(u[:, 0])[:, None] * np.exp(lq)
            ql = self.pdf(l[:, 0])[:, None] * np.exp(lq)
        return _faces_trace(t, C, m, l, u, ql, qu)


class TrimodalBeta(BetaMixture1D):
    """The trimodal target of sec:data, 0.30 B(8, 24) + 0.45 B(30, 28) + 0.25 B(24, 6) on
    [-1, 1] (64 panels in y)."""

    def __init__(self):
        super().__init__(w=[0.30, 0.45, 0.25], a=[8, 30, 24], b=[24, 28, 6], n_panels=64)


class Inhomogeneous(BetaMixture1D):
    """The inhomogeneous target of sec:data, 0.8 B(4, 5) + 0.2 B(1500, 500) on [-1, 1]: broad
    (sd 0.314) plus a sharp mode (sd 0.019) at y = 0.5; 256 panels in y resolve the sharp
    mode."""

    def __init__(self):
        super().__init__(w=[0.8, 0.2], a=[4, 1500], b=[5, 500], n_panels=256)


# ---------------------------------------------------------------- isotropic mixtures

def _iso_mixture(x, t, L, mus, logw, v, block_pairs=2_000_000):
    """Posterior of y ~ sum_k omega_k N(mu_k, v_k I) restricted to B.

    Component k: A_k = sigma^2 + t^2 v_k, y | x, k ~ N(mu_k(x), s_k^2 I),
    mu_k(x) = mu_k + (t v_k / A_k)(x - t mu_k), s_k^2 = v_k sigma^2 / A_k, prior weight
    log w_k = log omega_k + log N(x; t mu_k, A_k I); components with log w_k < max - 80 are
    dropped. Per axis (alpha, beta) = ((l_d, u_d) - mu_kd)/s_k and (log Z_kd, E_kd - alpha_kd,
    V_kd) = truncnorm_moments; weights ∝ w_k prod_d Z_kd; component mean mu_kd + s_k E_kd,
    variance s_k^2 V_kd; m, C by total expectation and covariance (centred sums);
    p_t = Z_L^{-D} sum_k omega_k N(x; t mu_k, A_k I) prod_d Z_kd. Returns (m, C, log p_t)."""
    n, D = x.shape
    K = mus.shape[0]
    v = np.broadcast_to(np.asarray(v, float), (K,))
    sig = 1.0 - t
    A = sig * sig + t * t * v
    s = np.sqrt(v * sig * sig / A)
    l, u = (x - sig * L) / t, (x + sig * L) / t
    width = 2.0 * sig * L / t
    m = np.empty((n, D))
    C = np.empty((n, D, D))
    logp = np.empty(n)
    logZL = D * _log_ZL(L)
    step = max(1, block_pairs // K)
    MM = np.sum(mus * mus, 1)
    for s0 in range(0, n, step):
        xb = x[s0:s0 + step]
        nb = xb.shape[0]
        d2 = np.sum(xb * xb, 1)[:, None] - 2 * t * xb @ mus.T + t * t * MM[None]
        lw = logw[None] - 0.5 * np.maximum(d2, 0.0) / A[None] - 0.5 * D * (_LOG2PI + np.log(A))[None]
        keep = lw >= lw.max(1, keepdims=True) - 80.0
        ii, kk = np.nonzero(keep)
        r = xb[ii] - t * mus[kk]
        mk = mus[kk] + (t * v[kk] / A[kk])[:, None] * r
        sk = s[kk][:, None]
        lo, hi = l[s0 + ii], u[s0 + ii]
        al, be = (lo - mk) / sk, (hi - mk) / sk
        logZ, Ea, V = truncnorm_moments(al, be, width / sk)
        lwt = lw[ii, kk] + logZ.sum(1)
        mx = np.full(nb, -np.inf)
        np.maximum.at(mx, ii, lwt)
        okp = np.isfinite(mx)
        wt = np.exp(lwt - np.where(okp, mx, 0.0)[ii])
        S = np.bincount(ii, wt, minlength=nb)
        with np.errstate(divide="ignore"):
            logp[s0:s0 + nb] = np.where(okp, np.log(np.where(S > 0, S, 1.0)) + mx, -np.inf) - logZL
        wt = wt / np.where(S > 0, S, 1.0)[ii]
        # the mean from the nearer end keeps the width information of narrow intervals
        wd = width / sk
        near_lo = np.abs(al) <= np.abs(be)
        mean_k = np.where(near_lo, lo + sk * Ea, hi + sk * (Ea - wd))
        var_k = sk * sk * V
        mb = np.stack([np.bincount(ii, wt * mean_k[:, d], minlength=nb) for d in range(D)], 1)
        dd = mean_k - mb[ii]
        Cb = np.empty((nb, D, D))
        for a_ in range(D):
            for b_ in range(a_, D):
                val = np.bincount(ii, wt * dd[:, a_] * dd[:, b_], minlength=nb)
                if a_ == b_:
                    val = val + np.bincount(ii, wt * var_k[:, a_], minlength=nb)
                Cb[:, a_, b_] = Cb[:, b_, a_] = val
        m[s0:s0 + nb], C[s0:s0 + nb] = mb, Cb
    return m, C, logp


class _IsoMixtureTarget(Target):
    """A target whose law is a finite mixture of isotropic Gaussians (``_components``)."""

    def _components(self):
        """(mus (K, D), log omega (K,), v (K,) or scalar)."""
        raise NotImplementedError

    def posterior(self, x, t, L):
        x = as_points(x, self.dim).reshape(-1, self.dim)
        t = float(t)
        if t <= 0:
            return self._at_zero(x, L)
        mus, logw, v = self._components()
        m, C, logp = _iso_mixture(x, t, L, mus, logw, v)
        return self._undefined(m, C, logp, ~np.isfinite(logp), t)


class EightGaussians(_IsoMixtureTarget):
    """The eight Gaussians of sec:data (FFJORD "8gaussians"): y = (c_k + 0.5 eta) / 1.414, k
    uniform on 8 centres c_k = 4 (+-1, 0), 4 (0, +-1), 4 (+-1/sqrt 2, +-1/sqrt 2),
    eta ~ N(0, I): the equal-weight mixture of N(ybar_k, s^2 I), ybar_k = c_k / 1.414,
    s = 0.5 / 1.414 (FFJORD's literal 1.414). The code names the centres mus."""

    DIV = 1.414

    def __init__(self):
        super().__init__(2)
        r = 1.0 / np.sqrt(2.0)
        c = 4.0 * np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (r, r), (r, -r), (-r, r), (-r, -r)])
        self.mus = c / self.DIV
        self.s = 0.5 / self.DIV
        self.logw = np.full(8, -np.log(8.0))

    def _components(self):
        return self.mus, self.logw, self.s**2

    def logpdf(self, x):
        """log sum_k (1/8) N(y; mu_k, s^2 I)."""
        y = as_points(x, 2)
        r = y[..., None, :] - self.mus
        lk = -0.5 * np.sum(r * r, -1) / self.s**2 - _LOG2PI - 2.0 * np.log(self.s)
        return logsumexp(lk + self.logw, axis=-1)

    def sample(self, n, rng=None):
        """All normal pairs first, then the centre indices (this rng order fixes the training sets)."""
        rng = np.random.default_rng(rng)
        n = int(n)
        e = rng.standard_normal((n, 2))
        k = rng.integers(0, 8, n)
        return self.mus[k] + self.s * e

    @property
    def mean(self):
        return self.mus.mean(0)

    @property
    def cov(self):
        mu = self.mean
        return (np.einsum("ki,kj->ij", self.mus, self.mus) / 8 - np.outer(mu, mu)
                + self.s**2 * np.eye(2))


class TwoSpirals(_IsoMixtureTarget):
    """The two spirals of sec:data: u = 3 pi sqrt(U), zeta(u) = (-u cos u, u sin u), arm
    +-zeta with probability 1/2, y = (+-zeta(u) + s eta) / 3, s = 0.15, eta ~ N(0, I) (noise
    sd c = 0.05 in y). pi(y) = (1/2) sum_+- int q(u) N(y; +-zeta(u)/3, c^2 I) du,
    q(u) = 2u / (3 pi)^2, evaluated by composite 8-point Gauss-Legendre in u on panels of
    equal arc length c of zeta/3: the law is the mixture of the 2Q node Gaussians (``nodes``,
    ``weights``). The code's ``curve`` is zeta/3."""

    def __init__(self, s: float = 0.15):
        super().__init__(2)
        self.s = float(s)
        self.c = self.s / 3.0
        self.umax = 3.0 * np.pi
        uq, wq = self._u_rule(self.c)
        g = self.curve(uq)
        self.nodes_u = np.concatenate([uq, uq])
        self.nodes = np.concatenate([g, -g])
        self.weights = 0.5 * np.concatenate([wq, wq])
        self.logweights = np.log(self.weights)
        self.block = max(1, 2_000_000 // self.nodes.shape[0])

    def _components(self):
        return self.nodes, self.logweights, self.c**2

    @staticmethod
    def arc_length(u):
        """S(u) = (u sqrt(1 + u^2) + asinh u) / 6, the arc length of zeta/3."""
        u = np.asarray(u, float)
        return (u * np.sqrt(1.0 + u * u) + np.arcsinh(u)) / 6.0

    def _u_rule(self, delta):
        """Nodes and weights of int_0^{3 pi} q(u) f(u) du: panels of arc length <= delta,
        8-point Gauss-Legendre per panel."""
        Smax = float(self.arc_length(self.umax))
        npan = int(np.ceil(Smax / delta))
        ut = np.linspace(0.0, self.umax, 200_001)
        edges = np.interp(np.linspace(0.0, Smax, npan + 1), self.arc_length(ut), ut)
        edges[0], edges[-1] = 0.0, self.umax
        a, b = edges[:-1, None], edges[1:, None]
        u = (a + 0.5 * (b - a) * (_GL8 + 1)).ravel()
        w = (0.5 * (b - a) * _WGL8).ravel()
        return u, w * 2.0 * u / self.umax**2

    @staticmethod
    def curve(u):
        """zeta(u) / 3, shape (..., 2)."""
        u = np.asarray(u, float)
        return np.stack([-u * np.cos(u), u * np.sin(u)], -1) / 3.0

    @staticmethod
    def curve_d(u):
        """d zeta(u) / du / 3."""
        u = np.asarray(u, float)
        return np.stack([-np.cos(u) + u * np.sin(u), np.sin(u) + u * np.cos(u)], -1) / 3.0

    def logpdf(self, x):
        y = as_points(x, 2)
        shp = y.shape[:-1]
        y = y.reshape(-1, 2)
        out = np.empty(y.shape[0])
        G, lW = self.nodes, self.logweights
        c2 = self.c**2
        for s0 in range(0, y.shape[0], self.block):
            yb = y[s0:s0 + self.block]
            d2 = np.sum(yb * yb, 1)[:, None] - 2 * yb @ G.T + np.sum(G * G, 1)[None]
            lf = lW[None] - 0.5 * np.maximum(d2, 0.0) / c2
            out[s0:s0 + self.block] = logsumexp(lf, axis=1) - _LOG2PI - np.log(c2)
        return out.reshape(shp)

    def sample(self, n, rng=None):
        """u = sqrt(U) 3 pi, then the arm, then eps (this rng order fixes the training sets)."""
        rng = np.random.default_rng(rng)
        n = int(n)
        u = np.sqrt(rng.uniform(size=n)) * self.umax
        sgn = 1.0 - 2.0 * rng.integers(0, 2, n)
        e = rng.standard_normal((n, 2))
        return sgn[:, None] * self.curve(u) + self.c * e

    @property
    def mean(self):
        return np.zeros(2)

    @property
    def cov(self):
        """sum_q W_q G_q G_q^T + c^2 I."""
        return (np.einsum("q,qi,qj->ij", self.weights, self.nodes, self.nodes)
                + self.c**2 * np.eye(2))

    def tangent(self, y, newton: int = 4):
        """Unit tangent of the spiral at the curve point nearest to each y (n, 2): nearest
        node by a k-d tree, then Newton steps on (a(u) - y) . a'(u) = 0, a = +-zeta/3."""
        from scipy.spatial import cKDTree
        y = as_points(y, 2)
        _, k = cKDTree(self.nodes).query(y)
        u = self.nodes_u[k].copy()
        sgn = np.where(k < self.nodes.shape[0] // 2, 1.0, -1.0)
        for _ in range(newton):
            a, d1 = sgn[:, None] * self.curve(u), sgn[:, None] * self.curve_d(u)
            d2 = sgn[:, None] * np.stack([2 * np.sin(u) + u * np.cos(u),
                                          2 * np.cos(u) - u * np.sin(u)], -1) / 3.0
            f = np.sum((a - y) * d1, 1)
            fp = np.sum(d1 * d1, 1) + np.sum((a - y) * d2, 1)
            step = np.where(fp > 0, f / np.where(fp > 0, fp, 1.0), 0.0)
            u = np.clip(u - step, 0.0, self.umax)
        e = self.curve_d(u)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


# ---------------------------------------------------------------- Checkerboard

class Checkerboard(Target):
    """The checkerboard of sec:data: the uniform law on the 8 dark squares of side 2 of a 4 x 4
    board on [-4, 4]^2, squares [-4 + 2i, -2 + 2i) x [-4 + 2j, -2 + 2j) with i + j even,
    density 1/32 there.

    Posterior: in u = (t y_d - x_d) / sigma = -z_d the posterior in square k is
    N(0, 1) on [alpha_kd, beta_kd] = [(t a_kd - x_d)/sigma, (t (a_kd + 2) - x_d)/sigma] cut to
    [-L, L]; weights ∝ Z_k1 Z_k2, E[y_d | k] = max(a_kd, l_d) + (sigma/t)(E[u] - alpha'_kd),
    Var = (sigma/t)^2 Var u, rho(x, t) = (32 t^2 Z_L^2)^{-1} sum_k Z_k1 Z_k2.
    """

    def __init__(self):
        super().__init__(2)
        ij = [(i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0]
        self.squares = np.array(ij)
        self.lower = -4.0 + 2.0 * self.squares

    def pdf(self, x):
        y = as_points(x, 2)
        i = np.floor((y[..., 0] + 4.0) / 2.0)
        j = np.floor((y[..., 1] + 4.0) / 2.0)
        inside = (i >= 0) & (i <= 3) & (j >= 0) & (j <= 3)
        return np.where(inside & ((i + j) % 2 == 0), 1.0 / 32.0, 0.0)

    def logpdf(self, x):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x))

    def sample(self, n, rng=None):
        """FFJORD: x1 = 4U - 2, x2 = U' - 2B + (floor(x1) mod 2), y = 2 (x1, x2)."""
        rng = np.random.default_rng(rng)
        n = int(n)
        x1 = rng.uniform(size=n) * 4.0 - 2.0
        x2 = rng.uniform(size=n) - rng.integers(0, 2, n) * 2.0
        x2 = x2 + np.floor(x1) % 2
        return 2.0 * np.stack([x1, x2], 1)

    @property
    def mean(self):
        return np.zeros(2)

    @property
    def cov(self):
        c = self.lower + 1.0
        return np.einsum("ki,kj->ij", c, c) / len(c) + np.eye(2) / 3.0

    def _moments(self, x, t, L):
        sig = 1.0 - t
        edges = -4.0 + 2.0 * np.arange(4)
        al = (t * edges[None, None, :] - x[:, :, None]) / sig          # (n, 2, 4)
        be = (t * (edges[None, None, :] + 2.0) - x[:, :, None]) / sig
        alc, bec = np.maximum(al, -L), np.minimum(be, L)
        cut_lo, cut_hi = al < -L, be > L
        w_full = 2.0 * t / sig
        w = np.where(cut_lo & cut_hi, 2.0 * L,
                     np.where(cut_lo | cut_hi, bec - alc, w_full))
        w = np.maximum(w, 0.0)
        bec = np.where(w > 0, bec, alc)
        logZ, Ea, Vu = truncnorm_moments(alc, bec, w)
        lower = np.where(cut_lo, ((x - sig * L) / t)[:, :, None], edges[None, None, :])
        my = lower + (sig / t) * Ea
        vy = (sig / t) ** 2 * Vu
        I, J = self.squares[:, 0], self.squares[:, 1]
        lw = logZ[:, 0, I] + logZ[:, 1, J]
        mx = lw.max(1, keepdims=True)
        ok = np.isfinite(mx[:, 0])
        wk = np.exp(lw - np.where(ok, mx[:, 0], 0.0)[:, None])
        S = wk.sum(1, keepdims=True)
        with np.errstate(divide="ignore"):
            logp = np.where(ok, np.log(np.where(ok, S[:, 0], 1.0)) + mx[:, 0], -np.inf) \
                - np.log(32.0 * t * t) - 2 * _log_ZL(L)
        wk = wk / np.where(S > 0, S, 1.0)
        mk = np.stack([my[:, 0, I], my[:, 1, J]], -1)
        m = np.einsum("nk,nkd->nd", wk, mk)
        vk = np.stack([vy[:, 0, I], vy[:, 1, J]], -1)
        d = mk - m[:, None, :]
        C = np.einsum("nk,nkd,nke->nde", wk, d, d)
        C[:, 0, 0] += np.einsum("nk,nk->n", wk, vk[..., 0])
        C[:, 1, 1] += np.einsum("nk,nk->n", wk, vk[..., 1])
        return m, C, logp

    def posterior(self, x, t, L):
        x = as_points(x, 2).reshape(-1, 2)
        t = float(t)
        if t <= 0:
            return self._at_zero(x, L)
        m, C, logp = self._moments(x, t, L)
        return self._undefined(m, C, logp, ~np.isfinite(logp), t)


# ---------------------------------------------------------------- noisy circle

def _angle_edges(n_panels=60, psi_min=1e-7, n_uniform=256):
    """Panel edges on [-pi, pi] in psi = theta - (angle of x): geometric toward psi = 0
    (widths down to psi_min) plus n_uniform uniform panels per half (256 bring m to 1e-13
    against brute-force quadrature, 64 leave 2e-10 at t = 0.9)."""
    e = np.concatenate([[0.0], np.geomspace(psi_min, np.pi, n_panels)])
    e = np.unique(np.concatenate([e, np.linspace(0.0, np.pi, n_uniform + 1)]))
    return np.unique(np.concatenate([-e[::-1], e]))


class NoisyCircle(Target):
    """The noisy circle of sec:data: y = r (cos theta, sin theta), r ~ N(R, S^2) (the paper's
    radius varrho ~ N(2, 0.1^2)) independent of theta ~ sum_k pi_k vM(mu_k, kappa_k);
    R = 2, S = 0.1, modes 0, 2 pi/3, 4 pi/3, kappa 4, 16, 64, weights 0.5, 0.3, 0.2.
    Density (the r < 0 branch, 20 sd away, neglected): pi(y) = g(theta) phi((r - R)/S) / (S r).

    Posterior: for fixed theta, r | theta, x ~ N(mu, 1/P), P = 1/S^2 + t^2/sigma^2,
    mu = (R/S^2 + t x.e/sigma^2)/P, restricted to the interval [r_lo, r_hi] of r with r e in B;
    angular log weight log g(theta) + P mu^2 / 2 + log(Phi(sqrt P (r_hi - mu)) - Phi(sqrt P (r_lo - mu))),
    by composite 8-point Gauss-Legendre in theta on the refined rule centred at the angle of x
    (256 uniform panels per half), with the corner angles of B (and their antipodes) added as panel edges (the interval
    ends switch faces there). No rho(x, t) (NaN) and no Jacobian trace.
    """

    chunk = 256

    def __init__(self, R=2.0, S=0.1, modes=(0.0, 2 * np.pi / 3, 4 * np.pi / 3),
                 kappa=(4.0, 16.0, 64.0), weights=(0.5, 0.3, 0.2)):
        super().__init__(2)
        self.R, self.S = float(R), float(S)
        self.modes = np.asarray(modes, float)
        self.kappa = np.asarray(kappa, float)
        self.weights = np.asarray(weights, float) / np.sum(weights)
        self._edges = _angle_edges()

    def angle_pdf(self, th):
        """g(theta) = sum_k pi_k exp(kappa_k (cos(theta - mu_k) - 1)) / (2 pi I0e(kappa_k))."""
        th = np.asarray(th, float)
        return sum(p * np.exp(k * (np.cos(th - m) - 1)) / (2 * np.pi * i0e(k))
                   for p, k, m in zip(self.weights, self.kappa, self.modes))

    def pdf(self, x):
        y = as_points(x, 2)
        r = np.hypot(y[..., 0], y[..., 1])
        th = np.arctan2(y[..., 1], y[..., 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (self.angle_pdf(th) * np.exp(-0.5 * ((r - self.R) / self.S) ** 2)
                 / (np.sqrt(2 * np.pi) * self.S * r))
        return np.where(r > 0, f, 0.0)

    def logpdf(self, x):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x))

    def sample(self, n, rng=None):
        """Component, von Mises angle, Gaussian radius (this rng order fixes the training sets)."""
        rng = np.random.default_rng(rng)
        k = rng.choice(self.modes.size, size=int(n), p=self.weights)
        th = rng.vonmises(self.modes[k], self.kappa[k])
        r = self.R + self.S * rng.standard_normal(int(n))
        return np.stack([r * np.cos(th), r * np.sin(th)], 1)

    def tangent(self, y):
        """Unit tangent of the circle through y about the origin, (-sin theta, cos theta)."""
        y = as_points(y, 2)
        th = np.arctan2(y[..., 1], y[..., 0])
        return np.stack([-np.sin(th), np.cos(th)], -1)

    @property
    def mean(self):
        A = i1e(self.kappa) / i0e(self.kappa)
        return self.R * np.array([np.sum(self.weights * A * np.cos(self.modes)),
                                  np.sum(self.weights * A * np.sin(self.modes))])

    @property
    def cov(self):
        A = i1e(self.kappa) / i0e(self.kappa)
        A2 = 1.0 - 2.0 * A / self.kappa
        c2 = np.sum(self.weights * A2 * np.cos(2 * self.modes))
        s2 = np.sum(self.weights * A2 * np.sin(2 * self.modes))
        Eee = 0.5 * np.eye(2) + 0.5 * np.array([[c2, s2], [s2, -c2]])
        m = self.mean
        return (self.R**2 + self.S**2) * Eee - np.outer(m, m)

    def posterior(self, x, t, L):
        x = as_points(x, 2).reshape(-1, 2)
        t = float(t)
        if t <= 0:
            return self._at_zero(x, L)
        n = x.shape[0]
        m, C = np.empty((n, 2)), np.empty((n, 2, 2))
        bad = np.zeros(n, bool)
        for s0 in range(0, n, self.chunk):
            sl = slice(s0, s0 + self.chunk)
            m[sl], C[sl], bad[sl] = self._block(x[sl], t, L)
        logp = np.full(n, np.nan)
        return self._undefined(m, C, logp, bad, t)

    def _block(self, x, t, L):
        n = x.shape[0]
        sig = 1.0 - t
        P = 1.0 / self.S**2 + t**2 / sig**2
        sP = np.sqrt(P)
        l, u = self._box(x, t, L)
        phi = np.arctan2(x[:, 1], x[:, 0])
        cx = np.stack([np.stack([l[:, 0], l[:, 1]], 1), np.stack([u[:, 0], l[:, 1]], 1),
                       np.stack([l[:, 0], u[:, 1]], 1), np.stack([u[:, 0], u[:, 1]], 1)], 1)
        th_c = np.arctan2(cx[..., 1], cx[..., 0])                          # (n, 4)
        th_c = np.concatenate([th_c, th_c + np.pi], 1) - phi[:, None]
        psi_c = np.mod(th_c + np.pi, 2 * np.pi) - np.pi
        E = np.sort(np.concatenate([np.broadcast_to(self._edges, (n, self._edges.size)), psi_c],
                                   1), axis=1)
        a, b = E[:, :-1], E[:, 1:]
        half = 0.5 * (b - a)
        psi = (a[..., None] + half[..., None] * (_GL8 + 1)).reshape(n, -1)
        wpsi = (half[..., None] * _WGL8).reshape(n, -1)
        th = phi[:, None] + psi
        c, s = np.cos(th), np.sin(th)
        ax = x[:, :1] * c + x[:, 1:] * s
        mu = (self.R / self.S**2 + t * ax / sig**2) / P
        # P (mu^2 - mu0^2)/2 relative to psi = 0, without the cancellation of two O(1/sigma^2)
        rx = np.hypot(x[:, :1], x[:, 1:])
        da = -2.0 * rx * np.sin(0.5 * psi) ** 2
        mu0 = (self.R / self.S**2 + t * rx / sig**2) / P
        # interval of r with r e in B, per axis then intersected
        rlo = np.full(psi.shape, -np.inf)
        rhi = np.full(psi.shape, np.inf)
        empty = np.zeros(psi.shape, bool)
        for e, ld, ud in ((c, l[:, :1], u[:, :1]), (s, l[:, 1:], u[:, 1:])):
            with np.errstate(divide="ignore", invalid="ignore"):
                q1, q2 = ld / e, ud / e
            pos, neg = e > 0, e < 0
            rlo = np.where(pos, np.maximum(rlo, q1), np.where(neg, np.maximum(rlo, q2), rlo))
            rhi = np.where(pos, np.minimum(rhi, q2), np.where(neg, np.minimum(rhi, q1), rhi))
            empty |= (e == 0) & ~((ld <= 0) & (ud >= 0))
        empty |= ~(rhi > rlo)
        rhi = np.where(empty, rlo, rhi)
        rlo = np.where(empty, 0.0, rlo)
        rhi = np.where(empty, 0.0, rhi)
        al, be = sP * (rlo - mu), sP * (rhi - mu)
        logZ, Ea, V = truncnorm_moments(al, be, sP * (rhi - rlo))
        with np.errstate(divide="ignore"):
            logw = (0.5 * t / sig**2) * da * (mu + mu0) + np.log(wpsi) + np.log(self.angle_pdf(th)) \
                + logZ
        logw = np.where(empty, -np.inf, logw)
        mx = logw.max(axis=1, keepdims=True)
        bad = ~np.isfinite(mx[:, 0])
        w = np.exp(logw - np.where(bad[:, None], 0.0, mx))
        w /= np.where(bad, 1.0, w.sum(axis=1))[:, None]
        near_lo = np.abs(al) <= np.abs(be)
        Er = np.where(near_lo, rlo + Ea / sP, rhi + (Ea - sP * (rhi - rlo)) / sP)
        Vr = V / P
        Er = np.where(empty, 0.0, Er)
        m = np.stack([(w * Er * c).sum(1), (w * Er * s).sum(1)], axis=1)
        d0, d1 = Er * c - m[:, :1], Er * s - m[:, 1:]
        C = np.empty((n, 2, 2))
        C[:, 0, 0] = (w * (d0 * d0 + Vr * c * c)).sum(1)
        C[:, 1, 1] = (w * (d1 * d1 + Vr * s * s)).sum(1)
        C[:, 0, 1] = C[:, 1, 0] = (w * (d0 * d1 + Vr * c * s)).sum(1)
        return m, C, bad


# ---------------------------------------------------------------- the sampler interface

class PopulationVelocity:
    """The population velocity v_pi of a target with the sampler interface:
    v_pi(x, t) = (m - x) / sigma (eq:velocity), div v_pi = (tr grad m - D) / sigma (the 1D
    targets, for the population flow of experiments/compute/floors.py).

    latent: the TruncatedGaussian (default ``target.latent``). The last (x, t) is cached,
    since the sampler asks for v and div v at the same point."""

    def __init__(self, target: Target, latent=None):
        self.target = target
        self.latent = latent if latent is not None else getattr(target, "latent", None)
        if self.latent is None:
            raise ValueError("PopulationVelocity needs the latent (argument or target.latent)")
        self.L = self.latent.L
        self.dim = target.dim
        self._key = None
        self.nfev = 0

    def _same(self, x, t):
        return self._key is not None and self._key[1] == t and np.array_equal(self._key[0], x)

    def __call__(self, x, t):
        x = as_points(x, self.dim).reshape(-1, self.dim)
        t = float(t)
        if not (self._same(x, t) and "m" in self._val):
            self.nfev += 1
            m = self.target.posterior(x, t, self.L)[0]
            self._key, self._val = (x.copy(), t), {"m": m}
        return (self._val["m"] - x) / (1.0 - t)

    def div(self, x, t):
        x = as_points(x, self.dim).reshape(-1, self.dim)
        t = float(t)
        if not self._same(x, t):
            self._key, self._val = (x.copy(), t), {}
        if "J" not in self._val:
            self._val["J"] = self.target.jacobian_trace(x, t, self.L)
        return (self._val["J"] - self.dim) / (1.0 - t)
