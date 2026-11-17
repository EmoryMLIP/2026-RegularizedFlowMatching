"""Gabriel et al. (arXiv:2505.22841) smoothing of the empirical velocity (S:sec:kde, last
paragraphs): v_h(., t) = G_{t sqrt h} * v_N(., t), with a constant h or h(t).

The code smooths the posterior mean instead, u_h = ((G_sigma * m_N)(x) - x)/(1 - t),
sigma = t sqrt(h); since G_sigma * x = x on R, this equals G_sigma * v_N, the paper's v_h
(the code's u_h). Equivalently m_N convolved in x/t with a Gaussian of variance h. m_N is the
empirical posterior mean of the truncated latent (the same v_N as the spline flow), and the
flow starts from the truncated latent. With L = None the Gaussian-latent m_N of Gabriel et al.
is smoothed (closed form), the reference that the truncated version matches to O(phi(L)).

Truncated m_N: w_i ∝ exp(-(x - t y_i)^2/(2 (1-t)^2)) on |x - t y_i| <= (1 - t) L. Between the
box ends t y_i -+ (1 - t) L the active set S is fixed and m_N = M_S' with
M_S(x) = (x^2/2 + (1-t)^2 log sum_{i in S} exp(-(x - t y_i)^2/(2 (1-t)^2)))/t, so the cell
integrals of m_N are sums of differences of M_S over the pieces between box ends (exact).
Where no box covers x (rho_N = 0) the untruncated weights are used, because an RK4 stage
can step there although the exact trajectory cannot.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.interpolate import CubicHermiteSpline
from scipy.signal import fftconvolve
from scipy.special import ndtr

from ..latent import as_points

__all__ = ["StdGabriel", "h_schedule"]

_LAT_W, _C_GH, _K_GH, _NC, _GMAX = 7.0, 0.25, 24, 20, 400_000


def _window_stats(x, t, ys, lo, hi, max_block=2e7):
    """m, m' = t Var_w(y)/(1-t)^2 and A = (1-t)^2 log Z (up to a constant) at x with the
    Gaussian weights restricted to the sorted samples ys[lo:hi] (per point)."""
    n = ys.size
    m, dm, A = np.empty_like(x), np.empty_like(x), np.empty_like(x)
    if x.size == 0:
        return m, dm, A
    W = int((hi - lo).max())
    step = max(1, int(max_block // max(W, 1)))
    ar = np.arange(W)
    v2 = (1 - t) ** 2
    for s0 in range(0, x.size, step):
        sl = slice(s0, s0 + step)
        idx = lo[sl, None] + ar[None]
        msk = idx < hi[sl, None]
        Y = ys[np.minimum(idx, n - 1)]
        a = np.where(msk, -0.5 * (x[sl, None] - t * Y) ** 2, -np.inf)
        amax = a.max(1)
        e = np.exp((a - amax[:, None]) / v2)
        S = e.sum(1)
        w = e / S[:, None]
        mm = (w * Y).sum(1)
        m[sl] = mm
        dm[sl] = t / v2 * (w * (Y - mm[:, None]) ** 2).sum(1)
        A[sl] = amax + v2 * np.log(S)
    return m, dm, A


def _gauss_windows(x, t, ys):
    """The windows of the Gaussian latent: samples within sqrt(d^2 + 80 s^2) of z = x/t,
    d the distance to the nearest sample (dropped terms below e^-40 of the largest)."""
    n = ys.size
    z = x / t
    s = (1 - t) / t
    pos = np.searchsorted(ys, z)
    left = ys[np.clip(pos - 1, 0, n - 1)]
    right = ys[np.clip(pos, 0, n - 1)]
    d = np.minimum(np.abs(z - left), np.abs(right - z))
    r = np.sqrt(d**2 + 80 * s**2)
    lo = np.searchsorted(ys, z - r, "left")
    hi = np.maximum(np.searchsorted(ys, z + r, "right"), lo + 1)
    return np.minimum(lo, n - 1), hi


def _trunc_windows(z, t, ys, L):
    """Index ranges of the samples whose box covers x = t z: |z - y_i| <= (1-t) L / t."""
    r = (1 - t) * L / t
    return np.searchsorted(ys, z - r, "left"), np.searchsorted(ys, z + r, "right")


def _post_stats(x, t, ys, L=None):
    """(m_N, m_N', A) at x for sorted 1D samples ys: Gaussian latent (L None) or truncated
    (fallback to the Gaussian weights where no box covers x)."""
    x = np.asarray(x, float)
    if L is None:
        return _window_stats(x, t, ys, *_gauss_windows(x, t, ys))
    lo, hi = _trunc_windows(x / t, t, ys, L)
    empty = hi <= lo
    m, dm, A = _window_stats(x, t, ys, np.where(empty, 0, lo), np.where(empty, 1, hi))
    if empty.any():
        xe = x[empty]
        m[empty], dm[empty], A[empty] = _window_stats(xe, t, ys, *_gauss_windows(xe, t, ys))
    return m, dm, A


def _cell_integrals(nodes, t, ys, L):
    """int_{nodes[j]}^{nodes[j+1]} m_N dx, exact: the pieces between box ends, each the
    difference of M_S for the active set S at its midpoint (module docstring)."""
    sig = 1.0 - t
    br = np.concatenate([t * ys - sig * L, t * ys + sig * L])
    br = br[(br > nodes[0]) & (br < nodes[-1])]
    P = np.unique(np.concatenate([nodes, br]))
    a, b = P[:-1], P[1:]
    mid = 0.5 * (a + b)
    lo, hi = _trunc_windows(mid / t, t, ys, L)
    empty = hi <= lo
    # gaps (rho_N = 0): the Gaussian antiderivative, with the union of the windows at both
    # ends so that A is accurate at each end of the piece
    la, ha = _gauss_windows(a, t, ys)
    lb, hb = _gauss_windows(b, t, ys)
    lo, hi = np.where(empty, np.minimum(la, lb), lo), np.where(empty, np.maximum(ha, hb), hi)
    _, _, Aa = _window_stats(a, t, ys, lo, hi)
    _, _, Ab = _window_stats(b, t, ys, lo, hi)
    inc = ((0.5 * b * b + Ab) - (0.5 * a * a + Aa)) / t
    F = np.concatenate([[0.0], np.cumsum(inc)])
    return np.diff(F[np.searchsorted(P, nodes)])


class _Smoothed:
    """m~ = G_sigma * m_N and m~' at one (t, h), sigma = t sqrt(h): Gauss-Hermite branch while
    sigma <= C_GH ell (ell the smallest length scale of m_N), exact cell-integral branch
    otherwise, cubic Hermite interpolation on a node grid; 'direct' when the grid would
    exceed GMAX nodes; 'const' at t = 0."""

    def __init__(self, ys, t, h, L=None, K=_K_GH, nc=_NC, c_gh=_C_GH):
        self.ys, self.t, self.h, self.L = ys, t, h, L
        sigma = t * np.sqrt(h)
        self.sigma = sigma
        if t <= 0:
            self.branch, self.mean = "const", ys.mean()
            return
        if sigma == 0:
            # no smoothing: m_N itself (its jumps at the box ends rule out interpolation)
            self.branch, self.xi, self.wk = "direct", np.zeros(1), np.ones(1)
            return
        # node range: the support t y + (1 - t) z of x for |y| <= Y, |z| <= the latent width;
        # Y = 1 for data in [-1, 1], else the data's own extent, so
        # that data outside [-1, 1] are not clipped to the end nodes
        Y = max(1.0, float(np.max(np.abs(ys)))) if ys.size else 1.0
        Lz = _LAT_W if L is None else max(_LAT_W, float(L))
        Lr = t * Y + Lz * (1 - t) + 8 * sigma
        D = np.diff(ys).max() if ys.size > 1 else 1.0
        ell = (1 - t) * min(1.0, (1 - t) / (t * D))
        if sigma <= c_gh * ell:
            self.branch, delta = "gh", ell / 20
        else:
            self.branch, delta = "cell", sigma / nc
        G = int(np.ceil(2 * Lr / delta)) + 1
        if G > _GMAX:
            self.branch = "direct"
            self.xi, self.wk = self._gh(K) if sigma > 0 else (np.zeros(1), np.ones(1))
            return
        nodes = np.linspace(-Lr, Lr, G)
        if self.branch == "gh":
            xi, wk = self._gh(K) if sigma > 0 else (np.zeros(1), np.ones(1))
            m, dm, _ = _post_stats((nodes[:, None] + sigma * xi[None]).ravel(), t, ys, L)
            mt, dmt = m.reshape(G, -1) @ wk, dm.reshape(G, -1) @ wk
        else:
            mt, dmt = self._cell(nodes, t, ys, sigma, L)
        self.spline = CubicHermiteSpline(nodes, mt, dmt)
        self.lo, self.hi = nodes[0], nodes[-1]
        self.dx = nodes[1] - nodes[0]

    @staticmethod
    def _gh(K):
        xi, wk = hermegauss(K)
        return xi, wk / wk.sum()

    @staticmethod
    def _cell(nodes, t, ys, sigma, L):
        delta = nodes[1] - nodes[0]
        m, _, A = _post_stats(nodes, t, ys, L)
        if L is None:
            c = np.diff((0.5 * nodes**2 + A) / t)
        else:
            c = _cell_integrals(nodes, t, ys, L)
        R = int(np.ceil(8 * sigma / delta)) + 1
        off = (np.arange(-R, R + 1) - 0.5) * delta
        g = np.exp(-0.5 * (off / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        G = nodes.size
        mt = fftconvolve(c, g)[R:R + G]
        dmt = fftconvolve(c, -off / sigma**2 * g)[R:R + G]
        zl, zr = (nodes[0] - nodes) / sigma, (nodes[-1] - nodes) / sigma
        phi = lambda u: np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        mt += m[0] * ndtr(zl) + m[-1] * (1 - ndtr(zr))
        dmt += -m[0] * phi(zl) / sigma + m[-1] * phi(zr) / sigma
        return mt, dmt

    def _direct(self, x):
        m, dm, _ = _post_stats((x[..., None] + self.sigma * self.xi).ravel(), self.t, self.ys, self.L)
        return m.reshape(x.shape + (-1,)) @ self.wk, dm.reshape(x.shape + (-1,)) @ self.wk

    def m(self, x):
        if self.branch == "const":
            return np.full(np.shape(x), self.mean)
        if self.branch == "direct":
            return self._direct(x)[0]
        return self.spline(np.clip(x, self.lo, self.hi))

    def dm(self, x):
        if self.branch == "const":
            return np.zeros(np.shape(x))
        if self.branch == "direct":
            return self._direct(x)[1]
        inside = (x >= self.lo) & (x <= self.hi)
        return np.where(inside, self.spline(np.clip(x, self.lo, self.hi), 1), 0.0)


def h_schedule(times, h_t):
    """h(t) from values at time points: sqrt(h) linear in tau = -log(1 - t)."""
    sg, rh = -np.log1p(-np.asarray(times, float)), np.sqrt(np.asarray(h_t, float))
    return lambda t: float(np.interp(-np.log1p(-t), sg, rh)) ** 2


class GabrielVelocity:
    """u_h(x, t) = ((G_sigma * m_N)(x) - x)/(1 - t), du_h/dx = ((G_sigma * m_N)'(x) - 1)/(1 - t),
    sigma^2 = t^2 h, 1D, data in original units (m_N of the truncated latent with half-width L,
    given as L or latent, or Gaussian with neither). h_of_t: callable t -> h or a constant; cached per t.
    The node grid of the smoothed mean covers t [-Y, Y] + (1 - t) [-Lz, Lz], Y = max(1, max |y_i|),
    so data outside [-1, 1] are supported; non-finite data raise
    ValueError."""

    def __init__(self, samples, h_of_t, L=None, latent=None, **kw):
        self.ys = np.sort(np.asarray(samples, float).ravel())
        if not np.all(np.isfinite(self.ys)):
            raise ValueError("GabrielVelocity: samples must be finite")
        self.h_of_t = h_of_t if callable(h_of_t) else (lambda t, h=float(h_of_t): h)
        if latent is not None:
            L = latent.L
        self.L = None if L is None else float(L)
        self.latent = latent
        self.dim = 1
        self.kw = kw
        self._cache = {}

    def at(self, t):
        t = float(t)
        if t not in self._cache:
            if len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[t] = _Smoothed(self.ys, t, self.h_of_t(t), self.L, **self.kw)
        return self._cache[t]

    def __call__(self, x, t):
        x = as_points(x, 1)[..., 0].ravel()
        return ((self.at(t).m(x) - x) / (1 - t))[:, None]

    def div(self, x, t):
        x = as_points(x, 1)[..., 0].ravel()
        return (self.at(t).dm(x) - 1) / (1 - t)


def gabriel_val_loss(smoothed_m, yv, t, dx=5e-4, L=None):
    """Held-out FM loss (1/n_val) sum_i E_z |u(x_t) - (y_i - z)|^2, the latent expectation by
    the midpoint rule on [-Lz, Lz] (Lz = 7, or L for the truncated latent) with
    J = min(ceil(2 Lz (1-t)/dx), 40000) + 32 nodes and normalized Gaussian weights."""
    yv = np.asarray(yv, float).ravel()
    Lz = 7.0 if L is None else float(L)
    J = int(min(np.ceil(2 * (1 - t) * Lz / dx), 40000)) + 32
    z = -Lz + (np.arange(J) + 0.5) * 2 * Lz / J
    wz = np.exp(-0.5 * z**2)
    wz /= wz.sum()
    X = (1 - t) * z[None] + t * yv[:, None]
    T = yv[:, None] - z[None]
    r = (smoothed_m(X.ravel()).reshape(X.shape) - X) / (1 - t) - T
    return float(np.sum(wz[None] * r**2) / yv.size)


class StdGabriel:
    """GabrielVelocity in the standardized frame (1D), through the time change of
    AffineTarget: with a = s + t (1 - s), tau = t/a, x_tau = (s x' + t c)/a,
        u'(x', t) = ((m~(x_tau, tau) - c)/s - x')/(1 - t),  du'/dx' = (m~'(x_tau, tau)/a - 1)/(1 - t),
    smoothing by G_{t sqrt h'} in x' equals G_{tau sqrt h} in x_tau with h = s^2 h'. The truncated
    latent keeps its L under the time change. h_std: constant or callable t -> h'."""

    def __init__(self, ys, h_std, frame, latent=None):
        self.frame = frame
        self.latent = latent
        self.dim = 1
        s = frame.s
        hf = h_std if callable(h_std) else (lambda t, h=float(h_std): h)
        self.g = GabrielVelocity(np.asarray(ys, float),
                                 lambda tau: s * s * hf(s * tau / (1.0 - tau * (1.0 - s))),
                                 L=None if latent is None else latent.L)

    def _change(self, x, t):
        s, c = self.frame.s, float(self.frame.c[0])
        a = s + t * (1.0 - s)
        x = np.asarray(x, float).reshape(-1)
        return (s * x + t * c) / a, t / a, a

    def __call__(self, x, t):
        xt, tau, _ = self._change(x, t)
        m = self.g.at(tau).m(xt)
        x = np.asarray(x, float).reshape(-1)
        return (((m - self.frame.c[0]) / self.frame.s - x) / (1.0 - t))[:, None]

    def div(self, x, t):
        xt, tau, a = self._change(x, t)
        return (self.g.at(tau).dm(xt) / a - 1.0) / (1.0 - t)

    def val_loss(self, yv_std, t) -> float:
        """Held-out FM loss at t in the frame (latent step mapped by a / s)."""
        s, c = self.frame.s, float(self.frame.c[0])
        a = s + t * (1.0 - s)
        S = self.g.at(t / a)
        dx = 5e-4 if S.branch in ("const", "direct") else min(5e-4, S.dx)
        return gabriel_val_loss(lambda x: (S.m((s * x + t * c) / a) - c) / s,
                                np.asarray(yv_std, float).ravel(), t, dx * a / s, self.g.L)
