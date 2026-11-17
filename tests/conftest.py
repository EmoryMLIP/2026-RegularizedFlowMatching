"""Shared references and API adapters for the smoothfm release tests.

Every expected value in this suite is computed here or in the test files from the
mathematics of main.tex and supplement.tex, never from the package.
mpmath (50 digits) and dense composite Gauss-Legendre (GL) quadrature supply the
references.

API adapters. The tests call the package through the small adapter functions at the bottom of this file, so that aligning a
name or an argument order is a one-line change here. The readings chosen are listed
in each adapter's docstring.

Tiers. Tests marked ``@pytest.mark.slow`` are excluded by ``pytest -m "not slow"``,
the fast tier (target: under 5 min on 2 cores).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math

import mpmath as mp
import numpy as np
import pytest
import scipy.sparse as sp
from scipy import special, stats


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long test, excluded from the fast CI tier")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def pick(obj, *names, what=None):
    """First attribute of obj among names; fail with the list if none exists."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    pytest.fail(f"{what or 'attribute'}: none of {names} on {obj!r}; align the adapter "
                "in tests/conftest.py or the name in the package")


def value(x):
    """Property or zero-argument method: return its value."""
    return x() if callable(x) else x


def dense(A):
    return A.toarray() if sp.issparse(A) else np.asarray(A, dtype=float)


LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
_GL = {k: np.polynomial.legendre.leggauss(k) for k in (8, 10, 16)}


def gl(a, b, hmax=None, npts=10, breaks=(), minpan=1):
    """Composite GL nodes and weights on [a, b] (empty if b <= a), split at breaks."""
    if not b > a:
        return np.zeros(0), np.zeros(0)
    pts = [a, b] + [float(c) for c in breaks if a < c < b]
    pts = np.unique(pts)
    edges = [pts[0]]
    for lo, hi in zip(pts[:-1], pts[1:]):
        k = minpan if hmax is None else max(minpan, int(math.ceil((hi - lo) / hmax)))
        edges.extend(list(np.linspace(lo, hi, k + 1)[1:]))
    e = np.asarray(edges)
    g, w = _GL[npts]
    lo, hi = e[:-1, None], e[1:, None]
    x = (lo + hi) / 2 + (hi - lo) / 2 * g[None, :]
    ww = (hi - lo) / 2 * w[None, :]
    return x.ravel(), ww.ravel()


def gl_rows(lo, hi, npan=24, npts=8):
    """Row-wise composite GL: lo, hi (J,) -> nodes, weights (J, npan * npts).

    Empty rows (hi <= lo) get zero weights."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    ok = hi > lo
    hi = np.where(ok, hi, lo)
    g, w = _GL[npts]
    s = np.linspace(0.0, 1.0, npan + 1)
    a = lo[:, None] + (hi - lo)[:, None] * s[None, :-1]
    b = lo[:, None] + (hi - lo)[:, None] * s[None, 1:]
    x = (a + b)[:, :, None] / 2 + (b - a)[:, :, None] / 2 * g
    ww = (b - a)[:, :, None] / 2 * w
    ww = np.where(ok[:, None, None], ww, 0.0)
    return x.reshape(len(lo), -1), ww.reshape(len(lo), -1)


def lse(a, axis=None):
    return special.logsumexp(a, axis=axis)


# ---------------------------------------------------------------------------
# Truncated standard normal: mpmath references
# ---------------------------------------------------------------------------

MP_DPS = 50


def mp_tmoment(a, b, j):
    """int_a^b u^j phi(u) du in mpmath (a <= b finite or infinite), exact recurrence.

    I_j = [-u^{j-1} phi(u)]_a^b + (j - 1) I_{j-2}, I_0 = Phi(b) - Phi(a)."""
    with mp.workdps(MP_DPS):
        a, b = mp.mpf(a), mp.mpf(b)
        if not b > a:
            return mp.mpf(0)
        phi = lambda u: mp.npdf(u) if mp.isfinite(u) else mp.mpf(0)
        if a >= 0:
            i0 = mp.ncdf(-a) - mp.ncdf(-b)
        else:
            i0 = mp.ncdf(b) - mp.ncdf(a)
        i1 = phi(a) - phi(b)
        if j == 0:
            return i0
        if j == 1:
            return i1
        I = [i0, i1]
        for k in range(2, j + 1):
            pa = a ** (k - 1) * phi(a) if mp.isfinite(a) else mp.mpf(0)
            pb = b ** (k - 1) * phi(b) if mp.isfinite(b) else mp.mpf(0)
            I.append(pa - pb + (k - 1) * I[k - 2])
        return I[j]


def mp_ZL(L):
    with mp.workdps(MP_DPS):
        return mp.erf(mp.mpf(L) / mp.sqrt(2))


def ZL(L):
    return float(special.erf(L / math.sqrt(2)))


def trunc_normal_ppf(q, L):
    """Reference ppf of N(0,1) | [-L, L] (float, scipy)."""
    return stats.truncnorm(-L, L).ppf(q)


def trunc_normal_sample(n, D, L, rng):
    return stats.truncnorm(-L, L).rvs(size=(n, D), random_state=rng)


# ---------------------------------------------------------------------------
# Cubic B-spline basis on [-L, L] (paper S:sec:splinedetails)
# ---------------------------------------------------------------------------

def bspline(u, deriv=0):
    """Centred cardinal cubic B-spline (support [-2, 2]) and derivatives 1, 2, 3."""
    u = np.asarray(u, dtype=float)
    a = np.abs(u)
    s = np.sign(u)
    out = np.zeros_like(u)
    i = a < 1
    o = (a >= 1) & (a < 2)
    if deriv == 0:
        out[i] = 2 / 3 - a[i] ** 2 + a[i] ** 3 / 2
        out[o] = (2 - a[o]) ** 3 / 6
    elif deriv == 1:
        out[i] = s[i] * (-2 * a[i] + 1.5 * a[i] ** 2)
        out[o] = -s[o] * (2 - a[o]) ** 2 / 2
    elif deriv == 2:
        out[i] = -2 + 3 * a[i]
        out[o] = 2 - a[o]
    elif deriv == 3:
        out[i] = 3 * s[i]
        out[o] = -s[o]
    else:
        raise ValueError(deriv)
    return out


def centres(m, L):
    """Centres x_a = -L + a h, a = -1..m+1 (n = m + 3), h = 2L/m."""
    h = 2.0 * L / m
    return -L + h * np.arange(-1, m + 2), h


def design1d(m, L, x, deriv=0):
    """Reference design matrix d^k b_a/dx^k at physical x (no restriction applied)."""
    c, h = centres(m, L)
    u = (np.asarray(x, float).reshape(-1)[:, None] - c[None, :]) / h
    return bspline(u, deriv) / h ** deriv


def design2d(m, L, X, dx=0, dy=0):
    """Tensor design, index a n + a' (second axis fastest), physical derivatives."""
    X = np.asarray(X, float).reshape(-1, 2)
    B1 = design1d(m, L, X[:, 0], dx)
    B2 = design1d(m, L, X[:, 1], dy)
    return (B1[:, :, None] * B2[:, None, :]).reshape(len(X), -1)


def gram_ref(m, L, k, npts=10):
    """(P_k)_{aa'} = int_{-L}^{L} b_a^(k) b_a'^(k) dx by GL on every knot cell (exact)."""
    knots = np.linspace(-L, L, m + 1)
    x, w = gl(-L, L, npts=npts, breaks=knots)
    Bk = design1d(m, L, x, k)
    return (Bk * w[:, None]).T @ Bk


# ---------------------------------------------------------------------------
# Truncated empirical field (eq:empirical with the truncated latent)
# ---------------------------------------------------------------------------

def log_kernel(u, sig, L):
    """log K_sig(u) for the truncated latent; u (..., D). -inf outside the box."""
    D = u.shape[-1]
    z = u / sig
    lk = (-0.5 * z ** 2 - LOG_SQRT_2PI).sum(-1) - D * math.log(sig) - D * math.log(ZL(L))
    return np.where(np.all(np.abs(z) <= L, axis=-1), lk, -np.inf)


def empirical_ref(samples, x, t, L):
    """rho_N, weights, m_N, v_N at x (P, D); NaN weights where rho_N = 0."""
    samples = np.asarray(samples, float)
    x = np.asarray(x, float)
    sig = 1.0 - t
    lk = log_kernel(x[:, None, :] - t * samples[None, :, :], sig, L)   # (P, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = lk.max(axis=1, keepdims=True)
        e = np.exp(lk - mx)
        w = e / e.sum(axis=1, keepdims=True)
        rho = np.exp(mx[:, 0]) * e.sum(axis=1) / len(samples)
    rho = np.where(np.isfinite(mx[:, 0]), rho, 0.0)
    m = w @ samples
    v = (m - x) / sig
    return rho, w, m, v


# ---------------------------------------------------------------------------
# Reference targets (paper sec:data; the truncated-latent posterior on the box B)
# ---------------------------------------------------------------------------

def _aggregate(logw, mean, cov):
    """Moments of a mixture of pieces with absolute log weights."""
    logw = np.asarray(logw, float)
    keep = np.isfinite(logw)
    if not keep.any():
        D = mean.shape[1]
        return np.full(D, np.nan), np.full((D, D), np.nan), -np.inf
    logw, mean, cov = logw[keep], mean[keep], cov[keep]
    M = logw.max()
    w = np.exp(logw - M)
    W = w.sum()
    m = (w[:, None] * mean).sum(0) / W
    d = mean - m
    C = (w[:, None, None] * (cov + d[:, :, None] * d[:, None, :])).sum(0) / W
    return m, C, M + math.log(W)


def _axis_gauss(c, s, xd, t, sig, L, lo, hi, npan=24):
    """Per row j: int over [lo_j, hi_j] of N(y; c_j, s_j^2) phi((xd - t y)/sig)/(sig Z_L).

    Returns log integral, mean and variance of y under the normalized integrand."""
    c = np.asarray(c, float)
    s = np.broadcast_to(np.asarray(s, float), c.shape)
    P = 1 / s ** 2 + t ** 2 / sig ** 2
    mu = (c / s ** 2 + t * xd / sig ** 2) / P
    r = 14 / np.sqrt(P)
    a = np.maximum(lo, mu - r)
    b = np.minimum(hi, mu + r)
    y, w = gl_rows(a, b, npan)
    lf = (-0.5 * ((y - c[:, None]) / s[:, None]) ** 2 - LOG_SQRT_2PI - np.log(s)[:, None]
          - 0.5 * ((xd - t * y) / sig) ** 2 - LOG_SQRT_2PI - math.log(sig * ZL(L)))
    return _moments_rows(y, w, lf)


def _axis_uniform(a0, a1, xd, t, sig, L, lo, hi, npan=24):
    """Per row: int over [lo, hi] ∩ [a0, a1] of phi((xd - t y)/sig)/(sig Z_L) dy."""
    cen = xd / t
    r = 14 * sig / t
    a = np.maximum(np.maximum(lo, a0), cen - r)
    b = np.minimum(np.minimum(hi, a1), cen + r)
    y, w = gl_rows(np.atleast_1d(a), np.atleast_1d(b), npan)
    lf = -0.5 * ((xd - t * y) / sig) ** 2 - LOG_SQRT_2PI - math.log(sig * ZL(L))
    return _moments_rows(y, w, lf)


def _moments_rows(y, w, lf):
    with np.errstate(divide="ignore", invalid="ignore"):
        lw = np.where(w > 0, np.log(np.where(w > 0, w, 1.0)) + lf, -np.inf)
        M = lw.max(axis=1, keepdims=True)
        e = np.where(np.isfinite(M), np.exp(lw - np.where(np.isfinite(M), M, 0)), 0.0)
        S = e.sum(axis=1)
        mean = (e * y).sum(1) / S
        var = (e * (y - mean[:, None]) ** 2).sum(1) / S
        logI = M[:, 0] + np.log(S)
    logI = np.where(S > 0, logI, -np.inf)
    return logI, np.nan_to_num(mean), np.nan_to_num(var)


class RefTarget:
    """Base: posterior pieces in base coordinates with a free kernel width sig.

    The posterior of y given x is q(y) ∝ pi(y) phi_D((x - t y)/sig) 1{|x - t y| <= sig L},
    and logp is log int pi(y) sig^-D phi_D((x - t y)/sig) / Z_L^D over that box; with
    sig = 1 - t this is log p_t(x). A frame y' = (y - c)/s is reduced to the base by
    x_eff = s x' + t c, sig_eff = s (1 - t) (see posterior_ref)."""

    D = 1

    def box(self, x, t, sig, L):
        x = np.atleast_1d(np.asarray(x, float))
        return (x - sig * L) / t, (x + sig * L) / t


class BetaMix1DRef(RefTarget):
    D = 1

    def __init__(self, w, a, b, hmax=0.004):
        self.w, self.a, self.b = map(np.asarray, (w, a, b))
        self.hmax = hmax

    def logpdf(self, y):
        y = np.asarray(y, float).reshape(-1)
        u = (y + 1) / 2
        with np.errstate(divide="ignore"):
            lp = lse(np.log(self.w)[:, None]
                     + stats.beta.logpdf(u[None, :], self.a[:, None], self.b[:, None]), axis=0)
        return np.where((u >= 0) & (u <= 1), lp - math.log(2), -np.inf)

    def cdf(self, y):
        u = (np.asarray(y, float) + 1) / 2
        return sum(wk * stats.beta.cdf(u, ak, bk) for wk, ak, bk in zip(self.w, self.a, self.b))

    def ppf(self, q):
        from scipy.optimize import brentq
        return np.array([brentq(lambda y: self.cdf(y) - qq, -1, 1, xtol=1e-15, rtol=1e-15)
                         for qq in np.atleast_1d(q)])

    def sample(self, n, rng):
        k = rng.choice(len(self.w), size=n, p=self.w / self.w.sum())
        u = rng.beta(self.a[k], self.b[k])
        return (2 * u - 1)[:, None]

    @property
    def mean(self):
        mu = self.a / (self.a + self.b)
        return np.array([2 * (self.w * mu).sum() - 1])

    @property
    def cov(self):
        mu = self.a / (self.a + self.b)
        v = self.a * self.b / ((self.a + self.b) ** 2 * (self.a + self.b + 1))
        e2 = (self.w * (v + mu ** 2)).sum()
        e1 = (self.w * mu).sum()
        return np.array([[4 * (e2 - e1 ** 2)]])

    def pieces(self, x, t, sig, L):
        x = float(np.asarray(x).reshape(-1)[0])
        l, u = (x - sig * L) / t, (x + sig * L) / t
        lo, hi = max(-1.0, l), min(1.0, u)
        y, w = gl(lo, hi, hmax=min(sig / t, self.hmax), breaks=[x / t], minpan=8)
        with np.errstate(divide="ignore"):
            lw = (np.log(w) + self.logpdf(y) - 0.5 * ((x - t * y) / sig) ** 2
                  - LOG_SQRT_2PI - math.log(sig * ZL(L)))
        return lw, y[:, None], np.zeros((len(y), 1, 1))


def TrimodalRef():
    return BetaMix1DRef([0.30, 0.45, 0.25], [8, 30, 24], [24, 28, 6])


def InhomRef():
    return BetaMix1DRef([0.8, 0.2], [4, 1500], [5, 500])


class IsoMixtureRef(RefTarget):
    """Mixture of N(c_k, s_k^2 I); separable per axis."""
    D = 2

    def __init__(self, w, c, s):
        self.w = np.asarray(w, float)
        self.c = np.asarray(c, float)
        self.s = np.broadcast_to(np.asarray(s, float), self.w.shape).copy()

    def logpdf(self, y):
        y = np.asarray(y, float).reshape(-1, 2)
        d2 = ((y[:, None, :] - self.c[None]) ** 2).sum(-1)
        return lse(np.log(self.w)[None] - 0.5 * d2 / self.s[None] ** 2
                   - np.log(2 * np.pi * self.s[None] ** 2), axis=1)

    def sample(self, n, rng):
        k = rng.choice(len(self.w), size=n, p=self.w / self.w.sum())
        return self.c[k] + self.s[k, None] * rng.normal(size=(n, 2))

    @property
    def mean(self):
        return (self.w[:, None] * self.c).sum(0) / self.w.sum()

    @property
    def cov(self):
        w = self.w / self.w.sum()
        m = self.mean
        d = self.c - m
        return (w[:, None, None] * (d[:, :, None] * d[:, None, :]
                                    + self.s[:, None, None] ** 2 * np.eye(2))).sum(0)

    def pieces(self, x, t, sig, L, prune=60.0):
        x = np.atleast_1d(np.asarray(x, float))
        l, u = self.box(x, t, sig, L)
        # prune pieces that cannot matter (the untruncated weight bounds the truncated one)
        A = sig ** 2 + t ** 2 * self.s ** 2
        approx = np.log(self.w) - 0.5 * ((x - t * self.c) ** 2).sum(1) / A - np.log(A)
        keep = approx > approx.max() - prune
        c, s, w = self.c[keep], self.s[keep], self.w[keep]
        out = [_axis_gauss(c[:, d], s, x[d], t, sig, L, l[d], u[d]) for d in range(2)]
        logw = np.log(w) + out[0][0] + out[1][0]
        mean = np.stack([out[0][1], out[1][1]], 1)
        cov = np.zeros((len(w), 2, 2))
        cov[:, 0, 0], cov[:, 1, 1] = out[0][2], out[1][2]
        return logw, mean, cov


def EightGaussiansRef():
    ang = np.arange(8) * np.pi / 4
    r0 = 4 / 1.414
    return IsoMixtureRef(np.full(8, 1 / 8), r0 * np.stack([np.cos(ang), np.sin(ang)], 1),
                         0.5 / 1.414)


class SpiralsRef(IsoMixtureRef):
    """y = (±zeta(u) + 0.15 eps)/3, u = 3 pi sqrt(U), zeta(u) = (-u cos u, u sin u).

    The density is an integral over u with weight 2u/(9 pi^2) (x 1/2 per sign); it is
    discretized here by composite GL in u with panels of 0.004 (8 nodes each), far
    finer than the curve's length scale, and treated as an isotropic mixture."""

    def __init__(self, hu=0.004):
        u, wu = gl(0.0, 3 * np.pi, hmax=hu, npts=8)
        z = np.stack([-u * np.cos(u), u * np.sin(u)], 1) / 3
        dens = 2 * u / (9 * np.pi ** 2)
        c = np.concatenate([z, -z])
        w = 0.5 * np.concatenate([dens * wu, dens * wu])
        super().__init__(w, c, 0.05)

    def sample(self, n, rng):
        u = 3 * np.pi * np.sqrt(rng.uniform(size=n))
        sgn = rng.choice([-1.0, 1.0], size=n)
        z = np.stack([-u * np.cos(u), u * np.sin(u)], 1)
        return (sgn[:, None] * z + 0.15 * rng.normal(size=(n, 2))) / 3


class CheckerboardRef(RefTarget):
    """Uniform on the 8 dark squares [2i, 2i+2) x [2j, 2j+2), i, j in -2..1, i + j even
    (FFJORD toy_data: x2 offset by floor(x1) % 2, all times 2)."""
    D = 2

    def __init__(self):
        sq = [(2 * i, 2 * j) for i in range(-2, 2) for j in range(-2, 2) if (i + j) % 2 == 0]
        self.sq = np.array(sq, float)

    def logpdf(self, y):
        y = np.asarray(y, float).reshape(-1, 2)
        ins = np.zeros(len(y), bool)
        for a, b in self.sq:
            ins |= (y[:, 0] >= a) & (y[:, 0] < a + 2) & (y[:, 1] >= b) & (y[:, 1] < b + 2)
        return np.where(ins, -math.log(32.0), -np.inf)

    def sample(self, n, rng):
        k = rng.integers(0, 8, size=n)
        return self.sq[k] + 2 * rng.uniform(size=(n, 2))

    @property
    def mean(self):
        return (self.sq + 1).mean(0)

    @property
    def cov(self):
        c = self.sq + 1
        d = c - c.mean(0)
        return (d[:, :, None] * d[:, None, :]).mean(0) + np.eye(2) / 3

    def pieces(self, x, t, sig, L):
        x = np.atleast_1d(np.asarray(x, float))
        l, u = self.box(x, t, sig, L)
        o = [_axis_uniform(self.sq[:, d], self.sq[:, d] + 2, x[d], t, sig, L,
                           np.full(8, l[d]), np.full(8, u[d])) for d in range(2)]
        logw = -math.log(32.0) + o[0][0] + o[1][0]
        mean = np.stack([o[0][1], o[1][1]], 1)
        cov = np.zeros((8, 2, 2))
        cov[:, 0, 0], cov[:, 1, 1] = o[0][2], o[1][2]
        return logw, mean, cov


class CircleRef(RefTarget):
    """y = r (cos th, sin th), r ~ N(2, 0.1^2), th ~ 0.5 vM(0,4) + 0.3 vM(2pi/3,16)
    + 0.2 vM(4pi/3,64). Polar quadrature: GL in theta, GL in r per theta."""
    D = 2
    R, S = 2.0, 0.1
    W = np.array([0.5, 0.3, 0.2])
    MU = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    KAP = np.array([4.0, 16.0, 64.0])

    def log_ftheta(self, th):
        th = np.asarray(th, float)
        return lse(np.log(self.W)[:, None]
                   + self.KAP[:, None] * (np.cos(th[None] - self.MU[:, None]) - 1)
                   - np.log(2 * np.pi * special.i0e(self.KAP))[:, None], axis=0)

    def logpdf(self, y):
        y = np.asarray(y, float).reshape(-1, 2)
        r = np.hypot(y[:, 0], y[:, 1])
        th = np.arctan2(y[:, 1], y[:, 0])
        return (self.log_ftheta(th) - 0.5 * ((r - self.R) / self.S) ** 2 - LOG_SQRT_2PI
                - math.log(self.S) - np.log(r))

    def sample(self, n, rng):
        k = rng.choice(3, size=n, p=self.W)
        th = stats.vonmises.rvs(self.KAP[k], loc=self.MU[k], random_state=rng)
        r = self.R + self.S * rng.normal(size=n)
        return r[:, None] * np.stack([np.cos(th), np.sin(th)], 1)

    def _trig(self, n):
        return (self.W * special.iv(n, self.KAP) / special.iv(0, self.KAP)
                * np.exp(1j * n * self.MU)).sum()

    @property
    def mean(self):
        e1 = self._trig(1)
        return self.R * np.array([e1.real, e1.imag])

    @property
    def cov(self):
        e2 = self._trig(2)
        er2 = self.R ** 2 + self.S ** 2
        Eyy = er2 * np.array([[(1 + e2.real) / 2, e2.imag / 2], [e2.imag / 2, (1 - e2.real) / 2]])
        m = self.mean
        return Eyy - np.outer(m, m)

    def pieces(self, x, t, sig, L, hth=0.004):
        x = np.atleast_1d(np.asarray(x, float))
        l, u = self.box(x, t, sig, L)
        thc = math.atan2(x[1], x[0])
        if l[0] <= 0 <= u[0] and l[1] <= 0 <= u[1]:
            d0, d1, br = -np.pi, np.pi, []
        else:
            cx = np.array([l[0], u[0], l[0], u[0]])
            cy = np.array([l[1], l[1], u[1], u[1]])
            dc = np.angle(np.exp(1j * (np.arctan2(cy, cx) - thc)))
            d0, d1, br = dc.min(), dc.max(), list(dc)
        dth, wth = gl(d0, d1, hmax=hth, npts=8, breaks=br, minpan=16)
        th = thc + dth
        e = np.stack([np.cos(th), np.sin(th)], 1)
        # r-interval of the box along e
        rlo = np.zeros(len(th))
        rhi = np.full(len(th), np.inf)
        for d in range(2):
            ed = e[:, d]
            with np.errstate(divide="ignore"):
                a, b = l[d] / ed, u[d] / ed
            lo_d = np.where(ed > 0, a, np.where(ed < 0, b, -np.inf))
            hi_d = np.where(ed > 0, b, np.where(ed < 0, a, np.inf))
            lo_d = np.where((ed == 0) & ~((l[d] <= 0) & (0 <= u[d])), np.inf, lo_d)
            rlo, rhi = np.maximum(rlo, lo_d), np.minimum(rhi, hi_d)
        xe = e @ x
        P = 1 / self.S ** 2 + t ** 2 / sig ** 2
        mu = (self.R / self.S ** 2 + t * xe / sig ** 2) / P
        rr = 14 / math.sqrt(P)
        a, b = np.maximum(rlo, mu - rr), np.minimum(rhi, mu + rr)
        r, wr = gl_rows(a, b, 24, 8)
        y1 = r * e[:, :1]
        y2 = r * e[:, 1:]
        lf = (-0.5 * ((r - self.R) / self.S) ** 2 - LOG_SQRT_2PI - math.log(self.S)
              - 0.5 * ((x[0] - t * y1) / sig) ** 2 - 0.5 * ((x[1] - t * y2) / sig) ** 2
              - 2 * LOG_SQRT_2PI - 2 * math.log(sig * ZL(L)))
        logI, mr, vr = _moments_rows(r, wr, lf)
        logw = self.log_ftheta(th) + np.log(wth) + logI
        mean = mr[:, None] * e
        cov = vr[:, None, None] * e[:, :, None] * e[:, None, :]
        return logw, mean, cov


def posterior_ref(ref, x, t, L, frame=None):
    """Brute-force (m, C, log p_t) of the truncated posterior at one point x (D,).

    frame = (c, s): the law of y' = (y - c)/s, evaluated at x' = x in frame units."""
    x = np.atleast_1d(np.asarray(x, float))
    D = ref.D
    if frame is None:
        c, s = np.zeros(D), 1.0
    else:
        c, s = np.broadcast_to(np.asarray(frame[0], float), (D,)), float(frame[1])
    if t == 0:
        m, C = np.asarray(ref.mean, float), np.asarray(ref.cov, float)
        lp = (-0.5 * x ** 2 - LOG_SQRT_2PI).sum() - D * math.log(ZL(L))
        if np.any(np.abs(x) > L):
            lp = -np.inf
        return (m - c) / s, C / s ** 2, lp
    xe = s * x + t * c
    sig = s * (1 - t)
    m, C, lp = _aggregate(*ref.pieces(xe, t, sig, L))
    return (m - c) / s, C / s ** 2, lp + D * math.log(s)


def path_points(ref, t, L, n, rng, edge=()):
    """x = (1 - t) z + t y with z ~ N(0, I) | Omega and y ~ pi, plus edge points whose z
    has the given coordinates (|z_d| > L - 0.5) and y ~ pi."""
    D = ref.D
    z = trunc_normal_sample(n, D, L, rng)
    y = ref.sample(n + len(edge), rng).reshape(-1, D)
    pts = list((1 - t) * z + t * y[:n])
    for k, ze in enumerate(edge):
        pts.append((1 - t) * np.asarray(ze, float) + t * y[n + k])
    return np.array(pts)


# ---------------------------------------------------------------------------
# Adapters to the package API (align here, nowhere else)
# ---------------------------------------------------------------------------

def latent(D, L=None):
    """TruncatedGaussian(dim, L); L defaults to config.L[D]."""
    from smoothfm import config
    from smoothfm.latent import TruncatedGaussian
    return TruncatedGaussian(D, config.L[D] if L is None else L)


def make_basis(D, m, L):
    """Cubic1D(m, half_width=L) and Cubic2D(m, m, L) (sec:solver, S:sec:splinedetails)."""
    from smoothfm.splines import Cubic1D, Cubic2D
    return Cubic1D(m, half_width=L) if D == 1 else Cubic2D(m, m, L)


def design(basis, x, deriv=0):
    """Design matrix at physical points x (n, D) with physical derivatives.

    Reading: ``eval(xi, deriv)`` takes reference coordinates xi = x / L in [-1, 1]^D and
    returns derivatives with respect to the physical x; in 2D deriv is a pair."""
    x = np.asarray(x, float)
    L = np.asarray(pick(basis, "half_width", "L", "ell"), float).reshape(-1)
    if x.ndim == 1:
        x = x[:, None]
    xi = x / L
    if x.shape[1] == 1:
        return dense(basis.eval(xi[:, 0], deriv))
    return dense(basis.eval(xi, deriv))


def posterior(target, x, t, L):
    """(m (n, D), C (n, D, D), log p_t (n,)) from target.posterior(x, t, L)."""
    x = np.atleast_2d(np.asarray(x, float))
    n, D = x.shape
    m, C, lp = target.posterior(x, t, L)
    lp = np.full(n, np.nan) if lp is None else np.asarray(lp, float).reshape(n)
    return (np.asarray(m, float).reshape(n, D), np.asarray(C, float).reshape(n, D, D), lp)


def normal_equations(basis, lat, samples, t, reg):
    from smoothfm.normal_equations import NormalEquations
    return NormalEquations(basis, lat, np.asarray(samples, float), t, reg)


def run_search(candidates_rel, rule, include_zero=True):
    """LambdaSearch with the candidates 10^j lam_s, j = lo..hi-1, relative to the rule's lam_s
    (rule.ne.lam_s); candidates_rel is (lo, hi) or the array of the factors 10^j.
    Selection.lam is absolute."""
    from smoothfm.selection import LambdaSearch
    if not isinstance(candidates_rel, tuple):
        j = np.round(np.log10(candidates_rel)).astype(int)
        assert np.array_equal(10.0 ** j, candidates_rel) and np.all(np.diff(j) == 1)
        candidates_rel = (int(j[0]), int(j[-1]) + 1)
    return LambdaSearch(candidates_rel, include_zero=include_zero)(rule)


def rule_lam_s(rule):
    if hasattr(rule, "lam_s"):
        return float(value(rule.lam_s))
    ne = pick(rule, "ne", "ne_fit", what="rule normal equations")
    return float(ne.lam_s)


# ---------------------------------------------------------------------------
# 2D time controls (tests/test_tend_2d.py, tests/test_integration_error_2d.py):
# smoke runs of the working tree, shared per session.
# ---------------------------------------------------------------------------

class SmokeRuns:
    """Smoke runs into one results root, each (script, args) once per session, after its
    dependencies."""

    def __init__(self, root):
        import pathlib
        self.root = pathlib.Path(root)
        self.done = {}

    def run(self, script, args=(), deps=(), timeout=1800):
        """deps: (script, args) pairs run first. Returns the CompletedProcess."""
        import pathlib
        import subprocess
        import sys
        key = (script, tuple(args))
        if key in self.done:
            return self.done[key]
        for ds, da in deps:
            r = self.run(ds, da)
            if r.returncode != 0:
                pytest.fail(f"dependency {ds} {' '.join(da)} failed:\n{r.stderr[-3000:]}")
        code = pathlib.Path(__file__).resolve().parents[1]
        e = dict(os.environ)
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
            e[v] = "1"
        e["NWORKERS"] = "2"
        e.pop("OVERWRITE", None)
        e["PYTHONPATH"] = os.pathsep.join([str(code / "src"), e.get("PYTHONPATH", "")])
        cmd = [sys.executable, str(code / "experiments" / "compute" / f"{script}.py"),
               "--smoke", *args, "--results", str(self.root), "--workers", "2"]
        r = subprocess.run(cmd, cwd=code, env=e, capture_output=True, text=True,
                           timeout=timeout)
        self.done[key] = r
        return r

    def npz(self, name):
        p = self.root / "smoke" / f"{name}.npz"
        assert p.is_file(), f"{p.name} missing (files: " \
                            f"{sorted(x.name for x in (self.root / 'smoke').glob('*'))})"
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


@pytest.fixture(scope="session")
def smoke_tree(tmp_path_factory):
    """Smoke runs of the working tree."""
    return SmokeRuns(tmp_path_factory.mktemp("smoke_tree") / "results")
