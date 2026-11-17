"""The latent distribution gamma: the standard Gaussian conditioned on Omega = [-L, L]^D
(main.tex sec:formulation, assumption (A1); sec:setup; S:sec:splinedetails, "Domains, Frames
and Latents").

Its density K, the kernel of eq:empirical, is a product of 1D factors:

    K(z) = prod_d k_L(z_d),   k_L(u) = phi(u) 1{|u| <= L} / Z_L,   Z_L = erf(L / sqrt 2),

with phi the standard normal density. Samples are z = F_L^{-1}(F_0(g)), g ~ N(0, 1)
coordinatewise, with F_0 = Phi and F_L the distribution function of k_L.

Every integral of a cubic spline against K_sigma(x - mu) over a grid cell reduces, after
u = (x - mu) / sigma, to the integral of a polynomial against k_L over a cell [alpha, beta]
in u; ``TruncatedGaussian.cell_rule`` supplies a positive quadrature rule for it (accurate to
5e-14 relative for polynomials of degree <= 6, measured against mpmath), the "Gauss-Legendre panels adapted to the decay of the kernel" of
S:sec:splinedetails. It keeps the data matrices accurate when a cell is wide compared with
the kernel (h >> sigma, late t).

Module functions shared by the targets: ``log_ndtr_diff`` and ``truncnorm_moments``; and by
the whole package: ``as_points`` and ``gauss_legendre01``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import erfc, erfcx, log_ndtr, ndtr, ndtri

__all__ = ["TruncatedGaussian", "as_points"]

_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)
_LOG2PI = np.log(2.0 * np.pi)
_LOG_SQRT2PI = 0.5 * _LOG2PI


def _phi(u):
    # dividing after exp avoids the rounding of a shifted exponent (relative ~u^2 eps)
    return np.exp(-0.5 * u * u) / _SQRT2PI


def as_points(x, dim):
    """Points as an array (..., dim); a 1D array is n points when dim = 1."""
    x = np.asarray(x, dtype=float)
    if dim == 1 and (x.ndim == 0 or x.shape[-1] != 1):
        x = x[..., None]
    if x.shape[-1] != dim:
        raise ValueError(f"expected points with last axis {dim}, got shape {x.shape}")
    return x


def gauss_legendre01(n):
    """Gauss-Legendre nodes x_q and weights W_q on [0, 1], and 1 - x_q.

    The nodes are symmetric, so 1 - x_q equals the reversed nodes exactly; returning it
    avoids the cancellation in 1 - x_q for nodes near 1.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    x01 = (x + 1.0) / 2.0
    x01 = 0.5 * (x01 + (1.0 - x01[::-1]))
    return x01, w / 2.0, x01[::-1].copy()


def _ranges(counts):
    """Concatenation of arange(c) for c in counts."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(total) - starts


class CellRule(NamedTuple):
    """Flat quadrature rule for many cells [alpha_k, beta_k] at once.

    Node q belongs to cell ``cell[q]``, sits at local coordinate ``tau[q]`` in [0, 1]
    (``taubar[q] = 1 - tau[q]`` without cancellation) and has weight ``weight[q] > 0`` with

        sum_{q: cell[q] = k} weight[q] g(tau[q]) = int_{alpha_k}^{beta_k} g((u - alpha_k)/delta_k) k_L(u) du

    up to a relative error of at most 5e-14 (measured against mpmath over test cells) for
    polynomials g of degree <= 6, delta_k = beta_k - alpha_k. The weight is not polynomial,
    so the rule is accurate, not exact.
    """

    cell: np.ndarray
    tau: np.ndarray
    taubar: np.ndarray
    weight: np.ndarray


# ------------------------------------------------------------------ truncated normal

def log_ndtr_diff(a, b):
    """log(Phi(b) - Phi(a)) for a <= b, without cancellation.

    Where a > 0 the interval is reflected, Phi(b) - Phi(a) = Phi(-a) - Phi(-b), so both
    log_ndtr arguments lie on the side where log_ndtr is accurate; then
    log(Phi(b) - Phi(a)) = log Phi(b) + log(-expm1(log Phi(a) - log Phi(b))).
    """
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    flip = a > 0
    lo = np.where(flip, -b, a)
    hi = np.where(flip, -a, b)
    lh, ll = log_ndtr(hi), log_ndtr(lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        return lh + np.log(-np.expm1(ll - lh))


def _mills_tail(p):
    """(c, d) with R(p) = phi(p)/Q(p) = p + c, c = 1/(p + d), d = 2/(p + 3/(p + ...)), Q = 1 - Phi:
    the continued fraction of the Mills ratio, 400 terms, for p >= 1 (relative error <= 5e-16
    against mpmath at p = 1.0001)."""
    p = np.asarray(p, float)
    big = p >= 1.0
    K = np.zeros_like(p)
    pb = np.where(big, p, 4.0)
    for k in range(400, 1, -1):
        K = k / (pb + K)
    d_cf = K
    c_cf = 1.0 / (pb + d_cf)
    ps = np.where(big, 2.0, p)
    R = np.sqrt(2.0 / np.pi) / erfcx(ps / _SQRT2)
    c_d = R - ps
    d_d = 1.0 / c_d - ps
    return np.where(big, c_cf, c_d), np.where(big, d_cf, d_d)


def _tail_moments(p, q, w):
    """(log Z, E[v] - p, Var v) of N(0, 1) on [p, q], 1 < p < q <= inf, w = q - p, written with
    the Mills-ratio remainders c, d of ``_mills_tail`` so that nothing cancels:

        rho = Q(q)/Q(p) = exp(-w (p + q)/2) (p + c_p)/(q + c_q),   log Z = log Q(p) + log(1 - rho),
        E[v] - p = (c_p - rho (w + c_q)) / (1 - rho),
        E[(v - p)^2] = (d_p/(p + d_p) - rho (1 + w^2 - c_q (p - w))) / (1 - rho).
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cp, dp = _mills_tail(p)
        fin = np.isfinite(q)
        qf = np.where(fin, q, p + 1.0)
        cq, _ = _mills_tail(qf)
        wf = np.where(fin, w, 1.0)
        rho = np.where(fin, np.exp(-0.5 * wf * (p + qf)) * (p + cp) / (qf + cq), 0.0)
        logQp = -0.5 * p * p - _LOG_SQRT2PI - np.log(p + cp)
        logZ = logQp + np.log1p(-rho)
        Ea = np.where(fin, (cp - rho * (wf + cq)) / (1.0 - rho), cp)
        s2 = np.where(fin, (dp / (p + dp) - rho * (1.0 + wf * wf - cq * (p - wf))) / (1.0 - rho),
                      dp / (p + dp))
    return logZ, Ea, s2 - Ea * Ea


_GL64 = np.polynomial.legendre.leggauss(64)


def _gl_moments(p, w):
    """(log Z, E[s], Var s) of N(0, 1) on [p, p + w], s = v - p, by 64-point Gauss-Legendre
    of exp(-p s - s^2/2) on [0, w] (accurate when p w + w^2/2 <= 30)."""
    x, wt = _GL64
    s = 0.5 * w[:, None] * (x[None] + 1.0)
    W = 0.5 * w[:, None] * wt[None]
    f = W * np.exp(-p[:, None] * s - 0.5 * s * s)
    Z = f.sum(1)
    Es = (f * s).sum(1) / Z
    d = s - Es[:, None]
    return -0.5 * p * p - _LOG_SQRT2PI + np.log(Z), Es, (f * d * d).sum(1) / Z


def truncnorm_moments(a, b, w=None):
    """(log Z, E[u] - a, Var u) of N(0, 1) truncated to [a, b], a <= b elementwise (ends may be
    infinite):

        Z = Phi(b) - Phi(a),  E[u] = (phi(a) - phi(b)) / Z,
        Var u = 1 + (a phi(a) - b phi(b)) / Z - E[u]^2.

    w: the width b - a when the caller knows it more accurately than the difference of a
    and b. Three regimes:
    (i) generic: the formulas above with phi/Z formed in log space;
    (ii) both ends in one tail (after reflection [p, q] with p > 1): Mills-ratio remainders
         (``_tail_moments``), relative accuracy kept for |a| up to 1e5 and beyond, also
         one-sided (q = inf);
    (iii) narrow (w max(1, |c|) < 1e-2, c = a + w/2): the Taylor series of phi about c,
        Z = w phi(c) [1 + He2 w^2/24 + He4 w^4/1920],
        E[u] - a = w/2 - (c w^2/12 + He3 w^4/480) / [.],
        E[(u - c)^2] = (w^2/12 + He2 w^4/160) / [.],   error O((w max(1, |c|))^6).
    An empty interval (w = 0) gives log Z = -inf. The offset E[u] - a is returned so that
    a + (E[u] - a) keeps the width information.
    """
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    w = b - a if w is None else np.broadcast_to(np.asarray(w, float), a.shape)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logZ = log_ndtr_diff(a, b)
        ra = np.where(np.isfinite(a), np.exp(-0.5 * a * a - _LOG_SQRT2PI - logZ), 0.0)
        rb = np.where(np.isfinite(b), np.exp(-0.5 * b * b - _LOG_SQRT2PI - logZ), 0.0)
        ta = np.where(np.isfinite(a), a * ra, 0.0)
        tb = np.where(np.isfinite(b), b * rb, 0.0)
        Eu = ra - rb
        Var = 1.0 + ta - tb - Eu * Eu
        Ea = Eu - a
    # (ii) one tail: reflect so that the interval is [p, q], p > 1
    flip = a > 0
    p = np.where(flip, a, -b)
    q = np.where(flip, b, -a)
    tail = (p > 1.0) & (w > 0)
    if tail.any():
        lz, ev, vv = _tail_moments(p[tail], q[tail], w[tail])
        logZ, Ea, Var = logZ.copy(), Ea.copy(), Var.copy()
        logZ[tail] = lz
        # flipped: [a, b] = [p, q], E[u] - a = E[v] - p; else u = -v, E[u] - a = w - (E[v] - p)
        with np.errstate(invalid="ignore"):
            Ea[tail] = np.where(flip[tail], ev, w[tail] - ev)
        Var[tail] = vv
    # (ii') one-signed intervals over which log phi drops by at most 30 (E = p w + w^2/2):
    # Gauss-Legendre (64 nodes) in s = v - p of exp(-p s - s^2/2), which keeps the moments of
    # moderately narrow intervals that the closed forms lose to cancellation
    with np.errstate(invalid="ignore"):
        Etot = np.where(np.isfinite(w), p * w + 0.5 * w * w, np.inf)
    mod = (p >= 0) & (w > 0) & (Etot <= 30.0)
    if mod.any():
        lz, ev, vv = _gl_moments(p[mod], w[mod])
        logZ, Ea, Var = logZ.copy(), Ea.copy(), Var.copy()
        logZ[mod] = lz
        Ea[mod] = np.where(flip[mod], ev, w[mod] - ev)
        Var[mod] = vv
    with np.errstate(invalid="ignore"):
        c = a + 0.5 * w
    nar = np.isfinite(w) & np.isfinite(c) & (w * np.maximum(1.0, np.abs(c)) < 1e-2) & (w > 0)
    if nar.any():
        wn, cn = w[nar], c[nar]
        c2, wn2 = cn * cn, wn * wn
        he2, he3, he4 = c2 - 1.0, cn * (c2 - 3.0), c2 * c2 - 6.0 * c2 + 3.0
        den = 1.0 + he2 * wn2 / 24.0 + he4 * wn2 * wn2 / 1920.0
        logZ, Ea, Var = logZ.copy(), Ea.copy(), Var.copy()
        logZ[nar] = np.log(wn) - 0.5 * c2 - _LOG_SQRT2PI + np.log(den)
        d1 = -(cn * wn2 / 12.0 + he3 * wn2 * wn2 / 480.0) / den
        Ea[nar] = 0.5 * wn + d1
        Var[nar] = (wn2 / 12.0 + he2 * wn2 * wn2 / 160.0) / den - d1 * d1
    empty = ~(w > 0)
    if empty.any():
        logZ, Ea, Var = (np.where(empty, -np.inf, logZ), np.where(empty, 0.0, Ea),
                         np.where(empty, 0.0, Var))
    return logZ, Ea, np.maximum(Var, 0.0)


# ------------------------------------------------------------------ the latent

class TruncatedGaussian:
    """The latent gamma: N(0, I_D) conditioned on Omega = [-L, L]^D (A1), with density
    K(z) = prod_d phi(z_d) 1{|z_d| <= L} / Z_L.

    TV(N(0, I), gamma) = 1 - Z_L^D (``tv_to_gaussian``; 5.7e-7 at L = 5, D = 1; 1.4e-5 at
    L = 4.5, D = 2; sec:setup). Draws are monotone images of standard normal draws,
    z = F_L^{-1}(F_0(g)) (``from_normal``), so the generator is consumed as for N(0, I) and
    every sample lies in Omega.
    """

    def __init__(self, dim: int, L: float):
        L = float(L)
        if not (np.isfinite(L) and L > 0):
            raise ValueError(f"L must be a finite float > 0, got {L!r}")
        self.dim = int(dim)
        self.L = L
        e = float(erfc(L / _SQRT2))                 # 2 Phi(-L), without the cancellation in 1 - Z
        self.Z = 1.0 - e
        self.p_lo = 0.5 * e                         # Phi(-L)
        self._logZ = float(np.log1p(-e))
        self.tv_to_gaussian = float(-np.expm1(self.dim * self._logZ))

    def __repr__(self):
        return f"TruncatedGaussian(dim={self.dim}, L={self.L:g})"

    @property
    def box(self) -> np.ndarray:
        """Omega as (D, 2) intervals."""
        return np.tile([-self.L, self.L], (self.dim, 1))

    # -- density ---------------------------------------------------------------------
    def logkernel(self, u):
        """log k_L(u) elementwise (-inf outside [-L, L])."""
        u = np.asarray(u, float)
        return np.where(np.abs(u) <= self.L, -0.5 * u * u - _LOG_SQRT2PI - self._logZ, -np.inf)

    def logpdf(self, z):
        """log K(z) = sum_d log k_L(z_d), z of shape (..., D)."""
        return self.logkernel(as_points(z, self.dim)).sum(axis=-1)

    def pdf(self, z):
        """K(z)."""
        return np.exp(self.logpdf(z))

    def cdf(self, u):
        """F_L(u) = (Phi(u) - Phi(-L)) / Z_L elementwise, from the tail nearer u (no cancellation)."""
        u = np.asarray(u, float)
        a = np.clip(-np.abs(u), -self.L, 0.0)
        with np.errstate(divide="ignore"):
            lower = np.where(a > -self.L, np.exp(log_ndtr_diff(-self.L, a)), 0.0) / self.Z
        return np.where(u > 0, 1.0 - lower, lower)

    def ppf(self, q):
        """F_L^{-1}(q) = Phi^{-1}(Phi(-L) + q Z_L) on the half with the smaller tail
        (q -> min(q, 1 - q), then the sign), clipped to [-L, L] against rounding."""
        q = np.asarray(q, float)
        lo = q < 0.5
        u = ndtri(self.p_lo + np.where(lo, q, 1.0 - q) * self.Z)
        return np.clip(np.where(lo, u, -u), -self.L, self.L)

    def from_normal(self, g):
        """F_L^{-1}(Phi(g)) for standard normal values g, elementwise (Phi taken of -|g|,
        accurate in the tail, then the sign)."""
        g = np.asarray(g, float)
        u = ndtri(self.p_lo + ndtr(-np.abs(g)) * self.Z)
        return np.clip(np.where(g > 0, -u, u), -self.L, self.L)

    def sample(self, n, rng=None):
        """n draws (n, D): from_normal of rng.standard_normal((n, D))."""
        rng = np.random.default_rng(rng)
        return self.from_normal(rng.standard_normal((int(n), self.dim)))

    def quantile_midpoints(self, M):
        """The M quantile midpoints F_L^{-1}((k - 1/2)/M), k = 1..M, as points (M, 1): the 1D
        latent points of the flows (sec:setup)."""
        if self.dim != 1:
            raise ValueError("quantile midpoints are the 1D latent")
        return self.ppf((np.arange(int(M)) + 0.5) / int(M))[:, None]

    def sobol(self, M, seed):
        """M scrambled Sobol points in [0, 1]^D (scipy.stats.qmc.Sobol, rng = seed) mapped
        coordinatewise by F_L^{-1}: the 2D latent points of the flows (sec:setup)."""
        from scipy.stats import qmc
        return self.ppf(qmc.Sobol(self.dim, scramble=True, rng=seed).random(int(M)))

    # -- cell rule -------------------------------------------------------------------
    # Panel parameters of the Gaussian rule, measured against mpmath (worst relative error
    # over test cells): (dE, nodes) = (2, 10): 5e-14, (8, 16): 7e-15.
    _dE_short, _n_short, _dE, _n_long = 2.0, 10, 8.0, 16

    def cell_rule(self, alpha, beta) -> CellRule:
        """Positive rule for int_alpha^beta g(tau(u)) k_L(u) du (see CellRule), accurate to
        5e-14 relative for polynomial g of degree <= 6 (measured against mpmath).

        The cell is cut to [a, b] = [alpha, beta] ∩ [-L, L] (empty: no nodes), split at the
        mode u = 0, and each part runs away from the mode, where phi(a + s) = phi(a) exp(-E(s)),
        E(s) = a s + s^2/2: one panel of 10 Gauss-Legendre nodes when E <= 2 over the part,
        else panels of equal increments E <= 8 with 16 nodes. Weights are divided by Z_L,
        and the local coordinates refer to the uncut cell [alpha, beta].
        """
        alpha = np.asarray(alpha, float).ravel()
        beta = np.asarray(beta, float).ravel()
        a = np.clip(alpha, -self.L, self.L)
        b = np.clip(beta, -self.L, self.L)
        r = self._gauss_rule(a, b)
        k = r.cell
        clipped = ((a != alpha) | (b != beta))[k]
        dc, dl = (b - a)[k], (beta - alpha)[k]
        tau = np.where(clipped, ((a - alpha)[k] + r.tau * dc) / dl, r.tau)
        taubar = np.where(clipped, ((beta - b)[k] + r.taubar * dc) / dl, r.taubar)
        return CellRule(k, tau, taubar, r.weight / self.Z)

    def _gauss_rule(self, alpha, beta) -> CellRule:
        """The rule for the untruncated phi on cells [alpha, beta]."""
        delta = beta - alpha
        u0 = np.clip(0.0, alpha, beta)
        K = alpha.size
        S = np.concatenate([beta - u0, u0 - alpha])
        a = np.concatenate([u0, -u0])
        near = np.concatenate([u0 - alpha, beta - u0])
        is_right = np.concatenate([np.ones(K, bool), np.zeros(K, bool)])
        cell_of = np.concatenate([np.arange(K), np.arange(K)])
        keep = (S > 0) & (a < 39.0)
        S, a, near, is_right, cell_of = (v[keep] for v in (S, a, near, is_right, cell_of))
        Etot = a * S + 0.5 * S * S
        # beyond E* the neglected mass is below 1e-17 of any nonnegative degree-6 integrand
        Estar = 40.0 + 7.0 * np.log(np.maximum(Etot, 1.0))
        trunc = Etot > Estar
        Eend = np.where(trunc, Estar, Etot)
        short = Eend <= self._dE_short
        rules = []
        for sel, dE, nq in ((short, self._dE_short, self._n_short), (~short, self._dE, self._n_long)):
            if sel.any():
                rules.append(_part_rule(S[sel], a[sel], near[sel], is_right[sel], cell_of[sel],
                                        Eend[sel], trunc[sel], delta, dE, nq))
        if not rules:
            z = np.zeros(0)
            return CellRule(np.zeros(0, np.int64), z, z, z)
        return CellRule(*(np.concatenate(f) for f in zip(*rules)))


def _part_rule(S, a, near, is_right, cell_of, Eend, trunc, delta, dE, nq):
    """Composite Gauss-Legendre (nq nodes) on panels of equal increments <= dE in E."""
    npan = np.maximum(np.ceil(Eend / dE).astype(np.int64), 1)
    part = np.repeat(np.arange(S.size), npan)
    k = _ranges(npan)
    frac0 = k / npan[part]
    frac1 = (k + 1) / npan[part]
    ap, Sp, Ep = a[part], S[part], Eend[part]

    def s_of(E):  # inverse of E(s) = a s + s^2/2 without cancellation
        den = ap + np.sqrt(ap * ap + 2.0 * E)
        return np.divide(2.0 * E, den, out=np.zeros_like(den), where=den > 0)

    s0 = s_of(frac0 * Ep)
    s1 = s_of(frac1 * Ep)
    last = (k == npan[part] - 1) & ~trunc[part]
    s1 = np.where(last, Sp, s1)
    far0 = np.where(last, 0.0, Sp - s1)
    x, W, xbar = gauss_legendre01(nq)
    width = (s1 - s0)[:, None]
    s = s0[:, None] + width * x
    d_from_u0_end = near[part][:, None] + s0[:, None] + width * x
    d_to_far_end = far0[:, None] + width * xbar
    wts = width * W * _phi(ap[:, None] + s)
    dl = delta[cell_of[part]][:, None]
    right = is_right[part][:, None]
    tau = np.where(right, d_from_u0_end, d_to_far_end) / dl
    taubar = np.where(right, d_to_far_end, d_from_u0_end) / dl
    return (np.repeat(cell_of[part], nq), tau.ravel(), taubar.ravel(), wts.ravel())
