"""latent.py: N(0, I) conditioned on Omega = [-L, L]^D (main.tex (A1), sec:setup).

Interface the tests assume:
- ``Z`` is the per-axis mass erf(L / sqrt 2); ``tv_to_gaussian`` = 1 - Z^D.
- ``pdf``/``logpdf`` take points (n, D) and return (n,); ``cdf``/``ppf``/``from_normal``
  act elementwise on the one-dimensional factor.
- ``quantile_midpoints(M)`` returns M values (any shape with M entries) of the 1D factor.
- ``sobol(M, seed)`` is ppf of scipy ``qmc.Sobol(D, scramble=True, seed=seed).random(M)``.
- ``cell_rule(alpha, beta)`` returns the flat CellRule (cell, tau, taubar, weight): node q
  of cell k sits at u = alpha_k + delta_k tau_q, and the rule integrates g(tau) phi(u)/Z_L
  over [alpha_k, beta_k] cut to [-L, L], to rounding for polynomials g of degree <= 6 (the
  tests check the property, not the layout).
- ppf(cdf(u)) = u holds to 1e-14 plus the error that rounding q = cdf(u) to double
  forces (du = ulp(q)/2 / F'(u)); near +L that term is about 5e-11, so a flat
  1e-14 is not attainable there by any implementation that passes q as a double.
- ``truncnorm_moments(a, b, w=None)`` uses w as the exact width b - a when given.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest
from scipy import stats

from conftest import MP_DPS, latent, mp_tmoment, mp_ZL, value

lat_mod = pytest.importorskip("smoothfm.latent")


def mp_cdf(u, L):
    with mp.workdps(MP_DPS):
        u = mp.mpf(u)
        return mp_tmoment(-L, u, 0) / mp_ZL(L)


def mp_ppf(q, L):
    """Bisection on the mpmath cdf (monotone), 120 halvings of [-L, L]."""
    with mp.workdps(MP_DPS):
        Z = mp_ZL(L)
        q = mp.mpf(q)
        lo, hi = mp.mpf(-L), mp.mpf(L)
        for _ in range(120):
            mid = (lo + hi) / 2
            if mp_tmoment(-L, mid, 0) / Z < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Mass, density, cdf, ppf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D, L, tv", [(1, 5.0, 5.7e-7), (2, 4.5, 1.4e-5), (2, 5.0, 1.1e-6)])
def test_mass_and_tv(D, L, tv):
    g = latent(D, L)
    with mp.workdps(MP_DPS):
        Z = mp_ZL(L)
        ref_tv = 1 - Z ** D
    assert float(value(g.Z)) == pytest.approx(float(Z), rel=1e-15)
    assert float(value(g.tv_to_gaussian)) == pytest.approx(float(ref_tv), rel=1e-10)
    assert float(ref_tv) == pytest.approx(tv, rel=0.05)


@pytest.mark.parametrize("D, L", [(1, 5.0), (2, 4.5)])
def test_pdf_logpdf(D, L):
    g = latent(D, L)
    rng = np.random.default_rng(0)
    x = rng.uniform(-L - 1, L + 1, size=(200, D))
    inside = np.all(np.abs(x) <= L, axis=1)
    ref = stats.norm.logpdf(x).sum(1) - D * math.log(math.erf(L / math.sqrt(2)))
    lp = np.asarray(g.logpdf(x), float).reshape(-1)
    np.testing.assert_allclose(lp[inside], ref[inside], rtol=1e-14, atol=1e-14)
    assert np.all(np.isneginf(lp[~inside]))
    p = np.asarray(g.pdf(x), float).reshape(-1)
    np.testing.assert_allclose(p[inside], np.exp(ref[inside]), rtol=1e-13)
    assert np.all(p[~inside] == 0)


def test_cdf_vs_mp():
    L = 5.0
    g = latent(1, L)
    u = np.array([-5.0, -4.999, -4.5, -3.0, -1.0, -1e-3, 0.0, 0.7, 2.5, 4.0, 4.99, 5.0])
    got = np.asarray(g.cdf(u), float).reshape(-1)
    ref = np.array([float(mp_cdf(v, L)) for v in u])
    # tolerance: 1e-13 relative plus the effect of rounding u itself (F' = pdf)
    tol = 1e-13 * ref + 4e-16 * np.abs(u) * stats.norm.pdf(u) / math.erf(L / math.sqrt(2))
    assert np.all(np.abs(got - ref) <= tol + 1e-300), (got - ref) / ref


@pytest.mark.parametrize("L", [5.0, 4.5])
def test_ppf_cdf_roundtrip(L):
    g = latent(1, L)
    u = np.concatenate([np.linspace(-L, L, 401), [-L + 1e-9, L - 1e-9, -1e-12, 1e-12]])
    q = np.asarray(g.cdf(u), float).reshape(-1)
    back = np.asarray(g.ppf(q), float).reshape(-1)
    err = np.abs(back - u)
    # 1e-14 plus the information lost by storing q in double precision: du = ulp(q)/2 / F'(u).
    # Near +L, q is close to 1 and this term dominates (5e-11 at u = L - 1e-9, L = 5).
    f = stats.norm.pdf(u) / math.erf(L / math.sqrt(2))
    tol = 1e-14 * np.maximum(1, np.abs(u)) + 2 * np.spacing(q) / f
    assert np.all(err <= tol), (u[np.argmax(err / tol)], err.max())


def test_ppf_vs_mp_and_symmetry():
    L = 4.5
    g = latent(1, L)
    q = np.array([1e-12, 1e-7, 1 / 1024, 0.1, 0.25, 0.5, 0.75, 0.9, 1023 / 1024])
    got = np.asarray(g.ppf(q), float).reshape(-1)
    ref = np.array([float(mp_ppf(v, L)) for v in q])
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-15)
    k = np.arange(1, 1024) / 1024.0           # 1 - k exact in binary
    a = np.asarray(g.ppf(k), float)
    b = np.asarray(g.ppf(1 - k), float)
    np.testing.assert_allclose(a, -b, rtol=0, atol=1e-14)
    assert np.all(np.abs(a) <= L)


def test_from_normal():
    L = 4.5
    g = latent(1, L)
    gs = np.sort(np.random.default_rng(1).normal(size=2001) * 2.5)
    z = np.asarray(g.from_normal(gs), float).reshape(-1)
    assert np.all(np.diff(z) >= 0)
    assert np.all(np.abs(z) <= L)
    np.testing.assert_allclose(np.asarray(g.from_normal(-gs), float).reshape(-1), -z,
                               rtol=0, atol=1e-14)
    # F_L^{-1}(Phi(g)) vs mpmath at a few moderate values
    for v in (-3.0, -0.4, 0.0, 1.3, 2.9):
        with mp.workdps(MP_DPS):
            ref = mp_ppf(mp.ncdf(v), L)
        got = float(np.asarray(g.from_normal(np.array([v]))).reshape(-1)[0])
        assert got == pytest.approx(float(ref), rel=1e-12, abs=1e-14)


def test_from_normal_maps_to_truncated_law():
    L = 4.5
    g = latent(1, L)
    z = np.asarray(g.from_normal(np.random.default_rng(2).normal(size=100_000)), float)
    p = stats.kstest(z.reshape(-1), stats.truncnorm(-L, L).cdf).pvalue
    assert p > 1e-3


@pytest.mark.parametrize("D, L", [(1, 5.0), (2, 4.5)])
def test_sample(D, L):
    g = latent(D, L)
    z = np.asarray(g.sample(50_000, np.random.default_rng(3)), float).reshape(50_000, D)
    assert np.all(np.abs(z) <= L)
    for d in range(D):
        assert stats.kstest(z[:, d], stats.truncnorm(-L, L).cdf).pvalue > 1e-3


def test_quantile_midpoints():
    L = 5.0
    g = latent(1, L)
    M = 3000
    q = np.asarray(g.quantile_midpoints(M), float).reshape(-1)
    k = np.arange(1, M + 1)
    np.testing.assert_allclose(q, np.asarray(g.ppf((k - 0.5) / M), float).reshape(-1),
                               rtol=0, atol=0)
    np.testing.assert_allclose(q, stats.truncnorm(-L, L).ppf((k - 0.5) / M),
                               rtol=1e-11, atol=1e-13)


def test_sobol():
    L = 4.5
    g = latent(2, L)
    z = np.asarray(g.sobol(4096, 11), float)
    assert z.shape == (4096, 2)
    assert np.all(np.abs(z) <= L)
    np.testing.assert_array_equal(z, np.asarray(g.sobol(4096, 11), float))
    assert not np.array_equal(z, np.asarray(g.sobol(4096, 12), float))
    # ppf of a scrambled Sobol' net of 2^12 points: every coordinate of F_L(z) has exactly
    # one point in each interval [k/4096, (k+1)/4096)
    u = np.asarray(g.cdf(z), float).reshape(4096, 2)
    for d in range(2):
        cnt = np.bincount(np.floor(u[:, d] * 4096).astype(int).clip(0, 4095), minlength=4096)
        assert np.all(cnt == 1)


# ---------------------------------------------------------------------------
# cell_rule: positive rule, accurate to 5e-14 for degree <= 6, on [alpha, beta] cut to [-L, L]
# ---------------------------------------------------------------------------

def _cells(L):
    cells = []
    for w in (1e-6, 1e-3, 0.1, 1.0, 5.0, 20.0):
        for a in (-3.0, 0.2, L - w / 2, -L - w / 2, L + 0.5, -L - 0.5 - w, -10.0):
            cells.append((a, a + w))
    return np.array(cells)


@pytest.mark.parametrize("L", [5.0, 4.5])
def test_cell_rule_exact(L):
    """sum_{q: cell = k} w_q tau_q^j = int_alpha^beta ((u - alpha)/delta)^j phi(u)/Z_L du over the
    cell cut to [-L, L], j <= 6, vs mpmath quadrature (1e-12 of the cell mass)."""
    g = latent(1, L)
    C = _cells(L)
    r = g.cell_rule(C[:, 0], C[:, 1])
    cell = np.asarray(r.cell).astype(int)
    tau, taubar, w = (np.asarray(v, float) for v in (r.tau, r.taubar, r.weight))
    assert np.all(w > 0)
    assert np.all((tau >= 0) & (tau <= 1))
    np.testing.assert_allclose(taubar, 1 - tau, rtol=0, atol=4.5e-16)   # 1 - tau rounds
    with mp.workdps(MP_DPS):
        Z = mp_ZL(L)
        for k, (a, b) in enumerate(C):
            lo, hi = max(a, -L), min(b, L)
            sel = cell == k
            if not hi > lo:
                assert not sel.any(), (a, b)
                continue
            u = a + (b - a) * tau[sel]
            assert u.min() >= lo - 1e-12 * (b - a) and u.max() <= hi + 1e-12 * (b - a)
            mass = float(mp_tmoment(lo, hi, 0) / Z)
            d = mp.mpf(b) - mp.mpf(a)
            for j in range(7):
                ref = mp.quad(lambda v: ((v - mp.mpf(a)) / d) ** j * mp.npdf(v),
                              [mp.mpf(lo), mp.mpf(hi)]) / Z
                got = float(np.sum(w[sel] * tau[sel] ** j))
                assert abs(got - float(ref)) <= 1e-12 * mass + 1e-300, (a, b, j, got, ref)


# ---------------------------------------------------------------------------
# truncnorm_moments, log_ndtr_diff
# ---------------------------------------------------------------------------

def mp_trunc(a, b):
    with mp.workdps(MP_DPS):
        Z = mp_tmoment(a, b, 0)
        m1 = mp_tmoment(a, b, 1) / Z
        m2 = mp_tmoment(a, b, 2) / Z
        ea = m1 - mp.mpf(a) if mp.isfinite(mp.mpf(a)) else None
        return float(mp.log(Z)), (None if ea is None else float(ea)), float(m2 - m1 ** 2)


CASES = [(-1.0, 2.0), (-3.0, 3.0), (8.0, 9.0), (40.0, 41.0), (1e3, 1e3 + 1), (1e5, 1e5 + 0.5),
         (-41.0, -40.0), (-1e5 - 1, -1e5), (2.0, np.inf), (30.0, np.inf), (1e5, np.inf),
         (-6.0, np.inf), (-np.inf, -2.0), (-np.inf, -1e4)]


@pytest.mark.parametrize("a, b", CASES)
def test_truncnorm_moments_tails(a, b):
    lz, ea, var = (np.asarray(v, float).reshape(-1)[0]
                   for v in lat_mod.truncnorm_moments(np.array([a]), np.array([b])))
    rlz, rea, rvar = mp_trunc(a, b)
    assert lz == pytest.approx(rlz, rel=1e-12, abs=1e-12)
    if rea is not None:
        assert ea == pytest.approx(rea, rel=1e-12)
    assert var == pytest.approx(rvar, rel=1e-12)


@pytest.mark.parametrize("a", [0.3, -7.0, 30.0, -2.5])
@pytest.mark.parametrize("w", [1e-12, 1e-8, 1e-4, 1e-2])
def test_truncnorm_moments_narrow_and_w(a, w):
    """Narrow intervals; w passed as the exact width (b = a + w is rounded)."""
    with mp.workdps(MP_DPS):
        bb = mp.mpf(a) + mp.mpf(w)
        Z = mp_tmoment(a, bb, 0)
        m1 = mp_tmoment(a, bb, 1) / Z
        m2 = mp_tmoment(a, bb, 2) / Z
        rlz, rea, rvar = float(mp.log(Z)), float(m1 - a), float(m2 - m1 ** 2)
    b = a + w
    out = lat_mod.truncnorm_moments(np.array([a]), np.array([b]), w=np.array([w]))
    lz, ea, var = (np.asarray(v, float).reshape(-1)[0] for v in out)
    assert lz == pytest.approx(rlz, rel=1e-12, abs=1e-12)
    assert ea == pytest.approx(rea, rel=1e-12)
    assert var == pytest.approx(rvar, rel=1e-10)


def test_truncnorm_moments_vectorized():
    a = np.array([c[0] for c in CASES[:8]])
    b = np.array([c[1] for c in CASES[:8]])
    lz, ea, var = (np.asarray(v, float) for v in lat_mod.truncnorm_moments(a, b))
    for k in range(len(a)):
        one = [np.asarray(v, float).reshape(-1)[0]
               for v in lat_mod.truncnorm_moments(a[k:k + 1], b[k:k + 1])]
        assert lz[k] == one[0] and ea[k] == one[1] and var[k] == one[2]


@pytest.mark.parametrize("a, b", [(-1.0, 2.0), (8.0, 9.0), (1e5, np.inf), (-np.inf, -1e3),
                                  (-40.0, 40.0), (30.0, 31.0)])
def test_log_ndtr_diff(a, b):
    got = float(np.asarray(lat_mod.log_ndtr_diff(np.array([a]), np.array([b]))).reshape(-1)[0])
    with mp.workdps(MP_DPS):
        ref = float(mp.log(mp_tmoment(a, b, 0)))
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-13)
