"""splines.py: cubic B-spline bases and the data matrices (main.tex sec:solver;
supplement S:eq:datamatrix, S:sec:splinedetails).

Basis (supplement): x = L xi, knots -L + j h, h = 2L/m, n = m + 3 functions
b_a(x) = b((x - x_a)/h), x_a = -L + a h, a = -1..m+1, restricted to [-L, L]; 2D index
a n + a' (second axis fastest).

Interface the tests assume (adapters in conftest.py):
- ``eval(xi, deriv)`` takes reference coordinates xi = x / L, returns physical
  derivatives; Cubic2D takes deriv as a pair (dx, dy);
- ``affine_coefficients()`` (or the attribute) returns (n_basis, D + 1) coefficient columns
  that span {1, x_1, ..., x_D};
- ``data_matrices(samples, t, latent)`` returns A (n_basis^D square, sparse or dense) and
  b (n_basis^D, D), normalized by 1/N as in eq:datamatrix.
"""

from __future__ import annotations

from fractions import Fraction as Fr

import mpmath as mp
import numpy as np
import pytest
import scipy.sparse as sp

import conftest as R
from conftest import MP_DPS, dense, design, latent, make_basis, mp_tmoment, mp_ZL, pick, value

splines = pytest.importorskip("smoothfm.splines")


# ---------------------------------------------------------------------------
# Basis values, partition of unity, affine reproduction, Gram matrices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m, L", [(8, 5.0), (31, 4.5), (64, 1.0)])
def test_design_matches_reference_1d(m, L):
    b = make_basis(1, m, L)
    x = np.concatenate([np.linspace(-L, L, 401), [-L, L]])
    for k in (0, 1, 2):
        np.testing.assert_allclose(design(b, x[:, None], k), R.design1d(m, L, x, k),
                                   rtol=1e-12, atol=1e-12 * (m / (2 * L)) ** k)


def test_design_matches_reference_2d():
    m, L = 6, 4.5
    b = make_basis(2, m, L)
    X = np.random.default_rng(0).uniform(-L, L, size=(200, 2))
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
        np.testing.assert_allclose(design(b, X, (dx, dy)), R.design2d(m, L, X, dx, dy),
                                   rtol=1e-12, atol=1e-11)


@pytest.mark.parametrize("D", [1, 2])
def test_partition_of_unity_and_affine_reproduction(D):
    m, L = 10, 4.5
    b = make_basis(D, m, L)
    X = np.random.default_rng(1).uniform(-L, L, size=(300, D))
    B = design(b, X, 0 if D == 1 else (0, 0))
    np.testing.assert_allclose(B.sum(1), 1.0, rtol=0, atol=1e-14)
    # the centres are the coefficients of x (Greville abscissae of cubic B-splines)
    c, _ = R.centres(m, L)
    if D == 1:
        np.testing.assert_allclose(B @ c, X[:, 0], rtol=0, atol=1e-13)
    else:
        n = m + 3
        np.testing.assert_allclose(B @ np.repeat(c, n), X[:, 0], atol=1e-13)
        np.testing.assert_allclose(B @ np.tile(c, n), X[:, 1], atol=1e-13)
    Z = np.asarray(value(pick(b, "affine_coefficients")), float)
    assert Z.shape[1] == D + 1
    F = B @ Z
    target = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(F, target, rcond=None)
    np.testing.assert_allclose(F @ coef, target, atol=1e-12)
    assert np.linalg.matrix_rank(F) == D + 1


@pytest.mark.parametrize("m, L", [(8, 5.0), (40, 4.5), (512, 5.0)])
def test_gram_exact(m, L):
    b = make_basis(1, m, L)
    for k in (0, 1, 2):
        P = dense(b.gram(k))
        ref = R.gram_ref(m, L, k)
        assert P.shape == (m + 3, m + 3)
        np.testing.assert_allclose(P, ref, rtol=0, atol=1e-13 * np.abs(ref).max())


def test_gram_null_spaces():
    m, L = 20, 5.0
    b = make_basis(1, m, L)
    P0, P1, P2 = (dense(b.gram(k)) for k in (0, 1, 2))
    n = m + 3
    c, _ = R.centres(m, L)
    one = np.ones(n)
    assert np.linalg.eigvalsh(P0).min() > 0
    assert np.abs(P1 @ one).max() <= 1e-13 * np.abs(P1).max()
    assert np.abs(P2 @ one).max() <= 1e-13 * np.abs(P2).max()
    assert np.abs(P2 @ c).max() <= 1e-13 * np.abs(P2).max() * L
    e1 = np.linalg.eigvalsh(P1)
    e2 = np.linalg.eigvalsh(P2)
    assert np.sum(e1 < 1e-12 * e1.max()) == 1 and e1.min() > -1e-12 * e1.max()
    assert np.sum(e2 < 1e-12 * e2.max()) == 2 and e2.min() > -1e-12 * e2.max()
    for P in (P0, P1, P2):          # bandwidth three
        i, j = np.nonzero(P)
        assert np.abs(i - j).max() <= 3


# ---------------------------------------------------------------------------
# Data matrices vs mpmath (truncated kernel, exact cell integrals)
# ---------------------------------------------------------------------------

# pieces of the cubic B-spline on [k, k+1], k = -2..1, as cubics in s in [0, 1]
_PIECES = {-2: [Fr(0), Fr(0), Fr(0), Fr(1, 6)],
           -1: [Fr(1, 6), Fr(1, 2), Fr(1, 2), Fr(-1, 2)],
           0: [Fr(2, 3), Fr(0), Fr(-1), Fr(1, 2)],
           1: [Fr(1, 6), Fr(-1, 2), Fr(1, 2), Fr(-1, 6)]}


def _pmul(p, q):
    r = [mp.mpf(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, c in enumerate(q):
            r[i + j] += a * c
    return r


def _compose(p, A, B):
    """Coefficients in u of p(A + B u)."""
    out, base = [mp.mpf(0)], [mp.mpf(1)]
    for c in p:
        term = [mp.mpf(c.numerator) / c.denominator * v for v in base]
        n = max(len(out), len(term))
        out = [(out[i] if i < len(out) else 0) + (term[i] if i < len(term) else 0)
               for i in range(n)]
        base = _pmul(base, [A, B])
    return out


def mp_tables(m, Lb, xi, t, L):
    """T_i and g_i of eq:datamatrix in mpmath for one sample x_i, truncated latent."""
    with mp.workdps(MP_DPS):
        n = m + 3
        Lb, xi, t, L = (mp.mpf(v) for v in (Lb, xi, t, L))
        sig = 1 - t
        h = 2 * Lb / m
        mu = t * xi
        Z = mp_ZL(L)
        T = [[mp.mpf(0)] * n for _ in range(n)]
        g = [mp.mpf(0)] * n
        for j in range(m):
            xj = -Lb + j * h
            lo = max((xj - mu) / sig, -L)
            hi = min((xj + h - mu) / sig, L)
            if not hi > lo:
                continue
            J = [mp_tmoment(lo, hi, k) / Z for k in range(8)]
            A, B = (mu - xj) / h, sig / h
            pc = {}
            for ai in range(n):              # basis a = ai - 1, piece k = j - a
                k = j - (ai - 1)
                if k in _PIECES:
                    pc[ai] = _compose(_PIECES[k], A, B)
            dot = lambda p: mp.fsum(c * J[k] for k, c in enumerate(p))
            for ai, pa in pc.items():
                # (x_i - x)/sigma = x_i - u for x = t x_i + sigma u, sigma = 1 - t
                g[ai] += dot(_pmul(pa, [xi, mp.mpf(-1)]))
                for bi, pb in pc.items():
                    T[ai][bi] += dot(_pmul(pa, pb))
        return (np.array([[float(v) for v in row] for row in T]),
                np.array([float(v) for v in g]))


# (basis half-width, m, t, x_i) chosen so that sigma / h spans 1e-4 .. 1e3
TABLE_CASES = [(5.0, 16, 1 - 6.25e-5, 0.3),      # sigma/h = 1e-4
               (5.0, 16, 1 - 6.25e-3, -1.1),     # 1e-2
               (5.0, 16, 0.375, 2.0),            # 1
               (1.0, 16, 0.2, 0.05),             # 6.4
               (0.05, 16, 0.4, 0.01),            # 96
               (0.01, 16, 1e-3, 0.0)]            # 799


@pytest.mark.parametrize("Lb, m, t, xi", TABLE_CASES)
def test_data_matrices_single_sample_vs_mp(Lb, m, t, xi):
    L = 5.0
    b = make_basis(1, m, Lb)
    A, bb = b.data_matrices(np.array([[xi]]), t, latent(1, L))
    A, bb = dense(A), np.asarray(bb, float).reshape(-1)
    T, g = mp_tables(m, Lb, xi, t, L)
    np.testing.assert_allclose(A, T, rtol=0, atol=1e-10 * np.abs(T).max())
    np.testing.assert_allclose(bb, g, rtol=0, atol=1e-10 * np.abs(g).max())


def test_data_matrices_average_and_sums():
    """A = (1/N) sum T_i; 1^T A 1 = 1 and 1^T b = mean x_i when every kernel box lies in
    Omega (1^T f_i = 1, 1^T g_i = x_i, E z = 0)."""
    L, m, t = 5.0, 24, 0.7
    b = make_basis(1, m, L)
    xs = np.array([[-0.9], [0.1], [0.45], [1.0]])
    lat = latent(1, L)
    A, bb = (dense(v) if k == 0 else np.asarray(v, float).reshape(-1)
             for k, v in enumerate(b.data_matrices(xs, t, lat)))
    parts = [b.data_matrices(xs[i:i + 1], t, lat) for i in range(len(xs))]
    np.testing.assert_allclose(A, sum(dense(p[0]) for p in parts) / len(xs), atol=1e-15)
    one = np.ones(m + 3)
    assert one @ A @ one == pytest.approx(1.0, abs=1e-13)
    assert one @ bb == pytest.approx(xs.mean(), abs=1e-13)


@pytest.mark.parametrize("D", [1, 2])
def test_data_matrix_symmetric_psd_pattern(D):
    from smoothfm.regularizers import H2
    L, m, t = 4.5, 12, 0.6
    b = make_basis(D, m, L)
    xs = np.random.default_rng(2).uniform(-2.5, 2.5, size=(40, D))
    A, _ = b.data_matrices(xs, t, latent(D, L))
    Ad = dense(A)
    np.testing.assert_allclose(Ad, Ad.T, rtol=0, atol=1e-16 * np.abs(Ad).max())
    assert np.linalg.eigvalsh(Ad).min() >= -1e-14 * np.abs(Ad).max()
    P = H2().matrix(b)
    n = m + 3
    i1 = np.arange(n)
    band = np.abs(i1[:, None] - i1[None, :]) <= 3
    stencil = band if D == 1 else np.kron(band, band)
    # outside the 7 (1D) or 7 x 7 (2D) stencil the data matrix vanishes
    assert np.all(Ad[~stencil] == 0)
    if sp.issparse(A) and sp.issparse(P):     # same stored pattern (shared symbolic analysis)
        Ac, Pc = sp.csc_matrix(A), sp.csc_matrix(P)
        Ac.sort_indices()
        Pc.sort_indices()
        assert np.array_equal(Ac.indptr, Pc.indptr) and np.array_equal(Ac.indices, Pc.indices)
        if D == 2:
            assert np.diff(Pc.indptr).max() == 49


def test_data_matrices_2d_kronecker():
    """N = 1: A = T1 (x) T2, b_1 = g1 (x) f2, b_2 = f1 (x) g2, f = T 1 (supplement)."""
    L, m, t = 4.5, 8, 0.8
    x = np.array([[0.7, -1.3]])
    lat1, lat2 = latent(1, L), latent(2, L)
    b1, b2 = make_basis(1, m, L), make_basis(2, m, L)
    A, bb = b2.data_matrices(x, t, lat2)
    A, bb = dense(A), np.asarray(bb, float).reshape(-1, 2)
    T1, g1 = (dense(v) if k == 0 else np.asarray(v, float).reshape(-1)
              for k, v in enumerate(b1.data_matrices(x[:, :1], t, lat1)))
    T2, g2 = (dense(v) if k == 0 else np.asarray(v, float).reshape(-1)
              for k, v in enumerate(b1.data_matrices(x[:, 1:], t, lat1)))
    f1, f2 = T1.sum(1), T2.sum(1)
    np.testing.assert_allclose(A, np.kron(T1, T2), atol=1e-14 * np.abs(A).max())
    np.testing.assert_allclose(bb[:, 0], np.kron(g1, f2), atol=1e-14 * np.abs(bb).max())
    np.testing.assert_allclose(bb[:, 1], np.kron(f1, g2), atol=1e-14 * np.abs(bb).max())


def test_data_matrices_support_cut_exact():
    """Cells beyond t x_i + sigma Omega carry exactly zero mass (the truncated kernel has support sigma Omega)."""
    L, m, t = 5.0, 64, 0.99
    b = make_basis(1, m, L)
    xi = 0.4
    A, _ = b.data_matrices(np.array([[xi]]), t, latent(1, L))
    Ad = dense(A)
    c, h = R.centres(m, L)
    lo, hi = t * xi - (1 - t) * L, t * xi + (1 - t) * L
    far = (c + 2 * h < lo) | (c - 2 * h > hi)
    assert np.all(Ad[far][:, :] == 0) and np.all(Ad[:, far] == 0)
