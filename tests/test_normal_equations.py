"""normal_equations.py: K = A + mu M + lam P and its solves (main.tex eq:normaleq,
eq:dist-vN, sec:solver; sec:wellposed).

Interface the tests assume:
- ``NormalEquations(basis, latent, samples, t, reg)`` with reg an instance; attributes
  ``A``, ``b`` (n, D), ``M``, ``P``, ``lam_s``, ``lam_min``, ``n_factorizations``;
- ``solve(lam)`` returns the coefficients of v (not of w = (1 - t) v), shape (n, D);
- ``validation(samples)`` returns (A_val, b_val) of eq:datamatrix for those samples, or an
  object with attributes A and b.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

import conftest as R
from conftest import dense, gl, latent, make_basis, normal_equations

config = pytest.importorskip("smoothfm.config")
regs = pytest.importorskip("smoothfm.regularizers")


def reg(name="H2"):
    r = getattr(regs, name)
    return r() if isinstance(r, type) else r


def setup_1d(t=0.7, N=12, m=24, L=5.0, name="H2", seed=0):
    xs = np.random.default_rng(seed).uniform(-3.5, 3.5, size=(N, 1))
    b = make_basis(1, m, L)
    return normal_equations(b, latent(1, L), xs, t, reg(name)), b, xs


def setup_2d(t=0.6, N=10, m=6, L=4.5, name="H2", seed=1):
    xs = np.random.default_rng(seed).uniform(-3.0, 3.0, size=(N, 2))
    b = make_basis(2, m, L)
    return normal_equations(b, latent(2, L), xs, t, reg(name)), b, xs


def B(ne):
    return np.asarray(ne.b, float).reshape(dense(ne.A).shape[0], -1)


def sol(ne, lam):
    return np.asarray(ne.solve(lam), float).reshape(dense(ne.A).shape[0], -1)


@pytest.mark.parametrize("make", [setup_1d, setup_2d])
def test_attributes(make):
    ne, b, xs = make()
    A, P, M = dense(ne.A), dense(ne.P), dense(ne.M)
    D = xs.shape[1]
    A0, b0 = b.data_matrices(xs, 0.7 if D == 1 else 0.6, latent(D, 5.0 if D == 1 else 4.5))
    np.testing.assert_array_equal(A, dense(A0))
    np.testing.assert_array_equal(B(ne), np.asarray(b0, float).reshape(B(ne).shape))
    np.testing.assert_allclose(M, dense(regs.mass_matrix(b)), rtol=0, atol=0)
    np.testing.assert_allclose(P, dense(reg().matrix(b)), rtol=0, atol=0)
    assert ne.lam_s == pytest.approx(np.diag(A).max() / np.diag(P).max(), rel=1e-15)
    assert ne.lam_min == pytest.approx(config.LAM_FLOOR_REL * ne.lam_s, rel=1e-15)


@pytest.mark.parametrize("make", [setup_1d, setup_2d])
def test_system_and_floor(make):
    ne, b, xs = make()
    A, P, M = dense(ne.A), dense(ne.P), dense(ne.M)
    for lam in (0.0, 0.5 * ne.lam_min, ne.lam_min, 1e-3 * ne.lam_s, ne.lam_s, 1e3 * ne.lam_s):
        K = dense(ne.system(lam))
        ref = A + config.MU * M + max(lam, ne.lam_min) * P
        np.testing.assert_allclose(K, ref, rtol=0, atol=4e-16 * np.abs(ref).max())
        # the mass term is exactly MU M (MU from config)
        np.testing.assert_allclose(K - A - max(lam, ne.lam_min) * P, config.MU * M,
                                   rtol=0, atol=4e-16 * np.abs(ref).max())


@pytest.mark.parametrize("make", [setup_1d, setup_2d])
def test_spd_including_zero(make):
    ne, b, xs = make()
    for lam in (0.0, ne.lam_min, ne.lam_s, 1e6 * ne.lam_s):
        K = dense(ne.system(lam))
        np.linalg.cholesky(K)
        assert np.linalg.eigvalsh(K).min() > 0


def test_solve_vs_dense_1d():
    """Against a 40-digit solve of the same K. At the floor K has condition ~3.5e10, so a
    backward-stable solve may have a forward error up to cond(K) eps ~ 8e-6 (SciPyLU gives
    7e-8, CHOLMOD 1e-11); there the forward error is checked against cond(K) eps and the
    backward error against eps, as in test_solve_vs_dense_2d."""
    ne, b, xs = setup_1d()
    for lam in (1e-3 * ne.lam_s, ne.lam_s, 10 * ne.lam_s):
        ref = _mp_solve(dense(ne.system(lam)), B(ne))
        np.testing.assert_allclose(sol(ne, lam), ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())
    K = dense(ne.system(0.0))
    ref = _mp_solve(K, B(ne))
    C = sol(ne, 0.0)
    u = np.spacing(1.0)   # machine epsilon (the identifier eps is reserved for the paper's epsilon)
    assert np.abs(C - ref).max() <= np.linalg.cond(K) * u * np.abs(ref).max()
    assert ne.backward_error(0.0, C) <= 10 * u


def test_solve_vs_dense_2d():
    ne, b, xs = setup_2d()
    for lam in (1e-3 * ne.lam_s, ne.lam_s, 10 * ne.lam_s):
        ref = np.linalg.solve(dense(ne.system(lam)), B(ne))
        np.testing.assert_allclose(sol(ne, lam), ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())
    K = dense(ne.system(0.0))            # floor: backward error instead of forward error
    C = sol(ne, 0.0)
    assert np.linalg.norm(K @ C - B(ne)) <= 1e-12 * np.linalg.norm(K) * np.linalg.norm(C)


def _mp_solve(K, rhs):
    with mp.workdps(40):
        Km = mp.matrix(K.tolist())
        out = []
        for d in range(rhs.shape[1]):
            out.append([float(v) for v in mp.lu_solve(Km, mp.matrix(rhs[:, d].tolist()))])
        return np.array(out).T


def test_split_and_plain_agree_at_threshold():
    """lam = 100 lam_s (split) and just below it (plain) vs a 40-digit solve."""
    ne, b, xs = setup_1d()
    A, P, M = dense(ne.A), dense(ne.P), dense(ne.M)
    for f in (config.SPLIT_RATIO * (1 - 1e-6), config.SPLIT_RATIO):
        lam = f * ne.lam_s
        ref = _mp_solve(A + config.MU * M + lam * P, B(ne))
        np.testing.assert_allclose(sol(ne, lam), ref, rtol=0, atol=1e-10 * np.abs(ref).max())


def _exact_null_P(m, L):
    c, _ = R.centres(m, L)
    return np.column_stack([np.ones(m + 3), c])


def test_forward_error_large_lambda():
    """lam = 1e8 lam_s: relative forward error <= 1e-13 against a 40-digit solve with the
    projected penalty (I - Pi) P (I - Pi), whose kernel is exactly the affine coefficients."""
    m, L = 24, 5.0
    ne, b, xs = setup_1d(m=m, L=L)
    A, P, M = dense(ne.A), dense(ne.P), dense(ne.M)
    Z = _exact_null_P(m, L)
    lam = 1e8 * ne.lam_s
    rhs = B(ne)
    with mp.workdps(40):
        Zm = mp.matrix(Z.tolist())
        Pi = Zm * mp.inverse(Zm.T * Zm) * Zm.T
        I = mp.eye(m + 3)
        Pt = (I - Pi) * mp.matrix(P.tolist()) * (I - Pi)
        K = mp.matrix(A.tolist()) + mp.mpf(config.MU) * mp.matrix(M.tolist()) + mp.mpf(lam) * Pt
        ref = np.array([float(v) for v in mp.lu_solve(K, mp.matrix(rhs[:, 0].tolist()))])
    got = sol(ne, lam)[:, 0]
    assert np.linalg.norm(got - ref) <= 1e-13 * np.linalg.norm(ref)


@pytest.mark.parametrize("make", [setup_1d, setup_2d])
def test_infinite_lambda_limit(make):
    """lam -> inf: the (A + MU M)-weighted fit in null(P)."""
    ne, b, xs = make()
    A, M = dense(ne.A), dense(ne.M)
    Z = np.asarray(reg().null_space(b), float).reshape(A.shape[0], -1)
    K0 = A + config.MU * M
    ref = Z @ np.linalg.solve(Z.T @ K0 @ Z, Z.T @ B(ne))
    np.testing.assert_allclose(sol(ne, 1e12 * ne.lam_s), ref, rtol=0,
                               atol=1e-8 * np.abs(ref).max())


@pytest.mark.parametrize("make", [setup_1d, setup_2d])
def test_residual(make):
    ne, b, xs = make()
    lam = ne.lam_s
    C = sol(ne, lam)
    r = float(ne.residual(lam, C))
    assert r <= 1e-12
    K = dense(ne.system(lam))
    mine = np.linalg.norm(K @ C - B(ne)) / np.linalg.norm(B(ne))
    assert mine <= 1e-12
    Cp = C + 1e-3 * np.abs(C).max()
    rp = float(ne.residual(lam, Cp))
    ref = np.linalg.norm(K @ Cp - B(ne)) / np.linalg.norm(B(ne))
    assert rp == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("make", [setup_1d, setup_2d])
def test_backward_error(make):
    """The logged quantity: ||K C - b|| / (||K||_F ||C|| + ||b||) against the dense formula,
    at the rounding level where the relative residual is inflated (lam >= 1e13 lam_s)."""
    ne, b, xs = make()
    for rel in (1.0, 1e13, 1e15):
        lam = rel * ne.lam_s
        C = sol(ne, lam)
        K = dense(ne.system(lam))
        ref = np.linalg.norm(K @ C - B(ne)) / (np.linalg.norm(K) * np.linalg.norm(C)
                                               + np.linalg.norm(B(ne)))
        be = float(ne.backward_error(lam, C))
        # both at the rounding level, where sparse and dense products round differently
        assert be <= 1e-13 and ref <= 1e-13, (rel, be, ref, float(ne.residual(lam, C)))
        Cp = C + 1e-3 * np.abs(C).max()
        refp = np.linalg.norm(K @ Cp - B(ne)) / (np.linalg.norm(K) * np.linalg.norm(Cp)
                                                 + np.linalg.norm(B(ne)))
        assert float(ne.backward_error(lam, Cp)) == pytest.approx(refp, rel=1e-8)


def ell_t(xs, t, L, vfun, m=24):
    """eq:efm loss at time t by quadrature over the truncated latent (1D)."""
    sig = 1 - t
    knots = np.linspace(-L, L, m + 1)
    tot = 0.0
    for x1 in xs[:, 0]:
        zb = (knots - t * x1) / sig
        z, w = gl(-L, L, hmax=0.05, npts=10, breaks=zb)
        wz = w * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi) / R.ZL(L)
        x = sig * z + t * x1
        tot += np.sum(wz * (vfun(x) - (x1 - z)) ** 2)
    return tot / len(xs)


def test_fit_loss_equals_flow_matching_loss():
    """fit_loss(C) + ell_t(0) = ell_t(v_h) (eq:dist-vN), 1e-10."""
    m, L, t = 24, 5.0, 0.7
    ne, b, xs = setup_1d(t=t, m=m, L=L)
    for lam in (1e-3 * ne.lam_s, ne.lam_s):
        C = sol(ne, lam)
        A, bb = dense(ne.A), B(ne)
        mine = float(C[:, 0] @ A @ C[:, 0] - 2 * C[:, 0] @ bb[:, 0])
        assert float(ne.fit_loss(C)) == pytest.approx(mine, rel=1e-12, abs=1e-14)
        vh = lambda x: R.design1d(m, L, x) @ C[:, 0]
        lhs = float(ne.fit_loss(C)) + ell_t(xs, t, L, lambda x: 0 * x)
        assert lhs == pytest.approx(ell_t(xs, t, L, vh), rel=1e-10)


def test_validation_matrices():
    ne, b, xs = setup_1d()
    xv = np.array([[-1.0], [0.5], [2.2]])
    out = ne.validation(xv)
    Av, bv = (out.A, out.b) if hasattr(out, "A") else out
    A0, b0 = b.data_matrices(xv, 0.7, latent(1, 5.0))
    np.testing.assert_array_equal(dense(Av), dense(A0))
    np.testing.assert_array_equal(np.asarray(bv, float).reshape(-1),
                                  np.asarray(b0, float).reshape(-1))


def test_n_factorizations_counts():
    ne, b, xs = setup_1d()
    n0 = int(ne.n_factorizations)
    ne.solve(1e-2 * ne.lam_s)
    n1 = int(ne.n_factorizations)
    ne.solve(1e-1 * ne.lam_s)
    n2 = int(ne.n_factorizations)
    assert n1 >= n0 + 1 and n2 >= n1 + 1


def test_memorizing_affine_field_has_zero_penalty():
    """N = 1, H2: v_N = (x_1 - x)/sigma is affine, so its coefficients lie in null(P)."""
    m, L, t = 24, 5.0, 0.99
    x1 = 1.3
    b = make_basis(1, m, L)
    P = dense(reg().matrix(b))
    c, _ = R.centres(m, L)
    v = (x1 - c) / (1 - t)
    assert v @ P @ v <= 1e-12 * np.abs(P).max() * (v @ v)


def test_bounded_norm_as_t_to_one():
    """With MU > 0 the computed ||v_t||_{L^2(Omega)} stays bounded as t -> 1 (Sec. 3)."""
    m, L, x1 = 24, 5.0, 1.3
    b = make_basis(1, m, L)
    norms = []
    for e in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        ne = normal_equations(b, latent(1, L), np.array([[x1]]), 1 - e, reg())
        C = sol(ne, 1e-6)
        M = dense(ne.M)
        norms.append(float(np.sqrt(C[:, 0] @ M @ C[:, 0])))
    norms = np.array(norms)
    assert np.all(np.isfinite(norms))
    assert norms.max() <= 10 * norms[0]
    memo = np.sqrt(2 * L * ((x1 / 1e-6) ** 2 + L ** 2 / 3e-12))   # ||(x1 - x)/sigma||
    assert norms[-1] < 1e-4 * memo
