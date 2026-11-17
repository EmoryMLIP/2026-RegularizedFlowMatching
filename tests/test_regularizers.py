"""regularizers.py: B_k penalties and the mass matrix (supplement S:eq:regmatrix,
S:sec:splinedetails).

Interface the tests assume:
- ``H2``, ``H1``, ``L2`` are Regularizer classes instantiated without arguments (an
  instance exported under the same name also works); ``order`` is the attribute k;
- ``null_space(basis)`` returns a matrix whose columns span the kernel;
- ``mass_matrix(basis)`` is a module function.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import conftest as R
from conftest import dense, make_basis, pick, value

regs = pytest.importorskip("smoothfm.regularizers")


def reg(name):
    r = getattr(regs, name)
    return r() if isinstance(r, type) else r


ORDERS = {"H2": 2, "H1": 1, "L2": 0}


@pytest.mark.parametrize("name", ["H2", "H1", "L2"])
def test_attributes(name):
    r = reg(name)
    k = ORDERS[name]
    assert int(value(pick(r, "order", "k"))) == k
    assert isinstance(r.name, str)


@pytest.mark.parametrize("D", [1, 2])
@pytest.mark.parametrize("name", ["H2", "H1", "L2"])
def test_symmetric_psd_and_kernel(D, name):
    m, L = (16, 5.0) if D == 1 else (6, 4.5)
    b = make_basis(D, m, L)
    r = reg(name)
    P = dense(r.matrix(b))
    np.testing.assert_allclose(P, P.T, rtol=0, atol=1e-15 * np.abs(P).max())
    e = np.linalg.eigvalsh(P)
    assert e.min() >= -1e-12 * e.max()
    Z = np.asarray(r.null_space(b), float).reshape(P.shape[0], -1) if ORDERS[name] else None
    kdim = {"H2": D + 1, "H1": 1, "L2": 0}[name]
    assert np.sum(e < 1e-11 * e.max()) == kdim
    if kdim:
        assert Z.shape[1] == kdim and np.linalg.matrix_rank(Z) == kdim
        assert np.abs(P @ Z).max() <= 1e-12 * np.abs(P).max() * np.abs(Z).max()
        # span(Z) equals the numerical kernel
        w, V = np.linalg.eigh(P)
        K = V[:, :kdim]
        proj = Z @ np.linalg.lstsq(Z, K, rcond=None)[0]
        np.testing.assert_allclose(proj, K, atol=1e-9)


def test_kronecker_forms_2d():
    m, L = 7, 4.5
    b = make_basis(2, m, L)
    P0, P1, P2 = (R.gram_ref(m, L, k) for k in (0, 1, 2))
    ref = {"H2": np.kron(P2, P0) + 2 * np.kron(P1, P1) + np.kron(P0, P2),
           "H1": np.kron(P1, P0) + np.kron(P0, P1),
           "L2": np.kron(P0, P0)}
    for name, M in ref.items():
        np.testing.assert_allclose(dense(reg(name).matrix(b)), M, rtol=0,
                                   atol=1e-13 * np.abs(M).max())
    np.testing.assert_allclose(dense(regs.mass_matrix(b)), ref["L2"], rtol=0,
                               atol=1e-13 * np.abs(ref["L2"]).max())


def test_mass_matrix_1d():
    m, L = 12, 5.0
    b = make_basis(1, m, L)
    M = R.gram_ref(m, L, 0)
    np.testing.assert_allclose(dense(regs.mass_matrix(b)), M, atol=1e-13 * np.abs(M).max())
    np.testing.assert_allclose(dense(reg("L2").matrix(b)), M, atol=1e-13 * np.abs(M).max())


def _fit_coeffs_2d(m, L, f):
    """Spline coefficients of a polynomial of degree <= 3 per axis (exact LS fit)."""
    X = np.array(np.meshgrid(np.linspace(-L, L, 3 * (m + 3)), np.linspace(-L, L, 3 * (m + 3)),
                             indexing="ij")).reshape(2, -1).T
    B = R.design2d(m, L, X)
    c, *_ = np.linalg.lstsq(B, f(X), rcond=None)
    assert np.abs(B @ c - f(X)).max() < 1e-10
    return c


@pytest.mark.parametrize("name, exact", [("H2", 32.0), ("H1", 32.0 / 3), ("L2", 112.0 / 45)])
def test_energies_of_polynomials_2d(name, exact):
    """x^2 + y^2 on [-1, 1]^2: Hessian energy 32, gradient 32/3, L2 112/45."""
    m, L = 6, 1.0
    b = make_basis(2, m, L)
    c = _fit_coeffs_2d(m, L, lambda X: X[:, 0] ** 2 + X[:, 1] ** 2)
    P = dense(reg(name).matrix(b))
    assert c @ P @ c == pytest.approx(exact, rel=1e-11)


def test_energy_mixed_term():
    """x y: zero in the pure second derivatives, 2 int (d_12 u)^2 = 2 * 4 = 8 on [-1,1]^2."""
    m, L = 5, 1.0
    b = make_basis(2, m, L)
    c = _fit_coeffs_2d(m, L, lambda X: X[:, 0] * X[:, 1])
    assert c @ dense(reg("H2").matrix(b)) @ c == pytest.approx(8.0, rel=1e-11)


def test_energy_1d_cubic():
    m, L = 9, 5.0
    b = make_basis(1, m, L)
    x = np.linspace(-L, L, 200)
    B = R.design1d(m, L, x)
    c, *_ = np.linalg.lstsq(B, x ** 3, rcond=None)
    assert c @ dense(reg("H2").matrix(b)) @ c == pytest.approx(24 * L ** 3, rel=1e-11)
    assert c @ dense(reg("H1").matrix(b)) @ c == pytest.approx(9 * 2 * L ** 5 / 5, rel=1e-11)


@pytest.mark.parametrize("D", [1, 2])
@pytest.mark.parametrize("name", ["H2", "H1", "L2"])
def test_frame_scaling(D, name):
    """u'(x') = u(s x')/s: |u'|^2_{H^k} = s^(2k - 2 - D) |u|^2_{H^k}."""
    m, L, s = 8, 4.5, 0.37
    k = ORDERS[name]
    bf, bp = make_basis(D, m, L), make_basis(D, m, s * L)
    c = np.random.default_rng(3).normal(size=(m + 3) ** D)
    Ef = (c / s) @ dense(reg(name).matrix(bf)) @ (c / s)
    Ep = c @ dense(reg(name).matrix(bp)) @ c
    assert Ef == pytest.approx(s ** (2 * k - 2 - D) * Ep, rel=1e-12)
