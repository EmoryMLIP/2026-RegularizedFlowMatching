"""solvers.py: sparse SPD solves (main.tex sec:solver).

Readings: ``Cholmod()`` and ``SciPyLU()`` construct solvers; ``factor(K)`` returns an
object with ``solve(rhs)`` for rhs (n,) or (n, k); ``default_solver()`` returns a solver
with the same interface (Cholmod when cvxopt is importable).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import scipy.sparse as sp

solvers = pytest.importorskip("smoothfm.solvers")

try:
    import cvxopt  # noqa: F401
    HAVE_CVXOPT = True
except ImportError:
    HAVE_CVXOPT = False

needs_cholmod = pytest.mark.skipif(not HAVE_CVXOPT, reason="cvxopt not installed")


def spd(n=12, seed=0, shift=1e-3, scale=1.0):
    rng = np.random.default_rng(seed)
    T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    I = sp.identity(n)
    D = sp.diags(shift + rng.uniform(0, 1, n * n))
    return (scale * (sp.kron(T, I) + sp.kron(I, T) + D)).tocsc()


def makers():
    out = [pytest.param(lambda: solvers.SciPyLU(), id="scipy")]
    out.append(pytest.param(lambda: solvers.Cholmod(), id="cholmod", marks=needs_cholmod))
    out.append(pytest.param(lambda: solvers.default_solver(), id="default"))
    return out


@pytest.mark.parametrize("make", makers())
def test_solve_matches_dense(make):
    K = spd()
    rng = np.random.default_rng(1)
    for rhs in (rng.normal(size=K.shape[0]), rng.normal(size=(K.shape[0], 3))):
        x = np.asarray(make().factor(K).solve(rhs), float)
        ref = np.linalg.solve(K.toarray(), rhs)
        assert x.shape == ref.shape
        np.testing.assert_allclose(x, ref, rtol=1e-10, atol=1e-12 * np.abs(ref).max())


@needs_cholmod
def test_cholmod_equals_scipy_and_same_pattern_reuse():
    K1 = spd(n=11, seed=4)
    K2 = spd(n=11, seed=5, shift=0.5, scale=3.0)       # same pattern, other values
    r = np.random.default_rng(6).normal(size=K1.shape[0])
    a1, a2 = solvers.Cholmod().factor(K1), solvers.Cholmod().factor(K2)
    for K, a in ((K1, a1), (K2, a2)):
        ref = np.linalg.solve(K.toarray(), r)
        np.testing.assert_allclose(a.solve(r), ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(solvers.SciPyLU().factor(K).solve(r), ref, rtol=1e-10,
                                   atol=1e-12)
    # first factor still valid after the second shares the symbolic analysis
    np.testing.assert_allclose(a1.solve(r), np.linalg.solve(K1.toarray(), r), rtol=1e-10)


@needs_cholmod
def test_cholmod_thread_safe():
    Ks = [spd(n=10, seed=s, shift=0.1 * (s + 1)) for s in range(8)]
    r = np.random.default_rng(7).normal(size=Ks[0].shape[0])
    out = [None] * len(Ks)

    def work(i):
        out[i] = solvers.Cholmod().factor(Ks[i]).solve(r)

    th = [threading.Thread(target=work, args=(i,)) for i in range(len(Ks))]
    [t.start() for t in th]
    [t.join() for t in th]
    for K, x in zip(Ks, out):
        np.testing.assert_allclose(x, np.linalg.solve(K.toarray(), r), rtol=1e-10, atol=1e-12)
