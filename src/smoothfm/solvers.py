"""Sparse solvers for the matrix K = A + mu M + lam P of eq:normaleq (main.tex sec:solver).

The paper factorizes K by a sparse Cholesky method and reuses the symbolic analysis across
t and lam, because the mesh, and hence the sparsity pattern, does not depend on them. Both
solvers share the interface ``fresh()``, ``factor(K)``, ``solve(rhs)``.

``Cholmod``: supernodal sparse Cholesky (SuiteSparse CHOLMOD via cvxopt). A, M and P share
one canonical sparsity pattern, so K has the same pattern for every t and lam at a fixed
basis and the symbolic analysis is done once per pattern (module cache). cvxopt's factor
cannot be copied and is overwritten by each numeric factorization, so instances with one
pattern share it; an instance whose values were replaced refactorizes before solving. A
module lock serializes factor/solve; use processes for parallel work. If K is reported not
positive definite, the solver retries with simplicial LDL' and then LU (``Cholmod.stats``).

``SciPyLU``: scipy.sparse.linalg.splu with COLAMD (fallback without cvxopt).
"""

from __future__ import annotations

import os
import threading

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

try:
    from cvxopt import cholmod as _cholmod
    from cvxopt import matrix as _cvx_matrix
    from cvxopt import spmatrix as _cvx_spmatrix
    HAVE_CHOLMOD = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_CHOLMOD = False

__all__ = ["Cholmod", "default_solver"]


class SciPyLU:
    """Sparse LU with COLAMD ordering (scipy.sparse.linalg.splu), the fallback when cvxopt is
    missing. Backward stable like Cholesky; at the floor lam_min, where K has condition about
    1e10, its forward error can exceed CHOLMOD's by orders of magnitude (test_normal_equations
    measures 7e-8 against 1e-11), so the paper's numbers come from Cholmod."""

    name = "scipy-splu"

    def __init__(self):
        self._lu = None

    def fresh(self):
        return type(self)()

    def factor(self, K):
        self._lu = splu(sp.csc_matrix(K), permc_spec="COLAMD")
        return self

    def solve(self, rhs):
        if self._lu is None:
            raise RuntimeError("call factor(K) first")
        return self._lu.solve(np.asarray(rhs, float))


_LOCK = threading.RLock()


def _supernodal() -> int:
    """CHOLMOD's factorization mode: 2 (supernodal, BLAS; default) or, with the environment
    variable SMOOTHFM_CHOLMOD_SUPERNODAL=0, simplicial. Supernodal solves depend in the last
    bits on memory alignment (BLAS kernel paths), so two processes can differ at ~1e-12 on
    ill-conditioned systems; the smoke runs, whose arrays must repeat bitwise, use the
    simplicial mode (no BLAS). Read at every factorization, so worker processes inherit it."""
    return int(os.environ.get("SMOOTHFM_CHOLMOD_SUPERNODAL", "2"))


class Cholmod:
    """Sparse Cholesky (CHOLMOD via cvxopt) with the symbolic analysis cached per sparsity
    pattern (module docstring); ``stats`` counts symbolic analyses and fallbacks per process,
    a solver self-check (a nonzero fallback count flags a K that was not numerically positive
    definite); no result file stores it."""

    name = "cholmod"
    _symbolic = {}
    stats = {"symbolic": 0, "ldl_fallback": 0, "lu_fallback": 0}

    def __init__(self, cache: bool = True):
        if not HAVE_CHOLMOD:
            raise ImportError("Cholmod needs cvxopt (install the 'cholmod' extra)")
        self.cache = cache
        self._F = self._lu = self._ent = self._vals = None

    def fresh(self):
        return Cholmod(self.cache)

    @staticmethod
    def _lower(K):
        L = sp.tril(sp.csc_matrix(K)).tocsc()
        L.sort_indices()
        return L

    def factor(self, K):
        with _LOCK:
            return self._factor(K)

    def solve(self, rhs):
        with _LOCK:
            return self._solve(rhs)

    def _factor(self, K):
        L = self._lower(K)
        key = (L.shape[0], L.nnz, hash(L.indptr.tobytes()), hash(L.indices.tobytes()))
        ent = self._symbolic.get(key) if self.cache else None
        _cholmod.options["supernodal"] = _supernodal()
        if ent is None:
            S = _cvx_spmatrix(_cvx_matrix(L.data), _cvx_matrix(L.indices.astype(np.int64)),
                              _cvx_matrix(np.repeat(np.arange(L.shape[1], dtype=np.int64),
                                                    np.diff(L.indptr))), L.shape)
            ent = [S, _cholmod.symbolic(S, uplo="L"), None]
            Cholmod.stats["symbolic"] += 1
            if self.cache:
                self._symbolic[key] = ent
        S, F, _ = ent
        self._vals = _cvx_matrix(L.data)
        S.V = self._vals
        self._lu = None
        self._ent = ent
        try:
            _cholmod.numeric(S, F)
            self._F = F
            ent[2] = self
        except ArithmeticError:
            # a half-updated factor must not be taken as anyone's own
            ent[2] = None
            self._symbolic.pop(key, None)
            self._ent = None
            Cholmod.stats["ldl_fallback"] += 1
            _cholmod.options["supernodal"] = 0
            try:
                F = _cholmod.symbolic(S, uplo="L")
                _cholmod.numeric(S, F)
                self._F = F
            except ArithmeticError:
                Cholmod.stats["lu_fallback"] += 1
                self._F = None
                self._lu = SciPyLU().factor(K)
            finally:
                _cholmod.options["supernodal"] = _supernodal()
        return self

    def _solve(self, rhs):
        rhs = np.asarray(rhs, float)
        if self._lu is not None:
            return self._lu.solve(rhs)
        if self._F is None:
            raise RuntimeError("call factor(K) first")
        if self._ent is not None and self._ent[2] is not self:
            S, F, _ = self._ent
            S.V = self._vals
            _cholmod.numeric(S, F)
            self._ent[2] = self
        B = _cvx_matrix(np.array(rhs.reshape(rhs.shape[0], -1), dtype=float, order="F"))
        _cholmod.solve(self._F, B)
        return np.array(B).reshape(rhs.shape)


def default_solver():
    """A new Cholmod when cvxopt imports, SciPyLU otherwise."""
    return Cholmod() if HAVE_CHOLMOD else SciPyLU()
