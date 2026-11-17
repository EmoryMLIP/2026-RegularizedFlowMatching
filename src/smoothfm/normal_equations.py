"""The normal equations eq:normaleq of the problem at one time t (main.tex sec:solver).

At fixed t, eq:regfm restricted to the spline space decouples into the components
d = 1..D. With v_d(x) = sum_a (bold v_d)_a phi_a(x) on Omega, the coefficient vector solves

    min 1/2 v_d^T A v_d - v_d^T b_d + lam/2 v_d^T P v_d + mu/2 v_d^T M v_d
    <=>  (A + mu M + lam P) v_d = b_d,                                     (eq:normaleq)

with A = int phi phi^T rho_N(., t) dx and b_d = int phi rho_N(., t) v_{N,d}(., t) dx
(eq:datamatrix), M the mass matrix, P the regularizer's matrix (regularizers.py) and
mu = config.MU. The code stores the D vectors v_d as the columns of one array C (size, D),
because one factorization serves all components. Every score (``fit_loss``, hold-out)
uses the data matrix A alone, never mu M.

The paper's two numerical safeguards (sec:solver, S:sec:splinedetails) live here:
lam = 0 is computed at the floor lam_min = 1e-14 lam_s (config.LAM_FLOOR_REL), and for
lam >= 100 lam_s (config.SPLIT_RATIO) the solve splits off null(P).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import qr

from . import config
from .regularizers import H2, get_regularizer, mass_matrix
from .solvers import default_solver

__all__ = ["NormalEquations"]


def _csr_sum(*terms):
    """sum_k a_k X_k for CSR matrices that share one pattern; summing the value arrays keeps
    the pattern, so K has the pattern of A at every t and lam and the symbolic factorization
    is reused (solvers.Cholmod)."""
    X0 = terms[0][1]
    data = sum(a * X.data for a, X in terms)
    return sp.csr_matrix((data, X0.indices, X0.indptr), shape=X0.shape)


class NormalEquations:
    """A, b, M and P of eq:normaleq for the training points x_i (N, D) at time t.

    basis: Cubic1D or Cubic2D on Omega = [-L, L]^D with L = latent.L (the latent box and the
    spline box coincide, sec:formulation); latent: TruncatedGaussian; samples: the training
    points in the standardized frame (S:sec:splinedetails); reg: a Regularizer or its key
    (default H2). ``A`` is (size, size) CSR, ``b`` is (size, D), one column b_d per component.

    lam_s = max_j A_jj / max_j P_jj, the scale at which data and regularizer balance
    (sec:solver); lam_min = config.LAM_FLOOR_REL lam_s, the floor that stands for lam = 0;
    ``system(lam)`` = A + mu M + max(lam, lam_min) P.

    Conditioning at large lam: the component in null(P) (the affine fields for the Hessian
    regularizer) is determined by A + mu M alone, and forming K in double precision rounds
    that part away (relative error about eps lam / lam_s). For lam >= config.SPLIT_RATIO lam_s
    and a nontrivial null(P), ``solve`` therefore splits null(P) off (S:sec:splinedetails,
    "Large Regularization Parameters"). P and M may be passed in to share them between
    instances on one basis.
    """

    n_factorizations = 0   # process-wide count of numeric factorizations (cost records)

    def __init__(self, basis, latent, samples, t, reg=None, P=None, M=None):
        self.basis = basis
        self.latent = latent
        self.t = float(t)
        self.reg = H2() if reg is None else get_regularizer(reg)
        self.samples = np.asarray(samples, float)
        if not np.allclose(basis.half_width, latent.L, rtol=1e-12, atol=0):
            raise ValueError(f"basis half-width {basis.half_width} != latent L {latent.L}")
        self.A, self.b = basis.data_matrices(self.samples, self.t, latent)
        self.P = self.reg.matrix(basis) if P is None else P
        self.M = mass_matrix(basis) if M is None else M
        for X, nm in ((self.P, "P"), (self.M, "M")):
            if not (np.array_equal(self.A.indptr, X.indptr) and np.array_equal(self.A.indices, X.indices)):
                raise ValueError(f"A and {nm} must share one sparsity pattern")
        dA, dP = self.A.diagonal().max(), self.P.diagonal().max()
        self.lam_s = float(dA / dP) if dA > 0 else 1.0
        self.lam_min = config.LAM_FLOOR_REL * self.lam_s
        Z = self.reg.null_space(basis)
        nz = 0 if Z is None else Z.shape[1]
        if nz == 0:
            self._Z = np.zeros((self.size, 0))
            self._E = np.arange(self.size)
        else:
            self._Z = np.linalg.qr(Z)[0]
            piv = qr(self._Z.T, mode="r", pivoting=True)[1]
            keep = np.ones(self.size, bool)
            keep[piv[:nz]] = False
            self._E = np.flatnonzero(keep)

    @property
    def size(self) -> int:
        return self.A.shape[0]

    @property
    def AM(self):
        """A + mu M, the lam-independent part of K (from the current A)."""
        return _csr_sum((1.0, self.A), (config.MU, self.M))

    def system(self, lam: float):
        """K = A + mu M + max(lam, lam_min) P, the matrix of eq:normaleq (CSR)."""
        lam = max(float(lam), self.lam_min)
        return _csr_sum((1.0, self.A), (config.MU, self.M), (lam, self.P))

    def factorize(self, lam, solver=None, split="auto") -> "_Factorization":
        """One factorization of K (plain), or of K_EE for the null-space split, which
        split="auto" selects for lam >= SPLIT_RATIO lam_s. Counts n_factorizations."""
        if split == "auto":
            split = float(lam) >= config.SPLIT_RATIO * self.lam_s
        split = bool(split) and self._Z.shape[1] > 0
        NormalEquations.n_factorizations += 1
        return _Factorization(self, lam, solver or default_solver(), split)

    def solve(self, lam, solver=None, split="auto") -> np.ndarray:
        """C (size, D) with K C = b: column d is the coefficient vector v_d of eq:normaleq.

        Split solve: v = Z alpha + E beta, Z the orthonormalized basis of null(P), E the
        coordinate vectors of all indices except the dim null(P) ones where Z is best
        conditioned (QR with column pivoting). Since P Z = 0, Z^T K Z = Z^T (A + mu M) Z and
        K Z = (A + mu M) Z, so the null-space block is formed from A + mu M and not swamped by
        the rounding of lam P; beta comes from K_EE, alpha from the small Schur complement.
        This is the transformation of general-form Tikhonov regularization to standard form
        that sec:solver cites; it reduces the relative error at lam = 1e8 lam_s from 1e-8 to
        1e-15 (S:sec:splinedetails).
        """
        return self.factorize(lam, solver, split).solve(self.b)

    def residual(self, lam, C) -> float:
        """Relative residual ||K C - b||_F / ||b||_F. A solver-accuracy check: Velocity.fit
        records it for every selected solve (the run logs and the selection history of oned.py,
        twod.py and controls.py store it); no table or figure reads it."""
        C = np.asarray(C, float).reshape(self.b.shape)
        nb = np.linalg.norm(self.b)
        return float(np.linalg.norm(self.system(lam) @ C - self.b) / (nb if nb > 0 else 1.0))

    def backward_error(self, lam, C) -> float:
        """Normwise backward error ||K C - b||_F / (||K||_F ||C||_F + ||b||_F) (Frobenius
        norms, consistent since ||K C||_F <= ||K||_F ||C||_F). The relative residual
        ``residual`` reads 0.5-2.5 at lam >= 1e13 lam_s from rounding alone, because
        ||K|| ||C|| >> ||b|| there; this quotient stays at the rounding level eps. A
        solver-accuracy check, recorded with ``residual`` (max_backward_error in the run
        logs); no table or figure reads it."""
        C = np.asarray(C, float).reshape(self.b.shape)
        K = self.system(lam)
        den = spla.norm(K, "fro") * np.linalg.norm(C) + np.linalg.norm(self.b)
        r = np.linalg.norm(K @ C - self.b)
        return float(r / den) if den > 0 else float(r)

    def null_space_limit(self) -> np.ndarray:
        """lim_{lam -> inf} C(lam) = Z (Z^T (A + mu M) Z)^{-1} Z^T b: the weighted least-squares
        fit of v_N in null(P) (for the Hessian regularizer, the affine field); zero if P is
        positive definite (smallness regularizer)."""
        Z = self._Z
        if Z.shape[1] == 0:
            return np.zeros_like(self.b)
        return Z @ np.linalg.solve(Z.T @ (self.AM @ Z), Z.T @ self.b)

    def fit_loss(self, C) -> float:
        """sum_d (v_d^T A v_d - 2 v_d^T b_d), data matrix A only: by eq:dist-vN the loss
        ell_t(v_h) of eq:efm minus a constant that does not depend on the field. With the A, b of
        held-out samples it is the hold-out criterion v_d^T A_val v_d - 2 v_d^T b_{d,val} of
        sec:cv."""
        C = np.asarray(C, float).reshape(self.b.shape)
        return float(np.sum(C * (self.A @ C)) - 2.0 * np.sum(C * self.b))

    def validation(self, samples_val) -> "NormalEquations":
        """NormalEquations of the held-out samples on the same basis, t and regularizer (shares
        P and M): its A and b are A_val and b_val of sec:cv."""
        return NormalEquations(self.basis, self.latent, samples_val, self.t, self.reg,
                               P=self.P, M=self.M)


class _Factorization:
    """K factorized once (plain: K; split: K_EE); ``solve(R)`` for any right-hand side R
    (size, k). It holds the solver, whose factor the next ``factor`` call overwrites, so use
    it before that solver factorizes anything else."""

    def __init__(self, ne: NormalEquations, lam, solver, split: bool):
        self.ne = ne
        self.split = split
        K = ne.system(lam)
        self.solver = solver.factor(K[ne._E][:, ne._E] if split else K)
        if split:
            self.AZ = ne.AM @ ne._Z      # K Z = (A + mu M) Z exactly

    def solve(self, R) -> np.ndarray:
        R = np.asarray(R, float)
        if not self.split:
            return self.solver.solve(R).reshape(R.shape)
        R2 = R.reshape(R.shape[0], -1)
        Z, E, AZ = self.ne._Z, self.ne._E, self.AZ
        X = self.solver.solve(np.hstack([AZ[E], R2[E]]))
        nz = Z.shape[1]
        XA, XR = X[:, :nz], X[:, nz:]
        S = Z.T @ AZ - AZ[E].T @ XA
        alpha = np.linalg.solve(S, Z.T @ R2 - AZ[E].T @ XR)
        C = Z @ alpha
        C[E] += XR - XA @ alpha
        return C.reshape(R.shape)
