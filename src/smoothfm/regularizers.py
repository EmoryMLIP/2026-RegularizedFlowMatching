"""The regularizers |u|_{H^k}^2 of eq:regfm as matrices of the normal equations eq:normaleq.

main.tex sec:formulation defines the regularizer of order k as the squared seminorm
|u|_{H^k}^2 = int_Omega ||nabla^k u||_F^2 dx of each component of the field u and calls the
orders k = 2, 1 and 0 the Hessian, gradient and smallness regularizers. Its null space is
the polynomials of degree < k. A ``Regularizer`` turns a spline basis into the matrix
bold P with v^T P v = |v_h|_{H^k}^2 (S:sec:splinedetails) and names a basis of null(P),
which NormalEquations splits off for lam >= 100 lam_s:

    class  key   paper name               |u|_{H^k}^2            null(P)
    H2     "h2"  Hessian (k = 2)          int ||nabla^2 u||_F^2  affine functions
    H1     "h1"  gradient (k = 1)         int ||nabla u||^2      constants
    L2     "l2"  smallness (k = 0)        int |u|^2              {0}  (P = P_0 = M)

1D: P = P_k. 2D (eq:regmatrix and the line after it): P = P_2 (x) P_0 + 2 P_1 (x) P_1 +
P_0 (x) P_2 for k = 2, P_1 (x) P_0 + P_0 (x) P_1 for k = 1, P_0 (x) P_0 for k = 0.

The table maps the class names and keys to the paper's names; the keys name the result files
("oned_h1", "..._l2") and index config.LAM_DECADES.

Frame scaling: under x = c + s x' (S:sec:frames) a field u'(x') = u(c + s x') / s has
|u'|_{H^k}^2 = s^(2k - 2 - D) |u|_{H^k}^2 while the data term scales like s^-2, so the
optimal lam scales by s^(D - 2k). lam_s transforms alike, which is why the candidates are
relative to lam_s.
"""

from __future__ import annotations

__all__ = ["Regularizer", "H2", "H1", "L2", "get_regularizer"]


def mass_matrix(basis):
    """Mass matrix bold M = int_Omega phi phi^T dx of the mass term mu ||u||_{L^2(Omega)}^2 in
    eq:regfm: P_0 in 1D, P_0 (x) P_0 in 2D (S:sec:splinedetails)."""
    return basis.energy(0)


class Regularizer:
    """The regularizer |v|_{H^k}^2 = int_Omega ||nabla^k v||_F^2 dx of order k = ``order``;
    ``name`` is the registry key."""

    name: str = ""
    order: int = 0

    def matrix(self, basis):
        """Bold P (size x size, CSR, symmetric positive semidefinite) on the basis's
        canonical sparsity pattern, which A and M share, with v^T P v = |v_h|_{H^k}^2 exactly.
        The basis must be C^{k-1} so that v_h lies in H^k."""
        if basis.continuity < self.order - 1:
            raise ValueError(f"{self.name} needs a C^{self.order - 1} basis")
        return basis.energy(self.order)

    def null_space(self, basis):
        """Coefficient vectors (size, dim null(P)) spanning null(P), the polynomials of degree
        < k (1 for k = 1, D + 1 for k = 2); None for k = 0, where P = M is positive
        definite."""
        return None if self.order == 0 else basis.polynomial_coefficients(self.order - 1)

    def __repr__(self):
        return f"{type(self).__name__}()"


class H2(Regularizer):
    """Hessian regularizer (k = 2), int_Omega ||nabla^2 v||_F^2 dx, the paper's default; in
    2D the thin-plate energy. No boundary conditions are imposed, so the minimizer satisfies
    the natural ones (sec:solver)."""

    name, order = "h2", 2


class H1(Regularizer):
    """Gradient regularizer (k = 1), int_Omega ||nabla v||^2 dx."""

    name, order = "h1", 1


class L2(Regularizer):
    """Smallness regularizer (k = 0), int_Omega |v|^2 dx; P = M (standard-form Tikhonov)."""

    name, order = "l2", 0


REGULARIZERS = {"h2": H2, "h1": H1, "l2": L2}


def get_regularizer(reg) -> Regularizer:
    """A Regularizer from an instance, a subclass, or a key of REGULARIZERS ("h2", "h1",
    "l2"); the registry lets scripts select the regularizer by a command-line string."""
    if isinstance(reg, Regularizer):
        return reg
    if isinstance(reg, type) and issubclass(reg, Regularizer):
        return reg()
    try:
        return REGULARIZERS[reg]()
    except (KeyError, TypeError):
        raise ValueError(f"unknown regularizer {reg!r}; choose from {tuple(REGULARIZERS)}") from None
