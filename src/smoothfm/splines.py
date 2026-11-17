"""Cubic B-spline bases on Omega = [-L, L]^D with exact Gram matrices and data matrices
(main.tex sec:solver; S:sec:splinedetails, eq:datamatrix, eq:regmatrix).

Notation of S:sec:splinedetails: reference coordinate xi in [-1, 1] per axis, x = L xi;
uniform knots xi_j = -1 + j h_xi, j = 0..m, h_xi = 2/m; mesh width h = L h_xi. The n = m + 3
basis functions are b_a(xi) = b((xi - xi_a)/h_xi), a = -1..m+1 (array index a + 1), with b the
cubic cardinal B-spline on [-2, 2], restricted to [-1, 1]. On cell j, with the local
coordinate tau = (xi - xi_j)/h_xi and taubar = 1 - tau, the four nonzero basis functions are
the pieces q_0..q_3 (``local_bsplines``)

    taubar^3/6,  (4 - 6 tau^2 + 3 tau^3)/6,  (4 - 6 taubar^2 + 3 taubar^3)/6,  tau^3/6.

Matrices of eq:normaleq:
- Gram matrices (P_k)_{aa'} = int_Omega b_a^(k) b_a'^(k) dx, k = 0, 1, 2 (``Cubic1D.gram``),
  exact by 4-point Gauss-Legendre per cell; bandwidth three, stored in band form (n, 7).
- Regularizer and mass matrices: ``energy(k)``; in 2D the Kronecker sums of eq:regmatrix.
- Data matrices of eq:datamatrix: A = (1/N) sum_i T_i, b = (1/N) sum_i g_i in 1D, and
  A = (1/N) sum_i T_i^(1) (x) T_i^(2), b_1 = (1/N) sum_i g_i^(1) (x) f_i^(2),
  b_2 = (1/N) sum_i f_i^(1) (x) g_i^(2) in 2D (``data_matrices``), from the per-sample tables
  T_i, f_i, g_i of ``Cubic1D.local_tables``.
Every matrix is stored on one canonical CSR pattern (7 diagonals in 1D, 49 entries per row in
2D, zeros stored), so A at any t, P and M share one symbolic factorization.

Outside Omega (only the sampler evaluates there, when an RK4 stage leaves the box) each spline
is continued by the cubic of its boundary cell, a C^2 extension that reproduces affine
functions; x is first clipped per axis to Omega widened by one cell (``far_clip``), so the
extension cannot grow cubically (S:sec:frames).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .latent import _ranges, as_points, gauss_legendre01

__all__ = ["Cubic1D", "Cubic2D", "far_clip"]

BAND = 3  # half-bandwidth of the 1D matrices (cubic: |a - a'| <= 3)
#: samples per assembly block of Cubic2D.data_matrices: N <= 10_000 (every paper set) is
#: one block, the unblocked computation
ASSEMBLY_BLOCK = 10_000


def local_bsplines(tau, taubar=None, deriv=0):
    """The pieces q_0..q_3 (module docstring) at the local coordinate tau, or their
    deriv-th tau-derivatives (deriv = 0, 1, 2), shape (..., 4). Passing taubar = 1 - tau
    computed without cancellation keeps the pieces accurate near tau = 1."""
    tau = np.asarray(tau, float)
    tb = 1.0 - tau if taubar is None else np.asarray(taubar, float)
    if deriv == 0:
        cols = (tb**3 / 6, (4 - 6 * tau**2 + 3 * tau**3) / 6,
                (4 - 6 * tb**2 + 3 * tb**3) / 6, tau**3 / 6)
    elif deriv == 1:
        cols = (-tb**2 / 2, (-4 * tau + 3 * tau**2) / 2, (4 * tb - 3 * tb**2) / 2, tau**2 / 2)
    elif deriv == 2:
        cols = (tb, 3 * tau - 2, 3 * tb - 2, tau)
    else:
        raise ValueError("deriv must be 0, 1 or 2")
    return np.stack(cols, axis=-1)


def band_to_sparse(Tb):
    """Band storage (n, 7), Tb[a, d + 3] = T[a, a + d], to CSR on the canonical 7-diagonal
    pattern (zeros stored), so that A at any t, the Gram matrices and P share one pattern and
    one symbolic factorization."""
    Tb = np.asarray(Tb, float)
    n = Tb.shape[0]
    cols = np.arange(n)[:, None] + np.arange(-BAND, BAND + 1)
    valid = (cols >= 0) & (cols < n)
    indptr = np.concatenate([[0], np.cumsum(valid.sum(axis=1))])
    return sp.csr_matrix((Tb[valid], cols[valid], indptr), shape=(n, n))


def band2_to_sparse(V):
    """2D band storage (nx, ny, 7, 7), V[a, b, da + 3, db + 3] = M[(a, b), (a + da, b + db)],
    to CSR on the canonical 49-point pattern (row a ny + b; canonical column order)."""
    V = np.asarray(V, float)
    nx, ny = V.shape[:2]
    d = np.arange(-BAND, BAND + 1)
    ca = np.arange(nx)[:, None, None, None] + d[None, None, :, None]
    cb = np.arange(ny)[None, :, None, None] + d[None, None, None, :]
    valid = (ca >= 0) & (ca < nx) & (cb >= 0) & (cb < ny)
    cols = np.broadcast_to(ca * ny + cb, valid.shape)[valid]
    indptr = np.concatenate([[0], np.cumsum(valid.reshape(nx * ny, -1).sum(axis=1))])
    return sp.csr_matrix((V[valid], cols, indptr), shape=(nx * ny, nx * ny))


def band_kron(X, Y):
    """X (x) Y of 1D band matrices (nx, 7), (ny, 7) in 2D band storage (nx, ny, 7, 7)."""
    return X[:, None, :, None] * Y[None, :, None, :]


@dataclass(frozen=True)
class LocalTables:
    """Per-sample tables T_i, f_i, g_i of eq:datamatrix on a window of W consecutive basis
    functions (the truncated kernel of x_i meets only those):
    (T_i)[start_i + r, start_i + r + d] = T[i, r, d + 3], (f_i)[start_i + r] = F[i, r],
    (g_i)[start_i + r] = G[i, r]; n is the basis size."""

    start: np.ndarray
    T: np.ndarray
    F: np.ndarray
    G: np.ndarray
    n: int

    @property
    def width(self) -> int:
        return self.F.shape[1]

    def rows(self) -> np.ndarray:
        return self.start[:, None] + np.arange(self.width)

    def _sum(self, V):
        rows = self.rows().ravel()
        flat = V.reshape(rows.size, -1)
        out = np.stack([np.bincount(rows, flat[:, k], minlength=self.n)
                        for k in range(flat.shape[1])], axis=1)
        return out[:, 0] if V.ndim == 2 else out

    def sum_T(self):
        return self._sum(self.T)

    def sum_G(self):
        return self._sum(self.G)

    def window_matrix(self, V, dense=False):
        """Per-sample window vectors V (N, W) placed at their basis indices: (N, n), CSR or
        dense."""
        N, W = V.shape
        if dense:
            out = np.zeros((N, self.n))
            out[np.arange(N)[:, None], self.rows()] = V
            return out
        return sp.csr_matrix((V.ravel(), self.rows().ravel(), W * np.arange(N + 1)),
                             shape=(N, self.n))


class Cubic1D:
    """C^2 cubic splines on m uniform cells of Omega = [-L, L] (n = m + 3 functions b_a),
    half_width = L. The space lies in H^2(Omega) and contains the affine functions, so it
    supports the regularizers of order k <= 2 and their null spaces (``continuity`` = 2)."""

    dim = 1
    continuity = 2

    def __init__(self, m: int, half_width: float = 1.0):
        if m < 1:
            raise ValueError("m must be >= 1")
        if not half_width > 0:
            raise ValueError("half_width must be positive")
        self.m = int(m)
        self.ell = float(half_width)
        self.h_xi = 2.0 / self.m
        self.h = self.ell * self.h_xi

    def __repr__(self):
        return f"Cubic1D(m={self.m}, half_width={self.ell:g})"

    @property
    def n(self) -> int:
        return self.m + 3

    @property
    def size(self) -> int:
        return self.m + 3

    @property
    def half_width(self):
        return np.array([self.ell])

    def to_reference(self, x):
        return np.asarray(x, float) / self.ell

    # -- evaluation ------------------------------------------------------------------
    def _locate(self, xi):
        s = (np.clip(np.asarray(xi, float), -1.0, 1.0) + 1.0) / self.h_xi
        j = np.clip(np.floor(s).astype(np.int64), 0, self.m - 1)
        return j, s - j, (j + 1) - s

    def eval(self, xi, deriv=0):
        """Design matrix (CSR, npts x n), V[p, a] = d^k b_a / dx^k at x = L xi_p, k = deriv
        (physical derivatives, d/dx = (1/h) d/dtau); points outside [-1, 1] are clipped to the
        boundary."""
        xi = np.asarray(xi, float).ravel()
        j, tau, tb = self._locate(xi)
        vals = local_bsplines(tau, tb, deriv) / self.h**deriv
        npts = xi.size
        return sp.csr_matrix((vals.ravel(), (j[:, None] + np.arange(4)).ravel(),
                              4 * np.arange(npts + 1)), shape=(npts, self.n))

    def extended_eval(self, xi, deriv=0):
        """Design matrix with the C^2 extension: the boundary cell's cubic continues the
        spline (local_bsplines at tau outside [0, 1]); far_clip bounds xi beforehand."""
        xi = np.asarray(xi, float).ravel()
        s = (xi + 1.0) / self.h_xi
        j = np.clip(np.floor(s).astype(np.int64), 0, self.m - 1)
        tau, tb = s - j, (j + 1) - s
        vals = local_bsplines(tau, tb, deriv) / self.h**deriv
        npts = xi.size
        return sp.csr_matrix((vals.ravel(), (j[:, None] + np.arange(4)).ravel(),
                              4 * np.arange(npts + 1)), shape=(npts, self.n))

    def extended_design(self, x, deriv=0):
        """extended_eval at physical points x."""
        return self.extended_eval(self.to_reference(as_points(x, 1)[..., 0]), int(np.squeeze(deriv)))

    # -- Gram matrices ---------------------------------------------------------------
    def element_matrix(self, order: int) -> np.ndarray:
        """int_0^1 q_p^(order) q_r^(order) dtau (4 x 4) of the cell pieces, by 4-node
        Gauss-Legendre (exact for degree <= 7)."""
        x, w, xb = gauss_legendre01(4)
        V = local_bsplines(x, xb, order)
        return (V * w[:, None]).T @ V

    def gram_band(self, order: int) -> np.ndarray:
        """P_order in band storage (n, 7): the element matrices scaled by h^(1 - 2 order)
        (dx = h dtau, d/dx = (1/h) d/dtau) and assembled over the m cells."""
        E = self.element_matrix(order) * self.h ** (1 - 2 * order)
        j = np.arange(self.m)[:, None, None]
        p = np.arange(4)[:, None]
        q = np.arange(4)[None, :]
        idx = ((j + p) * 7 + (q - p + BAND)).ravel()
        vals = np.broadcast_to(E, (self.m, 4, 4)).ravel()
        return np.bincount(idx, vals, minlength=self.n * 7).reshape(self.n, 7)

    def gram(self, k: int):
        """(P_k)_{aa'} = int_Omega b_a^(k) b_a'^(k) dx, k = 0, 1, 2 (S:sec:splinedetails),
        exact, CSR on the canonical pattern."""
        if k not in (0, 1, 2):
            raise ValueError("k must be 0, 1 or 2")
        return band_to_sparse(self.gram_band(k))

    def energy(self, order: int = 2):
        """Matrix of |v|_{H^k}^2 = int_Omega (v^(k))^2 dx for k = order: P_k in 1D."""
        return self.gram(order)

    def affine_coefficients(self):
        """Coefficients Z (n, 2) of the functions 1 and x: the constant has all coefficients 1,
        and x has the Greville points L (-1 + a h_xi), a = -1..m+1, as coefficients."""
        xa = self.ell * (-1.0 + self.h_xi * np.arange(-1, self.m + 2))
        return np.column_stack([np.ones(self.n), xa])

    def polynomial_coefficients(self, degree: int):
        """Coefficients (size, 1 + degree D) of the monomials of degree <= degree (0: the
        constant, 1: the affine functions), the null space of P_{degree + 1}."""
        if degree not in (0, 1):
            raise NotImplementedError("degree 0 or 1")
        return self.affine_coefficients()[:, :1 + degree * self.dim]

    # -- data tables (eq:datamatrix) -------------------------------------------------
    def local_tables(self, mu, sigma, targets, latent) -> LocalTables:
        """Per-sample integrals over Omega against the latent kernel K_sigma(x - mu_i):

            T[i, a, a'] = int b_a b_a' K_sigma(x - mu_i) dx,   F[i, a] = int b_a K_sigma(x - mu_i) dx,
            G[i, a] = int b_a K_sigma(x - mu_i) (y_i - x)/sigma dx,   y_i = targets[i].

        With centres mu_i = t x_i, bandwidth sigma = 1 - t and targets y_i = x_i these are
        (T_i)_{aa'}, (f_i)_a and (g_i)_a of eq:datamatrix. The argument mu is the kernel centre,
        not the mass weight mu of eq:regfm. With u = (x - mu)/sigma each cell is
        [alpha, alpha + h/sigma] in u, integrated by the latent's positive cell rule (accurate
        to 5e-14 relative for degree <= 6). Only cells that meet the box mu_i + sigma [-L, L] contribute: the
        others carry zero mass of the truncated kernel, so the cut is exact.
        """
        mu = np.atleast_1d(np.asarray(mu, float)).ravel()
        y = np.atleast_1d(np.asarray(targets, float)).ravel()
        if y.size != mu.size:
            raise ValueError("mu and targets must have the same length")
        sigma = float(sigma)
        if not sigma > 0:
            raise ValueError("sigma must be positive")
        N, n, m, h = mu.size, self.n, self.m, self.h
        x0 = -self.ell
        lo, hi = -latent.L, latent.L
        j0 = np.clip(np.floor((mu + lo * sigma - x0) / h), 0, m).astype(np.int64)
        j1 = np.clip(np.ceil((mu + hi * sigma - x0) / h), 0, m).astype(np.int64)
        count = np.maximum(j1 - j0, 0)
        W = min(int(count.max(initial=0)) + 3, n)
        start = np.minimum(j0, n - W)
        ii = np.repeat(np.arange(N), count)
        jj = np.repeat(j0, count) + _ranges(count)
        xj = x0 + h * jj
        alpha = (xj - mu[ii]) / sigma
        rule = latent.cell_rule(alpha, alpha + h / sigma)
        i, j = ii[rule.cell], jj[rule.cell]
        w, tau, tb = rule.weight, rule.tau, rule.taubar
        b = local_bsplines(tau, tb, 0)
        g = ((y[i] - xj[rule.cell]) - h * tau) / sigma
        size = N * W
        Tb = np.zeros(size * 7)
        F = np.zeros(size)
        G = np.zeros(size)
        row0 = i * W + (j - start[i])
        for p in range(4):
            wb = w * b[:, p]
            F += np.bincount(row0 + p, wb, minlength=size)
            G += np.bincount(row0 + p, wb * g, minlength=size)
            for q in range(p, 4):
                Tb += np.bincount((row0 + p) * 7 + (q - p + BAND), wb * b[:, q], minlength=size * 7)
        Tb = Tb.reshape(N, W, 7)
        for d in range(1, BAND + 1):
            Tb[:, d:, BAND - d] = Tb[:, :-d, BAND + d]
        return LocalTables(start, Tb, F.reshape(N, W), G.reshape(N, W), n)

    def data_matrices(self, samples, t, latent):
        """A = (1/N) sum_i T_i (CSR, n x n) and b = (1/N) sum_i g_i (n, 1) of eq:datamatrix for
        the training points x_i at time t (kernel centres t x_i, bandwidth 1 - t)."""
        x = as_points(samples, 1)[..., 0].ravel()
        t = float(t)
        tab = self.local_tables(t * x, 1.0 - t, x, latent)
        N = x.size
        return band_to_sparse(tab.sum_T() / N), (tab.sum_G() / N)[:, None]


def _rowwise_kron(U, V, ny):
    """Row-wise Kronecker products U_p (x) V_p of two design matrices with 4 sorted entries
    per row: the tensor-product basis at the points."""
    p = U.shape[0]
    Ud, Ui = U.data.reshape(p, 4), U.indices.reshape(p, 4)
    Vd, Vi = V.data.reshape(p, 4), V.indices.reshape(p, 4)
    data = (Ud[:, :, None] * Vd[:, None, :]).ravel()
    idx = (Ui[:, :, None] * ny + Vi[:, None, :]).ravel()
    return sp.csr_matrix((data, idx, 16 * np.arange(p + 1)), shape=(p, U.shape[1] * ny))


class Cubic2D:
    """Tensor-product cubic splines b_a(xi_1) b_a'(xi_2) on Omega = [-L, L]^2: two Cubic1D
    (``bx``, ``by``); coefficients ordered lexicographically with the second index running
    fastest (index a n_2 + a'), as in S:sec:splinedetails."""

    dim = 2
    continuity = 2

    def __init__(self, m_x: int, m_y: int, half_width: float = 1.0):
        self.bx = Cubic1D(m_x, half_width)
        self.by = Cubic1D(m_y, half_width)

    def __repr__(self):
        return f"Cubic2D(m_x={self.bx.m}, m_y={self.by.m}, half_width={self.bx.ell:g})"

    @property
    def shape(self):
        return (self.bx.n, self.by.n)

    @property
    def size(self) -> int:
        return self.bx.n * self.by.n

    @property
    def half_width(self):
        return np.array([self.bx.ell, self.by.ell])

    @property
    def h_xi(self):
        return np.array([self.bx.h_xi, self.by.h_xi])

    def to_reference(self, x):
        return np.asarray(x, float) / self.half_width

    def eval(self, xi, deriv=(0, 0)):
        """Design matrix (CSR, npts x size), V[p, a n_2 + a'] = d^k1 b_a(xi_p1) d^k2 b_a'(xi_p2)
        (physical derivatives) at reference points xi (npts, 2), deriv = (k1, k2)."""
        xi = np.asarray(xi, float).reshape(-1, 2)
        kx, ky = (deriv, 0) if np.isscalar(deriv) else deriv
        return _rowwise_kron(self.bx.eval(xi[:, 0], kx), self.by.eval(xi[:, 1], ky), self.by.n)

    def extended_design(self, x, deriv=(0, 0)):
        """Design matrix with the C^2 extension per axis (Cubic1D.extended_eval)."""
        xi = self.to_reference(as_points(x, 2)).reshape(-1, 2)
        kx, ky = (deriv, 0) if np.isscalar(deriv) else deriv
        return _rowwise_kron(self.bx.extended_eval(xi[:, 0], kx),
                             self.by.extended_eval(xi[:, 1], ky), self.by.n)

    def energy(self, order: int = 2):
        """Matrix of |v|_{H^k}^2 for k = order on the canonical 49-entry pattern (eq:regmatrix
        and the line after it; the first factor acts in direction 1):
            2: P_2 (x) P_0 + 2 P_1 (x) P_1 + P_0 (x) P_2,  1: P_1 (x) P_0 + P_0 (x) P_1,
            0: P_0 (x) P_0 = M."""
        gx = [self.bx.gram_band(k) for k in (0, 1, 2)]
        gy = [self.by.gram_band(k) for k in (0, 1, 2)]
        if order == 2:
            V = band_kron(gx[2], gy[0]) + 2 * band_kron(gx[1], gy[1]) + band_kron(gx[0], gy[2])
        elif order == 1:
            V = band_kron(gx[1], gy[0]) + band_kron(gx[0], gy[1])
        elif order == 0:
            V = band_kron(gx[0], gy[0])
        else:
            raise ValueError("order must be 0, 1 or 2")
        return band2_to_sparse(V)

    def affine_coefficients(self):
        """Coefficients Z (size, 3) of the functions 1, x_1, x_2 (tensor products of the 1D
        ones)."""
        zx, zy = self.bx.affine_coefficients(), self.by.affine_coefficients()
        ox, oy = zx[:, 0], zy[:, 0]
        return np.column_stack([np.kron(ox, oy), np.kron(zx[:, 1], oy), np.kron(ox, zy[:, 1])])

    def polynomial_coefficients(self, degree: int):
        if degree not in (0, 1):
            raise NotImplementedError("degree 0 or 1")
        return self.affine_coefficients()[:, :1 + degree * self.dim]

    def data_matrices(self, samples, t, latent):
        """A = (1/N) sum_i T_i^(1) (x) T_i^(2) (CSR) and b (size, 2) with the columns
        b_1 = (1/N) sum_i g_i^(1) (x) f_i^(2), b_2 = (1/N) sum_i f_i^(1) (x) g_i^(2)
        (S:sec:splinedetails); the sums factor because the truncated Gaussian kernel is a
        product kernel. All 28 band offset pairs with da >= 0 come from one product
        [X_0..X_3]^T [Y_-3..Y_3] of windowed per-sample columns; da < 0 by symmetry."""
        x = as_points(samples, 2).reshape(-1, 2)
        t = float(t)
        N = x.shape[0]
        nx, ny = self.shape
        R = bx = by = None
        for i0 in range(0, N, ASSEMBLY_BLOCK):
            Rk, bxk, byk = self._data_block(x[i0:i0 + ASSEMBLY_BLOCK], t, latent)
            R, bx, by = ((Rk, bxk, byk) if R is None else (R + Rk, bx + bxk, by + byk))
        R = R.reshape(BAND + 1, nx, 2 * BAND + 1, ny)
        V = np.zeros((nx, ny, 2 * BAND + 1, 2 * BAND + 1))
        V[:, :, BAND:, :] = R.transpose(1, 3, 0, 2)
        for da in range(1, BAND + 1):
            for db in range(-BAND, BAND + 1):
                b0, b1 = max(0, -db), min(ny, ny - db)
                V[da:, b0 + db:b1 + db, BAND - da, BAND - db] = V[:nx - da, b0:b1, BAND + da, BAND + db]
        A = band2_to_sparse(V / N)
        b = np.stack([np.asarray(bx).ravel(), np.asarray(by).ravel()], axis=1) / N
        return A, b

    def _data_block(self, x, t, latent):
        """Unnormalized band products R and right-hand sides of one block of samples."""
        sigma = 1.0 - t
        tx = self.bx.local_tables(t * x[:, 0], sigma, x[:, 0], latent)
        ty = self.by.local_tables(t * x[:, 1], sigma, x[:, 1], latent)
        nx, ny = self.shape
        # sparse products cost ~ N Wx Wy at a much higher constant than dense BLAS N nx ny
        dense = 20 * tx.width * ty.width > nx * ny
        Xs = [tx.window_matrix(tx.T[:, :, BAND + da], dense) for da in range(BAND + 1)]
        Ys = [ty.window_matrix(ty.T[:, :, BAND + db], dense) for db in range(-BAND, BAND + 1)]
        cat = np.hstack if dense else (lambda m: sp.hstack(m, format="csr"))
        R = cat(Xs).T @ cat(Ys)
        R = R if dense else R.toarray()
        Gx, Fx = tx.window_matrix(tx.G, dense), tx.window_matrix(tx.F, dense)
        Gy, Fy = ty.window_matrix(ty.G, dense), ty.window_matrix(ty.F, dense)
        bx, by = Gx.T @ Fy, Fx.T @ Gy
        if not dense:
            bx, by = bx.toarray(), by.toarray()
        return R, bx, by


def far_clip(basis, x):
    """x clipped per axis to Omega widened by one cell, and the mask (n, D) of the clipped
    coordinates: beyond one cell the C^2 extension would grow cubically, so the velocity is
    held at its value there and its derivative along a clipped axis is zero."""
    h_xi = np.atleast_1d(basis.h_xi)
    lim = basis.half_width * (1.0 + h_xi)
    return np.clip(x, -lim, lim), (x < -lim) | (x > lim)
