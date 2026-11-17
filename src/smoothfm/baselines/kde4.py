"""Fourth-order kernel density estimators in d = 1 and d = 2 (S:sec:kde).

The paper's G^[4](u) = (1/2)(3 - u^2) G(u) in 1D and the radial (2 - |u|^2/2) G(u) in 2D are
``k4`` and ``k4_2d``; the product kernel is ``k4_prod``. The code writes h for the paper's
bandwidth sigma and lam for the Abramson factors sigma_i / sigma.

Gaussian-based fourth-order kernels (bias O(h^4) instead of O(h^2), at the price of
negative values): the kernels, clipped and renormalized grid densities (wrapped by
grid_density.GridDensity1D / GridDensity2D for exact sampling), and closed-form LSCV curves.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fftfreq, irfft2, rfft2, rfftfreq

from .kde import _bin_2d, _phi, kde_grid_1d


# ------------------------------------------------------------------- kernels

def k4(u):
    """Gaussian-based fourth-order kernel K4(u) = (1/2)(3 - u^2) phi(u) = phi(u) - phi''(u)/2.

    Moments: int K4 = 1, int u K4 = int u^2 K4 = int u^3 K4 = 0, int u^4 K4 = -3. The
    estimator's bias is O(h^4) on densities with four derivatives (O(h^2) for phi), at the
    price of negative values in the tails and between modes.
    """
    u = np.asarray(u, float)
    return 0.5 * (3.0 - u * u) * _phi(u)


# The 2D kernels of S:sec:kde: the radial K4 = phi_2 - (1/2) Laplacian phi_2, the d = 2
# analogue of the 1D k4 = phi - phi''/2, whose fourth-moment tensor is minus that of phi_2,
# so that the bias -(h^4/8) Laplacian^2 f does not depend on the coordinate axes; and the
# product kernel k4(u_1) k4(u_2), whose bias -(h^4/8)(f_1111 + f_2222) does.

def k4_2d(u):
    """Radial fourth-order kernel in d = 2 (u of shape (..., 2)):

        K4(u) = (1 + d/2 - |u|^2/2) phi_d(u) = (2 - |u|^2/2) phi_2(u)
              = phi_2(u) - (1/2) Laplacian phi_2(u)      (Laplacian phi_2 = (|u|^2 - 2) phi_2).

    Moments: int K4 = 1, first, second and third moments 0, int u_1^4 K4 = -3,
    int u_1^2 u_2^2 K4 = -1 (minus the Gaussian's), so the bias is
    -(h^4/8) Laplacian^2 f + O(h^6)."""
    u = np.asarray(u, float)
    r2 = np.sum(u * u, -1)
    return (2.0 - 0.5 * r2) * np.exp(-0.5 * r2) / (2.0 * np.pi)


def k4_prod(u):
    """Product fourth-order kernel K(u) = k4(u_1) k4(u_2), k4(s) = (1/2)(3 - s^2) phi(s).
    Moments: int K = 1, second moments 0, int u_1^4 K = -3, int u_1^2 u_2^2 K = 0."""
    u = np.asarray(u, float)
    return k4(u[..., 0]) * k4(u[..., 1])


def _k4_cov_values(r, S):
    """K4_S(r) = (2 - q/2) N(r; 0, S), q = r^T S^{-1} r, for 2 x 2 covariances S
    broadcast against r (..., 2)."""
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    det = a * c - b * b
    q = (c * r[..., 0] ** 2 - 2 * b * r[..., 0] * r[..., 1] + a * r[..., 1] ** 2) / det
    return (2.0 - 0.5 * q) * np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(det))


# ------------------------------------------------------------ grid densities

def kde4_grid_1d(ys, h, xg, lam=None, log_bin: float = 0.01, return_clipped: bool = True):
    """Clipped and renormalized fourth-order KDE on the uniform grid xg:

        f_+(x) = max(f_h(x), 0) / (dx sum_k max(f_h(x_k), 0)),

    so f_+ >= 0 and its Riemann sum on xg is 1 (the clipped mass is redistributed in
    proportion to f_+, S:sec:kde). f_h(x) = (1/n) sum_i K4((x - y_i)/h_i) / h_i is the
    unclipped K4 estimate with h_i = h lam_i (Abramson; lam = None: h_i = h), evaluated by
    linear binning and FFT as kde_grid_1d (error O(dx^2/h^2) relative to max f).
    """
    xg = np.asarray(xg, float)
    ys = np.asarray(ys, float).ravel()
    f = kde_grid_1d(ys, h, xg, lam, log_bin=log_bin, kernel=k4)
    dx = xg[1] - xg[0]
    neg = float(np.sum(np.maximum(-f, 0.0)) * dx)
    f = np.maximum(f, 0.0)
    f = f / (np.sum(f) * dx)
    return (f, neg) if return_clipped else f


def _clip_grid(f, dx, return_clipped):
    """f_+ = max(f, 0) / (dx^2 sum max(f, 0)) and, optionally, the clipped mass
    dx^2 sum max(-f, 0) of the raw values."""
    neg = float(np.sum(np.maximum(-f, 0.0)) * dx * dx)
    fp = np.maximum(f, 0.0)
    fp = fp / (np.sum(fp) * dx * dx)
    return (fp, neg) if return_clipped else fp


def kde4_grid_2d(X, h, xg, kernel: str = "radial", clip: bool = True,
                 return_clipped: bool = True, nfft: int = 1024):
    """2D fourth-order KDE on the square grid xg x xg (ij indexing), clipped and
    renormalized like kde4_grid_1d:

        f_+(x) = max(f_h(x), 0) / (dx^2 sum_k max(f_h(x_k), 0)),

    f_h(x) = (1/n) sum_i h^{-2} K((x - y_i)/h) the raw estimate with the radial (k4_2d) or
    product (k4_prod) kernel. With return_clipped, also the clipped mass
    dx^2 sum max(-f_h, 0); clip=False returns f_h. Evaluation by bilinear binning and FFT with
    the kernel's Fourier transform, as kde_grid_2d (frequencies k in cycles per unit,
    g(k) = exp(-2 pi^2 h^2 |k|^2) the Gaussian's):
        radial   g(k) (1 + 2 pi^2 h^2 |k|^2)     (from K4 = phi - Laplacian phi / 2),
        product  g(k) (1 + 2 pi^2 h^2 k_1^2)(1 + 2 pi^2 h^2 k_2^2),
    error O(dx^2/h^2)."""
    X = np.asarray(X, float)
    xg = np.asarray(xg, float)
    dx = xg[1] - xg[0]
    fx2 = fftfreq(nfft, dx)[:, None] ** 2
    fy2 = rfftfreq(nfft, dx)[None] ** 2
    a = 2 * np.pi**2 * h * h
    tr = np.exp(-a * (fx2 + fy2))
    tr = tr * (1 + a * (fx2 + fy2)) if kernel == "radial" else tr * (1 + a * fx2) * (1 + a * fy2)
    c = rfft2(_bin_2d(X, xg, 1.0 / len(X)), s=(nfft, nfft))
    f = irfft2(c * tr, s=(nfft, nfft))[:xg.size, :xg.size]
    if not clip:
        return f
    return _clip_grid(f, dx, return_clipped)


def kde4_aniso_grid_2d(X, cov, xg, cut: float = 6.0, clip: bool = True,
                       return_clipped: bool = True):
    """Anisotropic fourth-order KDE f(x) = (1/n) sum_i K4_{S_i}(x - y_i) on the grid
    xg x xg (ij indexing), by direct summation over each kernel's ellipse
    r^T S_i^{-1} r <= cut^2 (|K4_S| there < 4e-7 of its peak for cut = 6); clipped and
    renormalized as kde4_grid_2d."""
    X = np.asarray(X, float)
    xg = np.asarray(xg, float)
    S = np.broadcast_to(np.asarray(cov, float), (len(X), 2, 2))
    dx = xg[1] - xg[0]
    f = np.zeros((xg.size, xg.size))
    # the ellipse's extent along axis d is cut sqrt(S_dd)
    wid = cut * np.sqrt(np.stack([S[:, 0, 0], S[:, 1, 1]], 1))
    lo = np.clip(np.ceil((X - wid - xg[0]) / dx).astype(int), 0, xg.size)
    hi = np.clip(np.floor((X + wid - xg[0]) / dx).astype(int) + 1, 0, xg.size)
    for i in range(len(X)):
        if lo[i, 0] >= hi[i, 0] or lo[i, 1] >= hi[i, 1]:
            continue
        r = np.stack(np.meshgrid(xg[lo[i, 0]:hi[i, 0]] - X[i, 0], xg[lo[i, 1]:hi[i, 1]] - X[i, 1],
                                 indexing="ij"), -1)
        a, b, c = S[i, 0, 0], S[i, 0, 1], S[i, 1, 1]
        det = a * c - b * b
        q = (c * r[..., 0] ** 2 - 2 * b * r[..., 0] * r[..., 1] + a * r[..., 1] ** 2) / det
        val = (2.0 - 0.5 * q) * np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(det))
        f[lo[i, 0]:hi[i, 0], lo[i, 1]:hi[i, 1]] += np.where(q <= cut * cut, val, 0.0)
    f /= len(X)
    if not clip:
        return f
    return _clip_grid(f, dx, return_clipped)


# ---------------------------------------------------------------------- LSCV

def _k4_conv(d, a2, b2):
    """(K4_a * K4_b)(d), with K4_h(x) = K4(x/h)/h = phi_h - (h^2/2) D^2 phi_h:

        phi_c - ((a^2 + b^2)/2) D^2 phi_c + (a^2 b^2/4) D^4 phi_c,     c^2 = a^2 + b^2,

    D^2 phi_c = phi_c (d^2/c^4 - 1/c^2), D^4 phi_c = phi_c (d^4/c^8 - 6 d^2/c^6 + 3/c^4).
    For a = b = 1 this is phi_{sqrt 2}(d) (27/16 - 7 d^2/16 + d^4/64)."""
    c2 = a2 + b2
    g = np.exp(-0.5 * d * d / c2) / np.sqrt(2 * np.pi * c2)
    r = d * d / c2
    D2 = (r - 1.0) / c2
    D4 = (r * r - 6.0 * r + 3.0) / (c2 * c2)
    return g * (1.0 - 0.5 * c2 * D2 + 0.25 * a2 * b2 * D4)


def _k4_conv_radial(d2, a2, b2, D: int = 2):
    """(K4_a * K4_b)(r) for the radial kernels K4_h = phi_h - (h^2/2) Laplacian phi_h in D
    dimensions (phi_h = N(0, h^2 I)), |r|^2 = d2, a2 = a^2, b2 = b^2, c2 = a2 + b2:

        phi_c - (c2/2) Laplacian phi_c + (a2 b2/4) Laplacian^2 phi_c,
        Laplacian phi_c   = phi_c (d2/c2^2 - D/c2),
        Laplacian^2 phi_c = phi_c (d2^2/c2^4 - 2(D + 2) d2/c2^3 + D(D + 2)/c2^2)."""
    c2 = a2 + b2
    g = np.exp(-0.5 * d2 / c2) / (2 * np.pi * c2) ** (D / 2)
    lap = d2 / c2**2 - D / c2
    bilap = d2 * d2 / c2**4 - 2 * (D + 2) * d2 / c2**3 + D * (D + 2) / c2**2
    return g * (1.0 - 0.5 * c2 * lap + 0.25 * a2 * b2 * bilap)


def lscv4_curve(ys, hs, lam=None) -> np.ndarray:
    """LSCV(h) = int f_h^2 - (2/n) sum_i f_{h,-i}(y_i) of the unclipped K4 estimate in
    closed form (no quadrature); with h_i = h lam_i,

        int f_h^2       = (1/n^2) sum_{i,j} (K4_{h_i} * K4_{h_j})(y_i - y_j)     (_k4_conv),
        f_{h,-i}(y_i)   = (1/(n - 1)) sum_{j != i} K4_{h_j}(y_i - y_j).

    The criterion ignores the clipping of kde4_grid_1d, as usual for higher-order kernels."""
    ys = np.asarray(ys, float).ravel()
    n = ys.size
    lam = np.ones(n) if lam is None else np.asarray(lam, float)
    d = ys[:, None] - ys[None, :]
    off = ~np.eye(n, dtype=bool)
    out = []
    for h in np.atleast_1d(hs):
        hi2 = (h * lam) ** 2
        int_f2 = np.sum(_k4_conv(d, hi2[:, None], hi2[None])) / n**2
        hj = np.sqrt(hi2)[None]
        loo = np.sum((k4(d / hj) / hj)[off]) / (n * (n - 1))
        out.append(int_f2 - 2.0 * loo)
    return np.array(out)


def lscv4_curve_2d(X, hs, kernel: str = "radial") -> np.ndarray:
    """LSCV(h) = int f_h^2 - (2/n) sum_i f_{h,-i}(y_i) of the unclipped 2D K4 estimate, in
    closed form per h in hs (d_ij = y_i - y_j):

        int f_h^2      = (1/n^2) sum_{i,j} (K4_h * K4_h)(d_ij),
        f_{h,-i}(y_i)  = (1/(n - 1)) sum_{j != i} K4_h(d_ij),

    radial: K4_h * K4_h = phi_c - h^2 Laplacian phi_c + (h^4/4) Laplacian^2 phi_c,
    c^2 = 2h^2 (_k4_conv_radial); product: the product over the axes of the 1D
    convolution (_k4_conv). The criterion ignores the clipping, as in 1D."""
    X = np.asarray(X, float)
    n = len(X)
    d = X[:, None] - X[None]
    d2 = np.sum(d * d, -1)
    off = ~np.eye(n, dtype=bool)
    out = []
    for h in np.atleast_1d(hs):
        h2 = h * h
        if kernel == "radial":
            int_f2 = np.sum(_k4_conv_radial(d2, h2, h2)) / n**2
            Kh = k4_2d(d / h) / h2
        else:
            int_f2 = np.sum(_k4_conv(d[..., 0], h2, h2) * _k4_conv(d[..., 1], h2, h2)) / n**2
            Kh = k4_prod(d / h) / h2
        out.append(int_f2 - 2.0 * np.sum(Kh[off]) / (n * (n - 1)))
    return np.array(out)


def lscv4_aniso(X, Sig, block: int = 256) -> float:
    """LSCV of the anisotropic K4 estimate with per-sample covariances Sig (n, 2, 2), in
    closed form. K4_S = (1 - L_S/2) phi_S with L_S = grad^T S grad, so for C = A + B,
    P = C^{-1}, a = P r:

        (K4_A * K4_B)(r) = phi_C [1 - (L_A + L_B)/2 + L_A L_B/4],
        L_A phi_C / phi_C     = a^T A a - tr(A P),
        L_A L_B phi_C / phi_C = (a^T A a)(a^T B a) - (a^T A a) tr(B P) - (a^T B a) tr(A P)
                                - 4 a^T A P B a + tr(A P) tr(B P) + 2 tr(A P B P)

    (the contraction of the fourth Hermite tensor of phi_C); int f^2 sums it over all
    pairs (A = S_i, B = S_j), the leave-one-out term sums K4_{S_j}(y_i - y_j) over j != i."""
    X = np.asarray(X, float)
    S = np.asarray(Sig, float)
    n = len(X)
    int_f2 = 0.0
    loo = 0.0
    for i0 in range(0, n, block):
        r = X[i0:i0 + block, None] - X[None]                    # (b, n, 2)
        A = S[i0:i0 + block, None]                              # (b, 1, 2, 2)
        Bm = S[None]                                            # (1, n, 2, 2)
        C = A + Bm
        det = C[..., 0, 0] * C[..., 1, 1] - C[..., 0, 1] ** 2
        P = np.stack([np.stack([C[..., 1, 1], -C[..., 0, 1]], -1),
                      np.stack([-C[..., 0, 1], C[..., 0, 0]], -1)], -2) / det[..., None, None]
        a = np.einsum("...ij,...j->...i", P, r)
        g = np.exp(-0.5 * np.einsum("...i,...i->...", r, a)) / (2 * np.pi * np.sqrt(det))
        AP = np.einsum("...ij,...jk->...ik", np.broadcast_to(A, C.shape), P)
        BP = np.einsum("...ij,...jk->...ik", np.broadcast_to(Bm, C.shape), P)
        aAa = np.einsum("...i,...ij,...j->...", a, np.broadcast_to(A, C.shape), a)
        aBa = np.einsum("...i,...ij,...j->...", a, np.broadcast_to(Bm, C.shape), a)
        trAP = AP[..., 0, 0] + AP[..., 1, 1]
        trBP = BP[..., 0, 0] + BP[..., 1, 1]
        aAPBa = np.einsum("...i,...ij,...j->...", a, np.einsum("...ij,...jk->...ik", AP,
                                                               np.broadcast_to(Bm, C.shape)), a)
        trAPBP = np.einsum("...ij,...ji->...", AP, BP)
        LA, LB = aAa - trAP, aBa - trBP
        LAB = (aAa * aBa - aAa * trBP - aBa * trAP - 4 * aAPBa + trAP * trBP + 2 * trAPBP)
        int_f2 += float(np.sum(g * (1.0 - 0.5 * (LA + LB) + 0.25 * LAB)))
        Kj = _k4_cov_values(r, np.broadcast_to(Bm, C.shape))
        rows = np.arange(i0, min(i0 + block, n))
        Kj[rows - i0, rows] = 0.0
        loo += float(np.sum(Kj))
    return int_f2 / n**2 - 2.0 * loo / (n * (n - 1))
