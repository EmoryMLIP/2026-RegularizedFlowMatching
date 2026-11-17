"""Bandwidth selection for the Gaussian KDEs of kde.py.

Rule of thumb (Silverman), Sheather-Jones solve-the-equation (as R's bw.SJ),
least-squares cross-validation curves, Abramson's per-sample factors, and
LSCV for per-sample covariances (anisotropic kernels).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import cdist, pdist, squareform

from .kde import _gauss_d2


def _sd(ys):
    return min(ys.std(ddof=1), np.subtract(*np.percentile(ys, [75, 25])) / 1.349)


def silverman(ys) -> float:
    """Silverman's rule of thumb sigma = 0.9 q N^{-1/5}, q = min(sd, IQR/1.349) (S:sec:kde;
    run_kde.py)."""
    ys = np.asarray(ys, float).ravel()
    return 0.9 * _sd(ys) * ys.size ** (-0.2)


def sheather_jones(ys) -> float:
    """Sheather-Jones solve-the-equation bandwidth (as R's bw.SJ; run_kde.py)."""
    ys = np.asarray(ys, float).ravel()
    n = ys.size
    d = pdist(ys[:, None])
    scale = _sd(ys)
    a, b = 1.24 * scale * n ** (-1 / 7), 1.23 * scale * n ** (-1 / 9)

    def phi4(g):
        x = d / g
        s = 2 * np.sum((x**4 - 6 * x**2 + 3) * np.exp(-x**2 / 2)) + 3 * n
        return s / (n * (n - 1) * g**5 * np.sqrt(2 * np.pi))

    def phi6(g):
        x = d / g
        s = 2 * np.sum((x**6 - 15 * x**4 + 45 * x**2 - 15) * np.exp(-x**2 / 2)) - 15 * n
        return s / (n * (n - 1) * g**7 * np.sqrt(2 * np.pi))

    alph2 = 1.357 * (phi4(a) / -phi6(b)) ** (1 / 7)
    c1 = 1 / (2 * np.sqrt(np.pi) * n)
    fsd = lambda h: (c1 / phi4(alph2 * h ** (5 / 7))) ** 0.2 - h
    hmax = 1.144 * scale * n ** (-0.2)
    lo, hi = 0.1 * hmax, hmax
    while fsd(lo) < 0:
        lo /= 2
    while fsd(hi) > 0:
        hi *= 2
    return brentq(fsd, lo, hi, xtol=1e-6 * hmax)


# ---------------------------------------------------------- cross-validation

def lscv_curve(X, hs, lam=None) -> np.ndarray:
    """The LSCV criterion of S:sec:kde, int hat pi^2 - (2/N) sum_i hat pi_{-i}(x_i), for the
    bandwidths sigma_i = h0 lam_i (h0 the paper's sigma, lam_i the Abramson factors, 1 if lam
    is None), per h0 in hs:

        int f^2      = (1/N^2) sum_{i,j} N(x_i - x_j; 0, (h_i^2 + h_j^2) I),
        f_{-i}(x_i)  = (1/(N - 1)) sum_{j != i} N(x_i - x_j; 0, h_j^2 I).

    lam = None is the fixed bandwidth (run_kde.py lscv, run_inhom.py lscv_scores,
    run_kde2d.py lscv, p4 run_baselines.py lscv)."""
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    n, D = X.shape
    lam = np.ones(n) if lam is None else np.asarray(lam, float)
    d2 = squareform(pdist(X, "sqeuclidean"))
    off = ~np.eye(n, dtype=bool)
    out = []
    for h0 in np.atleast_1d(hs):
        hi2 = (h0 * lam) ** 2
        int_f2 = np.sum(_gauss_d2(d2, hi2[:, None] + hi2[None], D)) / n**2
        loo = np.sum(_gauss_d2(d2, np.broadcast_to(hi2[None], d2.shape), D)[off]) / (n * (n - 1))
        out.append(int_f2 - 2 * loo)
    return np.array(out)


# ----------------------------------------------------------------- Abramson

def abramson_factors(X, pilot_h, dense_max: int = 20_000) -> np.ndarray:
    """Abramson factors lam_i = sigma_i / sigma = (f_pilot(x_i) / g)^{-1/2} of S:sec:kde,
    f_pilot (the paper's tilde pi) the fixed KDE with bandwidth pilot_h
    (self term included), g the geometric mean of f_pilot(x_i). 1D pilot: Sheather-Jones
    (oned.py, kde4.py); 2D pilot: the fixed-KDE LSCV bandwidth (kde2d.py).

    Above dense_max points the pilot density is summed in blocks of whole rows (about
    2^25 distances each): the dense n x n matrices need 24 n^2 bytes, 240 GB at n = 1e5.
    The blocks hold the same squared distances (cdist and pdist share the kernel) and each
    row mean runs over the full row, so both paths agree bit for bit."""
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    n, D = X.shape
    if n <= dense_max:
        d2 = squareform(pdist(X, "sqeuclidean"))
        fp = _gauss_d2(d2, pilot_h**2, D).mean(1)
    else:
        step = max(1, 2**25 // n)
        fp = np.concatenate([_gauss_d2(cdist(X[i0:i0 + step], X, "sqeuclidean"), pilot_h**2, D).mean(1)
                             for i0 in range(0, n, step)])
    return (fp / np.exp(np.mean(np.log(fp)))) ** -0.5


# -------------------------------------------------------- anisotropic LSCV

def _gauss_cov2(d, S):
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    det = a * c - b * b
    q = (c * d[..., 0] ** 2 - 2 * b * d[..., 0] * d[..., 1] + a * d[..., 1] ** 2) / det
    return np.exp(-q / 2) / (2 * np.pi * np.sqrt(det))


def aniso_lscv(X, Sig, block: int = 256) -> float:
    """LSCV for per-sample covariances: int f^2 = (1/N^2) sum N(x_i - x_j; 0, S_i + S_j),
    loo = (1/(N(N - 1))) sum_{i != j} N(x_i - x_j; 0, S_j) (run_akde2d.py)."""
    n = len(X)
    d = X[:, None] - X[None]
    off = ~np.eye(n, dtype=bool)
    int_f2 = sum(_gauss_cov2(d[i0:i0 + block], Sig[i0:i0 + block, None] + Sig[None]).sum()
                 for i0 in range(0, n, block)) / n**2
    loo = sum((_gauss_cov2(d[i0:i0 + block],
                           np.broadcast_to(Sig[None], (min(block, n - i0), n, 2, 2)))
               * off[i0:i0 + block]).sum() for i0 in range(0, n, block)) / (n * (n - 1))
    return float(int_f2 - 2 * loo)
