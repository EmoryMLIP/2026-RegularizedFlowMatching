"""Gaussian kernel density estimators (S:sec:kde).

f(x) = (1/N) sum_i N(x; x_i, Sigma_i); with Sigma_i = sigma^2 I it is the KDE
hat pi_sigma = (1/N) sum_i G_sigma(. - x_i) of S:sec:kde, with Sigma_i = sigma_i^2 I the
Abramson KDE, and with Sigma_i = sigma_T^2 e_i e_i^T + sigma_N^2 n_i n_i^T the anisotropic
KDE. The code writes h for sigma (module docstring of baselines/).

The module provides the estimator itself (``KDE``, a law with
exact logpdf and sampling), its values on uniform grids by binning and FFT convolution,
and the per-sample covariances of the anisotropic variant. Bandwidth selection is in
bandwidth.py.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fftfreq, irfft2, rfft2, rfftfreq
from scipy.signal import fftconvolve
from scipy.special import logsumexp, ndtr

from ..latent import as_points as _as_points

_LOG2PI = np.log(2.0 * np.pi)


def _phi(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)


def _gauss_d2(d2, var, D):
    """N(r; 0, var I) as a function of |r|^2 = d2, in D dimensions."""
    return np.exp(-d2 / (2 * var)) / (2 * np.pi * var) ** (D / 2)


def _bin_2d(Y, xg, weight):
    """Bilinear binning of the points Y onto the grid xg x xg, density units (each point
    carries mass ``weight``)."""
    dx = xg[1] - xg[0]
    s = (Y - xg[0]) / dx
    k = np.floor(s).astype(int)
    a = s - k
    c = np.zeros((xg.size, xg.size))
    w = np.full(len(Y), weight)
    for ddx, wx in [(0, 1 - a[:, 0]), (1, a[:, 0])]:
        for ddy, wy in [(0, 1 - a[:, 1]), (1, a[:, 1])]:
            np.add.at(c, (k[:, 0] + ddx, k[:, 1] + ddy), w * wx * wy)
    return c / dx**2


# ------------------------------------------------------------------ estimator

class KDE:
    """Gaussian KDE f(x) = (1/N) sum_i N(x; x_i, Sigma_i).

    Kernel covariances: ``bandwidth`` h (scalar, the paper's sigma) or h_i (array (N,), the
    Abramson sigma_i) gives Sigma_i = h_i^2 I;
    ``cov`` (D, D) is one shared matrix, (N, D, D) one per sample.
    ``sample(n, rng)`` draws the component index first (rng.integers(0, N, n)), then the
    standard normal (rng.standard_normal((n, D))); this order fixes the KDE draws.
    """

    def __init__(self, samples, bandwidth=None, cov=None, block: int = 2048):
        samples = np.asarray(samples, float)
        if samples.ndim == 1:
            samples = samples[:, None]
        self.dim = samples.shape[1]
        self.samples = samples
        N, D = samples.shape
        if (bandwidth is None) == (cov is None):
            raise ValueError("give exactly one of bandwidth and cov")
        if bandwidth is not None:
            h = np.broadcast_to(np.asarray(bandwidth, float), (N,))
            self.cov = h[:, None, None] ** 2 * np.eye(D)
        else:
            c = np.asarray(cov, float)
            c = np.atleast_2d(c) if c.ndim < 2 else c
            self.cov = np.broadcast_to(c, (N, D, D)).copy()
        self.chol = np.linalg.cholesky(self.cov)
        self.logdet = 2.0 * np.sum(np.log(np.diagonal(self.chol, axis1=1, axis2=2)), axis=1)
        self.block = int(block)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    @property
    def bandwidth(self):
        """Per-sample isotropic h_i (sqrt of the mean kernel variance per axis)."""
        return np.sqrt(np.trace(self.cov, axis1=1, axis2=2) / self.dim)

    def logpdf(self, x):
        """log f(x) = logsumexp_i [log N(x; x_i, Sigma_i)] - log N."""
        x = _as_points(x, self.dim)
        shp = x.shape[:-1]
        x = x.reshape(-1, self.dim)
        out = np.empty(x.shape[0])
        N, D = self.samples.shape
        # 2D lower-triangular solves in closed form keep this vectorized
        L = self.chol
        step = max(1, (self.block * 1024) // N)      # about 2e6 kernel evaluations per block
        for s0 in range(0, x.shape[0], step):
            r = x[s0:s0 + step][:, None, :] - self.samples[None]
            if D == 1:
                q = (r[..., 0] / L[None, :, 0, 0]) ** 2
            elif D == 2:
                u0 = r[..., 0] / L[None, :, 0, 0]
                u1 = (r[..., 1] - L[None, :, 1, 0] * u0) / L[None, :, 1, 1]
                q = u0**2 + u1**2
            else:
                u = np.linalg.solve(L[None], r[..., None])[..., 0]
                q = np.sum(u**2, -1)
            a = -0.5 * (q + self.logdet[None] + D * _LOG2PI)
            out[s0:s0 + a.shape[0]] = logsumexp(a, axis=1) - np.log(N)
        return out.reshape(shp)

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        k = rng.integers(0, self.samples.shape[0], int(n))
        e = rng.standard_normal((int(n), self.dim))
        return self.samples[k] + np.einsum("nij,nj->ni", self.chol[k], e)

    def cdf(self, x):
        """1D only: F(x) = (1/N) sum_i Phi((x - x_i) / h_i)."""
        if self.dim != 1:
            raise NotImplementedError("cdf is defined for D = 1")
        x = np.asarray(x, float)
        shp = x.shape
        x = x.ravel()
        y, h = self.samples[:, 0], self.chol[:, 0, 0]
        out = np.empty(x.size)
        step = max(1, (self.block * 1024) // y.size)
        for s0 in range(0, x.size, step):
            out[s0:s0 + step] = ndtr((x[s0:s0 + step, None] - y[None]) / h[None]).mean(1)
        return out.reshape(shp)

    def ppf(self, q, newton: int = 4):
        """1D only: the quantile function F^{-1}(q), q in (0, 1).

        Start from linear interpolation of F on 4097 grid points over
        [min x_i - 9 max h, max x_i + 9 max h], then ``newton`` Newton steps
        x <- x - (F(x) - q) / f(x) with the exact F and f (F is smooth and strictly
        increasing and the start lies within one grid cell, so the steps converge
        quadratically).
        """
        if self.dim != 1:
            raise NotImplementedError("ppf is defined for D = 1")
        q = np.asarray(q, float)
        shp = q.shape
        q = q.ravel()
        y, h = self.samples[:, 0], self.chol[:, 0, 0]
        xg = np.linspace(y.min() - 9 * h.max(), y.max() + 9 * h.max(), 4097)
        x = np.interp(q, self.cdf(xg), xg)
        for _ in range(newton):
            f = np.exp(self.logpdf(x[:, None]))
            x = x - (self.cdf(x) - q) / np.maximum(f, 1e-300)
        return x.reshape(shp)


# ----------------------------------------------------------------- FFT grids

def kde_grid_1d(ys, h, xg, lam=None, log_bin: float = 0.01, kernel=None) -> np.ndarray:
    """KDE values on the uniform grid xg by linear binning and FFT convolution (run_kde.py).

    lam (per-sample factors, Abramson): samples grouped into log-bandwidth bins of width
    log_bin, each group convolved with its own kernel (run_inhom.py abramson_grid).
    kernel: the standardized kernel K(u), truncated at |u| = 8 (default: the Gaussian).
    Binning error O(dx^2 / h^2); the grid must contain the kernels' mass.
    """
    ys = np.asarray(ys, float).ravel()
    dx = xg[1] - xg[0]

    def binned(y, wsum):
        s = (y - xg[0]) / dx
        k = np.floor(s).astype(int)
        a = s - k
        c = np.bincount(k, 1 - a, xg.size) + np.bincount(k + 1, a, xg.size)
        return c * wsum / (y.size * dx)

    def conv(c, hh):
        r = np.arange(-int(np.ceil(8 * hh / dx)), int(np.ceil(8 * hh / dx)) + 1) * dx
        if kernel is None:
            kr = np.exp(-0.5 * (r / hh) ** 2) / (np.sqrt(2 * np.pi) * hh) * dx
        else:
            kr = kernel(r / hh) / hh * dx
        return fftconvolve(c, kr, mode="same")

    if lam is None:
        return conv(binned(ys, 1.0), h)
    key = np.round(np.log(lam) / log_bin).astype(int)
    f = np.zeros(xg.size)
    for k in np.unique(key):
        sel = key == k
        f += conv(binned(ys[sel], sel.mean()), h * np.exp(log_bin * k))
    return f


def kde_grid_2d(X, h, xg, lam=None, nfft: int = 1024, log_bin: float = 0.05) -> np.ndarray:
    """2D KDE on the square grid xg x xg by bilinear binning and an FFT Gaussian filter
    (run_kde2d.py; Abramson factors grouped in log-bandwidth bins of width log_bin)."""
    X = np.asarray(X, float)
    dx = xg[1] - xg[0]
    K2 = fftfreq(nfft, dx)[:, None] ** 2 + rfftfreq(nfft, dx)[None] ** 2
    n = len(X)

    if lam is None:
        groups = [(1.0, X)]
    else:
        key = np.round(np.log(lam) / log_bin).astype(int)
        groups = [(np.exp(log_bin * k), X[key == k]) for k in np.unique(key)]
    tot = sum(rfft2(_bin_2d(Y, xg, 1.0 / n), s=(nfft, nfft)) * np.exp(-2 * np.pi**2 * (h * r) ** 2 * K2)
              for r, Y in groups)
    return irfft2(tot, s=(nfft, nfft))[:xg.size, :xg.size]


# ------------------------------------------------------- anisotropic kernels

def aniso_covs(e, h_t, h_n) -> np.ndarray:
    """Sigma_i = h_T^2 e_i e_i^T + h_N^2 n_i n_i^T (the paper's sigma_T, sigma_N), with e_i
    the unit tangent and n_i = e_i rotated by 90 degrees (S:sec:kde)."""
    nrm = np.stack([-e[:, 1], e[:, 0]], 1)
    return (h_t**2 * e[:, :, None] * e[:, None, :]
            + h_n**2 * nrm[:, :, None] * nrm[:, None, :])
