"""Densities interpolated on uniform grids (the clipped fourth-order KDEs; S:sec:kde):
exact sampling and the exact log density of the draws, as tv_pos needs.
Paper: the fourth-order KDE rows of Tables 2 and 3, via experiments/compute/kde4.py and kde4_2d.py."""

from __future__ import annotations

import numpy as np

from ..latent import as_points

__all__ = ["GridDensity1D", "GridDensity2D"]


class GridDensity1D:
    """The density that interpolates nonnegative values f_k linearly on the uniform grid
    x_0 < ... < x_K (0 outside), normalized so that int f = 1 (trapezoid rule, exact for
    the interpolant). Its CDF is piecewise quadratic,

        F(x_k + s) = F_k + f_k s + (f_{k+1} - f_k) s^2 / (2 dx),   0 <= s <= dx,

    and ``ppf`` inverts it exactly (root of the quadratic), so ``sample`` draws from f
    itself and ``logpdf`` is the exact log density of the draws, as tv_pos needs."""

    def __init__(self, xg, f):
        self.dim = 1
        self.xg = np.asarray(xg, float)
        f = np.maximum(np.asarray(f, float), 0.0)
        self.dx = self.xg[1] - self.xg[0]
        cell = 0.5 * (f[:-1] + f[1:]) * self.dx
        Z = cell.sum()
        self.f = f / Z
        self.F = np.concatenate([[0.0], np.cumsum(cell / Z)])

    def pdf(self, x):
        x = as_points(x, 1)[..., 0]
        return np.interp(x, self.xg, self.f, left=0.0, right=0.0)

    def logpdf(self, x):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x))

    def cdf(self, x):
        x = np.asarray(x, float)
        k = np.clip(np.searchsorted(self.xg, x, "right") - 1, 0, self.xg.size - 2)
        s = np.clip(x - self.xg[k], 0.0, self.dx)
        fk, fk1 = self.f[k], self.f[k + 1]
        F = self.F[k] + fk * s + (fk1 - fk) * s * s / (2 * self.dx)
        return np.where(x < self.xg[0], 0.0, np.where(x > self.xg[-1], 1.0, F))

    def ppf(self, q):
        q = np.asarray(q, float)
        k = np.clip(np.searchsorted(self.F, q, "right") - 1, 0, self.xg.size - 2)
        fk, fk1 = self.f[k], self.f[k + 1]
        r = np.maximum(q - self.F[k], 0.0)                   # mass to cover within cell k
        a = (fk1 - fk) / (2 * self.dx)
        # s solves a s^2 + fk s = r; 2r / (fk + sqrt(fk^2 + 4 a r)) is stable as a -> 0
        # and for fk = 0
        den = fk + np.sqrt(np.maximum(fk * fk + 4 * a * r, 0.0))
        s = np.where(den > 0, 2 * r / np.where(den > 0, den, 1.0), 0.0)
        return self.xg[k] + np.clip(s, 0.0, self.dx)

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        return self.ppf(rng.uniform(size=int(n)))[:, None]


class GridDensity2D:
    """The density that interpolates nonnegative values f[i, j] at (xg_i, xg_j) bilinearly
    on the square grid xg x xg (0 outside), normalized so that int f = 1 (the cell mass of
    the interpolant is dx^2 times the mean of its four corner values, exactly). ``sample``
    draws from the interpolant exactly: a cell with probability proportional to its mass,
    then the first coordinate from the cell's marginal (linear in s) and the second from
    the conditional (linear in u), each by inverting a quadratic CDF; ``logpdf`` is the
    exact log density of the draws, as tv_pos needs."""

    def __init__(self, xg, f):
        self.dim = 2
        self.xg = np.asarray(xg, float)
        self.dx = self.xg[1] - self.xg[0]
        f = np.maximum(np.asarray(f, float), 0.0)
        cell = 0.25 * (f[:-1, :-1] + f[1:, :-1] + f[:-1, 1:] + f[1:, 1:]) * self.dx**2
        Z = cell.sum()
        self.f = f / Z
        self._cum = np.cumsum(cell.ravel() / Z)

    def pdf(self, x):
        x = as_points(x, 2)
        shp = x.shape[:-1]
        x = x.reshape(-1, 2)
        m = self.xg.size
        s = (x - self.xg[0]) / self.dx
        k = np.clip(np.floor(s).astype(int), 0, m - 2)
        a = s - k
        f = self.f
        val = ((1 - a[:, 0]) * (1 - a[:, 1]) * f[k[:, 0], k[:, 1]]
               + a[:, 0] * (1 - a[:, 1]) * f[k[:, 0] + 1, k[:, 1]]
               + (1 - a[:, 0]) * a[:, 1] * f[k[:, 0], k[:, 1] + 1]
               + a[:, 0] * a[:, 1] * f[k[:, 0] + 1, k[:, 1] + 1])
        inside = np.all((s >= 0) & (s <= m - 1), axis=1)
        return np.where(inside, val, 0.0).reshape(shp)

    def logpdf(self, x):
        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x))

    @staticmethod
    def _inv_linear(p, q, u):
        """s in [0, 1] with int_0^s ((1 - v) p + v q) dv = u (p + q)/2: the root
        2r / (p + sqrt(p^2 + 2 (q - p) r)), r = u (p + q)/2, stable for q -> p and p = 0."""
        r = u * 0.5 * (p + q)
        den = p + np.sqrt(np.maximum(p * p + 2 * (q - p) * r, 0.0))
        s = np.where(den > 0, 2 * r / np.where(den > 0, den, 1.0), u)
        return np.clip(s, 0.0, 1.0)

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        U = rng.uniform(size=(int(n), 3))
        m = self.xg.size - 1
        c = np.minimum(np.searchsorted(self._cum, U[:, 0] * self._cum[-1], "right"), m * m - 1)
        i, j = c // m, c % m
        f = self.f
        f00, f10, f01, f11 = f[i, j], f[i + 1, j], f[i, j + 1], f[i + 1, j + 1]
        s = self._inv_linear(f00 + f01, f10 + f11, U[:, 1])
        u = self._inv_linear(f00 * (1 - s) + f10 * s, f01 * (1 - s) + f11 * s, U[:, 2])
        return np.stack([self.xg[i] + s * self.dx, self.xg[j] + u * self.dx], 1)
