"""The standardized frame x = c + s x' of the data (main.tex sec:setup; S:sec:splinedetails,
"Domains, Frames and Latents").

A fixed box that contains the training data is mapped onto Omega = [-L, L]^D by
y |-> (y - c)/s, c its centre and s > 0 the scalar that maps its largest half-width to L, so
the mapped training points satisfy (A3) and the velocity lives on one fixed box at every t.
All flow computations happen in this frame; in original coordinates this is flow matching
from the latent N(c, s^2 I) conditioned on c + s Omega. The weight mu and lam_s are frame
quantities (config.MU).

- ``AffineFrame(c, s)``: the map x' = (x - c)/s and its inverse x = c + s x'.
- ``AffineTarget(base, frame, latent)``: the law of y' = (y - c)/s for y ~ base, a Target
  whose posterior quantities come from the base target's by an exact time change.
"""

from __future__ import annotations

import numpy as np

from .latent import as_points
from .targets import Target

__all__ = ["AffineFrame", "AffineTarget"]


class AffineFrame:
    """The frame x' = (x - c)/s, c in R^D, s > 0 one scalar for every axis, so that the latent
    stays isotropic in original coordinates."""

    def __init__(self, c, s):
        self.c = np.atleast_1d(np.asarray(c, float)).ravel()
        self.s = float(s)
        if not self.s > 0 or not np.isfinite(self.s):
            raise ValueError("s must be a finite scalar > 0")
        self.dim = self.c.size
        # D = 1: a scalar centre, so that (n,) and scalar inputs keep their shape
        self._cb = self.c[0] if self.dim == 1 else self.c

    def __repr__(self):
        return f"AffineFrame(c={self.c.tolist()}, s={self.s:.6g})"

    @classmethod
    def from_box(cls, box, L) -> "AffineFrame":
        """The frame mapping box (D, 2) into [-L, L]^D: c the midpoint, s = max half-width / L
        (the largest half-width maps to L)."""
        b = np.asarray(box, float).reshape(-1, 2)
        if np.any(b[:, 1] <= b[:, 0]):
            raise ValueError("box intervals must have lo < hi")
        return cls(0.5 * (b[:, 0] + b[:, 1]), 0.5 * np.max(b[:, 1] - b[:, 0]) / float(L))

    @property
    def log_det(self) -> float:
        """D log s: log p'(x') = log p(c + s x') + log_det."""
        return self.dim * np.log(self.s)

    def to_std(self, x):
        """x' = (x - c) / s (same shape as x)."""
        return (np.asarray(x, float) - self._cb) / self.s

    def to_phys(self, xs):
        """x = c + s x' (same shape as xs)."""
        return self._cb + self.s * np.asarray(xs, float)

    def training_set(self, y, L, target, n, seed):
        """y' = to_std(y) for a training set (or a second data set), checked against (A3), that
        the training points lie in Omega: ValueError naming target, n, seed and max |y'_d| if
        any point leaves [-L, L]^D."""
        ys = self.to_std(as_points(y, self.dim).reshape(-1, self.dim))
        worst = float(np.max(np.abs(ys))) if ys.size else 0.0
        if worst > float(L):
            raise ValueError(f"(A3) violated: {target} n={n} seed={seed}: max |y'_d| = {worst:.6g} "
                             f"> L = {float(L):g} (max |y_d| = "
                             f"{float(np.max(np.abs(np.asarray(y, float)))):.6g} in original units)")
        return ys


class AffineTarget(Target):
    """The law of y' = (y - c) / s for y ~ base: p'(x') = s^D p(c + s x').

    Time change (exact for the truncated latent, because z is unchanged): in original
    coordinates s x'_t + t c = s (1 - t) z + t y = a [(1 - tau) z + tau y] with
    a = s + t (1 - s), tau = t / a, 1 - tau = s (1 - t) / a. So the event x'_t = x' is the base
    event x_tau = (s x' + t c) / a, whose posterior of y is the base posterior at (x_tau, tau)
    with the same L, and
        m' = (m - c) / s,  C' = C / s^2,  p'_t(x') = (s / a)^D p_tau(x_tau),
        tr grad' m' = tr grad m / a.
    ``latent`` (optional) is carried for PopulationVelocity.
    """

    def __init__(self, base: Target, frame: AffineFrame, latent=None):
        if frame.dim != base.dim:
            raise ValueError(f"frame dim {frame.dim} != target dim {base.dim}")
        super().__init__(base.dim)
        self.base, self.frame, self.latent = base, frame, latent
        if self.dim == 1 and hasattr(base, "cdf") and hasattr(base, "ppf"):
            self.cdf = lambda x: self.base.cdf(self.frame.to_phys(x))
            self.ppf = lambda u: self.frame.to_std(self.base.ppf(u))

    def __repr__(self):
        return f"AffineTarget({type(self.base).__name__}, {self.frame!r})"

    def sample(self, n, rng=None):
        return self.frame.to_std(self.base.sample(n, rng))

    def logpdf(self, x):
        return self.base.logpdf(self.frame.to_phys(x)) + self.frame.log_det

    @property
    def mean(self):
        return (np.asarray(self.base.mean, float) - self.frame.c) / self.frame.s

    @property
    def cov(self):
        return np.asarray(self.base.cov, float) / self.frame.s**2

    def time_change(self, x, t):
        """(x_tau, tau, a): the base event with the same posterior of y (class docstring).
        a = s + t (1 - s), so s = 1 gives a = 1 and tau = t without rounding."""
        s, c = self.frame.s, self.frame.c
        a = s + t * (1.0 - s)
        return (s * x + t * c) / a, t / a, a

    def posterior(self, x, t, L):
        """(m', C', log p'_t) in the frame from the base posterior at (x_tau, tau)."""
        x = as_points(x, self.dim).reshape(-1, self.dim)
        s, c = self.frame.s, self.frame.c
        xt, tau, a = self.time_change(x, float(t))
        m, C, logp = self.base.posterior(xt, tau, L)
        return (m - c) / s, C / s**2, logp + self.dim * np.log(s / a)

    def jacobian_trace(self, x, t, L):
        """tr grad' m'(x', t) = tr grad m(x_tau, tau) / a."""
        x = as_points(x, self.dim).reshape(-1, self.dim)
        xt, tau, a = self.time_change(x, float(t))
        return self.base.jacobian_trace(xt, tau, L) / a
