"""The regularized velocity v(., t) on [0, t_end] (main.tex sec:cv, last paragraphs;
sec:sampling).

Problem eq:regfm decouples across time. At the time points t_0 < ... < t_T (config.TIMES), a
selection rule picks lam(t_k) (selection.py, eq:oracle or hold-out), and the normal
equations eq:normaleq are solved again with all samples. Stored are the coefficients of
w = (1 - t) v = dx/dtau, which stays O(1) while v grows like 1/(1 - t) at late t.

Evaluation at any other t (``mode="refit"``, the paper's rule) solves the normal equations
at t with log lam interpolated linearly in tau = -log(1 - t), and with lam(t_0) before t_0
and lam(t_T) after t_T. A stored lam = 0 is the floor 1e-14 lam_s (sec:setup); if it sits
at either end of the bracket, the solve at t uses the floor 1e-14 lam_s(t). At t = t_k the
stored lam is used as is, so the refit reproduces the stored coefficients bitwise.
``mode="interpolate"`` interpolates the coefficients linearly in tau instead, for
compare_time, which measures the difference that sec:cv quotes (up to 0.005 in TV).

Parallel work over t (``workers`` > 1 or a ``TimePool``): the selection at each t_k and the
refit solves at given times are independent, so worker processes do them, each holding a
copy of the problem (sent once) and a fresh solver, with the serial operation sequence.
"""

from __future__ import annotations

import copy
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .latent import as_points
from .normal_equations import NormalEquations
from .regularizers import H2, get_regularizer, mass_matrix
from .selection import LambdaSearch
from .solvers import default_solver
from .splines import far_clip

__all__ = ["Velocity"]


class Velocity:
    """The regularized velocity v(., t), the minimizer of F_t in eq:regfm in the spline
    space, for training points in the standardized frame.

    basis: Cubic1D / Cubic2D with half-width latent.L; latent: TruncatedGaussian;
    samples (N, D) inside Omega (A3); times t_0 < ... < t_T in [0, 1); reg: a Regularizer
    or its key (default the Hessian regularizer H2). Sampler interface: ``v(x, t)`` (n, D)
    and ``v.div(x, t)`` (n,), both in frame units.

    After ``fit``, arrays over the T + 1 time points: ``lam`` (absolute), ``lam_s`` (of each
    rule's NormalEquations, the fitting part for hold-out), ``selections`` (Selection),
    ``nsolves`` (factorizations), ``wall`` (selection wall time, s), ``at_limit``,
    ``residual`` (relative residual of the selected all-sample solve), ``backward_error`` (its
    normwise backward error; both solver-accuracy checks, logged, never reported). After
    set_lam / set_floor, ``lam_s`` is that of all samples.
    """

    refit_cache_size = 8

    def __init__(self, basis, latent, samples, times, reg=None, solver=None, mode="refit"):
        self.latent = latent
        self.dim = latent.dim
        self.samples = as_points(samples, self.dim).reshape(-1, self.dim)
        self.times = np.asarray(times, float)
        if np.any(np.diff(self.times) <= 0) or self.times[0] < 0 or self.times[-1] >= 1:
            raise ValueError("times must increase within [0, 1)")
        self.tau = -np.log1p(-self.times)
        out = np.any(np.abs(self.samples) > latent.L, axis=1)
        if out.any():
            raise ValueError(f"{int(out.sum())} training samples lie outside Omega = [-L, L]^D")
        self.basis = basis
        self.reg = H2() if reg is None else get_regularizer(reg)
        self.P = self.reg.matrix(basis)
        self.M = mass_matrix(basis)
        self.solver = solver or default_solver()
        if mode not in ("refit", "interpolate"):
            raise ValueError("mode must be 'refit' or 'interpolate'")
        self.mode = mode
        self.coef = self.lam = self.lam_s = None
        self.selections = self.nsolves = self.wall = self.at_limit = self.residual = None
        self.backward_error = None
        self.floor = False
        self._refit, self._pre = {}, {}

    def with_mode(self, mode):
        """A shallow copy sharing the fit with another evaluation rule."""
        out = copy.copy(self)
        if mode not in ("refit", "interpolate"):
            raise ValueError("mode must be 'refit' or 'interpolate'")
        out.mode = mode
        out._refit, out._pre = {}, {}
        return out

    def normal_equations(self, t, samples=None) -> NormalEquations:
        """NormalEquations at t (default all samples), sharing P and M."""
        x = self.samples if samples is None else samples
        return NormalEquations(self.basis, self.latent, x, t, self.reg, P=self.P, M=self.M)

    # -- fitting -----------------------------------------------------------------------
    def set_lam(self, lam) -> "Velocity":
        """Fix lam(t_k) (array over the time points or scalar) without selection and solve
        eq:normaleq with all samples."""
        self.lam = np.broadcast_to(np.asarray(lam, float), self.times.shape).copy()
        self.floor = False
        self.selections = None
        coef, scale = [], []
        for k, t in enumerate(self.times):
            ne = self.normal_equations(t)
            scale.append(ne.lam_s)
            coef.append((1.0 - t) * ne.solve(self.lam[k], self.solver))
        self.coef, self.lam_s = np.stack(coef), np.array(scale)
        self._refit, self._pre = {}, {}
        return self

    def set_floor(self) -> "Velocity":
        """The limit lam -> 0 (sec:discretization): solve at the floor
        lam_min(t) = 1e-14 lam_s(t) at every t_k and, in refit mode, at every evaluation time.
        ``lam`` records lam_min(t_k)."""
        coef, lam, scale = [], np.empty(self.times.size), np.empty(self.times.size)
        for k, t in enumerate(self.times):
            ne = self.normal_equations(t)
            lam[k], scale[k] = ne.lam_min, ne.lam_s
            coef.append((1.0 - t) * ne.solve(0.0, self.solver))
        self.lam, self.lam_s, self.coef = lam, scale, np.stack(coef)
        self.selections = None
        self.floor = True
        self._refit, self._pre = {}, {}
        return self

    def fit(self, rule_factory, search: LambdaSearch, workers: int = 1, pool=None) -> "Velocity":
        """Select lam(t_k) at every time point by rule_factory(velocity, t) -> SelectionRule
        and the LambdaSearch, then solve again with all samples ("With the selected lam, we
        solve again with all samples", sec:cv), reusing the rule's solution when its normal
        equations already hold all samples (oracle). rule_factory and search must be
        picklable for workers > 1."""
        self.floor = False
        K = self.times.size
        if pool is None and workers <= 1:
            results = [self._fit_one(k, rule_factory, search) for k in range(K)]
        else:
            with (_Borrowed(pool) if pool is not None else TimePool(self, workers)) as tp:
                results = tp.map(_fit_task, [(k, rule_factory, search) for k in range(K)])
        self.selections = [r[0] for r in results]
        self.coef = np.stack([r[1] for r in results])
        self.wall = np.array([r[2] for r in results])
        self.residual = np.array([r[3] for r in results])
        self.backward_error = np.array([r[4] for r in results])
        self.lam = np.array([s.lam for s in self.selections])
        self.lam_s = np.array([s.lam_s for s in self.selections])
        self.nsolves = np.array([s.nsolves for s in self.selections])
        self.at_limit = np.array([s.at_limit for s in self.selections])
        self._refit, self._pre = {}, {}
        return self

    def _fit_one(self, k, rule_factory, search):
        t0 = time.time()
        t = self.times[k]
        rule = rule_factory(self, t)
        sel = search(rule)
        ne = rule.ne
        if ne.samples.shape == self.samples.shape and np.array_equal(ne.samples, self.samples):
            C = rule.solution(sel.lam)
        else:
            ne = self.normal_equations(t)
            C = ne.solve(np.float64(sel.lam), self.solver)
        # relative residual and backward error of the selected all-sample solve (run logs);
        # the logs report the backward error, which rounding cannot inflate
        res, berr = ne.residual(sel.lam, C), ne.backward_error(sel.lam, C)
        return sel, (1.0 - t) * C, time.time() - t0, res, berr

    @property
    def rel_lam(self):
        """lam(t_k) / lam_s(t_k), the quantity the figures report (sec:setup)."""
        return self.lam / self.lam_s

    # -- evaluation --------------------------------------------------------------------
    def lam_at(self, t) -> float:
        """lam(t): lam(t_k) at t = t_k; log lam linear in tau = -log(1 - t) between the time
        points (sec:cv); 0 (the floor) if either end of the bracket is 0 or the velocity is at
        the floor; constant beyond t_0 and t_T."""
        if self.floor:
            return 0.0
        t = float(t)
        hit = np.flatnonzero(self.times == t)
        if hit.size:
            return float(self.lam[hit[0]])
        tau = -np.log1p(-t)
        if tau <= self.tau[0]:
            return float(self.lam[0])
        if tau >= self.tau[-1]:
            return float(self.lam[-1])
        k = int(np.searchsorted(self.tau, tau, side="right")) - 1
        a, b = self.lam[k], self.lam[k + 1]
        if a <= 0 or b <= 0:
            return 0.0
        th = (tau - self.tau[k]) / (self.tau[k + 1] - self.tau[k])
        return float(np.exp((1 - th) * np.log(a) + th * np.log(b)))

    def _refit_solve(self, t):
        return (1.0 - t) * self.normal_equations(t).solve(self.lam_at(t), self.solver)

    def w_coefficients(self, t):
        """Coefficients (size, D) of w(., t) = (1 - t) v(., t): stored at t_k, solved at other t
        (refit) or interpolated (interpolate); the last few refits are cached per t."""
        if self.coef is None:
            raise RuntimeError("fit the velocity first (fit, set_lam or set_floor)")
        t = float(t)
        if self.mode == "interpolate":
            tau = -np.log1p(-t)
            if tau <= self.tau[0]:
                return self.coef[0]
            if tau >= self.tau[-1]:
                return self.coef[-1]
            k = int(np.searchsorted(self.tau, tau, side="right")) - 1
            th = (tau - self.tau[k]) / (self.tau[k + 1] - self.tau[k])
            return (1 - th) * self.coef[k] + th * self.coef[k + 1]
        if t in self._pre:
            return self._pre[t]
        if t not in self._refit:
            if len(self._refit) >= self.refit_cache_size:
                self._refit.pop(next(iter(self._refit)))
            self._refit[t] = self._refit_solve(t)
        return self._refit[t]

    def prefetch(self, ts, workers: int = 1, pool=None) -> "Velocity":
        """Refit mode: solve at the times ts in worker processes and hold the coefficients
        until the next prefetch (prefetch([]) releases them)."""
        if self.mode != "refit":
            return self
        ts = list(dict.fromkeys(float(t) for t in ts))
        held = {t: self._pre[t] for t in ts if t in self._pre}
        todo = [t for t in ts if t not in held]
        if todo:
            with (_Borrowed(pool) if pool is not None else TimePool(self, workers)) as tp:
                held.update(zip(todo, tp.map(_refit_task, [(t, self.lam, self.floor) for t in todo])))
        self._pre = held
        return self

    def w(self, x, t):
        """w(x, t) = (1 - t) v(x, t) at points x (n, D), with the C^2 extension of the spline
        outside Omega (splines.far_clip), shape (n, D)."""
        x = as_points(x, self.dim).reshape(-1, self.dim)
        W = self.w_coefficients(t)
        xc, _ = far_clip(self.basis, x)
        deriv = 0 if self.dim == 1 else (0, 0)
        return np.asarray(self.basis.extended_design(xc, deriv) @ W).reshape(x.shape[0], -1)

    def div_w(self, x, t):
        """div_x w (n,): the exact divergence of the extended spline (no contribution from an
        axis clipped by far_clip)."""
        x = as_points(x, self.dim).reshape(-1, self.dim)
        W = self.w_coefficients(t)
        xc, clipped = far_clip(self.basis, x)
        if self.dim == 1:
            g = np.asarray(self.basis.extended_design(xc, 1) @ W[:, 0]).ravel()
            return np.where(clipped[:, 0], 0.0, g)
        gx = np.asarray(self.basis.extended_design(xc, (1, 0)) @ W[:, 0]).ravel()
        gy = np.asarray(self.basis.extended_design(xc, (0, 1)) @ W[:, 1]).ravel()
        return np.where(clipped[:, 0], 0.0, gx) + np.where(clipped[:, 1], 0.0, gy)

    def __call__(self, x, t):
        return self.w(x, t) / (1.0 - t)

    def div(self, x, t):
        return self.div_w(x, t) / (1.0 - t)

    def _problem_copy(self):
        out = copy.copy(self)
        out.solver = self.solver.fresh()
        out.coef = out.selections = None
        out._refit, out._pre = {}, {}
        return out


# -- worker processes ---------------------------------------------------------------------
_WORKER = {}


def _worker_init(problem):
    _WORKER["v"] = problem


def _fit_task(args):
    k, rule_factory, search = args
    return _WORKER["v"]._fit_one(k, rule_factory, search)


def _refit_task(args):
    t, lam, floor = args
    v = _WORKER["v"]
    v.lam, v.floor = lam, floor
    return v._refit_solve(t)


class TimePool:
    """Worker processes holding a copy of one velocity's problem (samples, basis, P, M,
    latent, times); a context manager. A dying worker raises BrokenProcessPool."""

    def __init__(self, velocity: Velocity, workers: int, context=None):
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.workers = int(workers)
        self._ex = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context(context),
                                       initializer=_worker_init, initargs=(velocity._problem_copy(),))

    def map(self, fn, args):
        if self._ex is None:
            raise RuntimeError("this TimePool is closed")
        futs = [self._ex.submit(fn, a) for a in args]
        try:
            return [f.result() for f in futs]
        except BaseException:
            self.terminate()
            raise

    def terminate(self):
        ex, self._ex = self._ex, None
        if ex is not None:
            procs = list((getattr(ex, "_processes", None) or {}).values())
            ex.shutdown(wait=False, cancel_futures=True)
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=10)

    def close(self):
        ex, self._ex = self._ex, None
        if ex is not None:
            ex.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close() if exc_type is None else self.terminate()
        return False


class _Borrowed:
    def __init__(self, pool):
        self.pool = pool

    def __enter__(self):
        return self.pool

    def __exit__(self, *exc):
        return False
