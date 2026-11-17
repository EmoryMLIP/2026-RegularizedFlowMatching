"""Sampling: RK4 in tau = -log(1 - t) for the flow and its log density (main.tex
sec:sampling, sec:setup).

With dt/dtau = 1 - t = e^{-tau}, the state (x, log p) obeys

    dx/dtau     =  (1 - t) v(x, t),
    dlog p/dtau = -(1 - t) div_x v(x, t),       t = 1 - e^{-tau},

x(0) the latent points, log p(0) = log K(x(0)), tau in [0, -log(1 - t_end)]. p is the density
of the generated samples that eq:tv needs; the code names it f (``log f``, ``lf``). The
paper's sampler is classical RK4 with config.NSTEP = 96 uniform steps in tau to
t_end = 1 - 1e-4 (``sample``). Stage times come from the grid nodes only, so the reversed
grid (``backward_log_density``) evaluates v at bitwise the forward stage times and a refit
velocity reuses its solves. ``v`` is any object with ``v(x, t)`` and ``v.div(x, t)``:
Velocity, EmpiricalFlow (the exact v_N), PopulationVelocity (v_pi) or a Gabriel field.

``reference`` (adaptive DOP853) and ``reference_rk4`` (RK4 with step doubling) are the
references of the time-integration controls of sec:discretization.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from . import config
from .latent import as_points

__all__ = ["sample", "reference", "sample_path", "reference_rk4", "backward_log_density",
           "FlowSummary", "tau_of_t", "t_of_tau"]


def tau_of_t(t):
    """tau = -log(1 - t)."""
    return -np.log1p(-t)


def t_of_tau(tau):
    """t = 1 - e^{-tau} (the one formula for stage times, so per-t caches hit bitwise)."""
    return 1.0 - np.exp(-tau)


def _grid(t_end, nstep):
    tau_end = tau_of_t(t_end)
    return tau_end / nstep * np.arange(nstep + 1)


@dataclass
class FlowSummary:
    """Run record of one flow: nstep, t_end, nfev (right-hand-side evaluations), fraction of
    final points outside Omega, fraction of trajectories with any stage outside Omega, and
    max |x|_inf over all stages (frame units)."""

    nstep: int
    t_end: float
    nfev: int
    frac_outside_end: float
    frac_outside_any: float
    max_abs: float

    def as_dict(self):
        return dict(self.__dict__)


class _Rhs:
    """Right-hand side (x, log p) -> ((1 - t) v, -(1 - t) div v) in tau, recording the stages
    outside Omega."""

    def __init__(self, v, L, n):
        self.v, self.L = v, L
        self.nfev = 0
        self.any_out = np.zeros(n, bool)
        self.max_abs = 0.0

    def __call__(self, tau, y):
        t = t_of_tau(tau)
        sig = np.exp(-tau)            # 1 - t without the cancellation
        x = y[0]
        self.nfev += 1
        if x.size:
            ax = np.abs(x)
            self.max_abs = max(self.max_abs, float(ax.max()))
            if self.L is not None and self.any_out.size == x.shape[0]:
                self.any_out |= np.any(ax > self.L, axis=1)
        return sig * self.v(x, t), -sig * self.v.div(x, t)


def _rk4_step(f, ta, tb, y):
    h = tb - ta
    tm = 0.5 * (ta + tb)
    k1 = f(ta, y)
    k2 = f(tm, tuple(a + 0.5 * h * b for a, b in zip(y, k1)))
    k3 = f(tm, tuple(a + 0.5 * h * b for a, b in zip(y, k2)))
    k4 = f(tb, tuple(a + h * b for a, b in zip(y, k3)))
    return tuple(a + h / 6 * (p + 2 * q + 2 * r + s) for a, p, q, r, s in zip(y, k1, k2, k3, k4))


def _rk4(f, grid, y):
    for ta, tb in zip(grid[:-1], grid[1:]):
        y = _rk4_step(f, ta, tb, y)
    return y


def stage_times(grid) -> list:
    """The times t of RK4's evaluations over a tau grid, in order, repeats once."""
    out = []
    for ta, tb in zip(grid[:-1], grid[1:]):
        out += [t_of_tau(ta), t_of_tau(0.5 * (ta + tb)), t_of_tau(tb)]
    return list(dict.fromkeys(float(t) for t in out))


def _start(v, latent_points, latent):
    latent = latent if latent is not None else getattr(v, "latent", None)
    dim = latent.dim if latent is not None else getattr(v, "dim", None)
    x0 = np.array(as_points(latent_points, dim if dim else np.shape(latent_points)[-1]), float)
    x0 = x0.reshape(-1, x0.shape[-1])
    lf0 = latent.logpdf(x0) if latent is not None else np.zeros(x0.shape[0])
    return x0, lf0, latent


def _summary(rhs, x, nstep, t_end, L):
    out_end = float(np.mean(np.any(np.abs(x) > L, axis=1))) if L is not None else float("nan")
    out_any = float(np.mean(rhs.any_out)) if L is not None else float("nan")
    return FlowSummary(int(nstep), float(t_end), int(rhs.nfev), out_end, out_any, rhs.max_abs)


def sample(v, latent_points, nstep=config.NSTEP, t_end=config.T_END, latent=None,
           workers: int = 1, pool=None):
    """The paper's sampler: RK4 with nstep uniform steps in tau on [0, tau_end] from the
    latent points (M, D).

    Returns (x (M, D), log p (M,), FlowSummary): the generated points at t_end and the log
    density of their law there, as eq:tv needs. latent: the TruncatedGaussian (default
    v.latent; without one, log p is relative to log p(0) = 0). With workers > 1 or a
    TimePool and a refit Velocity, the solves at all stage times are done first in worker
    processes (same operation sequence, hence the same numbers)."""
    x0, lf0, latent = _start(v, latent_points, latent)
    L = None if latent is None else latent.L
    grid = _grid(t_end, nstep)
    rhs = _Rhs(v, L, x0.shape[0])
    parallel = (pool is not None or workers > 1) and hasattr(v, "prefetch") \
        and getattr(v, "mode", "") == "refit"
    if not parallel:
        x, lf = _rk4(rhs, grid, (x0, lf0))
    else:
        from .velocity import TimePool
        with (TimePool(v, workers) if pool is None else contextlib.nullcontext(pool)) as tp:
            try:
                v.prefetch(stage_times(grid), pool=tp)
                x, lf = _rk4(rhs, grid, (x0, lf0))
            finally:
                v.prefetch([])
    return x, lf, _summary(rhs, x, nstep, t_end, L)


def sample_path(v, latent_points, t_ends, dtau, latent=None):
    """RK4 with the uniform step dtau, the states at each t_end (increasing); each tau_end
    must be a multiple of dtau (1e-9 relative). Returns [(x, log p), ...] (the t_end study
    of sec:discretization)."""
    x0, lf0, latent = _start(v, latent_points, latent)
    kends = []
    for te in t_ends:
        te_tau = tau_of_t(te)
        k = int(round(te_tau / dtau))
        if abs(k * dtau - te_tau) > 1e-9 * te_tau:
            raise ValueError(f"tau_end = {te_tau} is not a multiple of dtau = {dtau}")
        kends.append(k)
    grid = dtau * np.arange(max(kends, default=0) + 1)
    rhs = _Rhs(v, None, 0)
    out, y, k = [], (x0, lf0), 0
    for j in kends:
        while k < j:
            y = _rk4_step(rhs, grid[k], grid[k + 1], y)
            k += 1
        out.append(tuple(np.array(a, copy=True) for a in y))
    return out


def _dop853(f, taus, y, rtol, atol):
    shapes = [np.shape(a) for a in y]
    cuts = np.cumsum([int(np.prod(s)) for s in shapes])[:-1]

    def unflat(u):
        return tuple(p.reshape(s) for p, s in zip(np.split(u, cuts), shapes))

    def g(tau, u):
        return np.concatenate([np.asarray(k, float).ravel() for k in f(tau, unflat(u))])

    u = np.concatenate([np.asarray(a, float).ravel() for a in y])
    out = []
    for ta, tb in zip(taus[:-1], taus[1:]):
        sol = solve_ivp(g, (ta, tb), u, method="DOP853", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        u = sol.y[:, -1]
        out.append((tuple(p.copy() for p in unflat(u)), int(sol.nfev)))
    return out


def reference(v, latent_points, t_end=config.T_END, rtol=1e-10, atol=1e-12, latent=None):
    """Adaptive DOP853 solution of the ODE of ``sample`` (joint error control over all
    points), the reference of the spline flows; for the exact v_N, which jumps, use
    ``reference_rk4``. Returns (x, log p, nfev)."""
    x0, lf0, latent = _start(v, latent_points, latent)
    ((x, lf), nfev), = _dop853(_Rhs(v, None, 0), [0.0, tau_of_t(t_end)], (x0, lf0), rtol, atol)
    return x, lf, nfev


def _absdiff(a, b):
    """(max, mean) of |a - b| over all entries; (0, 0) when empty."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return (float(np.max(d)), float(np.mean(d))) if d.size else (0.0, 0.0)


def reference_rk4(v, latent_points, t_end=config.T_END, n0=config.NSTEP,
                  tol_x=config.REFERENCE_RK4["tol_x"], tol_logf=config.REFERENCE_RK4["tol_logf"],
                  max_doublings=config.REFERENCE_RK4["max_doublings"], latent=None):
    """Reference solution of the ODE of ``sample`` by RK4 with step doubling: n0, 2 n0, 4 n0, ...
    uniform steps in tau (the grid, right-hand side and start of ``sample``), until
    max_i |x_n - x_{n/2}|_inf <= tol_x and mean_i |log f_n - log f_{n/2}| <= tol_logf, or
    after max_doublings doublings (then info['converged'] is False; no exception).

    The exact v_N jumps at the faces of the component boxes, where an adaptive method's error
    control rejects step after step; on a uniform grid a jump only costs O(h |jump|) per
    crossing, so the levels converge at a predictable cost. log f uses the mean over points:
    it bounds the change of a TV estimate, whereas the max sits on < 1% of the points, at
    the faces where div v_N jumps, and converges only to first order (kept as dlogf_max).

    Returns (x, log p, info) of the finest level; info: 'levels' (one dict per level: nstep,
    dx (max), dlogf (mean), dlogf_max against the previous level, None on the first, wall s),
    'nstep', 'converged', 'err_x', 'err_logf' (mean), 'err_logf_max' (the last differences,
    the reported error of the reference), 'nfev'."""
    x0, lf0, latent = _start(v, latent_points, latent)
    rhs = _Rhs(v, None, 0)
    levels, prev, conv = [], None, False
    n = int(n0)
    for j in range(int(max_doublings) + 1):
        t0 = time.perf_counter()
        x, lf = _rk4(rhs, _grid(t_end, n), (x0, lf0))
        dx = dl = dlm = None
        if prev is not None:
            dx = _absdiff(x, prev[0])[0]
            dlm, dl = _absdiff(lf, prev[1])
        levels.append({"nstep": n, "dx": dx, "dlogf": dl, "dlogf_max": dlm,
                       "wall": time.perf_counter() - t0})
        # NaN differences compare False, so a non-finite level never counts as converged
        conv = prev is not None and dx <= tol_x and dl <= tol_logf
        prev = (x, lf)
        if conv:
            break
        n *= 2
    last = levels[-1]
    info = {"levels": levels, "nstep": last["nstep"], "converged": bool(conv),
            "err_x": last["dx"], "err_logf": last["dlogf"], "err_logf_max": last["dlogf_max"],
            "nfev": int(rhs.nfev)}
    return prev[0], prev[1], info


def backward_log_density(v, latent, x, nstep=config.NSTEP, t_end=config.T_END):
    """log p(x) of the law that the forward RK4 flow generates, at arbitrary points x (m, D):
    RK4 backward in tau on the forward grid, at bitwise the forward stage times,

        log p(x) = log K(x(0)) - int_0^{tau_end} (1 - t) div v(x(tau), t(tau)) dtau,

    x(0) the backward image of x (density images; the TV between two flows in the 2D
    stability study)."""
    x = as_points(x, latent.dim).reshape(-1, latent.dim)
    grid = _grid(t_end, nstep)[::-1]
    rhs = _Rhs(v, None, 0)
    # forward rate dlog f/dtau integrated from tau_end down to 0 gives -(integral)
    x0, A = _rk4(rhs, grid, (x, np.zeros(len(x))))
    # A = int_{tau_end}^0 (dlog f/dtau) dtau = log f(x_0, 0) - log f(x, tau_end) relative
    return latent.logpdf(x0) - A
