"""1D helpers shared by oned_eval, hero_1d and gabriel_hero (measurements of a generated
flow, the figure grids, trajectories with a dense output), so that no compute script imports
another. Paper: Table 1 (tab:main), Table 2 (tab:oned-full), Figure 1 (fig:hero)."""
import common as C  # noqa: F401, I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import EmpiricalField, config, tv_pos
from smoothfm.sampler import t_of_tau

EV = config.ONED_EVAL
XF = np.linspace(*EV["xf"])

DXF = XF[1] - XF[0]

XF_EDGES = np.concatenate([XF - 0.5 * DXF, [XF[-1] + 0.5 * DXF]])

HIST_EDGES = np.linspace(*EV["hist_edges"])


# ------------------------------------------------------------------ measurements (oned_eval)

def latents(st, M, s):
    """M quantile midpoints (production latents) stacked on M iid truncated draws."""
    return np.vstack([st.latents(M), st.iid_latents(M, s)])


def flow_entry(st, z, lf):
    """tv_pos (frame), then the curves and the order violations (nonmono) of the outputs mapped
    to the original coordinates; z, lf the outputs of the M quantile-midpoint latents."""
    tv, se = tv_pos(st.std, z, lf)
    x = st.frame.to_phys(z[:, 0])
    f = np.exp(lf - st.frame.log_det)
    o = np.argsort(x, kind="stable")
    xs = x[o]
    u = (np.arange(x.size) + 0.5) / x.size
    # cell average from the generated CDF interpolated between the quantile points (x_k, u_k):
    # resolves the spikes of a memorizing flow, where interpolating f draws false plateaus
    F = np.interp(XF_EDGES, xs, u, left=0.0, right=1.0)
    return dict(tv_pos=tv, tv_pos_se=se,
                curve=np.interp(XF, xs, f[o], left=0.0, right=0.0),
                curve_cell=np.diff(F) / DXF, nonmono=int(np.sum(np.diff(x) < 0)))


def split_entry(st, z, lf, M):
    """flow_entry of the first M outputs plus tv_iid of the iid ones."""
    r = flow_entry(st, z[:M], lf[:M])
    r["tv_iid"], r["tv_iid_se"] = tv_pos(st.std, z[M:], lf[M:])
    return r


def spline_errors(v, times, pop):
    """e(t_k) = mean |v(x, t_k) - v*(x, t_k)|^2 at the oracle points (frame units)."""
    z, y, vpop = pop
    return np.array([np.mean(np.sum((v((1 - t) * z + t * y, t) - vpop[k]) ** 2, axis=1))
                     for k, t in enumerate(times)])


def exact_errors(ys, st, times, pop):
    z, y, vpop = pop
    fld = EmpiricalField(ys, st.latent)
    return np.array([np.mean(np.sum((fld.velocity((1 - t) * z + t * y, t) - vpop[k]) ** 2, 1))
                     for k, t in enumerate(times)])


def exact_extra(fl, law, z, lf):
    """Diagnostics of an exact v_N flow: evaluations with rho_N = 0 (fallback weights) and
    max |log f - log rho_N(x_end, t_end)|, the effect of the omitted delta terms of div v_N."""
    with np.errstate(invalid="ignore"):
        d = np.abs(lf - law.logpdf(z))
    return dict(n_outside=int(fl.n_outside_support), max_dlogf=float(np.nanmax(d)))


def row(d, key, n, s):
    """d[key][index of n, s] or NaN."""
    if d is None or key not in d or n not in [int(x) for x in d["ns"]]:
        return np.nan
    a = np.asarray(d[key])[[int(x) for x in d["ns"]].index(n)]
    return float(a[s]) if s < a.size else np.nan


# ------------------------------------------------------------------ figures (hero_1d, gabriel_hero)

HC = config.HERO_1D

S_END = -np.log1p(-config.T_END)

SUB = HC["sub"]


def image_grid(nt, nx):
    s = (np.arange(nt) + 0.5) * S_END / nt
    lo, hi = HC["x_range"]
    return s, lo + (np.arange(nx) + 0.5) * (hi - lo) / nx


def omega(st, t):
    """The velocity's box Omega in the original coordinates at every t, (lo, hi)."""
    lo, hi = st.frame.to_phys(np.array([-st.L, st.L]))
    t = np.asarray(t, float)
    return np.full(t.shape, lo), np.full(t.shape, hi)


def x_prior(st):
    return st.frame.to_phys(np.linspace(-HC["prior_sd"], HC["prior_sd"], HC["prior_points"]))


def prior_pdf(st):
    return st.latent.pdf(st.frame.to_std(x_prior(st))[:, None]) / st.frame.s


def traj_latents(st, k):
    """k truncated-latent quantile midpoints (frame units), (k, 1)."""
    return st.latent.quantile_midpoints(k)


def tau_grid(nstep):
    return S_END / nstep * np.arange(nstep + 1)


def rk4_with_path(v, x0, nstep, sub=SUB, k0=0):
    """The production RK4 of sampler.sample (x component; the same arithmetic and stage times)
    on x0 (n, 1), started at node k0 of the tau grid, with a cubic Hermite dense output between
    its nodes from the node states and the slopes dx/dtau = (1 - t) v there. Returns
    (tau ((nstep - k0) sub + 1,), x (n, (nstep - k0) sub + 1))."""
    grid = tau_grid(nstep)[k0:]

    def f(tau, x):
        return np.exp(-tau) * v(x, t_of_tau(tau))

    x = np.array(x0, float)
    xs, ws = [x[:, 0]], [f(grid[0], x)[:, 0]]
    for ta, tb in zip(grid[:-1], grid[1:]):
        h, tm = tb - ta, 0.5 * (ta + tb)
        k1 = f(ta, x)
        k2 = f(tm, x + 0.5 * h * k1)
        k3 = f(tm, x + 0.5 * h * k2)
        k4 = f(tb, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x[:, 0])
        ws.append(f(tb, x)[:, 0])
    ds = grid[1] - grid[0]
    th = np.arange(sub) / sub
    h00, h10 = 2 * th**3 - 3 * th**2 + 1, th**3 - 2 * th**2 + th
    h01, h11 = -2 * th**3 + 3 * th**2, th**3 - th**2
    seg = [h00 * xs[k][:, None] + h10 * ds * ws[k][:, None] + h01 * xs[k + 1][:, None]
           + h11 * ds * ws[k + 1][:, None] for k in range(len(grid) - 1)]
    path = np.concatenate(seg + [xs[-1][:, None]], axis=1)
    return grid[0] + np.arange((len(grid) - 1) * sub + 1) * (ds / sub), path


def rk4_path_doubling(v, x0, nstep, k0=0, max_doublings=config.REFERENCE_RK4["max_doublings"],
                      tol_x=config.REFERENCE_RK4["tol_x"]):
    """Reference trajectories of an exact v_N: rk4_with_path on the grid of nstep 2^j steps,
    j = 0, 1, ..., until the path at the output times (the SUB nstep cells of the base grid
    from its node k0) changes by <= tol_x (frame units, max over points and times) between
    levels, or after max_doublings doublings. Up to 2^j = SUB the dense output is Hermite with
    SUB / 2^j points per step; above, every (2^j / SUB)-th RK4 node. The differences include the
    Hermite interpolation, so the converged path is the plotted one.
    Returns (tau, x (n, len(tau)), info) with info as sampler.reference_rk4 (dlogf None)."""
    levels, prev, conv, nfev = [], None, False, 0
    for j in range(int(max_doublings) + 1):
        f = 2 ** j
        if f <= SUB and SUB % f:
            raise ValueError(f"SUB = {SUB} is not a multiple of 2^{j}")
        t0 = time.perf_counter()
        tau, path = rk4_with_path(v, x0, nstep * f, sub=max(SUB // f, 1), k0=k0 * f)
        stride = max(f // SUB, 1)
        tau, path = tau[::stride], path[:, ::stride]
        nfev += 4 * (nstep - k0) * f
        dx = None if prev is None else float(np.max(np.abs(path - prev)))
        levels.append({"nstep": nstep * f, "dx": dx, "dlogf": None, "dlogf_max": None,
                       "wall": time.perf_counter() - t0})
        conv = prev is not None and dx <= tol_x
        prev = path
        if conv:
            break
    info = {"levels": levels, "nstep": levels[-1]["nstep"], "converged": bool(conv),
            "err_x": levels[-1]["dx"], "err_logf": None, "err_logf_max": None, "nfev": nfev}
    return tau, prev, info


def zoom_k0(nstep):
    """Node of the production grid at ZOOM_T0 (exact for NSTEP divisible by 4)."""
    return int(round(-np.log1p(-HC["zoom_t0"]) / (S_END / nstep)))


def zoom_window(ys):
    """(lo, hi) around ZOOM_PTS consecutive sorted points (original coordinates): among the
    runs of span <= ZOOM_SPAN the one with the largest smallest gap, extended to the
    midpoints to its outer neighbours (half its smallest gap at the ends of the data)."""
    y = np.sort(np.asarray(ys, float).ravel())
    g = np.diff(y)
    npts = HC["zoom_pts"]
    best, ib = -1.0, None
    for i in range(y.size - npts + 1):
        if y[i + npts - 1] - y[i] <= HC["zoom_span"] and g[i:i + npts - 1].min() > best:
            best, ib = g[i:i + npts - 1].min(), i
    if ib is None:
        raise SystemExit(f"no {npts} consecutive training points within {HC['zoom_span']}; "
                         "pass --zoom LO HI")
    j = ib + npts - 1
    lo = 0.5 * (y[ib - 1] + y[ib]) if ib > 0 else y[ib] - 0.5 * best
    hi = 0.5 * (y[j] + y[j + 1]) if j + 1 < y.size else y[j] + 0.5 * best
    return float(lo), float(hi)


def zoom_seeds(window, k=HC["zoom_k"]):
    lo, hi = window
    return lo + (np.arange(k) + 0.5) * (hi - lo) / k


def choose_set(oned, d, n, requested=-1):
    """The requested set, or (requested < 0) the median hold-out tv_pos set of oned.npz."""
    if requested >= 0:
        return requested
    tv = oned[f"{d}/holdout/tv_pos"][[int(x) for x in oned["ns"]].index(n)]
    idx = np.argsort(tv, kind="stable")
    return int(idx[(idx.size - 1) // 2])
