"""Stability of the regularized flow under a perturbation of the training points (paper
Theorem thm:stability, "Stability in the data"), and the instability of the exact empirical field v_N
and of the smallness (k = 0) regularizer. Paper: Figure 3 (fig:stability), Table 8
(tab:stability), Table 9 (tab:stability-refine); Sec. 5.5 "Stability with Respect to the
Training Data".

Perturbation (original coordinates): x_i' = x_i + epsilon xi_i, xi_i uniform on the unit
sphere (1D: a random sign), so max_i |x_i - x_i'| = epsilon; xi_i -> -xi_i for a point that
would leave the data box config.DATA_BOX. epsilon in config.EPSILON (data units; epsilon / s
stored as epsilon_frame), STABILITY['draws'] draws of xi per (target, N, set), rng
default_rng([SEEDS['stability'][0], target index, N, set, draw]), one xi for every epsilon.
Both training sets are mapped by the production frame (common.Setting), so the two flows live
on one spline grid; m = M_SPLINE (1D: `--m M`), the 49 selection times, refit mode, RK4 NSTEP
steps to T_END, the truncated latent.
Runs: `--dim 1` (trimodal, inhom, N = 100) -> stability_1d; `--dim 1 --m M` (the refinement
rows, epsilon in STABILITY['refine_epsilon']) -> stability_1d_m<M>; `--target circle`
(N = 1000) -> stability_circle; `--target circle --reg l2` -> stability_circle_l2 (the
smallness regularizer on the circle, a separate result so that stability_circle stays as it
was). `--grid-dx DX` (2D only) replaces the spacing of
the TV grid STABILITY['grid_2d'] (half-width kept; smoke runs: of the smoke grid); the result
name gets the suffix _dx<DX in thousandths, 3 digits> when DX differs from the production
spacing (stability_circle_l2_dx005 for 0.005), a grid-convergence check of the 2D TV.
Flows: h2_fixed (hold-out lam(t) of X: oned / twod_circle), h2_resel (hold-out on X', production
grid and split), 1D also l2_fixed and l2_resel (oned_l2), and exact (v_N). stability_circle_l2:
l2_fixed and l2_resel only (source twod_circle_l2, candidates config.LAM_DECADES['l2'][2], the
same positive-candidate rule), no exact jobs (the exact v_N is in stability_circle; the same
perturbations, xi_draw does not depend on the flow), so its npz has no <d>/exact/ keys and
'flows' lists l2_fixed, l2_resel.
Positive candidates: every hold-out selection of this script (re-selected flows on X and X', and
the fixed lam(t)) minimizes over lam = 10^j lam_s(t) only, lam = 0 excluded (LambdaSearch
include_zero=False; thm:stability assumes lam bounded below, and the production floor 1e-14 lam_s
makes the H^2 field constant grow with m). The fixed lam(t) is the stored production lam(t) where
it is positive (the positive minimizer then equals it: the positive candidates are a subset that
contains it) and, at the t_k where it is 0, the hold-out minimizer over the positive candidates on
X at the production m, split and times (PositiveHoldOut.complete). The smallest evaluated
candidate is 10^(lo - 1/2) lam_s (lowest decade lo, then the half-decade refinement): for B_2
10^-8.5 lam_s in 1D and 10^-9.5 lam_s in 2D. For B_0 (1D) the rule is a no-op: its lowest
candidate 10^-14 lam_s equals the solve floor LAM_FLOOR_REL lam_s used for lam = 0, so a
replaced zero gives the same solve (its refinement neighbour 10^-14.5 lam_s is clamped to the
floor as well).
At --m != M_SPLINE[1] the fixed flows keep the production-m lam(t); the re-selected base TV is
checked against controls_1d[_l2] '<d>/m<m>/holdout/tv_pos' when that result exists (optional).
Measured per (target, N, set, epsilon, draw, flow) against the base flow on X:
  tv          TV(p_epsilon, p_0) of the generated laws (1D: linear interpolation of the
              (x_end, f) pairs of the M_LAT[1] quantile-midpoint outputs on nx_1d points,
              trapezoid rule; 2D: backward-RK4 log densities (sampler.backward_log_density) on
              the grid [-half, half]^2 spacing dx of STABILITY['grid_2d'], metrics.tv_grid);
              tv_se = 0 (deterministic quadrature). exact: TV between the stopped laws
              rho_N(., T_END) of X and X' (StoppedLaw.tv_to: 1D exact, 2D Monte Carlo of m_tv2
              draws with SE) and the bound tv_bound = (1/N) sum_i (2 Phi(t |dx_i| / 2 sigma) - 1)
              + 2 (1 - Z_L^D) (the paired bound of the untruncated components plus
              the truncation, by the triangle inequality still an upper bound);
  tv_target   tv_pos against the target with its SE (2D also tv_iid);
  dnorm2      (K, 3): ||d||^2_L2, |d|^2_H1, |d|^2_H2 of d = v_{t_k} - v'_{t_k} (frame units) from
              the spline coefficients and basis.energy(0, 1, 2);
  l2diff2     exact: E_{rho_N(., t)} |v_N - v_N'|^2 at 1 - t in STABILITY['tl'] (Monte Carlo of
              m_l2 draws x = t x_i + (1 - t) z, z truncated; SE stored);
  lam, mass, nflip, wall;
  lam_changed the number of t_k > 0 at which the selected candidate j (lam = 10^j lam_s(t),
              j in half decades with the refinement, lam_s of the flow's own selection) differs
              from the base flow's (candidate_index). Not a float comparison of lam: lam_s moves
              with the data, so lam changes at almost every t_k while j does not. Fixed flows: 0.
Checks (log): the base flows reproduce the stored hold-out lam(t) at every t_k > 0 where the stored
value is positive, and tv_pos (2D also tv_iid) for every (target, N, set) (hard checks): from the
base flow where no zero was replaced, and from an extra flow with the stored lam(t) (its zeros at
the floor, as in production) where one was; a check that covers no set prints n/a. Each replaced
zero is listed (t_k, stored 0, new lam and lam/lam_s of its selection, whether it is the lowest
candidate); at the production m the fixed and re-selected base lam(t) agree at t_k > 0 (both are
the positive hold-out minimizer on X). At another m the re-selected reference is the controls
hold-out flow, also selected with lam = 0 allowed: its zero times count as replaced.
Smoke: N = 100, set 0, epsilon indices STABILITY['smoke']['epsilon_idx'] (refinement: the first
two of refine_epsilon), one draw, smoke m and times, the coarse 2D grid, the smoke sources.

npz keys (n = len(ns), S sets, E epsilons, R draws, K times, J = len(tl)):
  epsilon (E), epsilon_frame (E), times (K), tl (J), sets (S), draws, flows, t_end, nstep, m,
  frame_c, frame_s, frame_L, grid_2d (2D: half, dx);
  <d>/ns (n), <d>/nflip (n, S, E, R);
  <d>/<flow>/{tv, tv_target, tv_target_se, mass, lam_changed, wall (n, S, E, R), lam (.., K),
    dnorm2 (.., K, 3), base_tv_target, base_tv_target_se, base_tv_ref, base_wall (n, S),
    base_lam, base_lam_ref (n, S, K), base_lam_replaced, base_lam_at_lower (n, S, K) bool,
    base_tv_stored, base_tv_stored_se (n, S; NaN unless a zero was replaced), lam_at_lower
    (n, S, E, R; t_k > 0 of a re-selected flow at the lowest candidate, fixed flows 0),
    base_norm2 (n, S, K, 3)} and 2D {tv_iid, tv_iid_se, base_tv_iid, base_tv_iid_se, base_tv_iid_ref, base_tv_iid_stored,
    base_mass};
  <d>/exact/{tv, tv_se, tv_bound, wall (n, S, E, R), l2diff2, l2diff2_se (n, S, E, R, J),
    base_l2norm2, base_l2norm2_se (n, S, J)}.
The re-selected flows search positive candidates only; lam_changed counts candidate-index
changes.
base_lam_replaced: fixed flows, the t_k whose stored 0 was replaced; re-selected flows, the t_k
whose reference lam(t) is 0 (production at the production m, controls at another m; the selection
differs there by construction); False without a reference. base_lam_at_lower: the replaced
(fixed) or selected (re-selected) t_k whose lam is the smallest effective candidate,
max(10^(lo - 1/2), LAM_FLOOR_REL) lam_s (the hold-out score still decreases toward lam = 0
there; for l2 the floor itself).
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import os
import time

import numpy as np
from scipy.special import ndtr

from smoothfm import (EmpiricalField, HoldOut, StoppedLaw, backward_log_density,
                      config, tv_grid, tv_pos)
from smoothfm.sampler import sample

ABOUT = """{name}: stability of {targets} under x_i' = x_i + epsilon xi_i, epsilon in {epsilon}
(data units); flows {flows}{exact}, m = {m}, 1 - t checks at {tl}.
Hold-out over the positive candidates 10^j lam_s(t) only (lam = 0 excluded; {op} smallest
effective {lo}); the fixed lam(t) is the stored one with its zeros replaced by that selection
on X. For B_0 the rule is a no-op: its lowest candidate 10^-14 lam_s equals the solve floor.
Base jobs: the flow on X; perturbation jobs: the flow on X', TV to the base (quadrature, se 0),
tv_pos with SE, H^k field differences; exact jobs: stopped-law TV, bound, E|v_N - v_N'|^2.
Outputs: {out}.npz (report: fig_stability, tab_stability{refine}), .log, .manifest.json."""

ST = config.STABILITY
TARGETS = ("trimodal", "inhom", "circle")
TOL_REF, TOL_RESEL = ST["tol_ref"], ST["tol_resel"]
OPERATOR_ORDER = {"h2": 2, "h1": 1, "l2": 0}     # derivative order k of each regularizer


# ------------------------------------------------------------------ perturbation

def xi_draw(target, n, s, draw, dim):
    """Unit directions (n, D): random signs in 1D, normalized standard normals otherwise."""
    rng = np.random.default_rng([config.SEEDS["stability"][0], TARGETS.index(target), n, s, draw])
    if dim == 1:
        return rng.choice([-1.0, 1.0], size=(n, 1))
    g = rng.standard_normal((n, dim))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def perturb(y, epsilon, xi, box):
    """y + epsilon xi, with -xi for the points that would leave the box; (y', flips)."""
    box = np.asarray(box, float)
    lo, hi = box[:, 0], box[:, 1]
    yp = y + epsilon * xi
    out = np.any((yp < lo) | (yp > hi), axis=1)
    yp[out] = y[out] - epsilon * xi[out]
    if np.any((yp < lo) | (yp > hi)):
        raise ValueError(f"epsilon = {epsilon}: a point leaves the data box in both directions")
    return yp, int(out.sum())


# ------------------------------------------------------------------ flows and measurements

class Run:
    """Per-run settings shipped with each job: protocol, m, 2D grid."""

    def __init__(self, P, m, grid):
        self.P, self.m, self.grid = P, m, grid

    def selection(self, reg, dim) -> "PositiveHoldOut":
        return PositiveHoldOut(reg, dim)


def tv_grid_2d(smoke, dx=None) -> tuple:
    """(half, dx) of the 2D TV grid: STABILITY['grid_2d'] (smoke: its smoke grid), the spacing
    replaced by dx when given (`--grid-dx`)."""
    grid = tuple(ST["smoke"]["grid_2d"] if smoke else ST["grid_2d"])
    return grid if dx is None else (grid[0], float(dx))


def dx_suffix(dx) -> str:
    """'_dx<code>' (code = dx in thousandths, 3 digits) when dx differs from the production
    spacing STABILITY['grid_2d'][1], else ''. dx must be a positive multiple of 0.001 below 1,
    so that the code names it uniquely."""
    if dx is None or dx == ST["grid_2d"][1]:
        return ""
    code = round(dx * 1000)
    if not (0 < code < 1000 and abs(dx * 1000 - code) < 1e-9):
        raise ValueError(f"grid spacing {dx} is not a positive multiple of 0.001 below 1")
    return f"_dx{code:03d}"


def grid_points(grid):
    half, dx = grid
    xg = np.arange(-half, half + dx / 2, dx)
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()]), float(dx)


class PositiveHoldOut:
    """The hold-out selection of this script: the production split and decades of
    config.LAM_DECADES[reg], lam = 0 excluded. For l2 the exclusion changes nothing: its lowest candidate 10^-14 lam_s is the
    solve floor of lam = 0. thm:stability bounds the field difference by a
    constant that grows as lam -> 0, so a flow whose lam(t) contains the floor 1e-14 lam_s is
    outside the theorem (inhom N = 100 set 1: the H^2 constant grew 30x from m = 256 to 1024)."""

    def __init__(self, reg, dim):
        self.reg = reg
        self.search = C.search(reg, dim, include_zero=False)
        # the smallest lam the search evaluates: the lower half-decade neighbour of the lowest
        # decade; the solve clamps lam to the floor, so nothing below LAM_FLOOR_REL lam_s is a
        # different fit (l2: 10^-14.5 solves as 10^-14)
        self.lowest_rel = max(config.LAM_FLOOR_REL, float(self.search.relative.min()) * 10**-0.5)

    def at_lower(self, lam, lam_s) -> np.ndarray:
        """lam(t_k) at the smallest evaluated candidate (lam_s of the selection): the hold-out
        score still decreases toward lam = 0 there, so the positive minimizer is an edge value."""
        return candidate_index(lam, lam_s) <= candidate_index(self.lowest_rel, 1.0)

    def fit(self, st, ys, s, times, m):
        """Velocity on frame points ys fitted by hold-out over the positive candidates."""
        yt, yv = st.split(ys, s)

        def rule(v, t, yt=yt, yv=yv):
            ne = v.normal_equations(t, yt)
            return HoldOut(ne, ne.validation(yv))

        return st.velocity(ys, times, m, self.reg).fit(rule, self.search)

    def complete(self, lam, st, ys, s, times, m):
        """(lam (K,), replaced (K,) bool, lam_s (K,)): the stored lam(t) with each 0 replaced by
        the positive minimizer at that t_k, and the lam_s of that selection (NaN elsewhere), so
        that lam / lam_s lies on the candidate grid. Only the zero times are re-selected: where
        the stored value is positive it is already the positive minimizer (the positive
        candidates are a subset of the production ones containing it), and each t_k is
        selected independently."""
        lam = np.array(lam, float)
        replaced = lam == 0
        lam_s = np.full(lam.shape, np.nan)
        if replaced.any():
            v = self.fit(st, ys, s, np.asarray(times)[replaced], m)
            lam[replaced], lam_s[replaced] = v.lam, v.lam_s
        return lam, replaced, lam_s


def candidate_index(lam, lam_s) -> np.ndarray:
    """j with lam = 10^j lam_s, rounded to half decades (the refinement steps); -inf at lam = 0."""
    with np.errstate(divide="ignore"):
        return np.round(2.0 * np.log10(np.asarray(lam, float) / np.asarray(lam_s, float))) / 2.0


def candidates_text(reg, dim) -> str:
    """The positive candidate grid of PositiveHoldOut(reg, dim), with its smallest evaluated
    value."""
    lo, hi = config.LAM_DECADES[reg][dim]
    low = max(lo - 0.5, float(np.log10(config.LAM_FLOOR_REL)))
    return (f"lam = 10^[{lo}..{hi - 1}] lam_s(t) ({hi - lo}) + half-decade refinement, "
            f"smallest effective 10^{low:g} lam_s" + (" (the solve floor)" if low > lo - 0.5
                                                       else ""))


def lam_changed(lam, lam_s, base_lam, base_lam_s) -> int:
    """Number of t_k > 0 whose candidate index differs from the base's. t_0 = 0 is excluded as
    in the reproduction check: v_N(., 0) is affine, the score is flat in lam to rounding."""
    j, j0 = candidate_index(lam, lam_s), candidate_index(base_lam, base_lam_s)
    return int(np.sum(j[1:] != j0[1:]))


def build(st, ys, s, flow, lam_fixed, R):
    """The spline flow '<reg>_<fixed|resel>' on frame points ys (lam_fixed: the completed
    fixed lam(t), PositiveHoldOut.complete)."""
    reg, mode = flow.split("_")
    if mode == "fixed":
        return st.velocity(ys, R.P.times, R.m, reg).set_lam(lam_fixed)
    return R.selection(reg, st.dim).fit(st, ys, s, R.P.times, R.m)


def measure(st, v, s, R):
    """Generation and the base quantities of one flow (frame points)."""
    P = R.P
    if st.dim == 1:
        x0 = st.latents(P.m_lat)
        z, lf, summ = sample(v, x0, nstep=P.nstep, t_end=config.T_END)
        tv, se = tv_pos(st.std, z, lf)
        return dict(z=z[:, 0], lf=lf, tv_target=tv, tv_target_se=se, summary=summ)
    M = P.m_lat
    x0 = np.vstack([st.latents(M), st.iid_latents(M, s)])
    v.refit_cache_size = 4 * P.nstep + 8              # forward and backward share solves
    z, lf, summ = sample(v, x0, nstep=P.nstep, t_end=config.T_END)
    tv, se = tv_pos(st.std, z[:M], lf[:M])
    ti, si = tv_pos(st.std, z[M:], lf[M:])
    pts, dx = grid_points(R.grid)
    lp = backward_log_density(v, st.latent, st.frame.to_std(pts), nstep=P.nstep,
                              t_end=config.T_END) - st.frame.log_det
    return dict(tv_target=tv, tv_target_se=se, tv_iid=ti, tv_iid_se=si, lp=lp,
                mass=float(np.sum(np.exp(lp)) * dx * dx), summary=summ)


def field_parts(basis, coef_a, coef_b, times):
    """(K, 3): ||d||^2_L2, |d|^2_H1, |d|^2_H2 of d = v_a - v_b at each t_k, summed over the
    components; v = W / (1 - t) from the coefficients W (K, size, D) of w = (1 - t) v."""
    G = [basis.energy(j) for j in (0, 1, 2)]
    out = np.empty((len(times), 3))
    for k, t in enumerate(times):
        d = (coef_a[k] - (0.0 if coef_b is None else coef_b[k])) / (1.0 - t)
        out[k] = [float(np.sum(d * (Gj @ d))) for Gj in G]
    return out


def interp_density(z, lf, xg):
    o = np.argsort(z, kind="stable")
    return np.interp(xg, z[o], np.exp(lf[o]), left=0.0, right=0.0)


def tv_1d(a, b, nx):
    """(TV(p_a, p_b), mass of p_b): trapezoid rule on nx points over both ranges."""
    xg = np.linspace(min(a["z"].min(), b["z"].min()), max(a["z"].max(), b["z"].max()), nx)
    pa, pb = interp_density(a["z"], a["lf"], xg), interp_density(b["z"], b["lf"], xg)
    return 0.5 * float(np.trapezoid(np.abs(pa - pb), xg)), float(np.trapezoid(pb, xg))


def exact_tv(st, ys, yp, n, s, ie, draw):
    """(tv, tv_se, tv_bound) between the stopped laws rho_N(., T_END) of X and X' (frame)."""
    t, sig = config.T_END, 1.0 - config.T_END
    dist = np.linalg.norm(t * (ys - yp), axis=1)
    bound = float(np.mean(2.0 * ndtr(dist / (2.0 * sig)) - 1.0)) + 2.0 * st.latent.tv_to_gaussian
    A, B = StoppedLaw(ys, t, st.latent), StoppedLaw(yp, t, st.latent)
    rng = np.random.default_rng([config.SEEDS["stability"][2], n, s, ie, draw])
    tv, se = A.tv_to(B, n=ST["m_tv2"], rng=rng)
    return tv, se, bound


def exact_l2(st, ys, yp, n, s):
    """(E|v_N - v_N'|^2, SE) under rho_N(., t_j) of X at 1 - t_j in tl (yp None: E|v_N|^2)."""
    Fa = EmpiricalField(ys, st.latent)
    Fb = None if yp is None else EmpiricalField(yp, st.latent)
    tl = np.asarray(ST["tl"], float)
    m, se = np.empty(tl.size), np.empty(tl.size)
    for j, sl in enumerate(tl):
        rng = np.random.default_rng([config.SEEDS["stability"][1], n, s, j])
        idx = rng.integers(0, ys.shape[0], ST["m_l2"])
        x = (1.0 - sl) * ys[idx] + sl * st.latent.from_normal(
            rng.standard_normal((ST["m_l2"], st.dim)))
        d = Fa.velocity(x, 1.0 - sl) - (0.0 if Fb is None else Fb.velocity(x, 1.0 - sl))
        q = np.sum(d * d, axis=1)
        m[j], se[j] = q.mean(), q.std(ddof=1) / np.sqrt(q.size)
    return m, se


# ------------------------------------------------------------------ jobs

def base_task(job):
    """('base', target, n, s, flow, lam_stored, R[, lam_ref]). Fixed flows complete lam_stored
    (production lam(t)) on X at the production m (R.P.m). lam_ref is the lam(t) of the stored
    flow whose tv_pos is the reference (default: lam_stored at the production m, none
    elsewhere; at another m the controls hold-out lam(t) of a re-selected flow).
    'lam_replaced' marks the t_k whose stored lam is 0: fixed flows, the replaced values;
    re-selected flows, the zeros of lam_ref. 'lam_at_lower' marks the replaced (fixed) or
    selected (re-selected) t_k at the smallest evaluated candidate. When a zero was replaced
    and lam_ref exists, the stored flow is measured too (set_lam(lam_ref); a stored 0 solves
    at the floor as in production): 'stored_tv_target' etc., so that the tv reproduction check
    covers every set."""
    _, target, n, s, flow, lam_stored, R = job[:7]
    lam_ref = job[7] if len(job) > 7 else (lam_stored if R.m == R.P.m else None)
    t0 = time.time()
    st = C.setting(target)
    _, ys = st.training_set(n, s)
    reg, mode = flow.split("_")
    sel = R.selection(reg, st.dim)
    lam_stored = np.asarray(lam_stored, float)
    if mode == "fixed":
        lam_fixed, replaced, lam_s_sel = sel.complete(lam_stored, st, ys, s, R.P.times, R.P.m)
        at_lower = replaced & sel.at_lower(lam_fixed, lam_s_sel)
    else:
        lam_fixed = None
        replaced = (np.zeros(lam_stored.shape, bool) if lam_ref is None
                    else np.asarray(lam_ref, float) == 0)
    v = build(st, ys, s, flow, lam_fixed, R)
    t1 = time.time()
    r = measure(st, v, s, R)
    rec = C.velocity_record(v, t1 - t0)
    if mode != "fixed":
        lam_s_sel = np.where(replaced, rec["lam_s"], np.nan)
        at_lower = sel.at_lower(rec["lam"], rec["lam_s"])
    r.update(rec, coef=v.coef, norm2=field_parts(v.basis, v.coef, None, v.times),
             lam_replaced=replaced, lam_s_sel=lam_s_sel, lam_at_lower=at_lower,
             wall_gen=time.time() - t1)
    if replaced.any() and lam_ref is not None:
        vs = st.velocity(ys, R.P.times, R.m, reg).set_lam(np.asarray(lam_ref, float))
        rs = measure(st, vs, s, R)
        r["stored_tv_target"], r["stored_tv_target_se"] = rs["tv_target"], rs["tv_target_se"]
        if st.dim == 2:
            r["stored_tv_iid"] = rs["tv_iid"]
    r["wall"] = time.time() - t0
    return r


def pert_task(job):
    """('pert', target, n, s, ie, draw, flow, lam_fixed, base, epsilon, R); lam_fixed is the
    base flow's completed lam(t)."""
    _, target, n, s, ie, draw, flow, lam_fixed, base, epsilon, R = job
    t0 = time.time()
    st = C.setting(target)
    y, _ = st.training_set(n, s)
    yp, nflip = perturb(y, epsilon, xi_draw(target, n, s, draw, st.dim), st.box)
    ysp = st.frame.to_std(yp)
    v = build(st, ysp, s, flow, lam_fixed, R)
    t1 = time.time()
    r = measure(st, v, s, R)
    # a fixed lam(t) has no candidate index of its own; against the base scale it never moves
    lam_s = v.lam_s if v.selections is not None else base["lam_s"]
    out = dict(tv_target=r["tv_target"], tv_target_se=r["tv_target_se"], lam=v.lam.copy(),
               nflip=nflip, lam_changed=lam_changed(v.lam, lam_s, base["lam"], base["lam_s"]),
               lam_at_lower=(int(np.sum(R.selection(flow.split("_")[0], st.dim).at_lower(
                   v.lam, v.lam_s)[1:])) if v.selections is not None else 0),
               dnorm2=field_parts(v.basis, v.coef, base["coef"], v.times), summary=r["summary"])
    if st.dim == 1:
        out["tv"], out["mass"] = tv_1d(base, r, ST["nx_1d"])
    else:
        _, dx = grid_points(R.grid)
        out["tv"] = tv_grid(np.exp(r["lp"]), np.exp(base["lp"]), dx, dim=2)
        out["mass"], out["tv_iid"], out["tv_iid_se"] = r["mass"], r["tv_iid"], r["tv_iid_se"]
    out["wall_fit"], out["wall_gen"], out["wall"] = t1 - t0, time.time() - t1, time.time() - t0
    return out


def exact_task(job):
    """('exact', target, n, s, ie, draw, epsilon): v_N on X and X' (ie < 0: base norms only)."""
    _, target, n, s, ie, draw, epsilon = job
    t0 = time.time()
    st = C.setting(target)
    y, ys = st.training_set(n, s)
    if ie < 0:
        m, se = exact_l2(st, ys, None, n, s)
        return dict(base_l2norm2=m, base_l2norm2_se=se, wall=time.time() - t0)
    yp, _ = perturb(y, epsilon, xi_draw(target, n, s, draw, st.dim), st.box)
    ysp = st.frame.to_std(yp)
    tv, tv_se, bound = exact_tv(st, ys, ysp, n, s, ie, draw)
    m, se = exact_l2(st, ys, ysp, n, s)
    return dict(tv=tv, tv_se=tv_se, tv_bound=bound, l2diff2=m, l2diff2_se=se,
                wall=time.time() - t0)


def task(job):
    return {"base": base_task, "pert": pert_task, "exact": exact_task}[job[0]](job)


# ------------------------------------------------------------------ sources

def source_names(dim, reg=None) -> dict:
    """{reg: production result} whose stored hold-out lam(t) the run reads: 1D both
    regularizers, the circle the one of --reg (h2 by default)."""
    if dim == 1:
        return {"h2": "oned", "l2": "oned_l2"}
    reg = reg or "h2"
    return {reg: "twod_circle" + ("" if reg == "h2" else f"_{reg}")}


def run_plan(dim, reg=None, m=None, grid_dx=None) -> dict:
    """Result name, flows and whether the exact v_N jobs run, from the command line."""
    if dim == 1:
        name, flows, exact = "stability_1d", list(ST["flows"][1]), True
    elif (reg or "h2") == "h2":
        name, flows, exact = "stability_circle", list(ST["flows"][2]), True
    else:
        # only the regularizer changes; the exact field is that of stability_circle
        name, flows, exact = f"stability_circle_{reg}", [f"{reg}_fixed", f"{reg}_resel"], False
    if m is not None:
        name += f"_m{m}"
    if grid_dx is not None:
        if dim != 2:
            raise ValueError("grid_dx applies to the 2D runs only")
        name += dx_suffix(grid_dx)
    return {"name": name, "flows": flows, "exact": exact}


def load_sources(a, dim, m, reg=None):
    """Stored hold-out lam(t) and tv of the production runs (frames checked), and the optional
    controls references of the refinement rows."""
    out = {}
    names = source_names(dim, reg)
    for reg, name in names.items():
        d = C.load(a, name)
        C.check_frame(d, C.setting("trimodal" if dim == 1 else "circle"), name)
        out[reg] = d
    if dim == 1 and m != C.Protocol(a, 1).m:
        for reg, name in (("h2", "controls_1d"), ("l2", "controls_1d_l2")):
            path = C.result_path(a, name, deps=True)
            if os.path.exists(path):
                with np.load(path, allow_pickle=False) as f:
                    out[reg + "_ctl"] = {k: f[k] for k in f.files}
    return out


def stored(src, dim, target, flow, n, s, m, m_prod):
    """(lam (K,), tv_pos, tv_iid, lam_ref) of the stored hold-out flow of (target, N, set):
    lam the production lam(t), lam_ref the lam(t) of the flow whose tv_pos is the reference
    (the production one at m_prod; at another m the controls '<d>/m<m>/holdout/lam', also
    selected with lam = 0 allowed; None without a reference)."""
    reg = flow.split("_")[0]
    d = src[reg]
    ns, sets = [int(x) for x in d["ns"]], [int(x) for x in d["sets"]]
    j, k = ns.index(n), sets.index(s)
    if dim == 1:
        lam = d[f"{target}/holdout/lam"][j, k]
        if m == m_prod:
            return lam, float(d[f"{target}/holdout/tv_pos"][j, k]), np.nan, lam
        c = src.get(reg + "_ctl")
        key = f"{target}/m{m}/holdout/tv_pos"
        if flow.endswith("fixed") or c is None or key not in c or n not in list(c["ns"]):
            return lam, np.nan, np.nan, None
        jc, kc = list(c["ns"]).index(n), list(c["sets"]).index(s)
        return (lam, float(c[key][jc, kc]), np.nan,
                np.asarray(c[f"{target}/m{m}/holdout/lam"][jc, kc], float))
    lam = d["holdout/lam"][j, k]
    return lam, float(d["holdout/tv_pos"][j, k]), float(d["holdout/tv_iid"][j, k]), lam


def lam_dev(a, b) -> float:
    """max_k |log10 a_k - log10 b_k| (0 where both are 0, inf where only one is)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    both0 = (a == 0) & (b == 0)
    one0 = (a == 0) ^ (b == 0)
    with np.errstate(divide="ignore"):
        d = np.abs(np.log10(np.where(both0, 1.0, a)) - np.log10(np.where(both0, 1.0, b)))
    d[one0] = np.inf
    return float(d.max()) if d.size else 0.0


def slope(x, y) -> float:
    """Least-squares slope of log10 y against log10 x over the finite positive y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0)
    return float(np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)[0]) if ok.sum() >= 2 else np.nan


# ------------------------------------------------------------------ driver

def main():
    p = C.parser(__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dim", type=int, choices=(1,), help="the 1D targets (trimodal, inhom)")
    g.add_argument("--target", choices=("circle",), help="the 2D target (circle)")
    p.add_argument("--reg", choices=("h2", "l2"), default=None,
                   help="circle only: the regularizer of the flows (default h2, result "
                        "stability_circle; l2: stability_circle_l2, flows l2_fixed and "
                        "l2_resel, no exact jobs)")
    p.add_argument("--m", type=int, help="1D spline cells: the refinement rows (result "
                                         "stability_1d_m<M>, epsilon in STABILITY['refine_epsilon'])")
    p.add_argument("--grid-dx", type=float, metavar="DX",
                   help="circle only: spacing of the TV grid (default STABILITY['grid_2d'], "
                        "smoke: its smoke grid; half-width unchanged); result name + _dx<DX in "
                        "thousandths, 3 digits> when DX differs from the production spacing")
    a = C.parse(p)
    dim = 1 if a.dim else 2
    if a.m is not None and dim != 1:
        p.error("--m applies to the 1D targets only")
    if a.reg is not None and dim == 1:
        p.error("--reg applies to --target circle only (1D runs both regularizers)")
    if a.grid_dx is not None:
        if dim == 1:
            p.error("--grid-dx applies to --target circle only (the 1D TV uses no grid)")
        try:
            dx_suffix(a.grid_dx)
        except ValueError as e:
            p.error(f"--grid-dx: {e}")
    P = a.protocol = C.Protocol(a, dim)
    m_prod = P.m
    m = a.m if a.m is not None else P.m
    plan = run_plan(dim, a.reg, a.m, a.grid_dx)
    name, with_exact = plan["name"], plan["exact"]
    targets = ("trimodal", "inhom") if dim == 1 else ("circle",)
    flows = plan["flows"]
    regs = sorted({f.split("_")[0] for f in flows})
    # the regularizer of the worked example and of the 'smallest effective' text: h2 when
    # the run has it (every run but stability_circle_l2)
    reg0 = "h2" if "h2" in regs else regs[0]
    epsilons = np.asarray(config.EPSILON, float)
    if a.m is not None:
        want = ST["refine_epsilon"][:2] if a.smoke else ST["refine_epsilon"]
    else:
        want = epsilons[list(ST["smoke"]["epsilon_idx"])] if a.smoke else epsilons
    eidx = [int(np.argmin(np.abs(np.log10(epsilons / w)))) for w in want]
    draws = ST["smoke"]["draws"] if a.smoke else ST["draws"]
    ns_of = {t: (config.SMOKE["N"],) if a.smoke else tuple(ST["ns"][t]) for t in targets}
    sets = list(P.sets)
    grid = tv_grid_2d(a.smoke, a.grid_dx)
    R = Run(P, m, grid)
    src = load_sources(a, dim, m, a.reg)             # before any heavy work
    st0 = C.setting(targets[0])
    run = C.open_run(name, a, {"targets": targets, "ns": ns_of, "sets": sets,
                               "epsilon": epsilons[eidx].tolist(), "draws": draws,
                               "flows": flows + (["exact"] if with_exact else []), "m": m,
                               "lam candidates": "positive only (lam = 0 excluded); " + "; ".join(
                                   f"{reg}: " + candidates_text(reg, dim)
                                   for reg in sorted({f.split('_')[0] for f in flows}))
                                   + "; fixed lam(t): stored hold-out lam(t), each 0 replaced by "
                                   f"the positive hold-out minimizer on X at m = {m_prod}"
                                   + ("; l2: a no-op, its lowest candidate 10^-14 lam_s equals "
                                      "the solve floor" if "l2" in regs else ""),
                               "tv": (f"1D interpolation of {P.m_lat} quantile-midpoint outputs on "
                                      f"{ST['nx_1d']} points" if dim == 1 else
                                      f"grid [-{grid[0]}, {grid[0]}]^2 dx {grid[1]}, backward RK4"
                                      + ("" if a.grid_dx is None else " (--grid-dx)")),
                               "exact tv": ("StoppedLaw 1D quadrature" if dim == 1 else
                                            f"StoppedLaw Monte Carlo {ST['m_tv2']} draws")
                                           if with_exact else "not run (see stability_circle)",
                               "xi seed": f"[{config.SEEDS['stability'][0]}, target, N, set, draw]"},
                     ABOUT.format(name=name, targets=", ".join(targets),
                                  epsilon=" ".join(f"{x:.0e}" for x in epsilons[eidx]),
                                  flows=", ".join(flows), m=m, tl=list(ST["tl"]),
                                  exact=" and the exact v_N" if with_exact else
                                        " (no exact v_N: stability_circle)",
                                  op=f"B_{OPERATOR_ORDER[reg0]}",
                                  refine="_refine" if a.m is not None else "",
                                  lo=candidates_text(reg0, dim).split("smallest effective ")[1],
                                  out=C.rel(C.result_path(a, name, ""))))
    n0, s0 = ns_of[targets[0]][0], sets[0]
    ys0 = st0.training_set(n0, s0)[1]
    P_m = C.Protocol(a, dim)
    P_m.m = m
    C.worked_example(run, st0, P_m, n0, s0, f"{flows[0]} (stored hold-out lam(t))", reg0,
                     split_sizes=[len(x) for x in st0.split(ys0, s0)],
                     extra={"perturbation": f"x' = x + epsilon xi, epsilon in "
                                            f"{epsilons[eidx].tolist()}, {draws} draws",
                            "lam grid": candidates_text(reg0, dim)
                                        + " (lam = 0 excluded; re-selected flows; fixed flows use "
                                          "the stored hold-out lam(t), zeros replaced)"})
    K = P.times.size

    def base_job(d, n, s, f):
        lam, _, _, lam_ref = stored(src, dim, d, f, n, s, m, m_prod)
        return ("base", d, n, s, f, lam, R, lam_ref)

    base_jobs = [base_job(d, n, s, f) for d in targets for n in ns_of[d] for s in sets
                 for f in flows]
    ex_jobs = [("exact", d, n, s, ie, r, float(epsilons[ie]) if ie >= 0 else 0.0)
               for d in targets for n in ns_of[d] for s in sets
               for ie in [-1] + eidx for r in (range(draws) if ie >= 0 else [0])] if with_exact else []
    if a.mode == "example":
        base_jobs, ex_jobs = base_jobs[:1], []
    base, ex = {}, {}

    def base_done(job, r):
        _, d, n, s, f = job[:5]
        jl = f"{d} N={n} set {s} {f} base"
        base[(d, n, s, f)] = r
        C.log_velocity(run, jl, f, r, P.times)
        rk = np.flatnonzero(r["lam_replaced"])
        if rk.size:
            run(f"[{jl}] stored lam(t) = 0 replaced at {rk.size} t_k (positive candidates): "
                + "; ".join(f"t={P.times[k]:.4g} 0 -> {r['lam'][k]:.4g} (lam/lam_s "
                            f"{r['lam'][k] / r['lam_s_sel'][k]:.3g}"
                            f"{', lowest candidate' if r['lam_at_lower'][k] else ''})"
                            for k in rk))
        if "stored_tv_target" in r:
            run(f"[{jl}] stored lam(t) (zeros at the floor) gives tv_pos "
                f"{r['stored_tv_target']:.6g}; positive candidates {r['tv_target']:.6g}")
        nl = int(np.sum(r["lam_at_lower"][1:]))
        if nl:
            run(f"[{jl}] lam at the lowest candidate at {nl} t_k > 0")
        run.flow(jl, None, r["summary"], {"fit": round(r["wall_fit"], 3),
                                          "generation": round(r["wall_gen"], 3)})
        mt = {"tv_target": r["tv_target"], "tv_target_se": r["tv_target_se"],
              "lam/lam_s": " ".join(f"{x:.3g}" for x in r["lam_rel"])}
        if dim == 2:
            mt |= {"tv_iid": r["tv_iid"], "tv_iid_se": r["tv_iid_se"], "mass": r["mass"]}
        run.metrics(jl, mt)

    def ex_done(job, r):
        _, d, n, s, ie, rr = job[:6]
        ex[(d, n, s, ie, rr)] = r
        jl = f"{d} N={n} set {s} exact" + ("" if ie < 0 else f" epsilon {epsilons[ie]:.0e} draw {rr}")
        mt = {"wall": round(r["wall"], 2)}
        if ie >= 0:
            mt |= {"tv": r["tv"], "tv_se": r["tv_se"], "tv_bound": r["tv_bound"]}
        run.metrics(jl, mt)

    order = sorted(range(len(base_jobs)), key=lambda i: -base_jobs[i][2])
    jobs1 = [base_jobs[i] for i in order] + ex_jobs
    C.pool_map(task, jobs1, a.workers, run,
               lambda j, r: base_done(j, r) if j[0] == "base" else ex_done(j, r))
    run("base flows and exact field done")

    pert = {}
    if a.mode != "example":
        jobs_p = [("pert", d, n, s, ie, r, f, base[(d, n, s, f)]["lam"],
                   {k: base[(d, n, s, f)][k] for k in ("lam", "lam_s", "coef", "z", "lf", "lp")
                    if k in base[(d, n, s, f)]}, float(epsilons[ie]), R)
                  for d in targets for n in ns_of[d] for s in sets
                  for ie in eidx for r in range(draws) for f in flows]
        jobs_p.sort(key=lambda j: (-j[2], j[6].endswith("fixed")))

        def pert_done(job, r):
            _, d, n, s, ie, rr, f = job[:7]
            pert[(d, n, s, ie, rr, f)] = r
            jl = f"{d} N={n} set {s} {f} epsilon {epsilons[ie]:.0e} draw {rr}"
            run.flow(jl, None, r["summary"], {"fit": round(r["wall_fit"], 3),
                                              "generation": round(r["wall_gen"], 3)})
            mt = {"tv": r["tv"], "tv_se": 0.0, "tv_target": r["tv_target"],
                  "tv_target_se": r["tv_target_se"], "lam_changed": r["lam_changed"],
                  "nflip": r["nflip"]}
            if dim == 2:
                mt |= {"tv_iid": r["tv_iid"], "tv_iid_se": r["tv_iid_se"]}
            run.metrics(jl, mt)

        C.pool_map(task, jobs_p, a.workers, run, pert_done)

    # ---- npz
    S, NE = len(sets), len(eidx)
    tl = np.asarray(ST["tl"], float)
    save = {"epsilon": epsilons[eidx], "epsilon_frame": epsilons[eidx] / st0.frame.s,
            "times": P.times, "tl": tl, "sets": np.array(sets), "draws": draws,
            "flows": np.array(flows + (["exact"] if with_exact else [])), "t_end": config.T_END,
            "nstep": P.nstep, "m": m,
            **st0.record()}
    if dim == 2:
        save["grid_2d"] = np.array(grid, float)
    for d in targets:
        ns = list(ns_of[d])
        nn = len(ns)
        sh = (nn, S, NE, draws)
        save[f"{d}/ns"] = np.array(ns)
        save[f"{d}/nflip"] = np.full(sh, np.nan)
        for f in flows:
            q = f"{d}/{f}/"
            for key in ["tv", "tv_target", "tv_target_se", "mass", "lam_changed", "lam_at_lower",
                        "wall"] + (
                    [] if dim == 1 else ["tv_iid", "tv_iid_se"]):
                save[q + key] = np.full(sh, np.nan)
            save[q + "lam"] = np.full(sh + (K,), np.nan)
            save[q + "dnorm2"] = np.full(sh + (K, 3), np.nan)
            for key in ["base_tv_target", "base_tv_target_se", "base_tv_ref", "base_wall"] + (
                    [] if dim == 1 else ["base_tv_iid", "base_tv_iid_se",
                                                  "base_tv_iid_ref", "base_mass"]):
                save[q + key] = np.full((nn, S), np.nan)
            for key in ["base_lam", "base_lam_ref"]:
                save[q + key] = np.full((nn, S, K), np.nan)
            save[q + "base_lam_replaced"] = np.zeros((nn, S, K), bool)
            save[q + "base_lam_at_lower"] = np.zeros((nn, S, K), bool)
            for key in ["base_tv_stored", "base_tv_stored_se"] + (
                    [] if dim == 1 else ["base_tv_iid_stored"]):
                save[q + key] = np.full((nn, S), np.nan)
            save[q + "base_norm2"] = np.full((nn, S, K, 3), np.nan)
        if not with_exact:
            continue
        q = f"{d}/exact/"
        for key in ["tv", "tv_se", "tv_bound", "wall"]:
            save[q + key] = np.full(sh, np.nan)
        for key in ["l2diff2", "l2diff2_se"]:
            save[q + key] = np.full(sh + (tl.size,), np.nan)
        for key in ["base_l2norm2", "base_l2norm2_se"]:
            save[q + key] = np.full((nn, S, tl.size), np.nan)
    for (d, n, s, f), r in base.items():
        j, k, q = ns_of[d].index(n), sets.index(s), f"{d}/{f}/"
        lam_ref, tv_ref, tv_iid_ref, _ = stored(src, dim, d, f, n, s, m, m_prod)
        save[q + "base_tv_target"][j, k] = r["tv_target"]
        save[q + "base_tv_target_se"][j, k] = r["tv_target_se"]
        save[q + "base_tv_ref"][j, k] = tv_ref
        save[q + "base_lam"][j, k], save[q + "base_lam_ref"][j, k] = r["lam"], lam_ref
        save[q + "base_lam_replaced"][j, k] = r["lam_replaced"]
        save[q + "base_lam_at_lower"][j, k] = r["lam_at_lower"]
        if "stored_tv_target" in r:
            save[q + "base_tv_stored"][j, k] = r["stored_tv_target"]
            save[q + "base_tv_stored_se"][j, k] = r["stored_tv_target_se"]
            if dim == 2:
                save[q + "base_tv_iid_stored"][j, k] = r["stored_tv_iid"]
        save[q + "base_norm2"][j, k], save[q + "base_wall"][j, k] = r["norm2"], r["wall"]
        if dim == 2:
            save[q + "base_tv_iid"][j, k], save[q + "base_tv_iid_ref"][j, k] = r["tv_iid"], tv_iid_ref
            save[q + "base_tv_iid_se"][j, k], save[q + "base_mass"][j, k] = r["tv_iid_se"], r["mass"]
    for (d, n, s, ie, rr, f), r in pert.items():
        idx = (ns_of[d].index(n), sets.index(s), eidx.index(ie), rr)
        save[f"{d}/nflip"][idx] = r["nflip"]
        for key, val in r.items():
            if key not in ("nflip", "summary", "wall_fit", "wall_gen"):
                save[f"{d}/{f}/{key}"][idx] = val
    for (d, n, s, ie, rr), r in ex.items():
        q = f"{d}/exact/"
        j, k = ns_of[d].index(n), sets.index(s)
        if ie < 0:
            save[q + "base_l2norm2"][j, k] = r["base_l2norm2"]
            save[q + "base_l2norm2_se"][j, k] = r["base_l2norm2_se"]
            continue
        idx = (j, k, eidx.index(ie), rr)
        for key in ["tv", "tv_se", "tv_bound", "l2diff2", "l2diff2_se", "wall"]:
            save[q + key][idx] = r[key]
    if a.mode != "example":
        report_log(run, save, targets, ns_of, flows, dim, m == m_prod, with_exact)
    run.save(**save)
    run.close()


def _fmt_dev(x) -> str:
    """A check deviation; n/a when no set was covered (never a 0 that reads as a pass)."""
    return "n/a" if np.isnan(x) else f"{x:.1e}"


def report_log(run, save, targets, ns_of, flows, dim, same_m, with_exact=True):
    """Reproduction checks and the summary per (target, N, flow, epsilon)."""
    epsilon = save["epsilon"]
    K = save["times"].size
    run("reproduction of the stored hold-out flows (base = flow on X; positive candidates, "
        "stored zeros replaced): max |tv_pos - stored| over the (N, set), from the base flow "
        "where no t_k was replaced and from the flow with the stored lam(t) (zeros at the floor) "
        f"where one was (FAILED above {TOL_REF:g} fixed, {TOL_RESEL:g} re-selected; n/a: no set "
        "covered), max over t_k > 0 with a positive stored lam of |log10 lam - log10 lam_stored|; "
        "low: t_k > 0 at the lowest positive candidate (replaced or re-selected)")
    bad = False
    for d in targets:
        for f in flows:
            q = f"{d}/{f}/"
            rep = save[q + "base_lam_replaced"]
            # a replaced zero changes the flow, so its stored tv_pos is no reference for the
            # base flow
            clean = ~rep.any(-1)
            rtxt = (f"; stored lam = 0 replaced at {int(rep.sum())} t_k ({int(rep[..., 0].sum())} "
                    f"at t_0) in {int(rep.any(-1).sum())} (N, set)" if rep.any() else "")
            if not np.isfinite(save[q + "base_tv_ref"]).any():
                run(f"  {d} {f:9s} no stored reference at this m (fixed lam of the production m)"
                    + rtxt)
                continue
            def deviation(key, stored_key, ref_key):
                """(max |tv - stored| over the covered (N, set), number covered): the flow with
                the stored lam(t) is the base where nothing was replaced, else its own run."""
                alt = save.get(q + stored_key)
                tv = np.where(clean, save[q + key], np.nan if alt is None else alt)
                dv = np.abs(tv - save[q + ref_key])
                cov = np.isfinite(dv)
                return (float(dv[cov].max()) if cov.any() else np.nan), int(cov.sum())

            dev, ncov = deviation("base_tv_target", "base_tv_stored", "base_tv_ref")
            # t_0 = 0 excluded: v_N(., 0) is affine there, the hold-out score is flat in lam to
            # rounding and its argmin decided by rounding (TV unaffected)
            keep = ~rep[..., 1:].reshape(-1, K - 1)
            ld = 0.0 if not same_m else max(lam_dev(x[kk], y[kk]) for x, y, kk in
                                            zip(save[q + "base_lam"][..., 1:].reshape(-1, K - 1),
                                                save[q + "base_lam_ref"][..., 1:].reshape(-1, K - 1),
                                                keep))
            extra = ""
            if dim == 2:
                di, _ = deviation("base_tv_iid", "base_tv_iid_stored", "base_tv_iid_ref")
                extra = (f", tv_iid {_fmt_dev(di)}, grid mass {np.nanmin(save[q + 'base_mass']):.5f}.."
                         f"{np.nanmax(save[q + 'base_mass']):.5g}")
                dev = np.nanmax([dev, di]) if np.isfinite([dev, di]).any() else np.nan
                if np.nanmax(np.abs(save[q + "base_mass"] - 1.0)) > 1e-2:
                    # the backward-RK4 density is unreliable on a coarse spline (the smoke
                    # m = 24 gives spurious peaks; m = 128 gives mass 1.0002 at N = 100)
                    extra += "  WARNING: grid mass off 1 by > 1e-2, grid TV unreliable"
            # n/a passes the tv part: no set has a reference flow (reported, not hidden)
            tv_ok = np.isnan(dev) or dev <= (TOL_REF if f.endswith("fixed") else TOL_RESEL)
            ok = tv_ok and (f.endswith("fixed") or ld == 0)
            bad |= not ok
            low = save.get(q + "base_lam_at_lower")
            ltxt = "" if low is None else f", low {int(low[..., 1:].sum())} t_k"
            run(f"  {d} {f:9s} tv {_fmt_dev(dev)} over {ncov} of {clean.size} (N, set), "
                f"lam {ld:.2g}{ltxt}{extra}{rtxt}{'' if ok else '  FAILED'}")
        if same_m:
            # both base flows are the positive hold-out minimizer on X at the production m
            for reg in sorted({f.split("_")[0] for f in flows}):
                a, b = f"{d}/{reg}_fixed/base_lam", f"{d}/{reg}_resel/base_lam"
                if a in save and b in save:
                    ld = max(lam_dev(x[1:], y[1:]) for x, y in
                             zip(save[a].reshape(-1, K), save[b].reshape(-1, K)))
                    ok = ld == 0
                    bad |= not ok
                    run(f"  {d} {reg} fixed vs re-selected base lam(t), t_k > 0: {ld:.2g}"
                        + ("" if ok else "  FAILED"))
    run("mean over sets and draws per epsilon (data units "
        + " ".join(f"{x:.0e}" for x in epsilon) + ")")
    small = epsilon <= 1e-3
    for d in targets:
        for j, n in enumerate(ns_of[d]):
            run(f"{d} N={n}  nflip max {np.nanmax(save[f'{d}/nflip'][j]):.0f}")
            for f in flows + (["exact"] if with_exact else []):
                q = f"{d}/{f}/"
                tv = np.nanmean(save[q + "tv"][j], axis=(0, 2))
                run(f"  {f:9s} TV " + " ".join(f"{x:.2e}" for x in tv)
                    + f"  slope epsilon<=1e-3 {slope(epsilon[small], tv[small]):.2f}, all "
                      f"{slope(epsilon, tv):.2f}")
                if f == "exact":
                    b = np.nanmean(save[q + "tv_bound"][j], axis=(0, 2))
                    run(f"  {'':9s} bound " + " ".join(f"{x:.2e}" for x in b))
                    l2 = np.sqrt(np.nanmean(save[q + "l2diff2"][j], axis=(0, 2)))
                    for i, sl in enumerate(save["tl"]):
                        run(f"  {'':9s} ||v_N - v_N'||_L2(rho_N) at 1-t={sl:.0e}: "
                            + " ".join(f"{x:.2e}" for x in l2[:, i]))
                    continue
                h2 = np.sqrt(np.sum(save[q + "dnorm2"][j], -1)).max(-1)
                hm = np.nanmean(h2, axis=(0, 2))
                run(f"  {'':9s} sup_t H2 " + " ".join(f"{x:.2e}" for x in hm)
                    + f"  slope epsilon<=1e-3 {slope(epsilon[small], hm[small]):.2f}, all "
                      f"{slope(epsilon, hm):.2f}")
                l2n = np.sqrt(save[q + "dnorm2"][j][..., 0]).max(-1)
                run(f"  {'':9s} sup_t L2 " + " ".join(f"{x:.2e}" for x in np.nanmean(l2n, axis=(0, 2))))
                if f.endswith("resel"):
                    lc = np.nanmean(save[q + "lam_changed"][j], axis=(0, 2))
                    run(f"  {'':9s} lam changed at (mean over sets, draws) "
                        + " ".join(f"{x:.1f}" for x in lc) + f" of {K} t_k")
                    if q + "lam_at_lower" in save:
                        ll = np.nanmean(save[q + "lam_at_lower"][j], axis=(0, 2))
                        run(f"  {'':9s} lam at the lowest candidate at (mean over sets, draws) "
                            + " ".join(f"{x:.1f}" for x in ll) + f" of {K - 1} t_k > 0")
                run(f"  {'':9s} tv_target " + " ".join(
                    f"{x:.4f}" for x in np.nanmean(save[q + "tv_target"][j], axis=(0, 2)))
                    + f"  (base {np.nanmean(save[q + 'base_tv_target'][j]):.4f})")
    if bad:
        run("FAILED: a base flow does not reproduce its stored hold-out flow")


if __name__ == "__main__":
    main()
