"""Every scalar constant of the paper's setup (main.tex sec:setup, sec:solver, sec:cv) and of
the experiment protocol.

No other module defines a number that this module defines; no command-line flag changes
MU, L, the latent, NSTEP or T_END. The lambda grid is relative to lam_s(t) (LAM_DECADES), and
L = 4.5 holds for every 2D target. The second half (experiment grids) is protocol of experiments/compute/, kept here so that
the manifests record it.
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ model (eq:regfm, eq:normaleq)

#: Weight mu of the mass term mu ||v||^2 (frame units), K = A + mu M + lam P (eq:normaleq).
MU = 1e-8
#: Half-width L of Omega = [-L, L]^D, the latent support and the velocity's box (sec:setup).
L = {1: 5.0, 2: 4.5}
#: End time of the sampler, t_end = 1 - 1e-4 (sec:setup).
T_END = 1.0 - 1e-4
#: RK4 steps in tau = -log(1 - t) from 0 to tau_end (sec:sampling).
NSTEP = 96
#: The 49 selection times: 19 points linspace(0, 0.9) and 30 points with 1 - t from
#: 10^-1.1 to 10^-4, 0.1 decade apart (sec:setup). The expression fixes the floats bitwise.
TIMES = np.concatenate([np.linspace(0.0, 0.9, 19), 1.0 - np.geomspace(0.1, 1e-4, 31)[1:]])
#: Spline cells m per axis (sec:solver).
M_SPLINE = {1: 512, 2: 128}
#: The floor lam_min = LAM_FLOOR_REL lam_s(t): "lam = 0" solves there (sec:solver).
LAM_FLOOR_REL = 1e-14
#: Null-space split of the solve for lam >= SPLIT_RATIO lam_s (S:sec:splinedetails).
SPLIT_RATIO = 100.0

# ------------------------------------------------------------------ protocol

#: Candidate decades of the lambda search RELATIVE to lam_s(t): the
#: candidates are 10^j lam_s(t), j = lo, ..., hi - 1, plus lam = 0 (the floor) and the two
#: half-decade neighbours of the best positive candidate.
#:
#: Derivation: q = log10(lam / lam_s) of every positive selection in pilot runs of the
#: selection rules (all targets, regularizers, sets and the 49 times), selections pinned at a
#: grid end excluded; lo = floor(min q) - 2, hi = ceil(max q) + 3 (hi exclusive, so the
#: largest candidate is 10^(ceil(max q) + 2)): the range extends two decades beyond the pilot
#: selections (S:sec:splinedetails). Exceptions: l2 lo = -14 (candidates below the floor
#: 1e-14 lam_s repeat the floor solve) and l2 1D hi = 7, so that the v = 0 limit selections
#: sit at 10^6 lam_s. The production runs select at the top candidate (h2, l2: the limit
#: lam -> inf; h1 1D: t <= 0.25) and, for h2 and h1, at the lowest one or its lower
#: half-decade neighbour.
LAM_DECADES = {"h2": {1: (-8, 16), 2: (-9, 14)},
               "h1": {1: (-12, 8), 2: (-13, 8)},
               "l2": {1: (-14, 7), 2: (-14, 7)}}
#: Training-set sizes and the number of sets per size.
NS = (100, 1000, 3000)
SETS = 5
#: Latent points per flow: 1D quantile midpoints, 2D scrambled Sobol points.
M_LAT = {1: 3000, 2: 4096}
#: Oracle reference points per time: 1D Sobol (rng SEEDS["oracle_1d"]), 2D normal draws.
M_ORACLE = {1: 2048, 2: 4096}
#: Held-out fraction of the hold-out rule (sec:cv).
HOLDOUT_FRAC = 0.2

#: Data boxes Omega_y (original coordinates); the frame maps each onto [-L, L]^D.
DATA_BOX = {"trimodal": [[-1.05, 1.05]], "inhom": [[-1.05, 1.05]],
            "circle": [[-2.6, 2.6]] * 2, "checkerboard": [[-4.2, 4.2]] * 2,
            "spirals": [[-3.35, 3.35]] * 2, "gauss8": [[-4.25, 4.25]] * 2}

#: Every rng seed base, which fixes training sets, splits and latents bitwise. "train" bases enter as default_rng(base * n + set);
#: the others as default_rng(base + set) unless stated.
SEEDS = {
    "train": {"trimodal": 10_000, "inhom": 20_000, "circle": 30_000,
              "checkerboard": 32_000, "spirals": 33_000, "gauss8": 34_000},
    "split": 1_000,              # hold-out permutation (1D, 2D)
    "oracle_1d": 3,              # Sobol(2, scramble) seed: column 0 latent, column 1 y quantile
    "oracle_2d": 7,              # normal draws: latent (4096, 2), then target.sample(4096)
    "sobol_2d": 11,              # 2D latents, Sobol(2, scramble)
    "iid": 60_000,               # iid latents (tv_iid)
    "kde_draws": 40_000,         # KDE draws for tv_pos
    "kde_crn": 50_000,           # common random numbers of the 2D KDE rules
    "kde_oracle_2d": 45_000,     # 2D KDE / EM draws
    "em_restarts": 46_000,       # EM random partitions: default_rng(46_000 n + set)
    "law": 70_000,               # draws of the stopped law rho_N(t_end)
    "controls_iid": 80_000,      # controls_iid latents
    "stability": (97_000, 98_000, 99_000),   # xi, L2 draws, TV draws (sequence seeds)
}

# ------------------------------------------------------------------ experiment grids
# Protocol constants of the compute scripts (experiments/compute/). Each block
# names the script(s) that read it.

#: oned.py: rules of the flow (every regularizer) and the KDEs (h2 only).
ONED_RULES = ("oracle", "holdout")
ONED_KDE_RULES = ("silverman", "sj", "lscv", "abr_lscv", "fix_oracle", "abr_oracle")
#: 1D KDE grid (binned FFT density, dx on [-XLIM, XLIM]) and bandwidth candidates per target
#: (geomspace(lo, hi, n)) of oned.py and kde4.py.
KDE_1D = {"dx": 1e-4, "xlim": 1.6,
          "hs": {"trimodal": (2e-3, 0.3, 241), "inhom": (5e-4, 0.3, 281)}}

#: lam_sweep.py (tab_main constant row, tab_lam_sweep_full):
#: training-set size, c0 = median stored hold-out lam over t_k >= c0_tmin; the half-decade
#: grids s = 10^(k/2) (scale of the hold-out lam(t)) and c = c0 10^(k/2) (constant lam). The
#: grids per (target, reg) are fixed: every const grid is k = -6..6; scale k = -4..4 except
#: checkerboard l2 and gauss8 l2 (-4..8) and circle l2 (-4..12).
LAM_SWEEP = {"n": 1000, "c0_tmin": 0.97,
             "const_k": tuple(range(-6, 7)),
             "scale_k": {f"{t}/{r}": tuple(range(-4, 5)) for t in
                         ("trimodal", "inhom", "circle", "checkerboard", "spirals", "gauss8")
                         for r in ("h2", "h1", "l2")}
             | {"checkerboard/l2": tuple(range(-4, 9)), "gauss8/l2": tuple(range(-4, 9)),
                "circle/l2": tuple(range(-4, 13))},
             "smoke_scale_k": (0,), "smoke_const_k": (0, 2)}

#: stability.py (fig_stability, tab_stability, tab_stability_refine): the perturbation sizes
#: epsilon (data units; paper Sec. 5.5 varepsilon), x_i' = x_i + epsilon xi_i, |xi_i| = 1.
EPSILON = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
#: draws of xi per (target, N, set); N per target (1D at N = 100 only, the paper's case);
#: flows per dimension; 1 - t of the exact-field L2(rho_N) checks and their Monte Carlo draws;
#: draws of the 2D stopped-law TV; 1D TV of two flows on nx_1d points; 2D TV on the grid
#: [-half, half]^2 with spacing dx (circle); the refinement resolutions (--m) and their
#: epsilon subset; tolerance of the reproduction of the stored hold-out flows (fixed lam,
#: re-selected); smoke: epsilon indices, draws and the 2D grid.
STABILITY = {"draws": 2, "ns": {"trimodal": (100,), "inhom": (100,), "circle": (1000,)},
             "flows": {1: ("h2_fixed", "h2_resel", "l2_fixed", "l2_resel"),
                       2: ("h2_fixed", "h2_resel")},
             "tl": (1e-1, 1e-2, 1e-3, 1e-4), "m_l2": 20_000, "m_tv2": 20_000,
             "nx_1d": 2**17 + 1, "grid_2d": (2.6, 0.02),
             "refine_m": (256, 512, 1024), "refine_epsilon": (1e-6, 1e-5, 1e-4),
             "tol_ref": 1e-6, "tol_resel": 1e-4,
             "smoke": {"epsilon_idx": (1, 4), "draws": 1, "grid_2d": (2.6, 0.1)}}

#: gabriel.py: bandwidth grids in ORIGINAL data units (the script divides by frame.s^2 for
#: the frame's h'): per-time candidates {0} U logspace(*h_t), the G3 sweep logspace(*h_sweep),
#: the G3 Brent refinement tolerance in log10 h.
GABRIEL = {"h_t": (-6.0, 0.0, 25), "h_sweep": (-5.0, 0.0, 21), "brent_xatol": 0.03}
#: floors.py: RK4 step counts of the population flow as multiples of the production NSTEP.
FLOORS_STEP_FACTORS = (0.5, 1, 2)
#: Reference integrator of the exact v_N (sampler.reference_rk4): RK4 with step doubling from
#: NSTEP steps until two consecutive levels differ by <= tol_x in x (frame units, max over
#: points) and by <= tol_logf in log f (MEAN over points), at most max_doublings doublings.
#: Why the mean: a change Delta log f moves each term (1 - pi/f)_+ of tv_pos by at most about
#: |Delta log f|, so the mean bounds |Delta TV|, and 1e-4 keeps the integration share of a TV
#: estimate well below its Monte Carlo SE (1e-3 to 1e-2). The max sits on < 1% of the points,
#: at the box faces where div v_N jumps, and converges only to first order (measured:
#: 1e-2 at 1536 steps for N = 3000), so it is logged (dlogf_max), not used.
#: tol_x 1e-6 (frame units) is 2e-7 of the box half-width L.
REFERENCE_RK4 = {"tol_x": 1e-6, "tol_logf": 1e-4, "max_doublings": 6}
#: tend.py: 1 - t_end values; added selection times 1 - 10^(-k/10), k in range(*new_k);
#: direct draws of rho_N(t_end) for tv_law; the added times kept in smoke runs (indices into
#: the added times: 1 - t = 1e-5 .. 1e-8). The exact v_N flow uses REFERENCE_RK4.
#: table_one_minus_t_end: the end times tab_tend shows, in this order (report only; the
#: results hold every value of one_minus_t_end).
TEND = {"one_minus_t_end": (1e-4, 1e-6, 1e-8), "new_k": (41, 81), "law_draws": 30_000,
        "smoke_new_idx": (9, 19, 29, 39), "table_one_minus_t_end": (1e-4, 1e-8)}
#: integration_error.py: RK4 step counts as multiples of NSTEP (production first), the DOP853
#: tolerances of the hold-out flow's reference (the exact v_N uses REFERENCE_RK4), iid latents
#: per flow (the first m of oned_eval's iid draws) and the
#: flows (smoke: SMOKE_ADAPTIVE_LATENTS points).
INTEGRATION_ERROR = {"step_factors": (1, 4, 16), "rtol": 1e-10, "atol": 1e-12, "m_lat": 1000,
                     "flows": ("exact", "holdout")}
#: tend.py, integration_error.py, oned_eval.py, hero_1d.py in smoke runs only: latent points of
#: the reference rows (exact v_N by sampler.reference_rk4) and the cap on its doublings, so that
#: a smoke run stays within the CI budget; the full runs use M_LAT / m_lat and REFERENCE_RK4.
SMOKE_ADAPTIVE_LATENTS = 32
SMOKE_REFERENCE_DOUBLINGS = 2
#: compare_time.py: cases, training-set sizes (set 0) and rules; the refined time grid
#: inserts the tau-midpoints between consecutive selection times.
COMPARE_TIME = {"cases": ("circle", "trimodal"), "ns": (1000, 100), "rules": ("oracle", "holdout")}

#: controls.py: spline resolutions m of the refinement study per dimension (production
#: M_SPLINE in the middle); smoke resolutions (they contain SMOKE["M_SPLINE"], the m whose
#: lam -> 0 rows tab_oned_full / tab_twod_full read from controls_iid in smoke runs).
CONTROLS_M = {1: (256, 512, 1024), 2: (64, 128, 256)}
CONTROLS_M_SMOKE = {1: (32, 64, 128), 2: (12, 16, 24)}
#: controls.py rows per regularizer: hold-out lam(t) and the lam -> 0 floor (set_floor). The
#: ablations need hold-out only (no table or figure reads an ablation lam -> 0 row).
CONTROLS_ROWS = {"h2": ("holdout", "lam0"), "h1": ("holdout",), "l2": ("holdout",)}
#: controls.py --m: the extra resolution of a 2D target (result controls_<T>[_<reg>]_m<m>), the
#: N of its hold-out rows per regularizer and of its production-latent lam -> 0 rows (h2 only);
#: controls_iid.py --m: the N of its iid lam -> 0 rows (result controls_iid_<T>_m<m>), which
#: fits the lam -> 0 flow itself and reads the dependency's lam -> 0 row only as the reference
#: ref_tv_pos where lam0_ns has it (NaN elsewhere). "smoke_m" replaces m in smoke runs.
#: tab_refine_2d reads every N of every target; the circle's lam0_ns is fig_accuracy's
#: N = 1000 point, the other targets need no such row.
_EXTRA_NS = {"h2": (100, 1000, 3000), "h1": (100, 1000, 3000), "l2": (100, 1000, 3000)}
CONTROLS_M_EXTRA = {
    "circle": {"m": 512, "smoke_m": 32, "holdout_ns": dict(_EXTRA_NS), "lam0_ns": (1000,),
               "iid_ns": (100, 1000, 3000)},
    **{t: {"m": 512, "smoke_m": 32, "holdout_ns": dict(_EXTRA_NS), "lam0_ns": (),
           "iid_ns": (100, 1000, 3000)} for t in ("checkerboard", "spirals", "gauss8")}}
#: controls_iid.py: draws of the stopped law rho_N(., t_end) for tv_law (smoke: fewer), and
#: the N per case (every N for every case: tab_refine_1d / _2d, and at the production m
#: tab_oned_full / tab_twod_full).
CONTROLS_IID = {"m_law": 30_000, "m_law_smoke": 4096,
                "ns": {"trimodal": (100, 1000, 3000), "inhom": (100, 1000, 3000),
                       "circle": (100, 1000, 3000), "checkerboard": (100, 1000, 3000),
                       "spirals": (100, 1000, 3000), "gauss8": (100, 1000, 3000)}}

#: oned_eval.py: the Gabriel rows regenerated from the stored h
#: (Gabriel label -> gabriel.npz bandwidth key), the figure grid XF and histogram
#: edges (linspace(lo, hi, n)), the draws of the stopped law rho_N(t_end) for tv_law.
ONED_EVAL = {"gabriel": {"G1": "h1", "G2": "h2", "G3": "h3", "G4": "h4"},
             "xf": (-1.1, 1.1, 881), "hist_edges": (-1.0, 1.0, 51),
             "law_draws": 30_000,
             # the reference (REFERENCE_RK4) of the exact flow in smoke runs integrates this
             # many midpoints (+ as many iid), so that the smoke group stays within its budget
             "ref_points_smoke": 4}
#: kde4.py: fourth-order KDE rules (kernel kind _ bandwidth selection).
KDE4_RULES = ("fix4_oracle", "fix4_lscv", "abr4_oracle", "abr4_lscv")
#: hero_1d.py (fig_hero): image cells
#: (nt in tau, nx in x) full and smoke, trajectories k, x range of the images (original
#: units), fine cells of the stopped law, prior grid (+-prior_sd frame sd, points), Hermite
#: points per RK4 step, zoom inset (start time, points, span, trajectories), hero N.
HERO_1D = {"n": 100, "image": (500, 400), "image_smoke": (40, 30), "k": 28,
           "x_range": (-1.1, 1.1), "law_cells": 2200, "prior_sd": 5.0, "prior_points": 1001,
           "sub": 8, "zoom_t0": 0.9, "zoom_pts": 3, "zoom_span": 0.04, "zoom_k": 12,
           "k_smoke": 6, "zoom_k_smoke": 4}
#: gabriel_hero.py: N of its flows, the regularizers of --regs, and the radius of the
#: "within r of a training point" fraction (original units; sec:operator).
GABRIEL_HERO = {"n": 100, "regs": ("h2", "h1", "l2"), "within_r": 1e-3}

#: twod.py, kde2d.py, kde4_2d.py, twod_eval.py: per 2D target the KDE
#: grid ("nodes": arange(-half, half + dx/2, dx), the circle's; "centres": -half + dx/2 + k dx,
#: so that no node lies on a checkerboard edge), whether the grid TV adds the KDE mass outside
#: the grid, the FFT size, the bandwidth candidates geomspace(lo, hi, n) (fixed h, Abramson h0,
#: fourth-order h), whether the target has an exact curve tangent (anisotropic rules), and the
#: figure grid linspace(-half, half, n) of the density images.
TWOD_TARGETS = {
    "circle": {"grid": ("nodes", 2.8, 0.01), "grid_tv_outside": False, "nfft": 1024,
               "hs": (0.02, 0.6, 61), "aniso": True, "xf2": (2.6, 261)},
    "checkerboard": {"grid": ("centres", 4.4, 0.02), "grid_tv_outside": True, "nfft": 1024,
                     "hs": (0.04, 2.0, 71), "aniso": False, "xf2": (4.2, 261)},
    "spirals": {"grid": ("centres", 3.5, 0.01), "grid_tv_outside": True, "nfft": 1536,
                "hs": (0.01, 0.6, 71), "aniso": True, "xf2": (3.35, 261)},
    "gauss8": {"grid": ("centres", 4.6, 0.02), "grid_tv_outside": True, "nfft": 1024,
               "hs": (0.02, 1.5, 71), "aniso": False, "xf2": (4.25, 261)},
}
#: Anisotropic (h_T, h_N) lattice H = h0 ratio^k, k < nh, h_T >= h_N, h_N <= H[nmax_n].
TWOD_ANISO_LATTICE = {"h0": 0.02, "ratio": 1.28, "nh": 17, "nmax_n": 11}
#: twod.py: selection rules (every regularizer).
TWOD_RULES = ("oracle", "holdout")
#: kde2d.py rules (ani_* only for targets with "aniso") and draws per KDE (tv_pos);
#: kde4_2d.py rules and the direct-summation cut of the anisotropic K4 (r^T S^-1 r <= cut^2);
#: em8.py EM settings (covariance ridge, random restarts, iterations, tolerance of the mean
#: log-likelihood).
KDE2D_RULES = ("fix_lscv", "fix_oracle", "abr_lscv", "abr_oracle", "ani_oracle", "ani_lscv")
KDE2D_DRAWS = 4096
KDE4_2D_RULES = ("k4_oracle", "k4_lscv", "k4p_oracle", "k4p_lscv", "k4_ani_oracle",
                 "k4_ani_lscv")
KDE4_2D_ANISO_CUT = 6.0
EM8 = {"ridge": 1e-3, "restarts": 20, "iters": 2000, "tol": 1e-10}
#: twod_eval.py and kde4_2d.py: points of a stored cloud, draws of the stopped law
#: rho_N(t_end) for tv_law (smoke: fewer), backward log-density check points.
TWOD_EVAL = {"cloud": 2000, "m_law": 30_000, "m_law_smoke": 4096, "image_check": 256}
#: Smoke runs of kde2d / kde4_2d / twod_eval: every k-th bandwidth candidate and every k-th
#: figure-grid node per axis.
TWOD_SMOKE_STRIDE = {"hs": 3, "xf2": 4}

# ------------------------------------------------------------------ smoke runs (CI)

#: Indices into TIMES of the 9 smoke times (t = 0 and the last, 1 - 1e-4, included).
_SMOKE_TIMES = [0, 9, 18, 24, 30, 36, 42, 46, 48]
SMOKE = {"N": 100, "SET": 0, "M_SPLINE": {1: 64, 2: 24}, "TIMES": TIMES[_SMOKE_TIMES],
         "NSTEP": 24, "M_LAT": 512}


#: The experiment-grid constants above, listed in the manifests (as_dict).
EXPERIMENT_CONSTANTS = ["ONED_RULES", "ONED_KDE_RULES", "KDE_1D",
                        "CONTROLS_M", "CONTROLS_M_SMOKE", "CONTROLS_ROWS", "CONTROLS_M_EXTRA",
                        "CONTROLS_IID", "TWOD_TARGETS", "TWOD_ANISO_LATTICE", "TWOD_RULES",
                        "KDE2D_RULES", "KDE2D_DRAWS", "KDE4_2D_RULES",
                        "KDE4_2D_ANISO_CUT", "EM8", "TWOD_EVAL", "TWOD_SMOKE_STRIDE",
                        "GABRIEL", "FLOORS_STEP_FACTORS", "REFERENCE_RK4", "TEND", "INTEGRATION_ERROR",
                        "SMOKE_ADAPTIVE_LATENTS", "SMOKE_REFERENCE_DOUBLINGS", "COMPARE_TIME",
                        "ONED_EVAL", "KDE4_RULES", "HERO_1D", "GABRIEL_HERO",
                        "LAM_SWEEP", "EPSILON", "STABILITY"]


def as_dict() -> dict:
    """Every constant of this module as a JSON-ready dict (runlog manifest 'config')."""
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [conv(x) for x in v]
        return v
    names = ["MU", "L", "T_END", "NSTEP", "TIMES", "M_SPLINE", "LAM_FLOOR_REL", "SPLIT_RATIO",
             "LAM_DECADES", "NS", "SETS", "M_LAT", "M_ORACLE", "HOLDOUT_FRAC",
             "DATA_BOX", "SEEDS", "SMOKE"]
    names += [k for k in EXPERIMENT_CONSTANTS if k not in names]
    return {k: conv(globals()[k]) for k in names}
