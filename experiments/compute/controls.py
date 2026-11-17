"""Discretization controls: does the spline space regularize?
Paper: Figure 2 (fig:accuracy), Table 5 (tab:refine-1d), Table 6 (tab:refine-2d); Sec. 5.4
"Excluding Implicit Regularization from Discretization". controls_iid regenerates the lam -> 0
rows from iid latents (the TV the paper reports) and records these production-latent rows
beside them as its reference ref_tv_pos (a check).

For each spline resolution m, two rows: hold-out lam(t) (selected per time as in oned.py /
twod.py: fit 80 / val 20, grid config.LAM_DECADES[reg][D] relative to lam_s(t)) and
lam -> 0 (Velocity.set_floor: lam = LAM_FLOOR_REL lam_s(t) at every time, with mu M; the one
lam -> 0 level of every script). If the tuned
TV does not change with m while the lam -> 0 TV approaches memorization as m grows, the
penalty, not the basis, causes the generalization.
  --dim 1            trimodal and inhom, m in config.CONTROLS_M[1] (256, 512, 1024; production
                     512): result controls_1d.
  --target T         one 2D target (circle, checkerboard, spirals, gauss8), m in
                     config.CONTROLS_M[2] (64, 128, 256; production 128): result controls_T.
  --reg h1 | l2      the penalty ablation (rows config.CONTROLS_ROWS: hold-out only), suffix
                     _<reg>: controls_1d_h1, controls_circle_l2, ...
  --m M              (with --target, M = config.CONTROLS_M_EXTRA[T]["m"] = 512 for every 2D
                     target): only that resolution, hold-out at the N of CONTROLS_M_EXTRA
                     holdout_ns[reg] (every N) and (h2) lam -> 0 at lam0_ns (circle N = 1000,
                     the other targets none: controls_iid_<T>_m512 fits its own lam -> 0
                     flows); suffix _m<M>: controls_<T>_m512, controls_<T>_h1_m512,
                     controls_<T>_l2_m512, the m = 512 rows of tab_refine_2d (b) and, for the
                     circle, of fig_accuracy.
N in config.NS, config.SETS sets; training sets, splits, latents (1D 3000 truncated quantile
midpoints, 2D 4096 scrambled Sobol), RK4 (config.NSTEP steps to T_END) and frames as oned.py.
Smoke: m in config.CONTROLS_M_SMOKE (contains SMOKE M_SPLINE), --m uses its smoke_m.

npz keys (arrays (len(ns), len(sets), ...), NaN where a cell is not computed):
<case>/m<m>/<rule>/{tv_pos, tv_pos_se, wall_fit, wall_gen, wall, frac_out, frac_out_any,
nfev, max_abs, lam (K,), lam_s (K,), lam_rel (K,)} for rule in {holdout, lam0}; holdout also
{at_limit, scores, limit_scores, nsolves, nevals, residual, wall_t} (K,) and the per-job
history <case>/m<m>/<rule>/N<n>_set<s>/{tables (K, k, 2), at_limit, nsolves, residual, lam,
lam_s, lam_rel, ...} (lam0: lam, lam_s, lam_rel only);
<case>/ms (resolutions of the case), <case>/frame_c, <case>/frame_s, <case>/frame_L;
cases, ns, sets, times, reg, rows, decades, lam0_rel (= LAM_FLOOR_REL).
Readers: controls_iid (frames, ms, the lam0 cells as ref_tv_pos), the report (tab_refine_1d,
tab_refine_2d, fig_accuracy: <case>/m<m>/holdout/tv_pos).
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import copy
import time

import numpy as np

from smoothfm import HoldOut, config

ABOUT = """{name}: discretization controls, {cases}, penalty {reg}, m in {ms}, rows {rows}
(holdout: lam(t) by hold-out per time, history below; lam0: lam = {floor:g} lam_s(t) at every
time, with mu M). Each job: case, N, m, row, set -> lam(t_k), RK4 from the production latents,
tv_pos with its Monte Carlo SE. Outputs: {out}.npz (read by {readers}, report tab_refine_1d,
tab_refine_2d, fig_accuracy), {out}.log, {out}.manifest.json."""


class HoldOutRule:
    """rule_factory of the hold-out rule (top-level, picklable)."""

    def __init__(self, yt, yv):
        self.yt, self.yv = yt, yv

    def __call__(self, v, t):
        ne = v.normal_equations(t, self.yt)
        return HoldOut(ne, ne.validation(self.yv))


def task(job):
    case, n, s, m, rule, reg, P = job
    st = C.setting(case)
    _, ys = st.training_set(n, s)
    t0 = time.time()
    v = st.velocity(ys, P.times, m, reg)
    if rule == "holdout":
        v.fit(HoldOutRule(*st.split(ys, s)), C.search(reg, st.dim))
    else:
        v.set_floor()
    rec = C.velocity_record(v, time.time() - t0)
    rec.update(C.flow_record(C.generate(v, st.latents(P.m_lat), st.std, P.nstep)))
    rec["wall"] = rec["wall_fit"] + rec["wall_gen"]
    return rec


def plan(a, P):
    """(result name, cases, {case: ms}, rows, jobs (case, n, set, m, rule)), largest first."""
    dim = 1 if a.dim is not None else 2
    cases = list(C.TARGETS_1D) if dim == 1 else [a.target]
    rows = config.CONTROLS_ROWS[a.reg]
    name = f"controls_{'1d' if dim == 1 else a.target}" + ("" if a.reg == "h2" else f"_{a.reg}")
    jobs = []
    if a.m is not None:
        ex = config.CONTROLS_M_EXTRA[a.target]
        name += f"_m{a.m}"
        m = ex["smoke_m"] if a.smoke else ex["m"]
        ms = {c: [m] for c in cases}
        cells = {"holdout": ex["holdout_ns"][a.reg], "lam0": ex["lam0_ns"]}
        for rule in rows:
            ns = P.ns if a.smoke else tuple(cells[rule])
            jobs += [(c, n, s, m, rule) for c in cases for n in ns for s in P.sets]
    else:
        mm = (config.CONTROLS_M_SMOKE if a.smoke else config.CONTROLS_M)[dim]
        ms = {c: list(mm) for c in cases}
        jobs = [(c, n, s, m, r) for c in cases for n in P.ns for m in mm for r in rows
                for s in P.sets]
    jobs.sort(key=lambda j: (-j[1], -j[3]))
    return name, cases, ms, rows, jobs


def main():
    p = C.parser(__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dim", type=int, choices=(1,), help="the 1D cases (trimodal, inhom)")
    g.add_argument("--target", choices=C.TARGETS_2D, help="one 2D target")
    p.add_argument("--reg", choices=("h2", "h1", "l2"), default="h2",
                   help="penalty: h2 (production), h1 / l2 (ablations, suffix _<reg>)")
    p.add_argument("--m", type=int, default=None,
                   help="the extra resolution of config.CONTROLS_M_EXTRA (512); "
                        "suffix _m<m>")
    a = C.parse(p)
    if a.m is not None and (a.target not in config.CONTROLS_M_EXTRA
                            or a.m != config.CONTROLS_M_EXTRA[a.target]["m"]):
        p.error("--m: only " + ", ".join(f"--target {t} --m {e['m']}"
                                         for t, e in config.CONTROLS_M_EXTRA.items()))
    dim = 1 if a.dim is not None else 2
    P = a.protocol = C.Protocol(a, dim)
    name, cases, ms, rows, jobs = plan(a, P)
    readers = "controls_iid"
    run = C.open_run(name, a, {"cases": cases, "ms": ms, "rows": rows, "reg": a.reg,
                               "lam grid": C.search_text(a.reg, dim),
                               "lam0": f"floor {config.LAM_FLOOR_REL:g} lam_s(t), with mu M",
                               "latents": (f"{P.m_lat} truncated-latent quantile midpoints"
                                           if dim == 1 else
                                           f"{P.m_lat} truncated-latent scrambled Sobol points")},
                     ABOUT.format(name=name, cases=", ".join(cases), reg=a.reg,
                                  ms=sorted({m for v in ms.values() for m in v}),
                                  rows=", ".join(rows), floor=config.LAM_FLOOR_REL,
                                  out=C.rel(C.result_path(a, name, "")), readers=readers))
    # worked example: hold-out, smallest N and set, the production m if the run has it
    m0 = P.m if P.m in ms[cases[0]] else ms[cases[0]][0]
    hold = [j for j in jobs if j[4] == "holdout" and j[0] == cases[0] and j[3] == m0]
    first = min(hold, key=lambda j: (j[1], j[2]))
    st0 = C.setting(first[0])
    ys0 = st0.training_set(first[1], first[2])[1]
    P0 = copy.copy(P)
    P0.m = first[3]
    C.worked_example(run, st0, P0, first[1], first[2], "holdout", a.reg,
                     split_sizes=[len(x) for x in st0.split(ys0, first[2])],
                     extra={"resolutions m": str(ms), "rows": ", ".join(rows),
                            "lam0 row": f"lam = {config.LAM_FLOOR_REL:g} lam_s(t) at every "
                                        f"time (Velocity.set_floor)"})
    if a.mode == "example":
        jobs = [first]
    ns = sorted({j[1] for j in jobs})
    sets = sorted({j[2] for j in jobs})
    jobs = [j + (a.reg, P) for j in jobs]
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))

    def done(job, rec):
        c, n, s, m, r = job[:5]
        jl = f"{c} N={n} set {s} m={m} {r}"
        C.log_velocity(run, jl, r, rec, P.times)
        walls = {"fit": rec["wall_fit"], "generation": rec["wall_gen"]}
        if "wall_t" in rec:
            walls["selection"] = float(np.sum(rec["wall_t"]))
        C.log_flow(run, jl, rec, walls)
        save.put_record(f"{c}/m{m}/{r}", (ns.index(n), sets.index(s)), rec)

    C.pool_map(task, jobs, a.workers, run, done)

    run("summary: tv_pos mean (SE over sets) [min, max]; hold-out selections at a grid end")
    for c in cases:
        for m in ms[c]:
            k = f"{c}/m{m}/holdout/at_limit"
            if k in save:
                al = save[k]
                run(f"  {c} m={m} holdout at_limit {int(np.nansum(al))} of "
                    f"{int(np.isfinite(al).sum())} (set, t) selections")
        for i, n in enumerate(ns):
            for m in ms[c]:
                have = [r for r in rows if f"{c}/m{m}/{r}/tv_pos" in save
                        and np.isfinite(save[f"{c}/m{m}/{r}/tv_pos"][i]).any()]
                if have:
                    cols = "".join(f"  {r} {C.ms(save[f'{c}/m{m}/{r}/tv_pos'][i])}" for r in have)
                    wall = np.nanmean(np.concatenate([save[f"{c}/m{m}/{r}/wall"][i] for r in have]))
                    run(f"  {c:12s} N={n:5d} m={m:5d}{cols}  wall {wall:.0f} s")
    frames = {}
    for c in cases:
        frames.update(C.setting(c).record(f"{c}/"))
        frames[f"{c}/ms"] = np.array(ms[c])
    run.save(cases=np.array(cases), ns=np.array(ns), sets=np.array(sets), times=P.times,
             reg=a.reg, rows=np.array(rows), decades=np.array(config.LAM_DECADES[a.reg][dim]),
             lam0_rel=config.LAM_FLOOR_REL, **frames, **save)
    run.close()


if __name__ == "__main__":
    main()
