"""2D spline flows with oracle and hold-out lam(t).
Paper: Table 1 (tab:main), Table 3 (tab:twod-full), Table 10 (tab:reg-full, --reg h1 / l2),
Figure 4 (fig:twod-ops, through twod_eval); Sec. 5.3, 5.6, 5.7.

Targets config.TWOD_TARGETS (circle, checkerboard, spirals, gauss8; `--target T`), each in its
frame (config.DATA_BOX onto [-4.5, 4.5]^2, c = 0), N in config.NS, config.SETS sets (training
sets rng SEEDS['train'][T] N + set). Flow: cubic splines with m cells per axis
(config.M_SPLINE[2] = 128), K = A + mu M + lam P,
lam(t) at the 49 config.TIMES by config.TWOD_RULES: oracle (velocity error against the
population velocity of the target in the frame at 4096 points x = (1 - t) z + t y',
z truncated-latent draws from_normal of rng SEEDS['oracle_2d'], y' target draws of the same
rng) and hold-out (80/20, rng SEEDS['split'] + set); grid config.LAM_DECADES[reg][2] relative
to lam_s(t) plus the floor and the half-decade refinement. Generation: RK4 with NSTEP
steps in tau to T_END (refit mode) from 4096 scrambled Sobol latents (rng SEEDS['sobol_2d'])
stacked on 4096 iid latents (rng SEEDS['iid'] + set), both through the truncated latent's
inverse CDF; tv_pos (Sobol half, the production estimate) and tv_iid (iid half) with their
Monte Carlo SE, measured in the frame. e(t_k) = mean |v(x, t_k) - v*(x, t_k)|^2 at the
oracle points from the fit at t_k (frame units).
--reg h1 | l2: the ablation penalty (result twod_T_<reg>). The generated points are not
stored: twod_eval regenerates every flow from the stored lam(t).

Result names: twod_<T>[_<reg>] (twod_circle, twod_circle_h1, ...).
npz keys (arrays (len(ns), len(sets), ...)), per rule r in TWOD_RULES:
  r/{tv_pos, tv_pos_se, tv_iid, tv_iid_se, wall_fit, wall_gen, frac_out, frac_out_any, nfev,
  max_abs}; per time (K,): r/{lam, lam_s, lam_rel, e, at_limit, scores, limit_scores, nsolves,
  nevals, residual, wall_t}; per job the selection history
  r/N<n>_set<s>/{tables (K, k, 2), at_limit, nsolves, nevals, residual, scores, limit_scores};
top level: ns, sets, times, m, reg, decades, target, box (data box), frame_c, frame_s, frame_L.
r/lam_s is lam_s(t) of the oracle's all-sample system and of the hold-out fit part.

The 2D helpers this script shares with kde2d, kde4_2d and twod_eval (target settings of
config.TWOD_TARGETS, grid TV, representative sets, e(t)) live in common2d.py.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import config
from common import HoldOutRule, OracleRule, population
from common2d import split_tv, stacked_latents, velocity_errors

ABOUT = """{name}: 2D spline flows of {target}, lam(t) per time by {rules}, penalty {reg}, m = {m}.
Each job: one N, set and rule -> lam(t_k) (history: score of every candidate lam/lam_s, selection,
factorizations, residual), refit, e(t_k), RK4 from {mlat} Sobol + {mlat} iid truncated latents,
tv_pos / tv_iid with their Monte Carlo SE. Read by kde4_2d, twod_eval, lam_sweep, stability and
the report (tab_main, tab_twod_full, tab_reg_full).
Outputs: {out}.npz, {out}.log (this file), {out}.manifest.json."""


# ------------------------------------------------------------------ rules and jobs

def flow_task(job):
    name, n, s, rule, reg, m, P, pop, tworkers = job
    st = C.setting(name)
    _, ys = st.training_set(n, s)
    factory = OracleRule(pop[0], pop[1], P.times, pop[2]) if rule == "oracle" \
        else HoldOutRule(*st.split(ys, s))
    t0 = time.time()
    v = st.velocity(ys, P.times, m, reg).fit(factory, C.search(reg, 2), workers=tworkers)
    rec = C.velocity_record(v, time.time() - t0)
    if pop is not None:
        rec["e"] = velocity_errors(v, pop)
    g = C.generate(v, stacked_latents(st, P.m_lat, s), st.std, P.nstep, workers=tworkers)
    rec.update(split_tv(st, g, P.m_lat))
    rec.update(C.flow_record(g))
    return rec


def result_name(target, reg):
    return f"twod_{target}" + (f"_{reg}" if reg != "h2" else "")


def main():
    p = C.parser(__doc__)
    p.add_argument("--target", choices=C.TARGETS_2D, default="circle", help="2D target")
    p.add_argument("--reg", choices=("h2", "h1", "l2"), default="h2",
                   help="penalty: h2 (production), h1 / l2 (ablations, suffix _<reg>)")
    p.add_argument("--tworkers", type=int, default=1,
                   help="per-time worker processes of each job (TimePool)")
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 2)
    m = P.m if a.smoke else config.M_SPLINE[2]
    name = result_name(a.target, a.reg)
    rules = config.TWOD_RULES
    st = C.setting(a.target)
    run = C.open_run(name, a, {"target": a.target, "rules": rules, "reg": a.reg, "m": m,
                               "lam grid": C.search_text(a.reg, 2), "tworkers": a.tworkers,
                               "latents": f"{P.m_lat} Sobol (rng {config.SEEDS['sobol_2d']}) + "
                                          f"{P.m_lat} iid (rng {config.SEEDS['iid']} + set), "
                                          "truncated latent"},
                     ABOUT.format(name=name, target=a.target, rules=", ".join(rules), reg=a.reg,
                                  m=m, mlat=P.m_lat, out=C.rel(C.result_path(a, name, ""))))
    first = (a.target, P.ns[0], P.sets[0], "holdout")
    ys0 = st.training_set(first[1], first[2])[1]
    P.m = m
    C.worked_example(run, st, P, first[1], first[2], "holdout", a.reg,
                     split_sizes=[len(x) for x in st.split(ys0, first[2])],
                     latent_kind=f"{P.m_lat} scrambled Sobol (rng {config.SEEDS['sobol_2d']}) "
                                 f"+ {P.m_lat} iid (rng {config.SEEDS['iid']} + set)",
                     extra={"oracle points": f"{P.m_oracle} draws, rng {config.SEEDS['oracle_2d']}",
                            "data box": st.box.tolist()})
    if a.mode == "example":
        rules = ("holdout",)
    t0 = time.time()
    pop = population(st, P.times, P.m_oracle)
    run(f"population velocity at {P.m_oracle} oracle points x {P.times.size} times: "
        f"{time.time() - t0:.1f} s")
    jobs = [(a.target, n, s, r, a.reg, m, P, pop, a.tworkers) for n in P.ns[::-1] for r in rules
            for s in P.sets]
    if a.mode == "example":
        jobs = [first + (a.reg, m, P, pop, a.tworkers)]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))

    def flow_done(job, rec):
        _, n, s, r = job[:4]
        jl = f"{a.target} N={n} set {s} {r}"
        C.log_velocity(run, jl, r, rec, P.times)
        C.log_flow(run, jl, rec, {"fit": rec["wall_fit"], "selection": float(np.sum(rec["wall_t"])),
                                  "generation": rec["wall_gen"]})
        run.metrics(jl, {"tv_iid": rec["tv_iid"], "tv_iid_se": rec["tv_iid_se"],
                         "e(t_end)": float(rec["e"][-1])})
        save.put_record(r, (ns.index(n), sets.index(s)), rec)

    C.pool_map(flow_task, jobs, a.workers, run, flow_done)

    run("summary: mean (SE over sets) [min, max]")
    for r in rules:
        for i, n in enumerate(ns):
            run(f"  {r:8s} N={n:5d} tv_pos {C.ms(save[f'{r}/tv_pos'][i])}  tv_iid "
                f"{C.ms(save[f'{r}/tv_iid'][i])}  fit {np.nanmean(save[f'{r}/wall_fit'][i]):.1f} s  "
                f"gen {np.nanmean(save[f'{r}/wall_gen'][i]):.1f} s  factorizations/t "
                f"{np.nanmean(save[f'{r}/nsolves'][i]):.1f}")
            med = np.nanmedian(save[f"{r}/lam_rel"][i], 0)
            run(f"  {r:8s} N={n:5d} median log10 lam/lam_s per t: " + " ".join(
                "floor" if np.isclose(x, config.LAM_FLOOR_REL) else f"{np.log10(x):.1f}"
                for x in med))
    if "oracle" in rules:
        dev = np.nanmax(np.abs(save["oracle/e"] - save["oracle/scores"])
                        / np.abs(save["oracle/scores"]))
        run(f"check: e(t_k) vs the selected oracle score, max rel dev {dev:.1e}")
    run.save(ns=np.array(ns), sets=np.array(sets), times=P.times, m=m, reg=a.reg,
             decades=np.array(config.LAM_DECADES[a.reg][2]), target=a.target, box=st.box,
             **st.record(), **save)
    run.close()


if __name__ == "__main__":
    main()
