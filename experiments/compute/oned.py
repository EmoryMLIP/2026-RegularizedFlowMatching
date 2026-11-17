"""1D flows with data-driven lam(t) and the KDE baselines.
Paper: Table 1 (tab:main), Table 2 (tab:oned-full), Table 10 (tab:reg-full, --reg h1 / l2);
Sec. 5.3, 5.6, 5.7; lam(t) of Figure 1 (fig:hero).

Targets trimodal and inhom (config.DATA_BOX, frame onto [-5, 5]), N in config.NS, config.SETS
sets. Flow: cubic splines (config.M_SPLINE[1] cells), K = A + mu M + lam P, lam(t) at the 49
config.TIMES by the rules config.ONED_RULES (oracle against the population velocity at 2048
Sobol points; hold-out 80/20); grid
config.LAM_DECADES[reg][1] relative to lam_s(t). Generation: RK4, NSTEP steps to T_END
from 3000 truncated-latent quantile midpoints; tv_pos with its Monte Carlo SE.
KDE (h2 run only): Silverman, Sheather-Jones, LSCV, Abramson-LSCV, oracle fixed and
Abramson h (grid TV); TV on the grid and tv_pos from 3000 draws (rng SEEDS['kde_draws'] + set).
--reg h1 | l2: the ablation (the same rules, no KDE), result oned_<reg>.

npz keys (arrays (len(ns), len(sets), ...)): <d>/<rule>/{tv_pos, tv_pos_se, wall_fit,
wall_gen, frac_out, frac_out_any, nfev, max_abs} and per time {lam, lam_s, lam_rel, at_limit,
scores, limit_scores, nsolves, nevals, residual, wall_t} (K,); per job the selection history
<d>/<rule>/N<n>_set<s>/{tables (K, k, 2), at_limit, nsolves, nevals, residual, scores,
limit_scores, lam, lam_s, lam_rel, wall_t};
<d>/kde_<r>/{h, tv_grid, tv_grid_se (0), tv_pos, tv_pos_se, wall}; ns, sets, times, reg, decades, frame_c,
frame_s, frame_L.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import config, tv_pos
from smoothfm import baselines as B
from common import HoldOutRule, OracleRule, population

ABOUT = """oned: 1D spline flows (trimodal, inhom) with lam(t) selected per time by {rules},
penalty {reg}{kde}.
Each flow job: one target, N, training set and rule -> lam(t_k) at the selection times (the
history below lists the score of every candidate lam/lam_s, the selection, factorizations,
residual), then RK4 sampling and tv_pos with its Monte Carlo SE.
Outputs: {out}.npz (read by oned_eval, gabriel, tend, integration_error, hero_1d,
gabriel_hero, lam_sweep, stability and the report: tab_main, tab_oned_full, tab_reg_full),
{out}.log (this file), {out}.manifest.json (provenance, cost, output checksums)."""


def flow_task(job):
    name, n, s, rule, reg, P, pop = job
    st = C.setting(name)
    _, ys = st.training_set(n, s)
    if rule == "oracle":
        factory = OracleRule(pop[0], pop[1], P.times, pop[2])
    else:
        factory = HoldOutRule(*st.split(ys, s))
    t0 = time.time()
    v = st.velocity(ys, P.times, P.m, reg).fit(factory, C.search(reg, 1))
    rec = C.velocity_record(v, time.time() - t0)
    rec.update(C.flow_record(C.generate(v, st.latents(P.m_lat), st.std, P.nstep)))
    return rec


def kde_task(job):
    name, n, s, P = job
    st = C.setting(name)
    ys = st.training_set(n, s)[0][:, 0]
    K1 = config.KDE_1D
    dx = K1["dx"]
    xg = np.arange(-K1["xlim"], K1["xlim"] + dx / 2, dx)
    hs = np.geomspace(*K1["hs"][name])
    pg = st.base.pdf(np.clip(xg, -1, 1)[:, None]) * (np.abs(xg) <= 1)

    def tvg(f):
        return 0.5 * float(np.sum(np.abs(f - pg)) * dx)

    t0 = time.time()
    lam = B.abramson_factors(ys, B.sheather_jones(ys))
    tv_fix = np.array([tvg(B.kde_grid_1d(ys, h, xg)) for h in hs])
    tv_abr = np.array([tvg(B.kde_grid_1d(ys, h, xg, lam)) for h in hs])
    h = dict(silverman=B.silverman(ys), sj=B.sheather_jones(ys),
             lscv=hs[np.argmin(B.lscv_curve(ys, hs))],
             abr_lscv=hs[np.argmin(B.lscv_curve(ys, hs, lam))],
             fix_oracle=hs[np.argmin(tv_fix)], abr_oracle=hs[np.argmin(tv_abr)])
    res = {}
    for r in config.ONED_KDE_RULES:
        abr = r.startswith("abr")
        k = B.KDE(ys, bandwidth=h[r] * lam if abr else h[r])
        z = k.sample(P.m_lat, np.random.default_rng(config.SEEDS["kde_draws"] + s))
        tv, se = tv_pos(st.base, z, k.logpdf(z))
        # tv_grid is a deterministic quadrature: se 0
        res[r] = dict(h=h[r], tv_grid=tvg(B.kde_grid_1d(ys, h[r], xg, lam if abr else None)),
                      tv_grid_se=0.0, tv_pos=tv, tv_pos_se=se)
    wall = time.time() - t0
    for r in res:
        res[r]["wall"] = wall
    return res


def main():
    p = C.parser(__doc__)
    p.add_argument("--reg", choices=("h2", "h1", "l2"), default="h2",
                   help="penalty: h2 (production), h1 / l2 (ablations, result oned_<reg>)")
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 1)
    ablation = a.reg != "h2"
    name = "oned" + (f"_{a.reg}" if ablation else "")
    rules = config.ONED_RULES
    kde_rules = () if ablation else config.ONED_KDE_RULES
    run = C.open_run(name, a, {"targets": C.TARGETS_1D, "rules": rules, "kde_rules": kde_rules,
                               "reg": a.reg, "lam grid": C.search_text(a.reg, 1),
                               "latents": f"{P.m_lat} truncated-latent quantile midpoints"},
                     ABOUT.format(rules=", ".join(rules), reg=a.reg,
                                  out=C.rel(C.result_path(a, name, "")),
                                  kde="" if ablation else ", and the KDE baselines "
                                  + ", ".join(kde_rules)))
    first = ("trimodal", P.ns[0], P.sets[0], "holdout")
    st0 = C.setting("trimodal")
    ys0 = st0.training_set(first[1], first[2])[1]
    C.worked_example(run, st0, P, first[1], first[2], "holdout", a.reg,
                     split_sizes=[len(x) for x in st0.split(ys0, first[2])],
                     extra={"oracle points": f"{P.m_oracle} Sobol pairs, rng "
                                             f"{config.SEEDS['oracle_1d']}"})
    names = C.TARGETS_1D
    if a.mode == "example":
        names, rules, kde_rules = ("trimodal",), ("holdout",), ()
    pops = {d: population(C.setting(d), P.times, P.m_oracle) if "oracle" in rules else None
            for d in names}
    jobs = [(d, n, s, r, a.reg, P, pops[d]) for n in P.ns[::-1] for d in names for r in rules
            for s in P.sets]
    if a.mode == "example":
        jobs = [first + (a.reg, P, None)]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))

    def flow_done(job, rec):
        d, n, s, r = job[:4]
        jl = f"{d} N={n} set {s} {r}"
        C.log_velocity(run, jl, r, rec, P.times)
        C.log_flow(run, jl, rec, {"fit": rec["wall_fit"], "generation": rec["wall_gen"]})
        save.put_record(f"{d}/{r}", (ns.index(n), sets.index(s)), rec)

    C.pool_map(flow_task, jobs, a.workers, run, flow_done)

    def kde_done(job, res):
        d, n, s = job[:3]
        for r in kde_rules:
            save.put_record(f"{d}/kde_{r}", (ns.index(n), sets.index(s)), res[r])
        run.metrics(f"{d} N={n} set {s} kde",
                    {f"{r}/{k}": res[r][k] for r in kde_rules
                     for k in ("tv_pos", "tv_pos_se", "tv_grid", "tv_grid_se")}
                    | {f"{r}/h": res[r]["h"] for r in kde_rules}
                    | {"wall": round(res[kde_rules[0]]["wall"], 2)})

    kjobs = [(d, n, s, P) for n in P.ns[::-1] for d in names for s in P.sets] if kde_rules else []
    C.pool_map(kde_task, kjobs, a.workers, run, kde_done)

    run("summary: tv_pos mean (SE over sets) [min, max]")
    for d in names:
        for i, n in enumerate(ns):
            for r in rules:
                run(f"  {d} N={n} flow {r:8s} {C.ms(save[f'{d}/{r}/tv_pos'][i])}  fit "
                    f"{np.nanmean(save[f'{d}/{r}/wall_fit'][i]):.1f} s  gen "
                    f"{np.nanmean(save[f'{d}/{r}/wall_gen'][i]):.1f} s")
            for r in kde_rules:
                run(f"  {d} N={n} kde  {r:10s} {C.ms(save[f'{d}/kde_{r}/tv_pos'][i])}  grid "
                    f"{C.ms(save[f'{d}/kde_{r}/tv_grid'][i])}")
    run.save(ns=np.array(ns), sets=np.array(sets), times=P.times, reg=a.reg,
             decades=np.array(config.LAM_DECADES[a.reg][1]), **st0.record(), **save)
    run.close()


if __name__ == "__main__":
    main()
