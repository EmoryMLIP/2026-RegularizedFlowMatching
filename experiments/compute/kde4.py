"""Fourth-order KDE baseline in 1D.
Paper: Table 1 (tab:main), Table 2 (tab:oned-full); clipped mass of Appendix B.

Kernel K4(u) = (1/2)(3 - u^2) phi(u) (int K4 = 1, int u^2 K4 = 0); the estimate is clipped,
f_+ = max(f_h, 0), and renormalized on the grid (baselines.kde4_grid_1d). Rules
config.KDE4_RULES on the bandwidth candidates config.KDE_1D['hs'] and the training sets of
oned.py:
  fix4_oracle   fixed h = argmin grid TV;     fix4_lscv  fixed h = argmin LSCV of the
                unclipped estimate (closed form, lscv4_curve);
  abr4_oracle   Abramson h_i = h lam_i (lam_i from the Sheather-Jones pilot, as oned.py's
                Abramson rules), h = argmin grid TV;   abr4_lscv  same factors, h by LSCV.
Grid: config.KDE_1D dx on [-xlim, xlim], binned FFT evaluation as for the Gaussian KDEs.
Per rule: grid TV = 0.5 sum |f_+ - p| dx (deterministic, se 0); tv_pos = mean (1 - p/f_+)_+
over M_LAT draws from f_+ (GridDensity1D: exact inversion of the piecewise-quadratic CDF; rng
SEEDS['kde_draws'] + set) with its Monte Carlo SE; the clipped (negative) mass int (f_h)_- dx before
renormalization; the density on the figure grid xf of oned_eval.npz (read, so that the curves
lie on the grid of the flow curves).
KDE only: no flow, no dependence on mu or the latent.

npz keys (arrays (len(ns), len(sets), ...)): <d>/<rule>/{h, tv_grid, tv_grid_se (0), tv_pos,
tv_pos_se, neg_mass, wall, curve (nx)}; <d>/{fix4,abr4}/{tv_curve, lscv_curve} (len(hs));
<d>/hs; ns, sets, xf.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import config, tv_pos
from smoothfm import baselines as B

ABOUT = """kde4: fourth-order (clipped, renormalized) KDEs in 1D, rules {rules}, on the training sets
of oned; per rule h, grid TV, tv_pos with its SE, clipped mass, the density on the figure
grid of {dep}.
Outputs: {out}.npz (read by the report: tab_main, tab_oned_full),
{out}.log (this file), {out}.manifest.json."""


def task(job):
    d, n, s, M, xf = job
    t0 = time.time()
    st = C.setting(d)
    ys = st.training_set(n, s)[0][:, 0]
    K1 = config.KDE_1D
    dx = K1["dx"]
    xg = np.arange(-K1["xlim"], K1["xlim"] + dx / 2, dx)
    hs = np.geomspace(*K1["hs"][d])
    pg = st.base.pdf(np.clip(xg, -1, 1)[:, None]) * (np.abs(xg) <= 1)

    def tvg(f):
        return 0.5 * float(np.sum(np.abs(f - pg)) * dx)

    def grid(h, lm):
        return B.kde4_grid_1d(ys, h, xg, lm, return_clipped=False)

    lam = B.abramson_factors(ys, B.sheather_jones(ys))
    curves = {"fix4": dict(tv=np.array([tvg(grid(h, None)) for h in hs]),
                           lscv=B.lscv4_curve(ys, hs)),
              "abr4": dict(tv=np.array([tvg(grid(h, lam)) for h in hs]),
                           lscv=B.lscv4_curve(ys, hs, lam))}
    res = {"curves": curves, "hs": hs}
    for rule in config.KDE4_RULES:
        kind, sel = rule.split("_")
        lm = lam if kind == "abr4" else None
        h = float(hs[np.argmin(curves[kind]["tv" if sel == "oracle" else "lscv"])])
        fplus, neg = B.kde4_grid_1d(ys, h, xg, lm)
        g = B.GridDensity1D(xg, fplus)
        z = g.sample(M, np.random.default_rng(config.SEEDS["kde_draws"] + s))
        tv, se = tv_pos(st.base, z, g.logpdf(z))
        res[rule] = dict(h=h, tv_grid=tvg(g.f), tv_grid_se=0.0, tv_pos=tv, tv_pos_se=se,
                         neg_mass=neg,
                         curve=g.pdf(xf[:, None]))
    wall = time.time() - t0
    for rule in config.KDE4_RULES:
        res[rule]["wall"] = wall
    return res


def main():
    p = C.parser(__doc__)
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 1)
    name = "kde4"
    ev = C.load(a, "oned_eval")
    xf = np.asarray(ev["xf"], float)
    rules = config.KDE4_RULES
    run = C.open_run(name, a, {"targets": C.TARGETS_1D, "rules": rules,
                               "hs": {d: config.KDE_1D["hs"][d] for d in C.TARGETS_1D},
                               "grid": f"dx {config.KDE_1D['dx']} on +-{config.KDE_1D['xlim']}",
                               "draws": P.m_lat},
                     ABOUT.format(rules=", ".join(rules),
                                  dep=C.rel(C.result_path(a, "oned_eval", deps=True)),
                                  out=C.rel(C.result_path(a, name, ""))))
    st0 = C.setting("trimodal")
    K1 = config.KDE_1D
    run.example({"job": f"trimodal N={P.ns[0]} set {P.sets[0]} rules {', '.join(rules)}",
                 **st0.describe(), "N": P.ns[0], "set": P.sets[0],
                 "training seed": f"default_rng({config.SEEDS['train']['trimodal']} * {P.ns[0]} "
                                  f"+ {P.sets[0]})",
                 "kernel": "K4(u) = (3 - u^2) phi(u) / 2, clipped at 0, renormalized on the grid",
                 "grid": f"dx {K1['dx']} on [-{K1['xlim']}, {K1['xlim']}], binned FFT",
                 "bandwidths": f"geomspace{tuple(K1['hs']['trimodal'])}",
                 "Abramson pilot": "Sheather-Jones",
                 "draws": f"{P.m_lat} from the clipped KDE, rng {config.SEEDS['kde_draws']} + set",
                 "figure grid": "xf of oned_eval.npz"})
    names = C.TARGETS_1D if a.mode != "example" else ("trimodal",)
    jobs = [(d, n, s, P.m_lat, xf) for n in P.ns[::-1] for d in names for s in P.sets]
    if a.mode == "example":
        jobs = [("trimodal", P.ns[0], P.sets[0], P.m_lat, xf)]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))
    hs = {}

    def done(job, res):
        d, n, s = job[:3]
        idx = (ns.index(n), sets.index(s))
        hs[d] = res["hs"]
        for kind, c in res["curves"].items():
            for key, v in c.items():
                save.put(f"{d}/{kind}/{key}_curve", idx, v)
        met = {}
        for r in rules:
            for key, v in res[r].items():
                save.put(f"{d}/{r}/{key}", idx, v)
            met.update({f"{r}/{k}": res[r][k] for k in ("h", "tv_pos", "tv_pos_se", "tv_grid",
                                                        "tv_grid_se", "neg_mass")})
        met["wall"] = round(res[rules[0]]["wall"], 3)
        run.metrics(f"{d} N={n} set {s} kde4", met)

    C.pool_map(task, jobs, a.workers, run, done)
    run("h at the ends of the candidates (fraction of sets):")
    for d in names:
        for r in rules:
            h = save[f"{d}/{r}/h"]
            run(f"  {d:8s} {r:12s} smallest {np.mean(h <= hs[d][0]):.0%}, "
                f"largest {np.mean(h >= hs[d][-1]):.0%}")
    run("summary: mean (SE over sets) [min, max]: tv_pos, grid TV, clipped mass")
    for d in names:
        for i, n in enumerate(ns):
            run(f"{d} N={n}")
            for r in rules:
                def q(k):
                    return C.ms(save[f"{d}/{r}/{k}"][i])
                run(f"  {r:12s} tv_pos {q('tv_pos')}  grid {q('tv_grid')}  "
                    f"neg {q('neg_mass')}")
    run.save(ns=np.array(ns), sets=np.array(sets), xf=xf,
             **{f"{d}/hs": hs[d] for d in names}, **save)
    run.close()


if __name__ == "__main__":
    main()
