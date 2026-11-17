"""Fourth-order 2D KDE baselines.
Paper: Table 1 (tab:main), Table 3 (tab:twod-full), Figure 4 (fig:twod-ops); clipped mass of
Appendix B.

Target `--target T` (config.TWOD_TARGETS), training sets of twod.py in the original
coordinates (no frame, no latent, no mu), N in config.NS, config.SETS sets. Kernels
(smoothfm.baselines.kde4):
  radial (primary)  K4(u) = (2 - |u|^2/2) phi_2(u) = phi_2 - Laplacian phi_2 / 2;
  product           k4(u_1) k4(u_2);
  anisotropic       K4_S(r) = (2 - r^T S^-1 r / 2) N(r; 0, S), S_i = h_T^2 e_i e_i^T +
                    h_N^2 n_i n_i^T with the exact curve tangent (targets with 'aniso').
Rules config.KDE4_2D_RULES:
  k4_oracle, k4p_oracle   h = argmin grid TV over the target's bandwidth candidates hs;
  k4_lscv, k4p_lscv       h = argmin closed-form LSCV (lscv4_curve_2d) over hs;
  k4_ani_oracle           (h_T, h_N) by lattice pattern search on the grid TV over the lattice of
                          config.TWOD_ANISO_LATTICE, started at the lattice point nearest the
                          k4_oracle h on both axes; moves to the best of the (up to 8) neighbours
                          while it improves (a lattice local minimum);
  k4_ani_lscv             the same search on the closed-form anisotropic LSCV (lscv4_aniso).
Every estimate is clipped and renormalized on the KDE grid, f_+ = max(f, 0)/(dx^2 sum max(f, 0))
(isotropic kernels by binned FFT, the anisotropic one by direct summation within
r^T S^-1 r <= KDE4_2D_ANISO_CUT^2). Per rule: tv_grid (as kde2d.py, se 0), tv_pos over
KDE2D_DRAWS exact draws of the bilinear interpolant of f_+ (GridDensity2D, exact log density;
rng SEEDS['kde_draws'] + set shared by the rules in order; a skipped rule advances the stream)
with its Monte Carlo SE, neg_mass (the clipped mass of the raw estimate); for the
representative set per N (lower median of the hold-out tv_pos of twod_<T>) the image f_+ on the figure grid and the first TWOD_EVAL['cloud'] draws.
Smoke: thinned bandwidth candidates and figure grid (config.TWOD_SMOKE_STRIDE).

Result kde4_2d_<T>; reads twod_<T>. npz keys (arrays (len(ns), len(sets), ...)):
<rule>/{h (2), tv_grid, tv_grid_se, tv_pos, tv_pos_se, neg_mass, wall}; <rule>/image
(len(ns), G, G) float32, <rule>/cloud (len(ns), cloud, 2) float32 (rep set); curve_{k4_tv,
k4_lscv, k4p_tv, k4p_lscv} (len(hs),), curve_ani_{oracle, lscv}_table (len(pairs), 3) (evaluated
(h_T, h_N, score), NaN-padded); wall_{k4, k4p, ani_oracle, ani_lscv}, wall; ns, sets, hs,
lattice, xg, xf2, rep_set, target, aniso.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import config, tv_pos
from smoothfm import baselines as B
from common2d import Twod, grid_points, grid_tv, pdf_grid, rep_sets

ABOUT = """{name}: fourth-order KDE baselines of {target}: {rules}.
Each job: one N and set -> grid-TV and LSCV curves of the radial and product K4 over {nh}
bandwidths{ani}, the selected h, {draws} exact draws per estimate: tv_pos with its SE, grid TV,
clipped mass; image and cloud of the representative set (median hold-out set of {src}).
Read by the report (tab_twod_full, tab_main, fig_twod_ops).
Outputs: {out}.npz, {out}.log (this file), {out}.manifest.json."""


def pattern_search(score, start, H, nmax_n):
    """Lattice pattern search over (i_T, i_N), 0 <= i_N <= min(i_T, nmax_n), i_T < H.size.
    Returns ((i_T, i_N), table of evaluated (h_T, h_N, score))."""
    cache = {}

    def f(p):
        if p not in cache:
            cache[p] = float(score(H[p[0]], H[p[1]]))
        return cache[p]

    def ok(p):
        return 0 <= p[1] <= min(p[0], nmax_n) and p[0] < H.size

    cur = start
    while True:
        nb = [p for p in ((cur[0] + a, cur[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                          if (a, b) != (0, 0)) if ok(p)]
        best = min(nb, key=f)
        if f(best) >= f(cur):
            break
        cur = best
    return cur, np.array([(H[p[0]], H[p[1]], v) for p, v in cache.items()])


def kde4_task(job):
    name, n, s, rep, smoke = job
    t0 = time.time()
    st = C.setting(name)
    tw = Twod(name, smoke)
    T = st.base
    ys = st.training_set(n, s)[0]
    pg = pdf_grid(T, tw.xg)
    HS, H = tw.hs, tw.H
    M = config.KDE2D_DRAWS
    npair = len(tw.pairs)

    def tvg(f):
        return grid_tv(f, pg, tw.dx, tw.outside)

    def grid(h, kern):
        return B.kde4_grid_2d(ys, h, tw.xg, kernel=kern, return_clipped=True,
                              nfft=tw.nfft)

    curves, sel, timing = {}, {}, {}
    for kern, lab in (("radial", "k4"), ("product", "k4p")):
        t1 = time.time()
        curves[f"{lab}_tv"] = np.array([tvg(grid(h, kern)[0]) for h in HS])
        curves[f"{lab}_lscv"] = B.lscv4_curve_2d(ys, HS, kernel=kern)
        sel[f"{lab}_oracle"] = ("iso", kern, HS[np.argmin(curves[f"{lab}_tv"])])
        sel[f"{lab}_lscv"] = ("iso", kern, HS[np.argmin(curves[f"{lab}_lscv"])])
        timing[f"wall_{lab}"] = time.time() - t1
    e = T.tangent(ys) if tw.aniso else None

    def ani(ht, hn):
        return B.kde4_aniso_grid_2d(ys, B.aniso_covs(e, ht, hn), tw.xg,
                                    cut=config.KDE4_2D_ANISO_CUT, return_clipped=True)

    for crit in ("oracle", "lscv"):
        pad = np.full((npair, 3), np.nan)
        if tw.aniso:
            i0 = int(np.argmin(np.abs(np.log(H / HS[np.argmin(curves["k4_tv"])]))))
            t1 = time.time()
            score = ((lambda ht, hn: tvg(ani(ht, hn)[0])) if crit == "oracle"
                     else (lambda ht, hn: B.lscv4_aniso(ys, B.aniso_covs(e, ht, hn))))
            p, tab = pattern_search(score, (i0, min(i0, tw.nmax_n)), H, tw.nmax_n)
            pad[:len(tab)] = tab
            sel[f"k4_ani_{crit}"] = ("ani", None, (H[p[0]], H[p[1]]))
            timing[f"wall_ani_{crit}"] = time.time() - t1
        curves[f"ani_{crit}_table"] = pad
    rng = np.random.default_rng(config.SEEDS["kde_draws"] + s)
    res = dict(curves=curves, timing=timing, rules=[r for r in config.KDE4_2D_RULES if r in sel])
    E = config.TWOD_EVAL
    for r in config.KDE4_2D_RULES:
        if r not in sel:                        # anisotropic without a tangent: keep the stream
            rng.uniform(size=(M, 3))
            continue
        t1 = time.time()
        kind, kern, h = sel[r]
        f, neg = ani(*h) if kind == "ani" else grid(h, kern)
        g = B.GridDensity2D(tw.xg, f)
        z = g.sample(M, rng)
        tv, se = tv_pos(T, z, g.logpdf(z))
        hh = np.full(2, np.nan)
        hh[:np.size(h)] = np.atleast_1d(h)
        o = dict(h=hh, tv_grid=tvg(f), tv_grid_se=0.0, tv_pos=tv, tv_pos_se=se,
                 neg_mass=neg)
        if rep:
            o["image"] = g.pdf(grid_points(tw.xf2)).astype(np.float32)
            o["cloud"] = z[:E["cloud"]].astype(np.float32)
        o["wall"] = time.time() - t1
        res[r] = o
    res["wall"] = time.time() - t0
    return res


def main():
    p = C.parser(__doc__)
    p.add_argument("--target", choices=C.TARGETS_2D, default="circle", help="2D target")
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 2)
    name = f"kde4_2d_{a.target}"
    src_name = f"twod_{a.target}"
    src = C.load(a, src_name)
    st = C.setting(a.target)
    tw = Twod(a.target, a.smoke)
    rules = [r for r in config.KDE4_2D_RULES if tw.aniso or "ani" not in r]
    src_path = C.rel(C.result_path(a, src_name, deps=True))
    run = C.open_run(name, a, {"target": a.target, "rules": rules, **tw.describe(),
                               "aniso cut": config.KDE4_2D_ANISO_CUT, "source": src_path,
                               "draws": f"{config.KDE2D_DRAWS}, rng {config.SEEDS['kde_draws']} + set"},
                     ABOUT.format(name=name, target=a.target, rules=", ".join(rules),
                                  nh=tw.hs.size, draws=config.KDE2D_DRAWS, src=src_path,
                                  ani=" and the anisotropic pattern searches" if tw.aniso else "",
                                  out=C.rel(C.result_path(a, name, ""))))
    rep = rep_sets(src, P.ns, a.smoke)
    n0, s0 = P.ns[0], P.sets[0]
    run.example({"job": f"{a.target} N={n0} set {s0} (all rules)", **st.describe(), "N": n0,
                 "set": s0, "training seed": f"default_rng({config.SEEDS['train'][a.target]} * "
                                             f"{n0} + {s0})",
                 "rules": rules, **tw.describe(), "representative sets": rep,
                 "coordinates": "original (KDEs do not use the frame or the latent)"})
    run(f"representative sets (median hold-out tv_pos of {src_path}): {rep}")
    if not tw.aniso:
        run(f"anisotropic rules k4_ani_oracle, k4_ani_lscv skipped: {a.target} has no curve "
            "tangent (stored as NaN)")
    jobs = [(a.target, n, s, s == rep[n], a.smoke) for n in P.ns[::-1] for s in P.sets]
    if a.mode == "example":
        jobs = [(a.target, n0, s0, s0 == rep[n0], a.smoke)]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))
    pics = {}

    def done(job, res):
        _, n, s, _, _ = job
        j = ns.index(n)
        idx = (j, sets.index(s))
        for k, v in res["curves"].items():
            save.put(f"curve_{k}", idx, v)
        for k, v in res["timing"].items():
            save.put(k, idx, v)
        save.put("wall", idx, res["wall"])
        m = {}
        for r in res["rules"]:
            o = dict(res[r])
            for key in ("image", "cloud"):
                if key in o:
                    v = o.pop(key)
                    arr = pics.setdefault(f"{r}/{key}", np.full((len(ns),) + v.shape, np.nan,
                                                                np.float32))
                    arr[j] = v
            save.put_record(r, idx, o)
            m |= {f"{r}/tv_pos": o["tv_pos"], f"{r}/tv_pos_se": o["tv_pos_se"],
                  f"{r}/tv_grid": o["tv_grid"], f"{r}/tv_grid_se": 0.0,
                  f"{r}/neg_mass": o["neg_mass"],
                  f"{r}/h": o["h"][np.isfinite(o["h"])].tolist()}
        run.metrics(f"{a.target} N={n} set {s}", m | {"wall": round(res["wall"], 2)})

    C.pool_map(kde4_task, jobs, a.workers, run, done)

    run("summary: mean (SE over sets) [min, max]: tv_pos, grid TV, clipped mass")
    for i, n in enumerate(ns):
        for r in rules:
            def q(k):
                return C.ms(save[f"{r}/{k}"][i])
            run(f"  N={n:5d} {r:14s} tv_pos {q('tv_pos')}  grid {q('tv_grid')}  "
                f"neg {q('neg_mass')}  h {np.round(np.nanmean(save[f'{r}/h'][i], 0), 4)}")
    for r in rules:
        h = save[f"{r}/h"][..., 0]
        lo, hi = (tw.H[0], tw.H[-1]) if "ani" in r else (tw.hs[0], tw.hs[-1])
        run(f"  h saturation {r}: at smallest {np.mean(h <= lo):.0%}, at largest {np.mean(h >= hi):.0%}")
    run.save(ns=np.array(ns), sets=np.array(sets), hs=tw.hs, lattice=tw.H, xg=tw.xg, xf2=tw.xf2,
             rep_set=np.array([rep[n] for n in ns]), target=a.target, aniso=tw.aniso,
             **save, **pics)
    run.close()


if __name__ == "__main__":
    main()
