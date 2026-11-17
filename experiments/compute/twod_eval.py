"""2D evaluation without reselection: regenerate the 2D flows and KDEs from the stored
lam(t) and h and measure them. Analogue of oned_eval.py.
Paper: Table 1 (tab:main), Table 3 (tab:twod-full), Figure 4 (fig:twod-ops).

`--target T [--reg R]`: the flows of twod_<T>[_<reg>] and the KDEs of kde2d_<T>, in the frame
of the source (checked), training sets, latents and draws as the sources. Methods per (N, set):
  oracle, holdout  Velocity(all samples).set_lam(stored lam(t)), refit mode, RK4 NSTEP steps
           to T_END from the Sobol latents stacked on the iid latents (twod.stacked_latents);
  exact_rk the exact empirical field v_N (EmpiricalFlow), the same integrator and latents;
           also tv_law = TV of the stopped law rho_N(., T_END) (StoppedLaw, mixture of
           box-truncated Gaussians) by TWOD_EVAL['m_law'] draws (rng SEEDS['law'] + set), the
           law the unregularized flow approaches; n_outside_support (weight evaluations where
           rho_N = 0) and max |log f - log rho_N(x_end, T_END)| over the Sobol outputs;
  kde_<r>  the kde2d rules from the stored h (Abramson factors from the fixed-LSCV pilot h, the
           exact tangent for the anisotropic rules; rules without h skipped), KDE2D_DRAWS draws
           from rng SEEDS['kde_draws'] + set in kde2d's rule order (h2 run only).
Measured: tv_pos (Sobol outputs of the flows / the KDE draws) and tv_iid (iid outputs) with their
Monte Carlo SE, in the frame for flows (TV is invariant); tv_ref = the TV stored by the
source run (a reproduction check: the log reports max |tv_pos - tv_ref|); e(t_k) at the
oracle points of twod.py (spline flows: the fit at t_k; exact: v_N);
lam_s(t_k); the FlowSummary (fraction leaving Omega, nfev) and walls.
For the representative set per N (lower median hold-out tv_pos of the source; set 0 in smoke):
image = the density on the figure grid (flows: backward_log_density with the forward stage
times, mapped by the frame; KDEs exact), cloud = the first TWOD_EVAL['cloud'] iid outputs /
draws; image_check = max |log f backward - log f forward| at TWOD_EVAL['image_check'] outputs
(a check of the backward density integration behind the image; logged, not reported).
--reg h1 | l2: the holdout group of the ablation (result twod_eval_<T>_<reg>; its image and
tv_pos at the representative sets feed fig_twod_ops).

Result names: twod_eval_<T>[_<reg>]. npz keys (arrays
(len(ns), len(sets), ...)): <T>/<method>/{tv_pos, tv_pos_se, tv_iid, tv_iid_se, tv_ref,
wall, wall_gen, nfev, frac_out, frac_out_any, max_abs} for every method; spline flows and
exact_rk also e (K,), lam (K,), lam_s (K,) (exact: NaN), image_check, wall_image; exact_rk also
tv_law, tv_law_se, n_outside_support, max_logf_dev; kde_<r> also h (2,); <T>/<method>/image
(len(ns), G, G) float32 (oracle, holdout, kde_<r>), <T>/<method>/cloud (len(ns), cloud, 2)
float32; <T>/rep_set, <T>/pdf_true (G, G), <T>/lam_s (oracle's); ns, sets, times, xf2, methods, nstep, m, reg, target, frame_c, frame_s,
frame_L.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import EmpiricalField, EmpiricalFlow, StoppedLaw, config, tv_pos
from smoothfm import baselines as B
from smoothfm.sampler import backward_log_density
from common import population
from common2d import (Twod, grid_points, pdf_grid, rep_sets, stacked_latents, split_tv,
                      velocity_errors)

FLOW_RULES = ("oracle", "holdout")
GROUPS = ("exact", "oracle", "holdout", "kde")
FLOW_KEYS = ("tv_pos", "tv_pos_se", "tv_iid", "tv_iid_se", "wall", "wall_gen", "nfev",
             "frac_out", "frac_out_any", "max_abs", "e", "lam", "lam_s", "image_check",
             "wall_image", "tv_law", "tv_law_se", "n_outside_support", "max_logf_dev", "h")

ABOUT = """{name}: {what}
Each job regenerates one flow or KDE family from the stored selection (no reselection) and
measures tv_pos / tv_iid with their Monte Carlo SE, e(t){extra}.
Sources: {src}.
Outputs: {out}.npz (read by the report: {readers}), {out}.log (this file),
{out}.manifest.json."""


# ------------------------------------------------------------------ measurements

def density_on_grid(v, st, xg, nstep):
    """The density of a frame flow on xg x xg in the original coordinates (ij indexing)."""
    x = grid_points(xg).reshape(-1, 2)
    lp = backward_log_density(v, st.latent, st.frame.to_std(x), nstep=nstep,
                              t_end=config.T_END) - st.frame.log_det
    return np.exp(lp).reshape(xg.size, xg.size)


def flow_measures(st, v, x0, M, nstep, pop):
    """Generation from the stacked latents x0 (M Sobol + M iid): flow record, tv_iid and the
    cloud of the iid outputs in original coordinates, e(t) (spline flows)."""
    g = C.generate(v, x0, st.std, nstep)
    r = split_tv(st, g, M)
    r.update(C.flow_record(g))
    x = st.frame.to_phys(g["z"][M:])
    r["cloud"] = x[:config.TWOD_EVAL["cloud"]].astype(np.float32)
    return g, r


def target_task(job):
    name, n, s, group, src, rep, P, pop, reg, m, smoke = job
    st = C.setting(name)
    tw = Twod(name, smoke)
    ys, yss = st.training_set(n, s)
    M = P.m_lat
    E = config.TWOD_EVAL
    x0 = stacked_latents(st, M, s)
    out = {}
    t0 = time.time()
    if group in FLOW_RULES:
        v = st.velocity(yss, P.times, m, reg)
        ns_src = list(np.asarray(src["twod"]["ns"]))
        sets_src = list(np.asarray(src["twod"]["sets"]))
        v.set_lam(src["twod"][f"{group}/lam"][ns_src.index(n), sets_src.index(s)])
        # keep every stage solve of the forward pass for the backward density image
        v.refit_cache_size = 2 * P.nstep + 2
        g, r = flow_measures(st, v, x0, M, P.nstep, pop)
        r.update(e=velocity_errors(v, pop), lam=v.lam, lam_s=v.lam_s)
        if rep:
            t1 = time.time()
            r["image"] = density_on_grid(v, st, tw.xf2, P.nstep).astype(np.float32)
            idx = np.linspace(0, M - 1, E["image_check"]).astype(int)
            back = backward_log_density(v, st.latent, g["z"][idx], nstep=P.nstep,
                                        t_end=config.T_END)
            r["image_check"] = float(np.max(np.abs(back - g["lf"][idx])))
            r["wall_image"] = time.time() - t1
        r["wall"] = time.time() - t0
        out[group] = r
    elif group == "exact":
        fl = EmpiricalFlow(yss, st.latent)
        g, r = flow_measures(st, fl, x0, M, P.nstep, pop)
        fld = EmpiricalField(yss, st.latent)
        z, y, vpop = pop
        r["e"] = np.array([np.mean(np.sum((fld.velocity((1 - t) * z + t * y, t) - vpop[k]) ** 2, 1))
                           for k, t in enumerate(P.times)])
        law = StoppedLaw(yss, config.T_END, st.latent)
        ml = E["m_law_smoke"] if smoke else E["m_law"]
        zl = law.sample(ml, np.random.default_rng(config.SEEDS["law"] + s))
        r["tv_law"], r["tv_law_se"] = tv_pos(st.std, zl, law.logpdf(zl))
        ll = law.logpdf(g["z"][:M])
        ok = np.isfinite(ll)
        r["max_logf_dev"] = float(np.max(np.abs(g["lf"][:M][ok] - ll[ok]))) if ok.any() else np.nan
        r["n_outside_support"] = fl.n_outside_support
        r["wall"] = time.time() - t0
        out["exact_rk"] = r
    elif group == "kde":
        k = src["kde2d"]
        j = list(np.asarray(k["ns"])).index(n), list(np.asarray(k["sets"])).index(s)
        h = {r: np.asarray(k[f"{r}/h"][j]) for r in config.KDE2D_RULES if f"{r}/h" in k}
        lam = B.abramson_factors(ys, h["fix_lscv"][0])
        e = st.base.tangent(ys) if tw.aniso else None
        D = config.KDE2D_DRAWS
        rng = np.random.default_rng(config.SEEDS["kde_draws"] + s)
        for r in config.KDE2D_RULES:            # kde2d's draw order
            if r not in h or not np.isfinite(h[r]).any():
                rng.integers(0, n, D)
                rng.standard_normal((D, 2))
                continue
            t1 = time.time()
            if r.startswith("ani"):
                kde = B.KDE(ys, cov=B.aniso_covs(e, *h[r]))
            else:
                kde = B.KDE(ys, bandwidth=h[r][0] * (lam if r.startswith("abr") else 1.0))
            zz = kde.sample(D, rng)
            tv, se = tv_pos(st.base, zz, kde.logpdf(zz))
            o = dict(tv_pos=tv, tv_pos_se=se, h=h[r],
                     cloud=zz[:E["cloud"]].astype(np.float32))
            if rep:
                o["image"] = np.exp(kde.logpdf(grid_points(tw.xf2))).astype(np.float32)
            o["wall"] = time.time() - t1
            out[f"kde_{r}"] = o
    return out


def task(job):
    return target_task(job[1:])


# ------------------------------------------------------------------ main

def main():
    p = C.parser(__doc__)
    p.add_argument("--target", choices=C.TARGETS_2D, default="circle", help="2D target")
    p.add_argument("--reg", choices=("h2", "h1", "l2"), default="h2",
                   help="penalty of the flows: h2 (production), h1 / l2 (ablations)")
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 2)
    T, reg = a.target, a.reg
    sfx = "" if reg == "h2" else f"_{reg}"
    name = f"twod_eval_{T}{sfx}"
    names = {"twod": f"twod_{T}{sfx}", "kde2d": f"kde2d_{T}"}
    src = {"twod": C.load(a, names["twod"])}
    if reg == "h2":
        src["kde2d"] = C.load(a, names["kde2d"])
    st = C.setting(T)
    tw = Twod(T, a.smoke)
    C.check_frame(src["twod"], st, C.rel(C.result_path(a, names["twod"], deps=True)))
    if not np.array_equal(np.asarray(src["twod"]["times"]), P.times):
        raise SystemExit(f"{names['twod']}: stored times differ from this run's times")
    m = int(src["twod"]["m"])
    groups = ["oracle", "holdout", "exact", "kde"] if reg == "h2" else ["holdout"]
    groups = [g for g in GROUPS if g in groups]
    paths = {k: C.rel(C.result_path(a, v, deps=True)) for k, v in names.items() if k in src}
    run = C.open_run(name, a, {"target": T, "reg": reg, "groups": groups, "m": m,
                               "sources": paths, **tw.describe()},
                     ABOUT.format(name=name, what=f"the {T} flows ({', '.join(groups)}) "
                                  "regenerated from the stored lam(t) and h.",
                                  extra=", density images and clouds of the representative sets",
                                  src=", ".join(paths.values()),
                                  readers="tab_twod_full, tab_main, tab_integrator, fig_twod_ops",
                                  out=C.rel(C.result_path(a, name, ""))))
    rep = rep_sets(src["twod"], P.ns, a.smoke)
    n0, s0 = P.ns[0], P.sets[0]
    P.m = m
    C.worked_example(run, st, P, n0, s0, "holdout (stored lam)", reg,
                     latent_kind=f"{P.m_lat} Sobol (rng {config.SEEDS['sobol_2d']}) + {P.m_lat} "
                                 f"iid (rng {config.SEEDS['iid']} + set)",
                     extra={"sources": paths, "representative sets": rep})
    t0 = time.time()
    pop = population(st, P.times, P.m_oracle)
    run(f"population velocity at {P.m_oracle} oracle points: {time.time() - t0:.1f} s; "
        f"representative sets {rep}")
    jobs = [("target", T, n, s, g, src, s == rep[n], P, pop, reg, m, a.smoke)
            for n in P.ns[::-1] for g in groups for s in P.sets]
    if a.mode == "example":
        jobs = [("target", T, n0, s0, "holdout", src, s0 == rep[n0], P, pop, reg, m, a.smoke)]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))
    pics = {}
    methods = [g for g in groups if g in FLOW_RULES] + \
        (["exact_rk"] if "exact" in groups else []) + \
        ([f"kde_{r}" for r in config.KDE2D_RULES if tw.aniso or not r.startswith("ani")]
         if "kde" in groups else [])

    def ref_tv(meth, n, s):
        if meth in FLOW_RULES:
            d, key = src["twod"], f"{meth}/tv_pos"
        elif meth.startswith("kde_"):
            d, key = src.get("kde2d"), f"{meth[4:]}/tv_pos"
        else:
            return np.nan
        if d is None or key not in d or n not in list(np.asarray(d["ns"])):
            return np.nan
        return float(np.asarray(d[key])[list(np.asarray(d["ns"])).index(n),
                                        list(np.asarray(d["sets"])).index(s)])

    def done(job, out):
        n, s = job[2], job[3]
        j = ns.index(n)
        idx = (j, sets.index(s))
        jl = f"{T} N={n} set {s} {job[4]}"
        for meth, r in out.items():
            r = dict(r)
            for key in ("image", "cloud"):
                if key in r:
                    v = r.pop(key)
                    pics.setdefault(f"{T}/{meth}/{key}",
                                    np.full((len(ns),) + v.shape, np.nan, np.float32))[j] = v
            r["tv_ref"] = ref_tv(meth, n, s)
            for k in FLOW_KEYS + ("tv_ref",):
                if k in r:
                    save.put(f"{T}/{meth}/{k}", idx, r[k])
            if meth == "oracle":
                save.put(f"{T}/lam_s", idx, r["lam_s"])
            lab = f"{jl} {meth}" if meth.startswith("kde_") else jl
            mets = {}
            if "summary" in r:
                C.log_flow(run, lab, r, {"generation": r["wall_gen"], "total": r["wall"]})
                mets |= {"tv_iid": r["tv_iid"], "tv_iid_se": r["tv_iid_se"],
                         "e(t_end)": float(r["e"][-1])}
                if "lam" in r:
                    run(f"[{lab}] lam/lam_s " + " ".join(f"{x:.3g}" for x in r["lam"] / r["lam_s"]))
            else:
                mets |= {"tv_pos": r["tv_pos"], "tv_pos_se": r["tv_pos_se"]}
            mets |= {"tv_ref": r["tv_ref"], "tv_ref_se": 0.0,
                     "wall": round(r["wall"], 3)}
            for k in ("tv_law", "tv_law_se", "n_outside_support", "max_logf_dev", "image_check"):
                if k in r:
                    mets[k] = r[k]
            run.metrics(lab, mets)

    C.pool_map(task, jobs, a.workers, run, done)

    run("tv_pos reproduction: max |tv_pos - stored| over (N, set) per method")
    dev = {mt: np.nanmax(np.abs(save[f"{T}/{mt}/tv_pos"] - save[f"{T}/{mt}/tv_ref"]))
           for mt in methods if f"{T}/{mt}/tv_ref" in save
           and np.isfinite(save[f"{T}/{mt}/tv_ref"]).any()}
    run("  " + ", ".join(f"{mt} {v:.1e}" for mt, v in dev.items()))
    run(f"{T}: mean (SE over sets) [min, max]")
    for i, n in enumerate(ns):
        run(f"N={n} (representative set {rep[n]})")
        for mt in methods:
            key = f"{T}/{mt}/tv_pos"
            if key not in save:
                continue
            line = f"  {mt:15s} tv_pos {C.ms(save[key][i])}"
            for k in ("tv_iid", "tv_law"):
                if f"{T}/{mt}/{k}" in save:
                    line += f"  {k} {C.ms(save[f'{T}/{mt}/{k}'][i])}"
            run(line)
    run.save(ns=np.array(ns), sets=np.array(sets), times=P.times, xf2=tw.xf2,
             methods=np.array(methods), nstep=P.nstep, m=m, reg=reg, target=T,
             **{f"{T}/rep_set": np.array([rep[n] for n in ns]),
                f"{T}/pdf_true": pdf_grid(st.base, tw.xf2)},
             **st.record(), **save, **pics)
    run.close()


if __name__ == "__main__":
    main()
