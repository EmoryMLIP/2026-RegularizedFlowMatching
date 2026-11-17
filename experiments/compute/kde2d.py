"""2D Gaussian KDE baselines on the training sets of twod.py.
Paper: Table 1 (tab:main), Table 3 (tab:twod-full), Figure 4 (fig:twod-ops) KDE column.

Target `--target T` (config.TWOD_TARGETS), N in config.NS, config.SETS sets, training sets
rng SEEDS['train'][T] N + set (the same y as twod.py; KDEs live in the original coordinates).
Rules config.KDE2D_RULES:
  fix_lscv     isotropic h by LSCV over the candidates hs (config.TWOD_TARGETS[T]['hs'])
  fix_oracle   isotropic h = argmin grid TV over hs
  abr_lscv     Abramson h_i = h0 (f_pilot(y_i)/g)^{-1/2}, pilot the fixed-LSCV KDE, h0 by LSCV
  abr_oracle   the same factors, h0 = argmin grid TV
  ani_oracle   Sigma_i = h_T^2 e_i e_i^T + h_N^2 n_i n_i^T with the exact curve tangent e_i of
               the target (circle, spirals; config 'aniso'), (h_T, h_N) = argmin tv_pos over the
               lattice config.TWOD_ANISO_LATTICE (138 pairs), KDE2D_DRAWS draws with common random
               numbers rng SEEDS['kde_crn'] + set
  ani_lscv     the same kernels, (h_T, h_N) by anisotropic LSCV
The anisotropic rules are skipped (NaN, said in the log) for targets without a tangent.
Grid TV on the target's KDE grid (binned FFT, isotropic rules): 0.5 dx^2 sum |f - p| (plus the
mass off the grid for every target but the circle); grid TVs are deterministic (se 0).
tv_pos for every rule from KDE2D_DRAWS draws (rng SEEDS['kde_draws'] + set, shared by the rules
in rule order; a skipped rule advances the stream by its draws) with the exact KDE log density,
with its Monte Carlo SE. Smoke: every config.TWOD_SMOKE_STRIDE['hs']-th candidate.
No flow, no mu, no latent.

Result kde2d_<T>. npz keys (arrays (len(ns), len(sets), ...)): <rule>/{h (2; isotropic rules h
in slot 0, Abramson h0), tv_pos, tv_pos_se, tv_grid, tv_grid_se, wall}; curve_{tv_fix, tv_abr,
cv_fix, cv_abr} (len(hs),), curve_{tv_ani, cv_ani} (len(pairs),), curve_abr_factors (max N,
NaN-padded); wall; ns, sets, hs, pairs, xg, target, aniso, frame_c, frame_s, frame_L.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import config, tv_pos
from smoothfm import baselines as B
from common2d import Twod, grid_tv, pdf_grid

ABOUT = """{name}: Gaussian KDE baselines of the target {target}: {rules}.
Each job: one N and training set -> the selection curves (LSCV and grid TV over the {nh}
bandwidth candidates{ani}), the selected h per rule, then {draws} draws per rule: tv_pos with its
Monte Carlo SE, and the grid TV of the isotropic rules.
Outputs: {out}.npz (read by twod_eval, which regenerates every KDE from the stored h, and the
report: tab_twod_full, tab_main, fig_twod_ops), {out}.log (this file), {out}.manifest.json."""


def kde_task(job):
    name, n, s, smoke = job
    t0 = time.time()
    st = C.setting(name)
    tw = Twod(name, smoke)
    rules = [r for r in config.KDE2D_RULES if tw.aniso or not r.startswith("ani")]
    M = config.KDE2D_DRAWS
    ys = st.training_set(n, s)[0]
    T = st.base
    pg = pdf_grid(T, tw.xg)
    HS = tw.hs

    def tvg(f):
        return grid_tv(f, pg, tw.dx, tw.outside)

    def kg(h, lam=None):
        return B.kde_grid_2d(ys, h, tw.xg, lam, nfft=tw.nfft)

    cv_fix = B.lscv_curve(ys, HS)
    tv_fix = np.array([tvg(kg(h)) for h in HS])
    lam = B.abramson_factors(ys, HS[np.argmin(cv_fix)])
    cv_abr = B.lscv_curve(ys, HS, lam)
    tv_abr = np.array([tvg(kg(h, lam)) for h in HS])
    sel = {"fix_lscv": ("fix", HS[np.argmin(cv_fix)]), "fix_oracle": ("fix", HS[np.argmin(tv_fix)]),
           "abr_lscv": ("abr", HS[np.argmin(cv_abr)]), "abr_oracle": ("abr", HS[np.argmin(tv_abr)])}
    npair = len(tw.pairs)
    tv_ani, cv_ani = np.full(npair, np.nan), np.full(npair, np.nan)
    e = None
    if tw.aniso:
        e = T.tangent(ys)
        nrm = np.stack([-e[:, 1], e[:, 0]], 1)
        rng = np.random.default_rng(config.SEEDS["kde_crn"] + s)
        k, xi = rng.integers(0, n, M), rng.standard_normal((M, 2))
        for i, (ht, hn) in enumerate(tw.pairs):
            kde = B.KDE(ys, cov=B.aniso_covs(e, ht, hn))
            z = ys[k] + ht * xi[:, :1] * e[k] + hn * xi[:, 1:] * nrm[k]
            tv_ani[i] = tv_pos(T, z, kde.logpdf(z))[0]
            cv_ani[i] = B.aniso_lscv(ys, kde.cov)
        sel["ani_oracle"] = ("ani", tw.pairs[int(np.argmin(tv_ani))])
        sel["ani_lscv"] = ("ani", tw.pairs[int(np.argmin(cv_ani))])
    t_sel = time.time() - t0
    res = {"curves": dict(tv_fix=tv_fix, tv_abr=tv_abr, cv_fix=cv_fix, cv_abr=cv_abr,
                          tv_ani=tv_ani, cv_ani=cv_ani, abr_factors=lam)}
    rng = np.random.default_rng(config.SEEDS["kde_draws"] + s)
    for r in config.KDE2D_RULES:
        if r not in sel:                      # anisotropic without a tangent: keep the stream
            rng.integers(0, n, M)
            rng.standard_normal((M, 2))
            continue
        t1 = time.time()
        kind, h = sel[r]
        if kind == "ani":
            kde, grid = B.KDE(ys, cov=B.aniso_covs(e, *h)), np.nan
        else:
            kde = B.KDE(ys, bandwidth=h * (lam if kind == "abr" else 1.0))
            grid = tvg(kg(h, lam if kind == "abr" else None))
        z = kde.sample(M, rng)
        tv, se = tv_pos(T, z, kde.logpdf(z))
        hh = np.full(2, np.nan)
        hh[:np.size(h)] = np.atleast_1d(h)
        res[r] = dict(h=hh, tv_pos=tv, tv_pos_se=se, tv_grid=grid,
                      tv_grid_se=0.0 if np.isfinite(grid) else np.nan,
                      wall=time.time() - t1)
    res["rules"] = rules
    res["wall_select"] = t_sel
    res["wall"] = time.time() - t0
    return res


def main():
    p = C.parser(__doc__)
    p.add_argument("--target", choices=C.TARGETS_2D, default="circle", help="2D target")
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 2)
    name = f"kde2d_{a.target}"
    st = C.setting(a.target)
    tw = Twod(a.target, a.smoke)
    rules = [r for r in config.KDE2D_RULES if tw.aniso or not r.startswith("ani")]
    run = C.open_run(name, a, {"target": a.target, "rules": rules, **tw.describe(),
                               "draws": f"{config.KDE2D_DRAWS}, rng {config.SEEDS['kde_draws']} + set"
                                        f" (common random numbers of ani_oracle: rng "
                                        f"{config.SEEDS['kde_crn']} + set)"},
                     ABOUT.format(name=name, target=a.target, rules=", ".join(rules),
                                  nh=tw.hs.size, draws=config.KDE2D_DRAWS,
                                  ani=f" and tv_pos / LSCV over {len(tw.pairs)} (h_T, h_N) pairs"
                                  if tw.aniso else "", out=C.rel(C.result_path(a, name, ""))))
    n0, s0 = P.ns[0], P.sets[0]
    run.example({"job": f"{a.target} N={n0} set {s0} (all rules)", **st.describe(),
                 "N": n0, "set": s0,
                 "training seed": f"default_rng({config.SEEDS['train'][a.target]} * {n0} + {s0})",
                 "rules": rules, **tw.describe(),
                 "draws": f"{config.KDE2D_DRAWS} per rule, rng {config.SEEDS['kde_draws']} + {s0}",
                 "coordinates": "original (KDEs do not use the frame or the latent)"})
    if not tw.aniso:
        run(f"anisotropic rules ani_oracle, ani_lscv skipped: {a.target} has no curve tangent "
            "(stored as NaN)")
    jobs = [(a.target, n, s, a.smoke) for n in P.ns[::-1] for s in P.sets]
    if a.mode == "example":
        jobs = [(a.target, n0, s0, a.smoke)]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))

    def done(job, res):
        _, n, s, _ = job
        idx = (ns.index(n), sets.index(s))
        for r in res["rules"]:
            save.put_record(r, idx, res[r])
        for k, v in res["curves"].items():
            save.put(f"curve_{k}", idx, v)
        save.put("wall", idx, res["wall"])
        m = {}
        for r in res["rules"]:
            m |= {f"{r}/tv_pos": res[r]["tv_pos"], f"{r}/tv_pos_se": res[r]["tv_pos_se"],
                  f"{r}/h": res[r]["h"][np.isfinite(res[r]["h"])].tolist()}
            if np.isfinite(res[r]["tv_grid"]):
                m |= {f"{r}/tv_grid": res[r]["tv_grid"], f"{r}/tv_grid_se": 0.0}
        m |= {"wall": round(res["wall"], 2), "wall_select": round(res["wall_select"], 2)}
        run.metrics(f"{a.target} N={n} set {s}", m)

    C.pool_map(kde_task, jobs, a.workers, run, done)

    run("summary: mean (SE over sets) [min, max]")
    for i, n in enumerate(ns):
        for r in rules:
            g = save[f"{r}/tv_grid"][i]
            run(f"  N={n:5d} {r:11s} tv_pos {C.ms(save[f'{r}/tv_pos'][i])}  grid "
                f"{C.ms(g) if np.isfinite(g).any() else '-'}  h "
                f"{np.round(np.nanmean(save[f'{r}/h'][i], 0), 4)}")
    for r in rules:
        h = save[f"{r}/h"][..., 0]
        lo, hi = (tw.H[0], tw.H[-1]) if r.startswith("ani") else (tw.hs[0], tw.hs[-1])
        run(f"  h saturation {r}: at smallest {np.mean(h <= lo):.0%}, at largest {np.mean(h >= hi):.0%}")
    run.save(ns=np.array(ns), sets=np.array(sets), hs=tw.hs, pairs=np.array(tw.pairs), xg=tw.xg,
             target=a.target, aniso=tw.aniso, **st.record(), **save)
    run.close()


if __name__ == "__main__":
    main()
