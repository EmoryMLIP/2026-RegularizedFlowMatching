"""Sensitivity of the flow to lam(t), and a time-constant lam with its own hold-out choice.
Paper: the hold-out constant row of Table 1 (tab:main), Table 11 (tab:lam-sweep-full);
Sec. 5.7 "Choice of the Regularization Parameter".

One target (`--target`: trimodal, inhom, circle, checkerboard, spirals, gauss8), one penalty
(`--reg` h2, h1, l2), N = config.LAM_SWEEP['n'] or `--n N` (one of config.NS; smoke:
config.SMOKE['N'] whatever --n says), every set of the source. Training sets, 80/20 hold-out splits, frame, spline resolution, latents and
integrator are those of the production runs (oned.py, twod.py): refit mode, RK4 NSTEP steps in
tau to T_END.
Source: the stored hold-out schedule lam_h(t_k) (the 49 config.TIMES) of the same target and
penalty: oned[_<reg>] '<T>/holdout/lam' (1D) or twod_<T>[_<reg>] 'holdout/lam' (2D); frame,
penalty and times checked (common.check_frame).
Schedules (lam at each t_k; refit-mode log-linear interpolation in tau between them):
  scale  lam(t) = s lam_h(t), s = 10^(k/2), k in config.LAM_SWEEP['scale_k'][<T>/<reg>]
         (zeros of lam_h stay zero, i.e. the floor). s = 1 is the production flow: its tv_pos
         must equal the stored one (checked in the log).
  const  lam(t) = c at every t_k, c = c0 10^(k/2), k in config.LAM_SWEEP['const_k'], c0 = median
         of the stored hold-out lam over every set and the t_k >= c0_tmin.
The grids are fixed constants (config.LAM_SWEEP); nothing extends at run time.
Per set and schedule: tv_pos (production latents: 1D M_LAT[1] truncated quantile midpoints,
2D M_LAT[2] Sobol), tv_iid (same integration, M_LAT iid truncated latents, rng SEEDS['iid'] +
set), both with their Monte Carlo SE; e(t_k) = mean |v_lam - v*|^2 at the oracle points of
the production script from the fit at t_k on all samples; the hold-out criterion crit(t_k) =
HoldOut(ne_fit, ne_val).score(lam(t_k)) of the production rule (its argmin over the candidate
grid is the stored hold-out lam(t_k), so at s = 1 crit equals the stored hold-out scores, checked
in 2D). crit_sum = sum_k crit(t_k); the hold-out constant is argmin_c crit_sum (sel_idx, per set).
`--const-only` (the tab_main hold-out constant row at the other N): the scale grid shrinks to
k = 0 (s = 1, which keeps the reproduction check against the stored hold-out flow); the
constant grid is the full one.
Smoke: set 0, s = 1, c in {c0, 10 c0}, the smoke source (--const-only changes nothing there).

Result names: lam_sweep_<T>[_<reg>] (h2 without suffix) at N = config.LAM_SWEEP['n'], and
lam_sweep_<T>[_<reg>]_n<N> for --n N != config.LAM_SWEEP['n'] (also in smoke mode, so that
every line of experiments/runs.txt has its own smoke result).
npz keys (sets S, scale values Ns, constants Nc, K times): times (K), sets (S), n (the N of the
training sets), const_only, target, reg, source, source_file_sha256, frame_c, frame_s, frame_L;
<T>/scale/{k, s, extended (Ns), lam (S, Ns, K), tv_pos, tv_pos_se, tv_iid, tv_iid_se,
  crit_sum, wall, frac_out, nfev (S, Ns), e, crit (S, Ns, K)};
<T>/const/{k, c, extended (Nc), c0, tv_pos, tv_pos_se, tv_iid, tv_iid_se, crit_sum, wall,
  frac_out, nfev (S, Nc), e, crit (S, Nc, K), sel_idx (S,)};
<T>/holdout_ref/{tv_pos, lam (S, K), and tv_iid, scores (2D)}.
extended marks the k outside the base grids s: -4..4, c: -6..6. wall is the flow wall (fit +
generation + e).
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import hashlib
import time

import numpy as np

from smoothfm import HoldOut, config, tv_pos
from smoothfm.splines import far_clip

ABOUT = """{name}: sensitivity of the {target} flow (penalty {reg}, N = {n}) to lam(t): the
stored hold-out lam_h(t) of {src} scaled by s = 10^(k/2){const_only}, and constant
lam = c0 10^(k/2).
Flow jobs (set, schedule): set_lam, e(t_k), RK4, tv_pos and tv_iid with SE; criterion jobs
(set): the hold-out criterion of every schedule at every t_k.
Outputs: {out}.npz (report: tab_main, tab_lam_sweep_full), .log, .manifest.json."""

LS = config.LAM_SWEEP
TARGETS = C.TARGETS_1D + C.TARGETS_2D
BASE_SCALE_K, BASE_CONST_K = range(-4, 5), range(-6, 7)


def half_decades(ks):
    return 10.0 ** (np.asarray(ks, float) / 2.0)


def result_name(target, reg, n=None) -> str:
    """lam_sweep_<T>[_<reg>], with _n<N> for a requested N other than LAM_SWEEP['n']."""
    sfx = "" if n is None or int(n) == int(LS["n"]) else f"_n{int(n)}"
    return f"lam_sweep_{target}" + ("" if reg == "h2" else f"_{reg}") + sfx


def schedule_k(target, reg, smoke=False, const_only=False) -> tuple:
    """(scale k, const k) of a run: the smoke grids, else the config grids with the scale
    grid reduced to k = 0 under --const-only (s = 1 keeps the reproduction check)."""
    if smoke:
        return list(LS["smoke_scale_k"]), list(LS["smoke_const_k"])
    scale_k = [0] if const_only else list(LS["scale_k"][f"{target}/{reg}"])
    return scale_k, list(LS["const_k"])


def source_name(target, reg):
    suf = "" if reg == "h2" else f"_{reg}"
    return ("oned" if target in C.TARGETS_1D else f"twod_{target}") + suf


def load_source(a, target, reg, n, times):
    """lam_h (S, K), tv_pos (S,) (2D also tv_iid, scores) of the stored hold-out flows at n."""
    name = source_name(target, reg)
    d = C.load(a, name)
    st = C.setting(target)
    C.check_frame(d, st, name)
    if str(d.get("reg", "h2")) != reg:
        raise SystemExit(f"{name}: penalty {d.get('reg')} != {reg}")
    if not np.allclose(np.asarray(d["times"]), times, rtol=0, atol=1e-12):
        raise SystemExit(f"{name}: selection times differ from this run's")
    ns = [int(x) for x in d["ns"]]
    if n not in ns:
        raise SystemExit(f"{name}: N = {n} not stored (N = {ns})")
    j = ns.index(n)
    pre = f"{target}/holdout/" if target in C.TARGETS_1D else "holdout/"
    out = {"lam": d[pre + "lam"][j], "tv_pos": d[pre + "tv_pos"][j],
           "sets": [int(x) for x in d["sets"]], "name": name}
    for k in ("tv_iid", "scores"):
        if pre + k in d:
            out[k] = d[pre + k][j]
    with open(C.result_path(a, name, deps=True), "rb") as f:
        out["sha256"] = hashlib.sha256(f.read()).hexdigest()
    return out


def population(st, times, M):
    """(z, y', v*(x, t_k)) at the oracle points of the production script."""
    z, y = st.oracle_points(M)
    pv = st.population()
    return z, y, np.array([pv((1 - t) * z + t * y, t) for t in times])


def fit_errors(v, pop):
    """e(t_k) from the stored fit at t_k (no refit solve)."""
    z, y, vpop = pop
    deriv = 0 if v.dim == 1 else (0, 0)
    out = []
    for k, t in enumerate(v.times):
        x = (1 - t) * z + t * y
        xc, _ = far_clip(v.basis, x)
        val = np.asarray(v.basis.extended_design(xc, deriv) @ v.coef[k]).reshape(x.shape)
        d = val / (1.0 - t) - vpop[k]
        out.append(float(np.mean(np.sum(d * d, 1))))
    return np.array(out)


def crit_task(job):
    """('crit', target, n, s, reg, P, items): {(kind, k): crit (K,)}."""
    _, target, n, s, reg, P, items = job
    t0 = time.time()
    st = C.setting(target)
    _, ys = st.training_set(n, s)
    yt, yv = st.split(ys, s)
    v = st.velocity(ys, P.times, P.m, reg)
    out = {key: np.empty(P.times.size) for key, _ in items}
    for j, t in enumerate(P.times):
        ne = v.normal_equations(t, yt)
        rule = HoldOut(ne, ne.validation(yv))
        for key, lam in items:
            out[key][j] = rule.score(float(lam[j]))
    return {"crit": out, "wall": time.time() - t0}


def flow_task(job):
    """('flow', target, n, s, reg, P, kind, k, lam, pop): the flow of schedule lam (K,)."""
    _, target, n, s, reg, P, kind, k, lam, pop = job
    t0 = time.time()
    st = C.setting(target)
    _, ys = st.training_set(n, s)
    v = st.velocity(ys, P.times, P.m, reg).set_lam(lam)
    rec = C.velocity_record(v, time.time() - t0)
    rec["e"] = fit_errors(v, pop)
    x0 = np.vstack([st.latents(P.m_lat), st.iid_latents(P.m_lat, s)])
    g = C.generate(v, x0, st.std, P.nstep)
    M = P.m_lat
    rec.update(C.flow_record(g))
    rec["tv_pos"], rec["tv_pos_se"] = tv_pos(st.std, g["z"][:M], g["lf"][:M])
    rec["tv_iid"], rec["tv_iid_se"] = tv_pos(st.std, g["z"][M:], g["lf"][M:])
    rec["wall"] = time.time() - t0
    return rec


def task(job):
    return crit_task(job) if job[0] == "crit" else flow_task(job)


def main():
    p = C.parser(__doc__)
    p.add_argument("--target", choices=TARGETS, required=True, help="target of the sweep")
    p.add_argument("--reg", choices=("h2", "h1", "l2"), default="h2",
                   help="penalty: h2 (production), h1 / l2 (ablations, result lam_sweep_<T>_<reg>)")
    p.add_argument("--n", type=int, choices=tuple(int(x) for x in config.NS), default=None,
                   help=f"training-set size (default config.LAM_SWEEP['n'] = {LS['n']}; another N "
                        "adds _n<N> to the result name; smoke runs use config.SMOKE['N'])")
    p.add_argument("--const-only", action="store_true",
                   help="scale grid reduced to s = 1 (the reproduction check); the full "
                        "constant grid (the tab_main hold-out constant row)")
    a = C.parse(p)
    T, reg = a.target, a.reg
    dim = 1 if T in C.TARGETS_1D else 2
    P = a.protocol = C.Protocol(a, dim)
    n_req = LS["n"] if a.n is None else a.n
    n = config.SMOKE["N"] if a.smoke else n_req
    name = result_name(T, reg, n_req)
    src = load_source(a, T, reg, n, P.times)          # before any heavy work
    src_sets = src["sets"]
    sets = list(P.sets) if a.smoke else src_sets
    lam_all = np.asarray(src["lam"], float)
    tmin = LS["c0_tmin"]
    c0 = float(np.median(lam_all[:, P.times >= tmin]))
    scale_k, const_k = schedule_k(T, reg, a.smoke, a.const_only)
    run = C.open_run(name, a, {"target": T, "reg": reg, "n": n, "const_only": bool(a.const_only),
                               "sets": sets,
                               "scale_k": scale_k, "const_k": const_k, "c0": c0,
                               "c0 rule": f"median stored hold-out lam, t >= {tmin}, all sets",
                               "source": f"{C.rel(C.result_path(a, src['name'], deps=True))} "
                                         f"sha256 {src['sha256'][:16]}",
                               "latents": f"{P.m_lat} production + {P.m_lat} iid truncated latents"},
                     ABOUT.format(name=name, target=T, reg=reg, n=n, src=src["name"],
                                  const_only=" (--const-only: s = 1 only)" if a.const_only else "",
                                  out=C.rel(C.result_path(a, name, ""))))
    st0 = C.setting(T)
    ys0 = st0.training_set(n, sets[0])[1]
    C.worked_example(run, st0, P, n, sets[0], "fixed schedule s lam_h(t) (s = 1)", reg,
                     split_sizes=[len(x) for x in st0.split(ys0, sets[0])],
                     extra={"lam grid": "no selection: lam(t_k) = s lam_h(t_k) or c (sweep)",
                            "c0": f"{c0:.6g}", "scale k": scale_k, "const k": const_k})
    pop = population(st0, P.times, P.m_oracle)
    lam_of = {si: lam_all[src_sets.index(si)] for si in sets}
    K = P.times.size
    sched = {}
    for si in sets:
        for k in scale_k:
            sched[(si, "scale", k)] = half_decades(k) * lam_of[si]
        for k in const_k:
            sched[(si, "const", k)] = np.full(K, c0 * half_decades(k))
    jobs = [("crit", T, n, si, reg, P, [((kd, k), lam) for (s2, kd, k), lam in sched.items()
                                       if s2 == si]) for si in sets]
    jobs += [("flow", T, n, si, reg, P, kd, k, lam, pop) for (si, kd, k), lam in sched.items()]
    if a.mode == "example":
        jobs = [("flow", T, n, sets[0], reg, P, "scale", 0, lam_of[sets[0]], pop)]
    res = {}

    def done(job, r):
        si = job[3]
        if job[0] == "crit":
            for key, c in r["crit"].items():
                res[("crit", si) + key] = c
            run.metrics(f"{T} set {si} crit", {"schedules": len(r["crit"]),
                                                "wall": round(r["wall"], 2)})
            return
        kd, k = job[6], job[7]
        jl = f"{T} set {si} {kd} k={k}"
        res[(si, kd, k)] = r
        C.log_velocity(run, jl, kd, r, P.times)
        C.log_flow(run, jl, r, {"fit": r["wall_fit"], "generation": r["wall_gen"]})
        run.metrics(jl, {"tv_iid": r["tv_iid"], "tv_iid_se": r["tv_iid_se"],
                         "wall": round(r["wall"], 2)})

    C.pool_map(task, jobs, a.workers, run, done)

    save = {"times": P.times, "sets": np.array(sets), "n": n,
            "const_only": bool(a.const_only), "target": T, "reg": reg,
            "source": src["name"], "source_file_sha256": src["sha256"], **st0.record()}
    pre = f"{T}/"
    for kd, ks, base in (("scale", scale_k, BASE_SCALE_K), ("const", const_k, BASE_CONST_K)):
        q = f"{pre}{kd}/"
        sh = (len(sets), len(ks))
        for key in ("tv_pos", "tv_pos_se", "tv_iid", "tv_iid_se", "wall", "frac_out", "nfev"):
            save[q + key] = np.full(sh, np.nan)
        for key in ("e", "crit"):
            save[q + key] = np.full(sh + (K,), np.nan)
        for i, si in enumerate(sets):
            for j, k in enumerate(ks):
                f = res.get((si, kd, k))
                if f is not None:
                    for key in ("tv_pos", "tv_pos_se", "tv_iid", "tv_iid_se", "wall", "frac_out",
                                "nfev", "e"):
                        save[q + key][i, j] = f[key]
                c = res.get(("crit", si, kd, k))
                if c is not None:
                    save[q + "crit"][i, j] = c
        save[q + "crit_sum"] = save[q + "crit"].sum(axis=2)
        save[q + "k"] = np.array(ks, int)
        save[q + "extended"] = ~np.isin(ks, list(base))
    save[pre + "scale/s"] = half_decades(scale_k)
    save[pre + "scale/lam"] = np.stack([[half_decades(k) * lam_of[si] for k in scale_k]
                                        for si in sets])
    save[pre + "const/c"] = c0 * half_decades(const_k)
    save[pre + "const/c0"] = c0
    cs = save[pre + "const/crit_sum"]
    save[pre + "const/sel_idx"] = np.array([int(np.nanargmin(r)) if np.isfinite(r).any() else -1
                                            for r in cs], int)
    idx = [src_sets.index(si) for si in sets]
    for key in ("tv_pos", "lam", "tv_iid", "scores"):
        if key in src:
            save[f"{pre}holdout_ref/{key}"] = np.asarray(src[key])[idx]
    summary(run, save, pre, scale_k)
    run.save(**save)
    run.close()


def summary(run, save, pre, scale_k):
    sc, co = pre + "scale/", pre + "const/"
    run("summary: tv_pos mean (SE over sets) [min, max]; tv_iid mean; crit_sum mean")
    for a_, k in enumerate(save[sc + "k"]):
        run(f"  scale s = 10^{k / 2:+.1f}{' ext' if save[sc + 'extended'][a_] else ''}  tv_pos "
            f"{C.ms(save[sc + 'tv_pos'][:, a_])}  tv_iid {np.nanmean(save[sc + 'tv_iid'][:, a_]):.4f}"
            f"  crit_sum {np.nanmean(save[sc + 'crit_sum'][:, a_]):.6g}")
    m = np.nanmean(save[sc + "tv_pos"], axis=0)
    if m.size > 1 and np.isfinite(m).any():
        b = int(np.nanargmin(m))
        run(f"best scale: s = 10^{save[sc + 'k'][b] / 2:+.1f}"
            + ("  WARNING: at an end of the s grid" if b in (0, m.size - 1) else ""))
    ref = save.get(pre + "holdout_ref/tv_pos")
    if 0 in scale_k and ref is not None:
        d = np.abs(save[sc + "tv_pos"][:, scale_k.index(0)] - ref)
        run(f"check s = 1 vs stored hold-out tv_pos: max |diff| {np.nanmax(d):.2e} "
            f"({'PASS' if np.nanmax(d) <= 1e-4 else 'FAIL'} at 1e-4)")
        sref = save.get(pre + "holdout_ref/scores")
        if sref is not None:
            cr = save[sc + "crit"][:, scale_k.index(0)]
            run(f"check s = 1 crit(t_k) vs stored hold-out scores: max rel diff "
                f"{np.nanmax(np.abs(cr - sref) / np.abs(sref)):.1e}")
    run(f"c0 = {float(save[co + 'c0']):.4g}")
    for a_, c in enumerate(save[co + "c"]):
        run(f"  const c = c0 10^{save[co + 'k'][a_] / 2:+.1f} = {c:.3g}  tv_pos "
            f"{C.ms(save[co + 'tv_pos'][:, a_])}  tv_iid {np.nanmean(save[co + 'tv_iid'][:, a_]):.4f}"
            f"  crit_sum {np.nanmean(save[co + 'crit_sum'][:, a_]):.6g}")
    sel = save[co + "sel_idx"]
    ok = sel >= 0
    if ok.any():
        tv_sel = save[co + "tv_pos"][np.flatnonzero(ok), sel[ok]]
        run("hold-out constant (argmin crit_sum per set): c / c0 = 10^("
            + ", ".join(f"{save[co + 'k'][i] / 2:+.1f}" for i in sel[ok])
            + f"); tv_pos {C.ms(tv_sel)}"
            + ("  WARNING: at an end of the c grid"
               if np.isin(sel[ok], [0, save[co + 'c'].size - 1]).any() and save[co + 'c'].size > 1
               else ""))
    if ref is not None:
        run(f"stored hold-out lam(t) tv_pos {C.ms(ref)}")


if __name__ == "__main__":
    main()
