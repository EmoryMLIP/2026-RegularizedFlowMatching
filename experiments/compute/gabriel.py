"""Gabriel et al. (arXiv:2505.22841) kernel-smoothed velocity against our flow, 1D.
Paper: the Gabriel et al. rows of Table 2 (tab:oned-full); Sec. 5.6, last paragraph; the
convolution baseline of Appendix B "Benchmark Methods".

v_h = G_{t sqrt h} * v_N (standard deviation t sqrt h; smoothfm.baselines.gabriel.GabrielVelocity)
in the frame of oned.py through StdGabriel (the exact time change onto GabrielVelocity on the
original data). v_N is the empirical velocity of the truncated latent (the same v_N as the spline flow) and
the flows start from the truncated latent (quantile midpoints, as oned.py). h is stored in the
frame's units h' (h = s^2 h' in the original ones); the grids config.GABRIEL are in original
units, so the candidates are HC = {0} U logspace(*h_t) / s^2 and HG = logspace(*h_sweep) / s^2.
Variants:
  G1  constant h by the hold-out FM loss (fit 80% / validation 20%, the split of oned.py)
      integrated over t (trapezoid weights on the selection times), generation with all N;
  G2  h(t) per time by the hold-out loss;
  G3  oracle constant h: argmin tv_pos over HG, refined by bounded Brent in log10 h between
      the sweep neighbours (xatol config.GABRIEL['brent_xatol']);
  G4  oracle h(t) = argmin mean |u_h - v|^2 at the oracle points of oned.py;
  exact  h = 0, the empirical flow v_N: a check, the reference tv_ref of oned_eval's
         exact_rk (the same flow by another code path); no table reads it.
h(t) between selection times: sqrt(h) linear in tau (h_schedule). Ours (oned hold-out and
oracle tv_pos with SE) is copied from oned.npz (frame checked).

npz keys (arrays (len(ns), len(sets), ...)): <d>/<lab>/{tv_pos, tv_pos_se, wall_gen,
frac_out, frac_out_any, nfev, max_abs} for lab in exact, G1..G4; <d>/<lab> (= tv_pos, the
key the report reads) and <d>/<lab>_se; <d>/h1, <d>/h3 (scalars, frame units), <d>/h2,
<d>/h4 (K,); <d>/sweep, <d>/sweep_se (len(HG),); <d>/loss_val, <d>/err (K, len(HC)) the
hold-out loss and velocity error tables; <d>/wall_tab, <d>/wall; <d>/ours_holdout,
<d>/ours_oracle (+ _se); ns, sets, times, hc, hg, frame_c, frame_s, frame_L.
``velocity(st, ys, h, times)`` builds the flow of a stored h (oned_eval regenerates G1..G4).
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np
from scipy.optimize import minimize_scalar

from smoothfm import config
from smoothfm.baselines import StdGabriel, h_schedule
from common import population

ABOUT = """gabriel: Gabriel et al.'s kernel-smoothed empirical velocity (truncated latent) in
1D (trimodal, inhom): G1 constant h by hold-out, G2 h(t) by hold-out, G3 oracle constant h
(TV sweep plus Brent), G4 oracle h(t) (velocity error), exact (h = 0), against our flows
(oned hold-out and oracle, copied from {dep}).
Each job: one target, N and training set: the per-time tables of the hold-out loss and the
velocity error over the h candidates, then RK4 generation of every variant, tv_pos with SE.
Outputs: {out}.npz (read by oned_eval, gabriel_hero and the report: tab_main, tab_oned_full),
{out}.log (this file), {out}.manifest.json (provenance, cost, output checksums)."""

LABELS = ("exact", "G1", "G2", "G3", "G4")


def grids(st):
    """(HC, HG) in the frame's units h' = h / s^2."""
    g = config.GABRIEL
    s2 = st.frame.s ** 2
    return (np.concatenate([[0.0], np.logspace(*g["h_t"][:2], int(g["h_t"][2]))]) / s2,
            np.logspace(*g["h_sweep"][:2], int(g["h_sweep"][2])) / s2)


def trapezoid(times):
    d = np.diff(times)
    return np.concatenate([[d[0] / 2], (d[:-1] + d[1:]) / 2, [d[-1] / 2]])


def velocity(st, ys, h, times):
    """The frame's Gabriel velocity of the original data ys (N,) for constant h' (scalar) or
    h' at the selection times (array), truncated latent."""
    hf = h_schedule(times, h) if np.ndim(h) else float(h)
    return StdGabriel(np.asarray(ys, float).ravel(), hf, st.frame, st.latent)


def task(job):
    name, n, s, P, pop = job
    st = C.setting(name)
    HC, HG = grids(st)
    times = P.times
    y, ys = st.training_set(n, s)
    yt, yv = st.split(ys, s)
    yt_phys = st.frame.to_phys(yt)
    z, yref, vpop = pop
    t0 = time.time()
    Lv = np.full((times.size, HC.size), np.nan)
    Er = np.full((times.size, HC.size), np.nan)
    for j, t in enumerate(times):
        x = (1 - t) * z + t * yref
        for k, h in enumerate(HC):
            Lv[j, k] = velocity(st, yt_phys, h, times).val_loss(yv[:, 0], t)
            Er[j, k] = float(np.mean((velocity(st, y, h, times)(x, t) - vpop[j]) ** 2))
    res = {"loss_val": Lv, "err": Er, "wall_tab": time.time() - t0,
           "h1": float(HC[int(np.argmin(trapezoid(times) @ Lv))]),
           "h2": HC[np.argmin(Lv, 1)], "h4": HC[np.argmin(Er, 1)], "flows": {}}
    x0 = st.latents(P.m_lat)

    def gen(h):
        return C.flow_record(C.generate(velocity(st, y, h, times), x0, st.std, P.nstep))

    for lab, h in (("exact", 0.0), ("G1", res["h1"]), ("G2", res["h2"]), ("G4", res["h4"])):
        res["flows"][lab] = gen(h)
    sweep = [gen(h) for h in HG]
    res["sweep"] = np.array([r["tv_pos"] for r in sweep])
    res["sweep_se"] = np.array([r["tv_pos_se"] for r in sweep])
    k = int(np.argmin(res["sweep"]))
    lo, hi = np.log10(HG[max(k - 1, 0)]), np.log10(HG[min(k + 1, HG.size - 1)])
    seen = {}

    def f(lh):
        seen[float(lh)] = gen(10.0 ** lh)
        return seen[float(lh)]["tv_pos"]

    r = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                        options={"xatol": config.GABRIEL["brent_xatol"]})
    if r.fun <= res["sweep"][k]:
        res["h3"], res["flows"]["G3"] = float(10.0 ** r.x), seen[float(r.x)]
    else:
        res["h3"], res["flows"]["G3"] = float(HG[k]), sweep[k]
    res["n_brent"] = len(seen)
    res["wall"] = time.time() - t0
    return res


def main():
    p = C.parser(__doc__)
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 1)
    name = "gabriel"
    on = C.load(a, "oned")
    st0 = C.setting("trimodal")
    C.check_frame(on, st0, "oned.npz")
    HC, HG = grids(st0)
    run = C.open_run(name, a, {"targets": C.TARGETS_1D, "labels": LABELS,
                               "hc (frame units)": f"{{0}} U logspace{config.GABRIEL['h_t']} / s^2",
                               "hg (frame units)": f"logspace{config.GABRIEL['h_sweep']} / s^2",
                               "latents": f"{P.m_lat} truncated-latent quantile midpoints",
                               "latent of v_N": "truncated (eq:empirical)"},
                     ABOUT.format(dep=C.rel(C.result_path(a, "oned", deps=True)),
                                  out=C.rel(C.result_path(a, name, ""))))
    first = ("trimodal", P.ns[0], P.sets[0])
    ys0 = st0.training_set(first[1], first[2])[1]
    C.worked_example(run, st0, P, first[1], first[2], "G1-G4, exact", "h2",
                     split_sizes=[len(x) for x in st0.split(ys0, first[2])],
                     extra={"lam grid": "not used", "regularizer": "none (Gabriel smoothing)",
                            "h candidates": f"HC: 0 and {HC.size - 1} values {HC[1]:.4g}..{HC[-1]:.4g} "
                                            f"(frame units); HG: {HG.size} values "
                                            f"{HG[0]:.4g}..{HG[-1]:.4g}",
                            "oracle points": f"{P.m_oracle} Sobol pairs, rng "
                                             f"{config.SEEDS['oracle_1d']}",
                            "smoothing": "v_h = G_{t sqrt h} * v_N (S:sec:kde), v_N of the "
                                         "truncated latent (eq:empirical), StdGabriel time "
                                         "change into the frame"})
    names = C.TARGETS_1D if a.mode != "example" else ("trimodal",)
    pops = {d: population(C.setting(d), P.times, P.m_oracle) for d in names}
    jobs = [(d, n, s, P, pops[d]) for n in P.ns[::-1] for d in names for s in P.sets]
    if a.mode == "example":
        jobs = [first + (P, pops["trimodal"])]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))

    def done(job, res):
        d, n, s = job[:3]
        idx = (ns.index(n), sets.index(s))
        jl = f"{d} N={n} set {s}"
        run(f"[{jl}] h' (frame units): h1 {res['h1']:.4g}  h3 {res['h3']:.4g}  h2(t) "
            + " ".join(f"{x:.3g}" for x in res["h2"]) + "  h4(t) "
            + " ".join(f"{x:.3g}" for x in res["h4"]))
        run(f"[{jl}] G3 sweep tv_pos: " + " ".join(f"{x:.4f}" for x in res["sweep"])
            + f"  Brent evaluations {res['n_brent']}")
        for lab in LABELS:
            rec = res["flows"][lab]
            C.log_flow(run, f"{jl} {lab}", rec, {"tables": res["wall_tab"],
                                                  "generation": rec["wall_gen"]})
            save.put_record(f"{d}/{lab}", idx, rec)
            save.put(f"{d}/{lab}", idx, rec["tv_pos"])
            save.put(f"{d}/{lab}_se", idx, rec["tv_pos_se"])
        for k in ("h1", "h2", "h3", "h4", "sweep", "sweep_se", "loss_val", "err", "wall_tab",
                  "wall"):
            save.put(f"{d}/{k}", idx, res[k])
        run.metrics(jl, {"wall": round(res["wall"], 2)})

    C.pool_map(task, jobs, a.workers, run, done)

    onn = list(on["ns"])
    for d in names:
        for r in ("holdout", "oracle"):
            for suf in ("", "_se"):
                arr = np.full((len(ns), len(sets)), np.nan)
                for i, n in enumerate(ns):
                    arr[i] = on[f"{d}/{r}/tv_pos{suf}"][onn.index(n)][list(sets)]
                save[f"{d}/ours_{r}{suf}"] = arr
    run("summary: tv_pos mean (SE over sets) [min, max]; paired ratio ours_holdout / G1")
    for d in names:
        for i, n in enumerate(ns):
            for lab in LABELS + ("ours_holdout", "ours_oracle"):
                run(f"  {d} N={n} {lab:13s} {C.ms(save[f'{d}/{lab}'][i])}")
            rr = save[f"{d}/ours_holdout"][i] / save[f"{d}/G1"][i]
            run(f"  {d} N={n} ratio ours_holdout/G1 {np.round(rr, 2)}  wall per set "
                f"{np.nanmean(save[f'{d}/wall'][i]):.0f} s")
    run.save(ns=np.array(ns), sets=np.array(sets), times=P.times, hc=HC, hg=HG,
             **st0.record(), **save)
    run.close()


if __name__ == "__main__":
    main()
