"""Time-integration error of the flows against a reference solution.
Paper: Table 7 (tab:integrator); Sec. 5.4 (RK4 against DOP853 and the step-doubled
reference).

Without --target: the 1D targets trimodal and inhom, lam(t) from oned.npz (result
integration_error). With `--target T` (T a 2D target, one per process as twod.py): lam(t) from
twod_<T>.npz (result integration_error_<T>). Per (target, N in config.NS, set of config.SETS;
training sets rng SEEDS['train'][T] N + set) and flow in config.INTEGRATION_ERROR['flows']:
  exact    the exact empirical field v_N of the truncated latent (EmpiricalFlow);
  holdout  the production hold-out flow: Velocity(all samples).set_lam(lam), B_2 with
           m = M_SPLINE[D], lam(t) from oned.npz '<d>/holdout/lam' or twod_<T>.npz
           'holdout/lam' (frame and times checked; 2D also penalty h2 and m,
           common.ProductionLam), refit mode,
both in the frame of oned.py / twod.py, from the same m iid truncated-latent draws (the first m
of the iid draws of oned_eval / twod, rng SEEDS['iid'] + set; m = INTEGRATION_ERROR['m_lat'],
smoke config.SMOKE_ADAPTIVE_LATENTS) to T_END, by RK4 with step_factors x NSTEP uniform steps in tau
and by the flow's reference (REFERENCE[flow]):
  exact    'ref': RK4 with step doubling from NSTEP steps until two levels agree to
           config.REFERENCE_RK4 (sampler.reference_rk4; smoke: at most
           SMOKE_REFERENCE_DOUBLINGS doublings). The truncated v_N jumps at the component box
           faces; adaptive error control (DOP853) stalls there at 60-130x the evaluations of
           step doubling.
  holdout  'dop853': solve_ivp DOP853 (rtol, atol of config.INTEGRATION_ERROR, joint error
           control over the points); the spline field is C^2, so DOP853 needs about 2.5-4.2e3
           evaluations.
One job per (d, N, set, flow, integrator); differences against the reference are taken in the
parent.

npz keys: <d>/<flow>/<integ>/{tv, tv_se, wall, nfev, frac_out} (ns, sets) for integ in
rk<k> (k in steps) and the flow's reference (exact: ref, holdout: dop853);
<d>/<flow>/rk<k>/{dx (ns, sets, m) in 1D, (ns, sets, m, D) in 2D, dlogf (ns, sets, m)}:
x_RK4(t_end) - x_ref(t_end) in original units (s times the frame difference) and
log f_RK4 - log f_ref; <d>/exact/ref/{ref_nstep,
ref_converged, ref_err_x, ref_err_logf, ref_err_logf_max} (the converged level and its last
level differences: max |dx|, mean and max |dlog f|, frame units); exact flows also <d>/exact/<integ>/{max_dlogf_law, n_outside_support} (log f
against log rho_N(x_end, t_end), the effect of the omitted delta terms of div v_N); ns, sets, flows, integrators (the RK4 runs
rk<k>), references ('<flow>:<reference>'), nstep, steps, m_latents, frame_c, frame_s, frame_L.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import EmpiricalFlow, config, tv_pos
from smoothfm.sampler import reference, reference_rk4, sample
from common import ProductionLam

ABOUT = """{name}: RK4 ({steps} steps) against a reference for the exact empirical
flow v_N (RK4 with step doubling, config.REFERENCE_RK4) and the production hold-out flow
(lam(t) from {dep}; DOP853 rtol {rtol}, atol {atol}), {cases}, {m} iid
truncated latents per flow.
Each job: one target, N, set, flow and integrator; integrator summary (the reference with its
per-level history), tv_pos with SE; the parent forms max |dx| and |dlog f| against the
reference (summary below).
Outputs: {out}.npz (read by the report: tab_integrator), {out}.log, {out}.manifest.json."""


#: The reference integrator per flow: step doubling for the exact v_N (jumps at the box faces),
#: DOP853 for the C^2 spline flow.
REFERENCE = {"exact": "ref", "holdout": "dop853"}


def integrators(P):
    steps = [int(f * P.nstep) for f in config.INTEGRATION_ERROR["step_factors"]]
    return steps, [f"rk{k}" for k in steps]


def label(g):
    return {"ref": "RK4 step doubling", "dop853": "DOP853"}.get(g, f"RK4 {g[2:]} steps")


def task(job):
    name, n, s, flow, integ, lam, P, m = job
    st = C.setting(name)
    _, ys = st.training_set(n, s)
    v = EmpiricalFlow(ys, st.latent) if flow == "exact" else \
        st.velocity(ys, P.times, P.m, "h2").set_lam(lam)
    x0 = st.iid_latents(m, s)
    t0 = time.time()
    info = None
    if integ == "ref":
        z, lf, info = reference_rk4(v, x0, config.T_END, n0=P.nstep,
                                    max_doublings=C.reference_doublings(P), latent=st.latent)
        nfev = info["nfev"]
    elif integ == "dop853":
        z, lf, nfev = reference(v, x0, config.T_END, rtol=config.INTEGRATION_ERROR["rtol"],
                                atol=config.INTEGRATION_ERROR["atol"], latent=st.latent)
    else:
        z, lf, summ = sample(v, x0, nstep=int(integ[2:]), t_end=config.T_END)
        nfev = summ.nfev
    wall = time.time() - t0
    tv, se = tv_pos(st.std, z, lf)
    out = {"z": C.to_phys_points(st, z), "lf": lf - st.frame.log_det, "tv": tv, "tv_se": se,
           "wall": wall, "nfev": int(nfev), "frac_out": C.outside(st, z)}
    if info is not None:
        out["info"] = info
        out.update({k: v for k, v in C.reference_arrays(info).items() if k != "nfev"})
    if flow == "exact":
        out["max_dlogf_law"] = float(np.max(np.abs(lf - v.field.log_density(z, config.T_END))))
        out["n_outside_support"] = v.n_outside_support
    return out


def main():
    p = C.parser(__doc__)
    p.add_argument("--target", choices=C.TARGETS_2D, default=None,
                   help="one 2D target (lam(t) from twod_<T>, result integration_error_<T>); "
                        "without it the 1D targets (oned, result integration_error)")
    a = C.parse(p)
    dim = 1 if a.target is None else 2
    P = a.protocol = C.Protocol(a, dim)
    src = ProductionLam(a, P, "integration_error", a.target)
    name = "integration_error" + src.suffix
    st0 = C.setting(src.targets[0])
    IE = config.INTEGRATION_ERROR
    # smoke: every flow on the reference subset, since dx and dlog f pair RK4 with reference points
    m = config.SMOKE_ADAPTIVE_LATENTS if a.smoke else IE["m_lat"]
    steps, integ = integrators(P)
    flows = tuple(IE["flows"])
    run = C.open_run(name, a, {"targets": src.targets, "flows": flows, "integrators": integ,
                               "reference": {"exact": C.reference_text(P),
                                             "holdout": f"DOP853 rtol {IE['rtol']} atol "
                                                        f"{IE['atol']}"},
                               "latents": f"{m} iid truncated latents (rng {config.SEEDS['iid']}"
                                          " + set, first m of "
                                          + ("oned_eval's" if dim == 1 else "twod's")
                                          + " draws)"},
                     ABOUT.format(name=name, steps=steps, rtol=IE["rtol"], atol=IE["atol"], m=m,
                                  cases=("1D trimodal and inhom" if dim == 1 else
                                         f"2D {src.targets[0]}"),
                                  dep=C.rel(src.path),
                                  out=C.rel(C.result_path(a, name, ""))))
    C.worked_example(run, st0, P, P.ns[0], P.sets[0],
                     f"holdout (lam(t) of {src.dep}), all integrators",
                     "h2", latent_kind=f"{m} iid truncated latents, rng {config.SEEDS['iid']} + set",
                     extra={"integrators": ", ".join(integ + ["dop853 (holdout reference)"]),
                            "lam grid": f"not used (lam(t) from {src.dep}.npz)"})
    names = src.targets if a.mode != "example" else src.targets[:1]
    # expensive first (large N, exact field, many steps), so the pool's tail is short
    per_flow = {f: [REFERENCE[f]] + integ[::-1] for f in flows}
    jobs = [(d, n, s, f, g, src.value(d, "holdout", "lam", n, s), P, m)
            for n in P.ns[::-1] for f in flows for g in per_flow[f] for d in names
            for s in P.sets]
    if a.mode == "example":
        jobs = [j for j in jobs if j[3] == "holdout" and j[4] == f"rk{steps[0]}"][:1]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))
    out = {}

    def done(job, r):
        d, n, s, f, g = job[:5]
        jl = f"{d} N={n} set {s} {f} {g}"
        if "info" in r:
            C.log_reference(run, jl, r["info"])
        run(f"[{jl}] flow: {label(g)}  t_end "
            f"{config.T_END}  nfev {r['nfev']}  outside Omega {100 * r['frac_out']:.3g}%  wall "
            f"{r['wall']:.2f} s")
        man = run._job(jl)
        nstep = (None if g == "dop853" else int(r["ref_nstep"]) if g == "ref" else int(g[2:]))
        man.setdefault("flows", []).append({"integrator": g, "nstep": nstep, "t_end": config.T_END,
                                            "nfev": r["nfev"], "outside": r["frac_out"],
                                            "wall": r["wall"]})
        mt = {"tv": r["tv"], "tv_se": r["tv_se"]}
        if f == "exact":
            mt.update(max_dlogf_law=r["max_dlogf_law"], n_outside_support=r["n_outside_support"])
        run.metrics(jl, mt)
        idx = (ns.index(n), sets.index(s))
        for k in ("tv", "tv_se", "wall", "nfev", "frac_out") + \
                (("max_dlogf_law", "n_outside_support") if f == "exact" else ()) + \
                (("ref_nstep", "ref_converged", "ref_err_x", "ref_err_logf", "ref_err_logf_max")
                 if g == "ref" else ()):
            save.put(f"{d}/{f}/{g}/{k}", idx, r[k])
        out[(d, n, s, f, g)] = r

    C.pool_map(task, jobs, a.workers, run, done)
    for (d, n, s, f, g), r in out.items():
        ref = out.get((d, n, s, f, REFERENCE[f]))
        if g == REFERENCE[f] or ref is None:
            continue
        idx = (ns.index(n), sets.index(s))
        save.put(f"{d}/{f}/{g}/dx", idx, r["z"] - ref["z"])
        save.put(f"{d}/{f}/{g}/dlogf", idx, r["lf"] - ref["lf"])

    run("summary per (target, flow, N): tv mean over sets; max over sets of max|dx| and of "
        "mean|dlog f|; wall and nfev mean over sets")
    for d in names:
        for f in flows:
            for i, n in enumerate(ns):
                for g in per_flow[f]:
                    pre = f"{d}/{f}/{g}/"
                    if pre + "tv" not in save:
                        continue
                    msg = (f"  {d} {f} N={n} {g:7s} tv {np.nanmean(save[pre + 'tv'][i]):.4f}  wall "
                           f"{np.nanmean(save[pre + 'wall'][i]):.1f} s  nfev "
                           f"{np.nanmean(save[pre + 'nfev'][i]):.0f}")
                    if pre + "dx" in save:
                        msg += (f"  max|dx| {np.nanmax(np.abs(save[pre + 'dx'][i])):.2e}  mean|dlogf| "
                                f"{np.nanmax(np.nanmean(np.abs(save[pre + 'dlogf'][i]), -1)):.2e}")
                    run(msg)
    run.save(ns=np.array(ns), sets=np.array(sets), flows=np.array(flows),
             integrators=np.array(integ),
             references=np.array([f"{f}:{REFERENCE[f]}" for f in flows]), nstep=P.nstep, steps=np.array(steps), m_latents=m,
             **st0.record(), **save)
    run.close()


if __name__ == "__main__":
    main()
