"""lam -> 0 controls with iid latents, and the stopped law.
Paper: every lam -> 0 entry: the rows of Table 5 (tab:refine-1d) and Table 6
(tab:refine-2d), the curves of Figure 2 (fig:accuracy), and, at the production m, the
lam -> 0 rows of Table 2 (tab:oned-full) and Table 3 (tab:twod-full); Sec. 5.4.

controls.py estimates tv_pos of the lam -> 0 flows at the production latents (1D quantile
midpoints, 2D Sobol points). Those are stratified samples, and on a memorizing flow (N narrow
spikes) they alias with the spikes, so the paper reports the lam -> 0 TV from iid latents.
This script regenerates the lam -> 0 flows of controls.py (same training sets, frame, m,
Velocity.set_floor: lam = LAM_FLOOR_REL lam_s(t) with mu M, RK4 config.NSTEP steps to T_END)
from iid truncated-latent draws (TruncatedGaussian.sample, rng SEEDS['controls_iid'] + set, the
same draws for every m; config.M_LAT[D] points) and stores tv_iid = mean (1 - pi/f)_+ with its
Monte Carlo SE. Per (case, N, set) also tv_law: the TV of the law every lam -> 0 flow stopped at
t_end approaches, rho_N(., t_end) = the equal-weight mixture of N(t y'_i, (1 - t)^2 I) truncated
to t y'_i + (1 - t) Omega (empirical.StoppedLaw), from CONTROLS_IID['m_law'] direct
draws (rng SEEDS['law'] + set), with its SE.
  --dim 1               reads controls_1d, result controls_iid_1d (trimodal, inhom).
  --target T [--m M]    reads controls_T[_mM], result controls_iid_T[_mM] (M = 512 for every
                        2D target, config.CONTROLS_M_EXTRA: controls_iid_<T>_m512, the m = 512
                        row of tab_refine_2d (a); the circle's N = 1000 is fig_accuracy's point).
N per case from config.CONTROLS_IID['ns'] (every N for every case); --m: CONTROLS_M_EXTRA
iid_ns (iid_ns). Resolutions m: the dependency's <case>/ms. The dependency must hold the same
frame and the lam -> 0 cell of every job (checked), except with --m at the N outside
CONTROLS_M_EXTRA lam0_ns (ref_required): there the dependency has hold-out rows only and
ref_tv_pos is NaN (ref_value). The lam -> 0 flow itself never comes from the dependency.

npz keys (arrays (len(ns), len(sets)), NaN where not computed): <case>/m<m>/lam0/{tv_iid,
tv_iid_se, ref_tv_pos, ref_tv_pos_se (the dependency's production-latent tv_pos), wall, wall_fit, wall_gen,
frac_out, frac_out_any, nfev, max_abs} and lam, lam_s, lam_rel (K,); <case>/{tv_law,
tv_law_se, wall_law}; <case>/ms, <case>/frame_c, <case>/frame_s, <case>/frame_L; cases, ns,
sets, times, lam0_rel. Read by the report (tab_refine_1d / _2d: <case>/m<m>/lam0/tv_iid and
<case>/tv_law; fig_accuracy: <case>/m<m>/lam0/tv_iid at N = 1000; tab_oned_full,
tab_twod_full: <case>/m<M_SPLINE>/lam0/tv_iid). ref_tv_pos is a check: the production-latent
TV of the same flow from the dependency, logged beside tv_iid.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import copy
import time

import numpy as np

from smoothfm import StoppedLaw, config, tv_pos
from smoothfm.sampler import FlowSummary

ABOUT = """{name}: lam -> 0 flows of {dep} (lam = {floor:g} lam_s(t), mu M), cases {cases}, m in
{ms}, from {m_lat} iid truncated-latent draws (rng {seed} + set): tv_iid with SE, beside the
production-latent tv_pos of {dep}; tv_law of rho_N(., t_end) from {m_law} draws (rng {law_seed}
+ set). Outputs: {out}.npz (report tab_refine_1d, tab_refine_2d, fig_accuracy, tab_oned_full,
tab_twod_full), {out}.log, {out}.manifest.json."""


def iid_latents(st, M, s):
    return st.latent.sample(M, np.random.default_rng(config.SEEDS["controls_iid"] + s))


def task(job):
    case, n, s, m, P = job
    st = C.setting(case)
    _, ys = st.training_set(n, s)
    t0 = time.time()
    if m is None:            # the stopped law, independent of m
        law = StoppedLaw(ys, config.T_END, st.latent)
        z = law.sample(P.m_law, np.random.default_rng(config.SEEDS["law"] + s))
        tv, se = tv_pos(st.std, z, law.logpdf(z))
        return {"tv_law": tv, "tv_law_se": se, "wall_law": time.time() - t0}
    v = st.velocity(ys, P.times, m, "h2").set_floor()
    rec = C.velocity_record(v, time.time() - t0)
    g = C.flow_record(C.generate(v, iid_latents(st, P.m_lat, s), st.std, P.nstep))
    rec.update(g)
    rec["tv_iid"], rec["tv_iid_se"] = rec.pop("tv_pos"), rec.pop("tv_pos_se")
    rec["wall"] = rec["wall_fit"] + rec["wall_gen"]
    return rec


def iid_ns(a, case, smoke_ns):
    """N of the iid lam -> 0 rows of a case: the protocol's (smoke_ns) in smoke runs,
    CONTROLS_M_EXTRA[T]['iid_ns'] with --m, else CONTROLS_IID['ns'][case]."""
    if a.smoke:
        return tuple(smoke_ns)
    if a.m is not None:
        return tuple(config.CONTROLS_M_EXTRA[a.target]["iid_ns"])
    return tuple(config.CONTROLS_IID["ns"][case])


def ref_required(a, case, n) -> bool:
    """Whether the dependency must hold the production-latent lam -> 0 cell of (case, N):
    always, except with --m in a full run at an N outside CONTROLS_M_EXTRA lam0_ns."""
    if a.m is None or a.smoke:
        return True
    return n in tuple(config.CONTROLS_M_EXTRA[a.target]["lam0_ns"])


def ref_value(ctl, c, m, dns, dsets, n, s):
    """(tv_pos, tv_pos_se) of the dependency's lam -> 0 cell, (NaN, NaN) where it has none."""
    k = f"{c}/m{m}/lam0/tv_pos"
    if k not in ctl or n not in dns or s not in dsets:
        return np.nan, np.nan
    i, j = dns.index(n), dsets.index(s)
    return float(ctl[k][i, j]), float(ctl[k + "_se"][i, j])


def main():
    p = C.parser(__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dim", type=int, choices=(1,), help="the 1D cases (reads controls_1d)")
    g.add_argument("--target", choices=C.TARGETS_2D, help="one 2D target (reads controls_T)")
    p.add_argument("--m", type=int, default=None,
                   help="the extra resolution of config.CONTROLS_M_EXTRA (512)")
    a = C.parse(p)
    if a.m is not None and (a.target not in config.CONTROLS_M_EXTRA
                            or a.m != config.CONTROLS_M_EXTRA[a.target]["m"]):
        p.error("--m: only " + ", ".join(f"--target {t} --m {e['m']}"
                                         for t, e in config.CONTROLS_M_EXTRA.items()))
    dim = 1 if a.dim is not None else 2
    P = a.protocol = C.Protocol(a, dim)
    P.m_law = config.CONTROLS_IID["m_law_smoke" if a.smoke else "m_law"]
    tag = "1d" if dim == 1 else a.target
    sfx = "" if a.m is None else f"_m{a.m}"
    dep, name = f"controls_{tag}{sfx}", f"controls_iid_{tag}{sfx}"
    ctl = C.load(a, dep)
    cases = list(C.TARGETS_1D) if dim == 1 else [a.target]
    ms = {c: [int(x) for x in ctl[f"{c}/ms"]] for c in cases}
    dns, dsets = [int(x) for x in ctl["ns"]], [int(x) for x in ctl["sets"]]
    jobs = []
    for c in cases:
        st = C.setting(c)
        C.check_frame(ctl, st, f"{dep} {c}", prefix=f"{c}/")
        for n in iid_ns(a, c, P.ns):
            for s in P.sets:
                for m in ms[c]:
                    k = f"{c}/m{m}/lam0/tv_pos"
                    if ref_required(a, c, n) and not np.isfinite(
                            ref_value(ctl, c, m, dns, dsets, n, s)[0]):
                        raise SystemExit(f"{C.result_path(a, dep, deps=True)}: no lam -> 0 cell "
                                         f"{k} N={n} set {s}; rerun compute-{dep}")
                    jobs.append((c, n, s, m))
                jobs.append((c, n, s, None))
    jobs.sort(key=lambda j: (-j[1], -(j[3] or 0)))
    run = C.open_run(name, a, {"dependency": C.rel(C.result_path(a, dep, deps=True)),
                               "cases": cases, "ms": ms, "lam0": f"floor {config.LAM_FLOOR_REL:g} "
                               "lam_s(t), with mu M", "latents": f"{P.m_lat} iid truncated-latent "
                               f"draws, rng {config.SEEDS['controls_iid']} + set",
                               "law": f"{P.m_law} draws of rho_N(., t_end), rng "
                                      f"{config.SEEDS['law']} + set"},
                     ABOUT.format(name=name, dep=dep, floor=config.LAM_FLOOR_REL, m_lat=P.m_lat,
                                  seed=config.SEEDS["controls_iid"], cases=", ".join(cases),
                                  ms=sorted({m for v in ms.values() for m in v}), m_law=P.m_law,
                                  law_seed=config.SEEDS["law"],
                                  out=C.rel(C.result_path(a, name, ""))))
    flows = [j for j in jobs if j[3] is not None]
    first = min(flows, key=lambda j: (j[1], j[2], j[3] != P.m))
    st0 = C.setting(first[0])
    P0 = copy.copy(P)
    P0.m = first[3]
    C.worked_example(run, st0, P0, first[1], first[2], "lam0 (floor)", "h2",
                     latent_kind=f"{P.m_lat} iid draws, rng {config.SEEDS['controls_iid']} + set",
                     extra={"lam0 row": f"lam = {config.LAM_FLOOR_REL:g} lam_s(t) at every time "
                                        "(Velocity.set_floor), no selection",
                            "stopped law": f"StoppedLaw(y', t_end), {P.m_law} draws, rng "
                                           f"{config.SEEDS['law']} + set"})
    if a.mode == "example":
        jobs = [first]
    ns = sorted({j[1] for j in jobs})
    sets = sorted({j[2] for j in jobs})
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))

    def done(job, rec):
        c, n, s, m = job[:4]
        idx = (ns.index(n), sets.index(s))
        if m is None:
            run.metrics(f"{c} N={n} set {s} law", {"tv_law": rec["tv_law"],
                                                    "tv_law_se": rec["tv_law_se"],
                                                    "wall": round(rec["wall_law"], 3)})
            save.put_record(c, idx, rec)
            return
        jl = f"{c} N={n} set {s} m={m} lam0 iid"
        ref, ref_se = ref_value(ctl, c, m, dns, dsets, n, s)
        rec["ref_tv_pos"], rec["ref_tv_pos_se"] = ref, ref_se
        run.flow(jl, None, FlowSummary(**rec["summary"]),
                 {"fit": round(rec["wall_fit"], 3), "generation": round(rec["wall_gen"], 3)})
        run.metrics(jl, {"tv_iid": rec["tv_iid"], "tv_iid_se": rec["tv_iid_se"],
                         "ref_tv_pos": ref, "ref_tv_pos_se": ref_se,
                         "lam/lam_s": " ".join(f"{x:.3g}" for x in rec["lam_rel"])})
        save.put_record(f"{c}/m{m}/lam0", idx, rec)

    C.pool_map(task, [j + (P,) for j in jobs], a.workers, run, done)

    run(f"summary: lam -> 0 tv_iid (iid latents) vs tv_pos of {dep} (production latents); "
        "mean (SE over sets) [min, max]")
    for c in cases:
        for i, n in enumerate(ns):
            if f"{c}/tv_law" in save and np.isfinite(save[f"{c}/tv_law"][i]).any():
                run(f"  {c:12s} N={n:5d}  law rho_N(t_end) {C.ms(save[f'{c}/tv_law'][i])}")
            for m in ms[c]:
                k = f"{c}/m{m}/lam0"
                if k + "/tv_iid" in save and np.isfinite(save[k + "/tv_iid"][i]).any():
                    run(f"  {c:12s} N={n:5d} m={m:5d}  iid {C.ms(save[k + '/tv_iid'][i])}  "
                        f"production {C.ms(save[k + '/ref_tv_pos'][i])}  wall "
                        f"{np.nanmean(save[k + '/wall'][i]):.0f} s")
    frames = {}
    for c in cases:
        frames.update(C.setting(c).record(f"{c}/"))
        frames[f"{c}/ms"] = np.array(ms[c])
    run.save(cases=np.array(cases), ns=np.array(ns), sets=np.array(sets), times=P.times,
             lam0_rel=config.LAM_FLOOR_REL, **frames, **save)
    run.close()


if __name__ == "__main__":
    main()
