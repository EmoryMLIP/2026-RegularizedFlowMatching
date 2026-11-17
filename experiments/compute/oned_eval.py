"""1D evaluation without reselection: tv_iid and the exact v_N rows.
Paper: Table 1 (tab:main), Table 2 (tab:oned-full); the 1D fallback of Table 7
(tab:integrator); e(t) of Figure 1 (fig:hero).

Every flow and KDE of oned.py (and of gabriel.py) is regenerated from the
stored lam(t) and h and measured. Flows run in the frame of oned.py (common.Setting, frame
onto [-5, 5]); tv_pos, tv_iid, tv_law and e are frame quantities (TV is invariant under the
map), the curves are computed after mapping back (x = c + s x', log f = log f' - log s).
Stored lam(t) and h must come from the same frame (common.check_frame on every source).
Per (target in {trimodal, inhom}, N in config.NS, set) and method:
  flows    oracle, holdout (config.ONED_RULES; lam(t) = oned '<d>/<rule>/lam'):
           Velocity(all samples).set_lam(lam), refit mode, RK4 with NSTEP steps to T_END.
  exact    the exact v_N (EmpiricalFlow, truncated latent, no splines) by RK4 with NSTEP
           steps ('exact_rk') and by the reference integrator, RK4 with step doubling
           from NSTEP steps until two levels agree to config.REFERENCE_RK4
           (sampler.reference_rk4; 'exact_ref'; per level nstep, dx, dlog f, wall logged, the
           converged level and its err_x, err_logf stored). Logged per flow: points outside the
           support of rho_N (n_outside_support, fallback weights) and
           max |log f - log rho_N(x_end, t_end)|.
  KDE      kde_<rule> for config.ONED_KDE_RULES from oned '<d>/kde_<rule>/h' (Abramson factors
           from the Sheather-Jones pilot as oned.py); draws rng SEEDS['kde_draws'] + set.
  Gabriel  G1..G4 from the stored frame bandwidths h' of gabriel.npz (config.ONED_EVAL
           ['gabriel'] maps label -> key): baselines.StdGabriel with the truncated latent. A
           reproduction check of gabriel.py (tv_ref = its tv_pos); the report reads the
           Gabriel rows from gabriel.npz.
tv_ref: the stored tv_pos the regenerated flow must reproduce (oned.npz for the flows and
KDEs, gabriel.npz for G1..G4 and, for exact_rk, gabriel.py's h = 0 flow, the same exact v_N
by another code path); the log reports max |tv_pos - tv_ref| per method.
Latents: M_LAT truncated-latent quantile midpoints stacked on M_LAT iid truncated draws (rng
SEEDS['iid'] + set), one integration (refit mode: each point's path is independent of the
others). The midpoints give tv_pos (reproduces the stored value; max deviation logged), nonmono
and the curves; the iid draws tv_iid (unbiased; the midpoints alias with the spikes of a
memorizing flow, so the unregularized rows use tv_iid). tv_law: TV of the stopped law
rho_N(., t_end) (StoppedLaw, truncated components) from ONED_EVAL['law_draws'] direct draws
(rng SEEDS['law'] + set). nonmono: order violations of the outputs of the latent quantile
midpoints (a 1D flow map is monotone). e(t_k) = mean |v - v*|^2 at the oracle
points of oned.py (frame units). curve: density on XF by interpolation of (x_end, f) (flows) or
exact (KDE); curve_cell: mass per XF cell from the quantile points (flows).

npz keys (arrays (len(ns), len(sets), ...)), per <d> in {trimodal, inhom} and method <m>:
  <d>/<m>/{tv_pos, tv_pos_se, tv_iid, tv_iid_se, tv_ref, nonmono, wall, curve (nx),
  curve_cell (nx)} for flows; flows also {frac_out, frac_out_any, nfev, max_abs, wall_fit,
  wall_gen} and spline flows {lam, lam_s, lam_rel} (K,) and e (K,); exact methods also
  {tv_law, tv_law_se (exact_rk only), npts, n_outside, max_dlogf, e (K,)}; exact_ref also
  {ref_nstep, ref_converged, ref_err_x, ref_err_logf (mean), ref_err_logf_max};
  KDE <d>/kde_<r>/{tv_pos, tv_pos_se, h, tv_ref, wall, curve (nx)};
  <d>/pdf_true (nx), <d>/hist (ns, sets, 50);
  top level: ns, sets, times, xf, hist_edges, methods, nstep, frame_c, frame_s,
  frame_L.
Method keys: oracle, holdout, exact_rk, exact_ref, kde_<rule>, G1..G4.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import EmpiricalFlow, Oracle, StoppedLaw, config, tv_pos
from smoothfm import baselines as B
from smoothfm.sampler import FlowSummary, reference_rk4, sample
from common import population
from common1d import (HIST_EDGES, XF, exact_errors, exact_extra, latents,
                      row, spline_errors, split_entry)

EV = config.ONED_EVAL
FLOW_RULES = tuple(config.ONED_RULES)
GAB = tuple(EV["gabriel"])

ABOUT = """oned_eval: the 1D flows and KDEs of oned (and gabriel) regenerated from the
stored lam(t) and h without reselection, plus the exact v_N by RK4
({steps} steps) and by RK4 with step doubling (the reference); per flow tv_pos (midpoint latents, reproduces the stored value),
tv_iid (iid latents), e(t), density curves; tv_law of the stopped law rho_N(t_end).
Reads {deps}.
Outputs: {out}.npz (read by kde4 (grid), hero_1d and the report: tab_oned_full,
tab_main, tab_integrator), {out}.log (this file), {out}.manifest.json."""


def methods():
    return (list(FLOW_RULES) + ["exact_rk", "exact_ref"]
            + [f"kde_{r}" for r in config.ONED_KDE_RULES] + list(GAB))


GROUPS = FLOW_RULES + ("exact_rk", "exact_ref", "kde", "gabriel")


# ------------------------------------------------------------------ measurements

def generate(v, st, M, s, nstep):
    """RK4 from the stacked latents; the measured entry with the flow record."""
    t0 = time.time()
    z, lf, summ = sample(v, latents(st, M, s), nstep=nstep, t_end=config.T_END)
    r = split_entry(st, z, lf, M)
    r.update(wall_gen=time.time() - t0, frac_out=summ.frac_outside_end,
             frac_out_any=summ.frac_outside_any, nfev=summ.nfev, max_abs=summ.max_abs,
             summary=summ.as_dict())
    return r, z, lf


def ref_points(P):
    """Midpoint latents of the reference exact-field flow: all M_LAT, smoke ONED_EVAL's few."""
    return EV["ref_points_smoke"] if P.smoke else P.m_lat


def task(job):
    d, n, s, group, P, pop, par = job
    st = C.setting(d)
    yo, ys = st.training_set(n, s)
    M, out = P.m_lat, {}
    t0 = time.time()
    if group in FLOW_RULES:
        v = st.velocity(ys, P.times, P.m).set_lam(par)
        rec = C.velocity_record(v, time.time() - t0)
        r, _, _ = generate(v, st, M, s, P.nstep)
        r.update({k: rec[k] for k in ("lam", "lam_s", "lam_rel", "wall_fit")})
        r["e"] = spline_errors(v, P.times, pop)
        if group == "oracle":
            # e(t_k) equals Oracle.score at the selected lam (first, middle, last time)
            ks = sorted({0, P.times.size // 2, P.times.size - 1})
            sc = np.array([Oracle(v.normal_equations(P.times[k]), (1 - P.times[k]) * pop[0]
                                  + P.times[k] * pop[1], pop[2][k]).score(v.lam[k]) for k in ks])
            r["oracle_check"] = float(np.max(np.abs(sc - r["e"][ks]) / sc))
        out[group] = r
    elif group == "exact_rk":
        e = exact_errors(ys, st, P.times, pop)
        law = StoppedLaw(ys, config.T_END, st.latent)
        zl = law.sample(EV["law_draws"], np.random.default_rng(config.SEEDS["law"] + s))
        tl, tl_se = tv_pos(st.std, zl, law.logpdf(zl))
        t1 = time.time()
        fl = EmpiricalFlow(ys, st.latent)
        r, z, lf = generate(fl, st, M, s, P.nstep)
        r.update(e=e, npts=M, wall=time.time() - t1, **exact_extra(fl, law, z, lf),
                 tv_law=tl, tv_law_se=tl_se)
        out["exact_rk"] = r
    elif group == "exact_ref":
        M = ref_points(P)
        fl = EmpiricalFlow(ys, st.latent)
        law = StoppedLaw(ys, config.T_END, st.latent)
        z, lf, info = reference_rk4(fl, latents(st, M, s), config.T_END, n0=P.nstep,
                                    max_doublings=C.reference_doublings(P), latent=st.latent)
        r = split_entry(st, z, lf, M)
        outside = np.any(np.abs(z) > st.L, axis=1)
        r.update(e=exact_errors(ys, st, P.times, pop), npts=M, info=info,
                 frac_out=float(np.mean(outside)), max_abs=float(np.max(np.abs(z))),
                 wall_gen=time.time() - t0, **exact_extra(fl, law, z, lf),
                 **C.reference_arrays(info),
                 summary=FlowSummary(info["nstep"], config.T_END, int(info["nfev"]),
                                     float(np.mean(outside)), float("nan"),
                                     float(np.max(np.abs(z)))).as_dict())
        out["exact_ref"] = r
    elif group == "kde":
        y1 = yo[:, 0]
        lam_abr = B.abramson_factors(y1, B.sheather_jones(y1))
        for rule, h in par.items():
            t1 = time.time()
            k = B.KDE(y1, bandwidth=h * lam_abr if rule.startswith("abr") else h)
            zz = k.sample(M, np.random.default_rng(config.SEEDS["kde_draws"] + s))
            tv, se = tv_pos(st.base, zz, k.logpdf(zz))
            out[f"kde_{rule}"] = dict(tv_pos=tv, tv_pos_se=se, h=h,
                                      curve=np.exp(k.logpdf(XF[:, None])), wall=time.time() - t1)
    elif group == "gabriel":
        for lab, h in par.items():
            t1 = time.time()
            # constant h' (G1, G3) or h'(t_k) at the selection times (G2, G4), as gabriel.py
            hf = B.h_schedule(P.times, h) if np.ndim(h) else float(h)
            g = B.StdGabriel(yo[:, 0], hf, st.frame, st.latent)
            r, _, _ = generate(g, st, M, s, P.nstep)
            r.update(h=h)
            r["wall"] = time.time() - t1
            out[lab] = r
    for r in out.values():
        r.setdefault("wall", time.time() - t0)
    return out


# ------------------------------------------------------------------ sources

def row_array(d, key, n, s):
    return np.asarray(d[key])[[int(x) for x in d["ns"]].index(n), s]


def reference_tv(src, d, n, s, m):
    """The stored tv_pos the regenerated method must reproduce (NaN if none)."""
    if m in FLOW_RULES or m.startswith("kde_"):
        return row(src["oned"], f"{d}/{m}/tv_pos", n, s)
    if m in GAB:
        g = src["gabriel"]
        return row(g, f"{d}/{m}/tv_pos" if f"{d}/{m}/tv_pos" in g else f"{d}/{m}", n, s)
    if m == "exact_rk":
        # gabriel.py's h = 0 flow: the same exact v_N by the production RK4, another code path
        return row(src["gabriel"], f"{d}/exact/tv_pos", n, s)
    return np.nan


def job_params(src, d, n, s, group):
    if group in FLOW_RULES:
        return row_array(src["oned"], f"{d}/{group}/lam", n, s)
    if group == "kde":
        return {r: float(row_array(src["oned"], f"{d}/kde_{r}/h", n, s))
                for r in config.ONED_KDE_RULES}
    if group == "gabriel":
        return {lab: np.asarray(row_array(src["gabriel"], f"{d}/{key}", n, s), float)
                for lab, key in EV["gabriel"].items()}
    return None


# ------------------------------------------------------------------ main

def main():
    p = C.parser(__doc__)
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 1)
    name = "oned_eval"
    deps = ("oned", "gabriel")
    src = {nm: C.load(a, nm) for nm in deps}
    st0 = C.setting("trimodal")
    for nm in deps:
        for d in C.TARGETS_1D:
            C.check_frame(src[nm], C.setting(d), f"{nm}.npz")
    meths = methods()
    run = C.open_run(name, a, {"targets": C.TARGETS_1D, "methods": meths,
                               "exact steps": P.nstep,
                               "reference": f"{C.reference_text(P)}; {ref_points(P)} midpoints "
                                            "+ as many iid",
                               "latents": f"{P.m_lat} truncated-latent quantile midpoints + "
                                          f"{P.m_lat} iid (rng {config.SEEDS['iid']} + set)",
                               "law draws": EV["law_draws"], "xf": EV["xf"],
                               "hist edges": EV["hist_edges"]},
                     ABOUT.format(steps=P.nstep,
                                  deps=", ".join(C.rel(C.result_path(a, nm, deps=True))
                                                 for nm in deps),
                                  out=C.rel(C.result_path(a, name, ""))))
    ys0 = st0.training_set(P.ns[0], P.sets[0])[1]
    C.worked_example(run, st0, P, P.ns[0], P.sets[0], "holdout (stored lam, no selection)", "h2",
                     split_sizes=[len(x) for x in st0.split(ys0, P.sets[0])],
                     latent_kind=f"{P.m_lat} quantile midpoints + {P.m_lat} iid draws",
                     extra={"oracle points": f"{P.m_oracle} Sobol pairs, rng "
                                             f"{config.SEEDS['oracle_1d']}",
                            "stored lam": C.rel(C.result_path(a, "oned", deps=True))})
    names = C.TARGETS_1D if a.mode != "example" else ("trimodal",)
    pops = {d: population(C.setting(d), P.times, P.m_oracle) for d in names}
    run("population velocities at the oracle points done")
    order = ("exact_ref", "exact_rk") + tuple(g for g in GROUPS
                                               if g not in ("exact_ref", "exact_rk"))
    jobs = [(d, n, s, g, P, pops[d], job_params(src, d, n, s, g))
            for n in P.ns[::-1] for g in order for d in names for s in P.sets]
    if a.mode == "example":
        n, s = P.ns[0], P.sets[0]
        jobs = [("trimodal", n, s, "holdout", P, pops["trimodal"],
                 job_params(src, "trimodal", n, s, "holdout"))]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))
    checks = []

    def done(job, out):
        d, n, s, g = job[:4]
        idx = (ns.index(n), sets.index(s))
        for m, r in out.items():
            jl = f"{d} N={n} set {s} {m}"
            r = dict(r)
            r["tv_ref"] = reference_tv(src, d, n, s, m)
            if "oracle_check" in r:
                checks.append((d, n, s, r.pop("oracle_check")))
            if "info" in r:
                C.log_reference(run, jl, r.pop("info"))
            if "summary" in r:
                if "lam_rel" in r:
                    C.log_velocity(run, jl, m, r, P.times)
                walls = {"generation": r.get("wall_gen", r["wall"])}
                if "wall_fit" in r:
                    walls["fit"] = r["wall_fit"]
                C.log_flow(run, jl, r, walls)
            met = {k: r[k] for k in ("tv_pos", "tv_pos_se", "tv_iid", "tv_iid_se", "tv_law",
                                     "tv_law_se", "h", "n_outside", "max_dlogf", "npts",
                                     "nonmono") if k in r}
            met["tv_ref"], met["tv_ref_se"] = r["tv_ref"], 0.0
            met["wall"] = round(float(r["wall"]), 3)
            run.metrics(jl, met)
            for k, v in r.items():
                if k != "summary":
                    save.put(f"{d}/{m}/{k}", idx, v)

    C.pool_map(task, jobs, a.workers, run, done)

    run("tv_pos reproduction: max |tv_pos - stored| over (N, set) per method (stored: oned, "
        "gabriel (G1-G4; exact_rk vs its h = 0 flow))")
    for d in names:
        dev = {}
        for m in meths:
            k = f"{d}/{m}/tv_pos"
            if k in save and np.isfinite(save[f"{d}/{m}/tv_ref"]).any():
                dev[m] = float(np.nanmax(np.abs(save[k] - save[f"{d}/{m}/tv_ref"])))
        run(f"  {d}: " + ", ".join(f"{m} {v:.1e}" for m, v in dev.items()))
    for d, n, s, c in checks:
        run(f"  e(t) vs Oracle.score at the selected lam ({d} N={n} set {s}, 3 times): "
            f"max rel dev {c:.1e}")
    run("summary: mean (SE over sets) [min, max]")
    for d in names:
        for i, n in enumerate(ns):
            run(f"{d} N={n}")
            for m in meths:
                k = f"{d}/{m}/tv_pos"
                if k not in save:
                    continue
                extra = ""
                for q in ("tv_iid", "tv_law", "max_dlogf"):
                    if f"{d}/{m}/{q}" in save and np.isfinite(save[f"{d}/{m}/{q}"][i]).any():
                        extra += f"  {q} {C.ms(save[f'{d}/{m}/{q}'][i])}"
                if f"{d}/{m}/nonmono" in save and np.nansum(save[f"{d}/{m}/nonmono"][i]) > 0:
                    extra += f"  NONMONOTONE {np.nansum(save[f'{d}/{m}/nonmono'][i]):.0f}"
                run(f"  {m:15s} tv_pos {C.ms(save[k][i])}{extra}")
    extra = {}
    for d in names:
        st = C.setting(d)
        extra[f"{d}/pdf_true"] = st.base.pdf(XF[:, None])
        extra[f"{d}/hist"] = np.array([[np.histogram(st.training_set(n, s)[0][:, 0],
                                                     HIST_EDGES)[0] for s in sets] for n in ns])
    run.save(ns=np.array(ns), sets=np.array(sets), times=P.times, xf=XF, hist_edges=HIST_EDGES,
             methods=np.array(meths), nstep=P.nstep,
             **st0.record(), **extra, **save)
    run.close()


if __name__ == "__main__":
    main()
