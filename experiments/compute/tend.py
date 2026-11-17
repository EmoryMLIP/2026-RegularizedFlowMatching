"""The t_end limit: as t_end -> 1, does the unregularized flow approach TV = 1 while
the tuned flows stay unchanged? Paper: Table 4 (tab:tend); Sec. 5.4.

Without --target: 1D trimodal and inhom in the frame of oned.py (result tend). With
`--target T` (T a 2D target, one per process as twod.py): T in the frame of twod.py, lam(t)
from twod_<T> (result tend_<T>). N in config.NS, config.SETS sets (training sets rng
SEEDS['train'][T] N + set), and 1 - t_end in config.TEND['one_minus_t_end'] (1e-4, 1e-6, 1e-8).
Latents: the M_LAT[D] production latents (1D truncated-latent quantile midpoints; 2D scrambled
Sobol points, rng SEEDS['sobol_2d']; tv_pos, 1D also nonmono) stacked on M_LAT[D] iid truncated-latent
draws (rng SEEDS['iid'] + set, as oned_eval and twod) for tv_iid; every flow is integrated once,
with the states taken at each t_end. Hold-out split rng SEEDS['split'] + set; oracle points
as oned.py / twod.py (1D Sobol rng SEEDS['oracle_1d'], 2D normal draws rng SEEDS['oracle_2d']).
  exact    the exact empirical field v_N of the truncated latent (EmpiricalFlow) by the
           reference integrator, RK4 with step doubling from NSTEP steps until two levels agree
           to config.REFERENCE_RK4 (sampler.reference_rk4, one doubling per t_end, each t_end
           its own pool task since the doublings share no state; the truncated v_N jumps at the
           component box faces, where adaptive error control stalls);
           per level nstep, dx, dlog f, wall logged, the converged level, its err_x, err_logf
           and the right-hand-side evaluations stored, the evaluations outside supp rho_N and
           max |log f - log rho_N(x_end, t_end)| recorded.
           tv_law: TV of the law it converges to, rho_N(., t_end) (StoppedLaw, box-truncated
           Gaussians), from config.TEND['law_draws'] direct draws (rng SEEDS['law'] + set).
  oracle,  the spline flow of oned.py / twod.py (B_2, m = M_SPLINE[D]): lam(t) at the selection
  holdout  times from oned.npz / twod_<T>.npz (frame and times checked; 2D also penalty h2 and
           m, common.ProductionLam); at the added times 1 - t = 10^(-k/10), k in
           range(*TEND['new_k']) (1e-4.1 .. 1e-8) selected by the same rule (oracle at the
           oracle points of oned.py / twod.py, hold-out 80/20 and refit on all; the lambda grid
           of the dimension). Generation: refit, RK4 with the production step dtau = tau(T_END)/NSTEP,
           so NSTEP, 1.5 NSTEP and 2 NSTEP steps; at
           1 - 1e-4 this is the production flow (tv_pos reproduces oned.npz / twod_<T>.npz,
           logged).
tv_pos, tv_iid and tv_law are measured in the frame (the estimator tv_pos of twod.py in 2D),
nonmono (1D only) after mapping back.
Smoke: the added times config.TEND['smoke_new_idx'] only, and the exact (reference) rows on
config.SMOKE_ADAPTIVE_LATENTS production latents (1D midpoints, 2D Sobol) + as many iid points
with at most config.SMOKE_REFERENCE_DOUBLINGS doublings from the smoke NSTEP (the full run is
unchanged).

npz keys (arrays (len(ns), len(sets), len(one_minus_t_end))), <d> the target:
<d>/<kind>/{tv_pos, tv_pos_se, tv_iid, tv_iid_se, nonmono (1D, order_measures),
frac_out, nfev}; exact also {tv_law, tv_law_se, max_dlogf, n_outside_support, m_points,
ref_nstep, ref_converged, ref_err_x, ref_err_logf (mean), ref_err_logf_max}; <d>/<kind>/wall
(ns, sets); tuned kinds: the new-time selection record
<d>/<kind>/new/{lam, lam_s, lam_rel, at_limit, scores, nsolves, nevals, residual, wall_t,
wall_fit, ...} (ns, sets, K_new), the per-job history <d>/<kind>/new/N<n>_set<s>/{tables,
...}, and the aliases <d>/<kind>/{lam_new, lam_scale_new, score_new}; ns, sets,
one_minus_t_end, new_times, nstep, dtau, frame_c, frame_s, frame_L.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import EmpiricalFlow, StoppedLaw, config, tv_pos
from smoothfm.sampler import FlowSummary, reference_rk4, sample_path, tau_of_t
from common import HoldOutRule, OracleRule, ProductionLam, population

ABOUT = """{name}: TV against 1 - t_end in {om} of the exact v_N flow (RK4 with step doubling), its
law rho_N(., t_end), and the oracle and hold-out spline flows (lam(t) from {dep} plus selections at
{knew} added times down to 1 - t = 1e-8), {cases}.
Each job: one target, N, set and kind (exact, oracle, holdout; exact also one 1 - t_end, the
pieces assembled per target, N and set); tuned kinds list the selection
history at the added times. TV over the {prod} latents (tv_pos) and over iid
latents (tv_iid), both with Monte Carlo SE.
Outputs: {out}.npz (read by the report: tab_tend), {out}.log (this file), {out}.manifest.json."""

KINDS = ("exact", "oracle", "holdout")


def new_times(smoke):
    k = np.arange(*config.TEND["new_k"])
    t = 1.0 - 10.0 ** (-k / 10)
    return t[list(config.TEND["smoke_new_idx"])] if smoke else t


def t_ends():
    return [1.0 - e for e in config.TEND["one_minus_t_end"]]


def latents(st, M, s):
    return np.vstack([st.latents(M), st.iid_latents(M, s)])


def order_measures(st, z):
    """1D only: nonmono, the order violations of the production-latent outputs mapped back (the
    latents are sorted quantile midpoints and a 1D flow map is increasing, so any violation is
    an integration artefact). No 2D counterpart: a 2D flow map can reverse the orientation of a triangle of latent points without
    folding (measured: the exact v_N flow of 5 circle points, reference converged,
    reverses 7 of the 501 Delaunay triangles of 256 Sobol latents at t = 0.99), so a triangle
    count would not detect folding."""
    if st.dim != 1:
        return {}
    zp = st.frame.to_phys(z[:, 0])
    return {"nonmono": int(np.sum(np.diff(zp) < 0))}


def entry(st, z, lf, M):
    """tv_pos (production half), tv_iid (iid half), each with SE; 1D also nonmono of the
    production half (order_measures); frac_out."""
    tv, se = tv_pos(st.std, z[:M], lf[:M])
    tvi, sei = tv_pos(st.std, z[M:], lf[M:])
    return {"tv_pos": tv, "tv_pos_se": se, "tv_iid": tvi, "tv_iid_se": sei,
            **order_measures(st, z[:M]), "frac_out": C.outside(st, z)}


def tuned(st, ys, s, rule, P, lam_stored, pop_new, tnew):
    if rule == "oracle":
        factory = OracleRule(pop_new[0], pop_new[1], tnew, pop_new[2])
    else:
        factory = HoldOutRule(*st.split(ys, s))
    t0 = time.time()
    vn = st.velocity(ys, tnew, P.m, "h2").fit(factory, C.search("h2", st.dim))
    rec = C.velocity_record(vn, time.time() - t0)
    v = st.velocity(ys, np.concatenate([P.times, tnew]), P.m, "h2")
    v.set_lam(np.concatenate([lam_stored, vn.lam]))
    dtau = tau_of_t(config.T_END) / P.nstep
    t1 = time.time()
    path = sample_path(v, latents(st, P.m_lat, s), t_ends(), dtau)
    per = [entry(st, z, lf, P.m_lat) for z, lf in path]
    for r, te in zip(per, t_ends()):
        r["nfev"] = 4 * int(round(tau_of_t(te) / dtau))
    return {"per_t": per, "new": rec, "wall_gen": time.time() - t1}


def ref_m(P):
    """Latent points of the reference rows: M_LAT, or SMOKE_ADAPTIVE_LATENTS in smoke runs."""
    return config.SMOKE_ADAPTIVE_LATENTS if P.smoke else P.m_lat


def exact(st, ys, s, P, t):
    """One t_end of the exact flow: its own reference_rk4 doubling from NSTEP on a fresh
    EmpiricalFlow. n_outside_support is this run's count; the parent (assemble_exact) stores
    the sum over the t_end runs."""
    fl = EmpiricalFlow(ys, st.latent)
    M = ref_m(P)
    t1 = time.time()
    z, lf, info = reference_rk4(fl, latents(st, M, s), t_end=t, n0=P.nstep,
                                max_doublings=C.reference_doublings(P), latent=st.latent)
    wall = time.time() - t1
    r = entry(st, z, lf, M)
    r["m_points"] = M
    r["info"] = info
    r.update(C.reference_arrays(info))
    law = StoppedLaw(ys, t, st.latent)
    zl = law.sample(config.TEND["law_draws"], np.random.default_rng(config.SEEDS["law"] + s))
    r["tv_law"], r["tv_law_se"] = tv_pos(st.std, zl, law.logpdf(zl))
    lr = fl.field.log_density(z, t)
    r.update(max_dlogf=float(np.max(np.abs(lf - lr))),
             n_outside_support=fl.n_outside_support)
    return {"per_t": [r], "wall_gen": wall}


def assemble_exact(pieces):
    """The per-t_end exact results of one (target, N, set), in t_end order, as one result."""
    per = [p["per_t"][0] for p in pieces]
    total = sum(r["n_outside_support"] for r in per)
    for r in per:
        r["n_outside_support"] = total
    return {"per_t": per, "wall_gen": sum(p["wall_gen"] for p in pieces),
            "wall": sum(p["wall"] for p in pieces)}


def task(job):
    """job: (target, N, set, kind, 1 - t_end (exact) or None, P, lam_stored, pop_new, tnew)."""
    name, n, s, kind, e, P, lam_stored, pop_new, tnew = job
    st = C.setting(name)
    _, ys = st.training_set(n, s)
    t0 = time.time()
    out = exact(st, ys, s, P, 1.0 - e) if kind == "exact" else tuned(st, ys, s, kind, P,
                                                                     lam_stored, pop_new, tnew)
    out["wall"] = time.time() - t0
    return out


def main():
    p = C.parser(__doc__)
    p.add_argument("--target", choices=C.TARGETS_2D, default=None,
                   help="one 2D target (lam(t) from twod_<T>, result tend_<T>); without it the "
                        "1D targets (oned, result tend)")
    a = C.parse(p)
    dim = 1 if a.target is None else 2
    P = a.protocol = C.Protocol(a, dim)
    src = ProductionLam(a, P, "tend", a.target)
    name = "tend" + src.suffix
    st0 = C.setting(src.targets[0])
    tnew = new_times(a.smoke)
    om = list(config.TEND["one_minus_t_end"])
    dtau = tau_of_t(config.T_END) / P.nstep
    steps = [int(round(tau_of_t(t) / dtau)) for t in t_ends()]
    prod = src.latent_text(P)
    run = C.open_run(name, a, {"targets": src.targets, "kinds": KINDS, "one_minus_t_end": om,
                               "added times": f"{tnew.size}: 1 - t = "
                                              + " ".join(f"{1 - t:.1e}" for t in tnew),
                               "rk4 steps": steps, "dtau": dtau,
                               "exact": C.reference_text(P),
                               "latents": f"{prod} + {P.m_lat} iid "
                                          f"(rng {config.SEEDS['iid']} + set), truncated"},
                     ABOUT.format(name=name, om=om, dep=C.rel(src.path), knew=tnew.size,
                                  cases=("1D trimodal and inhom" if dim == 1 else
                                         f"2D {src.targets[0]}"),
                                  prod=("quantile-midpoint" if dim == 1 else "Sobol"),
                                  out=C.rel(C.result_path(a, name, ""))))
    first = (src.targets[0], P.ns[0], P.sets[0], "holdout")
    ys0 = st0.training_set(first[1], first[2])[1]
    C.worked_example(run, st0, P, first[1], first[2],
                     f"holdout (lam(t) of {src.dep} + added times)",
                     "h2", split_sizes=[len(x) for x in st0.split(ys0, first[2])],
                     extra={"integrator": f"RK4 in tau, step {dtau:.5f}, {steps} steps to "
                                          f"1 - t_end = {om}",
                            "latent points": f"{prod} + {P.m_lat} iid",
                            "added times": " ".join(f"{t:.10g}" for t in tnew)})
    names = src.targets if a.mode != "example" else src.targets[:1]
    pops = {d: population(C.setting(d), tnew, P.m_oracle) for d in names}
    run("population velocities at the added times done; max |v - x/t| at the last added time "
        "(the sigma -> 0 limit): " + ", ".join(
            f"{d} {np.max(np.abs(pops[d][2][-1] - ((1 - tnew[-1]) * pops[d][0] + tnew[-1] * pops[d][1]) / tnew[-1])):.2e}"
            for d in names))
    jobs = []
    for n in P.ns[::-1]:
        for kind in KINDS:
            for d in names:
                for s in P.sets:
                    if kind == "exact":
                        # one task per t_end, the longest (1 - t_end smallest) first
                        jobs += [(d, n, s, kind, e, P, None, None, tnew) for e in om[::-1]]
                        continue
                    lam = src.value(d, kind, "lam", n, s)
                    jobs.append((d, n, s, kind, None, P, lam,
                                 pops[d] if kind == "oracle" else None, tnew))
    if a.mode == "example":
        jobs = [j for j in jobs if j[:4] == first][:1]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))
    # nonmono is a 1D measure (order_measures)
    keys = tuple(k for k in ("tv_pos", "tv_pos_se", "tv_iid", "tv_iid_se", "nonmono",
                             "frac_out", "nfev") if dim == 1 or k != "nonmono")
    xkeys = ("tv_law", "tv_law_se", "max_dlogf", "n_outside_support", "m_points", "ref_nstep",
             "ref_converged", "ref_err_x", "ref_err_logf", "ref_err_logf_max")

    pending = {}

    def collect(job, res):
        """Exact pieces wait until every t_end of their (target, N, set) is in."""
        if job[3] != "exact":
            return done(job, res)
        got = pending.setdefault(job[:3], {})
        got[job[4]] = res
        if len(got) == len(om):
            done(job, assemble_exact([got[e] for e in om]))
            del pending[job[:3]]

    def done(job, res):
        d, n, s, kind = job[:4]
        idx = (ns.index(n), sets.index(s))
        jl = f"{d} N={n} set {s} {kind}"
        pre = f"{d}/{kind}/"
        if kind != "exact":
            rec = res["new"]
            C.log_velocity(run, jl, kind, rec, tnew)
            save.put_record(pre + "new", idx, rec)
            save.put(pre + "lam_new", idx, rec["lam"])
            save.put(pre + "lam_scale_new", idx, rec["lam_s"])
            save.put(pre + "score_new", idx, rec["scores"])
            ref = src.value(d, kind, "tv_pos", n, s)
            run(f"[{jl}] check: tv_pos at 1 - t_end = 1e-4 {res['per_t'][0]['tv_pos']:.6f}, "
                f"{src.dep}.npz {ref:.6f}, |diff| {abs(res['per_t'][0]['tv_pos'] - ref):.1e}")
        for key in keys + (xkeys if kind == "exact" else ()):
            save.put(pre + key, idx, [r[key] for r in res["per_t"]])
        for r, e, k_end in zip(res["per_t"], om, steps):
            # metric names <1 - t_end>/<key>, so that each TV has its <name>_se beside it
            m = {f"{e:.0e}/{key}": r[key] for key in keys + (xkeys if kind == "exact" else ())
                 if key not in ("nonmono",)}
            run.metrics(jl, m)
            # integrator summary: the reference at its converged level, RK4 k_end steps
            if kind == "exact":
                C.log_reference(run, jl, r["info"], f"1 - t_end {e:.0e} reference")
            fs = FlowSummary(int(r["ref_nstep"]) if kind == "exact" else int(k_end), float(1 - e),
                             int(r["nfev"]),
                             float(r["frac_out"]), float("nan"), float("nan"))
            run.flow(jl, None, fs, {"fit": round(res["wall"] - res["wall_gen"], 3),
                                    "generation": round(res["wall_gen"], 3)})
            run(f"[{jl}] 1 - t_end {e:.0e}: integrator "
                f"{'RK4 step doubling' if kind == 'exact' else 'RK4'}"
                f"  nfev {r['nfev']}  outside Omega {100 * r['frac_out']:.3g}%"
                + (f"  latent points {r['m_points']} + {r['m_points']} iid" if kind == "exact"
                   else ""))
            if kind == "exact" and r["n_outside_support"]:
                run(f"[{jl}] NOTE: {r['n_outside_support']} field evaluations outside supp rho_N")
        save.put(pre + "wall", idx, res["wall"])
        run.metrics(jl, {"wall": round(res["wall"], 2), "wall_gen": round(res["wall_gen"], 2)})

    C.pool_map(task, jobs, a.workers, run, collect)
    assert not pending, sorted(pending)

    run("summary: tv_iid mean (SE over sets) [min, max]; tv_pos means")
    for d in names:
        for i, n in enumerate(ns):
            for kind in KINDS:
                pre = f"{d}/{kind}/"
                for k, e in enumerate(om):
                    line = (f"  {d} N={n} {kind:8s} 1-t_end={e:.0e} tv_iid "
                            f"{C.ms(save[pre + 'tv_iid'][i, :, k])}  tv_pos "
                            f"{np.nanmean(save[pre + 'tv_pos'][i, :, k]):.4f}")
                    if kind == "exact":
                        line += (f"  tv_law {np.nanmean(save[pre + 'tv_law'][i, :, k]):.4f}  nfev "
                                 f"{np.nanmean(save[pre + 'nfev'][i, :, k]):.0f}  max|dlog f| "
                                 f"{np.nanmax(save[pre + 'max_dlogf'][i, :, k]):.2e}")
                    if dim == 1 and np.nansum(save[pre + "nonmono"][i, :, k]) > 0:
                        line += f"  NONMONOTONE {np.nansum(save[pre + 'nonmono'][i, :, k]):.0f}"
                    run(line)
            for kind in ("oracle", "holdout"):
                lr = np.nanmedian(save[f"{d}/{kind}/new/lam_rel"][i], 0)
                run(f"  {d} N={n} {kind} log10(lam/lam_s) at the added times (median over sets): "
                    + " ".join("-inf" if not x > 0 else f"{np.log10(x):.1f}" for x in lr))
    run.save(ns=np.array(ns), sets=np.array(sets), one_minus_t_end=np.array(om),
             new_times=tnew, nstep=P.nstep, dtau=dtau, steps=np.array(steps), **st0.record(),
             **save)
    run.close()


if __name__ == "__main__":
    main()
