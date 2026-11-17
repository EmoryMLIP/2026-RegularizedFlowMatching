"""Time discretization of the velocity: does it add implicit regularization?
Paper: Sec. 4.2 "Selection of the Regularization Parameter", last paragraph (interpolating the
solution in time changes the TV by up to 0.005; compare_time.log), in the
production setup.

One training set (set 0) per case of config.COMPARE_TIME: the noisy circle (2D) and the
trimodal Beta mixture (1D), each in its production frame, spline resolution M_SPLINE, latent,
mu, lambda grid and generation (RK4, NSTEP, T_END, production latents) as oned.py / twod.py;
N in COMPARE_TIME['ns'], lam(t_k) by oracle and by hold-out at the production times. The same
lam(t_k) is used by
  (i)   field   : coefficients of w = (1 - t) v interpolated linearly in tau between the
                  selection times (Velocity.with_mode("interpolate"));
  (ii)  refit   : solve at every RK stage time, log lam interpolated in tau (production);
and on the refined grid (the tau-midpoints inserted, 2K - 1 times):
  (iii) field2  : field interpolation, lam selected at every refined time by the same rule;
  (iv)  field2i : field interpolation with the lam(t) of (i) evaluated at the refined times.

npz keys (arrays (len(ns), ...)): <case>/<rule>/<variant>/{tv_pos, tv_pos_se, wall_gen,
frac_out, frac_out_any, nfev, max_abs} for variant in field, refit, field2, field2i;
<case>/<rule>/{coarse, fine}/{lam, lam_s, lam_rel, at_limit, scores, nsolves, nevals, residual,
wall_t, wall_fit, ...} (the selections at the production and the refined times) and the
per-job histories <case>/<rule>/{coarse, fine}/N<n>/{tables, ...}; <case>/frame_c, frame_s,
frame_L; ns, times, times_fine, cases, rules.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import config
from common import HoldOutRule, OracleRule, population

ABOUT = """compare_time: TV of the same lam(t) under field interpolation in tau (coarse and refined
time grids) and under refit (production), for the circle and the trimodal target, set 0,
N in {ns}, rules oracle and hold-out, production setup.
Each job: one case, N and rule: selections at the production times and at the refined times
(histories below), then four generations; tv_pos with SE.
Outputs: {out}.npz (read by the report: the compare_time sentence of sec:cv), {out}.log,
{out}.manifest.json."""

VARIANTS = ("field", "refit", "field2", "field2i")


def refined(times):
    tau = -np.log1p(-times)
    return 1.0 - np.exp(-np.sort(np.concatenate([tau, 0.5 * (tau[:-1] + tau[1:])])))


def task(job):
    case, n, rule, P, pop, pop2 = job
    st = C.setting(case)
    _, ys = st.training_set(n, 0)
    T, T2 = P.times, refined(P.times)

    def factory(times, pp):
        if rule == "oracle":
            return OracleRule(pp[0], pp[1], times, pp[2])
        return HoldOutRule(*st.split(ys, 0))

    out = {}
    t0 = time.time()
    v = st.velocity(ys, T, P.m, "h2").fit(factory(T, pop), C.search("h2", st.dim))
    out["coarse"] = C.velocity_record(v, time.time() - t0)
    t0 = time.time()
    v2 = st.velocity(ys, T2, P.m, "h2").fit(factory(T2, pop2), C.search("h2", st.dim))
    out["fine"] = C.velocity_record(v2, time.time() - t0)
    t0 = time.time()
    v2i = st.velocity(ys, T2, P.m, "h2").set_lam([v.lam_at(t) for t in T2])
    out["wall_fit_field2i"] = time.time() - t0
    x0 = st.latents(P.m_lat)
    flows = {"field": v.with_mode("interpolate"), "refit": v,
             "field2": v2.with_mode("interpolate"), "field2i": v2i.with_mode("interpolate")}
    for k, f in flows.items():
        out[k] = C.flow_record(C.generate(f, x0, st.std, P.nstep))
    return out


def main():
    p = C.parser(__doc__)
    a = C.parse(p)
    P1 = a.protocol = C.Protocol(a, 1)
    name = "compare_time"
    CT = config.COMPARE_TIME
    ns = list(P1.ns) if a.smoke else list(CT["ns"])
    cases, rules = tuple(CT["cases"]), tuple(CT["rules"])
    run = C.open_run(name, a, {"cases": cases, "ns": ns, "rules": rules, "variants": VARIANTS,
                               "set": 0, "refined times": "tau-midpoints inserted"},
                     ABOUT.format(ns=ns, out=C.rel(C.result_path(a, name, ""))))
    Ps = {c: C.Protocol(a, C.setting(c).dim) for c in cases}
    st0 = C.setting(cases[0])
    P0 = Ps[cases[0]]
    C.worked_example(run, st0, P0, ns[-1], 0, "holdout", "h2",
                     split_sizes=[len(x) for x in st0.split(st0.training_set(ns[-1], 0)[1], 0)],
                     extra={"variants": "field (interpolate in tau), refit, field2 (refined "
                                        "grid), field2i (refined grid, lam of field)",
                            "refined times": f"{refined(P0.times).size}"})
    pops = {c: (population(C.setting(c), Ps[c].times, Ps[c].m_oracle),
                population(C.setting(c), refined(Ps[c].times), Ps[c].m_oracle))
            for c in cases if "oracle" in rules}
    jobs = [(c, n, r, Ps[c]) + (pops[c] if r == "oracle" else (None, None))
            for c in cases for n in ns for r in rules]
    if a.mode == "example":
        jobs = [j for j in jobs if j[0] == cases[0] and j[1] == ns[-1] and j[2] == "holdout"]
    save = C.Store((len(ns),), ([f"N{n}" for n in ns],))
    res = {}

    def done(job, out):
        c, n, r = job[:3]
        P = Ps[c]
        jl = f"{c} N={n} set 0 {r}"
        C.log_velocity(run, jl, r, out["coarse"], P.times)
        C.log_velocity(run, jl, r + " (refined times)", out["fine"], refined(P.times))
        for k in VARIANTS:
            rec = out[k]
            walls = {"fit": out["coarse"]["wall_fit"] if k in ("field", "refit") else
                     out["fine"]["wall_fit"] if k == "field2" else out["wall_fit_field2i"],
                     "generation": rec["wall_gen"]}
            C.log_flow(run, jl, rec, walls, label=f"{k}/")
            save.put_record(f"{c}/{r}/{k}", (ns.index(n),), rec)
        for g in ("coarse", "fine"):
            save.put_record(f"{c}/{r}/{g}", (ns.index(n),), out[g])
        res[(c, n, r)] = out

    C.pool_map(task, jobs, a.workers, run, done)
    run("summary: case N rule | tv_pos (i) field (ii) refit (iii) field2 (iv) field2i | "
        "|(i)-(ii)| |(iv)-(ii)| (SE of (ii))")
    for (c, n, r), o in sorted(res.items()):
        tv = {k: o[k]["tv_pos"] for k in VARIANTS}
        run(f"  {c} {n} {r} | " + " ".join(f"{tv[k]:.4f}" for k in VARIANTS)
            + f" | {abs(tv['field'] - tv['refit']):.1e} {abs(tv['field2i'] - tv['refit']):.1e} "
              f"({o['refit']['tv_pos_se']:.1e})")
    frames = {f"{c}/{k}": v for c in cases for k, v in C.setting(c).record().items()}
    run.save(ns=np.array(ns), cases=np.array(cases), rules=np.array(rules),
             **{f"{c}/times": Ps[c].times for c in cases},
             **{f"{c}/times_fine": refined(Ps[c].times) for c in cases}, **frames, **save)
    run.close()


if __name__ == "__main__":
    main()
