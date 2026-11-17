"""One hold-out flow per regularizer beside Gabriel et al.'s G1 (--regs, result
gabriel_hero_reg). Paper: the 31% (smallness, k = 0) / 14.5-15.2% (k = 2, 1) sample shares and
lam/lam_s ~ 3e6 of Sec. 5.6 (experiments/report/prose_numbers.py reads them from
gabriel_hero_reg).

Gabriel et al.'s kernel-smoothed velocity G1 (baselines.StdGabriel with the constant frame
bandwidth h' = gabriel.npz '<d>/h1', smoothing the truncated-latent v_N) beside our
penalized flows (Velocity(all samples).set_lam(lam), hold-out lam(t), refit mode) on the
trimodal target at N = config.GABRIEL_HERO['n'] (smoke: the smoke N), training set: the median
hold-out tv_pos set of oned.npz (common1d.choose_set, as fig_hero), so that every penalty is
shown on the same training set. The flows run in the frame of oned.py; RK4 with NSTEP steps to
T_END from M_LAT quantile midpoints stacked on M_LAT iid draws (common1d.latents).

--regs [R ...] (default config.GABRIEL_HERO['regs'] = h2 h1 l2): flows G1 plus one hold-out
flow per penalty, 'holdout' for h2 (lam(t) of oned.npz) and 'holdout_<R>' (lam(t) from the
hold-out rule of oned_<R>.npz, the penalty of oned.py --reg R).

Measured per flow: within = fraction of the midpoint outputs within GABRIEL_HERO['within_r']
(original units) of a training point (metrics.frac_within; the statistic of sec:operator);
tv_pos, tv_pos_se (midpoint outputs), tv_iid, tv_iid_se (iid outputs), nonmono. Reproduction
check: tv_ref = the stored tv_pos of the same flow (gabriel.npz G1, oned[_<R>].npz holdout);
a full run logs max |tv_pos - tv_ref| and flags it above 1e-6.

npz keys. '<d>/n<N>/set_median'; per flow '<d>/n<N>/s<set>/<flow>/': within, tv_pos,
tv_pos_se, tv_iid, tv_iid_se, nonmono, tv_ref, h (G1, frame units) or lam, lam_rel (K),
frac_out, nfev, wall. Top level: ns, flows, dataset, regs, times, frame_c, frame_s, frame_L.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

import common1d as C1
from smoothfm import baselines as B
from smoothfm import config, frac_within
from smoothfm.sampler import FlowSummary, sample

GH = config.GABRIEL_HERO

ABOUT = """{name}: Gabriel et al.'s G1 flow (constant bandwidth, truncated-latent v_N) beside the
hold-out flows {flows} of oned on the {d} target, N = {n}, on the median hold-out training set:
the fraction of samples within {r:g} of a training point, tv_pos and tv_iid.
Reads {deps}.
Outputs: {out}.npz (read by the report: prose_numbers), {out}.log (this file), {out}.manifest.json."""


def flow_name(reg):
    return "holdout" if reg == "h2" else f"holdout_{reg}"


def flow_reg(flow):
    return flow.split("_", 1)[1] if "_" in flow else "h2"


def gab_tv(g, d, lab, n, s):
    k = f"{d}/{lab}/tv_pos" if f"{d}/{lab}/tv_pos" in g else f"{d}/{lab}"
    return C1.row(g, k, n, s)


def task(job):
    d, n, s, flow, par, P = job
    t0 = time.time()
    st = C.setting(d)
    fr = st.frame
    yo, ys = st.training_set(n, s)
    M = P.m_lat
    r = {}
    if flow == "G1":
        h = float(par)
        v = B.StdGabriel(yo[:, 0], h, fr, st.latent)
        r["h"] = h
    else:
        v = st.velocity(ys, P.times, P.m, flow_reg(flow)).set_lam(par)
        r["lam"] = np.asarray(par, float)
        r["lam_rel"] = v.rel_lam
    z, lf, summ = sample(v, C1.latents(st, M, s), nstep=P.nstep, t_end=config.T_END)
    r.update(C1.split_entry(st, z, lf, M))
    for k in ("curve", "curve_cell"):
        r.pop(k)
    r["summary"] = summ.as_dict()
    r["frac_out"], r["nfev"] = summ.frac_outside_end, summ.nfev
    r["within"] = frac_within(fr.to_phys(z[:M, 0])[:, None], yo, GH["within_r"])
    r["wall"] = time.time() - t0
    return r


def main():
    p = C.parser(__doc__)
    p.add_argument("--regs", nargs="*", choices=("h2", "h1", "l2"), default=None,
                   help="the penalties (absent or no value: config.GABRIEL_HERO['regs']); "
                        "result gabriel_hero_reg")
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 1)
    d = "trimodal"
    regs = list(dict.fromkeys(a.regs or GH["regs"]))
    name = "gabriel_hero_reg"
    flows = ["G1"] + [flow_name(r) for r in regs]
    dep_names = ["oned", "gabriel"] + [f"oned_{r}" for r in regs if r != "h2"]
    src = {nm: C.load(a, nm) for nm in dep_names}
    st = C.setting(d)
    for nm in dep_names:
        C.check_frame(src[nm], st, f"{nm}.npz")
    for r in regs:
        if r != "h2" and str(src[f"oned_{r}"]["reg"]) != r:
            raise SystemExit(f"oned_{r}.npz records reg {src[f'oned_{r}']['reg']}, not {r}")
    on, gb = src["oned"], src["gabriel"]
    ond = {f: (on if flow_reg(f) == "h2" else src[f"oned_{flow_reg(f)}"]) for f in flows
           if f != "G1"}
    ns = list(P.ns) if a.smoke else [GH["n"]]
    run = C.open_run(name, a, {"dataset": d, "ns": ns, "set rule": "median hold-out tv_pos",
                               "flows": flows, "regs": regs, "within r": GH["within_r"]},
                     ABOUT.format(name=name, flows=", ".join(flows), d=d,
                                  n=", ".join(map(str, ns)), r=GH["within_r"],
                                  deps=", ".join(C.rel(C.result_path(a, nm, deps=True))
                                                 for nm in dep_names),
                                  out=C.rel(C.result_path(a, name, ""))))

    def gidx(n):
        return [int(x) for x in gb["ns"]].index(n)

    chosen = {n: C1.choose_set(on, d, n) for n in ns}
    for n in ns:
        run(f"N={n}: median set {chosen[n]}")
    pairs = sorted(chosen.items())
    h1 = {(n, s): float(gb[f"{d}/{config.ONED_EVAL['gabriel']['G1']}"][gidx(n), s])
          for n, s in pairs}
    n0, s0 = pairs[0]
    C.worked_example(run, st, P, n0, s0, f"{flows[-1]} (stored lam, no selection)",
                     flow_reg(flows[-1]) if flows[-1] != "G1" else "h2",
                     latent_kind=f"{P.m_lat} quantile midpoints + {P.m_lat} iid",
                     extra={"G1 h' (frame)": h1[n0, s0]})
    jobs = []
    for n, s in pairs:
        for f in flows:
            if f == "G1":
                par = h1[n, s]
            else:
                o = ond[f]
                par = o[f"{d}/holdout/lam"][[int(x) for x in o["ns"]].index(n), s]
            jobs.append((d, n, s, f, par, P))
    if a.mode == "example":
        jobs = [j for j in jobs if j[3] != "G1"][:1]
    save = {}
    worst = [0.0]

    def done(job, r):
        _, n, s, f = job[:4]
        pre = f"{d}/n{n}/s{s}/{f}/"
        jl = f"{d} N={n} set {s} {f}"
        for k, v in r.items():
            if k != "summary":
                save[pre + k] = np.asarray(v)
        o = gb if f == "G1" else ond[f]
        ref = (gab_tv(gb, d, "G1", n, s) if f == "G1"
               else C1.row(o, f"{d}/holdout/tv_pos", n, s))
        save[pre + "tv_ref"] = ref
        dev = abs(r["tv_pos"] - ref)
        if a.mode == "full" and np.isfinite(dev):
            worst[0] = max(worst[0], dev)
        if "lam_rel" in r:
            C.log_velocity(run, jl, f, {"lam_rel": r["lam_rel"]}, P.times)
        run.flow(jl, None, FlowSummary(**r["summary"]), {"generation": round(r["wall"], 3)})
        run.metrics(jl, {"tv_pos": r["tv_pos"], "tv_pos_se": r["tv_pos_se"],
                         "tv_iid": r["tv_iid"], "tv_iid_se": r["tv_iid_se"], "tv_ref": ref,
                         "tv_ref_se": 0.0, "abs_dev": dev, "within": r["within"],
                         "wall": round(r["wall"], 3)}
                    | ({"h": r["h"]} if "h" in r else {}))

    C.pool_map(task, jobs, a.workers, run, done)
    if a.mode == "full":
        run(f"max |tv_pos - stored| = {worst[0]:.1e} "
            + ("OK" if worst[0] <= 1e-6 else "FAILED (> 1e-6)"))
    run(f"summary: fraction of samples within {GH['within_r']:g} of a training point, tv_pos")
    for n, s in pairs:
        run(f"  N={n} set {s}: " + ", ".join(
            f"{f} {float(save[f'{d}/n{n}/s{s}/{f}/within']):.3f} / "
            f"{float(save[f'{d}/n{n}/s{s}/{f}/tv_pos']):.4f}"
            for f in flows if f"{d}/n{n}/s{s}/{f}/within" in save))
    for n in ns:
        save[f"{d}/n{n}/set_median"] = chosen[n]
    run.save(ns=np.array(ns), flows=np.array(flows), dataset=d,
             regs=np.array(regs, dtype=str), times=P.times, **st.record(), **save)
    run.close()


if __name__ == "__main__":
    main()
