"""Data of the hero figure: the unregularized and the penalized 1D flow side by
side on one training set of the trimodal target. Paper: Figure 1 (fig:hero).

Both flows run in the frame of oned.py (x = c + s x', s = 0.21, Omega = [-5, 5]); the flow
from the truncated latent to the standardized data is, in the original coordinates, the flow
from the latent scaled by s to the data. Everything stored is in the original coordinates:
w(x, t) = s w'(x', t), trajectories x = c + s x', densities f(x) = f'(x') / s, the latent
density on x_prior, omega_lo/hi the box c +- s L at every image time. TV and e are frame
quantities (TV is invariant; e in frame units, as oned_eval.npz).

Target trimodal, N = config.HERO_1D['n'] (smoke: the smoke N), the median hold-out tv_pos
set of oned.npz; per flow in FLOWS = (exact, holdout):
  exact    the exact v_N (EmpiricalFlow, truncated latent) integrated by the reference, RK4
           with step doubling (sampler.reference_rk4, config.REFERENCE_RK4), exactly as
           oned_eval 'exact_ref' (M_LAT quantile midpoints stacked on M_LAT iid draws); the TV
           printed in the figure is tv_iid.
  holdout  Velocity(all samples).set_lam(lam) with the hold-out lam(t_k) of oned.npz, refit
           mode, RK4 with NSTEP steps (production, as oned_eval); TV = tv_pos.
Stored per flow ('<d>/n<N>/<flow>/<field>'):
  w        (NT, NX) w(x, t) = (1 - t) v = dx/dtau on the image grid: tau at the centres of NT
           equal cells of [0, tau_end], x at the centres of NX cells of x_range. Exact:
           m_N(x, t) - x; penalized: the refit spline at each image time.
  traj, traj_s  trajectories of the K truncated-latent quantile midpoints at SUB x NSTEP + 1
           times uniform in tau. Penalized: the production RK4 (its node states every SUB-th
           point) with a cubic Hermite dense output from states and slopes (traj_end_dev:
           endpoints minus sampler.sample's on the same latents); exact: the same RK4 and
           dense output doubled from NSTEP steps until the path at the output times changes by
           <= REFERENCE_RK4 tol_x (common1d.rk4_path_doubling; Hermite points below 8 NSTEP
           steps, RK4 nodes from 8 NSTEP on); the level used is stored in traj_nstep and
           zoom_nstep (with traj_converged, zoom_converged, traj_err_x, zoom_err_x).
  zoom, zoom_s  ZOOM_K trajectories seeded at the cell midpoints of zoom_window at t = ZOOM_T0
           (a node of the production grid), integrated to t_end as traj; the same seeds for
           every flow. zoom_window: ZOOM_PTS consecutive training points of span <= ZOOM_SPAN
           with the largest smallest gap, bounded by the midpoints to their neighbours.
  curve    density on XF (oned_eval grid). Penalized: interpolation of (x_end, f), also
           curve_cell. Exact: the mass per XF cell (curve_cell of oned_eval; the pointwise
           interpolation is curve_pt) and curve_law: cell averages of the stopped law
           rho_N(., t_end) (StoppedLaw.cell_mass, truncated components) on law_cells fine
           cells of XD (edges xd_edges).
  tv_pos, tv_pos_se, tv_iid, tv_iid_se, tv (the printed one), tv_ref (oned_eval value; the
  reproduction check |tv - tv_ref| <= 1e-6, logged in full runs), nonmono, e (K; oned_eval's),
  e_check (the recomputation, a check against e), lam (K; zeros for exact), wall, frac_out,
  nfev; exact also ref_nstep,
  ref_converged, ref_err_x, ref_err_logf (mean), ref_err_logf_max of the generated points.
Per (target, N): set, ys, hist, pdf_true, lam_s (oned '<d>/oracle/lam_s', the all-sample
lam_s(t_k)), zoom_window. Top level: times, xf, xd_edges, hist_edges, s_img, x_img, t_img,
omega_lo, omega_hi, s_end (tau_end), x_range, flows, datasets, n, latents (frame units),
x_prior, prior_pdf, frame_c, frame_s, frame_L.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from common import population
from common1d import (HC, HIST_EDGES, S_END, SUB, XF, choose_set, exact_errors, exact_extra,
                      image_grid, latents, omega, prior_pdf, rk4_path_doubling, rk4_with_path,
                      spline_errors, split_entry, traj_latents, x_prior, zoom_k0,
                      zoom_seeds, zoom_window)
from smoothfm import EmpiricalField, EmpiricalFlow, StoppedLaw, config
from smoothfm.sampler import FlowSummary, reference_rk4, sample, t_of_tau

FLOWS = ("exact", "holdout")
TARGET = "trimodal"
XD_EDGES = np.linspace(HC["x_range"][0], HC["x_range"][1], HC["law_cells"] + 1)

ABOUT = """hero_1d: data of fig_hero: the exact v_N flow (RK4 with step doubling) beside the
penalized flow with the hold-out lam(t) of oned, on one trimodal training set (N = {n}): w(x, t)
images, trajectories with a zoom inset, generated densities, the stopped law, e(t) and lam(t).
Reads {deps}.
Outputs: {out}.npz (read by the report: fig_hero), {out}.log (this file), {out}.manifest.json."""


# ------------------------------------------------------------------ task

def task(job):
    d, n, s, flow, lam, P, pop, opt = job
    t0 = time.time()
    st = C.setting(d)
    yo, ys = st.training_set(n, s)
    fr = st.frame
    s_img, x_img = image_grid(*opt["image"])
    xs_img = fr.to_std(x_img)
    t_img = t_of_tau(s_img)
    zk = traj_latents(st, opt["k"])
    window = zoom_window(yo)
    k0 = zoom_k0(P.nstep)
    zseed = fr.to_std(zoom_seeds(window, opt["zoom_k"]))
    r = {}
    if flow == "exact":
        M = config.ONED_EVAL["ref_points_smoke"] if P.smoke else P.m_lat
        lat = latents(st, M, s)
        fld = EmpiricalField(ys, st.latent)
        w = fr.s * np.stack([fld.posterior_mean(xs_img[:, None], t)[:, 0] - xs_img
                             for t in t_img])
        fl = EmpiricalFlow(ys, st.latent)
        md = C.reference_doublings(P)
        z, lf, info = reference_rk4(fl, lat, config.T_END, n0=P.nstep, max_doublings=md,
                                    latent=st.latent)
        r.update(split_entry(st, z, lf, M))
        r["tv"], r["tv_se"] = r["tv_iid"], r["tv_iid_se"]
        r["curve_pt"], r["curve"] = r["curve"], r["curve_cell"]
        law = StoppedLaw(ys, config.T_END, st.latent)
        r["curve_law"] = law.cell_mass(fr.to_std(XD_EDGES)) / np.diff(XD_EDGES)
        r["e"] = exact_errors(ys, st, P.times, pop)
        r["lam"] = np.zeros(P.times.size)
        r.update(exact_extra(fl, law, z, lf))
        out = np.any(np.abs(z) > st.L, axis=1)
        r["summary"] = FlowSummary(info["nstep"], config.T_END, int(info["nfev"]),
                                   float(np.mean(out)), float("nan"),
                                   float(np.max(np.abs(z)))).as_dict()
        r.update(C.reference_arrays(info))
        r["infos"] = {"reference": info}
        fl2 = EmpiricalFlow(ys, st.latent)
        r["traj_s"], traj, ti = rk4_path_doubling(fl2, zk, P.nstep, max_doublings=md)
        r["zoom_s"], zt, zi = rk4_path_doubling(fl2, zseed[:, None], P.nstep, k0=k0,
                                                max_doublings=md)
        r["traj"], r["zoom"] = fr.to_phys(traj), fr.to_phys(zt)
        r["infos"].update(trajectories=ti, zoom=zi)
        for k, i in (("traj", ti), ("zoom", zi)):
            r[f"{k}_nstep"], r[f"{k}_converged"] = int(i["nstep"]), bool(i["converged"])
            r[f"{k}_err_x"] = float("nan") if i["err_x"] is None else float(i["err_x"])
    else:
        M = P.m_lat
        lat = latents(st, M, s)
        v = st.velocity(ys, P.times, P.m).set_lam(lam)
        z, lf, summ = sample(v, lat, nstep=P.nstep, t_end=config.T_END)
        r.update(split_entry(st, z, lf, M))
        r["tv"], r["tv_se"] = r["tv_pos"], r["tv_pos_se"]
        r["summary"] = summ.as_dict()
        r["e"] = spline_errors(v, P.times, pop)
        r["lam"] = np.asarray(lam, float)
        r["lam_rel"] = v.rel_lam
        r["traj_s"], traj = rk4_with_path(v, zk, P.nstep)
        zprod, _, _ = sample(v, zk, nstep=P.nstep, t_end=config.T_END)
        r["traj_end_dev"] = float(np.max(np.abs(traj[:, -1] - zprod[:, 0])))
        r["traj"] = fr.to_phys(traj)
        r["zoom_s"], zt = rk4_with_path(v, zseed[:, None], P.nstep, k0=k0)
        r["zoom"] = fr.to_phys(zt)
        w = fr.s * np.stack([v.w(xs_img[:, None], t)[:, 0] for t in t_img])
    r["w"] = np.asarray(w, float)
    r["zoom_window"] = np.array(window)
    r["wall"] = time.time() - t0
    return r


def main():
    p = C.parser(__doc__)
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 1)
    name = "hero_1d"
    oned, ev = C.load(a, "oned"), C.load(a, "oned_eval")
    st0 = C.setting(TARGET)
    C.check_frame(oned, st0, "oned.npz")
    C.check_frame(ev, st0, "oned_eval.npz")
    n = HC["n"] if not a.smoke else config.SMOKE["N"]
    opt = dict(image=HC["image_smoke"] if a.smoke else HC["image"],
               k=HC["k_smoke"] if a.smoke else HC["k"],
               zoom_k=HC["zoom_k_smoke"] if a.smoke else HC["zoom_k"])
    run = C.open_run(name, a, {"targets": [TARGET], "n": n, "set": "median hold-out tv_pos",
                               "flows": FLOWS,
                               "image (nt, nx)": opt["image"], "trajectories": opt["k"],
                               "x range": HC["x_range"], "tau_end": S_END,
                               "reference": C.reference_text(P) + "; trajectories doubled "
                                            "until the path changes by <= tol_x",
                               "law cells": HC["law_cells"], "hermite sub": SUB,
                               "zoom": {k: HC[k] for k in ("zoom_t0", "zoom_pts", "zoom_span",
                                                           "zoom_k")}},
                     ABOUT.format(n=n, deps=", ".join(C.rel(C.result_path(a, nm, deps=True))
                                                      for nm in ("oned", "oned_eval")),
                                  out=C.rel(C.result_path(a, name, ""))))
    names = (TARGET,)
    sets = {d: choose_set(oned, d, n) for d in names}
    run("sets: " + ", ".join(f"{d} {s}" for d, s in sets.items()))
    C.worked_example(run, st0, P, n, sets[TARGET], "holdout (stored lam, no selection)", "h2",
                     latent_kind=f"{P.m_lat} quantile midpoints + {P.m_lat} iid; trajectories "
                                 f"{opt['k']} quantile midpoints",
                     extra={"stored lam": C.rel(C.result_path(a, "oned", deps=True))})
    pops = {d: population(C.setting(d), P.times, P.m_oracle) for d in names}
    jn = [int(x) for x in oned["ns"]].index(n)

    def lam_of(d, f):
        return None if f == "exact" else oned[f"{d}/{f}/lam"][jn, sets[d]]

    jobs = [(d, n, sets[d], f, lam_of(d, f), P, pops[d], opt) for f in FLOWS for d in names]
    if a.mode == "example":
        jobs = [j for j in jobs if j[3] == "holdout"][:1]
    save = {}
    keep = ("w", "traj", "traj_s", "curve", "curve_cell", "curve_pt", "curve_law", "tv", "tv_se",
            "tv_pos", "tv_pos_se", "tv_iid", "tv_iid_se", "e", "lam", "nonmono", "wall",
            "traj_end_dev", "zoom", "zoom_s", "frac_out", "nfev", "n_outside", "max_dlogf",
            "ref_nstep", "ref_converged", "ref_err_x", "ref_err_logf", "ref_err_logf_max", "traj_nstep",
            "traj_converged", "traj_err_x", "zoom_nstep", "zoom_converged", "zoom_err_x")
    jev = [int(x) for x in ev["ns"]].index(n) if n in [int(x) for x in ev["ns"]] else None
    worst = [0.0]

    def done(job, r):
        d, n_, s, f = job[:4]
        pre = f"{d}/n{n_}/{f}/"
        jl = f"{d} N={n_} set {s} {f}"
        save[f"{d}/n{n_}/zoom_window"] = r["zoom_window"]
        meth = "exact_ref" if f == "exact" else f
        key = "tv_iid" if f == "exact" else "tv_pos"
        ref = (float(ev[f"{d}/{meth}/{key}"][jev, s]) if jev is not None
               and f"{d}/{meth}/{key}" in ev else np.nan)
        eref = (np.asarray(ev[f"{d}/{meth}/e"][jev, s], float) if jev is not None
                and f"{d}/{meth}/e" in ev else None)
        for k in keep:
            if k in r:
                save[pre + k] = np.asarray(r[k])
        save[pre + "tv_ref"] = ref
        if eref is not None:
            # the figure plots oned_eval's e(t) (the e(t) arrays of oned_eval.npz); the recomputation
            # is kept as the check
            save[pre + "e_check"] = np.asarray(r["e"])
            save[pre + "e"] = eref
        for lab, info in r.get("infos", {}).items():
            C.log_reference(run, jl, info, lab)
        if r.get("summary"):
            if "lam_rel" in r:
                C.log_velocity(run, jl, f, {"lam_rel": r["lam_rel"]}, P.times)
            run.flow(jl, None, FlowSummary(**r["summary"]), {"generation": round(r["wall"], 3)})
        dev = abs(r["tv"] - ref) if np.isfinite(ref) else np.nan
        if a.mode == "full" and np.isfinite(dev):
            worst[0] = max(worst[0], dev)
        edev = (float(np.max(np.abs(r["e"] - eref) / np.abs(eref))) if eref is not None
                else np.nan)
        run.metrics(jl, {"tv": r["tv"], "tv_se": r["tv_se"], "tv_ref": ref, "tv_ref_se": 0.0,
                         "abs_dev": dev, "e_max_rel_dev": edev,
                         "traj_end_dev": r.get("traj_end_dev", np.nan),
                         "w_max": float(np.nanmax(np.abs(r["w"]))), "wall": round(r["wall"], 3),
                         "zoom_window": np.round(r["zoom_window"], 4).tolist()})

    C.pool_map(task, jobs, a.workers, run, done)
    if a.mode == "full":
        run(f"max |tv - oned_eval| = {worst[0]:.1e} "
            + ("OK" if worst[0] <= 1e-6 else "FAILED (> 1e-6)"))
    for d in names:
        st = C.setting(d)
        pre = f"{d}/n{n}/"
        yo = st.training_set(n, sets[d])[0][:, 0]
        save[pre + "set"] = sets[d]
        save[pre + "ys"] = yo
        save[pre + "hist"] = np.histogram(yo, HIST_EDGES)[0]
        save[pre + "pdf_true"] = st.base.pdf(XF[:, None])
        save[pre + "lam_s"] = oned[f"{d}/oracle/lam_s"][jn, sets[d]]
    s_img, x_img = image_grid(*opt["image"])
    t_img = t_of_tau(s_img)
    lo, hi = omega(st0, t_img)
    run.save(times=P.times, xf=XF, xd_edges=XD_EDGES, hist_edges=HIST_EDGES, s_img=s_img,
             x_img=x_img, t_img=t_img, omega_lo=lo, omega_hi=hi, s_end=S_END,
             x_range=np.array(HC["x_range"]), flows=np.array(FLOWS), datasets=np.array(names),
             n=n, latents=traj_latents(st0, opt["k"])[:, 0], x_prior=x_prior(st0),
             prior_pdf=prior_pdf(st0), **st0.record(), **save)
    run.close()


if __name__ == "__main__":
    main()
