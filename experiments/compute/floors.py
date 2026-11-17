"""Population TV floors in 1D: the population velocity of each target
through the production sampler at config.FLOORS_STEP_FACTORS x NSTEP RK4 steps (48, 96, 192 at
NSTEP = 96) to T_END. The floor is the tv_pos of the population flow; it bounds the
integration and estimator error of every flow TV. Paper: Sec. 5.2 "Experimental Setup"
(1.0e-4 and 2.5e-4, the same order with 48 and 192 steps; floors.log). The report reads no
number from it.

Targets trimodal and inhom in the frame of oned.py; latent: the truncated latent, 3000
quantile midpoints (the production latents). The velocity is PopulationVelocity of the target
in the frame: the truncated posterior mean and its Jacobian trace with the boundary terms of
the indicator, so log f is transported exactly up to the integrator.

npz keys: <target>/<steps>/{tv_pos, tv_pos_se, wall_gen, frac_out, frac_out_any, nfev,
max_abs} (scalars); <target>_<steps> = [tv_pos, tv_pos_se] (index 0 tv_pos);
nstep (production), steps, targets, frame_c, frame_s, frame_L.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import numpy as np

from smoothfm import config

ABOUT = """floors: population TV floors in 1D (trimodal, inhom): the exact population velocity
(truncated posterior, Jacobian trace with boundary terms) through RK4 with {steps} steps to
t_end, from {m} truncated-latent quantile midpoints; tv_pos with its Monte Carlo SE.
Each job: one target and step count.
Outputs: {out}.npz (read by the report: the floor sentence of sec:setup), {out}.log (this
file), {out}.manifest.json (provenance, cost, output checksums)."""


def steps_of(P):
    return [int(round(f * P.nstep)) for f in config.FLOORS_STEP_FACTORS]


def task(job):
    name, nstep, P = job
    st = C.setting(name)
    pv = st.population()
    rec = C.flow_record(C.generate(pv, st.latents(P.m_lat), st.std, nstep))
    rec["nfev_posterior"] = pv.nfev
    return rec


def main():
    p = C.parser(__doc__)
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 1)
    name = "floors"
    steps = steps_of(P)
    run = C.open_run(name, a, {"targets": C.TARGETS_1D, "steps": steps,
                               "latents": f"{P.m_lat} truncated-latent quantile midpoints"},
                     ABOUT.format(steps=steps, m=P.m_lat, out=C.rel(C.result_path(a, name, ""))))
    st0 = C.setting("trimodal")
    C.worked_example(run, st0, P, 0, 0, f"population flow, {steps[0]} steps", "h2",
                     extra={"N": "none (population velocity)", "set": "none",
                            "training seed": "none", "spline": "none", "regularizer": "none",
                            "lam grid": "not used",
                            "velocity": "PopulationVelocity: truncated posterior mean, "
                                        "Jacobian trace with the indicator's boundary terms"})
    names = C.TARGETS_1D
    jobs = [(d, k, P) for d in names for k in steps]
    if a.mode == "example":
        jobs = jobs[:1]
    out = {}

    def done(job, rec):
        d, k = job[:2]
        jl = f"{d} steps {k}"
        C.log_flow(run, jl, rec, {"generation": rec["wall_gen"]})
        out[(d, k)] = rec

    C.pool_map(task, jobs, a.workers, run, done)
    save = {}
    run("summary: target  steps  tv_pos (SE)")
    for (d, k), rec in sorted(out.items()):
        run(f"  {d:9s} {k:4d}  {rec['tv_pos']:.3e} ({rec['tv_pos_se']:.1e})")
        for key in ("tv_pos", "tv_pos_se", "wall_gen", "frac_out", "frac_out_any", "nfev",
                    "max_abs"):
            save[f"{d}/{k}/{key}"] = np.asarray(rec[key], float)
        save[f"{d}_{k}"] = np.array([rec["tv_pos"], rec["tv_pos_se"]])
    run.save(nstep=P.nstep, steps=np.array(steps), targets=np.array(names), **st0.record(),
             **save)
    run.close()


if __name__ == "__main__":
    main()
