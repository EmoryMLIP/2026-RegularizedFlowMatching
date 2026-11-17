"""8 Gaussians: the 8-component full-covariance Gaussian mixture fitted by EM.
Paper: the "Gaussian mixture, EM" row of Table 3 (tab:twod-full).

Training sets of twod.py for gauss8 (rng SEEDS['train']['gauss8'] N + set), N in config.NS,
config.SETS sets, original coordinates (no frame, no latent, no mu). Fits (config.EM8):
  near   EM (em_from) started from the points nearest to each true centre, covariance ridge
  best   the largest training log-likelihood among `near` and EM8['restarts'] starts from
         random partitions (rng SEEDS['em_restarts'] N + set), same ridge (the row of tab_twod_full)
The ridge bounds the likelihood (unbounded for a full-covariance mixture on collinear points).
tv_pos from KDE2D_DRAWS draws (rng SEEDS['kde_oracle_2d'] + set) with the exact mixture log
density, and its Monte Carlo SE.

Result em8. npz keys (arrays (len(ns), len(sets))): <fit>/{tv_pos, tv_pos_se, ll (mean training
log-likelihood), iters, min_eig, max_eig, wall} for fit in near, best; best/from_near;
<fit>/{weights (8,), means (8, 2), covs (8, 2, 2)}; ns, sets.
"""
import common as C  # noqa: I001  (sets the thread variables before numpy)

import time

import numpy as np

from smoothfm import config, tv_pos
from smoothfm import baselines as B

ABOUT = """em8: EM fits of the 8-component full-covariance Gaussian mixture to the gauss8
training sets (near: nearest-centre start; best: the largest training likelihood of near and
{restarts} random starts), ridge {ridge}.
Each job: one N and set -> the two fits, tv_pos with its SE from {draws} draws each.
Outputs: {out}.npz (read by the report: tab_twod_full EM row = best/tv_pos), {out}.log (this file),
{out}.manifest.json."""

TARGET = "gauss8"
FITS = ("near", "best")


def em_task(job):
    n, s = job
    t0 = time.time()
    st = C.setting(TARGET)
    T = st.base
    ys = st.training_set(n, s)[0]
    E = config.EM8
    K = len(T.mus)
    D = config.KDE2D_DRAWS

    def score(w, mu, S):
        g = np.random.default_rng(config.SEEDS["kde_oracle_2d"] + s)
        k = g.choice(K, size=D, p=w)
        z = mu[k] + np.einsum("nij,nj->ni", np.linalg.cholesky(S)[k], g.standard_normal((D, 2)))
        return tv_pos(T, z, B.mix_logpdf(z, w, mu, S))

    def rec(w, mu, S, ll, its, t1):
        tv, se = score(w, mu, S)
        ev = np.linalg.eigvalsh(S)
        return dict(tv_pos=tv, tv_pos_se=se, ll=ll, iters=its, min_eig=ev.min(), max_eig=ev.max(),
                    weights=w, means=mu, covs=S, wall=time.time() - t1)

    out = {}
    t1 = time.time()
    lab = np.argmin(((ys[:, None] - np.asarray(T.mus)[None]) ** 2).sum(-1), 1)
    kw = dict(ridge=E["ridge"], iters=E["iters"], tol=E["tol"])
    fits = [B.em_from(ys, B.one_hot(lab, K), **kw)]
    out["near"] = rec(*fits[0], t1)
    t1 = time.time()
    rng = np.random.default_rng(config.SEEDS["em_restarts"] * n + s)
    fits += [B.em_from(ys, B.one_hot(rng.integers(0, K, n), K), **kw) for _ in range(E["restarts"])]
    ib = max(range(len(fits)), key=lambda i: fits[i][3])
    out["best"] = rec(*fits[ib], t1)
    out["best"]["from_near"] = float(ib == 0)
    out["wall"] = time.time() - t0
    return out


def main():
    p = C.parser(__doc__)
    a = C.parse(p)
    P = a.protocol = C.Protocol(a, 2)
    name = "em8"
    st = C.setting(TARGET)
    E = config.EM8
    run = C.open_run(name, a, {"target": TARGET, "fits": FITS, **E,
                               "draws": f"{config.KDE2D_DRAWS}, rng {config.SEEDS['kde_oracle_2d']} + set"},
                     ABOUT.format(restarts=E["restarts"], ridge=E["ridge"], draws=config.KDE2D_DRAWS,
                                  out=C.rel(C.result_path(a, name, ""))))
    n0, s0 = P.ns[0], P.sets[0]
    run.example({"job": f"{TARGET} N={n0} set {s0} (all fits)", **st.describe(), "N": n0, "set": s0,
                 "training seed": f"default_rng({config.SEEDS['train'][TARGET]} * {n0} + {s0})",
                 "restarts seed": f"default_rng({config.SEEDS['em_restarts']} * {n0} + {s0})",
                 "EM": E, "coordinates": "original (no frame, no latent)"})
    jobs = [(n, s) for n in P.ns[::-1] for s in P.sets]
    if a.mode == "example":
        jobs = [(n0, s0)]
    ns, sets = list(P.ns), list(P.sets)
    save = C.Store((len(ns), len(sets)), ([f"N{n}" for n in ns], [f"set{s}" for s in sets]))

    def done(job, res):
        n, s = job
        idx = (ns.index(n), sets.index(s))
        m = {}
        for f in FITS:
            save.put_record(f, idx, res[f])
            m |= {f"{f}/tv_pos": res[f]["tv_pos"], f"{f}/tv_pos_se": res[f]["tv_pos_se"],
                  f"{f}/ll": res[f]["ll"], f"{f}/iters": res[f]["iters"]}
        save.put("wall", idx, res["wall"])
        run.metrics(f"{TARGET} N={n} set {s}", m | {"wall": round(res["wall"], 2)})

    C.pool_map(em_task, jobs, a.workers, run, done)

    run("summary: mean (SE over sets) [min, max]")
    for f in FITS:
        for i, n in enumerate(ns):
            run(f"  {f:4s} N={n:5d} tv_pos {C.ms(save[f'{f}/tv_pos'][i])}  mean train loglik "
                f"{np.round(save[f'{f}/ll'][i], 4)}  min eig {np.nanmin(save[f'{f}/min_eig'][i]):.5f}"
                f"  max eig {np.nanmax(save[f'{f}/max_eig'][i]):.3f}")
    for i, n in enumerate(ns):
        run(f"  best from the nearest-centre start: N={n:5d} "
            f"{int(np.nansum(save['best/from_near'][i]))} of {len(sets)}")
    run.save(ns=np.array(ns), sets=np.array(sets), target=TARGET, **save)
    run.close()


if __name__ == "__main__":
    main()
