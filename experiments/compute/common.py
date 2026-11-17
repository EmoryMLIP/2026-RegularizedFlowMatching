"""Shared setup of the compute scripts.
Paper: every result behind docs/REPRODUCE.md; no table or figure of its own.

Every constant comes from ``smoothfm.config``; this module only builds objects from it: the
command line (``--smoke``, ``--example``, ``--workers``, ``--overwrite``/OVERWRITE=1,
``--results-dir``), the protocol of the mode (full or the reduced smoke configuration of
config.SMOKE), the targets in their frames, training sets, splits, latents, oracle points, the
lambda search, the parallel job map, and the log records of a fitted velocity and a generated
flow (RunLog of smoothfm.runlog).

Modes. full: config as is, results/<name>.{npz,log,manifest.json}; refuses a dirty tree and
an existing result unless --overwrite or OVERWRITE=1. smoke: config.SMOKE (N = 100, set 0,
m = 64 / 24, 9 times, NSTEP 24, 512 latents), results/smoke/, reads the smoke results of its
dependencies. example: only the worked example (the first job) with the full configuration
(or the smoke one with --smoke), printed and logged to results/example/, no npz; it reads the
dependencies of the configuration it uses.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from smoothfm import (AffineFrame, AffineTarget, Checkerboard, Cubic1D, Cubic2D, EightGaussians,
                      HoldOut, Inhomogeneous, LambdaSearch, NoisyCircle, Oracle, PopulationVelocity,
                      TruncatedGaussian, TrimodalBeta, TwoSpirals, Velocity, config,
                      default_solver, get_regularizer, tv_pos)
from smoothfm.runlog import ROOT, RunLog
from smoothfm.sampler import sample

__all__ = ["parser", "Protocol", "open_run", "load", "result_path", "rel", "Store", "TARGETS_1D", "TARGETS_2D",
           "Setting", "setting", "search", "pool_map", "velocity_record", "log_velocity",
           "generate", "flow_record", "worked_example", "ms", "OracleRule", "HoldOutRule",
           "population", "reference_doublings", "reference_text", "log_reference",
           "reference_arrays", "ProductionLam", "outside", "to_phys_points"]

TARGETS_1D = ("trimodal", "inhom")
TARGETS_2D = ("circle", "checkerboard", "spirals", "gauss8")
_TARGET_CLASS = {"trimodal": TrimodalBeta, "inhom": Inhomogeneous, "circle": NoisyCircle,
                 "checkerboard": Checkerboard, "spirals": TwoSpirals, "gauss8": EightGaussians}
_TARGET_DEF = {
    "trimodal": "TrimodalBeta 0.30 B(8,24) + 0.45 B(30,28) + 0.25 B(24,6) on [-1,1]",
    "inhom": "Inhomogeneous 0.8 B(4,5) + 0.2 B(1500,500) on [-1,1]",
    "circle": "NoisyCircle (von Mises angular mixture, R = 2, radial sd 0.1)",
    "checkerboard": "Checkerboard (8 unit-2 squares on [-4,4]^2)",
    "spirals": "TwoSpirals (two arms, Gaussian noise about the curve)",
    "gauss8": "EightGaussians (8 isotropic components on a circle)",
}


# ------------------------------------------------------------------ command line and modes

def parser(doc: str) -> argparse.ArgumentParser:
    """The common options; scripts add their own."""
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0], allow_abbrev=False)
    p.add_argument("--smoke", action="store_true",
                   help="reduced configuration config.SMOKE (CI); results/smoke/")
    p.add_argument("--example", action="store_true",
                   help="run and print only the worked example (first job); results/example/, no npz")
    p.add_argument("--workers", type=int, default=int(os.environ.get("NWORKERS", 1)),
                   help="worker processes (default env NWORKERS or 1)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace existing full results (also OVERWRITE=1)")
    p.add_argument("--results", "--results-dir", dest="results_dir",
                   default=os.environ.get("SMOOTHFM_RESULTS") or os.path.join(ROOT, "results"),
                   help="results root DIR (default <repo>/results; smoke runs use DIR/smoke, "
                        "--example DIR/example)")
    return p


def parse(p: argparse.ArgumentParser, argv=None):
    a = p.parse_args(argv)
    a.overwrite = bool(a.overwrite or os.environ.get("OVERWRITE", "") == "1")
    a.workers = max(1, int(a.workers))
    a.mode = "example" if a.example else ("smoke" if a.smoke else "full")
    if a.smoke:
        # simplicial CHOLMOD (no BLAS): bitwise repeatable smoke arrays (solvers._supernodal)
        os.environ["SMOOTHFM_CHOLMOD_SUPERNODAL"] = "0"
    return a


class Protocol:
    """The run's configuration: config, or config.SMOKE for --smoke."""

    def __init__(self, args, dim: int):
        self.smoke = bool(args.smoke)
        self.dim = dim
        S = config.SMOKE
        self.ns = (S["N"],) if self.smoke else tuple(config.NS)
        self.sets = (S["SET"],) if self.smoke else tuple(range(config.SETS))
        self.times = np.asarray(S["TIMES"] if self.smoke else config.TIMES, float)
        self.m = (S["M_SPLINE"] if self.smoke else config.M_SPLINE)[dim]
        self.nstep = S["NSTEP"] if self.smoke else config.NSTEP
        self.m_lat = S["M_LAT"] if self.smoke else config.M_LAT[dim]
        self.m_oracle = S["M_LAT"] if self.smoke else config.M_ORACLE[dim]

    def as_dict(self):
        return {"ns": list(self.ns), "sets": list(self.sets), "m": self.m, "nstep": self.nstep,
                "m_lat": self.m_lat, "m_oracle": self.m_oracle, "n_times": int(self.times.size)}


def result_path(args, name: str, ext: str = ".npz", deps: bool = False) -> str:
    """results/<name><ext> in the directory of the mode (deps=True: where this run reads its
    dependencies: smoke/ for --smoke, the full results otherwise)."""
    base = args.results_dir
    if args.smoke and (deps or args.mode == "smoke"):
        return os.path.join(base, "smoke", name + ext)
    if args.mode == "example" and not deps:
        return os.path.join(base, "example", name + ext)
    return os.path.join(base, name + ext)


def rel(path: str) -> str:
    """path relative to the repository root (log text); absolute outside the repository."""
    r = os.path.relpath(path, ROOT)
    return os.path.abspath(path) if r.startswith("..") else r


def load(args, name: str) -> dict:
    """The npz of a dependency (smoke run: its smoke result), as a dict of arrays."""
    path = result_path(args, name, deps=True)
    if not os.path.exists(path):
        raise SystemExit(f"missing dependency {path}: run compute-{name} first"
                         + (" with --smoke" if args.smoke else ""))
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def open_run(name: str, args, setup: dict, about: str) -> RunLog:
    """RunLog of this run: header, the 'what this run computes' block, the setup."""
    setup = dict(setup)
    setup.setdefault("protocol", args.protocol.as_dict() if hasattr(args, "protocol") else None)
    setup.setdefault("cholmod", "simplicial (smoke: bitwise repeatable)" if args.smoke
                     else "supernodal")
    try:
        return RunLog(name, args, setup, args.mode, workers=args.workers,
                      overwrite=args.overwrite, results_dir=args.results_dir, about=about)
    except (RuntimeError, FileExistsError) as e:
        # a refused full run (dirty tree, existing result) ends with a message, no traceback
        raise SystemExit(f"{name}: {e}") from None


# ------------------------------------------------------------------ targets, frames, data

class Setting:
    """One target in its frame: base target (original coordinates), the frame mapping the data
    box onto Omega = [-L, L]^D, the latent, the target in the frame (with the latent)."""

    def __init__(self, name: str, target=None, box=None):
        self.name = name
        self.base = target if target is not None else _TARGET_CLASS[name]()
        self.dim = self.base.dim
        self.L = config.L[self.dim]
        self.box = np.asarray(box if box is not None else config.DATA_BOX[name], float)
        self.frame = AffineFrame.from_box(self.box, self.L)
        self.latent = TruncatedGaussian(self.dim, self.L)
        self.std = AffineTarget(self.base, self.frame, self.latent)
        self.definition = _TARGET_DEF.get(name, repr(self.base))

    # data
    def training_set(self, n: int, s: int):
        """(y original (n, D), y' in the frame): rng SEEDS['train'][name] n + s; 1D sorted."""
        rng = np.random.default_rng(config.SEEDS["train"][self.name] * n + s)
        y = np.asarray(self.base.sample(n, rng), float).reshape(n, self.dim)
        if self.dim == 1:
            y = np.sort(y[:, 0])[:, None]
        return y, self.frame.training_set(y, self.L, self.name, n, s)

    def split(self, ys, s: int):
        """Hold-out split (fit, validation): rng SEEDS['split'] + s, HOLDOUT_FRAC held out;
        1D parts sorted, 2D in permutation order."""
        n = len(ys)
        perm = np.random.default_rng(config.SEEDS["split"] + s).permutation(n)
        k = int(round(config.HOLDOUT_FRAC * n))
        fit, val = ys[perm[k:]], ys[perm[:k]]
        if self.dim == 1:
            fit, val = np.sort(fit, axis=0), np.sort(val, axis=0)
        return fit, val

    # latents and oracle points
    def latents(self, M: int):
        """Production latents: 1D quantile midpoints, 2D scrambled Sobol (rng SEEDS['sobol_2d'])."""
        if self.dim == 1:
            return self.latent.quantile_midpoints(M)
        return self.latent.sobol(M, config.SEEDS["sobol_2d"])

    def iid_latents(self, M: int, s: int):
        """iid latents rng SEEDS['iid'] + s (tv_iid)."""
        return self.latent.sample(M, np.random.default_rng(config.SEEDS["iid"] + s))

    def oracle_points(self, M: int):
        """Fixed (z, y') pairs of the oracle reference points x = (1 - t) z + t y'. 1D: Sobol(2)
        rng SEEDS['oracle_1d'] (column 0 latent quantile, column 1 target quantile); 2D: normal
        draws rng SEEDS['oracle_2d'] (latent (M, 2) via from_normal, then target.sample(M))."""
        if self.dim == 1:
            from scipy.stats import qmc
            u = qmc.Sobol(2, scramble=True, rng=config.SEEDS["oracle_1d"]).random(M)
            return self.latent.ppf(u[:, :1]), self.frame.to_std(self.base.ppf(u[:, 1]))[:, None]
        rng = np.random.default_rng(config.SEEDS["oracle_2d"])
        z = self.latent.from_normal(rng.standard_normal((M, 2)))
        return z, self.frame.to_std(self.base.sample(M, rng))

    def population(self):
        return PopulationVelocity(self.std, self.latent)

    def basis(self, m: int):
        return Cubic1D(m, self.L) if self.dim == 1 else Cubic2D(m, m, self.L)

    def velocity(self, ys_std, times, m, reg="h2"):
        return Velocity(self.basis(m), self.latent, ys_std, times, get_regularizer(reg),
                        solver=default_solver())

    def describe(self) -> dict:
        lat = self.latent
        return {"target": self.definition,
                "frame": f"c={np.round(self.frame.c, 6).tolist()} s={self.frame.s:.6g} L={self.L:g}",
                "data box": self.box.tolist(),
                "latent": f"N(0,I_{self.dim}) | [-{self.L:g},{self.L:g}]^{self.dim}  "
                          f"Z_L={lat.Z:.10g}  TV to N(0,I) {lat.tv_to_gaussian:.3e}"}

    def record(self, prefix: str = "") -> dict:
        return {prefix + "frame_c": self.frame.c, prefix + "frame_s": self.frame.s,
                prefix + "frame_L": float(self.L)}


def setting(name: str) -> Setting:
    return Setting(name)


def check_frame(d: dict, st: Setting, what: str, prefix: str = ""):
    """Refuse a stored lam(t) or h selected in another frame."""
    c, s, half = (np.asarray(d[prefix + k], float) for k in ("frame_c", "frame_s", "frame_L"))
    if not (np.allclose(c, st.frame.c, rtol=0, atol=1e-12) and np.isclose(s, st.frame.s, rtol=1e-12)
            and np.isclose(half, st.L)):
        raise RuntimeError(f"{what}: stored frame c={c} s={s} L={half} differs from c={st.frame.c} "
                           f"s={st.frame.s} L={st.L}")


class ProductionLam:
    """The production result whose stored lam(t) a time control reuses (tend,
    integration_error), loaded and checked on construction.

    1D (target None): oned.npz, both 1D targets, keys <d>/<rule>/<field>, frame checked
    against trimodal's (both 1D targets share it). 2D (target T): twod_<T>.npz, the one target,
    keys <rule>/<field> (twod.py stores one target per file); the penalty must be h2 and the
    spline resolution the run's m. Both: the selection times must be the run's. `suffix` is
    the result-name suffix of the control ('' in 1D, '_<T>' in 2D), so that the 1D results keep
    their names."""

    def __init__(self, args, P, what: str, target: str | None = None):
        if target is None:
            self.dim, self.dep, self.targets, self.suffix = 1, "oned", TARGETS_1D, ""
        else:
            if target not in TARGETS_2D:
                raise ValueError(f"{what}: unknown 2D target {target!r}")
            self.dim, self.dep, self.targets = 2, f"twod_{target}", (target,)
            self.suffix = f"_{target}"
        if P.dim != self.dim:
            raise ValueError(f"{what}: protocol of dimension {P.dim}, source {self.dep} is {self.dim}D")
        self.d = load(args, self.dep)
        self.path = result_path(args, self.dep, deps=True)
        check_frame(self.d, setting(self.targets[0]), f"{self.dep}.npz")
        # atol, not equality: a result computed on another host evaluates config.TIMES to
        # different last bits (twod_circle.npz: 1 ulp, max 1.1e-16)
        if not np.allclose(np.asarray(self.d["times"]), P.times, rtol=0, atol=1e-12):
            raise SystemExit(f"{what}: the selection times of {self.dep}.npz differ from this run's")
        if self.dim == 2:
            reg = str(self.d.get("reg", "h2"))
            if reg != "h2":
                raise SystemExit(f"{what}: {self.dep}.npz has penalty {reg}, not h2")
            if int(self.d["m"]) != P.m:
                raise SystemExit(f"{what}: {self.dep}.npz has m = {int(self.d['m'])}, this run "
                                 f"m = {P.m}")
        self.ns = [int(x) for x in np.atleast_1d(self.d["ns"])]
        self.sets = [int(x) for x in np.atleast_1d(self.d["sets"])]

    def key(self, target: str, rule: str, field: str) -> str:
        return f"{target}/{rule}/{field}" if self.dim == 1 else f"{rule}/{field}"

    def value(self, target: str, rule: str, field: str, n: int, s: int):
        """The stored field of (target, rule) at training-set size n and set s."""
        k = self.key(target, rule, field)
        if k not in self.d or n not in self.ns or s not in self.sets:
            raise SystemExit(f"{self.dep}.npz: no {k} at N={n} set {s}")
        return self.d[k][self.ns.index(n), self.sets.index(s)]

    def latent_text(self, P) -> str:
        """The production latents of the dimension (setup lines)."""
        return (f"{P.m_lat} quantile midpoints" if self.dim == 1 else
                f"{P.m_lat} scrambled Sobol (rng {config.SEEDS['sobol_2d']})")


def outside(st, z) -> float:
    """Fraction of points with some coordinate outside Omega = [-L, L]^D."""
    return float(np.mean(np.any(np.abs(z) > st.L, axis=1)))


def to_phys_points(st, z):
    """Points z (m, D) of the frame in original units: (m,) in 1D (the 1D arrays of the npz
    files), (m, D) otherwise."""
    return st.frame.to_phys(z[:, 0]) if st.dim == 1 else st.frame.to_phys(z)


# ------------------------------------------------------------------ selection rules (picklable)

class OracleRule:
    """rule_factory: Oracle at the fixed reference points (1 - t) z + t y' with the
    population velocity vpop[k] (top-level, picklable)."""

    def __init__(self, z, y, times, vpop):
        self.z, self.y, self.vpop = z, y, vpop
        self.k = {float(t): k for k, t in enumerate(times)}

    def __call__(self, v, t):
        return Oracle(v.normal_equations(t), (1 - t) * self.z + t * self.y,
                      self.vpop[self.k[float(t)]])


class HoldOutRule:
    def __init__(self, yt, yv):
        self.yt, self.yv = yt, yv

    def __call__(self, v, t):
        ne = v.normal_equations(t, self.yt)
        return HoldOut(ne, ne.validation(self.yv))


def population(st, times, M):
    """(z, y', v*(x, t_k)) of the target in the frame at the oracle points, v* (K, M, D)."""
    z, y = st.oracle_points(M)
    pv = st.population()
    return z, y, np.array([pv((1 - t) * z + t * y, t) for t in times])


def search(reg: str, dim: int, include_zero: bool = True) -> LambdaSearch:
    """lam = 10^j lam_s(t), j in config.LAM_DECADES[reg][dim], plus 0 and refinement."""
    return LambdaSearch(tuple(config.LAM_DECADES[reg][dim]), include_zero=include_zero)


def search_text(reg: str, dim: int, include_zero=True) -> str:
    lo, hi = config.LAM_DECADES[reg][dim]
    return (f"lam = {'0 (floor 1e-14 lam_s) + ' if include_zero else ''}10^[{lo}..{hi - 1}] "
            f"lam_s(t) ({hi - lo}) + half-decade refinement")


# ------------------------------------------------------------------ jobs

def _indexed(args):
    fn, i, job = args
    return i, fn(job)


def _label(job):
    """The leading scalar fields of a job tuple (arrays carried by jobs stay out of the log)."""
    if not isinstance(job, tuple):
        return str(job)
    lead = []
    for x in job:
        if not isinstance(x, (str, int, float, np.integer, np.floating, bool)):
            break
        lead.append(x)
    return str(tuple(lead))


def pool_map(fn, jobs, workers: int, run=None, callback=None):
    """Parallel map over jobs (processes, not daemonic, so a job may use a TimePool); results
    in job order; one progress line per finished job; callback(job, result) in completion
    order (the log is appended as jobs finish). ``workers`` = 1: serial in this process."""
    out = [None] * len(jobs)
    t0 = time.time()

    def done(k, i, r):
        out[i] = r
        if callback:
            callback(jobs[i], r)
        if run:
            run(f"  [{k + 1}/{len(jobs)}] {_label(jobs[i])} done, {time.time() - t0:.0f} s")

    if workers <= 1 or len(jobs) <= 1:
        for i, j in enumerate(jobs):
            done(i, i, fn(j))
        return out
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        futs = [ex.submit(_indexed, (fn, i, j)) for i, j in enumerate(jobs)]
        try:
            for k, f in enumerate(as_completed(futs)):
                i, r = f.result()
                done(k, i, r)
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    return out


# ------------------------------------------------------------------ records and log lines

def velocity_record(v: Velocity, t_fit: float) -> dict:
    """Per-time arrays of a fitted Velocity for npz and log: lam, lam_s, lam/lam_s, at_limit,
    nsolves (factorizations), nevals (scores evaluated), residual, backward_error, wall per t,
    the search
    tables (K, k_max, 2) NaN-padded, and wall_fit (s; timing keys contain 'wall' so that
    the reproducibility check of two smoke runs skips them)."""
    K = v.times.size
    rec = {"lam": np.asarray(v.lam, float), "lam_s": np.asarray(v.lam_s, float),
           "wall_fit": float(t_fit)}
    rec["lam_rel"] = rec["lam"] / rec["lam_s"]
    if v.selections is not None:
        tabs = [np.asarray(s.table, float) for s in v.selections]
        kmax = max(len(t) for t in tabs)
        T = np.full((K, kmax, 2), np.nan)
        for k, tab in enumerate(tabs):
            T[k, :len(tab)] = tab
        rec.update(tables=T, scores=np.array([s.score for s in v.selections]),
                   at_limit=np.asarray(v.at_limit, bool), nsolves=np.asarray(v.nsolves, int),
                   nevals=np.array([len(t) for t in tabs]), wall_t=np.asarray(v.wall, float),
                   residual=np.asarray(v.residual, float),
                   backward_error=np.asarray(v.backward_error, float),
                   limit_scores=np.array([s.limit_score for s in v.selections]))
    return rec


def _fmt_rel(r):
    if r == 0:
        return "0"
    e = np.log10(r)
    return f"1e{e:.0f}" if abs(e - round(e)) < 1e-6 else f"1e{e:.1f}"


def log_velocity(run: RunLog, job: str, rule: str, rec: dict, times):
    """The per-time convergence history of one fitted velocity: every candidate's score
    (lam/lam_s: score), the selection, lam/lam_s, at-limit flag, factorizations, the normwise
    backward error of the selected solve."""
    if "tables" not in rec:
        run(f"[{job}] {rule}: lam fixed (no selection)  lam/lam_s "
            + " ".join(f"{x:.3g}" for x in rec["lam_rel"]))
        return
    j = run._job(job)
    j.setdefault("rule", rule)
    for k, t in enumerate(times):
        tab = rec["tables"][k]
        tab = tab[np.isfinite(tab[:, 0])]
        ls = rec["lam_s"][k]
        cells = " ".join(f"{_fmt_rel(l / ls)}:{s:+.6e}" for l, s in tab)
        run(f"[{job}] t={t:.6g} {rule} lam/lam_s: {cells}")
        run(f"[{job}]    -> lam {rec['lam'][k]:.4g} (lam/lam_s {_fmt_rel(rec['lam_rel'][k])}, "
            f"lam_s {ls:.4g})  score {rec['scores'][k]:+.6e}  at_limit "
            f"{'yes' if rec['at_limit'][k] else 'no'}  factorizations {rec['nsolves'][k]}  "
            f"solves {rec['nevals'][k]}  backward_error {rec['backward_error'][k]:.2e}  wall "
            f"{rec['wall_t'][k]:.2f} s")
    j["nsolves"] = j.get("nsolves", 0) + int(np.sum(rec["nsolves"]))
    j["at_limit"] = j.get("at_limit", 0) + int(np.sum(rec["at_limit"]))
    j["max_backward_error"] = max(j.get("max_backward_error", 0.0),
                                  float(np.nanmax(rec["backward_error"])))
    j["wall_fit"] = j.get("wall_fit", 0.0) + rec["wall_fit"]


def reference_doublings(P) -> int:
    """Doublings of sampler.reference_rk4: REFERENCE_RK4, capped in smoke runs."""
    md = config.REFERENCE_RK4["max_doublings"]
    return min(md, config.SMOKE_REFERENCE_DOUBLINGS) if P.smoke else md


def reference_text(P) -> str:
    """The reference integrator of the exact v_N, for setup lines and table notes."""
    R = config.REFERENCE_RK4
    md = reference_doublings(P)
    return (f"RK4 with step doubling from {P.nstep} steps, converged when two consecutive levels "
            f"differ by <= {R['tol_x']:g} in x (frame units, max over points) and <= "
            f"{R['tol_logf']:g} in log f (mean over points), at most {md} doublings"
            + (f" (smoke cap; full runs {R['max_doublings']})" if md != R["max_doublings"] else ""))


def log_reference(run: RunLog, job: str, info: dict, label: str = "reference"):
    """The per-level history of one sampler.reference_rk4 run (nstep, dx, dlog f against the
    previous level, wall) and its verdict on a check line; the record goes to the job's
    manifest entry, a non-converged run also to the manifest's top-level list."""
    R = config.REFERENCE_RK4
    for lv in info["levels"]:
        def f(k):
            return "-" if lv.get(k) is None else f"{lv[k]:.3e}"
        run(f"[{job}] {label} RK4 nstep {lv['nstep']}: dx {f('dx')}  dlogf (mean) {f('dlogf')}  "
            f"dlogf_max {f('dlogf_max')}  wall {lv['wall']:.2f} s")

    def g(k):
        return "-" if info.get(k) is None else f"{info[k]:.2e}"
    verdict = "converged" if info["converged"] else "NOT CONVERGED"
    run(f"[{job}] check: {label} {verdict} at nstep {info['nstep']}  err_x {g('err_x')} (max, tol "
        f"{R['tol_x']:g})  err_logf {g('err_logf')} (mean, tol {R['tol_logf']:g})  err_logf_max "
        f"{g('err_logf_max')}  nfev {info['nfev']}")
    rec = {"label": label, **{k: info.get(k) for k in ("nstep", "converged", "err_x", "err_logf",
                                                       "err_logf_max", "nfev", "levels")}}
    run._job(job).setdefault("reference", []).append(rec)
    if not info["converged"]:
        run.manifest.setdefault("reference_not_converged", []).append(
            {"job": job, "label": label, "nstep": info["nstep"], "err_x": info["err_x"],
             "err_logf": info["err_logf"], "err_logf_max": info.get("err_logf_max")})


def reference_arrays(info: dict) -> dict:
    """npz scalars of a reference run: final nstep, converged, err_x (max), err_logf (mean),
    err_logf_max, nfev."""
    def val(k):
        return float("nan") if info.get(k) is None else float(info[k])
    return {"ref_nstep": int(info["nstep"]), "ref_converged": bool(info["converged"]),
            "ref_err_x": val("err_x"), "ref_err_logf": val("err_logf"),
            "ref_err_logf_max": val("err_logf_max"), "nfev": int(info["nfev"])}


def generate(v, x0, target, nstep: int, workers: int = 1) -> dict:
    """RK4 generation (config.T_END) from latent points x0; TV against target (the target in
    the frame) with its Monte Carlo SE; summary (FlowSummary) and generation wall time."""
    t0 = time.time()
    z, lf, summ = sample(v, x0, nstep=nstep, t_end=config.T_END, workers=workers)
    tv, se = tv_pos(target, z, lf)
    return {"z": z, "lf": lf, "tv_pos": tv, "tv_pos_se": se, "summary": summ,
            "wall_gen": time.time() - t0}


def flow_record(g: dict) -> dict:
    """The scalars of a generation for npz and log: tv_pos, tv_pos_se, wall_gen, frac_out,
    frac_out_any, nfev, max_abs, and the FlowSummary as a dict ('summary', not for npz)."""
    s = g["summary"]
    return {"tv_pos": g["tv_pos"], "tv_pos_se": g["tv_pos_se"], "wall_gen": g["wall_gen"],
            "frac_out": s.frac_outside_end, "frac_out_any": s.frac_outside_any,
            "nfev": s.nfev, "max_abs": s.max_abs, "summary": s.as_dict()}


def log_flow(run: RunLog, job: str, rec: dict, walls: dict, label: str = ""):
    """Integrator summary (steps, t_end, nfev, fraction leaving Omega), walls,
    lam / lam_s per time (if rec has them) and tv_pos with its SE, from a job record."""
    from smoothfm.sampler import FlowSummary
    walls = {k: round(float(x), 3) for k, x in walls.items()}
    run.flow(job, None, FlowSummary(**rec["summary"]), walls)
    m = {f"{label}tv_pos": rec["tv_pos"], f"{label}tv_pos_se": rec["tv_pos_se"]}
    if "lam_rel" in rec:
        m[f"{label}lam/lam_s"] = " ".join(f"{x:.3g}" for x in rec["lam_rel"])
        run._job(job).setdefault("lam_rel", []).append(np.asarray(rec["lam_rel"]).tolist())
    run.metrics(job, m)


def worked_example(run: RunLog, st: Setting, P: Protocol, N: int, s: int, rule: str, reg: str,
                   extra: dict | None = None, split_sizes=None, latent_kind=None):
    """The worked example block of the first job: its full setup, logged once."""
    basis = st.basis(P.m)
    tau_end = -np.log1p(-config.T_END)
    d = {"job": f"{st.name} N={N} set {s} rule {rule}", **st.describe(),
         "N": N, "set": s, "training seed": f"default_rng({config.SEEDS['train'].get(st.name)} * {N} + {s})",
         "split": (f"default_rng({config.SEEDS['split']} + {s}): fit {split_sizes[0]} / val "
                   f"{split_sizes[1]}") if split_sizes else "none",
         "spline": f"cubic C^2, m={P.m} cells per axis, n={basis.size} basis functions on "
                   f"[-{st.L:g},{st.L:g}]^{st.dim}",
         "mu (mass term, frame units)": config.MU, "regularizer": reg,
         "lam grid": search_text(reg, st.dim),
         "times": f"{P.times.size}: " + " ".join(f"{t:.6g}" for t in P.times),
         "integrator": f"RK4 in tau, {P.nstep} steps, t_end {config.T_END}, tau_end {tau_end:.4f}, "
                       f"step {tau_end / P.nstep:.5f}",
         "latent points": latent_kind or (f"{P.m_lat} quantile midpoints" if st.dim == 1 else
                                          f"{P.m_lat} scrambled Sobol, rng {config.SEEDS['sobol_2d']}"),
         "solver": type(default_solver()).__name__}
    d.update(extra or {})
    run.example(d)


#: Per-time selection history of a fitted velocity (velocity_record keys), stored per job
#: under <prefix>/<job label>/ (<prefix>/tables (K, k_max, 2), at_limit (K,),
#: nsolves (K,), residual (K,)); "tables" is stored only there.
HISTORY = ("tables", "at_limit", "nsolves", "nevals", "residual", "backward_error", "scores",
           "limit_scores", "lam", "lam_s", "lam_rel", "wall_t")


class Store(dict):
    """npz arrays filled job by job: put(key, idx, value) creates key with the leading shape
    (NaN-filled float) on first use and grows trailing axes (NaN padding) when a later value is
    larger. labels: one list of names per leading axis (default i<k>), used for the per-job
    history prefix <prefix>/<label_0>_<label_1>/ of put_record. Strings: set directly."""

    def __init__(self, shape, labels=None):
        super().__init__()
        self.shape = tuple(shape)
        self.labels = labels

    def put(self, key, idx, value):
        value = np.asarray(value, float)
        if key not in self:
            self[key] = np.full(self.shape + value.shape, np.nan)
        arr = self[key]
        tail = arr.shape[len(self.shape):]
        if len(tail) != value.ndim:
            raise ValueError(f"{key}: value of shape {value.shape}, stored {tail}")
        if any(v > t for v, t in zip(value.shape, tail)):
            new = np.full(self.shape + tuple(max(v, t) for v, t in zip(value.shape, tail)), np.nan)
            new[tuple(slice(0, n) for n in arr.shape)] = arr
            self[key] = arr = new
        arr[idx + tuple(slice(0, n) for n in value.shape)] = value

    def job_label(self, idx):
        if self.labels is None:
            return "_".join(f"i{k}" for k in idx)
        return "_".join(str(lab[k]) for lab, k in zip(self.labels, idx))

    def put_record(self, prefix, idx, rec, keys=None):
        """put every numeric entry of a job record (or those in keys) under prefix/<key>,
        except 'tables'; the HISTORY entries also go, unstacked, to prefix/<job label>/<key>."""
        pre = f"{prefix}/" if prefix else ""
        for k, v in rec.items():
            if (keys is None or k in keys) and not isinstance(v, (dict, str)):
                if k in HISTORY:
                    self[f"{pre}{self.job_label(idx)}/{k}"] = np.asarray(v)
                if k != "tables":
                    self.put(f"{pre}{k}", idx, v)


def ms(v) -> str:
    """mean (SE over sets) [min, max]."""
    v = np.asarray(v, float).ravel()
    v = v[np.isfinite(v)]
    if not v.size:
        return "nan"
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else float("nan")
    return f"{v.mean():.4f} ({se:.4f}) [{v.min():.4f}, {v.max():.4f}]"
