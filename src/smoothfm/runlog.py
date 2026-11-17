"""Run logs and manifests of the compute scripts (named to avoid stdlib logging).

``run = RunLog(name, args, setup, mode)`` writes the header of results/<name>.log; then
``run.example``, ``run.selection``, ``run.flow``, ``run.metrics`` append, ``run.save`` writes an
npz and registers it, and ``run.close`` writes results/<name>.manifest.json (schema 1). Smoke
runs go to results/smoke/, ``--example`` runs to results/example/ and write no npz. A full run
refuses a dirty tree (git status --porcelain of src/ and experiments/) and an existing output
unless overwrite. Provenance: commit hash plus source_sha256 over the .py files of src/ and
experiments/compute/ (for copies without .git). Cost record: per flow
the wall times, factorizations per time, workers, CPU model and cores (host: HOST, a constant); per flow
also lam / lam_s per time and the fraction of samples leaving Omega.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from . import config

HOST = "unrecorded"

__all__ = ["RunLog", "source_sha256", "git_commit", "tree_clean", "MODES"]

MODES = ("full", "smoke", "example")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*args, root=ROOT):
    r = subprocess.run(["git", "-c", f"safe.directory={root}", *args], cwd=root,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:200]}")
    return r.stdout


def git_commit(root=ROOT) -> str:
    """HEAD hash, or 'unknown' without a usable repository."""
    try:
        return _git("rev-parse", "HEAD", root=root).strip()
    except (RuntimeError, OSError):
        return "unknown"


def tree_clean(root=ROOT) -> bool:
    """True if git status --porcelain of src/ and experiments/ is empty (results excluded)."""
    try:
        out = _git("status", "--porcelain", "src", "experiments", root=root)
    except (RuntimeError, OSError):
        return False
    lines = [ln for ln in out.splitlines() if "experiments/results" not in ln]
    return not lines


def source_sha256(root=ROOT) -> str:
    """sha256 over (relative path, bytes) of the .py files of src/ and experiments/compute/."""
    h = hashlib.sha256()
    for top in ("src", os.path.join("experiments", "compute")):
        for d, dirs, files in sorted(os.walk(os.path.join(root, top))):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for f in sorted(files):
                if f.endswith(".py"):
                    p = os.path.join(d, f)
                    h.update(os.path.relpath(p, root).encode() + b"\0")
                    with open(p, "rb") as fh:
                        h.update(fh.read())
    return h.hexdigest()


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip() or platform.processor()
        with open("/proc/cpuinfo") as f:
            for ln in f:
                if ln.startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def _json(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, "as_dict"):
        return _json(v.as_dict())
    return repr(v)


class RunLog:
    """Log, npz registry and manifest of one compute run (mode full | smoke | example)."""

    def __init__(self, name, args=None, setup=None, mode="full", workers=None, overwrite=None,
                 results_dir=None, repo=ROOT, about=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if workers is None:
            workers = getattr(args, "workers", None) or 1
        if overwrite is None:
            overwrite = bool(getattr(args, "overwrite", False))
        self.name, self.mode, self.workers = name, mode, int(workers)
        base = str(results_dir) if results_dir is not None else os.path.join(repo, "results")
        self.dir = {"full": base, "smoke": os.path.join(base, "smoke"),
                    "example": os.path.join(base, "example")}[mode]
        os.makedirs(self.dir, exist_ok=True)
        self.log_path = os.path.join(self.dir, f"{name}.log")
        self.manifest_path = os.path.join(self.dir, f"{name}.manifest.json")
        self.commit = git_commit(repo)
        self.clean = tree_clean(repo)
        if mode == "full":
            if not self.clean:
                raise RuntimeError("full run on a dirty tree (src/ or experiments/ modified)")
            if not overwrite and (os.path.exists(self.log_path) or os.path.exists(self.manifest_path)):
                raise FileExistsError(f"refusing to overwrite results of {name} (overwrite=True)")
        self.t0 = time.time()
        self.start = datetime.datetime.now().isoformat(timespec="seconds")
        self.manifest = {
            "schema": 1, "name": name, "command": " ".join(sys.argv),
            "argv": list(sys.argv), "args": _json(vars(args) if hasattr(args, "__dict__") else (args or {})), "mode": mode,
            "commit": self.commit, "clean": self.clean, "source_sha256": source_sha256(repo),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": _version("scipy"),
            "cvxopt": _version("cvxopt"), "platform": platform.platform(),
            # the host name is not recorded (published manifests); the key stays for schema 1
            "host": HOST, "cpu_model": _cpu_model(), "cpu_cores": os.cpu_count(),
            "workers": self.workers, "start": self.start, "end": None, "wall_s": None,
            "config": config.as_dict(), "setup": _json(setup or {}), "example": None,
            "jobs": {}, "outputs": [],
        }
        with open(self.log_path, "w") as f:
            f.write("")
        m = self.manifest
        self._write(f"# {name}  commit {self.commit[:12]} {'clean' if self.clean else 'dirty'}  "
                    f"source_sha256 {m['source_sha256'][:12]}  {self.start}  mode {mode}")
        self._write(f"# python {m['python']} numpy {m['numpy']} scipy {m['scipy']} cvxopt "
                    f"{m['cvxopt']}  host {m['host']} ({m['cpu_model']}, "
                    f"{m["cpu_cores"]} cores)  workers {self.workers}")
        self._write(f"# command: {m['command']}")
        self._write(f"# config: MU {config.MU:g}  L {config.L}  NSTEP {config.NSTEP}  T_END "
                    f"{config.T_END}  M_SPLINE {config.M_SPLINE}  LAM_FLOOR_REL {config.LAM_FLOOR_REL:g}")
        if about:
            # what the run computes and where its outputs go, for a reader of the log
            self._write("#")
            for ln in (about.splitlines() if isinstance(about, str) else about):
                self._write(f"# | {ln}".rstrip())
            self._write("#")
        for k, v in (setup or {}).items():
            self._write(f"# {k}: {v}")

    def _write(self, line):
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def __call__(self, line):
        self._write(line)

    def _job(self, job):
        return self.manifest["jobs"].setdefault(str(job), {})

    def example(self, setup_dict):
        """The worked example: the full setup of the first job (target, frame, basis, grid, times,
        integrator, latent, solver), logged once."""
        self.manifest["example"] = _json(setup_dict)
        self._write("== worked example")
        for k, v in setup_dict.items():
            self._write(f"{k}: {v}")

    def selection(self, job, t, rule, sel, residual=None):
        """One per-time selection: the table goes to the log, the counts to the manifest."""
        rel = sel.lam / sel.lam_s if sel.lam_s > 0 else float("nan")
        self._write(f"[{job}] t={t:.6g} {rule}: lam {sel.lam:.4g} (lam/lam_s {rel:.3g}, lam_s "
                    f"{sel.lam_s:.4g})  score {sel.score:.6g}  at_limit {'yes' if sel.at_limit else 'no'}"
                    f"  nsolves {sel.nsolves}" + ("" if residual is None else f"  residual {residual:.2e}"))
        j = self._job(job)
        j.setdefault("rule", rule)
        j["nsolves"] = j.get("nsolves", 0) + int(sel.nsolves)
        j["at_limit"] = j.get("at_limit", 0) + int(sel.at_limit)
        if residual is not None:
            j["max_residual"] = max(j.get("max_residual", 0.0), float(residual))

    def flow(self, job, flow, summary, wall):
        """A generated flow: FlowSummary, walls (dict fit / selection / generation or a number),
        and, for a Velocity, lam / lam_s per time and factorizations per time."""
        j = self._job(job)
        rec = {"summary": summary.as_dict(), "wall": _json(wall)}
        if getattr(flow, "lam", None) is not None and getattr(flow, "lam_s", None) is not None:
            rec["lam_rel"] = _json(np.asarray(flow.lam) / np.asarray(flow.lam_s))
        if getattr(flow, "nsolves", None) is not None:
            rec["nsolves_per_time"] = _json(flow.nsolves)
        j.setdefault("flows", []).append(rec)
        out = summary.frac_outside_end
        self._write(f"[{job}] flow: steps {summary.nstep}  t_end {summary.t_end}  nfev {summary.nfev}  "
                    f"outside Omega {100 * out:.3g}% (stages {100 * summary.frac_outside_any:.3g}%)  "
                    f"max|x| {summary.max_abs:.4g}  wall {wall}")
        if out > 0:
            self._write(f"[{job}] NOTE: {out:.3g} of the generated samples leave Omega")

    def metrics(self, job, values: dict):
        """Scalar metrics of a job (TV with its se, ...)."""
        self._job(job).setdefault("metrics", {}).update(_json(values))
        self._write(f"[{job}] metrics: " + "  ".join(
            f"{k} {v:.6g}" if isinstance(v, (float, int)) else f"{k} {v}" for k, v in values.items()))

    def save(self, path=None, **arrays):
        """np.savez of the arrays into the run's directory (default <name>.npz; a relative
        path is placed there), registered in the manifest; a no-op in example mode."""
        if self.mode == "example":
            self._write("(example mode: no npz written)")
            return None
        path = os.path.join(self.dir, path or f"{self.name}.npz")
        np.savez(path, **arrays)
        self._outputs.append(path)
        return path

    @property
    def _outputs(self):
        return self.__dict__.setdefault("_out_paths", [])

    def close(self):
        """End time, wall, and the output list with sizes and sha256; writes the manifest."""
        m = self.manifest
        m["end"] = datetime.datetime.now().isoformat(timespec="seconds")
        m["wall_s"] = time.time() - self.t0
        self._write(f"== done: {len(m['jobs'])} jobs  wall {m['wall_s']:.0f} s")
        files = self._outputs + [self.log_path]
        m["outputs"] = [{"path": os.path.relpath(p, self.dir), "bytes": os.path.getsize(p),
                         "sha256": _sha_file(p)} for p in files]
        with open(self.manifest_path, "w") as f:
            json.dump(_json(m), f, indent=1)
        return self.manifest_path
