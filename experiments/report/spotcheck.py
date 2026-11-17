"""Spot check of a code tree against the release archive of the results: recompute a few cheap
complete runs with this tree and compare their arrays with the archived ones key by key; or,
with --tree-only, compare this tree file by file with another tree. Paper: none (checks that
the public code reproduces the archived results behind the tables and figures).

Usage (from the root of a fresh git clone of the public repository):
  uv run python experiments/report/spotcheck.py --results DIR --out DIR [--workers N]
         [--speed F] [--dry-run]
  uv run python experiments/report/spotcheck.py --tree-only --against DIR

--results  the archived results: the release archive's paper/results directory
           (make fetch-results: RESULTS_ROOT/<name>/paper/results)
--out      scratch directory; the recomputed results go to OUT/results (an existing result
           there is reused, so an interrupted check resumes)
--workers  worker processes per run (default $NWORKERS, else 8)
--speed    time of one core here over one core of the production machines (default 1) for
           the estimate
--dry-run  print the plan, the commands and the estimate only

The runs (SPOT) are complete invocations of experiments/runs.txt: the compute scripts have no
option to restrict a run to some training sets or N (lam_sweep's --n aside), and the check adds
no code path. Their dependencies are copied from --results. Together they exercise the 1D and
2D spline solvers (Cholesky of the normal equations), hold-out selection over a candidate grid,
the RK4 sampler and the TV estimator, a KDE baseline with LSCV, and the stability test. Oracle
selection (oned, twod: 4 to 12 core-hours each) is not covered.

A full run refuses a dirty or non-git tree (runlog.tree_clean), so the check needs a clone;
a `git archive` export fails at the first run.

Comparison (compare_npz): the same keys; strings, integers and booleans exactly; selected
parameters (float keys whose last component starts with 'lam', 'sel', or is c, c0, k, s)
exactly, reporting how many entries changed (a changed hold-out candidate); timing keys (wall,
timing, elapsed) skipped; every other float array: identical NaN pattern, then the largest
absolute and relative difference, FAIL above ABS_FAIL, WARN above ABS_WARN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "experiments", "runs.txt")

# (result, what it exercises); cost from the archived manifests (wall_s x workers)
SPOT = [
    ("lam_sweep_trimodal_n100", "1D solver, hold-out over constant lambda, RK4, TV (N = 100)"),
    ("stability_1d_m256", "1D stability test: perturbed sets, re-selected hold-out lambda, TV, H2"),
    ("kde2d_checkerboard", "2D KDE baselines (fixed, Abramson) with LSCV, TV"),
    ("lam_sweep_checkerboard_n100", "2D solver, hold-out over constant lambda, RK4, TV (N = 100)"),
]
ABS_FAIL, ABS_WARN = 1e-8, 1e-12
VOLATILE = re.compile(r"(^|[/_])(wall|timing|elapsed)", re.I)
SELECTED = re.compile(r"^(lam|sel)|^(c|c0|k|s)$")
SKIP_TREE = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store",
             ".mypy_cache"}
SKIP_TREE_PREFIX = ("results/", "experiments/paper/build/")


# ------------------------------------------------------------------ comparison

def compare_npz(new: dict, old: dict) -> list:
    """[(key, status, detail)] for every key of either mapping; status OK, WARN, FAIL or SKIP."""
    out = []
    for k in sorted(set(new) | set(old)):
        if k not in new or k not in old:
            out.append((k, "FAIL", "only in the " + ("archive" if k in old else "recomputed")))
            continue
        a, b = np.asarray(new[k]), np.asarray(old[k])
        if VOLATILE.search(k):
            out.append((k, "SKIP", "timing"))
            continue
        if a.shape != b.shape:
            out.append((k, "FAIL", f"shape {a.shape} != {b.shape}"))
            continue
        last = k.rsplit("/", 1)[-1]
        if a.dtype.kind in "USObiu" or b.dtype.kind in "USObiu":
            if a.dtype.kind != b.dtype.kind:
                out.append((k, "FAIL", f"dtype {a.dtype} != {b.dtype}"))
                continue
            n = int(np.sum(a != b))
            out.append((k, "OK" if n == 0 else "FAIL", "exact" if n == 0 else
                        f"{n} of {a.size} entries differ"))
            continue
        a, b = a.astype(float), b.astype(float)
        na, nb = np.isnan(a), np.isnan(b)
        if not np.array_equal(na, nb):
            out.append((k, "FAIL", f"NaN pattern differs in {int(np.sum(na != nb))} entries"))
            continue
        fa, fb = a[~na], b[~nb]
        if SELECTED.search(last):
            n = int(np.sum(fa != fb))
            out.append((k, "OK" if n == 0 else "FAIL",
                        "exact" if n == 0 else f"selection changed in {n} of {fa.size} entries"))
            continue
        if fa.size == 0:
            out.append((k, "OK", "empty or all NaN"))
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.abs(fa - fb)
            dabs = float(np.max(d))
            scale = np.maximum(np.abs(fa), np.abs(fb))
            drel = float(np.max(np.where(scale > 0, d / scale, 0.0)))
        st = "FAIL" if dabs > ABS_FAIL else "WARN" if dabs > ABS_WARN else "OK"
        out.append((k, st, f"max abs {dabs:.3g}, max rel {drel:.3g}"))
    return out


def summarize(rows) -> dict:
    """Counts per status, the worst float difference and the overall verdict."""
    cnt = {s: sum(1 for _, t, _ in rows if t == s) for s in ("OK", "WARN", "FAIL", "SKIP")}
    verdict = "FAIL" if cnt["FAIL"] else "WARN" if cnt["WARN"] else "PASS"
    return {**cnt, "verdict": verdict}


def load(path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# ------------------------------------------------------------------ the plan

def run_line(name) -> tuple:
    """(deps, script, args) of a result in experiments/runs.txt."""
    with open(RUNS) as f:
        for ln in f:
            w = ln.split()
            if w and not w[0].startswith("#") and w[0] == name:
                deps = [] if w[2] == "-" else w[2].split(",")
                return deps, w[3], w[4:]
    raise SystemExit(f"spotcheck: {name} is not in experiments/runs.txt")


def estimate(results, name, workers, speed) -> tuple:
    """(core-hours of the archived run, estimated minutes here)."""
    with open(os.path.join(results, name + ".manifest.json")) as f:
        m = json.load(f)
    ch = float(m["wall_s"]) * int(m["workers"]) / 3600
    return ch, ch * 60 * speed / workers


def platform_lines() -> list:
    import platform
    import scipy
    out = [f"python {platform.python_version()}  numpy {np.__version__}  scipy {scipy.__version__}"
           f"  platform {platform.platform()}  machine {platform.machine()}"]
    try:
        import cvxopt
        out.append(f"cvxopt {cvxopt.__version__}")
    except ImportError:
        out.append("cvxopt not installed")
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        out.append(f"numpy BLAS {blas.get('name', '?')} {blas.get('version', '')}".rstrip())
    except (TypeError, AttributeError):
        pass
    return out


# ------------------------------------------------------------------ tree comparison

def tree_files(top) -> dict:
    """Relative path -> sha256 of the files of a tree: git ls-files (tracked and untracked
    unignored, minus deleted) in a git work tree, else every file; SKIP_TREE and
    SKIP_TREE_PREFIX excluded."""
    top = os.path.abspath(top)
    rels = None
    if os.path.isdir(os.path.join(top, ".git")):
        def ls(*a):
            return subprocess.run(["git", "-C", top, "ls-files", "-z", *a], check=True,
                                  capture_output=True).stdout.decode().split("\0")

        try:
            rels = (set(ls()) | set(ls("--others", "--exclude-standard"))) - set(ls("--deleted"))
            rels.discard("")
        except (OSError, subprocess.CalledProcessError):
            rels = None
    if rels is None:
        rels = set()
        for d, dirs, files in os.walk(top):
            dirs[:] = [x for x in dirs if x not in SKIP_TREE]
            rels |= {os.path.relpath(os.path.join(d, f), top).replace(os.sep, "/") for f in files}
    out = {}
    for r in sorted(rels):
        parts = r.split("/")
        if any(p in SKIP_TREE for p in parts) or r.startswith(SKIP_TREE_PREFIX):
            continue
        p = os.path.join(top, r)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                out[r] = hashlib.sha256(f.read()).hexdigest()
    return out


def compare_trees(a, b) -> tuple:
    """(only in a, only in b, differing) between two tree_files mappings, sorted."""
    return (sorted(set(a) - set(b)), sorted(set(b) - set(a)),
            sorted(k for k in set(a) & set(b) if a[k] != b[k]))


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--results")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NWORKERS") or 8))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tree-only", action="store_true")
    ap.add_argument("--against")
    a = ap.parse_args(argv)

    if a.tree_only:
        if not a.against:
            ap.error("--tree-only needs --against DIR")
        ta, tb = tree_files(ROOT), tree_files(a.against)
        oa, ob, diff = compare_trees(ta, tb)
        print(f"this tree {ROOT}: {len(ta)} files; other tree {a.against}: {len(tb)} files")
        for lab, lst in (("only here", oa), ("only there", ob), ("different", diff)):
            print(f"{lab}: {len(lst)}" + "".join(f"\n  {p}" for p in lst))
        ok = not (oa or ob or diff)
        print("IDENTICAL" if ok else "TREES DIFFER")
        sys.exit(0 if ok else 1)

    if not a.results or not a.out:
        ap.error("--results and --out are required (or --tree-only --against DIR)")
    res, out = os.path.abspath(a.results), os.path.abspath(a.out)
    newres = os.path.join(out, "results")
    plan, tot = [], 0.0
    print(f"spot check of {ROOT} against {res}")
    print(f"{'result':30s} {'archived core-h':>15s} {'est. min':>9s}  exercises")
    for name, what in SPOT:
        deps, script, args = run_line(name)
        ch, mins = estimate(res, name, a.workers, a.speed)
        tot += mins
        cmd = [sys.executable, os.path.join("experiments", "compute", script + ".py"), *args,
               "--results", newres, "--workers", str(a.workers)]
        plan.append((name, deps, cmd))
        print(f"{name:30s} {ch:15.2f} {mins:9.1f}  {what}")
    print(f"estimate: {tot:.0f} min with {a.workers} workers (speed factor {a.speed:g}; "
          "archived core-h count idle workers, so this is an upper estimate)")
    for name, deps, cmd in plan:
        print(f"  {name}: deps {', '.join(deps) or '-'}; " + " ".join(cmd[1:]))
    if a.dry_run:
        print("dry run: nothing computed")
        return

    os.makedirs(newres, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", NWORKERS=str(a.workers))
    for name, deps, cmd in plan:
        for d in deps:
            dst = os.path.join(newres, d + ".npz")
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(res, d + ".npz"), dst)
        if os.path.exists(os.path.join(newres, name + ".npz")):
            print(f"{name}: recomputed result exists, reused")
            continue
        t0 = time.time()
        print(f"{name}: running ...", flush=True)
        r = subprocess.run(cmd, cwd=ROOT, env=env)
        if r.returncode != 0:
            sys.exit(f"spotcheck: {name} exited with {r.returncode}")
        print(f"{name}: {(time.time() - t0) / 60:.1f} min")

    print("\nplatform: " + "\n          ".join(platform_lines()))
    print(f"\n{'result':30s} {'verdict':7s} {'OK':>5s} {'WARN':>5s} {'FAIL':>5s} {'SKIP':>5s}")
    bad = 0
    details = []
    for name, _, _ in plan:
        rows = compare_npz(load(os.path.join(newres, name + ".npz")),
                           load(os.path.join(res, name + ".npz")))
        s = summarize(rows)
        bad += s["verdict"] == "FAIL"
        print(f"{name:30s} {s['verdict']:7s} {s['OK']:5d} {s['WARN']:5d} {s['FAIL']:5d} "
              f"{s['SKIP']:5d}")
        details += [f"  {name}: {k}: {st} ({d})" for k, st, d in rows if st in ("WARN", "FAIL")]
    if details:
        print("\nkeys with WARN or FAIL:\n" + "\n".join(details))
    print("\nSPOT CHECK " + ("FAILED" if bad else "PASSED"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
