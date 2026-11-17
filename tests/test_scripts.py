"""Compute scripts (smoke and example runs, logs, manifests, Monte Carlo SEs, cost records)
and the CI workflow.

Tiers.
- Unmarked (fast): the scripts exist; no script redefines a config constant or offers a CLI
  flag for MU, L, the latent, NSTEP or T_END; the identifier ``eps`` occurs nowhere in the
  release code; OVERWRITE=1 maps to ``--overwrite``; the CI workflow has the three jobs.
- ``@pytest.mark.script`` (also ``slow``, so ``-m "not slow"`` keeps excluding them): every
  script run as a subprocess with ``--smoke`` / ``--example`` / a full-run refusal, into
  pytest temporary directories. CI runs them as their own job, one group of the run list
  per matrix entry: ``pytest -m script -k g1`` (g1 compare_time, g2 the other 1D runs and
  gauss8, g3 the circle and checkerboard, g4 spirals). Each group runs the dependencies it
  needs.
  ``test_report_from_smoke_results`` needs every group and carries no group id; run it with
  ``pytest -m script -k report_from_smoke`` (it runs every smoke invocation, about 15 min).

Interface the tests assume (see also tests/release_helpers.py):
- The per-(set, t) histories are checked in the npz (``<prefix>tables`` (K, k_max, 2) with
  columns (lam, score), ``<prefix>at_limit``, ``<prefix>nsolves``, ``<prefix>residual`` (K,))
  and in the log (one ``lam/lam_s`` line per selection); the manifest does not hold them.
- A TV metric is a manifest metric whose name contains ``tv`` (bounds excluded); it needs
  its se as ``<name>_se`` or ``se_<name>``, a dict with ``se``, or a pair (estimate, se).
- "Fails with a clear message" means a nonzero exit whose output names a missing dependency
  result and contains no Python traceback.
- ``--example`` is tested for the scripts without dependencies, since a dependent
  script's example has no defined input location.
- Two smoke runs give identical arrays except keys naming wall, timing, elapsed, host, cost.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tokenize

import numpy as np
import pytest

from release_helpers import (COMPUTE, DEP_INVS, INVOCATIONS, NO_FLOW, NO_TV, ROOT, ROOT_INVS,
                             SCRIPTS, SELECT, SMOKE_REFERENCE, first_invocation, npz_summaries,
                             outcome_mismatches, run_names, run_report, run_script,
                             summary_mismatches, tree_files,
                             yaml_load, read_outputs_txt, PAPER_INPUTS)

MANIFEST_KEYS = ["schema", "name", "command", "argv", "commit", "clean", "source_sha256",
                 "python", "numpy", "scipy", "cvxopt", "platform", "host", "workers",
                 "start", "end", "wall_s", "config", "example", "jobs", "outputs"]
VOLATILE = re.compile(r"wall|timing|elapsed|host|cost|cpu", re.I)


def _ids(invs):
    return [f"{i.group}-{i.id}" for i in invs]


def _py_files(*tops):
    for top in tops:
        for p in sorted((ROOT / top).rglob("*.py")):
            rel = p.relative_to(ROOT).as_posix()
            if "/results" in rel or "__pycache__" in rel or rel.startswith(".venv"):
                continue
            yield p


# ===========================================================================
# Fast static checks
# ===========================================================================

@pytest.mark.parametrize("name", SCRIPTS)
def test_script_exists(name):
    assert (COMPUTE / f"{name}.py").is_file(), f"experiments/compute/{name}.py missing"


def _config_names():
    import smoothfm.config as config
    return {k for k in vars(config) if k.isupper() and not k.startswith("_")}


LOWER_CONSTANTS = {"mu", "nstep", "t_end", "tend", "lam_floor_rel", "split_ratio"}


_ARITH = (ast.Constant, ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop)


def _literal(node):
    """The number a constant arithmetic expression (1e-8, 1 - 1e-4, 10 ** -4) denotes."""
    if not all(isinstance(n, _ARITH) for n in ast.walk(node)):
        return None
    try:
        v = eval(compile(ast.Expression(node), "<c>", "eval"), {"__builtins__": {}})
    except Exception:
        return None
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def test_no_script_redefines_config_constants():
    names = _config_names()
    assert {"MU", "L", "NSTEP", "T_END", "TIMES"} <= names
    bad = []
    for p in _py_files("experiments"):
        tree = ast.parse(p.read_text(), filename=str(p))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and n.id in names:
                        bad.append(f"{p.relative_to(ROOT)}:{node.lineno} assigns {n.id}")
                    if (isinstance(n, ast.Name) and n.id.lower() in LOWER_CONSTANTS
                            and isinstance(node, ast.Assign)
                            and _literal(node.value) is not None):
                        bad.append(f"{p.relative_to(ROOT)}:{node.lineno} {n.id} = literal")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                pos = a.posonlyargs + a.args
                pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + \
                    [(x, d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
                for arg, d in pairs:
                    if (arg.arg.lower() in LOWER_CONSTANTS or arg.arg in names) and \
                            _literal(d) is not None:
                        bad.append(f"{p.relative_to(ROOT)}:{node.lineno} default "
                                   f"{arg.arg}={ast.unparse(d)}")
    assert not bad, "constants defined outside config:\n" + "\n".join(bad)


FORBIDDEN_FLAGS = re.compile(r"^--?(mu|l|big-l|nstep|nsteps|t-end|t_end|tend|latent|prior|eps"
                             r"|lam0|merge|resume|extend.*|allow-code-change)$", re.I)


def test_no_cli_flag_changes_the_model():
    bad = []
    for p in _py_files("experiments/compute"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) and \
                            FORBIDDEN_FLAGS.match(a.value):
                        bad.append(f"{p.relative_to(ROOT)}:{node.lineno} {a.value}")
    assert not bad, "\n".join(bad)


def _code_files():
    skip = ("docs/", ".venv/", ".git/", ".pytest_cache/")
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if not p.is_file() or rel.startswith(skip) or "/results" in rel or \
                rel.startswith("results") or "__pycache__" in rel:
            continue
        yield p, rel


def test_identifier_eps_absent():
    bad = []
    for p, rel in _code_files():
        if p.suffix == ".py":
            toks = tokenize.generate_tokens(io.StringIO(p.read_text()).readline)
            bad += [f"{rel}:{t.start[0]}" for t in toks
                    if t.type == tokenize.NAME and t.string == "eps"]
        elif p.suffix in (".sh", ".yml", ".yaml", ".toml") or p.name == "Makefile":
            for k, ln in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if re.search(r"\beps\b", ln):
                    bad.append(f"{rel}:{k}")
    assert not bad, "identifier eps found:\n" + "\n".join(bad)


def test_make_overwrite_maps_to_flag():
    def dry(*extra):
        r = subprocess.run(["make", "-n", "-C", str(ROOT), "compute-oned", *extra],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout
    assert "--overwrite" not in dry()
    assert "--overwrite" in dry("OVERWRITE=1")


# ===========================================================================
# CI workflow
# ===========================================================================

def _ci():
    p = ROOT / ".github" / "workflows" / "ci.yml"
    assert p.is_file(), ".github/workflows/ci.yml missing"
    doc = yaml_load(p.read_text())
    assert isinstance(doc, dict) and isinstance(doc.get("jobs"), dict), "no jobs mapping"
    return doc


def _job_text(job):
    return json.dumps(job)


def _classify(doc):
    kinds = {}
    for name, job in doc["jobs"].items():
        runs = " ".join(str(s.get("run", "")) for s in job.get("steps", []) if isinstance(s, dict))
        if re.search(r"-m\s+['\"]?script|make\s+smoke", runs):
            kinds.setdefault("smoke", name)
        elif re.search(r"-m\s+['\"]?report|test_report|make\s+report", runs):
            kinds.setdefault("report", name)
        elif "pytest" in runs and re.search(r"not\s+script", runs):
            kinds.setdefault("unit", name)
    return kinds


def _matrix(job):
    return (job.get("strategy") or {}).get("matrix") or {}


def test_ci_three_jobs():
    kinds = _classify(_ci())
    assert set(kinds) == {"unit", "smoke", "report"}, kinds
    assert len(set(kinds.values())) == 3


def test_ci_python_versions_and_runner():
    doc = _ci()
    for name, job in doc["jobs"].items():
        assert "ubuntu-latest" in str(job.get("runs-on")), name
        m = _matrix(job)
        py = m.get("python-version", m.get("python"))
        assert py is not None, f"{name}: no python matrix"
        assert {"3.12", "3.13"} <= {str(x) for x in py}, (name, py)


def test_ci_uv_frozen_lock_and_extras():
    assert (ROOT / "uv.lock").is_file()
    doc = _ci()
    for name, job in doc["jobs"].items():
        t = _job_text(job)
        assert "setup-uv" in t or "install uv" in t or "pip install uv" in t, name
        sync = re.findall(r"uv sync[^\"\\]*", t)
        assert sync, f"{name}: no uv sync"
        s = " ".join(sync)
        assert "--frozen" in s or "--locked" in s, (name, s)
        assert "--extra cholmod" in s or "--all-extras" in s, (name, s)


def test_ci_single_threaded_blas():
    doc = _ci()
    for name, job in doc["jobs"].items():
        env = dict(doc.get("env") or {})
        env.update(job.get("env") or {})
        for s in job.get("steps", []):
            if isinstance(s, dict):
                env.update(s.get("env") or {})
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert str(env.get(v)) == "1", (name, v, env.get(v))


def test_ci_smoke_in_four_groups():
    doc = _ci()
    job = doc["jobs"][_classify(doc)["smoke"]]
    m = _matrix(job)
    groups = [v for k, v in m.items() if k not in ("python-version", "python", "include",
                                                     "exclude", "os")]
    assert any(isinstance(v, list) and len(v) == 4 for v in groups), m


def test_ci_unit_job_excludes_scripts():
    doc = _ci()
    job = doc["jobs"][_classify(doc)["unit"]]
    t = _job_text(job)
    assert re.search(r"not\s+slow", t) and re.search(r"not\s+script", t), t[:400]


# ===========================================================================
# Smoke runs (subprocess, temporary results root)
# ===========================================================================

class SmokeCache:
    """Runs each invocation once per session after its dependencies, all into one root."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.done = {}

    @property
    def dir(self):
        return self.root / "smoke"

    def ensure(self, inv_id):
        if inv_id in self.done:
            return self.done[inv_id]
        inv = INVOCATIONS[inv_id]
        for d in inv.deps:
            r = self.ensure(d)
            if r.returncode != 0:
                pytest.fail(f"dependency {d} of {inv_id} failed:\n{r.stderr[-3000:]}")
        r = run_script(inv.script, ["--smoke", *inv.args], self.root)
        self.done[inv_id] = r
        return r


@pytest.fixture(scope="session")
def smoke(tmp_path_factory):
    return SmokeCache(tmp_path_factory.mktemp("results"))


def _ok(smoke, inv):
    r = smoke.ensure(inv.id)
    assert r.returncode == 0, f"{inv.script} --smoke {' '.join(inv.args)} exit " \
                              f"{r.returncode}\n{r.stderr[-4000:]}"
    return r


def _manifest(smoke, inv):
    _ok(smoke, inv)
    p = smoke.dir / f"{inv.name}.manifest.json"
    assert p.is_file(), f"{p.name} missing in smoke/ (files: " \
                        f"{sorted(x.name for x in smoke.dir.glob('*'))})"
    return json.loads(p.read_text())


def _log(smoke, inv):
    _ok(smoke, inv)
    p = smoke.dir / f"{inv.name}.log"
    assert p.is_file(), f"{p.name} missing"
    return p.read_text()


def _npz(smoke, inv):
    man = _manifest(smoke, inv)
    paths = [smoke.dir / o["path"] for o in man["outputs"] if str(o["path"]).endswith(".npz")]
    assert paths, f"{inv.name}: no npz among the outputs"
    out = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            out.update({k: z[k] for k in z.files})
    return out


def _jobs(man):
    j = man["jobs"]
    return list(j.values()) if isinstance(j, dict) else list(j)


ALL = list(INVOCATIONS.values())


@pytest.mark.parametrize("inv", ALL, ids=_ids(ALL))
@pytest.mark.script
@pytest.mark.slow
def test_smoke_writes_log_manifest_npz(smoke, inv):
    _ok(smoke, inv)
    for suf in (".log", ".manifest.json"):
        assert (smoke.dir / f"{inv.name}{suf}").is_file(), inv.name + suf
    assert not list(smoke.root.glob(f"{inv.name}.*")), "smoke run wrote into the full results"
    _npz(smoke, inv)


@pytest.mark.parametrize("inv", ALL, ids=_ids(ALL))
@pytest.mark.script
@pytest.mark.slow
def test_manifest_fields(smoke, inv):
    import scipy
    man = _manifest(smoke, inv)
    for k in MANIFEST_KEYS:
        assert k in man, k
    assert man["schema"] == 1 and man["name"] == inv.name
    assert isinstance(man["clean"], bool)
    if (ROOT / ".git").exists():
        assert re.fullmatch(r"[0-9a-f]{40}", man["commit"]), man["commit"]
    assert re.fullmatch(r"[0-9a-f]{64}", man["source_sha256"])
    assert man["python"] == sys.version.split()[0]
    assert man["numpy"] == np.__version__ and man["scipy"] == scipy.__version__
    assert f"{inv.script}.py" in man["command"] and "--smoke" in man["argv"]
    assert int(man["workers"]) == 2
    assert float(man["wall_s"]) > 0 and man["start"] and man["end"]
    import smoothfm.config as config
    for k in ("MU", "L", "NSTEP", "T_END", "TIMES"):
        assert k in man["config"], k
    assert man["config"]["MU"] == config.MU
    assert len(man["config"]["TIMES"]) == 49
    assert isinstance(man["example"], dict) and man["example"], "worked example missing"
    jobs = _jobs(man)
    assert jobs, "no jobs"
    for j in jobs:
        assert "wall" in json.dumps(j), j


@pytest.mark.parametrize("inv", ALL, ids=_ids(ALL))
@pytest.mark.script
@pytest.mark.slow
def test_outputs_sha256_match_disk(smoke, inv):
    man = _manifest(smoke, inv)
    paths = [str(o["path"]) for o in man["outputs"]]
    assert any(p.endswith(".npz") for p in paths) and any(p.endswith(".log") for p in paths)
    for o in man["outputs"]:
        p = pathlib.Path(o["path"])
        p = p if p.is_absolute() else smoke.dir / p
        assert p.is_file(), o
        assert o["bytes"] == p.stat().st_size, o
        assert o["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest(), o


@pytest.mark.parametrize("inv", ALL, ids=_ids(ALL))
@pytest.mark.script
@pytest.mark.slow
def test_log_header_and_worked_example(smoke, inv):
    txt = _log(smoke, inv)
    head = "\n".join(txt.splitlines()[:12])
    assert re.search(r"commit ([0-9a-f]{7,40}|unknown)\s+(clean|dirty)", head), head
    for k in ("source_sha256", "python", "numpy", "scipy", "cvxopt", "host", "workers",
              "command", "MU", "NSTEP", "T_END"):
        assert k in head, k
    assert "worked example" in txt
    ex = json.dumps(_manifest(smoke, inv)["example"]).lower()
    assert "target" in ex, ex[:300]
    if inv.script in SELECT:
        for tok in ("mu", "lam", "latent", "seed", "spline"):
            assert tok in ex, (tok, ex[:500])
        assert "rk4" in ex or "nstep" in ex, ex[:500]


SEL = [i for i in ALL if i.script in SELECT]


@pytest.mark.parametrize("inv", SEL, ids=_ids(SEL))
@pytest.mark.script
@pytest.mark.slow
def test_selection_histories(smoke, inv):
    import smoothfm.config as config
    arrs = _npz(smoke, inv)
    prefixes = [k[:-len("tables")] for k in arrs if k.endswith("tables")]
    assert prefixes, f"no <prefix>tables arrays in {sorted(arrs)[:20]}"
    for pre in prefixes:
        tab = arrs[pre + "tables"]
        assert tab.ndim == 3 and tab.shape[2] == 2, (pre, tab.shape)
        K = tab.shape[0]
        for s in ("at_limit", "nsolves", "residual"):
            assert pre + s in arrs, pre + s
            assert arrs[pre + s].shape == (K,), (pre + s, arrs[pre + s].shape)
        for row in tab:
            lam = row[:, 0][np.isfinite(row[:, 0])]
            assert np.all(np.diff(lam) >= 0), "table not sorted in lam"
            assert np.isfinite(row[:, 1]).any(), "no finite score"
        assert np.all(arrs[pre + "nsolves"] >= 1)
    txt = _log(smoke, inv)
    nsel = sum("lam/lam_s" in ln for ln in txt.splitlines())
    assert nsel >= len(config.SMOKE["TIMES"]), f"{nsel} lam/lam_s lines"


FL = [i for i in ALL if i.script not in NO_FLOW]


@pytest.mark.parametrize("inv", FL, ids=_ids(FL))
@pytest.mark.script
@pytest.mark.slow
def test_flow_summary_and_leave_omega(smoke, inv):
    man = _manifest(smoke, inv)
    t = json.dumps(man["jobs"])
    assert "outside" in t, "no leave-Omega fraction in the flow summaries (A7)"
    for k in ("nstep", "t_end", "nfev"):
        assert k in t, k
    assert "outside omega" in _log(smoke, inv).lower()


TVI = [i for i in ALL if i.script not in NO_TV]


def _flatten(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict) and "se" not in v:
            yield from _flatten(v, f"{pre}{k}.")
        else:
            yield pre + str(k), k, v, d


@pytest.mark.parametrize("inv", TVI, ids=_ids(TVI))
@pytest.mark.script
@pytest.mark.slow
def test_tv_metrics_carry_se(smoke, inv):
    man = _manifest(smoke, inv)
    ntv, missing = 0, []
    for j in _jobs(man):
        for full, k, v, parent in _flatten(j.get("metrics", {}) or {}):
            k = str(k)
            if "tv" not in k.lower() or k.endswith("_se") or k.startswith("se_") or \
                    "bound" in k.lower():
                continue
            ntv += 1
            ok = (f"{k}_se" in parent or f"se_{k}" in parent
                  or (isinstance(v, dict) and "se" in v)
                  or (isinstance(v, list) and len(v) == 2 and not isinstance(v[0], list)))
            if not ok:
                missing.append(full)
    assert ntv > 0, "no TV metric in the manifest jobs"
    assert not missing, f"TV without se (A2): {missing[:10]}"


@pytest.mark.parametrize("inv", SEL, ids=_ids(SEL))
@pytest.mark.script
@pytest.mark.slow
def test_cost_record(smoke, inv):
    man = _manifest(smoke, inv)
    flat = json.dumps(man).lower()
    assert "cpu" in flat and "core" in flat and man["host"]
    jobs = json.dumps(man["jobs"]).lower()
    assert "generation" in jobs and ("fit" in jobs or "selection" in jobs), "A3 wall split"
    assert "nsolves" in jobs or "factoriz" in jobs, "A3 factorizations per time"
    assert "lam/lam_s" in _log(smoke, inv) or "lam_rel" in jobs, "A7 lam/lam_s per time"


ROOTS = ROOT_INVS


def _arrays(dirpath, name):
    """Arrays of the npz files that the manifest of ``name`` in dirpath lists."""
    man = json.loads((pathlib.Path(dirpath) / f"{name}.manifest.json").read_text())
    out = {}
    for o in man["outputs"]:
        p = pathlib.Path(dirpath) / o["path"]
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                out.update({f"{p.name}:{k}": z[k] for k in z.files})
    return out


@pytest.mark.parametrize("inv", ROOTS, ids=_ids(ROOTS))
@pytest.mark.script
@pytest.mark.slow
def test_two_smoke_runs_identical(smoke, inv, tmp_path):
    _ok(smoke, inv)
    r = run_script(inv.script, ["--smoke", *inv.args], tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    a = _arrays(smoke.dir, inv.name)
    b = _arrays(tmp_path / "smoke", inv.name)
    assert a and set(a) == set(b)
    diff = [k for k in a if not VOLATILE.search(k.split(":", 1)[1])
            and not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind in "fc")]
    assert not diff, f"arrays differ between two smoke runs: {diff[:10]}"


# The smoke reference (tests/fixtures/smoke_reference.json, tests/make_smoke_reference.py)
# holds per-array summaries of every smoke result of experiments/runs.txt, and the entries of
# the outcome metrics (TVs, within- and outside-fractions) and their Monte Carlo SEs. By
# default a rerun must match the structure (every array, shape, dtype) and the outcomes within
# statistical tolerance (release_helpers.outcome_differences). Selection internals (lam,
# lam_rel, scores, residuals, curves, trajectories, images, solve counts) are not compared:
# in smoke mode the hold-out and oracle criteria are flat at late times, so the order of
# floating-point reductions, which BLAS, LAPACK, SuiteSparse and the CPU's vector width
# change, flips ties between lambda candidates that lie decades apart. With
# SMOOTHFM_STRICT_REFERENCE=1 every summary must agree to SMOKE_RTOL, a check for reruns on
# the platform that wrote the reference, where a change of the numerics (another lambda, step
# count, sample or latent) moves the summaries by orders of magnitude more than rounding.
SMOKE_RTOL = 1e-9


def _strict_reference():
    return os.environ.get("SMOOTHFM_STRICT_REFERENCE") == "1"


def _reference_mismatches(got, ref):
    if _strict_reference():
        return summary_mismatches(got, ref, SMOKE_RTOL)
    return outcome_mismatches(got, ref)


def test_outcome_mismatches_ignore_selection_and_catch_outcomes():
    def summ(v):
        return {"shape": [len(v)], "dtype": "<f8", "values": list(v)}
    ref = {"a/lam": summ([1e7]), "a/tv": summ([0.190]), "a/tv_se": summ([0.008]),
           "a/frac_out": summ([0.0])}
    rerun = dict(ref, **{"a/lam": summ([1e14]), "a/tv": summ([0.205]),
                         "a/tv_se": summ([0.009]), "a/frac_out": summ([0.001])})
    assert outcome_mismatches(rerun, ref) == []
    assert outcome_mismatches(dict(rerun, **{"a/tv": summ([0.250])}), ref)
    assert outcome_mismatches(dict(rerun, **{"a/frac_out": summ([0.05])}), ref)
    assert outcome_mismatches(dict(rerun, **{"a/lam": summ([1.0, 2.0])}), ref)


@functools.lru_cache(maxsize=1)
def _smoke_reference():
    return json.loads(SMOKE_REFERENCE.read_text())


def test_smoke_reference_covers_the_run_list():
    assert sorted(_smoke_reference()) == sorted(run_names())


@pytest.mark.parametrize("inv", ALL, ids=[f"{i.group}-reference-{i.id}" for i in ALL])
@pytest.mark.script
@pytest.mark.slow
def test_smoke_matches_reference(smoke, inv):
    _ok(smoke, inv)
    bad = _reference_mismatches(npz_summaries(smoke.dir / f"{inv.name}.npz"),
                                _smoke_reference()[inv.name])
    assert not bad, f"{inv.name}: {len(bad)} differences from the smoke reference " \
                    f"(tests/make_smoke_reference.py rewrites it): {bad[:10]}"


@pytest.mark.slow
def test_smoke_directory_matches_reference():
    """Every result of a full `make smoke` (directory $SMOOTHFM_SMOKE_RESULTS, e.g.
    results/smoke) against the smoke reference."""
    d = os.environ.get("SMOOTHFM_SMOKE_RESULTS")
    if not d:
        pytest.skip("set SMOOTHFM_SMOKE_RESULTS to a directory of `make smoke` results")
    d = pathlib.Path(d)
    bad = {}
    for name in run_names():
        p = d / f"{name}.npz"
        if not p.is_file():
            bad[name] = ["missing"]
            continue
        diff = _reference_mismatches(npz_summaries(p), _smoke_reference()[name])
        if diff:
            bad[name] = diff[:3]
    assert not bad, f"{len(bad)} results differ from the smoke reference: " \
                    f"{dict(list(bad.items())[:5])}"


@pytest.mark.parametrize("inv", DEP_INVS, ids=_ids(DEP_INVS))
@pytest.mark.script
@pytest.mark.slow
def test_missing_dependency_fails_clearly(inv, tmp_path):
    r = run_script(inv.script, ["--smoke", *inv.args], tmp_path, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode != 0, "ran without its dependencies " + str(inv.deps)
    dep_names = {INVOCATIONS[d].name for d in inv.deps}
    assert any(d in out for d in dep_names), f"message names none of {dep_names}:\n{out[-1500:]}"
    assert "Traceback" not in out, out[-2000:]
    assert not list((tmp_path / "smoke").glob("*.npz"))


@pytest.mark.parametrize("inv", ROOTS, ids=_ids(ROOTS))
@pytest.mark.script
@pytest.mark.slow
def test_example_mode_prints_example_and_writes_no_results(inv, tmp_path):
    r = run_script(inv.script, ["--example", *inv.args], tmp_path, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "worked example" in r.stdout
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix == ".npz"], "example mode wrote an npz"
    stray = [p for p in files if "example" not in p.relative_to(tmp_path).parts]
    assert not stray, f"example mode wrote outside example/: {stray}"
    man = tmp_path / "example" / f"{inv.name}.manifest.json"
    if man.exists():
        assert len(_jobs(json.loads(man.read_text()))) <= 1, "example ran more than one job"


def _dirty():
    try:
        out = subprocess.run(["git", "status", "--porcelain", "src", "experiments"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool([ln for ln in out.splitlines() if "experiments/results" not in ln])


@pytest.mark.parametrize("inv", ROOTS, ids=_ids(ROOTS))
@pytest.mark.script
@pytest.mark.slow
def test_full_run_refuses_to_overwrite(inv, tmp_path):
    sentinel = "sentinel: existing results\n"
    for suf in (".log", ".manifest.json"):
        (tmp_path / f"{inv.name}{suf}").write_text(sentinel)
    r = run_script(inv.script, list(inv.args), tmp_path, timeout=300)
    out = (r.stdout + r.stderr).lower()
    assert r.returncode != 0, "full run did not refuse"
    for suf in (".log", ".manifest.json"):
        assert (tmp_path / f"{inv.name}{suf}").read_text() == sentinel, "existing result changed"
    d = _dirty()
    assert ("dirty" in out) if d else ("overwrite" in out), out[-1500:]


@pytest.mark.parametrize("name", SCRIPTS, ids=[f"{first_invocation(n).group}-help-{n}"
                                                for n in SCRIPTS])
@pytest.mark.script
@pytest.mark.slow
def test_help_lists_standard_flags(name):
    r = subprocess.run([sys.executable, str(COMPUTE / f"{name}.py"), "--help"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    for flag in ("--smoke", "--example", "--workers", "--overwrite", "--results"):
        assert flag in r.stdout, flag
    inv = first_invocation(name)
    for a in inv.args:
        if a.startswith("--"):
            assert a in r.stdout, a


@pytest.mark.script
@pytest.mark.slow
def test_report_from_smoke_results(smoke, tmp_path):
    """Every smoke result present: the report reads their keys (5.2) and is deterministic."""
    for inv in ALL:
        _ok(smoke, inv)
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        r = run_report(smoke.dir, out)
        assert r.returncode == 0, r.stderr[-3000:]
    fa, fb = tree_files(a), tree_files(b)
    expected = read_outputs_txt() or PAPER_INPUTS
    assert expected <= set(fa), sorted(expected - set(fa))
    assert fa == fb, sorted(k for k in fa if fa.get(k) != fb.get(k))[:10]
    # The report prints \\tbd for a missing source in every mode. ALL is a subset of the
    # run list (no ablation, lam_sweep or m512 chains beyond the ones listed above), so
    # a table may name missing sources, but none of them may be a result these runs produced.
    produced = {i.name for i in ALL}
    bad = {}
    for k, v in fa.items():
        if not k.endswith(".tex") or b"\\tbd" not in v:
            continue
        lines = [ln for ln in v.decode().splitlines() if "pending (missing) sources" in ln]
        missing = {x.strip().removesuffix(".npz") for ln in lines
                   for x in ln.split(":", 1)[1].split(",")}
        if not lines or missing & produced:
            bad[k] = sorted(missing & produced) if lines else "\\tbd without a missing source"
    assert not bad, f"outputs with \\tbd for sources the smoke runs produced: {bad}"
