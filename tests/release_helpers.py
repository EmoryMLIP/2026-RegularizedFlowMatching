"""Shared helpers of the script and report tests (test_scripts.py, test_report.py).

Written from the list of experiments, result names and dependencies, the log and manifest
format, Monte Carlo SEs and cost records, without reading experiments/.

Interface the tests assume (each is one line to change here):
- RESULTS_FLAG: a compute script takes ``--results DIR`` for the results root; smoke runs
  write ``DIR/smoke/<name>.*`` and read their dependencies from there, ``--example`` writes
  only into ``DIR/example/`` (RunLog's ``results_dir``).
- The report stage is run as ``make report RESULTS=<results dir> OUT=<out dir>`` and writes
  ``<out>/tables`` and ``<out>/figures`` (adapter ``run_report``).
- The committed generated files live in ``experiments/paper/{tables,figures}`` (the Makefile's
  OUT default).
- The script invocations use ``--dim 1`` / ``--target T`` / ``--reg R`` /
  ``--regs`` / ``--m M`` as the table writes them; the 1D
  datasets of lam_sweep are ``trimodal`` and ``inhom``.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPUTE = ROOT / "experiments" / "compute"
REPORT = ROOT / "experiments" / "report"
# the paper repository (main.tex, supplement.tex): only from $SMOOTHFM_PAPER, no default; the
# tests that read the paper skip without it
PAPER = (pathlib.Path(os.environ["SMOOTHFM_PAPER"]).expanduser()
         if os.environ.get("SMOOTHFM_PAPER") else None)

RESULTS_FLAG = "--results"
WORKERS = "2"

# Every compute script.
SCRIPTS = ["oned", "gabriel", "tend", "integration_error", "floors",
           "controls", "controls_iid", "oned_eval", "kde4", "hero_1d", "gabriel_hero",
           "twod", "kde2d", "kde4_2d", "em8", "twod_eval", "lam_sweep",
           "stability", "compare_time"]

# Scripts that select lambda per (set, t) and so write selection histories.
SELECT = {"oned", "controls", "twod", "tend"}
# KDE-only scripts: no flow, no FlowSummary.
NO_FLOW = {"kde4", "kde2d", "kde4_2d", "em8"}
# Scripts whose outputs carry TV estimates, hence an se next to each TV.
NO_TV = {"hero_1d", "gabriel_hero", "compare_time"}

# The \input targets of main.tex and supplement.tex (the supplement \inputs its tables
# through \fittable); OUTPUTS.txt lists these and the cost table.
PAPER_INPUTS = {
    "tables/" + t + ".tex" for t in (
        "tab_main", "tab_oned_full", "tab_twod_full", "tab_reg_full", "tab_lam_sweep_full",
        "tab_refine_1d", "tab_refine_2d", "tab_integrator", "tab_tend", "tab_stability",
        "tab_stability_refine")} | {
    "figures/" + f + ".tex" for f in (
        "fig_hero", "fig_accuracy", "fig_stability", "fig_twod_ops")}
REFINE_TABLES = ("tables/tab_refine_1d.tex", "tables/tab_refine_2d.tex")
# The cost table; the tests accept any name tab_cost*.tex.
COST_TABLE = re.compile(r"^tables/tab_cost[\w-]*\.tex$")
BUILD_JUNK = (".aux", ".log", ".pdf", ".fls", ".fdb_latexmk", ".synctex.gz", ".DS_Store")

T2 = ("circle", "checkerboard", "spirals", "gauss8")
#: CI group of the runs of each 2D target (see _invocations and experiments/runs.txt).
TARGET_GROUP = {"circle": "g3", "checkerboard": "g3", "spirals": "g4", "gauss8": "g2"}


@dataclasses.dataclass(frozen=True)
class Inv:
    """One script invocation of the run list: id, script, extra args, result name, deps, group."""
    id: str
    script: str
    args: tuple
    name: str
    deps: tuple
    group: str


def _invocations():
    v = []

    def add(id_, script, args, deps=(), group="g2", name=None):
        v.append(Inv(id_, script, tuple(args), name or id_, tuple(deps), group))

    # The CI groups balance the measured job times (GitHub runner, 2 cores): g1 compare_time,
    # whose --example run alone takes 18-27 min; g2 the other 1D runs and gauss8; g3 the
    # circle and checkerboard; g4 spirals, whose kde2d and twod runs are the slowest 2D ones.
    add("oned", "oned", [])
    add("oned_h1", "oned", ["--reg", "h1"])
    add("oned_l2", "oned", ["--reg", "l2"])
    add("floors", "floors", [])
    add("gabriel", "gabriel", [], ["oned"])
    add("tend", "tend", [], ["oned"])
    add("integration_error", "integration_error", [], ["oned"])
    add("compare_time", "compare_time", [], group="g1")
    g = "g2"
    add("controls_1d", "controls", ["--dim", "1"], group=g)
    add("controls_iid_1d", "controls_iid", ["--dim", "1"], ["controls_1d"], group=g)
    add("oned_eval", "oned_eval", [], ["oned", "gabriel", "controls_1d"], group=g)
    add("kde4", "kde4", [], ["oned_eval"], group=g)
    add("hero_1d", "hero_1d", [], ["oned", "oned_eval"], group=g)
    add("gabriel_hero_reg", "gabriel_hero", ["--regs"],
        ["oned", "oned_eval", "gabriel", "oned_h1", "oned_l2"], group=g)
    add("lam_sweep_trimodal", "lam_sweep", ["--target", "trimodal", "--reg", "h2"], ["oned"],
        group=g)
    add("stability_1d", "stability", ["--dim", "1"], ["oned", "oned_l2"], group=g)
    add("stability_1d_m256", "stability", ["--dim", "1", "--m", "256"], ["oned", "oned_l2"],
        group=g)
    for T in T2:
        g = TARGET_GROUP[T]
        add(f"twod_{T}", "twod", ["--target", T], group=g)
        add(f"kde2d_{T}", "kde2d", ["--target", T], group=g)
        add(f"controls_{T}", "controls", ["--target", T], group=g)
        add(f"kde4_2d_{T}", "kde4_2d", ["--target", T], [f"twod_{T}"], group=g)
        add(f"twod_eval_{T}", "twod_eval", ["--target", T],
            [f"twod_{T}", f"kde2d_{T}", f"controls_{T}"], group=g)
    g = "g3"
    add("controls_iid_circle", "controls_iid", ["--target", "circle"], ["controls_circle"],
        group=g)
    add("em8", "em8", [], group=g)
    add("lam_sweep_circle", "lam_sweep", ["--target", "circle", "--reg", "h2"], ["twod_circle"],
        group=g)
    add("stability_circle", "stability", ["--target", "circle"], ["twod_circle"], group=g)
    return {i.id: i for i in v}


INVOCATIONS = _invocations()
ROOT_INVS = [i for i in INVOCATIONS.values() if not i.deps]
DEP_INVS = [i for i in INVOCATIONS.values() if i.deps]


def first_invocation(script):
    return next(i for i in INVOCATIONS.values() if i.script == script)


def env():
    e = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        e[v] = "1"
    e["NWORKERS"] = WORKERS
    e.pop("OVERWRITE", None)
    e["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), e.get("PYTHONPATH", "")])
    return e


def run_script(script, args, results, timeout=900, extra_env=None):
    """Run experiments/compute/<script>.py with args into the results root ``results``."""
    cmd = [sys.executable, str(COMPUTE / f"{script}.py"), *args,
           RESULTS_FLAG, str(results), "--workers", WORKERS]
    e = env()
    e.update(extra_env or {})
    return subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True,
                          timeout=timeout)


def run_report(results, out, timeout=600):
    """The report stage from a results directory into out/{tables,figures}."""
    cmd = ["make", "-s", "-C", str(ROOT), "report", f"RESULTS={results}", f"OUT={out}"]
    return subprocess.run(cmd, cwd=ROOT, env=env(), capture_output=True, text=True,
                          timeout=timeout)


def make_n(*targets, **variables):
    """The commands `make -n <targets> VAR=value ...` prints, one per line."""
    cmd = ["make", "-n", "--no-print-directory", "-C", str(ROOT), *targets,
           *(f"{k}={v}" for k, v in variables.items())]
    e = env()
    e.pop("NWORKERS", None)
    r = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and not r.stderr, r.stderr
    return r.stdout.splitlines()


def committed_out():
    return ROOT / "experiments" / "paper"


def tree_files(top, subdirs=("tables", "figures")):
    """Relative path -> bytes of every generated file under top/{tables,figures}."""
    top = pathlib.Path(top)
    out = {}
    for s in subdirs:
        for p in sorted((top / s).rglob("*")):
            if p.is_file() and not p.name.endswith(BUILD_JUNK):
                out[p.relative_to(top).as_posix()] = p.read_bytes()
    return out


def strip_tex_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.splitlines())


def paper_inputs():
    """The \\input (and \\fittable) targets of main.tex and supplement.tex (comments
    removed), or None."""
    if PAPER is None:
        return None
    files = [PAPER / "main.tex", PAPER / "supplement.tex"]
    if not all(f.exists() for f in files):
        return None
    got = set()
    for f in files:
        for m in re.finditer(r"\\(?:input|fittable)\{([^}]+)\}",
                             strip_tex_comments(f.read_text())):
            p = m.group(1).strip()
            if p.startswith(("tables/", "figures/")):
                got.add(p if p.endswith(".tex") else p + ".tex")
    return got


def read_outputs_txt():
    p = REPORT / "OUTPUTS.txt"
    if not p.exists():
        return None
    got = set()
    for ln in p.read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        if "/" not in ln:
            ln = ("tables/" if ln.startswith("tab_") else "figures/") + ln
        got.add(ln if ln.endswith(".tex") else ln + ".tex")
    return got


# ---------------------------------------------------------------------------
# Reference summaries of the smoke results (tests/fixtures/smoke_reference.json, written by
# tests/make_smoke_reference.py)
# ---------------------------------------------------------------------------

SMOKE_REFERENCE = ROOT / "tests" / "fixtures" / "smoke_reference.json"
#: Keys that hold timings (wall clock) or provenance, not results.
SMOKE_UNSUMMARIZED = re.compile(r"wall[^/]*$|source_file_sha256$")


def run_names():
    """The result names of experiments/runs.txt, in file order."""
    out = []
    for ln in (ROOT / "experiments" / "runs.txt").read_text().splitlines():
        if ln.strip() and not ln.lstrip().startswith("#"):
            out.append(ln.split()[0])
    return out


def array_summary(a):
    """Shape, dtype and robust summaries of a numeric array (None for other dtypes): sum,
    sum of squares, min and max over the finite entries, and the count of the others."""
    import numpy as np
    a = np.asarray(a)
    if a.dtype.kind not in "biuf":
        return None
    f = a.astype(np.float64).ravel()
    fin = np.isfinite(f)
    v = f[fin]
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "sum": float(v.sum()), "sumsq": float((v * v).sum()),
            "min": float(v.min()) if v.size else None,
            "max": float(v.max()) if v.size else None,
            "nonfinite": int((~fin).sum())}


#: Outcome metrics by the last path component of their key: TV estimates, the fraction of
#: samples within a region, and the fraction outside the domain or support. Any other array
#: with a Monte Carlo SE partner ``<key>_se`` in the same npz is an outcome too.
SMOKE_OUTCOME = re.compile(r"^(tv|frac_)|within")
#: Outcome-named arrays left out of the cross-platform comparison. ``<d>/<flow>_resel/tv`` of
#: experiments/compute/stability.py is the TV between the laws generated from the training set
#: and from its perturbation, each with lambda selected anew; in smoke mode the selection is a
#: tie between candidates decades apart, so this TV follows the tie (circle at epsilon 1e-5:
#: 0.011 on macOS, 0.032 on Linux, with tv_target within its SE).
SMOKE_SELECTION_DEPENDENT = re.compile(r"_resel/tv$")
#: Absolute tolerance of an outcome that has no SE partner, and the floor of every tolerance.
SMOKE_OUTCOME_ATOL = 2e-3
#: Absolute tolerance of a sample fraction (frac_*, *within*) without SE partner. A smoke run
#: draws a few hundred samples (512 in the 2D runs, so one sample moves frac_out by 0.002),
#: and the binomial SE of a fraction 1/512 is 0.002 itself; 1e-2 admits five samples.
SMOKE_FRACTION_ATOL = 1e-2
SMOKE_FRACTION = re.compile(r"^frac_|within")


def outcome_roles(keys):
    """{key: ("outcome", se_key or None) or ("se", base_key)} for the outcome metrics among
    the keys of one npz and their SE partners. The partner of ``<key>`` is ``<key>_se``; a
    ``<dir>/tv_ref`` without one takes ``<dir>/tv_pos_se``."""
    keys = set(keys)
    out = {}
    for k in keys:
        if k.endswith("_se") and k[:-3] in keys:
            out[k] = ("se", k[:-3])
        elif SMOKE_OUTCOME.search(k.rsplit("/", 1)[-1]) or k + "_se" in keys:
            se = k + "_se"
            if se not in keys and k.endswith("/tv_ref"):
                # tv_ref is the stored tv_pos of the same flow that the script reproduces
                se = k[:-len("tv_ref")] + "tv_pos_se"
            out[k] = ("outcome", se if se in keys else None)
    return out


def npz_summaries(path):
    """{key: array_summary} of every numeric array of an npz outside SMOKE_UNSUMMARIZED; the
    summaries of outcome metrics and their SEs (outcome_roles) also hold the entries as
    ``values`` (row-major, None for non-finite)."""
    import numpy as np
    out = {}
    with np.load(path, allow_pickle=False) as z:
        roles = outcome_roles(k for k in z.files if not SMOKE_UNSUMMARIZED.search(k))
        for k in sorted(z.files):
            if SMOKE_UNSUMMARIZED.search(k):
                continue
            s = array_summary(z[k])
            if s is not None:
                if k in roles:
                    f = z[k].astype(np.float64).ravel()
                    s["values"] = [float(x) if np.isfinite(x) else None for x in f]
                out[k] = s
    return out


def _values(s):
    import numpy as np
    return np.array([np.nan if x is None else x for x in s["values"]], dtype=np.float64)


def summary_mismatches(got, ref, rtol):
    """Differences between two {key: summary} maps: key sets, shape, dtype and nonfinite count
    exactly; sum, min and max to rtol times the array's scale sqrt(size * sumsq) or its largest
    magnitude, sumsq to rtol times itself, and stored values to rtol times the largest
    magnitude."""
    import math
    import numpy as np
    bad = []
    if set(got) != set(ref):
        bad.append(f"keys: only new {sorted(set(got) - set(ref))[:5]}, "
                   f"only reference {sorted(set(ref) - set(got))[:5]}")
    for k in sorted(set(got) & set(ref)):
        g, r = got[k], ref[k]
        for f in ("shape", "dtype", "nonfinite"):
            if g[f] != r[f]:
                bad.append(f"{k}: {f} {g[f]} vs {r[f]}")
        if bad and bad[-1].startswith(k + ":"):
            continue
        n = math.prod(r["shape"]) - r["nonfinite"]
        scale = math.sqrt(n * r["sumsq"])
        peak = max(abs(r["min"] or 0.0), abs(r["max"] or 0.0))
        for f, tol in (("sum", rtol * scale), ("sumsq", rtol * r["sumsq"]),
                       ("min", rtol * peak), ("max", rtol * peak)):
            if (g[f] is None) != (r[f] is None) or (
                    r[f] is not None and abs(g[f] - r[f]) > tol):
                bad.append(f"{k}: {f} {g[f]!r} vs {r[f]!r}")
        if ("values" in g) != ("values" in r):
            bad.append(f"{k}: values stored in only one summary")
        elif "values" in r:
            a, b = _values(g), _values(r)
            if not np.array_equal(np.isnan(a), np.isnan(b)) or \
                    np.nanmax(np.abs(a - b), initial=0.0) > rtol * peak:
                bad.append(f"{k}: values differ by up to {np.nanmax(np.abs(a - b)):.3g}")
    return bad


def outcome_differences(got, ref):
    """{key: (max |got - ref|, max |got - ref| / tolerance)} over the stored values of the
    outcome metrics and their SEs (keys present in both). The tolerance of an outcome entry
    is max(3 * sqrt(se_got**2 + se_ref**2), SMOKE_OUTCOME_ATOL) with the SE partner's
    entries, or SMOKE_OUTCOME_ATOL without one; that of an SE entry is
    max(SMOKE_OUTCOME_ATOL, 0.5 * max(se_got, se_ref)). Entries non-finite in one map and
    not in the other give (inf, inf)."""
    import numpy as np
    roles = outcome_roles(ref)
    out = {}
    for k, (role, partner) in roles.items():
        if SMOKE_SELECTION_DEPENDENT.search(k):
            continue
        if k not in got or "values" not in got[k] or "values" not in ref[k]:
            continue
        a, b = _values(got[k]), _values(ref[k])
        if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
            out[k] = (np.inf, np.inf)
            continue
        if role == "se":
            tol = np.maximum(SMOKE_OUTCOME_ATOL, 0.5 * np.maximum(np.abs(a), np.abs(b)))
        else:
            frac = SMOKE_FRACTION.search(k.rsplit("/", 1)[-1])
            tol = np.full(a.shape, SMOKE_FRACTION_ATOL if frac else SMOKE_OUTCOME_ATOL)
            if partner and partner in got and "values" in got[partner]:
                sa, sb = _values(got[partner]), _values(ref[partner])
                if sa.shape == a.shape and sb.shape == a.shape:
                    s = np.sqrt(sa ** 2 + sb ** 2)
                    tol = np.where(np.isfinite(s), np.maximum(3 * s, SMOKE_OUTCOME_ATOL), tol)
        d = np.abs(a - b)
        fin = np.isfinite(d)
        out[k] = (float(d[fin].max(initial=0.0)), float((d[fin] / tol[fin]).max(initial=0.0)))
    return out


def outcome_mismatches(got, ref):
    """Differences that a smoke rerun on another platform must not show: key sets, shape and
    dtype of every array exactly, and the outcome metrics and their SEs within the
    tolerances of outcome_differences."""
    bad = []
    if set(got) != set(ref):
        bad.append(f"keys: only new {sorted(set(got) - set(ref))[:5]}, "
                   f"only reference {sorted(set(ref) - set(got))[:5]}")
    for k in sorted(set(got) & set(ref)):
        for f in ("shape", "dtype"):
            if got[k][f] != ref[k][f]:
                bad.append(f"{k}: {f} {got[k][f]} vs {ref[k][f]}")
    for k, (d, q) in sorted(outcome_differences(got, ref).items()):
        if q > 1:
            bad.append(f"{k}: differs by {d:.3g}, {q:.2f} times its tolerance")
    return bad


# ---------------------------------------------------------------------------
# Minimal YAML (block mappings, block sequences, flow lists, block scalars), used for
# the workflow only when PyYAML is not installed (it is not a dependency).
# ---------------------------------------------------------------------------

def _strip_comment(s):
    out, q = [], None
    for i, ch in enumerate(s):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#" and (i == 0 or s[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _scalar(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_scalar(x) for x in re.findall(r"""(?:'[^']*'|"[^"]*"|[^,])+""", inner)] \
            if inner else []
    if s == "{}":
        return {}
    return s


def yaml_load(text):
    try:
        import yaml  # noqa: PLC0415
        return yaml.safe_load(text)
    except ImportError:
        pass
    raw = text.splitlines()
    lines = []
    for ln in raw:
        c = _strip_comment(ln)
        if c.strip():
            lines.append([len(c) - len(c.lstrip(" ")), c.strip(), ln])

    def block(i, ind):
        if lines[i][1].startswith("- ") or lines[i][1] == "-":
            return seq(i, ind)
        return mapping(i, ind)

    def seq(i, ind):
        res = []
        while i < len(lines) and lines[i][0] == ind and (lines[i][1].startswith("- ")
                                                         or lines[i][1] == "-"):
            body = lines[i][1][1:].lstrip()
            if not body:
                v, i = block(i + 1, lines[i + 1][0])
            elif re.match(r"^[^'\"\[{][^:]*:(\s|$)", body):
                lines[i] = [ind + len(lines[i][1]) - len(body), body, lines[i][2]]
                v, i = mapping(i, lines[i][0])
            else:
                v, i = _scalar(body), i + 1
            res.append(v)
        return res, i

    def mapping(i, ind):
        res = {}
        while i < len(lines) and lines[i][0] == ind and not lines[i][1].startswith("- "):
            key, _, rest = lines[i][1].partition(":")
            key, rest = _scalar(key), rest.strip()
            i += 1
            if rest in ("|", ">", "|-", ">-", "|+", ">+"):
                buf = []
                while i < len(lines) and lines[i][0] > ind:
                    buf.append(lines[i][2].strip())
                    i += 1
                res[key] = "\n".join(buf)
            elif rest:
                res[key] = _scalar(rest)
            elif i < len(lines) and (lines[i][0] > ind or (lines[i][0] == ind and
                                                           lines[i][1].startswith("- "))):
                res[key], i = block(i, lines[i][0])
            else:
                res[key] = None
        return res, i

    return block(0, lines[0][0])[0] if lines else None


def load_isolated(stage_dir, name, tag):
    """(common, module) of experiments/<stage>/<name>.py loaded under private names.

    The compute and report stages each have a module 'common' imported as `import common as
    C`; whichever a test imports first occupies sys.modules['common'] for the whole session.
    This loads the stage's common.py as '<tag>_common', binds it to 'common' only while
    <name>.py executes, and restores sys.modules and sys.path afterwards."""
    import importlib.util

    def load(path, modname):
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    stage_dir = pathlib.Path(stage_dir)
    saved = sys.modules.get("common")
    sys.path.insert(0, str(stage_dir))
    try:
        c = load(stage_dir / "common.py", f"{tag}_common")
        sys.modules["common"] = c
        m = c if name == "common" else load(stage_dir / f"{name}.py", f"{tag}_{name}")
    finally:
        sys.path.remove(str(stage_dir))
        if saved is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = saved
    return c, m
