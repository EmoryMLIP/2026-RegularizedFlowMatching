"""Cost table tables/tab_cost.tex from the run manifests
<results>/<name>.manifest.json (runlog). Paper: none (the paper does not
\\input it); docs/REPRODUCE.md quotes its totals as the expected cost.

One row per compute script (experiment), in the order of experiments/runs.txt, summed over
the result files (manifests) the script wrote for the invocations of the run list (`make list`,
the first column of experiments/runs.txt; other manifests in the directory are not counted):
runs (manifests found), CPU model(s), worker
processes, wall time (h, sum of wall_s), core-hours (sum of wall_s x workers / 3600) and
factorizations (sum over the manifest jobs of nsolves; '--' for scripts that factorize
nothing). Scripts without any manifest print \\tbd. The per-result numbers go to comment
lines. A last row states the computational setup (prototype Python, one node, CPU only,
worker processes, sparse Cholesky by CHOLMOD through cvxopt, CPU model and cores from the
manifests). Deterministic: no start or end times, no paths.

Manifest keys read: command, wall_s, workers, cpu_model, cpu_cores,
jobs.<job>.nsolves.
"""

from __future__ import annotations

import os
import re

import common as C

# compute scripts in the order of experiments/runs.txt (1D, then the circle, then the other 2D
# targets, then the text-only runs)
SCRIPT_ORDER = ["oned", "gabriel", "tend", "integration_error", "floors",
                "controls", "controls_iid", "oned_eval", "kde4", "hero_1d", "gabriel_hero",
                "twod", "kde2d", "kde4_2d", "em8", "twod_eval", "lam_sweep", "stability",
                "compare_time"]
RUNS = os.path.join(C.ROOT, "experiments", "runs.txt")
_CMD = re.compile(r"experiments/compute/([A-Za-z0-9_]+)\.py")


def script_of(name, man):
    """The compute script of a manifest: from its command line, else the longest script
    name that prefixes the result name; None for a manifest that no compute script wrote
    (e.g. the test suite's unit_run)."""
    cmd = str(man.get("command", ""))
    m = _CMD.search(cmd)
    if m:
        return m.group(1)
    if cmd:
        return None
    cands = [s for s in SCRIPT_ORDER if name == s or name.startswith(s + "_")]
    return max(cands, key=len) if cands else None


def run_names() -> list:
    """Result names of the run list in file order (the invocations of `make list`)."""
    with open(RUNS) as f:
        return [ln.split()[0] for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def cpu_short(model) -> str:
    """'Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz' -> 'Intel Xeon Gold 6248R'."""
    s = re.sub(r"\((?:R|TM)\)", "", str(model or "?"))
    s = re.sub(r"@.*$", "", s)
    words = [w for w in s.split() if w not in ("CPU", "Processor")]
    return " ".join(words) or "?"


def nsolves(man):
    """Sum of nsolves over the manifest's jobs, None if no job records it."""
    jobs = man.get("jobs") or {}
    vals = jobs.values() if isinstance(jobs, dict) else jobs
    tot, found = 0, False
    for j in vals:
        rec = j[1] if isinstance(j, (list, tuple)) and len(j) == 2 else j
        if isinstance(rec, dict) and isinstance(rec.get("nsolves"), (int, float)):
            tot += int(rec["nsolves"])
            found = True
    return tot if found else None


def _num(x) -> str:
    """Three significant digits, no exponent below 1e6."""
    if x is None:
        return "--"
    if x == 0:
        return "$0$"
    s = f"{x:.3g}"
    if "e" in s:
        s = f"{x:.0f}"
    return f"${s}$"


def _int(k) -> str:
    return "--" if k is None else f"${k:,}$".replace(",", "{,}")


def _esc(s) -> str:
    return str(s).replace("_", r"\_")


def build() -> list:
    runs = run_names()
    present = C.RES.manifest_names()
    names = [n for n in present if n in set(runs)]
    pending = [n for n in runs if n not in set(present)]
    groups, detail, skipped = {}, [], []
    for nm in names:
        man = C.RES.manifest(nm) or {}
        sc = script_of(nm, man)
        if sc is None:
            skipped.append(nm)
            continue
        wall = float(man.get("wall_s") or 0.0)
        w = int(man.get("workers") or 1)
        k = nsolves(man)
        groups.setdefault(sc, []).append((nm, man, wall, w, k))
        detail.append(f"% {nm}: script {sc}, CPU {cpu_short(man.get('cpu_model'))}, workers {w}, "
                      f"wall {wall:.1f} s, core-h {wall * w / 3600:.3g}, factorizations "
                      f"{k if k is not None else '--'}")
    order = SCRIPT_ORDER + sorted(s for s in groups if s not in SCRIPT_ORDER)
    rows, tot_wall, tot_ch, tot_k = [], 0.0, 0.0, 0
    tbd = False
    for sc in order:
        g = groups.get(sc)
        if not g:
            if sc in SCRIPT_ORDER:
                rows.append(f"{_esc(sc)} & {C.tbd()} & {C.tbd()} & {C.tbd()} & {C.tbd()} & {C.tbd()} & "
                            f"{C.tbd()} \\\\")
                tbd = C.tbd() == C.TBD
            continue
        cpus_g = ", ".join(sorted({_esc(cpu_short(m.get("cpu_model"))) for _, m, *_ in g}))
        ws = sorted({w for *_, w, _ in g})
        wtxt = f"${ws[0]}$" if len(ws) == 1 else f"${ws[0]}$--${ws[-1]}$"
        wall = sum(x[2] for x in g)
        ch = sum(x[2] * x[3] for x in g) / 3600.0
        ks = [x[4] for x in g if x[4] is not None]
        k = sum(ks) if ks else None
        tot_wall, tot_ch, tot_k = tot_wall + wall, tot_ch + ch, tot_k + (k or 0)
        rows.append(f"{_esc(sc)} & ${len(g)}$ & {cpus_g} & {wtxt} & {_num(wall / 3600.0)} & "
                    f"{_num(ch)} & {_int(k)} \\\\")
    cpu = {}
    for g in groups.values():
        for _, m, *_ in g:
            key = cpu_short(m.get("cpu_model"))
            if m.get("cpu_cores") is not None or key not in cpu:
                cpu[key] = m.get("cpu_cores", cpu.get(key))
    cpus = sorted((mdl, "?" if c is None else str(c)) for mdl, c in cpu.items())
    allw = sorted({x[3] for g in groups.values() for x in g})
    cpu_txt = "; ".join(f"{_esc(mdl)}, {c} cores" for mdl, c in cpus) or C.tbd()
    wk_txt = (f"{allw[0]}" if len(allw) == 1 else f"{allw[0]}--{allw[-1]}") if allw else C.tbd()
    setup = ("Prototype Python code (numpy, scipy), one compute node per run, CPU only; "
             f"{wk_txt} worker processes (column workers), one BLAS thread each; sparse "
             "Cholesky factorizations by CHOLMOD through cvxopt. CPUs: " + cpu_txt + ".")
    ncol = 7
    lines = [r"\begin{tabular}{lrlrrrr}", r"\toprule",
             r"Experiment & runs & CPU & workers & wall (h) & core-h & factorizations \\",
             r"\midrule"] + rows + [
        r"\midrule",
        f"total & ${sum(len(g) for g in groups.values())}$ & & & {_num(tot_wall / 3600.0)} & "
        f"{_num(tot_ch)} & {_int(tot_k if groups else None)} \\\\",
        r"\midrule",
        fr"\multicolumn{{{ncol}}}{{@{{}}p{{12cm}}@{{}}}}{{\footnotesize {setup}}} \\",
        r"\bottomrule", r"\end{tabular}"]
    head = ["% Generated by experiments/report/cost.py; do not edit by hand.",
            C.results_line(),
            "% Cost per experiment (compute script) from the run manifests <name>.manifest.json:",
            "% runs = result files (manifests) of the script; wall = sum of wall_s; core-h = sum of",
            "% wall_s x workers / 3600; factorizations = sum of the jobs' nsolves (sparse Cholesky",
            "% factorizations of K = A + mu M + lam P). Scripts without a manifest: no numbers (italic tbd).",
            "% Computational setup (last row): " + setup.replace("\\_", "_"),
            f"% Runs of the run list (make list): {len(runs)}; with a manifest: {len(names)}; "
            f"without (pending): {len(pending)}" + (": " + ", ".join(pending) if pending else ""),
            "% Per result file:"] + detail + (
        ["% manifests of no compute script, not counted: " + ", ".join(skipped)] if skipped else [])
    if tbd:
        head.append(C.TBD_DEF)
    path = os.path.join(C.TABLES, "tab_cost.tex")
    C.write_text(path, "\n".join(head + lines) + "\n")
    return ["tables/tab_cost.tex"]
