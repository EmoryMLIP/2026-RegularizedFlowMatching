"""experiments/runs.txt, the run list of every compute invocation, against the Makefile targets
generated from it (checked through `make -n`)."""

from __future__ import annotations

import pytest

from release_helpers import ROOT, make_n

RUNS = ROOT / "experiments" / "runs.txt"
GROUPS = ("g1", "g2", "g3", "g4")


def runs():
    """[(name, group, deps, script, args)] of every line, in file order."""
    out = []
    for ln in RUNS.read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        name, group, deps, script, *args = ln.split()
        out.append((name, group, [] if deps == "-" else deps.split(","), script, args))
    return out


def by_name():
    d = runs()
    names = [r[0] for r in d]
    assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})
    return {r[0]: r[2:] for r in d}


def test_dependencies_are_listed_earlier():
    seen = set()
    for name, _, deps, _, _ in runs():
        for dep in deps:
            assert dep in seen, (name, dep)
        seen.add(name)


def test_every_run_has_a_ci_group():
    assert {g for _, g, *_ in runs()} == set(GROUPS)


def _cmd(script, args, smoke=False, overwrite=False):
    return " ".join(["uv run python", f"experiments/compute/{script}.py",
                     *(["--smoke"] if smoke else []), *args,
                     "--results", "results", "--workers", "10"]) + (" --overwrite" if overwrite
                                                                     else "")


def test_make_list_prints_the_run_list():
    import subprocess
    r = subprocess.run(["make", "-s", "-C", str(ROOT), "list"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [r_[0] for r_ in runs()]


@pytest.mark.parametrize("overwrite", [False, True])
def test_make_compute_runs_every_line_in_order(overwrite):
    """make compute: one compute-<name> per line, in file order, with the line's arguments."""
    got = make_n("compute", **({"OVERWRITE": "1"} if overwrite else {}))
    want = [_cmd(sc, a, overwrite=overwrite) for _, _, _, sc, a in runs()]
    assert got[1:] == want, [(g, w) for g, w in zip(got[1:], want) if g != w][:5]


def test_make_smoke_runs_every_line_in_order():
    assert make_n("smoke") == [_cmd(sc, a, smoke=True) for _, _, _, sc, a in runs()]


@pytest.mark.parametrize("group", GROUPS)
def test_make_smoke_group_runs_its_lines_after_their_dependencies(group):
    """smoke-<g>: the lines of group g and every line they depend on (transitively), in file
    order."""
    rs = runs()
    deps = {n: d for n, _, d, _, _ in rs}
    need = {n for n, g, *_ in rs if g == group}
    stack = list(need)
    while stack:
        for d in deps[stack.pop()]:
            if d not in need:
                need.add(d)
                stack.append(d)
    want = [_cmd(sc, a, smoke=True) for n, _, _, sc, a in rs if n in need]
    assert make_n(f"smoke-{group}") == want


@pytest.mark.parametrize("target,dep", [
    ("trimodal", "oned"), ("inhom", "oned"), ("circle", "twod_circle"),
    ("checkerboard", "twod_checkerboard"), ("spirals", "twod_spirals"),
    ("gauss8", "twod_gauss8")])
def test_hold_out_constant_runs(target, dep):
    d = by_name()
    for n in (100, 3000):
        deps, script, args = d[f"lam_sweep_{target}_n{n}"]
        assert (deps, script) == ([dep], "lam_sweep")
        assert args == ["--target", target, "--reg", "h2", "--n", str(n), "--const-only"]


@pytest.mark.parametrize("target", ["checkerboard", "spirals", "gauss8"])
def test_extra_resolution_runs_of_every_2d_target(target):
    """tab_refine_2d: every 2D target at m = 512, hold-out for k = 2, 1, 0 and
    the iid lam -> 0 row."""
    d = by_name()
    for sfx, args in (("", []), ("_h1", ["--reg", "h1"]), ("_l2", ["--reg", "l2"])):
        deps, script, a = d[f"controls_{target}{sfx}_m512"]
        assert (deps, script) == ([], "controls")
        assert a == ["--target", target, *args, "--m", "512"]
    deps, script, a = d[f"controls_iid_{target}_m512"]
    assert (deps, script) == ([f"controls_{target}_m512"], "controls_iid")
    assert a == ["--target", target, "--m", "512"]
