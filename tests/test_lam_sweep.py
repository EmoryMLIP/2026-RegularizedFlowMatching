"""experiments/compute/lam_sweep.py: result names with --n, the --const-only schedule, and
the records n and const_only (the hold-out constant row of tab_main at every N).

Unit level except the last test, which runs the script in smoke mode on the smoke oned result
(marker script; skipped without results/smoke/oned.npz)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from release_helpers import COMPUTE, ROOT, run_script
from release_helpers import load_isolated

_, LS = load_isolated(COMPUTE, "lam_sweep", "cmp_lam_sweep")
from smoothfm import config  # noqa: E402

N0 = int(config.LAM_SWEEP["n"])
TARGETS = ("trimodal", "inhom", "circle", "checkerboard", "spirals", "gauss8")


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("reg,sfx", [("h2", ""), ("h1", "_h1"), ("l2", "_l2")])
def test_result_name(target, reg, sfx):
    assert LS.result_name(target, reg) == f"lam_sweep_{target}{sfx}"
    assert LS.result_name(target, reg, N0) == f"lam_sweep_{target}{sfx}"
    for n in config.NS:
        if int(n) != N0:
            assert LS.result_name(target, reg, int(n)) == f"lam_sweep_{target}{sfx}_n{int(n)}"


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("reg", ["h2", "h1", "l2"])
def test_schedule_full_and_const_only(target, reg):
    scale, const = LS.schedule_k(target, reg)
    assert scale == list(config.LAM_SWEEP["scale_k"][f"{target}/{reg}"])
    assert const == list(config.LAM_SWEEP["const_k"])
    scale_c, const_c = LS.schedule_k(target, reg, const_only=True)
    # s = 1 only (the reproduction check), the full constant grid
    assert scale_c == [0] and 0 in scale
    assert const_c == const


@pytest.mark.parametrize("const_only", [False, True])
def test_smoke_schedule_unchanged(const_only):
    assert LS.schedule_k("trimodal", "h2", smoke=True, const_only=const_only) == (
        list(config.LAM_SWEEP["smoke_scale_k"]), list(config.LAM_SWEEP["smoke_const_k"]))


def test_n_restricted_to_production_sizes(tmp_path):
    r = run_script("lam_sweep", ["--target", "trimodal", "--n", "500"], tmp_path)
    assert r.returncode == 2 and "--n" in r.stderr


@pytest.mark.script
def test_smoke_const_only_records(tmp_path):
    src = ROOT / "results" / "smoke" / "oned.npz"
    if not src.is_file():
        pytest.skip("no results/smoke/oned.npz (make smoke)")
    (tmp_path / "smoke").mkdir()
    shutil.copy(src, tmp_path / "smoke" / "oned.npz")
    n = max(int(x) for x in config.NS)
    r = run_script("lam_sweep", ["--smoke", "--target", "trimodal", "--n", str(n),
                                 "--const-only"], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    out = tmp_path / "smoke" / f"lam_sweep_trimodal_n{n}.npz"
    with np.load(out) as d:
        # smoke data are at the smoke N whatever --n says; the name carries the request
        assert int(d["n"]) == int(config.SMOKE["N"]) and bool(d["const_only"])
        assert list(d["trimodal/scale/k"]) == [0]
    log = out.with_suffix(".log").read_text()
    assert "const_only" in log and "check s = 1 vs stored hold-out tv_pos" in log
