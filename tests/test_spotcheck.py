"""experiments/report/spotcheck.py: the key-by-key comparison of recomputed and archived
results, and the file-by-file tree comparison, on synthetic inputs. Paper: none."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from release_helpers import REPORT

pytestmark = pytest.mark.report


def _mod():
    spec = importlib.util.spec_from_file_location("spotcheck", REPORT / "spotcheck.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SC = _mod()


def _base():
    return {"target": np.array("trimodal"), "sets": np.arange(5), "times": np.linspace(0, 1, 4),
            "trimodal/const/sel_idx": np.array([3, 2, 3, 3, 4]),
            "trimodal/const/c": np.array([1e-3, 1e-2]),
            "trimodal/scale/lam": np.array([[1e-4, 2e-4, np.nan]]),
            "trimodal/const/tv_pos": np.array([[0.12, 0.13], [0.11, np.nan]]),
            "trimodal/const/e": np.array([1.0, 2.0]),
            "trimodal/const/wall": np.array([3.1, 2.9]),
            "ok_flag": np.array([True, False])}


def _status(rows):
    return {k: st for k, st, _ in rows}


def test_identical_pass():
    d = _base()
    rows = SC.compare_npz(d, {k: v.copy() for k, v in d.items()})
    assert SC.summarize(rows)["verdict"] == "PASS"
    assert _status(rows)["trimodal/const/wall"] == "SKIP"


def test_timing_ignored():
    a, b = _base(), _base()
    b["trimodal/const/wall"] = b["trimodal/const/wall"] * 2
    assert SC.summarize(SC.compare_npz(a, b))["verdict"] == "PASS"


@pytest.mark.parametrize("delta,expected", [(1e-14, "OK"), (1e-10, "WARN"), (1e-6, "FAIL")])
def test_float_thresholds(delta, expected):
    a, b = _base(), _base()
    b["trimodal/const/tv_pos"] = b["trimodal/const/tv_pos"] + delta
    rows = SC.compare_npz(a, b)
    assert _status(rows)["trimodal/const/tv_pos"] == expected
    assert SC.summarize(rows)["verdict"] == {"OK": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[expected]


def test_selected_lambda_must_be_exact():
    a, b = _base(), _base()
    b["trimodal/scale/lam"] = b["trimodal/scale/lam"] * (1 + 1e-15)
    rows = SC.compare_npz(a, b)
    st = {k: (s, d) for k, s, d in rows}["trimodal/scale/lam"]
    assert st[0] == "FAIL" and "selection changed in 2 of 2" in st[1]


def test_candidate_index_change_reported():
    a, b = _base(), _base()
    b["trimodal/const/sel_idx"] = np.array([3, 2, 4, 3, 4])
    st = {k: (s, d) for k, s, d in SC.compare_npz(a, b)}["trimodal/const/sel_idx"]
    assert st == ("FAIL", "1 of 5 entries differ")


@pytest.mark.parametrize("change", ["nan", "shape", "missing", "extra", "string", "dtype"])
def test_structural_differences_fail(change):
    a, b = _base(), _base()
    if change == "nan":
        b["trimodal/const/e"] = np.array([1.0, np.nan])
    elif change == "shape":
        b["trimodal/const/e"] = np.array([1.0, 2.0, 3.0])
    elif change == "missing":
        del b["trimodal/const/e"]
    elif change == "extra":
        b["new/key"] = np.zeros(2)
    elif change == "string":
        b["target"] = np.array("inhom")
    elif change == "dtype":
        b["sets"] = b["sets"].astype(float)
    assert SC.summarize(SC.compare_npz(a, b))["verdict"] == "FAIL"


def test_relative_difference_reported():
    a, b = _base(), _base()
    b["trimodal/const/e"] = np.array([1.0, 2.0 + 2e-10])
    d = {k: dd for k, _, dd in SC.compare_npz(a, b)}["trimodal/const/e"]
    assert "max abs 2e-10" in d and "max rel 1e-10" in d


def test_tree_comparison(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for t in (a, b):
        (t / "src").mkdir(parents=True)
        (t / "src" / "x.py").write_text("x = 1\n")
        (t / ".DS_Store").write_bytes(b"junk")
        (t / "src" / "__pycache__").mkdir()
        (t / "src" / "__pycache__" / "x.pyc").write_bytes(b"c")
    (b / ".DS_Store").write_bytes(b"other junk")
    assert SC.compare_trees(SC.tree_files(a), SC.tree_files(b)) == ([], [], [])
    (b / "src" / "x.py").write_text("x = 2\n")
    (b / "README.md").write_text("r\n")
    (a / "results").mkdir()
    (a / "results" / "r.npz").write_bytes(b"r")
    assert SC.compare_trees(SC.tree_files(a), SC.tree_files(b)) == ([], ["README.md"], ["src/x.py"])


def test_dry_run_prints_plan_without_computing(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    for name, _ in SC.SPOT:
        (res / f"{name}.manifest.json").write_text('{"wall_s": 360, "workers": 10}')
    r = subprocess.run([sys.executable, str(REPORT / "spotcheck.py"), "--results", str(res),
                        "--out", str(tmp_path / "out"), "--workers", "8", "--dry-run"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "dry run: nothing computed" in r.stdout
    assert "estimate: 30 min with 8 workers" in r.stdout      # 4 x 1 core-h x 60 / 8
    assert not (tmp_path / "out").exists()
