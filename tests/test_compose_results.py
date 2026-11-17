"""experiments/report/compose_results.py: the results directory of the paper composed from
several release runs (`make paper-report`), on temporary directories."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from release_helpers import REPORT, ROOT

pytestmark = pytest.mark.report

sys.path.insert(0, str(REPORT))
try:
    import compose_results as CR  # standard library only, no 'common'
finally:
    sys.path.remove(str(REPORT))


def _tree(root, spec):
    """Result directories {rel: {file: text}} under root."""
    for rel, files in spec.items():
        d = root / rel
        d.mkdir(parents=True)
        for name, text in files.items():
            (d / name).write_text(text)


@pytest.fixture
def root(tmp_path):
    _tree(tmp_path, {
        "a/results": {"x.npz": "a-x", "y.npz": "a-y", "y.log": "a-ylog"},
        "b/results": {"y.npz": "b-y", "z.npz": "b-z"},
    })
    (tmp_path / "a" / "results" / "smoke").mkdir()
    (tmp_path / "a" / "results" / "smoke" / "x.npz").write_text("smoke")
    return tmp_path


def test_later_directories_override_per_file(root):
    dest = root / "paper" / "results"
    chosen, skipped = CR.compose(["a/results", "b/results", "v6/results"], root, dest)
    assert skipped == ["v6/results"]
    assert chosen == {"x.npz": "a/results", "y.npz": "b/results", "y.log": "a/results",
                      "z.npz": "b/results"}
    got = {p.name: p.read_text() for p in dest.iterdir()}
    assert got == {"x.npz": "a-x", "y.npz": "b-y", "y.log": "a-ylog", "z.npz": "b-z"}
    assert all(p.is_symlink() and not os.path.isabs(os.readlink(p)) for p in dest.iterdir())
    assert "smoke" not in got          # subdirectories are not entered


def test_recompose_removes_stale_links_only(root):
    dest = root / "paper" / "results"
    CR.compose(["a/results", "b/results"], root, dest)
    CR.compose(["a/results"], root, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["x.npz", "y.log", "y.npz"]
    assert (dest / "y.npz").read_text() == "a-y"
    # the sources are untouched
    assert (root / "b" / "results" / "z.npz").read_text() == "b-z"
    assert (root / "b" / "results" / "y.npz").read_text() == "b-y"


def test_refuses_a_destination_with_real_files(root):
    dest = root / "paper" / "results"
    dest.mkdir(parents=True)
    (dest / "keep.npz").write_text("real")
    with pytest.raises(RuntimeError, match="not symlinks"):
        CR.compose(["a/results"], root, dest)
    assert (dest / "keep.npz").read_text() == "real"
    assert sorted(p.name for p in dest.iterdir()) == ["keep.npz"]


def test_spec_comments_and_sources_text(root, tmp_path):
    spec = tmp_path / "spec.txt"
    spec.write_text("# header\na/results   # first\n\nb/results\nv6/results  # later\n")
    assert CR.read_spec(spec) == ["a/results", "b/results", "v6/results"]
    r = subprocess.run([sys.executable, str(REPORT / "compose_results.py"), "--spec", str(spec),
                        "--root", str(root)], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "RESULTS_ROOT": ""})
    assert r.returncode == 0, r.stderr
    assert "v6/results absent, skipped" in r.stdout
    src = (root / "paper" / "SOURCES.txt").read_text().splitlines()
    assert "# skipped (absent): v6/results" in src
    rows = dict(ln.split("\t") for ln in src if not ln.startswith("#"))
    assert rows == {"x.npz": "a/results", "y.log": "a/results", "y.npz": "b/results",
                    "z.npz": "b/results"}


def test_committed_spec_order():
    spec = CR.read_spec(CR.DEFAULT_SPEC)
    assert spec == ["c2ddb03/results", "57a5270-v2/results", "c530cb8-stability/results",
                    "v6/results", "v7/results", "v8/results"]


def test_make_paper_report_target():
    mk = (ROOT / "Makefile").read_text()
    assert "\npaper-report:\n" in mk
    body = mk.split("\npaper-report:\n", 1)[1].split("\n\n", 1)[0]
    assert "compose_results.py" in body and '--dest "$(RESULTS_ROOT)/paper/results"' in body
    assert '--results "$(RESULTS_ROOT)/paper/results" --out $(OUT)' in body
    assert "RESULTS_ROOT ?= results\n" in mk     # no machine-specific default (C1)
    assert "PAPER ?= $(SMOOTHFM_PAPER)" in mk
