"""Axis and colour-bar labels of fig_hero: the paper caption describes the panels
in words, so fig_hero labels them in words."""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments" / "report"
RESULTS = pathlib.Path(os.environ.get("SMOOTHFM_HERO_RESULTS") or ROOT / "results" / "smoke")

HERO_LABEL_REF = {"xlabel": [r"time $t$", r"time $t$", r"scaled velocity $(1-t)\,v$"],
                  "ylabel": [r"position $x$", r"velocity\\error", r"$\lambda(t)/\lambda_s$"]}


@pytest.fixture(scope="module")
def hero_tex(tmp_path_factory):
    if not (RESULTS / "hero_1d.npz").is_file():
        pytest.skip(f"no hero_1d.npz under {RESULTS}")
    out = tmp_path_factory.mktemp("hero")
    # a fresh interpreter: experiments/compute has its own common.py, and another test module
    # may have imported it under the same name
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import common as C, hero; "
            "C.use(sys.argv[2]); C.set_out(sys.argv[3]); hero.build()")
    r = subprocess.run([sys.executable, "-c", code, str(REPORT), str(RESULTS), str(out)],
                       capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    tex = (out / "figures" / "fig_hero.tex").read_text()
    return "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))


def test_hero_axis_labels_in_words(hero_tex):
    """Two time axes, the colour bar, position, the two-line velocity error (centred) and
    lam(t)/lam_s."""
    if "missing" in hero_tex and "begin{axis}" not in hero_tex:
        pytest.skip("hero_1d.npz incomplete: placeholder figure")
    for key, want in HERO_LABEL_REF.items():
        got = re.findall(rf"(?<!\w){key}=\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}", hero_tex)
        assert sorted(got) == sorted(want), (key, got)
    assert "ylabel style={align=center}" in hero_tex
