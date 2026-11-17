"""config.py constants (main.tex sec:setup, sec:solver, sec:cv) and the repository-wide
invariants.

Interface the tests assume:
- TIMES is a sequence of 49 floats in increasing order (19 + 30 points).
- LAM_DECADES[reg][D] is a pair (lo, hi) of integer decades of lam / lam_s; the
  values are not known to the tests, only their structure and that the grid
  is relative (no frame factor).
- SMOKE is a dict or namespace with keys N, set, m (per D), times, NSTEP, latents.
"""

from __future__ import annotations

import ast
import io
import pathlib
import tokenize

import numpy as np
import pytest

from conftest import pick, value

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "smoothfm"
EXP = ROOT / "experiments"

config = pytest.importorskip("smoothfm.config")


def _get(obj, *keys):
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
        if not isinstance(obj, dict) and hasattr(obj, k):
            return getattr(obj, k)
    pytest.fail(f"SMOKE has none of {keys}")


def test_model_constants():
    assert config.MU == 1e-8
    assert dict(config.L) == {1: 5.0, 2: 4.5}
    assert config.T_END == 1 - 1e-4
    assert config.NSTEP == 96
    assert dict(config.M_SPLINE) == {1: 512, 2: 128}
    assert config.LAM_FLOOR_REL == 1e-14
    assert config.SPLIT_RATIO == 100
    assert not hasattr(config, "RADIUS")


def test_times():
    T = np.asarray(config.TIMES, float)
    ref = np.concatenate([np.linspace(0, 0.9, 19), 1 - np.geomspace(10 ** -1.1, 1e-4, 30)])
    assert T.shape == (49,)
    np.testing.assert_allclose(T, ref, rtol=0, atol=1e-15)
    assert np.all(np.diff(T) > 0)
    assert T[-1] == pytest.approx(config.T_END, abs=1e-15)


def test_protocol_constants():
    assert tuple(config.NS) == (100, 1000, 3000)
    assert config.SETS == 5
    assert dict(config.M_LAT) == {1: 3000, 2: 4096}
    assert isinstance(config.SEEDS, dict) and len(config.SEEDS) > 0


def test_lambda_decades_relative_grid():
    """A1: decades of lam / lam_s per penalty and dimension, integers, lo < hi."""
    dec = config.LAM_DECADES
    for reg in ("h2", "h1", "l2"):
        for D in (1, 2):
            lo, hi = dec[reg][D]
            assert float(lo).is_integer() and float(hi).is_integer()
            assert lo < hi
    # no frame factor anywhere in config (A1)
    names = [n for n in dir(config) if not n.startswith("_")]
    assert not any("FRAME_FACTOR" in n or "LAM_SCALE" in n for n in names)


def test_smoke_block():
    S = config.SMOKE
    assert _get(S, "N", "n") == 100
    assert _get(S, "set", "SET", "sets") in (0, (0,), [0])
    assert dict(_get(S, "m", "M_SPLINE", "M")) == {1: 64, 2: 24}
    assert _get(S, "NSTEP", "nstep") == 24
    lat = _get(S, "latents", "M_LAT", "n_latent")
    assert (set(dict(lat).values()) if isinstance(lat, dict) else {lat}) == {512}
    T = np.asarray(_get(S, "times", "TIMES"), float)
    assert len(T) == 9
    assert np.any(np.isclose(T, 1 - 1e-4, rtol=0, atol=1e-15))
    assert np.all(np.isin(np.round(T, 15), np.round(np.asarray(config.TIMES), 15)))


# ---------------------------------------------------------------------------
# Repository-wide invariants (Sec. 5.1 last item)
# ---------------------------------------------------------------------------

def _py_files():
    files = list(SRC.rglob("*.py"))
    if EXP.exists():
        files += list(EXP.rglob("*.py"))
    return [f for f in files if "__pycache__" not in f.parts]


def test_no_identifier_eps():
    bad = []
    for f in _py_files():
        src = f.read_text()
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.NAME and tok.string == "eps":
                bad.append(f"{f.relative_to(ROOT)}:{tok.start[0]}")
    assert not bad, f"identifier eps in: {bad[:20]}"


def test_no_prior_classes():
    bad = []
    for f in _py_files():
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Prior"):
                bad.append(f"{f.relative_to(ROOT)}:{node.name}")
    assert not bad, bad


@pytest.mark.parametrize("name", ["MU", "L", "NSTEP", "T_END"])
def test_config_is_only_definition(name):
    """No module other than config.py assigns these names at module or class level.

    Scope: src/ only; tests/test_scripts.py checks the compute scripts."""
    bad = []
    for f in [f for f in _py_files() if SRC in f.parents]:
        if f == SRC / "config.py":
            continue
        tree = ast.parse(f.read_text())
        scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        for sc in scopes:
            for node in sc.body:
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    targets = [node.target]
                for tg in targets:
                    for nm in ast.walk(tg):
                        if isinstance(nm, ast.Name) and nm.id == name:
                            bad.append(f"{f.relative_to(ROOT)}:{node.lineno}")
    assert not bad, bad


# ---------------------------------------------------------------------------
# Refinement study (tab_refine_1d, tab_refine_2d): every 2D target at the
# extra resolution, every N
# ---------------------------------------------------------------------------

T2 = ("circle", "checkerboard", "spirals", "gauss8")


def test_controls_iid_every_n_for_every_case():
    """lam -> 0 with iid latents at N = 100, 1000, 3000 for the 1D cases and every 2D
    target."""
    ns = config.CONTROLS_IID["ns"]
    assert set(ns) == {"trimodal", "inhom", *T2}
    for c, v in ns.items():
        assert tuple(v) == tuple(config.NS), c


def test_controls_m_extra_every_2d_target():
    """controls.py / controls_iid.py --m: every 2D target at m = 512, hold-out for
    the three penalties at every N, the iid lam -> 0 row at every N."""
    ex = config.CONTROLS_M_EXTRA
    assert set(ex) == set(T2)
    for t in T2:
        e = ex[t]
        assert e["m"] == 512
        assert e["m"] not in config.CONTROLS_M[2]
        assert e["smoke_m"] not in config.CONTROLS_M_SMOKE[2]
        assert {r: tuple(v) for r, v in e["holdout_ns"].items()} == {
            r: tuple(config.NS) for r in ("h2", "h1", "l2")}, t
        assert tuple(e["iid_ns"]) == tuple(config.NS), t


def test_controls_m_extra_lam0_rows_only_where_released():
    """The production-latent lam -> 0 row of controls_<T>_m512 is only a reference for
    controls_iid (ref_tv_pos), which fits the lam -> 0 flow itself. It stays where the
    release computed it (circle, N = 1000) and is not run for the other targets."""
    ex = config.CONTROLS_M_EXTRA
    assert tuple(ex["circle"]["lam0_ns"]) == (1000,)
    for t in ("checkerboard", "spirals", "gauss8"):
        assert tuple(ex[t]["lam0_ns"]) == (), t
    for t in T2:
        assert set(ex[t]["lam0_ns"]) <= set(ex[t]["iid_ns"]), t
