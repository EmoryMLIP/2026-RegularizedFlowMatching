"""experiments/compute/controls_iid.py: which N a run computes and where it requires the
dependency's production-latent lam -> 0 cell (iid_ns, ref_required, ref_value):
every N for every 2D target, at the base resolutions and at m = 512.

Unit level, and one smoke chain controls -> controls_iid at --m 512 for a target other than
the circle (marker script). It is not an invocation of release_helpers.INVOCATIONS: their
roots also run --example, which at m = 512 is one full-size hold-out job (hours)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from release_helpers import COMPUTE, load_isolated, run_script

_, CI = load_isolated(COMPUTE, "controls_iid", "cmp_controls_iid")
from smoothfm import config  # noqa: E402

T2 = ("circle", "checkerboard", "spirals", "gauss8")
NS = tuple(int(n) for n in config.NS)


def _a(target=None, m=None, smoke=False):
    return SimpleNamespace(target=target, m=m, smoke=smoke)


@pytest.mark.parametrize("case", ("trimodal", "inhom") + T2)
def test_base_resolution_every_n(case):
    a = _a(None if case in ("trimodal", "inhom") else case)
    assert tuple(CI.iid_ns(a, case, (100,))) == NS
    for n in NS:
        assert CI.ref_required(a, case, n)


@pytest.mark.parametrize("target", T2)
def test_extra_resolution_every_n(target):
    assert tuple(CI.iid_ns(_a(target, 512), target, (100,))) == NS


@pytest.mark.parametrize("target,need", [("circle", (1000,)), ("checkerboard", ()),
                                         ("spirals", ()), ("gauss8", ())])
def test_extra_resolution_reference_only_where_computed(target, need):
    a = _a(target, 512)
    assert tuple(n for n in NS if CI.ref_required(a, target, n)) == need


def test_smoke_uses_protocol_n_and_requires_every_reference():
    a = _a("gauss8", 512, smoke=True)
    assert tuple(CI.iid_ns(a, "gauss8", (100,))) == (100,)
    assert CI.ref_required(a, "gauss8", 100)


def test_ref_value_nan_without_the_cell():
    ctl = {"c/m512/lam0/tv_pos": np.array([[0.5, 0.6]]),
           "c/m512/lam0/tv_pos_se": np.array([[0.01, 0.02]])}
    assert CI.ref_value(ctl, "c", 512, [1000], [0, 1], 1000, 1) == (0.6, 0.02)
    for args in ((100, 0), (1000, 3)):
        v, se = CI.ref_value(ctl, "c", 512, [1000], [0, 1], *args)
        assert np.isnan(v) and np.isnan(se)
    v, se = CI.ref_value({}, "c", 512, [1000], [0, 1], 1000, 0)
    assert np.isnan(v) and np.isnan(se)


@pytest.mark.script
@pytest.mark.slow
def test_smoke_chain_at_the_extra_resolution(tmp_path):
    """controls --target gauss8 --m 512 --smoke, then controls_iid on it: the smoke m, the
    smoke N, a finite reference (the smoke dependency has its lam -> 0 row) and tv_law."""
    args = ["--target", "gauss8", "--m", "512"]
    r = run_script("controls", ["--smoke", *args], tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    r = run_script("controls_iid", ["--smoke", *args], tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    m = config.CONTROLS_M_EXTRA["gauss8"]["smoke_m"]
    with np.load(tmp_path / "smoke" / "controls_iid_gauss8_m512.npz") as z:
        assert list(z["gauss8/ms"]) == [m]
        assert list(z["ns"]) == [config.SMOKE["N"]]
        for k in ("tv_iid", "tv_iid_se", "ref_tv_pos"):
            assert np.isfinite(z[f"gauss8/m{m}/lam0/{k}"]).all(), k
        assert np.isfinite(z["gauss8/tv_law"]).all()
