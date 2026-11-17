"""experiments/compute/controls.py: the job plan of the refinement study (plan), in
particular --m at the extra resolution of every 2D target (tab_refine_2d).

Unit level: the plan is computed from config without running a job."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from release_helpers import COMPUTE, load_isolated

CC, CTL = load_isolated(COMPUTE, "controls", "cmp_controls")
from smoothfm import config  # noqa: E402

T2 = ("circle", "checkerboard", "spirals", "gauss8")


def _plan(target, reg="h2", m=None, smoke=False, dim=None):
    a = SimpleNamespace(dim=dim, target=target, reg=reg, m=m, smoke=smoke)
    P = CC.Protocol(SimpleNamespace(smoke=smoke), 1 if dim == 1 else 2)
    return CTL.plan(a, P)


def _cells(jobs, rule):
    return sorted({(j[1], j[2]) for j in jobs if j[4] == rule})


@pytest.mark.parametrize("target", T2)
@pytest.mark.parametrize("reg,sfx", [("h2", ""), ("h1", "_h1"), ("l2", "_l2")])
def test_extra_resolution_hold_out_at_every_n(target, reg, sfx):
    name, cases, ms, _, jobs = _plan(target, reg, m=512)
    assert name == f"controls_{target}{sfx}_m512"
    assert cases == [target] and ms == {target: [512]}
    assert {j[3] for j in jobs} == {512}
    every = sorted((n, s) for n in config.NS for s in range(config.SETS))
    assert _cells(jobs, "holdout") == every


@pytest.mark.parametrize("target,lam0", [("circle", (1000,)), ("checkerboard", ()),
                                         ("spirals", ()), ("gauss8", ())])
def test_extra_resolution_lam0_rows(target, lam0):
    """The production-latent lam -> 0 row only where config lam0_ns has it (circle N = 1000,
    as released); the iid row is controls_iid's."""
    _, _, _, _, jobs = _plan(target, "h2", m=512)
    assert _cells(jobs, "lam0") == sorted((n, s) for n in lam0 for s in range(config.SETS))


def test_extra_resolution_smoke_uses_smoke_m_and_both_rows():
    _, _, ms, _, jobs = _plan("checkerboard", "h2", m=512, smoke=True)
    m = config.CONTROLS_M_EXTRA["checkerboard"]["smoke_m"]
    assert ms == {"checkerboard": [m]}
    assert {j[4] for j in jobs} == {"holdout", "lam0"}
    assert {j[1] for j in jobs} == {config.SMOKE["N"]}


def test_base_resolutions_unchanged():
    name, _, ms, rows, jobs = _plan("spirals", "h2")
    assert name == "controls_spirals"
    assert ms == {"spirals": list(config.CONTROLS_M[2])}
    assert rows == ("holdout", "lam0")
    assert len(jobs) == len(config.NS) * config.SETS * len(config.CONTROLS_M[2]) * 2
