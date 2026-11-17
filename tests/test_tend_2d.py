"""The t_end limit in 2D (tab_tend, 2D rows): tend.py --target T.

Mathematics under test (main.tex eq:empirical):
- rho_N(., t) is the law of t x_i + sigma z, z ~ K, an equal-weight mixture of N boxes of
  side 2 L sigma, sigma = 1 - t. The exact flow of v_N carries K onto rho_N(., t).
- Late in the flow one weight w_k is 1 to rounding, so v_N = (x_k - x)/sigma and
  x(t) - x_k = sigma c with c constant: the distance to the nearest training point is
  O(sigma) (memorization), and the rescaled offset c = (x(t) - x_k)/sigma does not change
  between t_end = 1 - 1e-4 and 1 - 1e-6.
- 1 - TV(rho_N(t), pi) = int min(pi, rho_N) <= pi(supp rho_N) <= N (2 L sigma)^2 max pi, so
  1 - TV falls like sigma^2: by at least a factor 100 per factor 100 in 1 - t_end (the
  bound gives 1e4; 100 leaves room for Monte Carlo error and the box-face terms).

Contract: ``tend.py --target T`` writes result ``tend_<T>`` with keys
<T>/<kind>/{tv_pos, tv_pos_se, tv_iid, tv_iid_se, frac_out, nfev} for kind in exact,
oracle, holdout, arrays (len(ns), len(sets), len(one_minus_t_end)); exact also tv_law,
tv_law_se, ref_nstep, ref_converged, ref_err_x, ref_err_logf; no nonmono in 2D
(a 1D-only measure); ns, sets,
one_minus_t_end, frame_c, frame_s, frame_L. lam(t) from twod_<T>.
"""

from __future__ import annotations

import numpy as np
import pytest

from smoothfm import (AffineFrame, AffineTarget, Checkerboard, EmpiricalFlow, NoisyCircle,
                      StoppedLaw, TruncatedGaussian, config, tv_pos)
from smoothfm.sampler import reference_rk4

OM = (1e-2, 1e-4, 1e-6)
KINDS = ("exact", "oracle", "holdout")
PER_T = ("tv_pos", "tv_pos_se", "tv_iid", "tv_iid_se", "frac_out", "nfev")
#: 1D-only measure of the transport map (order violations): absent in 2D.
ONLY_1D = ("nonmono",)
EXACT_EXTRA = ("tv_law", "tv_law_se", "ref_nstep", "ref_converged", "ref_err_x", "ref_err_logf")


# ===========================================================================
# Library level (fast): the exact v_N flow of N = 5 points as t_end -> 1
# ===========================================================================

class Case:
    """N = 5 training points of a 2D target in its frame, 128 iid truncated latents, the
    exact flow by the reference integrator at each 1 - t_end in OM."""

    def __init__(self, target, name):
        L = config.L[2]
        self.L = L
        frame = AffineFrame.from_box(np.asarray(config.DATA_BOX[name], float), L)
        self.latent = TruncatedGaussian(2, L)
        self.std = AffineTarget(target, frame, self.latent)
        y = np.asarray(target.sample(5, np.random.default_rng(7)), float)
        self.ys = np.asarray(frame.to_std(y), float).reshape(5, 2)
        self.z = self.latent.sample(128, np.random.default_rng(11))
        flow = EmpiricalFlow(self.ys, self.latent)
        self.path = [reference_rk4(flow, self.z, t_end=1.0 - e, latent=self.latent) for e in OM]

    def nearest(self, x):
        d = np.linalg.norm(np.asarray(x)[:, None, :] - self.ys[None], axis=-1)
        return d.min(1), d.argmin(1)


@pytest.fixture(scope="module", params=["circle", "checkerboard"])
def case(request):
    target = {"circle": NoisyCircle, "checkerboard": Checkerboard}[request.param]()
    return Case(target, request.param)


def test_reference_converges(case):
    for (x, lf, info), e in zip(case.path, OM):
        assert info["converged"], (e, info["nstep"], info["err_x"], info["err_logf"])
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(lf))


def test_fraction_near_a_training_point_grows_to_one(case):
    """Fraction of endpoints within r = 2e-3 (frame units) of a training point: about
    P(|z| < 0.2) at 1 - t_end = 1e-2, and 1 once sigma (sqrt(2) L + max |x_i|) < r."""
    r = 2e-3
    frac = [float(np.mean(case.nearest(x)[0] < r)) for x, _, _ in case.path]
    assert frac[0] < 0.1, frac
    assert all(b >= a for a, b in zip(frac, frac[1:])), frac
    assert frac[-1] == 1.0, frac


def test_nearest_distance_is_order_sigma(case):
    """Every endpoint lies in a box t x_k + sigma Omega, so its distance to x_k is at most
    sigma (sqrt(2) L + |x_k|) (plus the reference's own error)."""
    bound = np.sqrt(2) * case.L + np.linalg.norm(case.ys, axis=1).max()
    for (x, _, info), e in zip(case.path, OM):
        d, _ = case.nearest(x)
        assert np.all(d <= e * bound + 10 * info["err_x"]), (e, (d / e).max(), bound)


def test_late_flow_is_frozen_relative_to_its_training_point(case):
    """Late in the flow x(t) = x_k + sigma c with c constant: the nearest training point
    is the same at 1 - t_end = 1e-4 and 1e-6, and c agrees to the integration error."""
    (x4, _, i4), (x6, _, i6) = case.path[1], case.path[2]
    _, k4 = case.nearest(x4)
    _, k6 = case.nearest(x6)
    np.testing.assert_array_equal(k4, k6)
    c4 = (x4 - case.ys[k4]) / 1e-4
    c6 = (x6 - case.ys[k6]) / 1e-6
    tol = 10 * (i4["err_x"] / 1e-4 + i6["err_x"] / 1e-6) + 1e-6
    assert np.abs(c4 - c6).max() <= tol * (1 + np.abs(c4).max()), (np.abs(c4 - c6).max(), tol)


def test_tv_to_target_tends_to_one(case):
    """TV(rho_N(t_end), pi) and the exact flow's tv (tv_pos of the endpoints with the
    flow's log f) increase to 1; 1 - TV falls at least 100-fold per step (sigma^2 law);
    flow and law agree within their Monte Carlo SEs (the flow transports K to rho_N)."""
    law_tv, flow_tv = [], []
    for (x, lf, _), e in zip(case.path, OM):
        law = StoppedLaw(case.ys, 1.0 - e, case.latent)
        zl = law.sample(4000, np.random.default_rng(3))
        law_tv.append(tv_pos(case.std, zl, law.logpdf(zl)))
        flow_tv.append(tv_pos(case.std, x, lf))
    for tvs in (law_tv, flow_tv):
        for (a, sa), (b, sb) in zip(tvs, tvs[1:]):
            assert b >= a - 2 * np.hypot(sa, sb), tvs
        assert all(0.0 <= v <= 1.0 for v, _ in tvs), tvs
    gap = [1.0 - v for v, _ in law_tv]
    assert gap[0] > 0, gap
    for a, b in zip(gap, gap[1:]):
        assert b <= a / 100, gap
    assert gap[-1] < 1e-6, gap
    for (a, sa), (b, sb) in zip(law_tv, flow_tv):
        assert abs(a - b) <= 4 * np.hypot(sa, sb) + 1e-3, (law_tv, flow_tv)


# ===========================================================================
# Script runs (slow): tend.py --target circle --smoke and the 1D tend.py --smoke
# ===========================================================================

T = "circle"


def _ok(r, what):
    assert r.returncode == 0, f"{what} exit {r.returncode}\n{r.stderr[-4000:]}"


@pytest.fixture(scope="module")
def tend2d(smoke_tree):
    r = smoke_tree.run("tend", ["--target", T], deps=[("twod", ("--target", T))])
    _ok(r, f"tend --target {T} --smoke")
    return smoke_tree.npz(f"tend_{T}")


@pytest.fixture(scope="module")
def tend1d(smoke_tree):
    r = smoke_tree.run("tend", [], deps=[("oned", ())])
    _ok(r, "tend --smoke")
    return smoke_tree.npz("tend")


@pytest.mark.slow
@pytest.mark.script
def test_tend2d_keys_and_shapes(tend2d):
    a = tend2d
    for k in ("ns", "sets", "one_minus_t_end", "frame_c", "frame_s", "frame_L"):
        assert k in a, k
    shape = (len(a["ns"]), len(a["sets"]), len(a["one_minus_t_end"]))
    assert list(a["ns"]) == [config.SMOKE["N"]] and list(a["sets"]) == [config.SMOKE["SET"]]
    np.testing.assert_allclose(a["one_minus_t_end"], config.TEND["one_minus_t_end"])
    assert np.shape(a["frame_c"]) == (2,) and float(a["frame_L"]) == config.L[2]
    missing, wrong = [], []
    for kind in KINDS:
        for k in PER_T + (EXACT_EXTRA if kind == "exact" else ()):
            key = f"{T}/{kind}/{k}"
            if key not in a:
                missing.append(key)
            elif a[key].shape != shape:
                wrong.append((key, a[key].shape))
    assert not missing, missing
    assert not wrong, (shape, wrong)


@pytest.mark.slow
@pytest.mark.script
def test_tend2d_value_ranges(tend2d):
    a = tend2d
    for kind in KINDS:
        p = f"{T}/{kind}/"
        tvs = [p + "tv_pos", p + "tv_iid"] + ([p + "tv_law"] if kind == "exact" else [])
        for k in tvs:
            v = a[k]
            assert np.all(np.isfinite(v)) and np.all((v >= 0) & (v <= 1)), (k, v)
            se = a[k + "_se"]
            assert np.all(np.isfinite(se)) and np.all(se >= 0), (k + "_se", se)
        fo = a[p + "frac_out"]
        assert np.all((fo >= 0) & (fo <= 1)), fo
        assert np.all(a[p + "nfev"] > 0)
    ex = f"{T}/exact/"
    n0 = config.SMOKE["NSTEP"]
    ns = a[ex + "ref_nstep"]
    assert np.all(ns >= n0) and np.all(ns % n0 == 0), ns
    assert np.all(np.isin(a[ex + "ref_converged"], (0, 1))), a[ex + "ref_converged"]
    assert np.all(a[ex + "ref_err_x"] >= 0) and np.all(a[ex + "ref_err_logf"] >= 0)


@pytest.mark.slow
@pytest.mark.script
def test_tend2d_exact_tv_iid_nondecreasing_in_t_end(tend2d):
    """Smoke: N = 100, set 0, SMOKE_ADAPTIVE_LATENTS iid points; the SE is the Monte Carlo SE
    within the set (one set, so no SE over sets). The exact flow memorizes more as t_end ->
    1 (1 - TV ~ sigma^2), so tv_iid must not fall by more than 2 SE between t_end values."""
    v = tend2d[f"{T}/exact/tv_iid"][0, 0]
    se = tend2d[f"{T}/exact/tv_iid_se"][0, 0]
    for k in range(len(v) - 1):
        assert v[k + 1] >= v[k] - 2 * np.hypot(se[k], se[k + 1]), (v, se)


@pytest.mark.slow
@pytest.mark.script
def test_tend2d_exact_flow_matches_its_law(tend2d):
    """The exact flow carries K to rho_N(t_end): its tv_iid estimates the same TV as tv_law
    (the smoke reference may stop at SMOKE_REFERENCE_DOUBLINGS, hence the 0.02 slack)."""
    p = f"{T}/exact/"
    d = np.abs(tend2d[p + "tv_iid"] - tend2d[p + "tv_law"])
    s = np.hypot(tend2d[p + "tv_iid_se"], tend2d[p + "tv_law_se"])
    assert np.all(d <= 4 * s + 0.02), (tend2d[p + "tv_iid"], tend2d[p + "tv_law"], s)


@pytest.mark.slow
@pytest.mark.script
def test_tend2d_has_no_1d_measures_and_tend1d_keeps_them(tend2d, tend1d):
    """nonmono is 1D-only: absent from the 2D result, present for every 1D target and kind."""
    bad = [k for k in tend2d if any(m in k.lower().split("/")[-1] for m in ONLY_1D)]
    assert not bad, bad[:5]
    for d in ("trimodal", "inhom"):
        for kind in KINDS:
            for m in ONLY_1D:
                assert f"{d}/{kind}/{m}" in tend1d, f"{d}/{kind}/{m}"
    assert not [k for k in tend1d if k.startswith(f"{T}/")], "1D result carries 2D keys"
