"""Time-integration error in 2D (tab_integrator, 2D rows): integration_error.py --target T.

Mathematics under test (main.tex sec:sampling): generation is classical RK4 in
tau = -log(1 - t) on dx/dtau = (1 - t) v(x, t), dlog f/dtau = -(1 - t) div v(x, t). On a
smooth field its global error is C h^4 + O(h^5), so doubling the step count divides the
error by about 16. Two 2D fields with a closed-form flow map (neither in RK4's exactness
class in tau, because of the factor e^{-tau}):
- linear, v = (1 + t) A x: x(t) = expm(A (t + t^2/2)) x_0, log f(t) = log f_0
  - tr(A) (t + t^2/2);
- nonlinear rotation, v = w(|x|) J x with w(r) = 1 + r^2, J the rotation by pi/2: |x| is
  conserved, x(t) = R(w(|x_0|) t) x_0, and div v = w'(r) (x/r) . J x = 0, so log f is
  constant.

Contract: ``integration_error.py --target T`` writes
``integration_error_<T>`` with keys <T>/<flow>/<integ>/{tv, tv_se, wall, nfev, frac_out}
(ns, sets) and <T>/<flow>/rk<k>/dx (ns, sets, m, 2) in original units, dlogf (ns, sets, m);
flows exact (reference 'ref', RK4 step doubling) and holdout (reference 'dop853'); RK4 at 1,
4, 16 x NSTEP. Without --target the result is the 1D one, ``integration_error``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from smoothfm import config
from smoothfm.sampler import sample, sample_path, tau_of_t

A = np.array([[-0.3, 1.2], [-0.8, 0.1]])


class Linear:
    dim = 2

    def __call__(self, x, t):
        return (1 + t) * np.asarray(x, float).reshape(-1, 2) @ A.T

    def div(self, x, t):
        return np.full(np.asarray(x).reshape(-1, 2).shape[0], (1 + t) * np.trace(A))

    @staticmethod
    def exact(z, t):
        p = t + t * t / 2
        return z @ expm(A * p).T, np.full(len(z), -np.trace(A) * p)


class Rotation:
    dim = 2

    def __call__(self, x, t):
        x = np.asarray(x, float).reshape(-1, 2)
        w = 1 + np.sum(x * x, 1)
        return w[:, None] * np.stack([-x[:, 1], x[:, 0]], 1)

    def div(self, x, t):
        return np.zeros(np.asarray(x).reshape(-1, 2).shape[0])

    @staticmethod
    def exact(z, t):
        a = (1 + np.sum(z * z, 1)) * t
        c, s = np.cos(a), np.sin(a)
        return np.stack([c * z[:, 0] - s * z[:, 1], s * z[:, 0] + c * z[:, 1]], 1), \
            np.zeros(len(z))


FIELDS = {"linear": (Linear, 2.0), "rotation": (Rotation, 0.8)}


def _points(scale):
    return np.random.default_rng(0).uniform(-scale, scale, (40, 2))


def _errors(v, z, t_end, runs):
    xe, le = v.exact(z, t_end)
    return [(np.abs(x - xe).max(), np.abs(lf - le).max()) for x, lf in runs]


def _assert_fourth_order(err, what):
    ex = np.array([e[0] for e in err])
    rx = ex[:-1] / ex[1:]
    assert np.all((rx > 12) & (rx < 20)), (what, "x", ex, rx)
    el = np.array([e[1] for e in err])
    if el.max() > 1e-13:          # the rotation conserves log f exactly
        rl = el[:-1] / el[1:]
        assert np.all((rl > 12) & (rl < 20)), (what, "log f", el, rl)


@pytest.mark.parametrize("field", FIELDS)
def test_sample_rk4_is_fourth_order(field):
    """``sample`` (integration_error's rk<k>) at NSTEP, 2 NSTEP, 4 NSTEP to T_END: the error
    against the exact flow map falls by about 16 per doubling."""
    cls, scale = FIELDS[field]
    v, z = cls(), _points(scale)
    runs = [sample(v, z, nstep=k * config.NSTEP, t_end=config.T_END)[:2] for k in (1, 2, 4)]
    _assert_fourth_order(_errors(v, z, config.T_END, runs), field)


@pytest.mark.parametrize("field", FIELDS)
def test_sample_path_rk4_is_fourth_order_at_each_t_end(field):
    """``sample_path`` (tend's tuned flows) with dtau = tau(t_e)/K, K = NSTEP, 2 NSTEP,
    4 NSTEP, read off at 1 - t_end in {1e-2, 1e-4} (both multiples of dtau): fourth order
    at each t_end, and the state at an intermediate t_end equals ``sample`` to it."""
    cls, scale = FIELDS[field]
    v, z = cls(), _points(scale)
    tes = [1 - 1e-2, 1 - 1e-4]
    # tau(1 - 1e-4) = 2 tau(1 - 1e-2): a common grid for both
    errs = {te: [] for te in tes}
    for k in (1, 2, 4):
        dtau = tau_of_t(tes[0]) / (k * config.NSTEP)
        path = sample_path(v, z, tes, dtau)
        for te, (x, lf) in zip(tes, path):
            errs[te].append((x, lf))
        xs, ls, _ = sample(v, z, nstep=k * config.NSTEP, t_end=tes[0])
        np.testing.assert_allclose(path[0][0], xs, rtol=0, atol=1e-12)
        np.testing.assert_allclose(path[0][1], ls, rtol=0, atol=1e-12)
    for te in tes:
        _assert_fourth_order(_errors(v, z, te, errs[te]), f"{field} t_end {te}")


# ===========================================================================
# Script runs (slow): integration_error.py --target spirals --smoke; 1D unchanged
# ===========================================================================

T = "spirals"
FLOWS = {"exact": "ref", "holdout": "dop853"}
SCALARS = ("tv", "tv_se", "wall", "nfev", "frac_out")


def _ok(r, what):
    assert r.returncode == 0, f"{what} exit {r.returncode}\n{r.stderr[-4000:]}"


def _rk():
    return [f"rk{int(f * config.SMOKE['NSTEP'])}" for f in config.INTEGRATION_ERROR["step_factors"]]


@pytest.fixture(scope="module")
def ie2d(smoke_tree):
    r = smoke_tree.run("integration_error", ["--target", T], deps=[("twod", ("--target", T))])
    _ok(r, f"integration_error --target {T} --smoke")
    return smoke_tree.npz(f"integration_error_{T}")


@pytest.fixture(scope="module")
def ie1d(smoke_tree):
    r = smoke_tree.run("integration_error", [], deps=[("oned", ())])
    _ok(r, "integration_error --smoke")
    return smoke_tree.npz("integration_error")


@pytest.mark.slow
@pytest.mark.script
def test_ie2d_keys_and_shapes(ie2d):
    a = ie2d
    assert config.INTEGRATION_ERROR["step_factors"] == (1, 4, 16)
    ns, sets = list(a["ns"]), list(a["sets"])
    assert ns == [config.SMOKE["N"]] and sets == [config.SMOKE["SET"]]
    shape = (len(ns), len(sets))
    missing, wrong = [], []
    for flow, ref in FLOWS.items():
        for g in [ref] + _rk():
            for k in SCALARS:
                key = f"{T}/{flow}/{g}/{k}"
                if key not in a:
                    missing.append(key)
                elif a[key].shape != shape:
                    wrong.append((key, a[key].shape))
    assert not missing, missing
    assert not wrong, (shape, wrong)
    m = None
    for flow in FLOWS:
        for g in _rk():
            dx, dl = a[f"{T}/{flow}/{g}/dx"], a[f"{T}/{flow}/{g}/dlogf"]
            assert dx.ndim == 4 and dx.shape[:2] == shape and dx.shape[3] == 2, dx.shape
            assert dl.shape == dx.shape[:3], (dl.shape, dx.shape)
            m = dx.shape[2] if m is None else m
            assert dx.shape[2] == m, "all flows and integrators use the same latents"
    assert m == config.SMOKE_ADAPTIVE_LATENTS, m
    assert not [k for k in a if k.startswith(f"{T}/") and k.endswith(("ref/dx", "dop853/dx"))]


@pytest.mark.slow
@pytest.mark.script
def test_ie2d_value_ranges(ie2d):
    a = ie2d
    for flow, ref in FLOWS.items():
        for g in [ref] + _rk():
            p = f"{T}/{flow}/{g}/"
            tv, se = a[p + "tv"], a[p + "tv_se"]
            assert np.all(np.isfinite(tv)) and np.all((tv >= 0) & (tv <= 1)), (p, tv)
            assert np.all(np.isfinite(se)) and np.all(se >= 0), (p, se)
            assert np.all((a[p + "frac_out"] >= 0) & (a[p + "frac_out"] <= 1))
            assert np.all(a[p + "nfev"] > 0) and np.all(a[p + "wall"] >= 0)
        for g in _rk():
            p = f"{T}/{flow}/{g}/"
            assert np.all(np.isfinite(a[p + "dx"])), p + "dx"
            assert np.all(np.isfinite(a[p + "dlogf"])), p + "dlogf"


@pytest.mark.slow
@pytest.mark.script
def test_ie2d_rk4_nfev_is_four_per_step(ie2d):
    for flow in FLOWS:
        for g in _rk():
            nfev = ie2d[f"{T}/{flow}/{g}/nfev"]
            assert np.all(nfev == 4 * int(g[2:])), (flow, g, nfev)


@pytest.mark.slow
@pytest.mark.script
def test_ie2d_holdout_error_falls_with_the_step(ie2d):
    """The hold-out spline field is C^2: RK4 converges to the DOP853 reference, at least at
    first order (a factor >= 4 per factor 4 in steps) until the reference floor, taken as
    1e-7 (DOP853 rtol 1e-10 on coordinates of order 1-10)."""
    e = [np.abs(ie2d[f"{T}/holdout/{g}/dx"]).max() for g in _rk()]
    for a, b in zip(e, e[1:]):
        assert b <= max(a / 4, 1e-7), e
    assert e[0] > 0, "RK4 at the smoke NSTEP equals DOP853 exactly"


@pytest.mark.slow
@pytest.mark.script
def test_ie1d_has_no_2d_keys(ie1d):
    """Without --target the result holds the 1D targets only."""
    assert ie1d, "empty 1D result"
    assert not [k for k in ie1d if k.startswith(f"{T}/")], "1D result carries 2D keys"
