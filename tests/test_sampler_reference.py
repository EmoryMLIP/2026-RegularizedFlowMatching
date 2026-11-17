"""sampler.reference_rk4: RK4 under step doubling as the reference integrator.

Spec under test:

    reference_rk4(v, latent_points, t_end=config.T_END, n0=config.NSTEP,
                  tol_x=config.REFERENCE_RK4["tol_x"],
                  tol_logf=config.REFERENCE_RK4["tol_logf"],
                  max_doublings=config.REFERENCE_RK4["max_doublings"], latent=None)
      -> (x, logf, info)

RK4 at n0, 2 n0, 4 n0, ... uniform steps in tau (the ODE, grid, right-hand side and start
of ``sample``); stop at the first level n with max |x_n - x_{n/2}|_inf <= tol_x and
mean |logf_n - logf_{n/2}| <= tol_logf, or after max_doublings doublings; return the
finest level. info: levels (dicts nstep, dx, dlogf, dlogf_max, wall; None on the first),
nstep, converged, err_x, err_logf, err_logf_max, nfev.

The log f part of the rule uses the mean over points,
mean_i |logf_n - logf_{n/2}| <= tol_logf; dlogf and err_logf are that mean, dlogf_max and
err_logf_max the max. dx and err_x stay the max over points and coordinates.

Expected values come from ``sample`` (the fixed-step RK4 the reference must reproduce
level by level), from the exact affine flow, from DOP853, and, for the empirical field,
from the transport property of the exact flow (the endpoint of the latent quantile q has
stopped-law CDF q).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import latent

sampler = pytest.importorskip("smoothfm.sampler")
config = pytest.importorskip("smoothfm.config")
from smoothfm.empirical import EmpiricalFlow, StoppedLaw  # noqa: E402

reference_rk4 = sampler.reference_rk4

A0 = np.array([[0.6, 1.1], [-0.9, 0.3]])
B0 = np.array([0.4, -0.2])


class Affine:
    """v = A x + b, div v = tr A."""

    def __init__(self, A, b, L=4.5):
        self.A, self.b = np.asarray(A, float), np.asarray(b, float)
        self.D = self.dim = len(self.b)
        self.latent = latent(self.D, L)

    def __call__(self, x, t):
        return np.asarray(x, float).reshape(-1, self.D) @ self.A.T + self.b

    def div(self, x, t):
        return np.full(np.asarray(x).reshape(-1, self.D).shape[0], np.trace(self.A))


class Nonlinear:
    """v = -x + a (1 + t) sin(x) componentwise, div v = sum_d (-1 + a (1 + t) cos(x_d)).

    Smooth in (x, t) and not in the RK4 exactness class, so the step-doubling differences
    show the h^4 rate."""

    def __init__(self, D=2, a=0.8, L=4.5):
        self.D = self.dim = D
        self.a = a
        self.latent = latent(D, L)

    def __call__(self, x, t):
        x = np.asarray(x, float).reshape(-1, self.D)
        return -x + self.a * (1 + t) * np.sin(x)

    def div(self, x, t):
        x = np.asarray(x, float).reshape(-1, self.D)
        return np.sum(-1 + self.a * (1 + t) * np.cos(x), axis=1)


def z0(n=40, seed=0, D=2):
    return np.random.default_rng(seed).uniform(-2, 2, size=(n, D))


def as_xl(out, shape):
    return np.asarray(out[0], float).reshape(shape), np.asarray(out[1], float).reshape(-1)


def fixed(v, z, nstep, t_end):
    x, lf, _ = sampler.sample(v, z, nstep=nstep, t_end=t_end)
    return as_xl((x, lf), z.shape)


def test_config_defaults():
    assert config.REFERENCE_RK4 == {"tol_x": 1e-6, "tol_logf": 1e-4, "max_doublings": 6}


def test_levels_are_fixed_step_rk4():
    """Every level is ``sample`` at that step count: the finest level is returned, and the
    recorded differences are those of ``sample`` at successive step counts (this pins
    level 0 to ``sample`` at n0)."""
    v, z, n0, te = Nonlinear(), z0(), 12, 0.99
    x, lf, info = reference_rk4(v, z, t_end=te, n0=n0, tol_x=0.0, tol_logf=0.0,
                                max_doublings=3)
    x, lf = as_xl((x, lf), z.shape)
    runs = [fixed(v, z, n0 * 2**k, te) for k in range(4)]
    np.testing.assert_allclose(x, runs[-1][0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(lf, runs[-1][1], rtol=0, atol=1e-14)
    lev = info["levels"]
    assert [int(L["nstep"]) for L in lev] == [n0 * 2**k for k in range(4)]
    assert lev[0]["dx"] is None and lev[0]["dlogf"] is None and lev[0]["dlogf_max"] is None
    for k in range(1, 4):
        dx = np.abs(runs[k][0] - runs[k - 1][0]).max()
        e = np.abs(runs[k][1] - runs[k - 1][1])
        assert lev[k]["dx"] == pytest.approx(dx, rel=1e-9, abs=1e-15)
        assert lev[k]["dlogf"] == pytest.approx(e.mean(), rel=1e-9, abs=1e-15)
        assert lev[k]["dlogf_max"] == pytest.approx(e.max(), rel=1e-9, abs=1e-15)
        assert lev[k]["dlogf"] <= lev[k]["dlogf_max"]
    for L in lev:
        assert float(L["wall"]) >= 0


def test_max_doublings_zero_is_sample():
    """With no doubling allowed the result is ``sample`` at n0 (level 0 alone)."""
    v, z, n0, te = Nonlinear(), z0(), 12, 0.99
    x, lf, info = reference_rk4(v, z, t_end=te, n0=n0, max_doublings=0)
    xs, ls = fixed(v, z, n0, te)
    np.testing.assert_allclose(as_xl((x, lf), z.shape)[0], xs, rtol=0, atol=1e-14)
    np.testing.assert_allclose(as_xl((x, lf), z.shape)[1], ls, rtol=0, atol=1e-14)
    assert int(info["nstep"]) == n0 and len(info["levels"]) == 1


@pytest.mark.parametrize("field", ["affine", "nonlinear"])
def test_fourth_order_differences(field):
    """|x_n - x_{n/2}| ~ C (1 - 2^-4) (tau_end / (n/2))^4: successive ratios tend to 16."""
    v = Affine(A0, B0) if field == "affine" else Nonlinear()
    z = z0()
    _, _, info = reference_rk4(v, z, t_end=0.99, n0=96, tol_x=0.0, tol_logf=0.0,
                               max_doublings=4)
    dx = np.array([L["dx"] for L in info["levels"][1:]], float)
    dl = np.array([L["dlogf"] for L in info["levels"][1:]], float)
    dm = np.array([L["dlogf_max"] for L in info["levels"][1:]], float)
    rx, rl, rm = dx[:-1] / dx[1:], dl[:-1] / dl[1:], dm[:-1] / dm[1:]
    assert np.all((rx > 12) & (rx < 20)), rx
    assert np.all((rl > 12) & (rl < 20)), rl
    assert np.all((rm > 12) & (rm < 20)), rm


def test_agrees_with_dop853():
    """With the default tolerances the RK4 reference matches DOP853 (rtol 1e-12) to within
    a small multiple of its own reported step-doubling difference."""
    v, z = Nonlinear(), z0()
    x, lf, info = reference_rk4(v, z)
    assert info["converged"]
    x, lf = as_xl((x, lf), z.shape)
    xr, lr = as_xl(sampler.reference(v, z, rtol=1e-12, atol=1e-13), z.shape)
    assert np.abs(x - xr).max() <= max(2 * info["err_x"], 1e-8)
    assert np.abs(lf - lr).max() <= max(2 * info["err_logf_max"], 1e-8)
    assert np.abs(lf - lr).mean() <= max(2 * info["err_logf"], 1e-8)
    assert info["err_x"] <= config.REFERENCE_RK4["tol_x"]
    assert info["err_logf"] <= config.REFERENCE_RK4["tol_logf"]
    assert int(info["nstep"]) % config.NSTEP == 0


def test_stops_after_one_doubling_with_loose_tol():
    v, z, n0 = Nonlinear(), z0(), 12
    x, lf, info = reference_rk4(v, z, t_end=0.99, n0=n0, tol_x=1e3, tol_logf=1e3,
                                max_doublings=5)
    assert info["converged"]
    assert int(info["nstep"]) == 2 * n0 and len(info["levels"]) == 2
    xs, ls = fixed(v, z, 2 * n0, 0.99)
    np.testing.assert_allclose(as_xl((x, lf), z.shape)[0], xs, rtol=0, atol=1e-14)
    assert info["err_x"] == info["levels"][-1]["dx"]
    assert info["err_logf"] == info["levels"][-1]["dlogf"]
    assert info["err_logf_max"] == info["levels"][-1]["dlogf_max"]


def test_tol_zero_runs_all_doublings():
    v, z, n0, md = Nonlinear(), z0(), 12, 3
    x, lf, info = reference_rk4(v, z, t_end=0.99, n0=n0, tol_x=0.0, tol_logf=0.0,
                                max_doublings=md)
    assert not info["converged"]
    assert int(info["nstep"]) == n0 * 2**md
    assert len(info["levels"]) == md + 1
    assert info["err_x"] == info["levels"][-1]["dx"]
    assert info["err_logf"] == info["levels"][-1]["dlogf"]
    assert info["err_logf_max"] == info["levels"][-1]["dlogf_max"]
    # classical RK4: 4 right-hand-side evaluations per step, every level from the start
    assert int(info["nfev"]) == 4 * sum(int(L["nstep"]) for L in info["levels"])


def test_stopping_needs_both_tolerances():
    """The rule is a conjunction: a loose tol_x alone or a loose tol_logf alone does not stop
    the doubling when the other tolerance is 0."""
    v, z, n0, md = Nonlinear(), z0(), 12, 2
    for tx, tl in [(1e3, 0.0), (0.0, 1e3)]:
        _, _, info = reference_rk4(v, z, t_end=0.99, n0=n0, tol_x=tx, tol_logf=tl,
                                   max_doublings=md)
        assert not info["converged"], (tx, tl)
        assert int(info["nstep"]) == n0 * 2**md


def test_stops_at_first_level_meeting_tol():
    """tol between two successive differences: stop at the first level below it."""
    v, z, n0 = Nonlinear(), z0(), 12
    _, _, full = reference_rk4(v, z, t_end=0.99, n0=n0, tol_x=0.0, tol_logf=0.0,
                               max_doublings=4)
    dx = [L["dx"] for L in full["levels"]]
    dl = [L["dlogf"] for L in full["levels"]]
    k = 2                                    # stop at level 2: tol between dx[2] and dx[1]
    tx = np.sqrt(dx[1] * dx[2])
    tl = max(dl[1:]) * 10                    # logf never binding
    _, _, info = reference_rk4(v, z, t_end=0.99, n0=n0, tol_x=tx, tol_logf=tl,
                               max_doublings=4)
    assert dx[1] > tx >= dx[2]
    assert info["converged"] and int(info["nstep"]) == n0 * 2**k
    assert len(info["levels"]) == k + 1


def test_logf_rule_uses_the_mean():
    """tol_logf strictly between the mean and the max of |dlogf| at level 1 (tol_x loose):
    the mean rule stops there; a max rule would not."""
    v, z, n0 = Nonlinear(), z0(), 12
    _, _, full = reference_rk4(v, z, t_end=0.99, n0=n0, tol_x=0.0, tol_logf=0.0,
                               max_doublings=1)
    lv = full["levels"][1]
    assert lv["dlogf"] < lv["dlogf_max"]
    tl = np.sqrt(lv["dlogf"] * lv["dlogf_max"])
    _, _, info = reference_rk4(v, z, t_end=0.99, n0=n0, tol_x=1e3, tol_logf=tl,
                               max_doublings=3)
    assert info["converged"] and int(info["nstep"]) == 2 * n0
    assert info["err_logf"] <= tl < info["err_logf_max"]


@pytest.fixture(scope="module")
def tiny_empirical():
    lat = latent(1, 5.0)
    Y = np.array([-0.6, 0.1, 0.7])[:, None]
    z = lat.quantile_midpoints(30)
    te = 1.0 - 1e-4
    v = EmpiricalFlow(Y, lat)
    x, lf, info = reference_rk4(v, z, t_end=te)
    return Y, lat, z, te, np.asarray(x, float).reshape(-1, 1), \
        np.asarray(lf, float).reshape(-1), info


def test_empirical_field_converges(tiny_empirical):
    _, _, _, _, x, lf, info = tiny_empirical
    assert info["converged"]
    assert int(info["nstep"]) <= config.NSTEP * 2 ** config.REFERENCE_RK4["max_doublings"]
    assert info["err_x"] <= 1e-6 and info["err_logf"] <= 1e-4
    assert info["err_logf"] <= info["err_logf_max"] < np.inf
    assert np.all(np.isfinite(lf))
    assert np.all(np.abs(x) <= 5.0)


def test_empirical_field_transports_the_latent(tiny_empirical):
    """The exact flow carries K to rho_N(., t_end): every endpoint lies in a box
    t x_i + sigma Omega, the 1D flow map is monotone, and the endpoint of the latent
    quantile q = (k - 1/2)/30 has stopped-law CDF q, so each of the 3 equal-weight
    components receives 10 points. log f is the stopped-law log density there (the
    omitted delta terms on box faces weigh about phi(5) relative)."""
    Y, lat, z, te, x, lf, info = tiny_empirical
    sig = 1.0 - te
    u = np.abs(x - te * Y.T)                      # (30, 3)
    assert np.all(u.min(1) <= sig * lat.L + 10 * info["err_x"])
    assert np.all(np.diff(x[:, 0]) > 0)
    assert np.bincount(np.argmin(u, 1), minlength=3).tolist() == [10, 10, 10]
    law = StoppedLaw(Y, te, lat)
    q = (np.arange(30) + 0.5) / 30
    assert np.abs(np.asarray(law.cdf(x), float).ravel() - q).max() <= 1e-6
    assert np.abs(lf - np.asarray(law.logpdf(x), float).ravel()).max() <= 1e-3


@pytest.mark.parametrize("field", ["nonlinear", "empirical"])
def test_2d_shapes(field):
    if field == "nonlinear":
        v = Nonlinear(D=2)
    else:
        lat = latent(2, 4.5)
        v = EmpiricalFlow(np.array([[0.5, -0.3], [-0.4, 0.6], [0.1, 0.1]]), lat)
    m = 17
    z = v.latent.sample(m, rng=3)
    x, lf, info = reference_rk4(v, z, t_end=0.99, n0=12, max_doublings=3)
    assert np.shape(x) == (m, 2)
    assert np.shape(lf) == (m,)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(lf))
    assert set(info) >= {"levels", "nstep", "converged", "err_x", "err_logf",
                         "err_logf_max", "nfev"}
