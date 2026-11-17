"""sampler.py: RK4 in tau with log-density transport (main.tex sec:sampling).

Known field: v(x, t) = A x + b (affine fields are in the spline space). Exact flow
x(t) = e^{A t}(x0 + A^{-1} b) - A^{-1} b, log f(t) = log K(x0) - t tr A. In tau the ODE
is dx/dtau = (1 - t) v, t = 1 - e^{-tau}.

Interface the tests assume:
- a field is any object with ``__call__(x, t)`` -> (n, D), ``div(x, t)`` -> (n,), and an
  attribute ``latent`` (the sampler takes log f(0) = latent.logpdf(z));
- ``sample(v, z, nstep=, t_end=)`` -> (x, log f, FlowSummary);
- ``reference(v, x0, rtol, atol)`` -> (x, log f, ...) at t_end = config.T_END
  (``t_end=`` keyword accepted);
- ``backward_log_density(v, latent, x, nstep=, t_end=)`` -> log f at the end points x;
- ``sample_path(v, z, t_ends=, dtau=)`` -> one (x, log f, ...) per t_end;
- FlowSummary fields: nstep, t_end, nfev, fraction of final points outside Omega,
  fraction of trajectories with a stage outside Omega, max |x|_inf (names in _summary).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from conftest import latent, pick

sampler = pytest.importorskip("smoothfm.sampler")
config = pytest.importorskip("smoothfm.config")

A0 = np.array([[0.6, 1.1], [-0.9, 0.3]])
B0 = np.array([0.4, -0.2])


class Affine:
    def __init__(self, A, b, L=4.5):
        self.A, self.b = np.asarray(A, float), np.asarray(b, float)
        self.latent = latent(len(self.b), L)
        self.D = self.dim = len(self.b)
        self.nfev = 0

    def __call__(self, x, t):
        self.nfev += 1
        return np.asarray(x, float).reshape(-1, self.D) @ self.A.T + self.b

    def div(self, x, t):
        return np.full(np.asarray(x).reshape(-1, self.D).shape[0], np.trace(self.A))

    def exact(self, z, t):
        Ai = np.linalg.solve(self.A, self.b)
        return (z + Ai) @ expm(self.A * t).T - Ai, \
            np.asarray(self.latent.logpdf(z), float).reshape(-1) - t * np.trace(self.A)


def z0(n=50, seed=0, D=2):
    return np.random.default_rng(seed).uniform(-2, 2, size=(n, D))


def run(v, z, nstep, t_end):
    x, lf, summ = sampler.sample(v, z, nstep=nstep, t_end=t_end)
    return np.asarray(x, float).reshape(z.shape), np.asarray(lf, float).reshape(-1), summ


def test_rk4_fourth_order():
    v = Affine(A0, B0)
    z = z0()
    te = 0.99
    xe, le = v.exact(z, te)
    ex, el = [], []
    for n in (96, 192, 384):
        x, lf, _ = run(v, z, n, te)
        ex.append(np.abs(x - xe).max())
        el.append(np.abs(lf - le).max())
    rx = np.array(ex[:-1]) / np.array(ex[1:])
    rl = np.array(el[:-1]) / np.array(el[1:])
    assert np.all(np.abs(rx - 16) <= 1), rx
    assert np.all(np.abs(rl - 16) <= 1), rl


def test_defaults_and_log_density():
    v = Affine(0.2 * A0, B0)
    z = z0()
    x, lf, summ = sampler.sample(v, z)
    xe, le = v.exact(z, config.T_END)
    np.testing.assert_allclose(np.asarray(x).reshape(z.shape), xe, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(lf).reshape(-1), le, rtol=0, atol=1e-7)
    assert int(summ.nstep) == config.NSTEP
    assert float(summ.t_end) == config.T_END


def test_t_end_zero_identity():
    v = Affine(A0, B0)
    z = z0()
    x, lf, _ = run(v, z, 10, 0.0)
    np.testing.assert_array_equal(x, z)
    np.testing.assert_allclose(lf, np.asarray(v.latent.logpdf(z)).reshape(-1), atol=0)


def test_dop853_reference():
    v = Affine(A0, B0)
    z = z0()
    out = sampler.reference(v, z, rtol=1e-10, atol=1e-12)
    x, lf = np.asarray(out[0], float).reshape(z.shape), np.asarray(out[1], float).reshape(-1)
    xe, le = v.exact(z, config.T_END)
    np.testing.assert_allclose(x, xe, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(lf, le, rtol=0, atol=1e-9)


def test_backward_log_density_inverts_forward():
    v = Affine(0.2 * A0, B0)
    z = z0()
    x, lf, _ = run(v, z, 96, 0.99)
    lb = np.asarray(sampler.backward_log_density(v, v.latent, x, nstep=96, t_end=0.99),
                    float).reshape(-1)
    np.testing.assert_allclose(lb, lf, rtol=0, atol=1e-10)


def _summary(s):
    return dict(nstep=int(s.nstep), t_end=float(s.t_end), nfev=int(s.nfev),
                out=float(pick(s, "frac_outside_end", "frac_outside", "outside_final")),
                any=float(pick(s, "frac_outside_any", "frac_any_outside", "outside_any")),
                xmax=float(pick(s, "max_abs", "max_x", "xmax", "max_abs_x", "max_inf")))


def test_flow_summary_counts_points_outside():
    """v = 1.5 x expands: x(t) = e^{1.5 t} z leaves Omega when |z|_inf e^{1.5 t} > L."""
    L, te = 4.5, 0.3
    v = Affine(1.5 * np.eye(2), np.zeros(2), L)
    z = np.random.default_rng(1).uniform(-L, L, size=(400, 2))
    g = math.exp(1.5 * te)
    zi = np.abs(z).max(1)
    z = z[np.abs(zi * g - L) > 0.05 * L]           # no borderline points
    x, lf, s = run(v, z, 96, te)
    S = _summary(s)
    exp_out = np.mean(np.abs(z).max(1) * g > L)
    assert 0.1 < exp_out < 0.9
    assert S["out"] == pytest.approx(exp_out, abs=1e-12)
    assert S["any"] == pytest.approx(exp_out, abs=1e-12)
    # max |x|_inf over the final points or over every RK4 stage (either reading)
    assert np.abs(x).max() * (1 - 1e-12) <= S["xmax"] <= np.abs(x).max() * (1 + 1e-6)
    assert S["nstep"] == 96 and S["t_end"] == te
    assert S["nfev"] == 4 * 96


def test_flow_summary_inside():
    v = Affine(0.05 * A0, np.zeros(2))
    z = 0.5 * z0()
    _, _, s = run(v, z, 24, 0.99)
    S = _summary(s)
    assert S["out"] == 0 and S["any"] == 0


def test_sample_path_several_t_end():
    v = Affine(0.3 * A0, B0)
    z = z0(10)
    tes = [0.9, 0.99, 1 - 1e-4]
    out = sampler.sample_path(v, z, t_ends=tes, dtau=-np.log(1e-4) / 96)
    for te, o in zip(tes, out):
        xe, le = v.exact(z, te)
        np.testing.assert_allclose(np.asarray(o[0], float).reshape(z.shape), xe, atol=1e-6)
        np.testing.assert_allclose(np.asarray(o[1], float).reshape(-1), le, atol=1e-6)
