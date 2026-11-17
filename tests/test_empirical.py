"""empirical.py: the empirical field of the truncated latent (main.tex eq:empirical,
eq:efm).

rho_N(x, t) = (1/N) sum_i K_sigma(x - t x_i), K_sigma the truncated latent scaled by
sigma = 1 - t; w_i ∝ K_sigma(x - t x_i); m_N = sum_i w_i x_i; v_N = (m_N - x)/sigma.

Interface the tests assume:
- ``EmpiricalField(samples, latent)`` methods (first existing name is used):
  weights(x, t) (P, N); mean / m(x, t) (P, D); velocity(x, t) (P, D);
  log_density / log_rho(x, t) (P,);
- ``EmpiricalFlow(samples, latent)`` with ``__call__(x, t)`` and ``div(x, t)``; where
  rho_N = 0 it returns the field of the untruncated Gaussian weights;
- ``StoppedLaw(samples, t, latent)`` with ``logpdf(x)``, ``sample(n, rng)``,
  ``cell_mass(edges)`` (1D edges -> masses of the cells), ``tv_to(other)`` (scalar or
  (estimate, se)).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special, stats
from scipy.optimize import brentq

import conftest as R
from conftest import gl, latent, pick

emp = pytest.importorskip("smoothfm.empirical")


def field(xs, L):
    return emp.EmpiricalField(np.asarray(xs, float), latent(xs.shape[1], L))


def call(obj, names, x, t):
    return np.asarray(pick(obj, *names)(x, t), float)


def data(D, N=7, seed=0):
    return np.random.default_rng(seed).uniform(-3, 3, size=(N, D))


@pytest.mark.parametrize("D, L", [(1, 5.0), (2, 4.5)])
@pytest.mark.parametrize("t", [0.3, 0.9, 0.995])
def test_field_vs_reference(D, L, t):
    xs = data(D)
    F = field(xs, L)
    rng = np.random.default_rng(1)
    x = t * xs[rng.integers(0, len(xs), 60)] + (1 - t) * rng.uniform(-L, L, size=(60, D))
    rho, w, m, v = R.empirical_ref(xs, x, t, L)
    ok = rho > 0
    assert ok.all()
    np.testing.assert_allclose(call(F, ("weights",), x, t).reshape(60, -1), w, rtol=1e-12,
                               atol=1e-300)
    np.testing.assert_allclose(call(F, ("weights",), x, t).reshape(60, -1).sum(1), 1.0,
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(call(F, ("mean", "m", "m_N", "posterior_mean"), x, t)
                               .reshape(60, D), m, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(call(F, ("velocity", "v", "v_N"), x, t).reshape(60, D), v,
                               rtol=1e-11, atol=1e-11 / (1 - t))
    np.testing.assert_allclose(call(F, ("log_density", "log_rho", "logrho"), x, t).reshape(60),
                               np.log(rho), rtol=0, atol=1e-12)


@pytest.mark.parametrize("D, L", [(1, 5.0), (2, 4.5)])
def test_single_sample_velocity(D, L):
    x1 = np.full((1, D), 0.7)
    t = 0.8
    F = field(x1, L)
    z = np.random.default_rng(2).uniform(-L, L, size=(30, D))
    x = t * x1 + (1 - t) * z
    np.testing.assert_allclose(call(F, ("velocity", "v", "v_N"), x, t).reshape(30, D),
                               (x1 - x) / (1 - t), rtol=1e-13, atol=1e-13)


def test_density_mass_and_support_1d():
    L, t = 5.0, 0.97
    xs = data(1)
    F = field(xs, L)
    sig = 1 - t
    br = np.concatenate([t * xs[:, 0] - sig * L, t * xs[:, 0] + sig * L])
    x, w = gl(-L - 1, L + 1, hmax=sig / 4, breaks=br)
    rho = np.exp(call(F, ("log_density", "log_rho", "logrho"), x[:, None], t).reshape(-1))
    assert np.sum(w * rho) == pytest.approx(1.0, abs=1e-12)
    outside = np.all(np.abs(x[:, None] - t * xs[None, :, 0]) > sig * L, axis=1)
    assert np.all(rho[outside] == 0)


def _stopped_cdf(xs, t, L, x):
    sig = 1 - t
    u = (np.asarray(x)[:, None] - t * xs[None, :, 0]) / sig
    return stats.truncnorm(-L, L).cdf(u).mean(1)


def test_stopped_law_logpdf_sample_cellmass():
    L, t = 5.0, 0.99
    xs = data(1, N=9, seed=3)
    lat = latent(1, L)
    S = emp.StoppedLaw(xs, t, lat)
    x = t * xs[[0, 3, 5]] + (1 - t) * np.array([[0.2], [-4.9], [3.0]])
    rho, _, _, _ = R.empirical_ref(xs, x, t, L)
    np.testing.assert_allclose(np.asarray(S.logpdf(x), float).reshape(-1), np.log(rho),
                               atol=1e-12)
    y = np.asarray(S.sample(40000, np.random.default_rng(4)), float).reshape(-1)
    assert stats.kstest(y, lambda q: _stopped_cdf(xs, t, L, q)).pvalue > 1e-3
    edges = np.linspace(-L, L, 201)
    cm = np.asarray(S.cell_mass(edges), float).reshape(-1)
    assert cm.sum() == pytest.approx(1.0, abs=1e-13)
    np.testing.assert_allclose(cm, np.diff(_stopped_cdf(xs, t, L, edges)), rtol=0, atol=1e-13)


def _tv_val(r):
    return float(np.asarray(r, float).reshape(-1)[0])


def test_stopped_law_tv_1d():
    """int (p - q)_+ by quadrature split at every component edge."""
    L = 5.0
    lat = latent(1, L)
    xa, xb = data(1, N=6, seed=5), data(1, N=6, seed=6)
    for t in (0.6, 0.95):
        sig = 1 - t
        Sa, Sb = emp.StoppedLaw(xa, t, lat), emp.StoppedLaw(xb, t, lat)
        br = np.concatenate([t * np.r_[xa[:, 0], xb[:, 0]] - sig * L,
                             t * np.r_[xa[:, 0], xb[:, 0]] + sig * L])
        dens = lambda xx: (R.empirical_ref(xa, np.atleast_1d(xx)[:, None], t, L)[0]
                           - R.empirical_ref(xb, np.atleast_1d(xx)[:, None], t, L)[0])
        xf = np.linspace(-L, L, 200001)
        df = dens(xf)
        roots = [brentq(lambda s: dens(s)[0], xf[k], xf[k + 1], xtol=1e-15)
                 for k in np.nonzero(np.sign(df[:-1]) * np.sign(df[1:]) < 0)[0]]
        x, w = gl(-L, L, hmax=sig / 4, breaks=np.concatenate([br, roots]))
        ref = np.sum(w * np.maximum(dens(x), 0))
        assert _tv_val(Sa.tv_to(Sb)) == pytest.approx(ref, abs=1e-10)
        assert _tv_val(Sa.tv_to(Sa)) == pytest.approx(0.0, abs=1e-12)


def test_stopped_law_tv_2d_extremes():
    L, t = 4.5, 0.99
    lat = latent(2, L)
    xa = np.array([[-2.0, -2.0], [2.0, 1.0]])
    xb = np.array([[0.0, 2.5], [-2.5, 2.0]])
    Sa, Sb = emp.StoppedLaw(xa, t, lat), emp.StoppedLaw(xb, t, lat)
    assert _tv_val(Sa.tv_to(Sb)) == pytest.approx(1.0, abs=1e-12)       # disjoint boxes
    assert _tv_val(Sa.tv_to(emp.StoppedLaw(xa, t, lat))) == pytest.approx(0.0, abs=1e-12)


def _flow(xs, L):
    return emp.EmpiricalFlow(np.asarray(xs, float), latent(xs.shape[1], L))


@pytest.mark.parametrize("D, L", [(1, 5.0), (2, 4.5)])
@pytest.mark.parametrize("t", [0.4, 0.9])
def test_flow_divergence(D, L, t):
    """div v_N = ((t / sigma^2) tr Cov_w - D)/sigma off the jump set, vs formula and FD."""
    xs = data(D, N=5, seed=7)
    Fl = _flow(xs, L)
    sig = 1 - t
    rng = np.random.default_rng(8)
    x = t * xs[rng.integers(0, 5, 20)] + sig * rng.uniform(-0.8 * L, 0.8 * L, size=(20, D))
    # stay away from the faces of every box
    far = np.all(np.abs(np.abs(x[:, None, :] - t * xs[None]) - sig * L) > 1e-3 * sig, axis=(1, 2))
    x = x[far]
    rho, w, m, v = R.empirical_ref(xs, x, t, L)
    d = xs[None] - m[:, None]
    trc = np.einsum("pn,pnd->p", w, d ** 2)
    ref = ((t / sig ** 2) * trc - D) / sig
    got = np.asarray(Fl.div(x, t), float).reshape(-1)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10 / sig)
    np.testing.assert_allclose(np.asarray(Fl(x, t), float).reshape(-1, D), v, rtol=1e-11,
                               atol=1e-11 / sig)
    h = 1e-6 * sig
    fd = np.zeros(len(x))
    for dd in range(D):
        e = np.zeros(D)
        e[dd] = h
        fd += (np.asarray(Fl(x + e, t), float).reshape(-1, D)[:, dd]
               - np.asarray(Fl(x - e, t), float).reshape(-1, D)[:, dd]) / (2 * h)
    np.testing.assert_allclose(got, fd, rtol=1e-6, atol=1e-6 / sig)


def test_flow_finite_fallback_outside_support():
    """Where rho_N = 0 the flow uses the untruncated Gaussian weights."""
    L, t = 5.0, 0.99
    xs = np.array([[-2.0], [2.0]])
    Fl = _flow(xs, L)
    x = np.array([[0.0], [1.0]])                       # between the boxes
    assert np.all(R.empirical_ref(xs, x, t, L)[0] == 0)
    v = np.asarray(Fl(x, t), float).reshape(-1)
    assert np.all(np.isfinite(v))
    sig = 1 - t
    lw = -0.5 * ((x - t * xs[:, 0][None]) / sig) ** 2
    w = np.exp(lw - special.logsumexp(lw, axis=1, keepdims=True))
    ref = ((w * xs[:, 0][None]).sum(1) - x[:, 0]) / sig
    np.testing.assert_allclose(v, ref, rtol=1e-10, atol=1e-10)
    assert np.all(np.isfinite(np.asarray(Fl.div(x, t), float)))
