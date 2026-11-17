"""metrics.py: TV estimators with standard errors, grid TV,
neighbour fractions (main.tex eq:tv, sec:setup, sec:operator).

Interface the tests assume:
- ``tv_pos(target, x, logf)`` -> (estimate, se); target is any object with ``logpdf(x)``
  for x (n, D); se = std((1 - pi/f)_+)/sqrt(n) (ddof 0 or 1 accepted);
- ``tv_grid(p, q, dx)`` = (dx^D / 2) sum |p - q| over arrays of any dimension D;
- ``frac_within(z, X, r)``: fraction of the points z whose nearest X is within r
  (r scalar or array).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

import conftest as R
from conftest import latent

metrics = pytest.importorskip("smoothfm.metrics")


class Normal1D:
    def __init__(self, mu=0.0, s=1.0):
        self.mu, self.s = mu, s

    def logpdf(self, x):
        return stats.norm.logpdf(np.asarray(x, float).reshape(-1), self.mu, self.s)


def tvse(target, x, lf):
    out = metrics.tv_pos(target, x, lf)
    est, se = out
    return float(est), float(se)


def test_tv_identical_is_zero():
    p = Normal1D()
    x = np.random.default_rng(0).normal(size=(1000, 1))
    est, se = tvse(p, x, p.logpdf(x))
    assert est == 0.0 and se == 0.0


@pytest.mark.parametrize("d", [0.3, 1.0, 2.5])
def test_tv_shifted_normals(d):
    p, q = Normal1D(), Normal1D(d)
    n = 200_000
    x = np.random.default_rng(1).normal(size=(n, 1))
    est, se = tvse(q, x, p.logpdf(x))
    exact = 2 * stats.norm.cdf(d / 2) - 1
    assert abs(est - exact) <= 3 * se
    g = np.maximum(1 - np.exp(q.logpdf(x) - p.logpdf(x)), 0)
    assert se == pytest.approx(g.std() / math.sqrt(n), rel=1e-3)


def test_tv_disjoint_is_one():
    p, q = Normal1D(0.0, 0.01), Normal1D(10.0, 0.01)
    x = np.random.default_rng(2).normal(0.0, 0.01, size=(500, 1))
    est, se = tvse(q, x, p.logpdf(x))
    assert est == 1.0 and se == 0.0


def test_tv_memorizing_mixture_about_one():
    tr = R.TrimodalRef()
    rng = np.random.default_rng(3)
    xi = tr.sample(100, rng)[:, 0]
    s = 1e-7
    k = rng.integers(0, 100, 20000)
    x = (xi[k] + s * rng.normal(size=20000))[:, None]
    lf = stats.norm.logpdf(x[:, :1] - xi[None, :], 0, s)
    lf = np.log(np.mean(np.exp(lf - lf.max(1, keepdims=True)), 1)) + lf.max(1)
    est, se = tvse(tr, x, lf)
    assert est > 0.99


def test_tv_standard_error_matches_spread():
    """A2: se against the spread over 200 independent latent draws of a fixed flow."""
    L = 5.0
    lat = latent(1, L)
    a, c = 0.2, 0.1
    tg = Normal1D(0.3, 1.2)
    ests, ses = [], []
    for r in range(200):
        z = R.trunc_normal_sample(2000, 1, L, np.random.default_rng(100 + r))
        x = math.exp(a) * z + c
        lf = np.asarray(lat.logpdf(z), float).reshape(-1) - a
        e, s = tvse(tg, x, lf)
        ests.append(e)
        ses.append(s)
    spread = np.std(ests, ddof=1)
    assert np.mean(ses) == pytest.approx(spread, rel=0.2)


@pytest.mark.parametrize("D", [1, 2])
def test_tv_grid(D):
    d = 0.8
    dx = 0.01
    g = np.arange(-8, 8 + dx / 2, dx)
    if D == 1:
        p, q = stats.norm.pdf(g), stats.norm.pdf(g, d)
    else:
        X, Y = np.meshgrid(g, g, indexing="ij")
        p = stats.norm.pdf(X) * stats.norm.pdf(Y)
        q = stats.norm.pdf(X, d) * stats.norm.pdf(Y)
    got = float(metrics.tv_grid(p, q, dx))
    assert got == pytest.approx(0.5 * dx ** D * np.abs(p - q).sum(), rel=1e-12)
    assert got == pytest.approx(2 * stats.norm.cdf(d / 2) - 1, abs=1e-5)


def test_frac_within():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(30, 2))
    z = rng.normal(size=(500, 2))
    d = np.sqrt(((z[:, None] - X[None]) ** 2).sum(-1)).min(1)
    rs = np.array([0.01, 0.1, 0.3, 1.0])
    got = np.asarray(metrics.frac_within(z, X, rs), float).reshape(-1)
    np.testing.assert_allclose(got, [np.mean(d <= r) for r in rs], atol=1e-15)
