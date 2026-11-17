"""selection.py: rules for lambda and the grid search (main.tex eq:oracle, sec:cv).

Interface the tests assume:
- ``LambdaSearch((lo, hi), include_zero=True)(rule)``: the candidates are the factors
  10^j, j = lo..hi-1, relative to the rule's lam_s, see ``conftest.run_search``; the returned
  ``Selection`` has ``lam`` (absolute), ``score``, ``table`` (k, 2) of (lam, score) sorted
  by lam, ``at_limit`` (best at the largest candidate), ``nsolves``;
- lam = 0 in a selection may be reported as 0 or as the floor lam_min;
- a rule has ``score(lam)``, ``limit_score()`` and ``ne`` (the fit equations);
- ``Oracle.score`` is the mean over x_ref of |v_lam - v_ref|^2;
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import conftest as R
from conftest import gl, latent, make_basis, normal_equations, rule_lam_s, run_search

config = pytest.importorskip("smoothfm.config")
sel = pytest.importorskip("smoothfm.selection")
regs = pytest.importorskip("smoothfm.regularizers")

M_, L_, T_ = 24, 5.0, 0.8


def H2():
    r = regs.H2
    return r() if isinstance(r, type) else r


def samples(N, seed=0):
    tr = R.TrimodalRef()
    return tr.sample(N, np.random.default_rng(seed)) / 0.21      # production frame s = 0.21


def ne_of(xs, t=T_):
    return normal_equations(make_basis(1, M_, L_), latent(1, L_), xs, t, H2())


def coef(ne, lam):
    return np.asarray(ne.solve(lam), float).reshape(M_ + 3, -1)


def vfun(C):
    return lambda x: R.design1d(M_, L_, np.asarray(x).reshape(-1)) @ C[:, 0]


def wdist(xs, t, f):
    """int (f - v_N)^2 rho_N dx for the truncated empirical field of xs (1D, quadrature)."""
    sig = 1 - t
    br = np.concatenate([t * xs[:, 0] - sig * L_, t * xs[:, 0] + sig * L_,
                         np.linspace(-L_, L_, M_ + 1)])
    x, w = gl(-L_, L_, hmax=sig / 4, npts=10, breaks=br)
    rho, _, _, v = R.empirical_ref(xs, x[:, None], t, L_)
    v = np.where(rho > 0, v[:, 0], 0.0)
    return float(np.sum(w * rho * (f(x) - v) ** 2))


GRID = 10.0 ** np.arange(-8, 5)


# ---------------------------------------------------------------------------
# Oracle and hold-out on planted problems
# ---------------------------------------------------------------------------

def test_oracle_selects_planted_lambda():
    xs = samples(30)
    ne = ne_of(xs)
    lam_star = 1e-2 * ne.lam_s
    xr = np.linspace(-4.0, 4.0, 400)[:, None]
    vr = R.design1d(M_, L_, xr[:, 0]) @ coef(ne, lam_star)
    rule = sel.Oracle(ne, xr, vr)
    for lam in (1e-4 * ne.lam_s, ne.lam_s):
        e = np.mean(np.sum((R.design1d(M_, L_, xr[:, 0]) @ coef(ne, lam) - vr) ** 2, axis=1))
        assert float(rule.score(lam)) == pytest.approx(e, rel=1e-10)
    s = run_search(GRID, rule)
    assert s.lam / ne.lam_s == pytest.approx(1e-2, rel=1e-12)
    assert float(s.score) <= 1e-20 * np.mean(vr ** 2)
    assert not s.at_limit


def test_holdout_on_training_data_selects_zero():
    xs = samples(30)
    ne = ne_of(xs)
    rule = sel.HoldOut(ne, ne_of(xs))
    s = run_search(GRID, rule)
    assert s.lam <= ne.lam_min * (1 + 1e-12)


def test_holdout_is_weighted_distance_up_to_constant():
    """score(lam) = int |v_lam - v_N^val|^2 rho_N^val dx + const (eq:dist-vN), slope 1."""
    xs = samples(40, seed=3)
    fit, val = xs[:32], xs[32:]
    ne_f, ne_v = ne_of(fit), ne_of(val)
    rule = sel.HoldOut(ne_f, ne_v)
    lams = ne_f.lam_s * 10.0 ** np.array([-4, -2, -1, 0, 1])
    sc = np.array([float(rule.score(l)) for l in lams])
    d = np.array([wdist(val, T_, vfun(coef(ne_f, l))) for l in lams])
    np.testing.assert_allclose(sc - sc[0], d - d[0], rtol=1e-8, atol=1e-10 * np.abs(d).max())


# ---------------------------------------------------------------------------
# LambdaSearch on synthetic scores
# ---------------------------------------------------------------------------

class Fake:
    """Synthetic rule with lam_s = 1; counts its solves."""

    def __init__(self, f, f0=1e6, flim=None):
        self.f, self.f0, self.flim = f, f0, flim
        self.calls = 0
        self.lam_s = 1.0
        self.ne = types.SimpleNamespace(lam_s=1.0, lam_min=1e-14, n_factorizations=0,
                                        n_solves=0)
        self.ne_fit = self.ne
        self.seen = {}

    def score(self, lam):
        self.calls += 1
        self.ne.n_factorizations += 1
        self.ne.n_solves += 1
        v = self.f0 if lam <= 1e-14 else float(self.f(lam))
        self.seen[float(lam)] = v
        return v

    def limit_score(self):
        return self.flim if self.flim is not None else self.f(1e300)


def quad_log(c):
    return lambda lam: (np.log10(lam) - c) ** 2


def test_search_half_decade_refinement():
    cands = 10.0 ** np.arange(-3, 7)
    rule = Fake(quad_log(2.3))
    s = run_search(cands, rule)
    assert s.lam == pytest.approx(10 ** 2.5, rel=1e-12)
    assert float(s.score) == pytest.approx(0.04, rel=1e-10)
    assert s.nsolves == rule.calls == rule.ne.n_factorizations
    assert rule.calls == len(cands) + 1 + 2
    tab = np.asarray(s.table, float)
    assert tab.ndim == 2 and tab.shape[1] == 2
    assert np.all(np.diff(tab[:, 0]) > 0)
    for lam, v in tab:
        key = min(rule.seen, key=lambda k: abs(k - lam) / max(lam, 1e-300))
        assert v == pytest.approx(rule.seen[key], rel=1e-12)
    assert np.any(np.isclose(tab[:, 0], 10 ** 1.5)) and np.any(np.isclose(tab[:, 0], 10 ** 2.5))
    assert not s.at_limit


def test_search_ties_to_first_candidate():
    cands = 10.0 ** np.arange(-2, 3)
    s = run_search(cands, Fake(lambda lam: 1.0), include_zero=False)
    assert s.lam == pytest.approx(cands[0], rel=1e-15)


def test_search_zero_and_at_limit():
    cands = 10.0 ** np.arange(-2, 3)
    s = run_search(cands, Fake(lambda lam: 1.0 + lam, f0=0.5))
    assert s.lam <= 1e-14
    s2 = run_search(cands, Fake(lambda lam: 1.0 / lam))
    assert s2.at_limit
    assert s2.lam >= cands[-1]
    s3 = run_search(cands, Fake(quad_log(0.2)))
    assert not s3.at_limit
