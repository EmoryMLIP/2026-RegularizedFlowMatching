"""baselines/: KDEs, bandwidth rules, fourth-order KDE, Gabriel smoothing,
EM (supplement S:sec:kde).

Names are looked up in ``smoothfm.baselines`` and its submodules (first match):
- ``KDE(X, bandwidth=h)`` / ``KDE(X, cov=S)`` (S (D, D) or (n, D, D)) with ``pdf``/``logpdf``;
  ``kde_grid_1d(ys, h, xg)``, ``kde_grid_2d(X, h, xg)`` (density on xg or xg x xg);
- ``silverman(y)`` = 0.9 min(sd, IQR/1.34) n^-1/5 (sd with ddof 1), ``sheather_jones(y)``,
  ``lscv_curve(X, hs, lam=None)`` (lam: per-point bandwidth factors), ``abramson_factors(X,
  hp)``, ``aniso_lscv(X, covs)``;
- ``k4(u)`` and ``k4_radial(u)`` kernels;
  ``kde4_grid_1d(ys, h, xg)`` -> a tuple with the density on xg and the clipped mass
  int (f_h)_- dx (in either order);
  ``lscv4_curve(ys, hs)``;
- ``GabrielVelocity(samples, h, L=None, latent=None, nc=, c_gh=)`` callable g(x, t):
  G_{t sqrt h} * v_N with G the Gaussian of standard deviation t sqrt h (variance t^2 h), v_N the truncated-latent
  empirical velocity, Gaussian latent when neither L nor latent is given; samples
  in original units, inside [-1, 1];
- ``em_from(X, R, ridge=, iters=, tol=)`` -> (weights, means, covs, mean log-likelihood,
  iterations), ``one_hot(labels, K)``.
GridDensity1D/2D are not tested here.
"""

from __future__ import annotations

import importlib
import inspect
import math

import numpy as np
import pytest
from scipy import special, stats

import conftest as R
from conftest import gl, latent

B = pytest.importorskip("smoothfm.baselines")
_SUBS = []
for _n in ("kde", "bandwidth", "kde4", "grid_density", "gabriel", "em"):
    try:
        _SUBS.append(importlib.import_module(f"smoothfm.baselines.{_n}"))
    except ImportError:
        pass


def fn(*names):
    for mod in [B] + _SUBS:
        for n in names:
            if hasattr(mod, n) and not inspect.ismodule(getattr(mod, n)):
                return getattr(mod, n)
    pytest.fail(f"baselines: none of {names} found; align tests/test_baselines.py")


def arr(x):
    return np.asarray(x, float)


# ---------------------------------------------------------------------------
# Second-order KDE and bandwidth rules
# ---------------------------------------------------------------------------

def test_kde_values_and_mass_1d():
    ys = R.TrimodalRef().sample(80, np.random.default_rng(0))
    h = 0.07
    k = fn("KDE")(ys, bandwidth=h)
    x, w = gl(-2, 2, hmax=h / 4)
    p = arr(k.pdf(x[:, None])).reshape(-1)
    ref = stats.norm.pdf((x[:, None] - ys[None, :, 0]) / h).mean(1) / h
    np.testing.assert_allclose(p, ref, rtol=1e-12, atol=1e-300)
    assert np.sum(w * p) == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(arr(k.logpdf(x[:, None])).reshape(-1)[p > 1e-200],
                               np.log(p[p > 1e-200]), atol=1e-11)


def _tensor(lo, hi, h):
    """Uniform grid with the trapezoid weights: spectrally accurate for integrands that
    are Gaussians (times polynomials) much wider than h and negligible at the edges."""
    g = np.arange(lo, hi + h / 2, h)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()]), np.full(X.size, h * h)


def test_kde_mass_2d_iso_and_per_point_cov():
    X = R.CircleRef().sample(25, np.random.default_rng(1))
    G, w = _tensor(-4, 4, 0.02)
    assert np.sum(w * arr(fn("KDE")(X, bandwidth=0.2).pdf(G)).reshape(-1)) == \
        pytest.approx(1.0, abs=1e-9)
    th = np.arctan2(X[:, 1], X[:, 0])
    e = np.stack([-np.sin(th), np.cos(th)], 1)
    nrm = np.stack([np.cos(th), np.sin(th)], 1)
    S = 0.3 ** 2 * e[:, :, None] * e[:, None, :] + 0.08 ** 2 * nrm[:, :, None] * nrm[:, None, :]
    k = fn("KDE")(X, cov=S)
    p = arr(k.pdf(G)).reshape(-1)
    assert np.sum(w * p) == pytest.approx(1.0, abs=1e-8)
    ref = np.mean([stats.multivariate_normal(X[i], S[i]).pdf(G[:200]) for i in range(len(X))], 0)
    np.testing.assert_allclose(p[:200], ref, rtol=1e-11, atol=1e-300)


def test_kde_grids_match_direct():
    ys = R.TrimodalRef().sample(60, np.random.default_rng(2))
    xg = np.linspace(-1.5, 1.5, 601)
    h = 0.05
    f = arr(fn("kde_grid_1d")(ys, h, xg)).reshape(-1)
    ref = stats.norm.pdf((xg[:, None] - ys[None, :, 0]) / h).mean(1) / h
    # binned FFT evaluation: error O((dx/h)^2) = O(1e-2) relative to the peak, here 2e-3
    np.testing.assert_allclose(f, ref, rtol=0, atol=2e-3 * ref.max())
    X = R.CircleRef().sample(40, np.random.default_rng(3))
    g2 = np.linspace(-3, 3, 121)
    f2 = arr(fn("kde_grid_2d")(X, 0.2, g2))
    GX, GY = np.meshgrid(g2, g2, indexing="ij")
    ref2 = (stats.norm.pdf((GX[..., None] - X[:, 0]) / 0.2)
            * stats.norm.pdf((GY[..., None] - X[:, 1]) / 0.2)).mean(-1) / 0.04
    # 2D binning on a grid with dx/h = 0.25: 1e-2 of the peak
    np.testing.assert_allclose(f2.reshape(ref2.shape), ref2, rtol=0, atol=1e-2 * ref2.max())


def test_silverman():
    y = np.random.default_rng(4).normal(size=300) * 1.3 + 0.2
    iqr = np.subtract(*np.percentile(y, [75, 25]))
    ref = 0.9 * min(y.std(ddof=1), iqr / 1.34) * 300 ** -0.2
    assert float(fn("silverman")(y)) == pytest.approx(ref, rel=1e-12)


def test_sheather_jones_normal_case():
    """For normal data SJ approaches the AMISE bandwidth 1.0592 sigma n^-1/5."""
    n = 5000
    y = np.random.default_rng(5).normal(size=n) * 0.7
    h = float(fn("sheather_jones")(y))
    assert h == pytest.approx(1.0592 * 0.7 * n ** -0.2, rel=0.1)


def _lscv_brute(X, h_i, grid_h):
    """int f^2 (quadrature) - (2/n) sum_i f_{-i}(y_i), per-point isotropic widths h_i."""
    X = np.atleast_2d(X)
    n, D = X.shape
    lo, hi = X.min() - 7 * h_i.max(), X.max() + 7 * h_i.max()
    if D == 1:
        G, w = gl(lo, hi, hmax=grid_h)
        G = G[:, None]
    else:
        G, w = _tensor(lo, hi, h_i.min() / 5)
    K = lambda d, h: np.exp(-0.5 * (d ** 2).sum(-1) / h ** 2) / (2 * np.pi * h ** 2) ** (D / 2)
    f = np.mean(K(G[:, None, :] - X[None], h_i[None, :]), 1)
    Kij = K(X[:, None, :] - X[None], h_i[None, :])
    np.fill_diagonal(Kij, 0)
    return np.sum(w * f ** 2) - 2 / n * np.sum(Kij.sum(1) / (n - 1))


@pytest.mark.parametrize("D", [1, 2])
@pytest.mark.parametrize("adaptive", [False, True])
def test_lscv_curve_vs_numerical(D, adaptive):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(15, D)) * np.array([1.0, 0.5])[:D]
    hs = np.array([0.15, 0.3, 0.6])
    lam = np.exp(rng.uniform(-0.3, 0.3, 15)) if adaptive else None
    lam = None if lam is None else lam / np.exp(np.mean(np.log(lam)))
    got = arr(fn("lscv_curve")(X if D == 2 else X[:, 0], hs, lam)).reshape(-1)
    for k, h in enumerate(hs):
        hi = h * (np.ones(15) if lam is None else lam)
        ref = _lscv_brute(X, hi, h * 0.6 / 4 if D == 2 else h / 8)
        assert got[k] == pytest.approx(ref, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("D", [1, 2])
def test_abramson_factors(D):
    X = np.random.default_rng(7).normal(size=(50, D))
    hp = 0.4
    lam = arr(fn("abramson_factors")(X if D == 2 else X[:, 0], hp)).reshape(-1)
    assert np.exp(np.mean(np.log(lam))) == pytest.approx(1.0, rel=1e-12)
    f = arr(fn("KDE")(X, bandwidth=hp).pdf(X)).reshape(-1)
    r = lam ** 2 * f                       # lam ∝ f^-1/2
    np.testing.assert_allclose(r, r[0], rtol=1e-10)


def test_aniso_lscv():
    X = R.CircleRef().sample(12, np.random.default_rng(8))
    th = np.arctan2(X[:, 1], X[:, 0])
    e = np.stack([-np.sin(th), np.cos(th)], 1)
    nr = np.stack([np.cos(th), np.sin(th)], 1)
    S = 0.35 ** 2 * e[:, :, None] * e[:, None, :] + 0.15 ** 2 * nr[:, :, None] * nr[:, None, :]
    G, w = _tensor(-3.5, 3.5, 0.02)
    f = np.mean([stats.multivariate_normal(X[i], S[i]).pdf(G) for i in range(12)], 0)
    loo = np.array([[stats.multivariate_normal(X[j], S[j]).pdf(X[i]) if i != j else 0.0
                     for j in range(12)] for i in range(12)])
    ref = np.sum(w * f ** 2) - 2 / 12 * np.sum(loo.sum(1) / 11)
    assert float(fn("aniso_lscv")(X, S)) == pytest.approx(ref, rel=1e-7)
    iso = np.broadcast_to(0.2 ** 2 * np.eye(2), (12, 2, 2))
    assert float(fn("aniso_lscv")(X, iso)) == pytest.approx(
        float(arr(fn("lscv_curve")(X, np.array([0.2]))).reshape(-1)[0]), rel=1e-12)


# ---------------------------------------------------------------------------
# Fourth-order KDE
# ---------------------------------------------------------------------------

def test_k4_moments_1d():
    k4 = fn("k4", "kernel4", "K4")
    u, w = gl(-14, 14, hmax=0.1)
    K = arr(k4(u)).reshape(-1)
    np.testing.assert_allclose(K, 0.5 * (3 - u ** 2) * stats.norm.pdf(u), rtol=1e-14, atol=1e-300)
    assert np.sum(w * K) == pytest.approx(1.0, abs=1e-13)
    assert np.sum(w * u * K) == pytest.approx(0.0, abs=1e-13)
    assert np.sum(w * u ** 2 * K) == pytest.approx(0.0, abs=1e-12)
    assert np.sum(w * u ** 4 * K) == pytest.approx(-3.0, abs=1e-11)


def test_k4_moments_radial_2d():
    k4r = fn("k4_radial", "k4_2d", "radial_k4", "kernel4_radial")
    G, w = _tensor(-12, 12, 0.05)
    K = arr(k4r(G)).reshape(-1)
    r2 = (G ** 2).sum(1)
    np.testing.assert_allclose(K, (2 - r2 / 2) * np.exp(-r2 / 2) / (2 * np.pi), rtol=1e-13,
                               atol=1e-300)
    assert np.sum(w * K) == pytest.approx(1.0, abs=1e-12)
    assert np.sum(w * G[:, 0] ** 2 * K) == pytest.approx(0.0, abs=1e-11)
    assert np.sum(w * G[:, 0] ** 4 * K) == pytest.approx(-3.0, abs=1e-10)


def test_kde4_clip_renormalize_and_clipped_mass():
    ys = R.InhomRef().sample(100, np.random.default_rng(9))[:, 0]
    h = 0.04
    xg = np.linspace(-1.5, 1.5, 3001)
    dx = xg[1] - xg[0]
    out = fn("kde4_grid_1d")(ys, h, xg)
    parts = list(out) if isinstance(out, tuple) else [out]
    f = next(arr(p).reshape(-1) for p in parts if np.size(p) == len(xg))
    clipped = next(float(p) for p in parts if np.size(p) == 1)
    raw = (0.5 * (3 - ((xg[:, None] - ys) / h) ** 2) * stats.norm.pdf((xg[:, None] - ys) / h)
           ).mean(1) / h
    assert np.all(f >= 0)
    assert np.sum(f) * dx == pytest.approx(1.0, abs=1e-10)
    pos = np.maximum(raw, 0)
    # linear binning: error O(dx^2 / h^2) relative to max f (6.5e-5 here)
    np.testing.assert_allclose(f, pos / (pos.sum() * dx), rtol=0, atol=2e-4 * f.max())
    assert clipped == pytest.approx(np.maximum(-raw, 0).sum() * dx, rel=1e-3, abs=1e-12)
    assert clipped > 0


def test_lscv4_closed_form_vs_numerical():
    ys = np.random.default_rng(10).normal(size=20)
    hs = np.array([0.2, 0.5])
    got = arr(fn("lscv4_curve")(ys, hs)).reshape(-1)
    for k, h in enumerate(hs):
        x, w = gl(ys.min() - 12 * h, ys.max() + 12 * h, hmax=h / 8)
        K = lambda d: 0.5 * (3 - (d / h) ** 2) * stats.norm.pdf(d / h) / h
        f = K(x[:, None] - ys[None]).mean(1)
        Kij = K(ys[:, None] - ys[None])
        np.fill_diagonal(Kij, 0)
        ref = np.sum(w * f ** 2) - 2 / 20 * np.sum(Kij.sum(1) / 19)
        assert got[k] == pytest.approx(ref, rel=1e-9)


# ---------------------------------------------------------------------------
# Gabriel et al.: G_{t sqrt h} * v_N of the truncated latent (A6)
# ---------------------------------------------------------------------------

def _vN(xs, x, t, L):
    """Truncated empirical velocity; L = None gives the Gaussian latent."""
    sig = 1 - t
    u = (x[:, None] - t * xs[None, :]) / sig
    lw = -0.5 * u ** 2
    if L is not None:
        lw = np.where(np.abs(u) <= L, lw, -np.inf)
    w = np.exp(lw - special.logsumexp(lw, axis=1, keepdims=True))
    return ((w * xs[None]).sum(1) - x) / sig


def _smoothed(xs, x, t, h, L):
    sig, a = 1 - t, t * math.sqrt(h)
    out = []
    for x0 in x:
        faces = [] if L is None else list((x0 - np.r_[t * xs - sig * L, t * xs + sig * L]) / a)
        u, w = gl(-12, 12, hmax=0.01, breaks=faces)
        out.append(np.sum(w * stats.norm.pdf(u) * _vN(xs, x0 - a * u, t, L)))
    out = np.array(out)
    assert np.all(np.isfinite(out)), "test points must lie in the union of the boxes"
    return out


@pytest.fixture(scope="module")
def gab_data():
    """Trimodal samples in original units: GabrielVelocity takes data in [-1, 1] (its node
    grid spans t + 7 (1 - t) + 8 sigma), not frame units."""
    return R.TrimodalRef().sample(50, np.random.default_rng(11))[:, 0]


PHI5 = float(stats.norm.pdf(5.0))      # 1.5e-6


@pytest.mark.parametrize("t", [0.5, 0.9])
@pytest.mark.parametrize("sqrth", [0.02, 0.05])
def test_gabriel_truncated_vs_brute_force(gab_data, t, sqrth):
    """Default scheme vs brute-force G * v_N of the truncated latent. Tolerance 1e-8 + phi(L):
    A6 allows O(phi(L)); the Gauss-Hermite branch integrates across the jumps of the
    truncated m_N and stagnates near 4e-7 = 0.27 phi(5) at t = 0.9 (measured,
    independent of the number of GH nodes), while the cell branch converges (next test)."""
    if t == 0.9 and sqrth == 0.05:
        pytest.skip("smoothing window reaches gaps between the boxes, where v_N is undefined")
    L, h = 5.0, sqrth ** 2
    xs = gab_data
    G = fn("GabrielVelocity")(xs[:, None], h, latent=latent(1, L))
    x = np.linspace(-0.8, 0.8, 17) * t
    got = arr(G(x[:, None], t)).reshape(-1)
    np.testing.assert_allclose(got, _smoothed(xs, x, t, h, L), rtol=0, atol=1e-8 + PHI5)


@pytest.mark.parametrize("t, L", [(0.5, 5.0), (0.9, 5.0), (0.9, None)])
def test_gabriel_cell_branch_converges(gab_data, t, L):
    """Cell-integral branch (c_gh = 0): error vs brute force falls by 3-5x per doubling of
    the nodes per cell (second-order interpolation), from about 2e-6 at the default nc = 20."""
    h = 0.02 ** 2
    xs = gab_data
    x = np.linspace(-0.8, 0.8, 17) * t
    ref = _smoothed(xs, x, t, h, L)
    err = []
    for nc in (20, 40, 80, 160):
        G = fn("GabrielVelocity")(xs[:, None], h, L=L, nc=nc, c_gh=0.0)
        err.append(np.abs(arr(G(x[:, None], t)).reshape(-1) - ref).max())
    err = np.array(err)
    assert err[0] <= 1e-5
    r = err[:-1] / err[1:]
    assert np.all((r > 3) & (r < 5)), err


def test_gabriel_large_L_matches_gaussian_latent(gab_data):
    """A6: in the limit of large L the truncated smoothing equals the Gaussian-latent one."""
    t, h = 0.9, 0.02 ** 2
    xs = gab_data
    x = np.linspace(-0.8, 0.8, 9) * t
    ref = _smoothed(xs, x, t, h, None)
    for G in (fn("GabrielVelocity")(xs[:, None], h, latent=latent(1, 8.0)),
              fn("GabrielVelocity")(xs[:, None], h)):           # Gaussian latent
        np.testing.assert_allclose(arr(G(x[:, None], t)).reshape(-1), ref, rtol=0, atol=1e-8)


def test_gabriel_zero_bandwidth_is_vN(gab_data):
    from smoothfm.empirical import EmpiricalField
    L, t = 5.0, 0.9
    xs = gab_data
    x = np.linspace(-0.8, 0.8, 11)[:, None] * t
    G = fn("GabrielVelocity")(xs[:, None], 0.0, latent=latent(1, L))
    ref = _vN(xs, x[:, 0], t, L)
    np.testing.assert_allclose(arr(G(x, t)).reshape(-1), ref, rtol=1e-12, atol=1e-12)


def test_gabriel_reproduces_affine_field():
    """N = 1: v_N = (x_1 - x)/sigma is affine on its box; smoothing keeps it there."""
    L, t, h = 5.0, 0.5, 0.1 ** 2
    x1 = np.array([[0.4]])
    G = fn("GabrielVelocity")(x1, h, latent=latent(1, L))
    x = np.linspace(-1.0, 1.0, 9)[:, None] + t * 0.4
    np.testing.assert_allclose(arr(G(x, t)).reshape(-1), (0.4 - x[:, 0]) / (1 - t),
                               rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Baptista et al.: Tikhonov score, Gaussian ML; EM
# ---------------------------------------------------------------------------

def test_em_likelihood_non_decreasing():
    """em_from(X, R, iters=k) from the same responsibilities (nearest of 8 sample points):
    the training log-likelihood of the returned fit (computed here) does not decrease with k."""
    X = R.EightGaussiansRef().sample(400, np.random.default_rng(14))
    rng = np.random.default_rng(15)
    mus = X[rng.choice(len(X), 8, replace=False)]
    lab = np.argmin(((X[:, None] - mus[None]) ** 2).sum(-1), 1)
    em, onehot = fn("em_from"), fn("one_hot")
    ll = []
    for k in (1, 2, 3, 5, 8, 13, 21, 34):
        w, m, S, _, _ = em(X, onehot(lab, 8), ridge=1e-9, iters=k, tol=0.0)
        comp = [np.log(w[j]) + stats.multivariate_normal(m[j], S[j]).logpdf(X) for j in range(8)]
        ll.append(special.logsumexp(np.array(comp), axis=0).mean())
    ll = np.array(ll)
    assert np.all(np.diff(ll) >= -1e-10 * np.abs(ll).max()), ll
    assert ll[-1] > ll[0]
