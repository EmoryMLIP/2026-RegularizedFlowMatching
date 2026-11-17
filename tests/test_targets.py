"""targets.py: target laws and the truncated-latent oracle (main.tex sec:data, eq:velocity).

Every expected value comes from the reference laws in conftest.py, written from the
definitions of main.tex sec:data, and from brute-force quadrature of
q(y | x) ∝ pi(y) phi_D((x - t y)/sigma) 1{|x - t y|_inf <= sigma L}:
composite GL in y (1D), per-axis GL per mixture piece (eight Gaussians,
spirals as a fine GL mixture along the curve, checkerboard squares), polar GL with the
angular domain cut to the sector of the box (circle). The references agree with dense
2D tensor quadrature to 1e-12 (checked when writing the tests).

Interface the tests assume:
- constructors take no arguments;
- ``posterior(x, t, L)`` takes x (n, D) and returns m (n, D), C (n, D, D), log p_t (n,)
  (any shape with these entries); NoisyCircle may return None or NaN for log p_t;
- ``mean``/``cov`` are properties or zero-argument methods;
- ``PopulationVelocity(target, latent)`` with L = config.L[D]; ``pv(x, t)`` -> (n, D),
  ``pv.div(x, t)`` -> (n,);
- ``tangent(y)`` returns unit tangents (n, 2) at points y (n, 2), sign arbitrary;
- the TwoSpirals density in the package is checked to 1e-9 in m and log p_t like every
  other target; a curve-node rule has to be fine enough for that.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

import conftest as R
from conftest import gl, latent, posterior, posterior_ref, value

targets = pytest.importorskip("smoothfm.targets")
frame_mod = pytest.importorskip("smoothfm.frame")


# name: (factory, reference, L, production-like data box)
CASES = {
    "trimodal": (lambda: targets.TrimodalBeta(), R.TrimodalRef, 5.0, [[-1.05, 1.05]]),
    "inhom": (lambda: targets.Inhomogeneous(), R.InhomRef, 5.0, [[-1.05, 1.05]]),
    "circle": (lambda: targets.NoisyCircle(), R.CircleRef, 4.5, [[-2.61, 2.61]] * 2),
    "checkerboard": (lambda: targets.Checkerboard(), R.CheckerboardRef, 4.5, [[-4.2, 4.2]] * 2),
    "gauss8": (lambda: targets.EightGaussians(), R.EightGaussiansRef, 4.5, [[-4.23, 4.23]] * 2),
    "spirals": (lambda: targets.TwoSpirals(), R.SpiralsRef, 4.5, [[-3.33, 3.33]] * 2),
}
NAMES = list(CASES)
HAS_JTR = ["trimodal", "inhom"]
NO_JTR = ["circle", "checkerboard", "gauss8", "spirals"]
HAS_PT = [n for n in NAMES if n != "circle"]

_cache = {}


def get(name):
    if name not in _cache:
        f, ref, L, box = CASES[name]
        _cache[name] = (f(), ref(), L, np.asarray(box, float))
    return _cache[name]


def edges(D, L):
    if D == 1:
        return [(L - 0.25,), (-(L - 0.4),)]
    return [(L - 0.25, 0.3), (-(L - 0.3), -(L - 0.45))]


def check_posterior(tg, ref, x, t, L, frame=None, with_pt=True):
    m, C, lp = posterior(tg, x, t, L)
    for k in range(len(x)):
        rm, rC, rlp = posterior_ref(ref, x[k], t, L, frame)
        assert np.all(np.abs(m[k] - rm) <= 1e-9 * np.maximum(1, np.abs(rm))), \
            (t, x[k], m[k], rm)
        assert np.abs(C[k] - rC).max() <= 1e-7 * np.abs(rC).max(), (t, x[k], C[k], rC)
        if with_pt:
            assert abs(lp[k] - rlp) <= 1e-9, (t, x[k], lp[k], rlp)


# ---------------------------------------------------------------------------
# Laws: density, moments, sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_logpdf_matches_reference(name):
    tg, ref, L, box = get(name)
    rng = np.random.default_rng(0)
    D = ref.D
    y = np.concatenate([ref.sample(300, rng).reshape(-1, D),
                        rng.uniform(box[:, 0], box[:, 1], size=(100, D))])
    got = np.asarray(tg.logpdf(y), float).reshape(-1)
    exp = ref.logpdf(y)
    fin = np.isfinite(exp) & (exp > -60)        # deep tails (density < 1e-26) not compared
    np.testing.assert_allclose(got[fin], exp[fin], rtol=0, atol=1e-9)
    assert np.all((got[~np.isfinite(exp)] == -np.inf) | (np.exp(got[~np.isfinite(exp)]) == 0))


def _norm_1d(tg):
    y, w = gl(-1.0, 1.0, hmax=0.002)
    return (np.exp(np.asarray(tg.logpdf(y[:, None]), float).reshape(-1)) * w).sum()


def _tensor(lo, hi, h, breaks=()):
    ys, ws = zip(*[gl(lo[d], hi[d], hmax=h, npts=8, breaks=breaks) for d in range(2)])
    G = np.meshgrid(*ys, indexing="ij")
    W = np.meshgrid(*ws, indexing="ij")
    return np.stack([g.ravel() for g in G], 1), W[0].ravel() * W[1].ravel()


@pytest.mark.parametrize("name", ["trimodal", "inhom", "checkerboard", "gauss8", "circle"])
def test_logpdf_normalized(name):
    tg, ref, L, box = get(name)
    if ref.D == 1:
        mass = _norm_1d(tg)
    elif name == "circle":
        th, wth = gl(-np.pi, np.pi, hmax=0.02, npts=8)
        r, wr = gl(1.2, 2.8, hmax=0.05, npts=8)
        T, Rr = np.meshgrid(th, r, indexing="ij")
        Y = np.stack([(Rr * np.cos(T)).ravel(), (Rr * np.sin(T)).ravel()], 1)
        w = np.outer(wth, wr).ravel() * Rr.ravel()
        mass = (np.exp(np.asarray(tg.logpdf(Y), float).reshape(-1)) * w).sum()
    elif name == "checkerboard":
        Y, w = _tensor([-4, -4], [4, 4], 1.0, breaks=np.arange(-4, 5, 2.0))
        mass = (np.exp(np.asarray(tg.logpdf(Y), float).reshape(-1)) * w).sum()
    elif name == "gauss8":
        g = np.arange(-7.5, 7.5 + 0.025, 0.05)          # trapezoid, spectrally accurate
        GX, GY = np.meshgrid(g, g, indexing="ij")
        Y = np.column_stack([GX.ravel(), GY.ravel()])
        mass = np.exp(np.asarray(tg.logpdf(Y), float).reshape(-1)).sum() * 0.05 ** 2
    else:
        mu, S = ref.mean, ref.cov
        sd = np.sqrt(np.diag(S))
        Y, w = _tensor(mu - 12 * sd, mu + 12 * sd, 0.1)
        mass = (np.exp(np.asarray(tg.logpdf(Y), float).reshape(-1)) * w).sum()
    assert mass == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("name", NAMES)
def test_mean_cov(name):
    tg, ref, L, box = get(name)
    D = ref.D
    np.testing.assert_allclose(np.asarray(value(tg.mean), float).reshape(D), ref.mean,
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(value(tg.cov), float).reshape(D, D), ref.cov,
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_sample_matches_law(name):
    """1D: chi^2 goodness of fit on 20 equiprobable bins. 2D: chi^2 homogeneity test
    against the reference sampler on a 12 x 12 grid."""
    tg, ref, L, box = get(name)
    n = 20000
    y = np.asarray(tg.sample(n, np.random.default_rng(1)), float).reshape(n, -1)
    if ref.D == 1:
        assert np.all(np.abs(y) <= 1)
        q = ref.ppf(np.linspace(0, 1, 21)[1:-1])
        cnt = np.bincount(np.searchsorted(q, y[:, 0]), minlength=20)
        p = stats.chisquare(cnt).pvalue
    else:
        yr = ref.sample(n, np.random.default_rng(2))
        both = np.concatenate([y, yr])
        ex = [np.quantile(both[:, d], np.linspace(0, 1, 13)) for d in range(2)]
        ex = [np.concatenate([[-np.inf], e[1:-1], [np.inf]]) for e in ex]
        h1 = np.histogram2d(y[:, 0], y[:, 1], bins=ex)[0].ravel()
        h2 = np.histogram2d(yr[:, 0], yr[:, 1], bins=ex)[0].ravel()
        keep = (h1 + h2) >= 10
        p = stats.chi2_contingency(np.stack([h1[keep], h2[keep]])).pvalue
    assert p > 1e-3


# ---------------------------------------------------------------------------
# Posterior (m, C, p_t) vs brute force
# ---------------------------------------------------------------------------

def _points(name, t, n_rho, seed):
    tg, ref, L, box = get(name)
    rng = np.random.default_rng(seed)
    return R.path_points(ref, t, L, n_rho, rng, edge=edges(ref.D, L))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("t", [0.5, 0.999])
def test_posterior_brute_force_fast(name, t):
    tg, ref, L, box = get(name)
    x = _points(name, t, 1, 10)
    check_posterior(tg, ref, x, t, L, with_pt=name in HAS_PT)


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("t", [0.01, 0.5, 0.9, 0.999, 1 - 1e-4])
def test_posterior_brute_force_full(name, t):
    tg, ref, L, box = get(name)
    x = _points(name, t, 3, 20)
    check_posterior(tg, ref, x, t, L, with_pt=name in HAS_PT)


def _affine(name, L):
    tg, ref, _, box = get(name)
    fr = frame_mod.AffineFrame.from_box(box, L)
    return frame_mod.AffineTarget(tg, fr, latent(ref.D, L)), fr


def _frame_points(name, fr, t, n_rho, seed):
    tg, ref, L, box = get(name)
    c, s = np.asarray(fr.c, float).reshape(ref.D), float(fr.s)
    rng = np.random.default_rng(seed)
    z = R.trunc_normal_sample(n_rho, ref.D, L, rng)
    e = np.asarray(edges(ref.D, L), float)
    y = (ref.sample(n_rho + len(e), rng).reshape(-1, ref.D) - c) / s
    return (1 - t) * np.concatenate([z, e]) + t * y


@pytest.mark.parametrize("name", ["trimodal", "gauss8"])
def test_affine_posterior_brute_force_fast(name):
    tg, ref, L, box = get(name)
    at, fr = _affine(name, L)
    c, s = np.asarray(fr.c, float).reshape(ref.D), float(fr.s)
    for t in (0.5, 0.999):
        x = _frame_points(name, fr, t, 1, 30)
        check_posterior(at, ref, x, t, L, frame=(c, s), with_pt=True)


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("t", [0.01, 0.5, 0.9, 0.999, 1 - 1e-4])
def test_affine_posterior_brute_force_full(name, t):
    tg, ref, L, box = get(name)
    at, fr = _affine(name, L)
    c, s = np.asarray(fr.c, float).reshape(ref.D), float(fr.s)
    x = _frame_points(name, fr, t, 3, 40)
    check_posterior(at, ref, x, t, L, frame=(c, s), with_pt=name in HAS_PT)


@pytest.mark.slow
@pytest.mark.parametrize("framed", [False, True])
@pytest.mark.parametrize("t", [0.5, 0.9, 0.99])
def test_circle_covariance(t, framed):
    """NoisyCircle C (and m) vs the polar reference, 4 path points and 2 edge points, base
    and production frame; C to 1e-7 of |C|. The reference is stable to 1e-15 under a 20x
    finer angular rule (checked when writing the test)."""
    tg, ref, L, box = get("circle")
    if framed:
        at, fr = _affine("circle", L)
        c, s = np.asarray(fr.c, float).reshape(2), float(fr.s)
        x = _frame_points("circle", fr, t, 4, 50)
        check_posterior(at, ref, x, t, L, frame=(c, s), with_pt=False)
    else:
        x = _points("circle", t, 4, 60)
        check_posterior(tg, ref, x, t, L, with_pt=False)


@pytest.mark.parametrize("name, x", [("gauss8", [3.6, 0.2]), ("trimodal", [2.59])])
def test_truncation_is_active(name, x):
    """At a point whose latent sits near L the truncated m differs from the untruncated
    posterior by more than 1e-3 and matches the truncated brute force."""
    tg, ref, L, box = get(name)
    t = 0.5
    x = np.asarray(x, float)
    rm, _, _ = posterior_ref(ref, x, t, L)
    um, _, _ = posterior_ref(ref, x, t, 40.0)
    assert np.abs(rm - um).max() > 1e-3       # the reference itself sees the truncation
    m, _, _ = posterior(tg, x[None], t, L)
    assert np.abs(m[0] - um).max() > 1e-3
    np.testing.assert_allclose(m[0], rm, rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_t0_posterior_is_prior_moments(name):
    tg, ref, L, box = get(name)
    D = ref.D
    x = np.random.default_rng(3).uniform(-0.9 * L, 0.9 * L, size=(4, D))
    m, C, _ = posterior(tg, x, 0.0, L)
    for k in range(4):
        np.testing.assert_allclose(m[k], ref.mean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(C[k], ref.cov, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name, x, t", [("trimodal", [1.1], 0.999), ("inhom", [-1.2], 0.999),
                                        ("checkerboard", [0.999, 2.997], 0.999)])
def test_posterior_nan_where_density_vanishes(name, x, t):
    tg, ref, L, box = get(name)
    x = np.asarray(x, float)[None]
    assert not np.isfinite(posterior_ref(ref, x[0], t, L)[2])
    with pytest.warns(Warning):
        m, C, lp = posterior(tg, x, t, L)
    assert np.all(np.isnan(m)) and np.all(np.isnan(C))


# ---------------------------------------------------------------------------
# Jacobian trace with the face terms, continuity equation, population velocity
# ---------------------------------------------------------------------------

def _fd_trace(tg, x, t, L):
    sig = 1 - t
    # step study (spirals, t = 0.999): truncation error < 1e-10 relative up to h = 1e-2 sigma,
    # rounding error ~ eps |m| / h reaches 8e-7 at h = 1e-6 sigma (platform-dependent misses of
    # the 1e-6 bound); 1e-4 sigma gives < 4e-9
    h = 1e-4 * sig
    D = x.shape[1]
    J = np.zeros(len(x))
    for d in range(D):
        e = np.zeros(D)
        e[d] = h
        J += (posterior(tg, x + e, t, L)[0][:, d] - posterior(tg, x - e, t, L)[0][:, d]) / (2 * h)
    return J


@pytest.mark.parametrize("name", HAS_JTR)
@pytest.mark.parametrize("t", [0.5, 0.9, 0.999])
def test_jacobian_trace_fd_and_face_term(name, t):
    tg, ref, L, box = get(name)
    x = R.path_points(ref, t, L, 0, np.random.default_rng(5), edge=edges(ref.D, L))
    J = np.asarray(tg.jacobian_trace(x, t, L), float).reshape(-1)
    Jfd = _fd_trace(tg, x, t, L)
    assert np.all(np.abs(J - Jfd) <= 1e-6 * np.maximum(1, np.abs(J))), (J, Jfd)
    _, C, _ = posterior(tg, x, t, L)
    interior = t / (1 - t) ** 2 * np.trace(C, axis1=1, axis2=2)
    assert np.all(J - interior > 1e-6), (J - interior)


@pytest.mark.parametrize("name", NO_JTR)
def test_jacobian_trace_not_implemented(name):
    tg, ref, L, box = get(name)
    with pytest.raises(NotImplementedError):
        tg.jacobian_trace(np.zeros((1, 2)), 0.5, L)


@pytest.mark.parametrize("name", ["trimodal", "inhom"])
def test_continuity_equation_1d(name):
    """d_t p_t + d_x (p_t v) = 0 at interior points by central differences."""
    tg, ref, L, box = get(name)
    h = 1e-5
    for t in (0.3, 0.6, 0.9):
        x = np.array([[-0.5 * t], [0.05 * t], [0.4 * t]])

        def p_and_flux(xx, tt):
            m, _, lp = posterior(tg, xx, tt, L)
            p = np.exp(lp)
            return p, p * (m[:, 0] - xx[:, 0]) / (1 - tt)

        dpt = (p_and_flux(x, t + h)[0] - p_and_flux(x, t - h)[0]) / (2 * h)
        dfx = (p_and_flux(x + h, t)[1] - p_and_flux(x - h, t)[1]) / (2 * h)
        p0 = p_and_flux(x, t)[0]
        assert np.all(np.abs(dpt + dfx) <= 1e-6 * (np.abs(dpt) + np.abs(dfx) + p0)), (dpt, dfx)


@pytest.mark.parametrize("name", ["trimodal", "gauss8", "circle"])
def test_population_velocity(name):
    from smoothfm import config
    tg, ref, L, box = get(name)
    D = ref.D
    pv = targets.PopulationVelocity(tg, latent(D, config.L[D]))
    Lc = config.L[D]
    t = 0.8
    x = R.path_points(ref, t, Lc, 5, np.random.default_rng(6))
    m, _, _ = posterior(tg, x, t, Lc)
    v = np.asarray(pv(x, t), float).reshape(-1, D)
    np.testing.assert_allclose(v, (m - x) / (1 - t), rtol=1e-12, atol=1e-12)
    v2 = np.asarray(pv(x, t), float).reshape(-1, D)          # cached (x, t)
    np.testing.assert_array_equal(v, v2)
    if name in HAS_JTR:
        J = np.asarray(tg.jacobian_trace(x, t, Lc), float).reshape(-1)
        np.testing.assert_allclose(np.asarray(pv.div(x, t), float).reshape(-1),
                                   (J - D) / (1 - t), rtol=1e-12)


# ---------------------------------------------------------------------------
# Curve tangents
# ---------------------------------------------------------------------------

def test_spiral_tangent():
    tg = get("spirals")[0]
    u = np.linspace(0.5, 3 * np.pi - 0.1, 25)
    z = np.stack([-u * np.cos(u), u * np.sin(u)], 1) / 3
    dz = np.stack([-np.cos(u) + u * np.sin(u), np.sin(u) + u * np.cos(u)], 1)
    dz /= np.linalg.norm(dz, axis=1, keepdims=True)
    for sgn in (1.0, -1.0):
        e = np.asarray(tg.tangent(sgn * z), float).reshape(-1, 2)
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, rtol=1e-12)
        np.testing.assert_allclose(np.abs((e * dz).sum(1)), 1.0, rtol=0, atol=1e-8)


def test_circle_tangent():
    tg = get("circle")[0]
    th = np.linspace(-3, 3, 13)
    y = 2.05 * np.stack([np.cos(th), np.sin(th)], 1)
    e = np.asarray(tg.tangent(y), float).reshape(-1, 2)
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose((e * y).sum(1), 0.0, atol=1e-12)
