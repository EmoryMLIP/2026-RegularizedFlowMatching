"""velocity.py: the time-dependent spline field (main.tex sec:cv last paragraph,
sec:sampling).

Interface the tests assume:
- ``Velocity(basis, latent, samples, times, reg)``; ``set_lam(lam)`` takes absolute
  values, one per time (a scalar applies to every time); ``set_floor()`` puts every time at
  the floor; ``fit(rule_factory, search, workers)`` calls ``rule_factory(velocity, t)`` at
  each t_k and ``search(rule)``;
- after ``fit``: ``lam`` (K,), ``lam_s`` (K,), ``selections`` (K Selection records) and the
  stored coefficients of w = (1 - t) v, shape (K, n_basis, D) (first existing name);
- ``v(x, t)`` -> (n, D) for x (n, D) by refit at t with log lam linear in tau = -log(1 - t);
  ``v.div(x, t)`` -> (n,); ``lam_at(t)`` (or ``lam_of``) gives the interpolated lam;
- outside Omega the field is the C^2 continuation of the boundary cell up to one mesh
  width h beyond the face and constant (clipped to that point) further out.
The refit-reproduces-stored check uses a 1e-14 relative tolerance, not bitwise equality,
because the order of the operations that form w is not specified.
"""

from __future__ import annotations

import numpy as np
import pytest

import conftest as R
from conftest import latent, make_basis, normal_equations, pick, run_search

vel = pytest.importorskip("smoothfm.velocity")
sel = pytest.importorskip("smoothfm.selection")
regs = pytest.importorskip("smoothfm.regularizers")
frame_mod = pytest.importorskip("smoothfm.frame")
sampler = pytest.importorskip("smoothfm.sampler")

TIMES = np.array([0.0, 0.3, 0.6, 0.9, 0.99])


def H2():
    r = regs.H2
    return r() if isinstance(r, type) else r


def data(D, N=20, seed=0):
    return np.random.default_rng(seed).uniform(-3.5, 3.5, size=(N, D))


def make(D=1, m=24, times=TIMES, xs=None):
    L = 5.0 if D == 1 else 4.5
    xs = data(D) if xs is None else xs
    b = make_basis(D, m, L)
    return vel.Velocity(b, latent(D, L), xs, times, H2()), b, xs, L


def direct(b, L, xs, t, lam, x):
    D = xs.shape[1]
    ne = normal_equations(b, latent(D, L), xs, t, H2())
    C = np.asarray(ne.solve(lam), float).reshape(-1, D)
    m = int(round(C.shape[0] ** (1 / D))) - 3
    return (R.design1d(m, L, x[:, 0]) @ C if D == 1 else R.design2d(m, L, x) @ C), ne


def lam_s_at(b, L, xs, t):
    return normal_equations(b, latent(xs.shape[1], L), xs, t, H2()).lam_s


def V(v, x, t):
    return np.asarray(v(x, t), float).reshape(len(x), -1)


@pytest.mark.parametrize("D", [1, 2])
def test_set_lam_reproduces_solves_at_time_points(D):
    v, b, xs, L = make(D, m=24 if D == 1 else 8)
    lam = np.array([1e-2 * lam_s_at(b, L, xs, t) for t in TIMES])
    v.set_lam(lam)
    x = np.random.default_rng(1).uniform(-0.9 * L, 0.9 * L, size=(40, D))
    for k, t in enumerate(TIMES):
        ref, _ = direct(b, L, xs, t, lam[k], x)
        np.testing.assert_allclose(V(v, x, t), ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_log_lambda_linear_in_tau():
    v, b, xs, L = make(1)
    lam = np.array([10.0 ** (-k) * lam_s_at(b, L, xs, t) for k, t in enumerate(TIMES)])
    v.set_lam(lam)
    x = np.linspace(-4, 4, 30)[:, None]
    tau = -np.log(1 - TIMES)
    for t in (0.45, 0.75, 0.95):
        lt = np.exp(np.interp(-np.log(1 - t), tau, np.log(lam)))
        lam_at = pick(v, "lam_at", "lam_of", "interp_lam")
        assert float(lam_at(t)) == pytest.approx(lt, rel=1e-12)
        ref, _ = direct(b, L, xs, t, lt, x)
        np.testing.assert_allclose(V(v, x, t), ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())


def test_zero_lambda_means_floor_in_between():
    v, b, xs, L = make(1)
    lam = np.array([1e-2 * lam_s_at(b, L, xs, t) for t in TIMES])
    lam[3] = 0.0
    v.set_lam(lam)
    x = np.linspace(-4, 4, 30)[:, None]
    for t in (0.7, 0.95):           # between t_2 and t_3, and t_3 and t_4
        ref, ne = direct(b, L, xs, t, 0.0, x)
        np.testing.assert_allclose(V(v, x, t), ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())
    v.set_floor()
    for t in TIMES:
        ref, _ = direct(b, L, xs, t, 0.0, x)
        np.testing.assert_allclose(V(v, x, t), ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())


def held_out(xs):
    """Validation samples of the hold-out rule: every other sample, shifted so that they differ
    from the fitting samples."""
    return xs[::2] + 0.05


def holdout_rule(ne, xs):
    return sel.HoldOut(ne, ne.validation(held_out(xs)))


class HoldOutFactory:
    """rule_factory(velocity, t) -> HoldOut on the all-sample normal equations at t, scored on
    held_out(xs) (picklable)."""

    def __init__(self, basis, lat, xs):
        self.basis, self.lat, self.xs = basis, lat, xs

    def __call__(self, velocity, t):
        return holdout_rule(normal_equations(self.basis, self.lat, self.xs, t, H2()), self.xs)


def _fit(v, basis, lat, xs, decades=(-6, 3)):
    from smoothfm.selection import LambdaSearch
    v.fit(HoldOutFactory(basis, lat, xs), LambdaSearch(decades), workers=1)
    return v


def test_fit_stores_selection_and_coefficients():
    v, b, xs, L = make(1)
    _fit(v, b, latent(1, L), xs)
    lam = np.asarray(pick(v, "lam", "lams"), float).reshape(-1)
    lams = np.asarray(pick(v, "lam_s"), float).reshape(-1)
    recs = list(pick(v, "selections", "records"))
    W = np.asarray(pick(v, "coef", "coefs", "coefficients", "W", "C"), float)
    assert len(lam) == len(TIMES) == len(recs) == len(lams)
    x = np.linspace(-4, 4, 25)[:, None]
    for k, t in enumerate(TIMES):
        ne = normal_equations(b, latent(1, L), xs, t, H2())
        assert lams[k] == ne.lam_s
        s = run_search(10.0 ** np.arange(-6, 3), holdout_rule(ne, xs))
        assert recs[k].lam == s.lam == lam[k]
        Ck = np.asarray(ne.solve(lam[k]), float).reshape(-1, 1)
        np.testing.assert_allclose(W[k].reshape(-1, 1), (1 - t) * Ck, rtol=1e-14,
                                   atol=1e-14 * np.abs(Ck).max())
        np.testing.assert_allclose(V(v, x, t) * (1 - t), R.design1d(24, L, x[:, 0]) @ W[k].reshape(-1, 1),
                                   rtol=1e-12, atol=1e-12 * np.abs(W[k]).max())
        # A1: candidates sit at integer and half-integer decades of lam_s, plus zero
        tab = np.asarray(recs[k].table, float)
        pos = tab[tab[:, 0] > ne.lam_min * (1 + 1e-9), 0]
        e2 = 2 * np.log10(pos / ne.lam_s)
        np.testing.assert_allclose(e2, np.round(e2), atol=1e-9)


@pytest.mark.parametrize("D", [1, 2])
def test_divergence_fd(D):
    v, b, xs, L = make(D, m=24 if D == 1 else 8)
    v.set_lam(np.array([1e-2 * lam_s_at(b, L, xs, t) for t in TIMES]))
    x = np.random.default_rng(2).uniform(-0.9 * L, 0.9 * L, size=(20, D))
    for t in (0.3, 0.8):
        h = 1e-5
        fd = np.zeros(len(x))
        for d in range(D):
            e = np.zeros(D)
            e[d] = h
            fd += (V(v, x + e, t)[:, d] - V(v, x - e, t)[:, d]) / (2 * h)
        got = np.asarray(v.div(x, t), float).reshape(-1)
        np.testing.assert_allclose(got, fd, rtol=1e-6, atol=1e-6 * np.abs(got).max())


def test_extension_outside_omega_c2_and_affine():
    m = 24
    v, b, xs, L = make(1, m=m)
    h = 2 * L / m
    t = 0.6
    # nearly affine field (lam -> inf): the continuation stays affine
    v.set_lam(np.array([1e12 * lam_s_at(b, L, xs, tt) for tt in TIMES]))
    xi = np.array([[-L + 0.1], [L - 0.1]])
    vi = V(v, xi, t)[:, 0]
    slope = (vi[1] - vi[0]) / (xi[1, 0] - xi[0, 0])
    xo = np.array([[L + 0.3 * h], [L + 0.9 * h], [-L - 0.5 * h]])
    np.testing.assert_allclose(V(v, xo, t)[:, 0], vi[0] + slope * (xo[:, 0] - xi[0, 0]),
                               rtol=1e-8, atol=1e-8 * np.abs(vi).max())
    # generic field: value, first and second derivative continuous at the faces
    v.set_lam(np.array([1e-3 * lam_s_at(b, L, xs, tt) for tt in TIMES]))
    d = 1e-4 * h
    for face in (L, -L):
        s = np.sign(face)
        pts = face + s * d * np.array([-2, -1, 0, 1, 2])
        f = V(v, pts[:, None], t)[:, 0]
        assert abs(f[3] - f[1]) <= 1e-3 * (np.abs(f).max() + 1)
        d1_in, d1_out = (f[2] - f[1]) / (s * d), (f[3] - f[2]) / (s * d)
        assert abs(d1_in - d1_out) <= 1e-2 * (abs(d1_in) + 1)
        d2_in = (f[2] - 2 * f[1] + f[0]) / d ** 2
        d2_out = (f[4] - 2 * f[3] + f[2]) / d ** 2
        assert abs(d2_in - d2_out) <= 1e-2 * (abs(d2_in) + abs(d2_out) + 1)
    # beyond one cell the value is clipped
    far = V(v, np.array([[L + 3 * h], [-L - 5 * h]]), t)[:, 0]
    lim = V(v, np.array([[L + h], [-L - h]]), t)[:, 0]
    np.testing.assert_allclose(far, lim, rtol=1e-14, atol=1e-14 * np.abs(lim).max())


@pytest.mark.parametrize("alpha, beta", [(3.7, 0.0), (0.02, -5.0), (1.0, 11.0)])
def test_invariance_under_shift_and_scale(alpha, beta):
    """A1: rescaling data and box by alpha (and shifting by beta) leaves lam / lam_s and
    the generated samples, mapped back, unchanged to rounding."""
    L, m = 5.0, 24
    times = np.array([0.0, 0.5, 0.9, 0.99, 0.999])
    y = R.TrimodalRef().sample(15, np.random.default_rng(3))
    box = np.array([[-1.05, 1.05]])
    z = np.asarray(latent(1, L).quantile_midpoints(200), float).reshape(-1, 1)
    out = []
    for a, bt in ((1.0, 0.0), (alpha, beta)):
        fr = frame_mod.AffineFrame.from_box(a * box + bt, L)
        ys = np.asarray(fr.to_std(a * y + bt), float).reshape(-1, 1)
        bb = make_basis(1, m, L)
        v = vel.Velocity(bb, latent(1, L), ys, times, H2())
        _fit(v, bb, latent(1, L), ys)
        rel = (np.asarray(pick(v, "lam", "lams"), float)
               / np.asarray(pick(v, "lam_s"), float))
        x, _, _ = sampler.sample(v, z, nstep=24)
        xp = (np.asarray(fr.to_phys(np.asarray(x, float).reshape(-1, 1)), float) - bt) / a
        out.append((rel, xp))
    np.testing.assert_allclose(out[1][0], out[0][0], rtol=1e-12)
    np.testing.assert_allclose(out[1][1], out[0][1], rtol=0, atol=1e-10)
