"""frame.py: AffineFrame and AffineTarget (main.tex sec:formulation; supplement
S:sec:frames).

Interface the tests assume:
- ``AffineFrame.from_box(box, L)`` takes box as an array (D, 2) of [lo, hi] per axis;
  attributes ``c`` (D,) and ``s`` (scalar).
- ``to_std(y) = (y - c)/s``, ``to_phys(y') = c + s y'`` on arrays (n, D).
- ``log_det`` is D log s up to sign (the Jacobian of one of the two maps).
- ``AffineTarget(base, frame, latent)`` exposes the Target API in frame units.
The brute-force check of AffineTarget.posterior against quadrature is in test_targets.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import latent, posterior, value

frame_mod = pytest.importorskip("smoothfm.frame")
targets = pytest.importorskip("smoothfm.targets")


def _frame(D):
    if D == 1:
        return frame_mod.AffineFrame(np.array([0.1]), 0.21)
    return frame_mod.AffineFrame(np.array([0.3, -0.2]), 1.1)


@pytest.mark.parametrize("D", [1, 2])
def test_to_std_to_phys_inverse(D):
    f = _frame(D)
    y = np.random.default_rng(0).normal(size=(50, D)) * 3
    ys = np.asarray(f.to_std(y), float).reshape(50, D)
    np.testing.assert_allclose(ys, (y - np.asarray(f.c)) / float(f.s), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np.asarray(f.to_phys(ys), float).reshape(50, D), y,
                               rtol=1e-14, atol=1e-14)
    assert abs(abs(float(value(f.log_det))) - D * abs(math.log(float(f.s)))) < 1e-14


@pytest.mark.parametrize("box, L", [(np.array([[-1.05, 1.05]]), 5.0),
                                    (np.array([[-3.0, 1.0], [0.5, 1.5]]), 4.5),
                                    (np.array([[2.0, 2.4], [-7.0, 1.0]]), 4.5)])
def test_from_box(box, L):
    f = frame_mod.AffineFrame.from_box(box, L)
    c = np.asarray(f.c, float).reshape(-1)
    s = float(f.s)
    np.testing.assert_allclose(c, box.mean(1), rtol=0, atol=1e-15)
    hw = (box[:, 1] - box[:, 0]) / 2
    assert s == pytest.approx(hw.max() / L, rel=1e-15)
    corners = np.array(np.meshgrid(*box, indexing="ij")).reshape(len(box), -1).T
    ys = np.asarray(f.to_std(corners), float).reshape(-1, len(box))
    assert np.all(np.abs(ys) <= L * (1 + 1e-15))
    wide = np.argmax(hw)
    assert np.abs(ys[:, wide]).max() == pytest.approx(L, rel=1e-15)


def test_time_change_identity():
    """(1 - t) z + t y' in the frame equals (s x' + t c)/a = (1 - tau) z + tau y in the
    base, a = s + t (1 - s), tau = t / a, y = c + s y'."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        c = rng.normal(size=2)
        s = rng.uniform(0.2, 3.0)
        t = rng.uniform(0.0, 1.0)
        z, yp = rng.normal(size=2), rng.normal(size=2)
        xp = (1 - t) * z + t * yp
        a = s + t * (1 - s)
        tau = t / a
        y = c + s * yp
        np.testing.assert_allclose((s * xp + t * c) / a, (1 - tau) * z + tau * y,
                                   rtol=1e-14, atol=1e-14)
        assert 1 - tau == pytest.approx(s * (1 - t) / a, rel=1e-13)


@pytest.mark.parametrize("D", [1, 2])
def test_affine_target_law(D):
    base = targets.TrimodalBeta() if D == 1 else targets.EightGaussians()
    f = _frame(D)
    c, s = np.asarray(f.c, float).reshape(D), float(f.s)
    at = frame_mod.AffineTarget(base, f, latent(D))
    y = np.asarray(base.sample(500, np.random.default_rng(2)), float).reshape(-1, D)
    yp = (y - c) / s
    np.testing.assert_allclose(np.asarray(at.logpdf(yp), float).reshape(-1),
                               np.asarray(base.logpdf(y), float).reshape(-1) + D * math.log(s),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(value(at.mean), float).reshape(D),
                               (np.asarray(value(base.mean), float).reshape(D) - c) / s,
                               rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(value(at.cov), float).reshape(D, D),
                               np.asarray(value(base.cov), float).reshape(D, D) / s ** 2,
                               rtol=1e-13)
    ys = np.asarray(at.sample(20000, np.random.default_rng(3)), float).reshape(-1, D)
    se = np.sqrt(np.diag(np.asarray(value(at.cov), float).reshape(D, D)) / len(ys))
    assert np.all(np.abs(ys.mean(0) - (np.asarray(value(base.mean)).reshape(D) - c) / s) < 5 * se)


@pytest.mark.parametrize("D", [1, 2])
@pytest.mark.parametrize("t", [0.3, 0.9, 0.999])
def test_affine_target_posterior_time_change(D, t):
    """m' = (m - c)/s, C' = C/s^2, p'_t = (s/a)^D p_tau at x_tau = (s x' + t c)/a."""
    base = targets.TrimodalBeta() if D == 1 else targets.EightGaussians()
    L = 5.0 if D == 1 else 4.5
    f = _frame(D)
    c, s = np.asarray(f.c, float).reshape(D), float(f.s)
    at = frame_mod.AffineTarget(base, f, latent(D, L))
    rng = np.random.default_rng(4)
    y = (np.asarray(base.sample(6, rng), float).reshape(-1, D) - c) / s
    z = rng.uniform(-0.9 * L, 0.9 * L, size=(6, D))
    xp = (1 - t) * z + t * y
    a = s + t * (1 - s)
    tau = t / a
    xt = (s * xp + t * c) / a
    m1, C1, l1 = posterior(at, xp, t, L)
    m0, C0, l0 = posterior(base, xt, tau, L)
    np.testing.assert_allclose(m1, (m0 - c) / s, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(C1, C0 / s ** 2, rtol=1e-10, atol=1e-14 * np.abs(C1).max())
    np.testing.assert_allclose(l1, l0 + D * math.log(s / a), rtol=0, atol=1e-11)
    if D == 1:
        j1 = np.asarray(at.jacobian_trace(xp, t, L), float).reshape(-1)
        j0 = np.asarray(base.jacobian_trace(xt, tau, L), float).reshape(-1)
        np.testing.assert_allclose(j1, j0 / a, rtol=1e-12)


def test_training_set_maps_and_checks_box():
    """A.3: training_set returns y' = to_std(y) and raises ValueError naming the target, n,
    seed and the largest |y_d| when a point leaves [-L, L]^D."""
    import re
    L = 4.5
    f = frame_mod.AffineFrame(np.array([0.3, -0.2]), 0.5)
    y = np.random.default_rng(7).uniform(-1.5, 1.5, size=(40, 2))
    ys = np.asarray(f.training_set(y, L, "checkerboard", 40, 3), float).reshape(40, 2)
    np.testing.assert_allclose(ys, (y - f.c) / f.s, rtol=1e-15, atol=1e-15)
    bad = y.copy()
    bad[5] = [0.3 + 0.5 * 4.7, 0.1]                     # y'_1 = 4.7 > L
    with pytest.raises(ValueError) as e:
        f.training_set(bad, L, "checkerboard", 40, 3)
    msg = str(e.value)
    assert "checkerboard" in msg and re.search(r"\b40\b", msg) and re.search(r"\b3\b", msg)
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", msg)]
    worst = np.abs((bad - f.c) / f.s).max()
    assert any(abs(v - worst) <= 1e-5 * worst for v in nums), (msg, worst)
