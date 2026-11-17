"""Stability runs over the positive candidates only (S1-S3, S5 below).

Mathematics under test.
- Hold-out selection over C_+ = {10^j lam_s(t)} (plus the half-decade refinement) never
  returns lam = 0, even where lam = 0 scores best (S1).
- Since C_+ is a subset of the production set C = {0} u C_+ and contains the production
  minimizer whenever that is positive, the positive-set minimizer equals the production
  minimizer at every t_k where the latter is positive, and differs from it only where the
  production minimizer is 0. Hence the fixed lam(t) of X (stored lam with each 0 replaced
  by the positive-set minimizer on X) equals the re-selected positive lam(t) on X, at every
  t_k (S1, "test that both agree").
- A set whose stored lam(t) has no zero is reproduced exactly: same lam, same coefficients,
  same tv_pos (S2, hard check).
- lam_changed counts t_k > 0 at which the candidate index j = log10(lam / lam_s) (in
  half-decade steps, the refinement) differs between the perturbed and the base flow; a
  change of lam_s alone (frame change, data-dependent scale) changes no index (S3).
- Shared defaults are unchanged: the production search keeps lam = 0 (S1, last bullet).

Cases (found by probing, 49 production times, m = 64 instead of the production m = 512 so
that each fit takes about 0.5 s): inhom N = 100 set 1 selects lam = 0 at 8 of 49 times
for h2 and for l2; trimodal N = 20 set 0 selects no zero for h2.

Assumed interface (stability.py; one adapter each, see the helpers below):
- ``stability.base_task(("base", target, n, s, flow, lam_stored, R))`` with ``R =
  stability.Run(P, m, grid)``, P exposing times, m_lat, nstep, returns a dict with ``lam``
  (K,), ``lam_s`` (K,), ``coef``, ``tv_target`` and a boolean ``lam_replaced`` (K,) (any key
  containing "replaced"); for a fixed flow ``lam_stored`` is the production lam(t), which
  may contain 0.
- ``stability.pert_task`` keeps its job tuple; ``base`` may be the full base_task dict.
- ``stability.lam_changed(lam, lam_s, lam_base, lam_s_base) -> int``.
- ``stability.report_log(run, save, targets, ns_of, flows, dim, same_m)``, with the npz key
  ``<d>/<flow>/base_lam_replaced`` (n, S, K) bool.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
import types

import numpy as np
import pytest

from release_helpers import COMPUTE, make_n

if str(COMPUTE) not in sys.path:
    sys.path.insert(0, str(COMPUTE))

import common as C  # noqa: E402
import stability as S  # noqa: E402

from smoothfm import HoldOut, LambdaSearch, NormalEquations, config  # noqa: E402

M = 64                                  # spline cells of the fast cases
TIMES = np.asarray(config.TIMES, float)
ZERO_CASE = ("inhom", 100, 1)           # production hold-out picks lam = 0 at some t_k
CLEAN_CASE = ("trimodal", 20, 0)        # no zero for h2


def run_spec(m=M, times=TIMES, m_lat=256, nstep=24):
    P = types.SimpleNamespace(times=np.asarray(times, float), m=m, m_lat=m_lat, nstep=nstep,
                              dim=1, sets=(0,), ns=(100,), m_oracle=m_lat, smoke=True)
    return S.Run(P, m, None)


def production_fit(target, n, s, reg, include_zero, m=M, perturbed=None):
    """Hold-out lam(t) on X (or on the given frame points), production split, from the
    library alone (not from stability.py)."""
    st = C.setting(target)
    _, ys = st.training_set(n, s)
    if perturbed is not None:
        ys = perturbed
    yt, yv = st.split(ys, s)
    v = st.velocity(ys, TIMES, m, reg)
    return v.fit(C.HoldOutRule(yt, yv), C.search(reg, 1, include_zero=include_zero))


_CACHE = {}


def prod(target, n, s, reg, include_zero):
    key = (target, n, s, reg, include_zero)
    if key not in _CACHE:
        _CACHE[key] = production_fit(target, n, s, reg, include_zero)
    return _CACHE[key]


def base(target, n, s, flow, lam_stored, R=None):
    return S.base_task(("base", target, n, s, flow, lam_stored, R or run_spec()))


def replaced_of(r):
    keys = [k for k in r if "replaced" in k]
    assert keys, f"base_task result has no '*replaced*' key (keys: {sorted(r)})"
    return np.asarray(r[keys[0]], bool)


# ===========================================================================
# S1: the selection option (library level)
# ===========================================================================

class Fake:
    """Synthetic rule, lam_s = 1; lam = 0 (and anything at the floor) scores f0."""

    def __init__(self, f, f0):
        self.f, self.f0 = f, f0
        self.ne = types.SimpleNamespace(lam_s=1.0, lam_min=1e-14, n_factorizations=0)
        self.seen = []

    def score(self, lam):
        self.ne.n_factorizations += 1
        self.seen.append(float(lam))
        return self.f0 if lam <= 1e-14 else float(self.f(lam))

    def limit_score(self):
        return float(self.f(1e300))


def test_positive_search_never_returns_zero_when_zero_scores_best():
    cands = 10.0 ** np.arange(-3, 4)
    f = lambda lam: (np.log10(lam) - 0.7) ** 2 + 1.0       # noqa: E731
    with_zero = LambdaSearch((-3, 4), include_zero=True)(Fake(f, f0=0.0))
    assert with_zero.lam <= 1e-14                          # teeth: 0 scores best
    rule = Fake(f, f0=0.0)
    s = LambdaSearch((-3, 4), include_zero=False)(rule)
    assert s.lam > 0 and min(rule.seen) >= cands.min()
    assert s.lam == pytest.approx(10 ** 0.5, rel=1e-12)    # best of grid + half decades


def test_positive_holdout_on_fit_data():
    """Hold-out with validation = fit samples scores lam -> 0 best (test_selection); over
    C_+ it returns the smallest positive candidate's neighbourhood, never the floor."""
    st = C.setting("trimodal")
    _, ys = st.training_set(30, 0)
    basis, lat = st.basis(24), st.latent
    from smoothfm import get_regularizer
    ne = NormalEquations(basis, lat, ys, 0.8, get_regularizer("h2"))
    rule = HoldOut(ne, ne.validation(ys))
    assert C.search("h2", 1)(rule).lam <= ne.lam_min * (1 + 1e-12)
    s = C.search("h2", 1, include_zero=False)(HoldOut(ne, ne.validation(ys)))
    lo = 10.0 ** config.LAM_DECADES["h2"][1][0] * ne.lam_s
    assert s.lam >= lo * 10 ** -0.5 * (1 - 1e-12) and s.lam > ne.lam_min


def test_shared_defaults_keep_zero():
    assert C.search("h2", 1).include_zero is True
    assert C.search("l2", 1).include_zero is True
    assert inspect.signature(LambdaSearch).parameters["include_zero"].default is True
    assert config.LAM_DECADES == {"h2": {1: (-8, 16), 2: (-9, 14)},
                                  "h1": {1: (-12, 8), 2: (-13, 8)},
                                  "l2": {1: (-14, 7), 2: (-14, 7)}}


def test_stability_source_explicit_option_no_target_branching():
    src = (COMPUTE / "stability.py").read_text()
    assert re.search(r"include_zero\s*=\s*False", src), \
        "stability.py should request the positive candidates explicitly (include_zero=False)"
    names = set(C.TARGETS_1D) | set(C.TARGETS_2D)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Compare):
            for c in [node.left, *node.comparators]:
                assert not (isinstance(c, ast.Constant) and c.value in names), \
                    f"string branching on a target name at line {node.lineno}"


# ===========================================================================
# S1: the flows of stability.py on X
# ===========================================================================

@pytest.mark.parametrize("reg", ["h2", "l2"])
def test_case_has_zeros(reg):
    """Teeth of the next tests: the production lam(t) of ZERO_CASE contains 0."""
    lam = prod(*ZERO_CASE, reg, True).lam
    assert np.any(lam == 0) and np.any(lam > 0)


@pytest.mark.parametrize("reg", ["h2", "l2"])
def test_positive_minimizer_equals_production_where_positive(reg):
    """C_+ subset of C containing the production minimizer when that is positive."""
    lam0, lamp = prod(*ZERO_CASE, reg, True).lam, prod(*ZERO_CASE, reg, False).lam
    assert np.all(lamp > 0)
    pos = lam0 > 0
    np.testing.assert_array_equal(lamp[pos], lam0[pos])


@pytest.mark.parametrize("reg", ["h2", "l2"])
def test_base_fixed_replaces_only_zeros(reg):
    stored = prod(*ZERO_CASE, reg, True).lam
    want = prod(*ZERO_CASE, reg, False).lam
    r = base(*ZERO_CASE, f"{reg}_fixed", stored.copy())
    lam = np.asarray(r["lam"], float)
    assert np.all(lam > 0), f"fixed lam has zeros at {np.where(lam == 0)[0].tolist()}"
    pos = stored > 0
    np.testing.assert_array_equal(lam[pos], stored[pos])
    np.testing.assert_array_equal(lam[~pos], want[~pos])
    np.testing.assert_array_equal(replaced_of(r), ~pos)


@pytest.mark.parametrize("reg", ["h2", "l2"])
def test_base_resel_positive_and_agrees_with_fixed(reg):
    stored = prod(*ZERO_CASE, reg, True).lam
    rr = base(*ZERO_CASE, f"{reg}_resel", stored.copy())
    rf = base(*ZERO_CASE, f"{reg}_fixed", stored.copy())
    lr = np.asarray(rr["lam"], float)
    assert np.all(lr > 0)
    np.testing.assert_array_equal(lr, prod(*ZERO_CASE, reg, False).lam)
    np.testing.assert_array_equal(lr, np.asarray(rf["lam"], float))      # both ways agree


def test_base_fixed_without_zero_reproduces_exactly():
    v0 = prod(*CLEAN_CASE, "h2", True)
    stored = v0.lam.copy()
    assert np.all(stored > 0)
    R = run_spec()
    r = base(*CLEAN_CASE, "h2_fixed", stored.copy(), R)
    np.testing.assert_array_equal(np.asarray(r["lam"], float), stored)
    assert not replaced_of(r).any()
    st = C.setting(CLEAN_CASE[0])
    _, ys = st.training_set(*CLEAN_CASE[1:])
    ref = st.velocity(ys, TIMES, M, "h2").set_lam(stored)
    np.testing.assert_array_equal(r["coef"], ref.coef)
    # same coefficients, same generation: tv_pos equal to rounding of one code path
    m_ref = S.measure(st, ref, CLEAN_CASE[2], R)
    assert r["tv_target"] == m_ref["tv_target"]
    # and the hold-out flow of production (with zero allowed) gives that tv as well
    m_prod = S.measure(st, v0, CLEAN_CASE[2], R)
    assert abs(r["tv_target"] - m_prod["tv_target"]) <= config.STABILITY["tol_ref"]


# ===========================================================================
# S3: lam_changed counts index changes
# ===========================================================================

def lam_changed(lam, lam_s, lam_b, lam_s_b):
    return int(S.lam_changed(np.asarray(lam, float), np.asarray(lam_s, float),
                             np.asarray(lam_b, float), np.asarray(lam_s_b, float)))


def test_lam_changed_frame_change_is_zero():
    K = 9
    rng = np.random.default_rng(0)
    j = rng.integers(-4, 6, K) + 0.5 * rng.integers(0, 2, K)   # grid and half decades
    lam_s = 10.0 ** rng.uniform(-3, 3, K)
    lam = 10.0 ** j * lam_s
    c = 1.37                                                   # lam_s rescaled
    assert lam_changed(c * lam, c * lam_s, lam, lam_s) == 0
    assert np.sum((c * lam)[1:] != lam[1:]) == K - 1           # the float count is K - 1
    # lam_s moved by a factor below a quarter decade, rel computed with rounding noise
    lam_s2 = lam_s * 10.0 ** rng.uniform(-0.2, 0.2, K)
    assert lam_changed(10.0 ** j * lam_s2 * (1 + 1e-13), lam_s2, lam, lam_s) == 0


def test_lam_changed_counts_index_steps_and_skips_t0():
    K = 7
    lam_s = np.linspace(1.0, 4.0, K)
    j = np.array([0.0, 1, 2, 3, 4, 5, 6])
    jp = j.copy()
    jp[0] += 3                    # t_0 excluded
    jp[2] += 1                    # one decade
    jp[4] -= 0.5                  # half-decade refinement
    jp[6] += 2
    lam_s_p = 1.1 * lam_s
    assert lam_changed(10.0 ** jp * lam_s_p, lam_s_p, 10.0 ** j * lam_s, lam_s) == 3


def test_stability_docstring_states_index_meaning():
    doc = S.__doc__ or ""
    i = doc.find("lam_changed")
    assert i >= 0
    assert re.search(r"index|decade", doc[i:i + 400]), \
        "the docstring should define lam_changed by the candidate index"


def test_pert_lam_changed_is_index_count():
    """A real perturbation at epsilon = 1e-6: lam_changed equals the index-change count
    computed here from the library fits on X and X'."""
    target, n, s = CLEAN_CASE
    R = run_spec()
    st = C.setting(target)
    y, ys = st.training_set(n, s)
    rb = base(target, n, s, "h2_resel", None, R)
    epsilon = 1e-6
    yp, _ = S.perturb(y, epsilon, S.xi_draw(target, n, s, 0, 1), st.box)
    ysp = st.frame.to_std(yp)
    vb = production_fit(target, n, s, "h2", False)
    vp = production_fit(target, n, s, "h2", False, perturbed=ysp)
    out = S.pert_task(("pert", target, n, s, 0, 0, "h2_resel", None, rb, epsilon, R))
    np.testing.assert_array_equal(np.asarray(out["lam"], float), vp.lam)
    jb = np.round(2 * np.log10(vb.lam / vb.lam_s))
    jp = np.round(2 * np.log10(vp.lam / vp.lam_s))
    assert int(out["lam_changed"]) == int(np.sum(jb[1:] != jp[1:]))


# ===========================================================================
# S2: the base-reproduction check of report_log
# ===========================================================================

FLOWS_1D = ["h2_fixed", "h2_resel", "l2_fixed", "l2_resel"]


def synthetic_save(Sets=2, K=5, E=2, Rd=1):
    nn = 1
    sh = (nn, Sets, E, Rd)
    rng = np.random.default_rng(1)
    save = {"epsilon": np.array([1e-4, 1e-3])[:E], "times": np.linspace(0, 0.8, K),
            "tl": np.array([1e-2]), "sets": np.arange(Sets), "draws": Rd}
    d = "trimodal"
    save[f"{d}/ns"] = np.array([100])
    save[f"{d}/nflip"] = np.zeros(sh)
    lam_ref = np.broadcast_to(1e-3 * 10.0 ** np.arange(K), (nn, Sets, K)).copy()
    for f in FLOWS_1D:
        q = f"{d}/{f}/"
        for key in ["tv", "tv_target", "tv_target_se", "mass", "lam_changed", "wall"]:
            save[q + key] = rng.uniform(0.01, 0.1, sh)
        save[q + "lam_changed"] = np.zeros(sh)
        save[q + "lam"] = np.broadcast_to(lam_ref[:, :, None, None, :], sh + (K,)).copy()
        save[q + "dnorm2"] = rng.uniform(1e-8, 1e-6, sh + (K, 3))
        for key in ["base_tv_target", "base_tv_target_se", "base_wall"]:
            save[q + key] = rng.uniform(0.05, 0.2, (nn, Sets))
        save[q + "base_tv_ref"] = save[q + "base_tv_target"].copy()
        save[q + "base_lam"], save[q + "base_lam_ref"] = lam_ref.copy(), lam_ref.copy()
        save[q + "base_lam_replaced"] = np.zeros((nn, Sets, K), bool)
        save[q + "base_norm2"] = rng.uniform(1e-3, 1e-2, (nn, Sets, K, 3))
    q = f"{d}/exact/"
    for key in ["tv", "tv_se", "tv_bound", "wall"]:
        save[q + key] = rng.uniform(0.01, 0.1, sh)
    for key in ["l2diff2", "l2diff2_se"]:
        save[q + key] = rng.uniform(1e-4, 1e-3, sh + (1,))
    for key in ["base_l2norm2", "base_l2norm2_se"]:
        save[q + key] = rng.uniform(1e-2, 1e-1, (nn, Sets, 1))
    return save


def log_of(save):
    lines = []

    class Run:
        def __call__(self, msg, *a, **k):
            lines.append(str(msg))

        def __getattr__(self, name):
            return lambda *a, **k: lines.append(" ".join(map(str, a)))

    S.report_log(Run(), save, ["trimodal"], {"trimodal": (100,)}, FLOWS_1D, 1, True)
    return "\n".join(lines)


def failed(text):
    """A failed check: a line ending in FAILED or the closing 'FAILED:' line (the header
    names the tolerance as '(FAILED above ...)')."""
    return bool(re.search(r"FAILED\s*$|^FAILED:", text, re.M))


def test_report_log_clean_sets_pass():
    assert not failed(log_of(synthetic_save()))


@pytest.mark.parametrize("flow", FLOWS_1D)
def test_report_log_lam_mismatch_without_zero_fails(flow):
    """A set whose stored lam(t) has no zero must reproduce lam (fixed flows included)."""
    save = synthetic_save()
    save[f"trimodal/{flow}/base_lam"][0, 1, 2] *= 10.0
    assert failed(log_of(save))


@pytest.mark.parametrize("flow", FLOWS_1D)
def test_report_log_tv_mismatch_without_zero_fails(flow):
    save = synthetic_save()
    save[f"trimodal/{flow}/base_tv_target"][0, 1] += 1e-2
    assert failed(log_of(save))


def test_report_log_replaced_zero_is_reported_not_failed():
    save = synthetic_save()
    for f in FLOWS_1D:
        q = f"trimodal/{f}/"
        save[q + "base_lam_ref"][0, 1, 2] = 0.0             # stored zero in set 1
        save[q + "base_lam_replaced"][0, 1, 2] = True       # base re-selected a positive lam
        save[q + "base_tv_target"][0, 1] += 1e-2            # and its tv moved
    text = log_of(save)
    assert not failed(text), text
    assert re.search(r"replac", text, re.I), "the log should state the replaced zeros"


def test_report_log_clean_set_still_checked_next_to_replaced_set():
    save = synthetic_save()
    for f in FLOWS_1D:
        q = f"trimodal/{f}/"
        save[q + "base_lam_ref"][0, 1, 2] = 0.0
        save[q + "base_lam_replaced"][0, 1, 2] = True
    save["trimodal/h2_fixed/base_tv_target"][0, 0] += 1e-2  # clean set 0 off
    assert failed(log_of(save))


# ===========================================================================
# S1, S2, S5: the smoke run
# ===========================================================================

@pytest.fixture(scope="module")
def smoke_root(tmp_path_factory):
    from test_scripts import SmokeCache
    return SmokeCache(tmp_path_factory.mktemp("results_stab"))


@pytest.mark.script
@pytest.mark.slow
def test_smoke_stability_1d_positive(smoke_root):
    r = smoke_root.ensure("stability_1d")
    assert r.returncode == 0, r.stderr[-4000:]
    p = smoke_root.dir / "stability_1d.npz"
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for tgt in ("trimodal", "inhom"):
        K = d["times"].size
        for f in FLOWS_1D:
            q = f"{tgt}/{f}/"
            bl = d[q + "base_lam"]
            assert np.all(bl[np.isfinite(bl)] > 0), q
            pl = d[q + "lam"]
            assert np.all(pl[np.isfinite(pl)] > 0), q
            rep = d[q + "base_lam_replaced"]
            assert rep.dtype == bool and rep.shape == bl.shape and rep.shape[-1] == K
            # holds at the production m only (stability_1d); at --m the re-selected flows
            # mark the zeros of the controls lam(t)
            np.testing.assert_array_equal(rep, (d[q + "base_lam_ref"] == 0)
                                          & np.isfinite(d[q + "base_lam_ref"]))
            if f.endswith("fixed"):                        # X' keeps lam of X
                np.testing.assert_array_equal(pl, np.broadcast_to(
                    bl[:, :, None, None, :], pl.shape))
            lc = d[q + "lam_changed"]
            assert np.all(lc[np.isfinite(lc)] <= K - 1)
    log = (smoke_root.dir / "stability_1d.log").read_text()
    assert not failed(log)
    assert re.search(r"positive|lam = 0 excluded|without lam = 0|excludes? (lam = )?0",
                     log, re.I), "the log header should state the positive-candidate rule"


# ===========================================================================
# Round 3: stored-lam tv check, lower-edge flag
# ===========================================================================
#
# Interface (developer, round 3): base_task returns 'lam_at_lower', 'stored_tv_target';
# Run(P, m, grid); Run.selection(reg, dim) -> PositiveHoldOut with lowest_rel and
# at_lower(lam, lam_s); npz base_tv_stored, base_lam_at_lower.

def test_report_log_stored_tv_checks_replaced_set():
    """A replaced set is checked through the flow with the stored lam(t): a matching stored
    tv passes, a mismatch fails (the replaced set no longer escapes the check)."""
    def save_with(off):
        save = synthetic_save()
        for f in FLOWS_1D:
            q = f"trimodal/{f}/"
            save[q + "base_lam_ref"][0, 1, 2] = 0.0
            save[q + "base_lam_replaced"][0, 1, 2] = True
            save[q + "base_tv_target"][0, 1] += 1e-2           # the replaced flow moved
            save[q + "base_tv_stored"] = np.full((1, 2), np.nan)
            save[q + "base_tv_stored_se"] = np.full((1, 2), np.nan)
            save[q + "base_tv_stored"][0, 1] = save[q + "base_tv_ref"][0, 1] + off
        return save
    assert not failed(log_of(save_with(0.0)))
    assert failed(log_of(save_with(1e-2)))


def test_report_log_no_covered_set_prints_na_not_zero():
    """Every set replaced and no stored-lam flow measured: the deviation is n/a, never a
    0.0 that reads as a pass."""
    save = synthetic_save()
    for f in FLOWS_1D:
        q = f"trimodal/{f}/"
        save[q + "base_lam_ref"][0, :, 2] = 0.0
        save[q + "base_lam_replaced"][0, :, 2] = True
        save[q + "base_tv_stored"] = np.full((1, 2), np.nan)
        save[q + "base_tv_stored_se"] = np.full((1, 2), np.nan)
    text = log_of(save)
    lines = [ln for ln in text.splitlines() if re.match(r"\s+trimodal \w+_\w+\s+tv ", ln)]
    assert len(lines) == len(FLOWS_1D), text
    for ln in lines:
        assert re.search(r"tv n/a", ln) and not re.search(r"tv 0\.0e\+00", ln), ln


@pytest.mark.parametrize("reg, low_exp", [("h2", -8.5), ("l2", -14.0)])
def test_lowest_effective_candidate(reg, low_exp):
    """Smallest evaluated lam / lam_s: 10^(lo - 1/2) (refinement below the lowest decade),
    clamped to the solve floor LAM_FLOOR_REL = 1e-14 (l2: 10^-14.5 solves as 10^-14)."""
    sel = S.Run(None, M, None).selection(reg, 1)
    assert np.log10(sel.lowest_rel) == pytest.approx(low_exp, abs=1e-12)
    lam_s = np.array([0.3, 7.0, 2e4])
    at = lambda e: np.asarray(sel.at_lower(10.0 ** e * lam_s, lam_s), bool)   # noqa: E731
    assert at(low_exp).all()
    assert not at(low_exp + 0.5).any()                 # one half-decade up: interior
    assert at(low_exp + 1e-13).all()                   # rounding noise of lam / lam_s


@pytest.mark.parametrize("flow", ["h2_resel", "h2_fixed"])
def test_at_lower_flag_matches_selection(flow):
    """lam_at_lower is true exactly where the selected (resel) or replaced (fixed) lam sits at
    the smallest effective candidate, computed here from the library positive fit."""
    stored = prod(*ZERO_CASE, "h2", True).lam
    vp = prod(*ZERO_CASE, "h2", False)
    r = base(*ZERO_CASE, flow, stored.copy())
    j = np.round(2 * np.log10(vp.lam / vp.lam_s)) / 2
    want = j <= -8.5
    if flow.endswith("fixed"):
        want &= stored == 0
    np.testing.assert_array_equal(np.asarray(r["lam_at_lower"], bool), want)


# ---------------------------------------------------------------- circle, smallness
# `--target circle --reg l2` -> stability_circle_l2: flows l2_fixed, l2_resel from
# twod_circle_l2, no exact jobs; every existing run keeps its name, flows and exact jobs.

@pytest.mark.parametrize("dim,reg,m,name,flows,exact", [
    (1, None, None, "stability_1d", list(config.STABILITY["flows"][1]), True),
    (1, None, 256, "stability_1d_m256", list(config.STABILITY["flows"][1]), True),
    (2, None, None, "stability_circle", list(config.STABILITY["flows"][2]), True),
    (2, "h2", None, "stability_circle", list(config.STABILITY["flows"][2]), True),
    (2, "l2", None, "stability_circle_l2", ["l2_fixed", "l2_resel"], False)])
def test_run_plan(dim, reg, m, name, flows, exact):
    assert S.run_plan(dim, reg, m) == {"name": name, "flows": flows, "exact": exact}


def test_source_names():
    assert S.source_names(1) == {"h2": "oned", "l2": "oned_l2"}
    assert S.source_names(2) == {"h2": "twod_circle"}
    assert S.source_names(2, "l2") == {"l2": "twod_circle_l2"}


def test_circle_l2_selection_is_the_positive_rule():
    sel = S.PositiveHoldOut("l2", 2)
    lo, hi = config.LAM_DECADES["l2"][2]
    assert sel.search.include_zero is False
    assert np.isclose(np.log10(sel.search.relative.min()), lo)
    # lowest decade at the solve floor: the exclusion of lam = 0 is a no-op, as in 1D
    assert sel.lowest_rel == pytest.approx(config.LAM_FLOOR_REL)


@pytest.mark.parametrize("args", [["--dim", "1", "--reg", "l2"]])
def test_reg_option_refusals(args):
    import subprocess
    r = subprocess.run([sys.executable, str(COMPUTE / "stability.py"), "--smoke", *args],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "Traceback" not in r.stderr, r.stderr[-2000:]


def test_circle_l2_makefile_and_run_list():
    root = COMPUTE.parents[1]
    (cmd,) = make_n("compute-stability_circle_l2")
    assert "stability.py --target circle --reg l2 " in cmd
    runs = (root / "experiments" / "runs.txt").read_text().split()
    i = runs.index("stability_circle_l2")
    assert runs[i + 2: i + 7] == ["twod_circle_l2", "stability", "--target", "circle", "--reg"]


@pytest.mark.script
def test_smoke_stability_circle_l2(tmp_path):
    """Smoke run on the smoke twod_circle_l2: two flows, no exact keys, reproduction checks
    pass (grid-mass warnings of the coarse smoke grid aside)."""
    import shutil

    from release_helpers import ROOT, run_script
    src = ROOT / "results" / "smoke" / "twod_circle_l2.npz"
    if not src.is_file():
        pytest.skip("no results/smoke/twod_circle_l2.npz (make smoke)")
    (tmp_path / "smoke").mkdir()
    shutil.copy(src, tmp_path / "smoke" / src.name)
    r = run_script("stability", ["--smoke", "--target", "circle", "--reg", "l2"], tmp_path,
                   timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    with np.load(tmp_path / "smoke" / "stability_circle_l2.npz", allow_pickle=False) as z:
        assert list(z["flows"]) == ["l2_fixed", "l2_resel"]
        assert not [k for k in z.files if "/exact/" in k or "/h2_" in k]
        for f in ("l2_fixed", "l2_resel"):
            assert np.isfinite(z[f"circle/{f}/tv"]).all()
            assert np.all(z[f"circle/{f}/base_lam"] > 0)
    log = (tmp_path / "smoke" / "stability_circle_l2.log").read_text()
    assert not failed(log)


# ---------------------------------------------------------------- circle TV grid spacing
# `--grid-dx DX` (2D only) replaces the spacing of STABILITY['grid_2d'] (half-width kept); the
# result name gets _dx<DX in thousandths, 3 digits> when DX differs from the production spacing.

@pytest.mark.parametrize("dx,suffix", [(None, ""), (config.STABILITY["grid_2d"][1], ""),
                                       (0.02, ""), (0.01, "_dx010"), (0.005, "_dx005"),
                                       (0.1, "_dx100"), (0.001, "_dx001")])
def test_dx_suffix(dx, suffix):
    assert S.dx_suffix(dx) == suffix


@pytest.mark.parametrize("dx", [0.0025, 0.0, -0.01, 1.0, 0.0101])
def test_dx_suffix_refuses_ambiguous_spacing(dx):
    with pytest.raises(ValueError):
        S.dx_suffix(dx)


@pytest.mark.parametrize("reg,dx,name", [
    ("l2", 0.01, "stability_circle_l2_dx010"), ("l2", 0.005, "stability_circle_l2_dx005"),
    ("l2", 0.02, "stability_circle_l2"), (None, 0.005, "stability_circle_dx005")])
def test_run_plan_grid_dx(reg, dx, name):
    plan = S.run_plan(2, reg, None, dx)
    assert plan == dict(S.run_plan(2, reg), name=name)


def test_run_plan_grid_dx_1d_refused():
    with pytest.raises(ValueError):
        S.run_plan(1, None, None, 0.01)


def test_tv_grid_2d():
    ST = config.STABILITY
    assert S.tv_grid_2d(False) == tuple(ST["grid_2d"])
    assert S.tv_grid_2d(True) == tuple(ST["smoke"]["grid_2d"])
    assert S.tv_grid_2d(False, 0.005) == (ST["grid_2d"][0], 0.005)
    assert S.tv_grid_2d(True, 0.01) == (ST["smoke"]["grid_2d"][0], 0.01)
    pts, dx = S.grid_points(S.tv_grid_2d(False, 0.005))
    assert dx == 0.005 and pts.shape == (1041**2, 2)
    assert pts.min() == pytest.approx(-2.6) and pts.max() == pytest.approx(2.6)


@pytest.mark.parametrize("extra,grid,name", [
    ([], tuple(config.STABILITY["smoke"]["grid_2d"]), "stability_circle_l2"),
    (["--grid-dx", "0.005"], (config.STABILITY["grid_2d"][0], 0.005), "stability_circle_l2_dx005")])
def test_grid_dx_reaches_the_jobs_and_log(monkeypatch, tmp_path, extra, grid, name):
    """main() up to the first pool_map: the Run shipped with every job carries the grid, the
    log header names it (sources, worked example and the jobs themselves stubbed out)."""
    class Stop(Exception):
        pass

    seen = {}

    def pool_map(fn, jobs, workers, run=None, callback=None):
        seen["jobs"] = jobs
        raise Stop

    K = len(config.SMOKE["TIMES"])
    monkeypatch.setenv("SMOOTHFM_CHOLMOD_SUPERNODAL", "1")       # C.parse sets it for smoke
    monkeypatch.setattr(S, "load_sources", lambda *a, **k: {})
    monkeypatch.setattr(S, "stored", lambda *a, **k: (np.ones(K), np.nan, np.nan, None))
    monkeypatch.setattr(S.C, "worked_example", lambda *a, **k: None)
    monkeypatch.setattr(S.C, "pool_map", pool_map)
    monkeypatch.setattr(sys, "argv", ["stability.py", "--smoke", "--target", "circle", "--reg",
                                      "l2", *extra, "--results", str(tmp_path)])
    with pytest.raises(Stop):
        S.main()
    assert seen["jobs"] and all(j[6].grid == grid for j in seen["jobs"] if j[0] == "base")
    log = (tmp_path / "smoke" / f"{name}.log").read_text()
    txt = f"grid [-{grid[0]}, {grid[0]}]^2 dx {grid[1]}, backward RK4"
    assert txt in log
    assert ("(--grid-dx)" in log) is bool(extra)


@pytest.mark.parametrize("args", [["--dim", "1", "--grid-dx", "0.01"],
                                  ["--target", "circle", "--reg", "l2", "--grid-dx", "0.0025"]])
def test_grid_dx_refusals(args):
    import subprocess
    r = subprocess.run([sys.executable, str(COMPUTE / "stability.py"), "--smoke", *args],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "Traceback" not in r.stderr, r.stderr[-2000:]
    assert "--grid-dx" in r.stderr


def test_stability_run_names_follow_run_plan():
    """Every stability line of experiments/runs.txt is named as stability.py names its result,
    and the grid refinement of stability_circle_l2 that sec:stability reads (dx 0.005) is in
    the run list and the Makefile."""
    root = COMPUTE.parents[1]
    lines = {}
    for ln in (root / "experiments" / "runs.txt").read_text().splitlines():
        parts = ln.split()
        if len(parts) >= 4 and not ln.lstrip().startswith("#") and parts[3] == "stability":
            lines[parts[0]] = ("runs.txt", parts[2], parts[4:])
    for name, (_, _, args) in lines.items():
        opt = dict(zip(args[::2], args[1::2]))
        dim = 1 if "--dim" in opt else 2
        plan = S.run_plan(dim, opt.get("--reg"), int(opt["--m"]) if "--m" in opt else None,
                          float(opt["--grid-dx"]) if "--grid-dx" in opt else None)
        assert plan["name"] == name, (name, args)
    for dx, code in (("0.005", "005"),):
        name = f"stability_circle_l2_dx{code}"
        assert lines[name][1] == "twod_circle_l2"
        assert lines[name][2] == ["--target", "circle", "--reg", "l2", "--grid-dx", dx]
        (cmd,) = make_n(f"compute-{name}")
        assert f"stability.py --target circle --reg l2 --grid-dx {dx} " in cmd
