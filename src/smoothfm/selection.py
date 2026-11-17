"""Selection of the regularization parameter lam at one time t (main.tex sec:cv, sec:setup;
S:sec:splinedetails, last paragraph of "Large Regularization Parameters").

Every rule scores the solution C(lam) = K(lam)^{-1} b of a NormalEquations ``ne`` (smaller
is better); the columns of C are the coefficient vectors v_d of eq:normaleq. With
fit_loss(C) = sum_d v_d^T A v_d - 2 v_d^T b_d and the latent's second moment m2,

    R(C) = fit_loss(C) + mean_i |x_i|^2 + D m2,

the empirical loss ell_t(v_h) of eq:efm (eq:dist-vN up to the constant).

The interface ``SelectionRule`` (score(lam), solution(lam)) lets the rules be swapped:
- ``Oracle``: eq:oracle, the paper's reference rule;
- ``HoldOut``: the paper's data-driven rule (sec:cv).

``LambdaSearch`` minimizes a rule over lam = 0 (the floor) and the candidates
lam = 10^j lam_s(t), j in config.LAM_DECADES, lam_s of the rule's NormalEquations, and then
over the two half-decade neighbours of the best positive candidate (sec:setup).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .latent import as_points
from .normal_equations import NormalEquations
from .solvers import default_solver

__all__ = ["Oracle", "HoldOut", "LambdaSearch"]


class SelectionRule:
    """Interface of a selection rule: ``score(lam)`` of the fit ``ne.solve(lam)`` (smaller is
    better), solutions cached per lam."""

    def __init__(self, ne: NormalEquations, solver=None):
        self.ne = ne
        self.solver = solver or default_solver()
        self._sol = {}

    def solution(self, lam) -> np.ndarray:
        lam = float(lam)
        if lam not in self._sol:
            self._sol[lam] = self.ne.solve(lam, self.solver)
        return self._sol[lam]

    def score(self, lam) -> float:
        raise NotImplementedError

    def __call__(self, lam):
        return self.score(lam)

    def score_coefficients(self, C) -> float:
        raise NotImplementedError(f"{type(self).__name__} scores lam, not coefficients")

    def limit_score(self) -> float:
        """The score of the limit lam -> inf (the fit in null(P), or v = 0 for the smallness
        regularizer); NaN if the rule cannot score coefficients."""
        try:
            return float(self.score_coefficients(self.ne.null_space_limit()))
        except NotImplementedError:
            return float("nan")


class Oracle(SelectionRule):
    """The oracle eq:oracle: e(t) = mean_j |v_lam(x_j, t) - v_pi(x_j, t)|^2 at a fixed sample
    x_j (M, D) of the population path rho(., t), with v_ref = v_pi(x_j, t) (M, D) the
    population velocity. x_j lies in Omega; the design matrix clips points outside it."""

    def __init__(self, ne, x_ref, v_ref, solver=None):
        super().__init__(ne, solver)
        D = ne.basis.dim
        self.x_ref = as_points(x_ref, D).reshape(-1, D)
        self.v_ref = np.asarray(v_ref, float).reshape(-1, D)
        self._E = ne.basis.eval(ne.basis.to_reference(self.x_ref), 0)

    def score(self, lam):
        return self.score_coefficients(self.solution(lam))

    def score_coefficients(self, C):
        return float(np.mean(np.sum((self._E @ C - self.v_ref) ** 2, axis=1)))


class HoldOut(SelectionRule):
    """The hold-out rule of sec:cv: v_d(lam) from the normal equations of the fitting part
    (ne_fit), scored by sum_d v_d^T A_val v_d - 2 v_d^T b_{d,val} of the held-out part
    (ne_val = ne_fit.validation(x_val)), i.e. by the loss eq:efm on the held-out samples up
    to a constant. The candidates are relative to lam_s of the fitting part."""

    def __init__(self, ne_fit, ne_val, solver=None):
        super().__init__(ne_fit, solver)
        self.ne_val = ne_val

    def score(self, lam):
        return self.score_coefficients(self.solution(lam))

    def score_coefficients(self, C):
        return self.ne_val.fit_loss(C)


class Selection(NamedTuple):
    """Result of LambdaSearch: lam (absolute), its score, table (k, 2) of (lam, score) sorted
    by lam, at_limit (the selection is the limit lam -> inf), nsolves (factorizations), lam_s
    of the rule's NormalEquations and the limit score."""

    lam: float
    score: float
    table: np.ndarray
    at_limit: bool
    nsolves: int
    lam_s: float
    limit_score: float

    @property
    def rel(self) -> float:
        """lam / lam_s."""
        return self.lam / self.lam_s


class LambdaSearch:
    """The search of sec:setup: minimize a rule over lam = 10^j lam_s(t), j in ``decades`` =
    (lo, hi) (hi exclusive), plus lam = 0 (the floor) if include_zero, then over the two
    half-decade neighbours of the best positive value. Ties go to the first candidate in
    evaluation order (zero, then increasing lam). at_limit: the best lam is at or above the
    top candidate and |score - limit score| <= LIMIT_RTOL |score(0) - limit score|."""

    LIMIT_RTOL = 1e-6

    def __init__(self, decades, include_zero=True):
        lo, hi = decades
        self.relative = np.sort(10.0 ** np.arange(lo, hi))
        self.include_zero = include_zero

    def candidates(self, lam_s) -> np.ndarray:
        return self.relative * float(lam_s)

    def __call__(self, rule) -> Selection:
        lam_s = float(rule.ne.lam_s)
        cand = self.candidates(lam_s)
        cache = {}
        nf0 = rule.ne.n_factorizations

        def s(lam):
            lam = float(lam)
            if lam not in cache:
                cache[lam] = float(rule.score(lam))
            return cache[lam]

        order = ([0.0] if self.include_zero else []) + list(cand)
        best = min(order, key=s)
        if best > 0:
            for f in (10**-0.5, 10**0.5):
                s(best * f)
        lam = min(cache, key=cache.get)
        score = cache[lam]
        table = np.array(sorted(cache.items()))
        lim = rule.limit_score()
        at_limit = False
        if lam >= cand.max() and np.isfinite(lim):
            j0 = s(0.0)
            span = abs(j0 - lim) if np.isfinite(j0) else np.nan
            if not np.isfinite(span):
                sc = table[np.isfinite(table[:, 1]), 1]
                span = float(sc.max() - sc.min()) if sc.size else 0.0
            at_limit = bool(abs(score - lim) <= self.LIMIT_RTOL * span)
            table = np.array(sorted(cache.items()))
        return Selection(float(lam), float(score), table, at_limit,
                         rule.ne.n_factorizations - nf0, lam_s, float(lim))
