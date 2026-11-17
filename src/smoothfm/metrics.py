"""Error measures of a generated law (main.tex eq:tv, sec:setup).

x_1..x_M are generated points, p their density (transported along the flow, sec:sampling,
or the KDE itself), pi the target density. Notation: the argument z holds the points x_j
(not latent points) and logf holds log p(x_j).
"""

from __future__ import annotations

import numpy as np

__all__ = ["tv_pos", "tv_grid", "frac_within"]


def tv_pos(target, z, logf):
    """The estimator of eq:tv: TV(p, pi) = E_{x~p}[(1 - pi(x)/p(x))_+]
    ~ (1/M) sum_j (1 - pi(x_j)/p(x_j))_+, with x_j = z[j] (M, D) and log p(x_j) = logf[j],
    and its Monte Carlo standard error std((1 - pi/p)_+)/sqrt(M) (an upper bound for the
    stratified or quasi-random latent points of sec:setup). Returns (estimate, se). The same
    function gives tv_iid (iid latents) and the KDE errors (p the KDE density)."""
    logp = np.asarray(target.logpdf(z), float)
    with np.errstate(over="ignore", invalid="ignore"):
        r = np.exp(logp - np.asarray(logf, float))
    q = np.clip(1.0 - np.where(np.isnan(r), 0.0, r), 0.0, None)
    se = float(q.std(ddof=1) / np.sqrt(q.size)) if q.size > 1 else float("nan")
    return float(q.mean()), se


def tv_grid(p, q, dx, dim=None) -> float:
    """(dx^D / 2) sum_k |p_k - q_k| on a uniform grid (D = p.ndim unless given).
    Paper: the 2D TV between the base and the perturbed flow of stability.py (Figure 3, Table 8)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    D = p.ndim if dim is None else int(dim)
    return float(0.5 * np.sum(np.abs(p - q)) * dx**D)


def frac_within(z, X, r):
    """Fraction of the points z within Euclidean distance r (strict) of a row of X, per r
    (exact nearest distances by a k-d tree; sec:operator)."""
    from scipy.spatial import cKDTree
    z = np.asarray(z, float).reshape(len(z), -1)
    X = np.asarray(X, float).reshape(len(X), z.shape[1])
    d = cKDTree(X).query(z)[0]
    r = np.asarray(r, float)
    out = np.array([np.mean(d < ri) for ri in np.atleast_1d(r)])
    return float(out[0]) if r.ndim == 0 else out
