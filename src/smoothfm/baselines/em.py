"""EM for a K-component full-covariance Gaussian mixture (the gauss8 EM row of tab:twod-full).
Paper: the "Gaussian mixture, EM" row of Table 3 (tab:twod-full), via experiments/compute/em8.py."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

__all__ = ["em_from", "mix_logpdf", "one_hot"]


def _log_joint(X, w, m, S):
    """log w_k + log N(x_n; m_k, S_k), (K, n)."""
    D = X.shape[1]
    L = np.linalg.cholesky(S)
    r = X[None] - m[:, None]
    u = np.linalg.solve(L[:, None], r[..., None])[..., 0]
    return (np.log(w)[:, None] - 0.5 * np.sum(u * u, -1)
            - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)[:, None] - 0.5 * D * np.log(2 * np.pi))


def mix_logpdf(z, w, mu, S):
    """log sum_k w_k N(z; mu_k, S_k)."""
    return logsumexp(_log_joint(np.asarray(z, float), w, mu, S), axis=0)


def em_from(X, R, ridge=1e-3, iters=2000, tol=1e-10):
    """EM started from responsibilities R (K, n), M-step first. Returns
    (w, m, S, mean log-likelihood, iterations)."""
    X = np.asarray(X, float)
    n, D = X.shape
    prev = -np.inf
    for it in range(iters):
        Nk = R.sum(1) + 1e-300
        w, m = Nk / n, (R @ X) / Nk[:, None]
        r = X[None] - m[:, None]
        S = np.einsum("kn,kni,knj->kij", R, r, r) / Nk[:, None, None] + ridge * np.eye(D)
        lj = _log_joint(X, w, m, S)
        ll = logsumexp(lj, axis=0)
        R = np.exp(lj - ll)
        if abs(ll.mean() - prev) < tol:
            break
        prev = ll.mean()
    return w, m, S, ll.mean(), it + 1


def one_hot(lab, K):
    """Responsibilities (K, n) of hard labels."""
    R = np.zeros((K, lab.size))
    R[lab, np.arange(lab.size)] = 1.0
    return R
