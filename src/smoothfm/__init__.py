"""smoothfm: flow matching with explicit regularization of the velocity, the code of
"Explicit Regularization for Flow Matching" (L. Ruthotto and E. Treister).

Map of the modules to the paper (main.tex; S: labels of supplement.tex, Appendices A-C of the
arXiv version):
- latent.py          the latent gamma, N(0, I) conditioned on Omega = [-L, L]^D (sec:formulation)
- frame.py           the frame x = c + s x' (sec:setup, S:sec:splinedetails)
- targets.py         the targets pi and the population velocity v_pi (sec:data, eq:velocity)
- empirical.py       v_N, rho_N (eq:empirical)
- regularizers.py    the Hessian, gradient and smallness regularizers (eq:regfm, eq:regmatrix)
- splines.py         cubic B-splines, Gram and data matrices (sec:solver, eq:datamatrix)
- normal_equations.py, solvers.py  eq:normaleq and its sparse Cholesky solve (sec:solver)
- selection.py       oracle eq:oracle, hold-out and the lam search (sec:cv, sec:setup)
- velocity.py        lam(t) over the time points and the refit (sec:cv)
- sampler.py         RK4 in tau for x and log p (sec:sampling)
- metrics.py         the TV estimators (eq:tv, sec:setup)
- baselines/         KDEs, fourth-order KDEs, Gabriel smoothing, EM (S:sec:kde)
- config.py          every constant of the setup; runlog.py  run logs and manifests
Swappable components: the regularizer (regularizers.Regularizer, get_regularizer with the keys "h2", "h1",
"l2"), the selection rule (Oracle, HoldOut: score(lam), solution(lam)), the target
(targets.Target), and the sampler's velocity protocol ``v(x, t)``, ``v.div(x, t)``.
"""

from . import config
from .empirical import EmpiricalField, EmpiricalFlow, StoppedLaw
from .frame import AffineFrame, AffineTarget
from .latent import TruncatedGaussian
from .metrics import frac_within, tv_grid, tv_pos
from .normal_equations import NormalEquations
from .regularizers import get_regularizer
from .sampler import backward_log_density
from .selection import HoldOut, LambdaSearch, Oracle
from .solvers import default_solver
from .splines import Cubic1D, Cubic2D
from .targets import (Checkerboard, EightGaussians, Inhomogeneous, NoisyCircle,
                      PopulationVelocity, TrimodalBeta, TwoSpirals)
from .velocity import Velocity

__all__ = ["config", "EmpiricalField", "EmpiricalFlow", "StoppedLaw", "AffineFrame",
           "AffineTarget", "TruncatedGaussian", "tv_pos", "tv_grid", "frac_within",
           "NormalEquations", "get_regularizer", "backward_log_density", "Oracle", "HoldOut",
           "LambdaSearch", "default_solver", "Cubic1D", "Cubic2D", "TrimodalBeta",
           "Inhomogeneous", "NoisyCircle", "Checkerboard", "EightGaussians", "TwoSpirals",
           "PopulationVelocity", "Velocity"]
