"""Benchmark methods of S:sec:kde: Gaussian KDEs hat pi_sigma and their bandwidth rules
(LSCV, Silverman, Sheather-Jones, Abramson), fourth-order KDEs with the kernel G^[4] and
their grid densities, the anisotropic KDE, Gabriel et al.'s convolution
v_h = G_{t sqrt h} * v_N, and EM for the eight-Gaussian mixture (the parametric row of
tab:twod-full).

Notation (paper -> code): bandwidth sigma -> h; standard normal density G -> phi; Abramson
bandwidths sigma_i = sigma (tilde pi(x_i)/g)^{-1/2} -> h * lam_i, so ``lam`` holds the
factors sigma_i / sigma (not the regularization parameter); G^[4] -> K4 / k4;
(sigma_T, sigma_N) -> (h_T, h_N). The experiments pass these names by keyword.
"""

from .bandwidth import abramson_factors, aniso_lscv, lscv_curve, sheather_jones, silverman
from .em import em_from, mix_logpdf, one_hot
from .gabriel import StdGabriel, h_schedule
from .grid_density import GridDensity1D, GridDensity2D
from .kde import KDE, aniso_covs, kde_grid_1d, kde_grid_2d
from .kde4 import (kde4_aniso_grid_2d, kde4_grid_1d, kde4_grid_2d, lscv4_aniso, lscv4_curve,
                   lscv4_curve_2d)

__all__ = ["KDE", "silverman", "sheather_jones", "lscv_curve", "abramson_factors",
           "aniso_lscv", "kde_grid_1d", "kde_grid_2d", "aniso_covs", "StdGabriel", "h_schedule",
           "kde4_grid_1d", "kde4_grid_2d", "kde4_aniso_grid_2d", "lscv4_curve", "lscv4_curve_2d",
           "lscv4_aniso", "GridDensity1D", "GridDensity2D", "em_from", "mix_logpdf", "one_hot"]
