"""2D helpers shared by twod, kde2d, kde4_2d and twod_eval (target settings of
config.TWOD_TARGETS, grid TV, representative sets, e(t)), so that no compute
script imports another. Paper: Table 1 (tab:main), Table 3 (tab:twod-full), Figure 4
(fig:twod-ops)."""
import common as C  # noqa: F401, I001  (sets the thread variables before numpy)

import numpy as np

from smoothfm import config, tv_pos
from smoothfm.splines import far_clip


class Twod:
    """The per-target settings of config.TWOD_TARGETS (KDE grid, bandwidths, lattice, figure
    grid); smoke=True thins the bandwidth candidates and the figure grid
    (config.TWOD_SMOKE_STRIDE)."""

    def __init__(self, name: str, smoke: bool = False):
        cfg = config.TWOD_TARGETS[name]
        self.name, self.cfg = name, cfg
        kind, half, dx = cfg["grid"]
        if kind == "nodes":
            self.xg = np.arange(-half, half + dx / 2, dx)
        else:
            self.xg = -half + dx / 2 + dx * np.arange(int(round(2 * half / dx)))
        self.dx = float(dx)
        self.half = float(half)
        self.outside = bool(cfg["grid_tv_outside"])
        self.nfft = int(cfg["nfft"])
        st = config.TWOD_SMOKE_STRIDE
        self.hs = np.geomspace(*cfg["hs"])[::st["hs"] if smoke else 1]
        lat = config.TWOD_ANISO_LATTICE
        self.H = lat["h0"] * lat["ratio"] ** np.arange(lat["nh"])
        self.nmax_n = int(lat["nmax_n"])
        self.pairs = [(self.H[i], self.H[j]) for i in range(self.H.size)
                      for j in range(min(i + 1, self.nmax_n + 1))]
        self.aniso = bool(cfg["aniso"])
        xh, nx = cfg["xf2"]
        self.xf2 = np.linspace(-xh, xh, nx)[::st["xf2"] if smoke else 1]

    def describe(self) -> dict:
        return {"kde grid": f"{self.cfg['grid'][0]} on [-{self.half:g}, {self.half:g}]^2, dx "
                            f"{self.dx:g} ({self.xg.size}^2), nfft {self.nfft}, grid TV adds "
                            f"outside mass: {self.outside}",
                "bandwidths": f"geomspace{tuple(self.cfg['hs'])} -> {self.hs.size} candidates",
                "aniso lattice": (f"{len(self.pairs)} pairs H = {self.H[0]:g} "
                                  f"{config.TWOD_ANISO_LATTICE['ratio']:g}^k, k < {self.H.size}, "
                                  f"h_N <= H[{self.nmax_n}]") if self.aniso else "none (no tangent)",
                "figure grid": f"linspace(-{self.xf2[-1]:g}, {self.xf2[-1]:g}, {self.xf2.size})"}


def grid_points(xg):
    """The grid xg x xg (ij indexing) as (G, G, 2)."""
    return np.stack(np.meshgrid(xg, xg, indexing="ij"), -1)


def pdf_grid(target, xg):
    """Target density on xg x xg; non-finite values (the circle's origin, 0 * inf) set to 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f = target.pdf(grid_points(xg))
    return np.where(np.isfinite(f), f, 0.0)


def grid_tv(f, pg, dx, outside):
    """0.5 dx^2 sum |f - p|, plus 0.5 max(0, 1 - dx^2 sum f) if outside (the KDE mass off the
    grid, where p has none)."""
    tv = 0.5 * float(np.sum(np.abs(f - pg)) * dx * dx)
    if outside:
        tv += 0.5 * max(0.0, 1.0 - float(np.sum(f)) * dx * dx)
    return tv


def rep_sets(d, ns, smoke, rule="holdout"):
    """Representative set per N: the lower median of the stored hold-out tv_pos of a twod
    result (set 0 in a smoke run)."""
    out = {}
    stored = list(np.asarray(d["ns"])) if d is not None else []
    for n in ns:
        if smoke or n not in stored:
            out[n] = 0
            continue
        v = np.asarray(d[f"{rule}/tv_pos"])[stored.index(n)]
        o = np.argsort(v, kind="stable")
        out[n] = int(np.asarray(d["sets"])[o[(o.size - 1) // 2]])
    return out


def fit_values(v, k, x):
    """v(x, t_k) of the stored fit at t_k (no refit solve)."""
    t = float(v.times[k])
    xc, _ = far_clip(v.basis, np.asarray(x, float))
    return np.asarray(v.basis.extended_design(xc, (0, 0)) @ v.coef[k]) / (1.0 - t)


def velocity_errors(v, pop):
    """e(t_k) = mean |v(x, t_k) - v*(x, t_k)|^2 at the oracle points of pop = (z, y, v*)."""
    z, y, vpop = pop
    out = []
    for k, t in enumerate(v.times):
        d = fit_values(v, k, (1 - t) * z + t * y) - vpop[k]
        out.append(float(np.mean(np.sum(d * d, 1))))
    return np.array(out)


def stacked_latents(st, M, s):
    """M Sobol latents (rng SEEDS['sobol_2d']) on M iid latents (rng SEEDS['iid'] + s)."""
    return np.vstack([st.latents(M), st.iid_latents(M, s)])


def split_tv(st, g, M):
    """tv_pos of the Sobol half and tv_iid of the iid half of a stacked generation g."""
    tv, se = tv_pos(st.std, g["z"][:M], g["lf"][:M])
    ti, si = tv_pos(st.std, g["z"][M:], g["lf"][M:])
    g.update(tv_pos=tv, tv_pos_se=se)
    return {"tv_iid": ti, "tv_iid_se": si}
