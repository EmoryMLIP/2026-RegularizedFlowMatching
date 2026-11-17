"""numbers.txt: the quantitative sentences of the paper (main text and appendices) whose
numbers no generated table or figure prints, recomputed from the result arrays (report stage;
seconds).
Paper: Sec. 5.3-5.7 (sec:flowaccuracy, sec:discretization, sec:stability, sec:operator,
sec:lambda).

One line per number: paper location | quantity | value | source arrays | compute runs. The
last field names each result with the compute script and arguments that write it (the line of
experiments/runs.txt; T a 2D target, [--reg R] the ablation runs). A missing source prints
'pending' as the value. The paper text states rounded or bounding versions of these
values; this file holds the exact ones, so that a reader can check each sentence.
"""

from __future__ import annotations

import os

import numpy as np

from smoothfm import config

import common as C
import stability as ST
import tables as T

NAME = "numbers.txt"
CASES_2D = [t for t, _ in T.TWOD_CASES]
CASES = list(C.DATASETS) + CASES_2D
PENDING = "pending"
# the stability blocks of sec:stability: (result, target, N)
STAB = [(ST.SRC_1D, "trimodal", 100), (ST.SRC_1D, "inhom", 100), (ST.SRC_2D, "circle", 1000)]
# the compute runs of each line (result = script and arguments, as in experiments/runs.txt)
RUN_STAB = "stability_1d = stability.py --dim 1; stability_circle = stability.py --target circle"
RUN_REG = ("oned[_<R>] = oned.py [--reg R]; twod_<T>[_<R>] = twod.py --target T [--reg R] "
           "(R = h1, l2)")
RUN_HERO = "gabriel_hero_reg = gabriel_hero.py --regs"
RUN_CIRCLE_L2 = {"stability_circle_l2_dx005": "stability.py --target circle --reg l2 --grid-dx 0.005",
                 "stability_circle_l2": "stability.py --target circle --reg l2"}


def _arr(v):
    return None if v is None or v is T.PENDING else np.asarray(v, float)


def _line(loc, what, value, source, runs):
    return f"{loc} | {what} | {value} | {source} | {runs}"


def _stab_index(d, target, n):
    """(index of N, epsilon) of a stability file, or None."""
    if d is None:
        return None
    j = ST._nidx(d, target, n)
    return None if j is None else (j, ST.epsilon_of(d))


def lam_changed():
    """sec:stability: the largest number of times t_k > 0 at which the re-selected Hessian
    lambda differs from the unperturbed selection, epsilon <= 1e-4."""
    out, n_t = [], None
    for name, target, n in STAB:
        d = C.load(name)
        ix = _stab_index(d, target, n)
        key = f"{target}/h2_resel/lam_changed"
        if ix is None or key not in d:
            out.append(f"{target} {PENDING}")
            continue
        j, epsilon = ix
        v = d[key][j][:, epsilon <= 1e-4]                  # (S, E, R)
        n_t = int(np.sum(np.asarray(d["times"], float) > 0))
        out.append(f"{target} N={n}: {int(np.nanmax(v))}")
    return _line("sec:stability, 'no more than 3 of the 48 times'",
                 f"max over sets, draws, epsilon <= 1e-4 of the times (of {n_t} with t > 0) at which "
                 "the re-selected Hessian lambda changes", "; ".join(out),
                 "stability_1d, stability_circle: <T>/h2_resel/lam_changed", RUN_STAB)


def resel_factors():
    """sec:stability: re-selected over fixed lambda, mean distance of the laws (TV) and of
    the fields (sup_t H^2), the largest ratio over epsilon <= 1e-3 and the three blocks."""
    tv, h2 = [], []
    for name, target, n in STAB:
        d = C.load(name)
        for which, fn in (("tv", ST.tv_values), ("h2", ST.h2_values)):
            a, b = fn(d, target, n, "h2_resel"), fn(d, target, n, "h2_fixed")
            if a is None or b is None:
                continue
            epsilon = a[0]
            r = ST.stats(a[1])[0] / ST.stats(b[1])[0]
            r = r[epsilon <= ST.SLOPE_MAX]
            (tv if which == "tv" else h2).append((float(np.nanmax(r)), target))
    fmt = lambda v: (f"{max(v)[0]:.2f} ({max(v)[1]}); per block "
                     + ", ".join(f"{t} {x:.2f}" for x, t in v)) if v else PENDING
    return [_line("sec:stability, 'within a factor of 3.4 of those'",
                  "max over epsilon <= 1e-3 of mean TV(p_eps, p_0) re-selected / fixed lambda, Hessian",
                  fmt(tv), "stability_1d, stability_circle: <T>/h2_{resel,fixed}/tv", RUN_STAB),
            _line("sec:stability, 'larger by a factor of up to 42'",
                  "max over epsilon <= 1e-3 of mean sup_t ||v_t - v_t'||_H2 re-selected / fixed, Hessian",
                  fmt(h2), "stability_1d, stability_circle: <T>/h2_{resel,fixed}/dnorm2",
                  RUN_STAB)]


def circle_ratio():
    """sec:stability: law and field of the smallness over the Hessian flow on the circle at
    epsilon = 1e-6, fixed lambda."""
    d2, dl = C.load(ST.SRC_2D), C.load(ST.circle_l2_source())
    out = []
    for which, fn in (("TV", ST.tv_values), ("sup_t H2", ST.h2_values)):
        a, b = fn(dl, "circle", 1000, "l2_fixed"), fn(d2, "circle", 1000, "h2_fixed")
        if a is None or b is None:
            out.append(f"{which} {PENDING}")
            continue
        k = int(np.argmin(np.abs(np.log10(a[0]) + 6)))
        out.append(f"{which} {ST.stats(a[1])[0][k] / ST.stats(b[1])[0][k]:.1f}")
    return _line("sec:stability, '68 and 108 times as much'",
                 "circle N = 1000, m = 128, epsilon = 1e-6: mean distance k = 0 over k = 2, fixed lambda",
                 ", ".join(out), f"{ST.circle_l2_source()}, {ST.SRC_2D}: circle/{{l2,h2}}_fixed/"
                 "{tv,dnorm2}", f"{ST.circle_l2_source()} = {RUN_CIRCLE_L2[ST.circle_l2_source()]}; "
                 "stability_circle = stability.py --target circle")


def exact_field_growth():
    """sec:stability: the field difference of v_N in L2(rho_N(., t)) per decade of 1 - t
    (epsilon = 1e-6), mean over sets and draws."""
    out = []
    for name, target, n in STAB[::2]:
        d = C.load(name)
        ix = _stab_index(d, target, n)
        key = f"{target}/exact/l2diff2"
        if ix is None or key not in d:
            out.append(f"{target} {PENDING}")
            continue
        j, epsilon = ix
        omt = np.asarray(d["tl"], float)                 # the values of 1 - t (decreasing)
        k = int(np.argmin(np.abs(np.log10(epsilon) + 6)))
        v = np.nanmean(np.sqrt(d[key][j][:, k]), axis=(0, 1))       # (K,)
        o = np.argsort(-omt)
        steps = ", ".join(f"1-t {omt[o][i]:.0e} -> {omt[o][i + 1]:.0e}: "
                          f"x{v[o][i + 1] / v[o][i]:.1f}" for i in range(len(o) - 1))
        out.append(f"{target} N={n} ({steps})")
    return _line("sec:stability, 'grows tenfold per decade of 1-t'",
                 "||v_N - v_N'||_{L2(rho_N(., t))}, ratio between consecutive times, epsilon = 1e-6",
                 "; ".join(out), "stability_1d, stability_circle: <T>/exact/l2diff2, tl", RUN_STAB)


def null_space_share():
    """sec:operator: share of the (set, t) pairs at which hold-out selects the top candidate
    of the lambda grid (the minimizer then lies numerically in the null space of the
    regularizer), per regularizer and target, pooled over N."""
    shares = []
    for reg in ("h2", "h1", "l2"):
        for case in CASES:
            dim = T.CASE_DIM[case]
            d = C.load(T.reg_flow_name(case, reg))
            key = (f"{case}/holdout/lam_rel" if dim == 1
                   else T.twod_flow_key(d, case, "holdout", "lam_rel") if d is not None else "")
            if d is None or key not in d:
                continue
            top = 10.0 ** (config.LAM_DECADES[reg][dim][1] - 1)
            lr = np.asarray(d[key], float)
            shares.append((float(np.mean(lr >= top * (1 - 1e-9))), reg, case))
    if not shares:
        return _line("sec:operator, 'at most 5.4% of the pairs'", "", PENDING, "", RUN_REG)
    big = sorted(shares, reverse=True)[:3]
    return _line("sec:operator, 'at most 5.4% of the pairs'",
                 "max over regularizer and target of the share of hold-out (N, set, t) at the top "
                 "candidate 10^(hi-1) lambda_s",
                 f"{100 * big[0][0]:.1f}% ({big[0][1]}, {big[0][2]}); next "
                 + ", ".join(f"{100 * s:.1f}% ({r}, {c})" for s, r, c in big[1:]),
                 "oned[_h1,_l2]: <T>/holdout/lam_rel; twod_<T>[_h1,_l2]: holdout/lam_rel", RUN_REG)


def hessian_vs_gradient():
    """sec:operator: training sets with N >= 1000 where the Hessian hold-out flow has a lower
    TV than the gradient hold-out flow (six targets, two N, five sets)."""
    k = m = 0
    for case in CASES:
        for n in C.SIZES[1:]:
            a, b = (_arr(T.reg_flow_tv(T.Sources(), case, r, "holdout", n)) for r in ("h2", "h1"))
            if a is None or b is None:
                continue
            k += int(np.sum(a < b))
            m += a.size
    return _line("sec:operator, 'in 57 of the 60 training sets'",
                 "sets with N >= 1000 where k = 2 hold-out TV < k = 1 hold-out TV",
                 f"{k} of {m}" if m else PENDING,
                 "oned[_h1], twod_<T>[_h1]: hold-out tv_pos (tab_reg_full)",
                 "oned[_h1] = oned.py [--reg h1]; twod_<T>[_h1] = twod.py --target T [--reg h1]")


def const_vs_holdout():
    """sec:lambda: 2D training sets where the hold-out constant lambda is less accurate than
    the hold-out lambda(t)."""
    S, k, m = T.Sources(), 0, 0
    for case in CASES_2D:
        tv = T.main_rows(S, case)[4]
        for n in C.SIZES:
            h, c = _arr(tv("holdout", n)), _arr(T.main_const(S, case, n))
            if h is None or c is None:
                continue
            k += int(np.sum(c > h))
            m += h.size
    return _line("sec:lambda, 'on all 60 training sets'",
                 "2D sets where the hold-out constant lambda has a higher TV than hold-out lambda(t)",
                 f"{k} of {m}" if m else PENDING,
                 "twod_eval_<T> / twod_<T> hold-out tv_pos, lam_sweep_<T>[_nN] sel_tv (tab_main)",
                 "twod_<T> = twod.py --target T; twod_eval_<T> = twod_eval.py --target T; "
                 "lam_sweep_<T>[_nN] = lam_sweep.py --target T --reg h2 [--n N --const-only] "
                 "(N = 100, 3000)")


def _mean_tv(v):
    """Mean over sets of a per-set TV array, NaN if missing or pending."""
    a = _arr(v)
    return np.nan if a is None else T.stats(a)[0]


def kde_best_per_n():
    """sec:flowaccuracy: the hold-out flow against the best data-driven KDE chosen for each N
    separately (tab_main picks one KDE per target over all N): relative difference of the mean
    TV over sets, (flow - KDE) / KDE, per target and N. Candidates: the data-driven KDE rows
    of select_oned / select_twod."""
    S, cells = T.Sources(), []
    for case, _ in T.MAIN_CASES:
        sel, _, _, _, tv = T.main_rows(S, case)
        cands = [m for m, _ in dict(sel.rankings)["data-driven KDE"]]
        for n in C.SIZES:
            h = _mean_tv(tv("holdout", n))
            kde = [(_mean_tv(tv(m, n)), m) for m in cands]
            kde = [(v, m) for v, m in kde if np.isfinite(v)]
            if not np.isfinite(h) or not kde:
                continue
            v, m = min(kde)
            cells.append((case, n, 100 * (h / v - 1), m))
    loc = ("sec:flowaccuracy, 'its error is 18 to 39% lower on the noisy circle and on the spirals "
           "with N >= 1000, and between 2% lower and 26% higher in the other cases'")
    if not cells:
        return _line(loc, "", PENDING, "", "")
    lower = [c for c in cells if c[0] == "circle" or (c[0] == "spirals" and c[1] >= 1000)]
    other = [c for c in cells if c not in lower]
    rng = (f"{min(-c[2] for c in lower):.1f} to {max(-c[2] for c in lower):.1f}% lower"
           if all(c[2] < 0 for c in lower) else "not all lower")
    worst, best = max(other, key=lambda c: c[2]), min(other, key=lambda c: c[2])
    return _line(loc, "hold-out flow vs the data-driven KDE with the lowest mean TV at each N, "
                 "(TV flow - TV KDE) / TV KDE, mean over sets",
                 f"circle and spirals N >= 1000: {rng}; other cases {best[2]:+.1f}% "
                 f"({best[0]} N={best[1]}) to {worst[2]:+.1f}% ({worst[0]} N={worst[1]}); "
                 "per target and N: "
                 + ", ".join(f"{c} N={n} {r:+.1f}% ({m})" for c, n, r, m in cells),
                 "oned, oned_eval, kde4; twod_<T>, kde2d_<T>, twod_eval_<T>, kde4_2d_<T> "
                 "(the rows of tab_oned_full / tab_twod_full)",
                 "oned = oned.py; oned_eval = oned_eval.py; kde4 = kde4.py; twod_<T> = twod.py "
                 "--target T; kde2d_<T> = kde2d.py --target T; twod_eval_<T> = twod_eval.py "
                 "--target T; kde4_2d_<T> = kde4_2d.py --target T")


def circle_late_share():
    """sec:lambda: circle, N = 1000: share of the (set, t < 0.9) pairs at which hold-out
    selects lambda below a tenth of its median over t >= 0.97."""
    d = C.load("twod_circle")
    j = C.n_index(d, 1000)
    if j is None:   # also a smoke directory, which holds N = 100 only
        return _line("sec:lambda, '77% of the pairs'", "", PENDING, "twod_circle",
                     "twod_circle = twod.py --target circle")
    lam = np.asarray(d[T.twod_flow_key(d, "circle", "holdout", "lam")][j], float)   # (S, K)
    t = np.asarray(d["times"], float)
    early, late = t < 0.9 - 1e-12, t >= 0.97 - 1e-12
    pooled = np.mean(lam[:, early] < 0.1 * np.median(lam[:, late]))
    per_set = np.mean([np.mean(r[early] < 0.1 * np.median(r[late])) for r in lam])
    return _line("sec:lambda, 'at 77% of the pairs of training set and time with t < 0.9'",
                 "circle N = 1000: share of (set, t < 0.9) with hold-out lambda < 0.1 x median "
                 "over t >= 0.97", f"{100 * pooled:.1f}% (median pooled over sets), "
                 f"{100 * per_set:.1f}% (median per set)", "twod_circle: holdout/lam, times",
                 "twod_circle = twod.py --target circle")


def integrator_tv():
    """sec:discretization: |TV(RK4, 96 steps) - TV(DOP853)| of the hold-out flow, max over
    training sets (and N), 1D and 2D."""
    out = []
    for dim, names in ((1, [("integration_error", c) for c in C.DATASETS]),
                       (2, [(f"integration_error_{c}", c) for c in CASES_2D])):
        best = None
        for name, case in names:
            d = C.load(name)
            ka, kb = f"{case}/holdout/rk96/tv", f"{case}/holdout/dop853/tv"
            if d is None or ka not in d or kb not in d:
                continue
            v = float(np.nanmax(np.abs(np.asarray(d[ka], float) - np.asarray(d[kb], float))))
            best = max(best or (v, case), (v, case))
        out.append(f"{dim}D {best[0]:.2e} ({best[1]})" if best else f"{dim}D {PENDING}")
    return _line("sec:discretization, 'the 96 RK4 steps change the total variation error of "
                 "the hold-out flows by at most 2.6e-4 in one and 9e-4 in two dimensions'",
                 "max over sets of |TV RK4 (96 steps) - TV DOP853|, hold-out flow",
                 ", ".join(out), "integration_error[_<T>]: <T>/holdout/{rk96,dop853}/tv",
                 "integration_error = integration_error.py; integration_error_<T> = "
                 "integration_error.py --target T")


def operator_1d():
    """sec:operator, trimodal N = 100, the median hold-out training set: share of
    the generated points within GABRIEL_HERO['within_r'] of a training point, and the
    largest hold-out lambda / lambda_s of the smallness flow."""
    d = C.load("gabriel_hero_reg")
    loc = "sec:operator, '31% of its samples within 1e-3 ... against 15 to 17%'"
    if d is None:
        return [_line(loc, "", PENDING, "gabriel_hero_reg", RUN_HERO)]
    pre = f"trimodal/n100/s{int(d['trimodal/n100/set_median'])}"   # the set of the figure
    r = config.GABRIEL_HERO["within_r"]
    lab = {"holdout_l2": "k = 0", "holdout_h1": "k = 1", "holdout": "k = 2", "G1": "Gabriel G1"}
    shares = [f"{lab[f]} {100 * float(d[f'{pre}/{f}/within']):.1f}%" for f in lab
              if f"{pre}/{f}/within" in d.keys()]
    lr = np.asarray(d[f"{pre}/holdout_l2/lam_rel"], float)
    return [_line(loc, f"share of generated points within {r:g} of a training point, "
                  f"{pre.replace('/', ' ')} (median set)", ", ".join(shares),
                  "gabriel_hero_reg: trimodal/n100/<set>/<flow>/within", RUN_HERO),
            _line("sec:operator, 'hold-out selects lambda/lambda_s ~ 3e6'",
                  "max hold-out lambda/lambda_s of the k = 0 flow, same set",
                  f"{np.nanmax(lr):.3g}", "gabriel_hero_reg: <prefix>/holdout_l2/lam_rel", RUN_HERO)]


def holdout_vs_oracle_1d():
    """sec:lambda: 1D, decades between the hold-out and the oracle lambda(t) at late times
    (t >= 0.9): share within one decade and the largest distance, over N, sets and times."""
    d = C.load("oned")
    out = []
    for case in C.DATASETS:
        kh, ko = f"{case}/holdout/lam_rel", f"{case}/oracle/lam_rel"
        if d is None or kh not in d or ko not in d:
            out.append(f"{case} {PENDING}")
            continue
        t = np.asarray(d["times"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            dec = np.abs(np.log10(np.asarray(d[kh], float)) - np.log10(np.asarray(d[ko], float)))
        dec = dec[..., t >= 0.9 - 1e-12]
        dec = dec[np.isfinite(dec)]
        out.append(f"{case}: {100 * np.mean(dec <= 1 + 1e-9):.0f}% within one decade, "
                   f"max {dec.max():.2f} decades")
    return _line("sec:lambda, 'within one decade of the oracle at most late times ... up to "
                 "three decades'", "|log10 lambda_holdout - log10 lambda_oracle|, t >= 0.9, "
                 "all N and sets", "; ".join(out), "oned: <T>/{holdout,oracle}/lam_rel, times",
                 "oned = oned.py")


def build() -> list:
    lines = ["# Generated by experiments/report/prose_numbers.py; do not edit by hand.",
             f"# {C.results_line()[2:]}",
             "# Numbers of the paper (main text and appendices) that no generated table or figure prints.",
             "# paper location | quantity | value | source arrays | compute runs (result = script "
             "and arguments, experiments/runs.txt)"]
    lines += [lam_changed(), *resel_factors(), circle_ratio(),
              exact_field_growth(), integrator_tv(), null_space_share(), hessian_vs_gradient(),
              *operator_1d(), holdout_vs_oracle_1d(), const_vs_holdout(),
              kde_best_per_n(),
              circle_late_share()]
    C.write_text(os.path.join(C.OUT, NAME), "\n".join(lines) + "\n")
    return [NAME]
