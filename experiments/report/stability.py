"""Report of the stability perturbation test: fig_stability.tex,
tab_stability.tex and tab_stability_refine.tex from stability_1d, stability_circle,
stability_circle_l2_dx005 (else stability_circle_l2) and stability_1d_m{256,512,1024}
(compute/stability.py; config.EPSILON, config.STABILITY). Paper: Figure 3 (fig:stability),
Table 8 (tab:stability), Table 9 (tab:stability-refine); Sec. 5.5.

fig_stability.tex, one row (printed at the siamart \\textwidth, 13.02 cm):
  (a) trimodal, N = 100: TV(p_epsilon, p_0) against epsilon (log-log) for the exact v_N
      (stopped mixtures), the Hessian regularizer (k = 2) with fixed and re-selected lam(t),
      the smallness regularizer (k = 0) fixed and re-selected;
  (b) circle, N = 1000: the same; the k = 0 curves come from stability_circle_l2 and are
      omitted (header note) without it;
  (c) sup_t ||v_t - v_t'||_{H^2(Omega)} (frame units) against epsilon for k = 2 and k = 0,
      fixed and re-selected, trimodal N = 100 (solid) and circle N = 1000 (dotted), slope-1
      references of the k = 2 fixed curves.
  Markers: mean over sets and draws; error bars: min and max over them. A panel without any
  source draws empty with the text "missing".
tab_stability.tex: the stability constants (a) mean TV / epsilon and (b) mean sup_t H^2
difference / epsilon per (target, N, flow) at each epsilon, with the least-squares slope of the
log-log means over epsilon <= 1e-3; one block per (target, N) under a rotated label; the circle
k = 0 rows from stability_circle_l2 when present (else omitted, header note).
tab_stability_refine.tex: the constants TV/epsilon, sup_t L2/epsilon, sup_t H2/epsilon per
spline resolution m (stability_1d_m<m>), mean [min, max], and the growth m = 1024 / 256.

Keys read (arrays (len(ns), sets, len(epsilon), draws, ...)): epsilon; <target>/ns;
<target>/<flow>/tv; <target>/<flow>/dnorm2 (..., 49, 3: L2, H1, H2 seminorm parts squared).
"""

from __future__ import annotations

import os
import warnings

import numpy as np

import common as C
from smoothfm import config

NAME = "fig_stability"
PRINT_CM = 13.02                     # siamart \textwidth
TICK_PT = 8.0                        # \footnotesize tick labels
# native width estimate: YLAB + 3 W + 2 SEP (cm), axes with 'scale only axis'
W, H, SEP, YLAB = 3.35, 3.0, 1.05, 1.42      # 1.42: measured 13.57 cm (pdfinfo)
FLOW_STYLE = {   # flow: (label, pgf style); regularizer labels from common.penalty_label
    "exact": ("exact $v_N$", "black, thick, mark=x, mark size=2pt"),
    "h2_fixed": (C.penalty_label("h2", short=True) + ", fixed $\\lambda$",
                 "rblue, thick, mark=*, mark size=1.4pt"),
    "h2_resel": (C.penalty_label("h2", short=True) + ", re-selected $\\lambda$",
                 "rgreen, thick, dashed, mark=square*, mark size=1.3pt"),
    "l2_fixed": (C.penalty_label("l2", short=True) + ", fixed $\\lambda$",
                 "rverm, thick, mark=triangle*, mark size=1.7pt"),
    "l2_resel": (C.penalty_label("l2", short=True) + ", re-selected $\\lambda$",
                 "rpurple, thick, dashed, mark=diamond*, mark size=1.7pt"),
}
# the smallest evaluated candidate of the stability selections (compute/stability.py
# PositiveHoldOut): the lowest decade, then its lower half-decade neighbour
LOW_H2 = {d: f"10^{config.LAM_DECADES['h2'][d][0] - 0.5:g}" for d in (1, 2)}
LOW_L2 = f"10^{config.LAM_DECADES['l2'][1][0]:g}"
TABLE_FLOWS = ["exact", "h2_fixed", "h2_resel", "l2_fixed", "l2_resel"]
# names in the rotated block labels (C.ROT_LINE_WIDTH_PT has their widths)
ROT_NAME = {"trimodal": "Trimodal", "inhom": "Inhom.", "circle": "Circle"}
SLOPE_MAX = 1e-3
SRC_1D, SRC_2D = "stability_1d", "stability_circle"
# the smallness regularizer on the circle, a separate run (compute/stability.py --target
# circle --reg l2) so that the B_2 circle result stays as it was; optional
SRC_2D_L2 = "stability_circle_l2"
# the same run with the TV grid spacing 0.005 (--grid-dx 0.005). It is preferred when present:
# at spacing 0.02 the k = 0 densities lose up to 1.1% of the grid mass (log warning), at 0.005
# the mass stays within 1e-3 of one; the TV of the two agrees within 2% (grid convergence)
SRC_2D_L2_CANDIDATES = ("stability_circle_l2_dx005", "stability_circle_l2")


def circle_l2_source() -> str:
    """Name of the circle smallness run the report reads: the first candidate present."""
    for name in SRC_2D_L2_CANDIDATES:
        if C.load(name) is not None:
            return name
    return SRC_2D_L2_CANDIDATES[-1]
L2_FLOWS = ["l2_fixed", "l2_resel"]
H2_FLOWS = ["h2_fixed", "h2_resel"]
REF_MS = [256, 512, 1024]
REF_FLOWS = ["l2_resel", "l2_fixed", "h2_resel", "h2_fixed"]
# the perturbation sizes in the npz (the one place to change if the compute script renames it)
EPSILON_KEY = "epsilon"


def epsilon_of(d):
    return np.asarray(d[EPSILON_KEY], float)


def _nidx(d, target, n):
    if d is None or f"{target}/ns" not in d:
        return None
    ns = [int(x) for x in np.atleast_1d(d[f"{target}/ns"])]
    return ns.index(n) if n in ns else None


def tv_values(d, target, n, flow):
    """(epsilon, per-(set, draw) TV (E, S*R)) or None."""
    j = _nidx(d, target, n)
    key = f"{target}/{flow}/tv"
    if j is None or key not in d:
        return None
    v = d[key][j]                                    # (S, E, R)
    return epsilon_of(d), np.moveaxis(v, 1, 0).reshape(v.shape[1], -1)


def h2_values(d, target, n, flow):
    """(epsilon, per-(set, draw) sup_t ||v_t - v_t'||_{H^2} (E, S*R)) or None."""
    j = _nidx(d, target, n)
    key = f"{target}/{flow}/dnorm2"
    if j is None or key not in d:
        return None
    v = np.sqrt(np.sum(d[key][j], -1)).max(-1)      # (S, E, R)
    return epsilon_of(d), np.moveaxis(v, 1, 0).reshape(v.shape[1], -1)


def stats(vals):
    """mean, min, max over the last axis (finite entries)."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(vals, 1), np.nanmin(vals, 1), np.nanmax(vals, 1)


def slope(epsilon, y, emax=SLOPE_MAX) -> float:
    epsilon, y = np.asarray(epsilon, float), np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0) & (epsilon <= emax)
    return (float(np.polyfit(np.log10(epsilon[ok]), np.log10(y[ok]), 1)[0]) if ok.sum() >= 2
            else np.nan)


def _slope_txt(x) -> str:
    return f"${x:.2f}$" if np.isfinite(x) else "--"


def write_curve(path, epsilon, vals, comment):
    m, lo, hi = stats(vals)
    C.write_dat(path, {"eps": epsilon, "mean": m, "lo": lo, "hi": hi, "ep": hi - m, "em": m - lo},
                comment)
    return m


def addplot(flow, rel, style=None):
    """One curve with min-max error bars; legend entries are drawn separately (legend_images)."""
    st = style or FLOW_STYLE[flow][1]
    return (f"\\addplot[{st}, forget plot, error bars/.cd, y dir=both, y explicit] "
            f"table[x=eps, y=mean, y error plus=ep, y error minus=em] {{\\figdir/{rel}}};\n")


def legend_images():
    """The shared legend (collected in panel (a)): the five flows, then panel (c)'s line kinds."""
    out = "".join(f"\\addlegendimage{{{st}}}\\addlegendentry{{{lab}}}\n"
                  for lab, st in FLOW_STYLE.values())
    return out + ("\\addlegendimage{black, thick}\\addlegendentry{(c) trimodal}\n"
                  "\\addlegendimage{black, thick, densely dotted}\\addlegendentry{(c) circle}\n"
                  "\\addlegendimage{gray, thin, dashdotted}\\addlegendentry{slope 1}\n")


def missing_node():
    """An empty panel: an invisible curve keeps the x range of the others, then the text."""
    return ("\\addplot[draw=none, forget plot] coordinates {(1e-6,1e-6) (1e-1,1)};\n"
            "\\node[font=\\footnotesize, gray] at (rel axis cs:0.5,0.5) {missing};\n")


def build() -> list:
    global SRC_2D_L2
    SRC_2D_L2 = circle_l2_source()
    d1, d2, d2l = C.load(SRC_1D), C.load(SRC_2D), C.load(SRC_2D_L2)
    return build_figure(d1, d2, d2l) + [write_table(d1, d2, d2l), write_refine_table()]


def l2_circle_note(d2l) -> list:
    """Header line when the circle smallness run is absent (its curves and rows omitted)."""
    return ([] if d2l is not None else
            [f"% {SRC_2D_L2}.npz not in the results directory: the circle curves and rows of "
             "the smallness regularizer are omitted"])


def build_figure(d1, d2, d2l=None) -> str:
    written = []
    notes = [C.src_line(SRC_1D), C.src_line(SRC_2D), C.src_line(SRC_2D_L2)] + l2_circle_note(d2l)
    panels = []
    ymin_tv = np.inf
    # (a), (b): TV against epsilon; one panel may draw from several sources
    for sources in [[(d1, SRC_1D, "trimodal", 100, ["exact"] + H2_FLOWS + L2_FLOWS)],
                    [(d2, SRC_2D, "circle", 1000, ["exact"] + H2_FLOWS),
                     (d2l, SRC_2D_L2, "circle", 1000, L2_FLOWS)]]:
        body = ""
        for (d, src, target, n, f) in [(d, src, target, n, f) for d, src, target, n, flows
                                       in sources for f in flows]:
            tv = tv_values(d, target, n, f)
            if tv is None:
                continue
            rel = f"{NAME}/tv_{target}_n{n}_{f}.dat"
            lo = np.nanmin(tv[1]) if np.isfinite(tv[1]).any() else np.nan
            if np.isfinite(lo) and lo > 0:
                ymin_tv = min(ymin_tv, 10.0 ** np.floor(np.log10(lo)))
            m = write_curve(os.path.join(C.FIGURES, rel), tv[0], tv[1],
                            f"{NAME}: TV(p_eps, p_0), {target} N={n} {f}; mean, min, max over "
                            f"sets x draws; {src}.npz")
            written.append(rel)
            body += addplot(f, rel)
            notes.append(f"% {target} N={n} {f}: TV " + " ".join(f"{x:.2e}" for x in m)
                         + f"; slope (eps <= 1e-3) {slope(tv[0], m):.2f}")
        panels.append(body or missing_node())
    # (c): sup_t H^2 field difference
    body, refs = "", []
    for (d, src, target, n, dash, flows) in [
            (d1, SRC_1D, "trimodal", 100, False, H2_FLOWS + L2_FLOWS),
            (d2, SRC_2D, "circle", 1000, True, H2_FLOWS),
            (d2l, SRC_2D_L2, "circle", 1000, True, L2_FLOWS)]:
        for f in flows:
            hv = h2_values(d, target, n, f)
            if hv is None:
                continue
            rel = f"{NAME}/h2_{target}_n{n}_{f}.dat"
            m = write_curve(os.path.join(C.FIGURES, rel), hv[0], hv[1],
                            f"{NAME}: sup_t ||v_t - v_t'||_H2 (frame units), {target} N={n} {f}; "
                            f"mean, min, max over sets x draws; {src}.npz")
            written.append(rel)
            _, st = FLOW_STYLE[f]
            st = st.replace(", dashed", "")
            st = st + (", densely dotted" if dash else "")
            body += addplot(f, rel, style=st)
            notes.append(f"% {target} N={n} {f}: sup_t H2 " + " ".join(f"{x:.2e}" for x in m)
                         + f"; slope (eps <= 1e-3) {slope(hv[0], m):.2f}")
            if f == "h2_fixed" and np.isfinite(m[0]):
                refs.append((hv[0], 3.0 * m[0] * hv[0] / hv[0][0]))
    for k, (e, r) in enumerate(refs):
        rel = f"{NAME}/ref{k}.dat"
        C.write_dat(os.path.join(C.FIGURES, rel), {"eps": e, "ref": r},
                    f"{NAME}: slope-1 reference, 3 x the fixed-lam mean at the smallest eps")
        written.append(rel)
        body += (f"\\addplot[gray, thin, dashdotted, forget plot] table[x=eps, y=ref] "
                 f"{{\\figdir/{rel}}};\n")
    panels.append(body or missing_node())
    width = YLAB + 3 * W + 2 * SEP
    printed = TICK_PT * PRINT_CM / width
    if printed < 7.0:
        raise SystemExit(f"printed tick labels {printed:.1f} pt < 7 pt (figure width {width:.2f} cm)")
    notes.append(f"% figure width {width:.2f} cm (estimate); printed at {PRINT_CM} cm: ticks "
                 f"{printed:.1f} pt")
    titles = ["(a) trimodal, $N = 100$", "(b) circle, $N = 1000$",
              "(c) $\\sup_t \\|v_t - v_t'\\|_{H^2(\\Omega)}$"]
    ylabels = ["$\\mathrm{TV}(p_\\varepsilon, p_0)$", "", ""]
    # (a) and (b) share the TV range: decades enclosing every min..max bar (TV <= 1)
    tvr = [ymin_tv, 1.0] if np.isfinite(ymin_tv) else None
    axes = ""
    for k, body in enumerate(panels):
        opts = [f"title={{{titles[k]}}}", f"ylabel={{{ylabels[k]}}}" if ylabels[k] else ""]
        if k < 2 and tvr is not None:
            opts.append(f"ymin={tvr[0]:.0e}, ymax=2")
        if k == 0:
            opts.append("legend to name=stablegend")
            body = legend_images() + body
        axes += f"\\nextgroupplot[{', '.join(o for o in opts if o)}]\n" + body
    tex = (
        "% Generated by experiments/report/stability.py; do not edit by hand.\n"
        + C.results_line() + "\n"
        "% Standalone: pdflatex fig_stability.tex in this directory. In the paper:\n"
        "% \\usepackage{standalone} (strips this preamble) and\n"
        "% \\newcommand{\\figdir}{figures}\\resizebox{\\textwidth}{!}{\\input{figures/fig_stability.tex}}\n"
        "% Stability test (compute/stability.py): x_i' = x_i + eps xi_i, |xi_i| = 1, eps in data units;\n"
        "% mean over 5 sets x 2 draws, error bars min..max. (a), (b) TV between the generated laws\n"
        "% (exact v_N: the stopped mixtures rho_N(., t_end)); (c) sup over the 49 t_k of the H^2(Omega)\n"
        "% norm of the field difference (frame units), k = 2 and k = 0 regularizers. Hold-out over the positive\n"
        "% candidates 10^j lam_s(t) only (lam = 0 excluded; stored zeros of the fixed lam(t) re-selected;\n"
        f"% smallest evaluated B_2 candidate {LOW_H2[1]} lam_s in 1D, {LOW_H2[2]} in 2D; a no-op for B_0, whose\n"
        f"% lowest candidate {LOW_L2} lam_s equals the solve floor).\n"
        + "\n".join(notes) + "\n"
        "\\documentclass{standalone}\n\\usepackage{pgfplots}\n\\usepgfplotslibrary{groupplots}\n"
        "\\usetikzlibrary{calc}\n\\pgfplotsset{compat=1.18}\n\\providecommand{\\figdir}{.}\n"
        "\\begin{document}\n\\begin{tikzpicture}\n"
        "\\definecolor{rblue}{HTML}{0072B2}\n\\definecolor{rverm}{HTML}{D55E00}\n"
        "\\definecolor{rgreen}{HTML}{009E73}\n\\definecolor{rpurple}{HTML}{CC79A7}\n"
        f"\\begin{{groupplot}}[group style={{group size=3 by 1, horizontal sep={SEP}cm}},\n"
        f"  scale only axis, width={W}cm, height={H}cm, xmode=log, ymode=log,\n"
        "  xmin=4e-7, xmax=2.5e-1, xtick={1e-6,1e-4,1e-2}, xlabel={$\\varepsilon$},\n"
        "  unbounded coords=jump, tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n"
        "  title style={font=\\footnotesize, yshift=-1ex}, error bars/error bar style={thin},\n"
        "  legend style={font=\\footnotesize, draw=none, legend columns=4, "
        "/tikz/every even column/.append style={column sep=0.25cm}}, legend cell align=left]\n"
        + axes
        + "\\end{groupplot}\n"
        "\\node[anchor=north] at ($(group c2r1.south)+(0,-0.9cm)$) {\\pgfplotslegendfromname{stablegend}};\n"
        "\\end{tikzpicture}\n\\end{document}\n")
    C.write_text(os.path.join(C.FIGURES, f"{NAME}.tex"), tex)
    return [f"figures/{NAME}.tex"] + [f"figures/{r}" for r in written]


def rot_lines(t, n) -> list:
    """Lines of the rotated label of a (target, N) block: the abbreviated name above N."""
    return [ROT_NAME[t], f"$N = {n}$"]


def rot_block(lines, rows, inner_space=0) -> list:
    r"""Rows (LaTeX lines without the first cell) of one block under a rotated label over all
    of them (C.rot_label), with \addlinespace[pad] above and below where the label is taller
    than the block (C.rot_pad_pt)."""
    nrow = sum(1 for r in rows if not r.startswith("\\"))     # raw lines start with a macro
    lab = C.rot_label(lines, nrow, inner_space, smash=True)
    sp = C.rot_pad_pt(C.rot_width(lines), nrow)
    out, first = [], True
    for r in rows:
        if r.startswith("\\"):
            out.append(r)
            continue
        out.append((lab if first else "") + " & " + r)
        first = False
    return ([f"\\addlinespace[{sp:g}pt]"] + out + [f"\\addlinespace[{sp:g}pt]"]) if sp else out


def write_table(d1, d2, d2l=None) -> str:
    """tab_stability.tex: per (target, N) block under a rotated label and flow, (a) the mean
    TV(p_eps, p_0)/eps and (b) the mean sup_t ||v_t - v_t'||_{H^2}/eps per epsilon (the
    stability constants; mean over sets x draws divided by the column's eps) and the
    least-squares slope of the log-log means (not divided) over eps <= SLOPE_MAX."""
    rows = []
    for d, targets, flows in [(d1, ["trimodal", "inhom"], TABLE_FLOWS),
                              (d2, ["circle"], TABLE_FLOWS), (d2l, ["circle"], L2_FLOWS)]:
        if d is None:
            continue
        for t in targets:
            if f"{t}/ns" not in d:
                continue
            for n in [int(x) for x in np.atleast_1d(d[f"{t}/ns"])]:
                for f in flows:
                    tv = tv_values(d, t, n, f)
                    if tv is None:
                        continue
                    hv = None if f == "exact" else h2_values(d, t, n, f)
                    rows.append((t, n, f, epsilon_of(d), stats(tv[1])[0],
                                 None if hv is None else stats(hv[1])[0]))
    epsilon = rows[0][3] if rows else np.asarray(config.EPSILON, float)
    head = " & ".join(f"$10^{{{int(round(np.log10(e)))}}}$" for e in epsilon)
    ncol = 2 + len(epsilon) + 1
    comments = ["% Generated by experiments/report/stability.py; do not edit by hand.",
                C.results_line(), C.src_line(SRC_1D), C.src_line(SRC_2D),
                C.src_line(SRC_2D_L2)] + l2_circle_note(d2l) + [
                "% Stability constants per eps column: (a) mean TV(p_eps, p_0) / eps, (b) mean sup_t",
                "% ||v_t - v_t'||_{H^2(Omega)} / eps (mean over sets x draws, then divided by the column's eps).",
                "% slope: least squares of log10 mean (not divided by eps) against log10 eps, eps <= 1e-3.",
                "% (b) sup over the 49 t_k of the H^2(Omega) norm in frame units.",
                "% First column: target and N, rotated, spanning the block (needs multirow, graphicx).",
                "% lambda: hold-out over the positive candidates 10^j lam_s(t) (lambda = 0 excluded); "
                "fixed: stored lambda(t), zeros re-selected.",
                f"% smallest evaluated B_2 candidate {LOW_H2[1]} lam_s (1D), {LOW_H2[2]} lam_s (2D); a no-op for "
                f"B_0: its lowest candidate {LOW_L2} lam_s equals the solve floor."]
    missing = [nm for nm, d in ((SRC_1D, d1), (SRC_2D, d2)) if d is None]
    if missing:
        comments.append("% pending (missing) sources, their rows print \\tbd: "
                        + ", ".join(f"{nm}.npz" for nm in missing))
    lines = [f"\\begin{{tabular}}{{cl{'r' * (len(epsilon) + 1)}}}", "\\toprule",
             f" & Estimator & \\multicolumn{{{len(epsilon)}}}{{c}}{{$\\varepsilon$}} & slope \\\\",
             f" & & {head} & \\\\", "\\midrule"]
    pend = ([("trimodal", 100), ("inhom", 100)] if d1 is None else []) + \
        ([("circle", 1000)] if d2 is None else [])
    tbd_row = lambda t, n: ("-- & " + " & ".join([C.tbd()] * len(epsilon)) + f" & {C.tbd()} \\\\")
    for p, (title, col) in enumerate([("(a) $\\mathrm{TV}(p_\\varepsilon, p_0)/\\varepsilon$", 4),
                                      ("(b) $\\sup_t \\|v_t - v_t'\\|_{H^2(\\Omega)}/\\varepsilon$", 5)]):
        if p:
            lines.append("\\midrule[\\heavyrulewidth]")
        lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{{title}}} \\\\")
        blocks = {}
        for r in rows:
            if r[col] is not None:
                blocks.setdefault((r[0], r[1]), []).append(r)
        for b, ((t, n), br) in enumerate(blocks.items()):
            if b:
                lines.append("\\midrule")
            body = []
            for v in br:
                f, e, y = v[2], v[3], v[col]
                body.append(f"{FLOW_STYLE[f][0]} & " + " & ".join(C.fmt_e(x) for x in y / e)
                            + f" & {_slope_txt(slope(e, y))} \\\\")
            lines += rot_block(rot_lines(t, n), body)
        for i, (t, n) in enumerate(pend):
            if blocks or i:
                lines.append("\\midrule")
            lines += rot_block(rot_lines(t, n), [tbd_row(t, n)])
    lines += ["\\bottomrule", "\\end{tabular}"]
    if pend:
        comments.append(C.TBD_DEF)
    C.write_text(os.path.join(C.TABLES, "tab_stability.tex"), "\n".join(comments + lines) + "\n")
    return "tables/tab_stability.tex"


def refine_constants(d, target, flow):
    """Per-(set, draw, epsilon) constants of one refinement file: TV/epsilon, sup_t L2/epsilon,
    sup_t H2/epsilon (flattened), or None."""
    j = _nidx(d, target, 100)
    if j is None or f"{target}/{flow}/tv" not in d:
        return None
    e = epsilon_of(d)[None, :, None]
    dn = d[f"{target}/{flow}/dnorm2"][j]                      # (S, E, R, K, 3)
    return dict(tv=(d[f"{target}/{flow}/tv"][j] / e).ravel(),
                l2=(np.sqrt(dn[..., 0]).max(-1) / e).ravel(),
                h2=(np.sqrt(dn.sum(-1)).max(-1) / e).ravel())


def write_refine_table() -> str:
    """tab_stability_refine.tex from stability_1d_m<M> (the mesh-refinement check): per
    target, penalty, lam mode and m the constants TV/epsilon, sup_t ||d||_L2/epsilon and
    sup_t ||d||_H2/epsilon, mean [min, max] over sets x draws x epsilon, and the growth
    m = 1024 / 256. Rows of the present m; \\tbd rows only if no file exists."""
    names = {m: f"stability_1d_m{m}" for m in REF_MS}
    files = {m: C.load(nm) for m, nm in names.items()}
    qs = [("tv", "TV$/\\varepsilon$"), ("l2", "$L^2$ field$/\\varepsilon$"), ("h2", "$H^2$ field$/\\varepsilon$")]
    comments = ["% Generated by experiments/report/stability.py; do not edit by hand.",
                C.results_line()] + [C.src_line(names[m]) for m in REF_MS] + [
        "% Mesh-refinement check: N = 100, eps in {1e-6, 1e-5, 1e-4}; constants mean [min, max] over",
        "% sets x draws x eps; field norms sup over the 49 t_k (frame units); growth = mean(1024)/mean(256).",
        "% fixed: lambda(t) of the m = 512 hold-out selection at every m; re-selected: hold-out at m;",
        "% both over the positive candidates 10^j lam_s(t) (lambda = 0 excluded; stored zeros re-selected;",
        f"% smallest evaluated B_2 candidate {LOW_H2[1]} lam_s; a no-op for B_0, whose lowest candidate",
        f"% {LOW_L2} lam_s equals the solve floor).",
        "% First column: target and N, rotated, spanning the block (needs multirow, graphicx)."]
    present = [m for m in REF_MS if files[m] is not None]
    missing = [names[m] for m in REF_MS if files[m] is None]
    if missing and present:
        # a partial set (the smoke runs compute m = 256 only): rows of the present m only
        comments.append("% not in the results directory, rows omitted: "
                        + ", ".join(f"{nm}.npz" for nm in missing))
    elif missing:
        comments.append("% pending (missing) sources, their rows print \\tbd: "
                        + ", ".join(f"{nm}.npz" for nm in missing))
        comments.append(C.TBD_DEF)
    ms = present or REF_MS
    lines = ["\\begin{tabular}{clr" + "c" * len(qs) + "}", "\\toprule",
             " & Estimator & $m$ & " + " & ".join(h for _, h in qs) + " \\\\"]
    for t in ["trimodal", "inhom"]:
        body = []
        for f in REF_FLOWS:
            vals = {m: (refine_constants(files[m], t, f) if files[m] is not None else None)
                    for m in ms}
            if present and not any(v is not None for v in vals.values()):
                continue
            if body:
                body.append("\\addlinespace")
            # the estimator label on two rows ("$k = 0$" over "re-selected $\lambda$"), which
            # narrows its column; a single m keeps it on one row
            label = FLOW_STYLE[f][0].split(", ", 1)
            row_label = (label + [""] * len(ms))[:len(ms)] if len(ms) > 1 else [FLOW_STYLE[f][0]]
            for i, m in enumerate(ms):
                v = vals.get(m)
                cells = []
                for q, _ in qs:
                    if v is None:
                        cells.append(C.tbd() if files[m] is None else "--")
                        continue
                    x = v[q][np.isfinite(v[q])]
                    cells.append(f"{C.fmt_e(x.mean())} [{C.fmt_e(x.min())}, {C.fmt_e(x.max())}]"
                                 if x.size else "--")
                body.append(f"{row_label[i]} & {m} & " + " & ".join(cells)
                            + " \\\\")
            if vals.get(256) is not None and vals.get(1024) is not None:
                g = [np.nanmean(vals[1024][q]) / np.nanmean(vals[256][q]) for q, _ in qs]
                body.append(" & growth & " + " & ".join(f"${x:.2f}$" for x in g) + " \\\\")
                comments.append(f"% {t} {f}: growth 1024/256 TV {g[0]:.2f}, L2 {g[1]:.2f}, H2 {g[2]:.2f}")
        if body:
            # one block per target under a rotated label (N = 100 throughout), \\addlinespace
            # between the estimators
            lines.append("\\midrule")
            lines += rot_block(rot_lines(t, 100), body, body.count("\\addlinespace"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    C.write_text(os.path.join(C.TABLES, "tab_stability_refine.tex"),
                 "\n".join(comments + lines) + "\n")
    return "tables/tab_stability_refine.tex"
