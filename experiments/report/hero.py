"""Hero figure fig_hero.tex from hero_1d.npz (report stage; numpy only).

One row of flow panels (_column): (a) the unregularized flow (exact v_N) and
(b) the flow with the Hessian penalty B_2 and hold-out lambda(t), trimodal target, N = config.HERO_1D['n'], the
training set chosen by compute/hero_1d.py (median hold-out TV). Per column, on one time axis
linear in tau = -log(1 - t) (ticks labelled by t):
  main panel   w(x, t) = (1 - t) v(x, t) = dx/dtau as an image (diverging map centred at 0,
               one symmetric limit for both columns: the larger per-panel CLIP_PCT percentile
               of |w|), K trajectories from the latent quantiles; the panel is Omega.
  zoom panels  the window zoom_window of hero_1d (+-ZOOM_PAD) from t = 0.9 to t_end, between
               the columns, joined to its outline in each main panel (same seeds).
  prior panel  the latent density at t = 0 (first column).
  data panel   at t_end: training histogram and rug, generated density (blue; exact v_N: the
               stopped law rho_N(., t_end) as cell averages, whose spikes run off the axis),
               true density (dashed).
  strip        e(t) (black, log) and lambda(t)/lambda_s (gray, twin axis); 'lambda = 0' for (a).
Colour map: ColorBrewer PuOr without its two darkest classes, linear in sign(w) |w/c|^GAMMA.
Fonts: the figure is built at its estimated native width and the paper scales it to
\\textwidth (PRINT_WIDTH_CM = 13.02 cm, siamart); tick labels must print at >= 7 pt.

hero_1d.npz keys read: s_end, x_img, omega_lo, omega_hi, xf,
xd_edges, hist_edges, x_prior, prior_pdf, frame_s, times and per <d>/n<N>/: set, ys, hist,
pdf_true, lam_s, zoom_window, and per flow (exact, holdout): w (image), traj, traj_s, zoom,
zoom_s, curve (holdout), curve_law (exact), e, lam (holdout), tv, tv_ref.
"""

from __future__ import annotations

import os

import numpy as np

from smoothfm import config

import common as C

NAME = "fig_hero"
GENERATOR = "experiments/report/hero.py"
DATASET, RULE = "trimodal", "holdout"
GAMMA, CLIP_PCT = 0.5, 99.0
PRINT_WIDTH_CM = 13.02                 # siamart \textwidth (5.125 in)
# PuOr (ColorBrewer, 11 classes) without the two darkest at each end; negative w orange
CMAP = np.array([[224, 130, 20], [253, 184, 99], [254, 224, 182], [247, 247, 247],
                 [216, 218, 235], [178, 171, 210], [128, 115, 172]], float)
BLUE = "0072B2"                        # Okabe-Ito blue: the generated density only
T_TICKS = [0.0, 0.9, 0.99, 0.999, 0.9999]

# layout in cm at the native width (the paper scales the whole figure)
PRIOR_W, MAIN_W, DATA_W = 0.8, 3.2, 1.45
COL_GAP, LEFT = 0.6, 1.0
ZOOM_W, ZOOM_GAP, ZOOM_MID = 0.95, 0.2, 0.15
MAIN_H, STRIP_H, STRIP_GAP = 4.0, 1.35, 0.15
MIN_TICK_CM = 0.75          # spacing of labelled t ticks
MIN_END_CM = 1.0            # ... and before the end tick, whose label is longer (0.9999)
ZOOM_PAD = 0.10                        # the zoom's x range: the window +-10%
ZOOM_TICK_PT = 2.2                     # half length of the training-point ticks in the zoom
ROW_TITLES = {"exact": r"(a) exact $v_N$", "holdout": "(b) " + C.penalty_label("h2")}
# fig_hero's axis labels in words (the paper caption no longer spells out the symbols).
# "velocity error" on one line is longer than the strip and runs into the -1 tick of the x
# axis above, so it breaks into two centred lines
LABELS = {"x": r"position $x$", "t": r"time $t$", "e": r"velocity\\error",
               "e_style": "align=center", "lam": r"$\lambda(t)/\lambda_s$",
               "w": r"scaled velocity $(1-t)\,v$"}
ROW_RIGHT = 0.75                       # beyond column (b): the lam axis and its label


def diverging_rgba(u, mask):
    """RGBA of u in [-1, 1] by piecewise-linear interpolation of CMAP; transparent where mask."""
    pos = (np.clip(u, -1, 1) + 1) / 2 * (len(CMAP) - 1)
    k = np.minimum(np.floor(pos).astype(int), len(CMAP) - 2)
    a = (pos - k)[..., None]
    rgb = np.round((1 - a) * CMAP[k] + a * CMAP[k + 1]).astype(np.uint8)
    alpha = np.where(mask, 0, 255).astype(np.uint8)[..., None]
    return np.concatenate([rgb, alpha], axis=-1)


def to_u(w, c, gamma):
    """Colour coordinate sign(w) min(|w|/c, 1)^gamma in [-1, 1]."""
    return np.sign(w) * np.minimum(np.abs(w) / c, 1.0) ** gamma


def log_ticks(lo, hi, nmax=4):
    """Decades between lo and hi, every k-th so that at most nmax remain."""
    a, b = int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi)))
    k = max(1, int(np.ceil((b - a + 1) / nmax)))
    return [10.0 ** e for e in range(a, b + 1, k)]


def _ticks(vals):
    return ",".join(f"{v:.6g}" for v in vals)


def _loglabels(vals):
    return ",".join(f"$10^{{{int(round(np.log10(v)))}}}$" for v in vals)


def crop_to_box(W, x_img, lo, hi):
    """Columns of the images W (tau, x) whose cells lie inside a constant Omega = [lo, hi],
    and the panel's x range: the outer edges of those cells."""
    dx = x_img[1] - x_img[0]
    keep = (x_img - 0.5 * dx >= lo - 1e-12) & (x_img + 0.5 * dx <= hi + 1e-12)
    x_in = x_img[keep]
    return {f: w[:, keep] for f, w in W.items()}, x_in, x_in[0] - 0.5 * dx, x_in[-1] + 0.5 * dx


def placeholder(name, missing):
    """A standalone figure that says which result files are missing (never \\tbd)."""
    notes = [C.results_line()] + [C.src_line(m) for m in missing]
    body = ("\\node[font=\\itshape, text=red!70!black] at (0,0) {missing "
            + ", ".join(f"{m}.npz" for m in missing).replace("_", "\\_") + "};\n")
    path = os.path.join(C.FIGURES, name + ".tex")
    C.write_text(path, _document(name, notes, body))
    return f"figures/{name}.tex"


def row_width(zoom_on):
    """(estimated figure width, column width, gap between the columns), cm."""
    col_w = PRIOR_W + MAIN_W + DATA_W
    mid = 2 * ZOOM_GAP + 2 * ZOOM_W + ZOOM_MID if zoom_on else COL_GAP
    return LEFT + 2 * col_w + mid + ROW_RIGHT, col_w, mid


def _written(sub_rel, sub):
    """Paths relative to C.OUT of every file in the figure's data directory (sorted)."""
    return sorted(f"figures/{sub_rel}/{f}" for f in os.listdir(sub))


# ------------------------------------------------------------------ data files

# Curves are thinned in printed coordinates (cm at native size) before pgfplots reads them: a
# figure with every coordinate exceeds TeX's main memory (5e6 words; each plotted coordinate
# costs on the order of 10^2). TOL_CM is far below the 0.4 pt (0.014 cm) line width.
TOL_CM = 0.004
TRAJ_NMAX = 80              # points per trajectory at most (tolerance raised until it fits)


def rdp_keep(u, v, tol):
    """Ramer-Douglas-Peucker: mask of the points of the polyline (u, v) to keep so that the
    dropped ones lie within tol of the simplified polyline (end points kept)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    keep = np.zeros(u.size, bool)
    keep[[0, -1]] = True
    stack = [(0, u.size - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        du, dv = u[j] - u[i], v[j] - v[i]
        seg = np.hypot(du, dv)
        uu, vv = u[i + 1:j] - u[i], v[i + 1:j] - v[i]
        d = np.abs(du * vv - dv * uu) / seg if seg > 0 else np.hypot(uu, vv)
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return keep


def thin_tol(u, v, tol=TOL_CM, nmax=None):
    """(rdp_keep mask, tolerance used): tol is raised by 1.5x until at most nmax points remain
    (nmax None: no cap, tol unchanged)."""
    keep = rdp_keep(u, v, tol)
    while nmax is not None and keep.sum() > nmax:
        tol *= 1.5
        keep = rdp_keep(u, v, tol)
    return keep, tol


def write_traj(path, s, X, comment, cm_per_s, cm_per_x, nmax=TRAJ_NMAX):
    """Trajectories as one table (blank line between them), each thinned on its own in printed
    coordinates (cm at native size; cm_per_s, cm_per_x) to at most nmax points; the header
    records the tolerance each trajectory used."""
    s, X = np.asarray(s, float), np.asarray(X, float)
    kept = [thin_tol(s * cm_per_s, row * cm_per_x, nmax=nmax) for row in X]
    tols = [t for _, t in kept]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ln in comment.splitlines():
            f.write(f"# {ln}\n")
        f.write(f"# thinned: Ramer-Douglas-Peucker, base tolerance {TOL_CM} cm at native size, "
                f"raised by 1.5x per step until at most {nmax} points per trajectory\n")
        f.write("# tolerance used per trajectory (cm at native size): "
                + " ".join(f"{t:.4g}" for t in tols) + "\n")
        f.write("s x\n")
        for row, (k, _) in zip(X, kept):
            f.write("".join(f"{a:.6g} {b:.6g}\n" for a, b in zip(s[k], row[k])) + "\n")


def thin_columns(y, cols, cm_per_y, cm_per_val, vmax):
    """Union over the columns of the RDP masks of the curves (val, y) in printed coordinates;
    values beyond vmax (off the axis) count as vmax."""
    keep = np.zeros(np.asarray(y).size, bool)
    for c in cols:
        keep |= thin_tol(np.minimum(np.asarray(c, float), vmax) * cm_per_val,
                         np.asarray(y, float) * cm_per_y)[0]
    return keep


def write_prior(sub, h, xlo, xhi, pmax):
    """prior.dat, thinned in the prior panel's printed coordinates."""
    x, pdf = np.asarray(h["x_prior"], float), np.asarray(h["prior_pdf"], float)
    k = thin_columns(x, [pdf], MAIN_H / (xhi - xlo), PRIOR_W / pmax, pmax)
    C.write_dat(os.path.join(sub, "prior.dat"), {"x": x[k], "pdf": pdf[k]},
                f"prior N(0, {float(h['frame_s']):g}^2) (thinned, tolerance {TOL_CM} cm at native size)")


def write_zoom(h, pre, label, flows, sub, xlo, xhi):
    """The zoom data: <flow>_zoom.dat per flow (thinned in the zoom panel's printed scale) and
    zoom_rug.dat; returns the zoom record (window, x range with ZOOM_PAD, start s0,
    magnification, training points inside)."""
    s_end = float(h["s_end"])
    win = np.asarray(h[pre + "zoom_window"], float)
    pad = ZOOM_PAD * (win[1] - win[0])
    zlo, zhi = win[0] - pad, win[1] + pad
    for f in flows:
        zs = np.asarray(h[pre + f + "/zoom_s"], float)
        # thinned in the zoom panel's own printed scale (magnified in x)
        write_traj(os.path.join(sub, f"{f}_zoom.dat"), zs, h[pre + f + "/zoom"],
                   f"{label} {f}: zoom trajectories", ZOOM_W / (s_end - zs[0]),
                   MAIN_H / (zhi - zlo))
    ys = np.asarray(h[pre + "ys"], float)
    yin = ys[(ys >= zlo) & (ys <= zhi)]
    C.write_dat(os.path.join(sub, "zoom_rug.dat"), {"y": yin},
                f"training points in the zoom window [{zlo:.4g}, {zhi:.4g}]")
    return dict(s0=float(np.asarray(h[pre + flows[0] + "/zoom_s"])[0]), lo=zlo, hi=zhi,
                win=tuple(float(v) for v in win), npts=int(yin.size),
                mag=(xhi - xlo) / (zhi - zlo))


# ------------------------------------------------------------------ TikZ

def zoom_tex(f, flow, nm, xz, zoom, s_end, right):
    """(rectangle and connector coordinates inside the main axis, the zoom axis) of one
    zoom panel with its top left corner at (xz, 0) cm. right: the zoom lies right of the
    main panel (the connectors leave the rectangle's right edge), else left of it. A white
    line under the rectangle keeps its edges visible where late-time trajectories run
    parallel to them."""
    sc0 = s_end if right else zoom["s0"]
    zm = (f"\\draw[white, line width=1.6pt, opacity=0.85] (axis cs:{zoom['s0']:.6g},{zoom['lo']:.6g})"
          f" rectangle (axis cs:{s_end:.6g},{zoom['hi']:.6g});\n"
          f"\\draw[hzoom] (axis cs:{zoom['s0']:.6g},{zoom['lo']:.6g}) rectangle "
          f"(axis cs:{s_end:.6g},{zoom['hi']:.6g});\n"
          f"\\coordinate (zhi{nm}) at (axis cs:{sc0:.6g},{zoom['hi']:.6g});\n"
          f"\\coordinate (zlo{nm}) at (axis cs:{sc0:.6g},{zoom['lo']:.6g});\n")
    tick_s = s_end - ZOOM_TICK_PT / 28.4528 / ZOOM_W * (s_end - zoom["s0"])
    ax = (f"\\begin{{axis}}[name=zoom{nm}, at={{({xz:.4g}cm,0cm)}}, anchor=north west,"
          f" scale only axis, width={ZOOM_W}cm, height={MAIN_H}cm,"
          f" xmin={zoom['s0']:.6g}, xmax={s_end:.6g}, ymin={zoom['lo']:.6g}, ymax={zoom['hi']:.6g},"
          " xtick=\\empty, ytick=\\empty, axis line style={hzoom}]\n"
          f"\\addplot[black!85, line width=0.4pt] table[x=s, y=x] {{{f}/{flow}_zoom.dat}};\n"
          + (f"\\addplot[only marks, mark=-, mark size={ZOOM_TICK_PT}pt, black!75,"
             f" mark options={{line width=0.5pt}}] table[x expr={tick_s:.6g}, y=y]"
             f" {{{f}/zoom_rug.dat}};\n" if zoom["npts"] else "")
          + "\\end{axis}\n")
    return zm, ax


def _axes(g, r):
    """The axis options the columns share: the s ticks, the x range, the e and lam ticks."""
    s_end, xlo, xhi = g["s_end"], r["xlo"], r["xhi"]
    sc = MAIN_W / s_end
    tin, last = [], None
    for t in T_TICKS[:-1]:
        st_ = -np.log1p(-t)
        if (last is None or (st_ - last) * sc >= MIN_TICK_CM) and (s_end - st_) * sc >= MIN_END_CM:
            tin.append(t)
            last = st_
    s_ticks = [-np.log1p(-t) for t in tin] + [s_end]
    return dict(tlabels=",".join(f"{t:g}" for t in tin) + f",{g['t_end']:.4g}",
                common_x=(f"xmin=0, xmax={s_end:.6g}, scale only axis, width={MAIN_W}cm, "
                          f"xtick={{{_ticks(s_ticks)}}}"),
                ylim=f"ymin={xlo:.6g}, ymax={xhi:.6g}",
                ytick_e=log_ticks(*g["elim"], nmax=3), ytick_l=log_ticks(*g["llim"], nmax=3))


def _column(f, flow, nm, xm, g, r, ax, comment, title, first, zm="", zoom_tex="",
            lam_col=None, side_extra="", strip_extra="", gen=True):
    """TikZ of one column (main panel with trajectories, prior, data panel, strip) with the
    top left corner of its main panel at (xm, 0) cm. f: the data directory as TeX path; nm:
    the axis name suffix; first: the leftmost column (x axis and e labels); zm, zoom_tex: the
    zoom rectangle (inside the main axis) and the zoom panel (after it); lam_col: the strip
    column of lam/lam_s for a twin axis (None: no twin axis); side_extra, strip_extra: further
    \\addplot lines of the data panel (drawn over the histogram, under the generated density)
    and of the e axis; gen=False omits the generated density of side.dat (the exact v_N
    column draws its stopped law through side_extra instead)."""
    lb = LABELS
    s_end, ylim, common_x = g["s_end"], ax["ylim"], ax["common_x"]
    title = f", title={{{title}}}, htitle" if title else ""
    out = [
        comment
        + f"\\begin{{axis}}[name=main{nm}, at={{({xm:.4g}cm,0cm)}}, anchor=north west,"
        f" {common_x}, height={MAIN_H}cm, {ylim}, axis lines=none, xtick=\\empty,"
        f" ytick=\\empty, enlargelimits=false{title}]\n"
        f"\\addplot graphics[xmin=0, xmax={s_end:.6g}, {ylim}] {{{f}/{flow}_w.png}};\n"
        f"\\addplot[white, line width=1.1pt, opacity=0.55] table[x=s, y=x] {{{f}/{flow}_traj.dat}};\n"
        f"\\addplot[black!85, line width=0.4pt] table[x=s, y=x] {{{f}/{flow}_traj.dat}};\n"
        + zm + "\\end{axis}\n" + zoom_tex,
        f"\\begin{{axis}}[at={{(main{nm}.north west)}}, anchor=north east, hmarg,"
        f" width={PRIOR_W}cm, xmin=0, xmax={g['pmax']:.4g}, x dir=reverse, {ylim}]\n"
        f"\\addplot[black, line width=0.6pt] table[x=pdf, y=x] {{{f}/prior.dat}};\n"
        "\\node[anchor=north west, font=\\small, inner xsep=0pt] at (rel axis cs:0,0.99) {prior};\n"
        "\\end{axis}\n"]
    if first:
        out.append(
            f"\\begin{{axis}}[at={{($(main{nm}.north west)+(-{PRIOR_W}cm,0)$)}},"
            f" anchor=north east, scale only axis, width=0.01cm, height={MAIN_H}cm, {ylim},"
            " xmin=0, xmax=1, axis y line*=left, axis x line=none, ytick={-1,0,1},"
            f" ylabel={{{lb['x']}}}]\n"
            "\\addplot[draw=none] coordinates {(0,0)};\n\\end{axis}\n")
    out.append(
        f"\\begin{{axis}}[at={{(main{nm}.north east)}}, anchor=north west, hmarg,"
        f" width={DATA_W}cm, xmin=0, xmax={r['dmax']:.4g}, {ylim}]\n"
        f"\\addplot[xbar interval, fill=black!30, draw=none] table[x=dens, y=y] {{{f}/hist.dat}};\n"
        + side_extra
        + (f"\\addplot[white, line width=2.0pt] table[x=gen_{flow}, y=x] {{{f}/side.dat}};\n"
           f"\\addplot[hblue, line width=1.0pt] table[x=gen_{flow}, y=x] {{{f}/side.dat}};\n"
           if gen else "") +
        f"\\addplot[black, line width=0.7pt, dashed] table[x=true, y=x] {{{f}/side.dat}};\n"
        f"\\addplot[xcomb, black!75, line width=0.3pt] table[x=len, y=y] {{{f}/rug.dat}};\n"
        # bottom left: the trimodal target has no mass below x = -0.8
        "\\node[anchor=south west, font=\\small, align=left, inner xsep=2pt] at "
        "(rel axis cs:0.04,0.01) {data};\n"
        "\\end{axis}\n")
    yoff = 0.0 - MAIN_H - STRIP_GAP
    e_style = (f"ymode=log, ymin={g['elim'][0]:.3g}, ymax={g['elim'][1]:.3g},"
               f" ytick={{{_ticks(ax['ytick_e'])}}}, yticklabels={{{_loglabels(ax['ytick_e'])}}}")
    if not first:
        e_style += ", yticklabels={}"
    el = f", ylabel={{{lb['e']}}}, ylabel style={{{lb['e_style']}}}" if first else ""
    out.append(
        f"\\begin{{axis}}[name=strip{nm}, at={{({xm:.4g}cm,{yoff:.4g}cm)}}, anchor=north west,"
        f" {common_x}, xticklabels={{{ax['tlabels']}}}, height={STRIP_H}cm, {e_style}{el},"
        f" axis x line*=bottom, axis y line*=left, xlabel={{{lb['t']}}}, clip mode=individual]\n"
        f"\\addplot[black, line width=0.8pt] table[x=s, y=e_{flow}] {{{f}/strip.dat}};\n"
        + strip_extra + "\\end{axis}\n")
    if lam_col:
        out.append(
            f"\\begin{{axis}}[at={{({xm:.4g}cm,{yoff:.4g}cm)}}, anchor=north west, {common_x},"
            f" height={STRIP_H}cm, axis y line*=right, axis x line=none, ymode=log,"
            f" ymin={g['llim'][0]:.3g}, ymax={g['llim'][1]:.3g}, ytick={{{_ticks(ax['ytick_l'])}}},"
            f" yticklabels={{{_loglabels(ax['ytick_l'])}}}, y axis line style={{hlam}},"
            " ytick style={hlam}, yticklabel style={text=hlam},"
            f" ylabel={{{lb['lam']}}}, ylabel style={{text=hlam}}]\n"
            f"\\addplot[hlam, line width=0.8pt] table[x=s, y={lam_col}] {{{f}/strip.dat}};\n"
            "\\end{axis}\n")
    return "".join(out)


def _colorbar_legend(c, gamma, zoom):
    """TikZ of the colour bar under the strip of column (a) and the legend under the strip of
    column (b) (zoom: the zoom record, or None)."""
    lb = LABELS
    wt = [w for w in (0.1, 0.5, 2.0) if w < 0.5 * c] + [c]
    cb_w = sorted([-w for w in wt] + [0.0] + wt)
    cb_u = [float(to_u(np.array(w), c, gamma)) for w in cb_w]
    cb_lab = ",".join(f"${w:.2g}$" for w in cb_w)
    cmap = "colormap={heropuor}{" + " ".join(
        f"rgb255=({int(r)},{int(gg)},{int(b)})" for r, gg, b in CMAP) + "}"
    out = []
    out.append(
        "\\begin{axis}[at={($(strip00.south west)+(0.05cm,-1.05cm)$)}, anchor=north west,"
        f" hide axis, scale only axis, width=4.2cm, height=0.01cm, {cmap}, point meta min=-1,"
        " point meta max=1, colorbar horizontal, colorbar style={at={(0,0)}, anchor=north west,"
        f" width=4.2cm, height=0.22cm, axis line style={{draw=none}}, xtick={{{_ticks(cb_u)}}},"
        f" xticklabels={{{cb_lab}}}, xticklabel style={{font=\\small}},"
        f" xlabel={{{lb['w']}}}, xlabel style={{font=\\small, yshift=3pt}}}}]\n"
        "\\addplot[draw=none] coordinates {(0,0)};\n\\end{axis}\n")

    def sym_line(sty):
        return f"\\draw[{sty}] (0,0) -- (0.55cm,0);"

    data_sym = ("\\fill[black!30] (0,-0.09cm) rectangle (0.55cm,0.09cm);"
                "\\foreach \\u in {0.06,0.17,0.24,0.38,0.47} "
                "\\draw[black!75, line width=0.3pt] (\\u cm,-0.09cm) -- ++(0,0.18cm);")
    leg = [[(sym_line("black!85, line width=0.4pt"), "trajectories"),
            (sym_line("black, line width=0.7pt, dashed"), "true density"),
            (sym_line("hblue, line width=1.0pt"), "generated density")],
           [(data_sym, "training data")]
           + ([("\\draw[hzoom] (0.05cm,-0.12cm) rectangle (0.5cm,0.12cm);",
                f"zoom: $t \\geq {-np.expm1(-zoom['s0']):.2g}$")] if zoom else [])]
    out.append("\\begin{scope}[font=\\small]\n"
               f"\\coordinate (L) at ($(strip01.south west)+({-3.3 if zoom else -1.05}cm,-1.05cm)$);\n")
    for col, items in enumerate(leg):
        for row, (sym, lab) in enumerate(items):
            p = f"($(L)+({col * 3.75:.2f}cm,{-row * 0.40:.2f}cm)$)"
            out.append(f"\\begin{{scope}}[shift={{{p}}}]{sym}\\end{{scope}}\n"
                       f"\\node[anchor=west] at ($(L)+({col * 3.75 + 0.6:.2f}cm,{-row * 0.40:.2f}cm)$) {{{lab}}};\n")
    out.append("\\end{scope}\n")
    return "".join(out)


def _document(name, notes, body):
    """The standalone document around a figure body."""
    return (f"% Generated by {GENERATOR}; do not edit by hand.\n"
            f"% Standalone: pdflatex {name}.tex in this directory. In the paper:\n"
            "% \\usepackage{standalone} (strips this preamble) and\n"
            f"% \\newcommand{{\\figdir}}{{figures}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{figures/{name}.tex}}}}\n"
            + "\n".join(notes) + "\n"
            "\\documentclass{standalone}\n"
            "\\usepackage{pgfplots}\n"
            "\\usetikzlibrary{calc}\n\\pgfplotsset{compat=1.18}\n"
            "\\providecommand{\\figdir}{.}\n"
            "\\begin{document}\n\\begin{tikzpicture}\n"
            f"\\definecolor{{hblue}}{{HTML}}{{{BLUE}}}\\colorlet{{hlam}}{{black!50}}\n"
            "\\tikzset{hzoom/.style={black!70, line width=0.5pt}, hconn/.style={black!55, line width=0.3pt}}\n"
            "\\pgfplotsset{every axis/.append style={tick label style={font=\\small},"
            " label style={font=\\normalsize}, tick align=outside, major tick length=2pt},"
            " htitle/.style={title style={font=\\normalsize, anchor=base, at={(0.5,1)}, yshift=4pt}},"
            f" hmarg/.style={{scale only axis, height={MAIN_H}cm, hide axis, clip mode=individual}}}}\n"
            + body + "\\end{tikzpicture}\n\\end{document}\n")


# ------------------------------------------------------------------ fig_hero

def build():
    """fig_hero.tex and fig_hero/*; returns the written paths relative to C.OUT."""
    h = C.load("hero_1d")
    if h is None:
        return [placeholder(NAME, ["hero_1d"])]
    d, n = DATASET, int(config.HERO_1D["n"])
    pre = f"{d}/n{n}/"
    flows = ["exact", RULE]
    if pre + "set" not in h or pre + RULE + "/w" not in h or pre + "exact/w" not in h:
        return [placeholder(NAME, ["hero_1d"])]
    seed = int(h[pre + "set"])
    zoom_on = all(pre + f + "/zoom" in h for f in flows)
    width, col_w, mid = row_width(zoom_on)
    scale = PRINT_WIDTH_CM / width
    if 9.0 * scale < 7.0:
        raise SystemExit(f"fig_hero: printed tick labels {9 * scale:.1f} pt < 7 pt")
    sub = os.path.join(C.FIGURES, NAME)
    f = rf"\figdir/{NAME}"
    label = f"{d} N={n} set={seed}"
    om_lo, om_hi = np.asarray(h["omega_lo"], float), np.asarray(h["omega_hi"], float)
    W, _, xlo, xhi = crop_to_box({fl: np.asarray(h.image(pre + fl + "/w"), float) for fl in flows},
                                 np.asarray(h["x_img"], float), om_lo[0], om_hi[0])
    clip = {fl: float(np.percentile(np.abs(W[fl]), CLIP_PCT)) for fl in flows}
    c = max(clip.values())
    s_end = float(h["s_end"])
    nomask = np.zeros(W[flows[0]].T.shape, bool)
    for fl in flows:
        C.write_png(os.path.join(sub, f"{fl}_w.png"),
                    diverging_rgba(to_u(W[fl], c, GAMMA).T[::-1], nomask))
        write_traj(os.path.join(sub, f"{fl}_traj.dat"), h[pre + fl + "/traj_s"],
                      h[pre + fl + "/traj"], f"{label} {fl}: trajectories x(s) from the latent "
                      "quantiles, original coordinates", MAIN_W / s_end, MAIN_H / (xhi - xlo))
    xf = np.asarray(h["xf"], float)
    true = np.asarray(h[pre + "pdf_true"], float)
    gen = np.asarray(h[pre + RULE + "/curve"], float)
    dmax = 1.25 * float(max(np.max(true), np.max(gen)))
    cm_y, cm_v = MAIN_H / (xhi - xlo), DATA_W / dmax
    keep = thin_columns(xf, [true, gen], cm_y, cm_v, 1.05 * dmax)
    C.write_dat(os.path.join(sub, "side.dat"),
                {"x": xf[keep], "true": true[keep], f"gen_{RULE}": gen[keep]},
                f"{label}: true density, generated density of the H2 {RULE} flow; rows thinned "
                f"(tolerance {TOL_CM} cm at native size, union over the curves)")
    xd = np.asarray(h["xd_edges"], float)
    xc = 0.5 * (xd[1:] + xd[:-1])
    law = np.minimum(np.asarray(h[pre + "exact/curve_law"], float), 4 * dmax)
    kl = thin_columns(xc, [law], cm_y, cm_v, 1.05 * dmax)
    C.write_dat(os.path.join(sub, "exact_gen.dat"), {"x": xc[kl], "gen": law[kl]},
                f"{label}: stopped law rho_N(., t_end) of the exact v_N flow, cell averages "
                f"(dx = {xd[1] - xd[0]:.4g}), clipped at {4 * dmax:.4g}; the axis ends at "
                f"{dmax:.4g}; rows thinned (tolerance {TOL_CM} cm at native size)")
    ys = np.asarray(h[pre + "ys"], float)
    edges = np.asarray(h["hist_edges"], float)
    dens = np.asarray(h[pre + "hist"], float) / (ys.size * np.diff(edges))
    C.write_dat(os.path.join(sub, "hist.dat"), {"y": edges, "dens": np.r_[dens, 0.0]},
                f"training-set histogram, density units, {label}")
    C.write_dat(os.path.join(sub, "rug.dat"), {"y": ys, "len": np.full(ys.size, 0.06 * dmax)},
                "training points")
    pmax = 1.12 * float(np.max(np.asarray(h["prior_pdf"], float)))
    write_prior(sub, h, xlo, xhi, pmax)
    st = -np.log1p(-np.asarray(h["times"], float))
    e = {fl: np.asarray(h[pre + fl + "/e"], float) for fl in flows}
    lam = np.asarray(h[pre + RULE + "/lam"], float) / np.asarray(h[pre + "lam_s"], float)
    C.write_dat(os.path.join(sub, "strip.dat"),
                {"s": st, **{f"e_{fl}": e[fl] for fl in flows}, "lam": lam},
                f"e(t) (oned_eval.npz) and lam/lam_s (oned.npz, oracle lam_s) at the 49 "
                f"selection times, {label}, flow {RULE}")
    ev = np.concatenate(list(e.values()))
    ev = ev[np.isfinite(ev) & (ev > 0)]
    lv = lam[np.isfinite(lam) & (lam > 0)]
    g = dict(width=width, col_w=col_w, scale=scale, s_end=s_end, c=c, pmax=pmax,
             elim=(10 ** np.floor(np.log10(ev.min())), 10 ** np.ceil(np.log10(ev.max()))),
             llim=(10 ** np.floor(np.log10(lv.min())), 10 ** np.ceil(np.log10(lv.max()))),
             t_end=-np.expm1(-s_end))
    r = dict(xlo=xlo, xhi=xhi, dmax=dmax)
    ax = _axes(g, r)
    zoom = write_zoom(h, pre, label, flows, sub, xlo, xhi) if zoom_on else None
    tv = {fl: float(h[pre + fl + "/tv"]) for fl in flows}
    out = []
    for ci, fl in enumerate(flows):
        first = ci == 0
        nm = f"0{ci}"
        xm = LEFT + ci * (col_w + mid) + PRIOR_W
        zm = ztex = ""
        if zoom:
            xz = LEFT + col_w + ZOOM_GAP + ci * (ZOOM_W + ZOOM_MID)
            zm, ztex = zoom_tex(f, fl, nm, xz, zoom, s_end, first)
        if fl == "exact":
            extra = dict(gen=False, side_extra=(
                f"\\addplot[hblue, line width=0.4pt] table[x=gen, y=x] {{{f}/exact_gen.dat}};\n"),
                strip_extra=("\\node[text=hlam, font=\\small, anchor=east] at "
                             "(rel axis cs:0.97,0.25) {$\\lambda = 0$};\n"))
        else:
            extra = dict(lam_col="lam")
        out.append(_column(
            f, fl, nm, xm, g, r, ax, f"% column {ci + 1}: {fl}\n",
            f"{ROW_TITLES[fl]}, TV {tv[fl]:.3f}", first, zm, ztex, **extra))
    if zoom:
        out.append("\\draw[hconn] (zhi00) -- (zoom00.north west); "
                   "\\draw[hconn] (zlo00) -- (zoom00.south west);\n"
                   "\\draw[hconn] (zhi01) -- (zoom01.north east); "
                   "\\draw[hconn] (zlo01) -- (zoom01.south east);\n"
                   "\\node[font=\\small, anchor=north, black!70] at "
                   f"($(zoom00.south east)!0.5!(zoom01.south west)$) {{${zoom['mag']:.0f}\\times$ in $x$}};\n")
    out.append(_colorbar_legend(c, GAMMA, zoom))
    late = lam[np.asarray(h["times"], float) >= 1 - 1e-2 - 1e-12]
    notes = [
        C.results_line(), C.src_line("hero_1d"),
        f"% dataset {d}, N = {n}, training set {seed}",
        "% layout row: one row of flow panels (hero._column), (a) exact v_N, (b) B_2 "
        f"{RULE}; zoom {'on' if zoom else 'off'}",
        f"% options: gamma {GAMMA}, clip percentile {CLIP_PCT}, print width {PRINT_WIDTH_CM} cm",
        f"% colour limit +-{c:.4g} (the larger per-panel {CLIP_PCT:g}th percentile of |w|: "
        + ", ".join(f"{fl} {v:.4g}" for fl, v in clip.items())
        + f"); colour coordinate sign(w) (|w|/c)^{GAMMA:g}",
        f"% TV: exact v_N {tv['exact']:.4f} (tv_iid; ref {float(h[pre + 'exact/tv_ref']):.4f}), "
        f"{RULE} {tv[RULE]:.4f} (tv_pos; ref {float(h[pre + RULE + '/tv_ref']):.4f}); data "
        f"axis ends at {dmax:.4g} (1.25 x the largest true or penalized density)",
        f"% strip axes: e(t) {g['elim'][0]:.0e} .. {g['elim'][1]:.0e}, lam/lam_s {g['llim'][0]:.0e} "
        f".. {g['llim'][1]:.0e}; late time (1 - t <= 1e-2) lam/lam_s "
        + (f"{late.min():.3g} .. {late.max():.3g}" if late.size else "--"),
        f"% figure width {width:.2f} cm (estimate); printed font sizes at {PRINT_WIDTH_CM} cm: "
        f"ticks {9 * scale:.1f} pt, labels {10 * scale:.1f} pt",
    ] + ([f"% zoom: window [{zoom['win'][0]:.4g}, {zoom['win'][1]:.4g}], t from "
          f"{-np.expm1(-zoom['s0']):.4g}, magnification {zoom['mag']:.1f} in x, {zoom['npts']} "
          "training points"] if zoom else [])
    tex = _document(NAME, notes, "".join(out))
    C.write_text(os.path.join(C.FIGURES, NAME + ".tex"), tex)
    return [f"figures/{NAME}.tex"] + _written(NAME, sub)
