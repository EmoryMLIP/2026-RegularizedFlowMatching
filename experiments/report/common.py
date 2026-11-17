"""Shared helpers of the report stage: the results loader, statistics, the
unresolved rule, LaTeX / .dat / PNG formatting. Paper: every generated table and figure.

Every key access of the report goes through one loader (Results / Store below), so that a
key the compute scripts do not write shows up in one list (Results.absent_keys, printed by
report.py) and so that `report.py --compact-out` can write exactly the keys the report read
(the compact results).

Results directory layout: <dir>/<name>.npz and <dir>/<name>.manifest.json
with the result names of experiments/runs.txt (oned, controls_1d, twod_circle, ...). One directory; no smoke
fallback (smoke results are another directory). A missing file is None for the readers,
and its cells print \\tbd.

Statistics over training sets: mean +- SE with SE = std(ddof=1) / sqrt(k) over the k sets
(NaN entries dropped); paired comparisons use per-set differences. Arrays of a result file
have the leading axes (len(ns), len(sets)) unless a script's docstring says otherwise.
"""

from __future__ import annotations

import json
import math
import os
import struct
import zlib

import numpy as np

from smoothfm import config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_RESULTS = os.path.join(ROOT, "results")
DEFAULT_OUT = os.path.join(ROOT, "experiments", "paper")

# output directories of the current run (report.set_out); read at call time
OUT = DEFAULT_OUT
TABLES = os.path.join(OUT, "tables")
FIGURES = os.path.join(OUT, "figures")

SIZES = tuple(int(n) for n in config.NS)       # training-set sizes N of the tables
DATASETS = ["trimodal", "inhom"]
DATASET_LABEL = {"trimodal": "Trimodal", "inhom": "Inhomogeneous"}


def set_out(out):
    """Point TABLES and FIGURES at out/tables and out/figures."""
    global OUT, TABLES, FIGURES
    OUT = os.path.abspath(out)
    TABLES = os.path.join(OUT, "tables")
    FIGURES = os.path.join(OUT, "figures")


# ------------------------------------------------------------------ loading

class Store:
    """One result file, read-only mapping key -> array. Records every key the report reads
    (`touched`), the keys read as images (`images`, cast to float32 for every reader, so
    that a float32 compact copy gives the same report), and keys asked for but absent."""

    def __init__(self, name, path, owner):
        self.name, self.path, self._owner = name, path, owner
        with np.load(path, allow_pickle=False) as f:
            self._d = {k: f[k] for k in f.files}
        self.touched, self.images = set(), set()

    # Mapping protocol (the readers use `key in d`, d[key], d.get(key))
    def __contains__(self, key):
        ok = key in self._d
        if ok:
            self.touched.add(key)
        else:
            self._owner.absent_keys.add((self.name, key))
        return ok

    def has(self, key):
        """key in d without recording an absence (probing alternative spellings)."""
        ok = key in self._d
        if ok:
            self.touched.add(key)
        return ok

    def __getitem__(self, key):
        if key not in self._d:
            self._owner.absent_keys.add((self.name, key))
            raise KeyError(f"{self.name}.npz has no key {key!r}")
        self.touched.add(key)
        if key in self.images:
            return self._d[key].astype(np.float32)
        return self._d[key]

    def get(self, key, default=None):
        return self[key] if key in self else default

    def keys(self):
        """Every key of the file (not recorded: use only to discover per-job prefixes, then
        read the keys themselves)."""
        return sorted(self._d)

    files = property(keys)

    def __iter__(self):
        return iter(self.keys())

    def image(self, key):
        """d[key] as float32 (density and velocity images; the compact copy stores float32)."""
        self.images.add(key)
        return self[key]

    def compact(self) -> dict:
        """The touched keys, images as float32 (Results.write_compact)."""
        out = {}
        for k in sorted(self.touched):
            v = self._d[k]
            out[k] = v.astype(np.float32) if k in self.images else v
        return out


class Results:
    """The results directory of one report run: load(name) -> Store or None (cached)."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self._files, self._manifests = {}, {}
        self.absent_keys = set()

    def load(self, name):
        if name not in self._files:
            p = os.path.join(self.root, name + ".npz")
            self._files[name] = Store(name, p, self) if os.path.exists(p) else None
        return self._files[name]

    def manifest(self, name):
        if name not in self._manifests:
            p = os.path.join(self.root, name + ".manifest.json")
            m = None
            if os.path.exists(p):
                with open(p) as f:
                    m = json.load(f)
            self._manifests[name] = m
        return self._manifests[name]

    def manifest_names(self):
        """Names of every manifest in the directory (sorted)."""
        if not os.path.isdir(self.root):
            return []
        sfx = ".manifest.json"
        return sorted(f[:-len(sfx)] for f in os.listdir(self.root) if f.endswith(sfx))

    @property
    def smoke(self) -> bool:
        """True if the directory holds manifests and every one records mode 'smoke'."""
        if not hasattr(self, "_smoke"):
            modes = [(self.manifest(n) or {}).get("mode") for n in self.manifest_names()]
            self._smoke = bool(modes) and all(m == "smoke" for m in modes)
        return self._smoke

    def missing(self):
        """Sorted names requested but absent."""
        return sorted(n for n, d in self._files.items() if d is None)

    def label(self) -> str:
        """The results directory as printed in '% results directory' lines: relative to the
        repository root when inside it, else its last two path components."""
        try:
            r = os.path.relpath(self.root, ROOT)
        except ValueError:
            r = None
        if r is None or r.startswith(".."):
            r = os.path.join(*self.root.split(os.sep)[-2:])
        return r

    def write_compact(self, out):
        """out/<name>.npz with the touched keys (compressed) and out/<name>.manifest.json for
        every manifest of the results directory (the cost table reads them)."""
        os.makedirs(out, exist_ok=True)
        for name, d in sorted(self._files.items()):
            if d is None:
                continue
            # a file the report loaded but read nothing from is still written (empty), so
            # that the headers name the same sources
            np.savez_compressed(os.path.join(out, name + ".npz"), **d.compact())
        for name in self.manifest_names():
            src = os.path.join(self.root, name + ".manifest.json")
            with open(src, "rb") as f, open(os.path.join(out, name + ".manifest.json"), "wb") as g:
                g.write(f.read())


RES = None   # the Results of the current run (report.main sets it)


def use(root) -> Results:
    global RES
    RES = Results(root)
    return RES


def load(name):
    """Store of <results>/<name>.npz, or None if missing."""
    return RES.load(name)


def src_line(name) -> str:
    """'% source: <name>.npz' (MISSING if absent); no directory (the '% results directory'
    line names it once)."""
    return f"% source: {name}.npz" + ("" if load(name) is not None else " MISSING")


def results_line() -> str:
    return f"% results directory: {RES.label()}"


def get(d, key, j=None):
    """d[key] (row j of the leading N axis if given), or None if d or the key is missing."""
    if d is None or key not in d:
        return None
    return d[key] if j is None else d[key][j]


def row(d, key, n):
    """d[key][index of n in d['ns']], or None if d, the key or n is missing."""
    j = n_index(d, n)
    return None if j is None else get(d, key, j)


def n_index(d, n):
    """Index of n in d['ns'], or None."""
    if d is None or "ns" not in d:
        return None
    ns = [int(x) for x in np.atleast_1d(d["ns"])]
    return ns.index(n) if n in ns else None


def ns_of(d):
    return [] if d is None or "ns" not in d else [int(x) for x in np.atleast_1d(d["ns"])]


def pick_n(d, n):
    """n if d holds it, else the N of d closest to n in log (ties: the smaller), so that a
    figure that shows one N still draws from a smoke directory with N = 100 only (its header
    names the N used); None without any N."""
    ns = ns_of(d)
    if not ns:
        return None
    if n in ns:
        return n
    return min(ns, key=lambda m: (abs(math.log(m / n)), m))


# ------------------------------------------------------------------ statistics

def mean_se(v):
    """(mean, SE) over the finite entries of v; SE = std(ddof=1)/sqrt(k), NaN if k < 2."""
    if v is None:
        return np.nan, np.nan
    v = np.asarray(v, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
    return float(v.mean()), float(se)


def unresolved(a, se_a, b, se_b, spread=None) -> bool:
    """The difference a - b is unresolved when |a - b| < 2 sqrt(se_a^2 +
    se_b^2) (Monte Carlo SEs of the two estimates; NaN SEs count as 0) or, with spread
    given, when |a - b| < spread (the set-to-set spread). False if a or b is not finite."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    sa = float(se_a) if se_a is not None and np.isfinite(se_a) else 0.0
    sb = float(se_b) if se_b is not None and np.isfinite(se_b) else 0.0
    d = abs(float(a) - float(b))
    if d < 2.0 * math.sqrt(sa * sa + sb * sb):
        return True
    return spread is not None and np.isfinite(spread) and d < float(spread)


def mc_se(se):
    """Monte Carlo SE of a mean over sets from the per-set Monte Carlo SEs se (k,):
    sqrt(sum se_i^2) / k over the finite entries; NaN if none."""
    if se is None:
        return np.nan
    s = np.asarray(se, float).ravel()
    s = s[np.isfinite(s)]
    return float(np.sqrt(np.sum(s * s)) / s.size) if s.size else np.nan


# ------------------------------------------------------------------ regularizer labels
# The paper names the regularizers by their derivative order k and a word; every table and legend takes its label
# from here. Keys are the result-name suffixes of the ablation (h2 is the production
# regularizer, unsuffixed in the result names).
PENALTY_OP = {"l2": 0, "h1": 1, "h2": 2}
PENALTY_WORD = {"l2": "smallness", "h1": "gradient", "h2": "Hessian"}
#: Column header of the regularizer column (tab_refine panel (b), tab_lam_sweep_full).
REGULARIZER_HEAD = "Regularizer"


def penalty_label(reg, short=False) -> str:
    """LaTeX label of a regularizer: 'Hessian ($k = 2$)', or '$k = 2$' where space requires."""
    k = f"$k = {PENALTY_OP[reg]}$"
    return k if short else f"{PENALTY_WORD[reg]} ({k})"


# ------------------------------------------------------------------ formatting

def column_exponent(means) -> int:
    """Common power of ten e of a table column: the smallest finite positive mean m scaled to
    m 10^{-e} lies in [1, 10) after rounding to one decimal. 0 if the column has no finite
    positive mean."""
    m = np.asarray([x for x in means if np.isfinite(x) and x > 0], float)
    if m.size == 0:
        return 0
    lo = float(m.min())
    e = int(np.floor(np.log10(lo)))
    if round(lo / 10.0 ** e, 1) >= 10:
        e += 1
    return e


def fmt_sci(mean, se, e, bold=False) -> str:
    """LaTeX math 'mean \\pm SE' in units of 10^e with one decimal place for both; '--' if
    the mean is missing, the SE omitted when undefined."""
    if not np.isfinite(mean):
        return "--"
    s = 10.0 ** e
    body = f"{mean / s:.1f}" + (fr" \pm {se / s:.1f}" if np.isfinite(se) else "")
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def power_label(e) -> str:
    """Header text of a column in units of 10^e: '($10^{-2}$)', empty for e = 0."""
    return "" if e == 0 else f"($10^{{{e}}}$)"


def fmt_e(x, digits=1) -> str:
    """One number in LaTeX scientific notation, '$3.2 \\cdot 10^{-5}$'; '--' if missing."""
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    mant = x / 10.0 ** e
    if round(abs(mant), digits) >= 10:
        e, mant = e + 1, mant / 10
    return f"${mant:.{digits}f} \\cdot 10^{{{e}}}$"


def write_dat(path, columns: dict, comment: str = ""):
    """Whitespace-separated columns with a header line (pgfplots 'table'); NaN as 'nan'.
    Optional '#' comment lines first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = list(columns)
    cols = [np.asarray(columns[k], float).ravel() for k in names]
    n = max((c.size for c in cols), default=0)
    with open(path, "w") as f:
        for ln in comment.splitlines():
            f.write(f"# {ln}\n")
        f.write(" ".join(names) + "\n")
        for i in range(n):
            f.write(" ".join("nan" if (i >= c.size or not np.isfinite(c[i])) else f"{c[i]:.6g}"
                             for c in cols) + "\n")


def write_text(path, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


# ------------------------------------------------------------------ images (PNG, stdlib only)

# sequential colormap for density images: equally spaced RGB anchors, white at 0 (the same
# anchors define the pgfplots colormap 'densblue' in the figure, so the colorbar matches)
CMAP_ANCHORS = np.array([[255, 255, 255], [198, 219, 239], [107, 174, 214], [33, 113, 181],
                         [8, 48, 107]], float)


def colorize(v, vmax):
    """RGB uint8 image (rows, cols, 3) of v / vmax clipped to [0, 1], by piecewise-linear
    interpolation of CMAP_ANCHORS; NaN maps to white."""
    u = np.clip(np.nan_to_num(np.asarray(v, float) / vmax, nan=0.0), 0.0, 1.0)
    pos = u * (len(CMAP_ANCHORS) - 1)
    k = np.minimum(np.floor(pos).astype(int), len(CMAP_ANCHORS) - 2)
    a = (pos - k)[..., None]
    rgb = (1 - a) * CMAP_ANCHORS[k] + a * CMAP_ANCHORS[k + 1]
    return np.round(rgb).astype(np.uint8)


def pgf_colormap(name="densblue") -> str:
    return (f"colormap={{{name}}}{{"
            + " ".join(f"rgb255=({int(r)},{int(g)},{int(b)})" for r, g, b in CMAP_ANCHORS) + "}")


def write_png(path, rgb):
    """8-bit RGB or RGBA PNG of the array rgb (rows, cols, 3 or 4), top row first (zlib, no
    filter; deterministic bytes)."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    h, w, nc = rgb.shape
    ctype = {3: 2, 4: 6}[nc]

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def grid_image(f):
    """A density on the grid x_i, y_j (ij indexing, x across, y up) as image rows (top row =
    largest y)."""
    return np.asarray(f).T[::-1]


# Rotated block labels (tab_main, tab_refine, tab_stability, tab_stability_refine): the label
# lines stacked in a one-column tabular, rotated by 90 degrees and centred by \multirow on
# the rows of its block. Widths of label lines at \footnotesize (pt, \settowidth in pdflatex,
# article 10pt and siamart251216 alike): the vertical extent of the rotated label.
ROT_LINE_WIDTH_PT = {"Trimodal": 34.0, "Inhom.": 26.2, "Noisy circle": 43.0,
                     "Checkerb.": 36.6, "Two spirals": 42.8, "8 Gaussians": 44.1,
                     "Circle": 21.7, "(1D)": 17.4, "(2D)": 17.4,
                     "$N = 100$": 31.7, "$N = 1000$": 35.9, "$N = 3000$": 35.9}
ROT_ROW_PT = 9.5         # \footnotesize \baselineskip of a 10pt class: one row at stretch 1
ROT_MARGIN_PT = 2.0      # clearance kept free above and below the longest label


def rot_label(lines, nrow, inner_space=0, smash=False, last=False) -> str:
    r"""\multirow over nrow rows of the label lines, rotated by 90 degrees. inner_space: the
    number of \addlinespace (\defaultaddspace each) inside the block; \multirow centres on
    nrow rows of nominal height, so the label is lowered by half that space to sit at the
    centre of the block. smash: wrap the rotated label in \smash. \multirow sets its text in
    a box of nrow rows with unshrinkable glue, so a label taller than that box hangs from the
    top row (Overfull \vbox) instead of being centred; smashed, it is centred on the block
    and overhangs it equally above and below (rot_pad_pt sizes the space for that).
    last: the label goes into the LAST row of its block (\multirow{-nrow}, centred on the
    rows above it, raised by half the inner space), so that the \rowcolor of a shaded row
    further down cannot paint over it (tab_oned_full, tab_twod_full). The lines of the inner tabular
    are then \rowcolor{white}: they would inherit the colour of a shaded label row."""
    sign = 1 if last else -1
    vmove = f"[{sign * 0.5 * inner_space:g}\\defaultaddspace]" if inner_space else ""
    if last:
        nrow = -nrow
        lines = [r"\rowcolor{white}" + ln for ln in lines]
    body = r"\\".join(lines)
    rot = (fr"\rotatebox[origin=c]{{90}}"
           fr"{{\begin{{tabular}}{{@{{}}c@{{}}}}{body}\end{{tabular}}}}")
    if smash:
        rot = fr"\smash{{{rot}}}"
    return fr"\multirow{{{nrow}}}{{*}}{vmove}{{{rot}}}"


def rot_width(lines) -> float:
    """Vertical extent (pt) of a rotated label: its longest line (ROT_LINE_WIDTH_PT)."""
    return max(ROT_LINE_WIDTH_PT[s] for s in lines)


def rot_pad_pt(width_pt, nrow, row_pt=ROT_ROW_PT, margin_pt=ROT_MARGIN_PT) -> float:
    r"""Space (pt, a multiple of 0.5) to add above and below a block of nrow rows so that a
    rotated label of width_pt centred on it keeps margin_pt from the rules around the block;
    0 if the block is tall enough. The rule separations of booktabs are not counted."""
    need = (width_pt + 2.0 * margin_pt - nrow * row_pt) / 2.0
    return 0.0 if need <= 0 else math.ceil(need * 2.0 - 1e-9) / 2.0


TBD = r"\tbd"
TBD_DEF = r"\providecommand{\tbd}{\textit{tbd}}"


def tbd() -> str:
    """Text of a cell whose source file is missing: \\tbd in every mode, so that a report
    from smoke results compiles with the paper's \\tbd macro like a pending full report (a
    present source that lacks the entry still prints '--')."""
    return TBD
