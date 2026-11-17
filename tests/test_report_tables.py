"""Report stage, experiments/report/common.py and tables.py: regularizer labels (the paper's
notation), the \\arraystretch of tab_main, and the hold-out constant row of tab_main at
every N (lam_sweep_<T> at N = config.LAM_SWEEP['n'], lam_sweep_<T>_n<N> otherwise).

Unit level: synthetic lam_sweep files in a temporary results directory, no compute."""

from __future__ import annotations

import re

import numpy as np
import pytest

from release_helpers import REPORT, load_isolated

pytestmark = pytest.mark.report

C, T = load_isolated(REPORT, "tables", "rep_tables")
LS_N = T.LS_N
SIZES = list(C.SIZES)


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("reg,k,word", [("l2", 0, "smallness"), ("h1", 1, "gradient"),
                                        ("h2", 2, "Hessian")])
def test_penalty_label_names_the_order(reg, k, word):
    assert C.penalty_label(reg) == f"{word} ($k = {k}$)"
    assert C.penalty_label(reg, short=True) == f"$k = {k}$"


def test_penalty_labels_drop_the_operator_and_the_word_penalty():
    for reg in ("l2", "h1", "h2"):
        for short in (False, True):
            lab = C.penalty_label(reg, short=short)
            assert "B_" not in lab and "penalt" not in lab.lower(), lab
    assert C.REGULARIZER_HEAD == "Regularizer"


def test_tab_lam_sweep_all_header_says_regularizer(results):
    assert T.LS_HEAD[2].startswith("Regularizer &")


# ---------------------------------------------------------------- rotated labels

def test_stretch_holds_the_longest_label_with_margin():
    s = T.main_rot_stretch()
    block = T.MAIN_BLOCK_ROWS * T.MAIN_ROW_PT * s
    need = max(T.MAIN_ROT_WIDTH_PT.values()) + 2 * T.MAIN_ROT_MARGIN_PT
    assert block >= need
    # the smallest such multiple of 0.05: one step less does not fit
    assert T.MAIN_BLOCK_ROWS * T.MAIN_ROW_PT * (s - 0.05) < need or s == 1.0
    assert abs(s / 0.05 - round(s / 0.05)) < 1e-9


def test_stretch_of_the_measured_widths():
    # 8 Gaussians, 44.1 pt, in 4 rows of 9.5 pt with 2 pt above and below: 48.1 / 38 -> 1.30
    assert T.main_rot_stretch() == pytest.approx(1.30)


@pytest.mark.parametrize("w,expected", [(30.0, 1.0), (34.0, 1.0), (38.0, 1.15),
                                        (57.0, 1.65)])
def test_stretch_rounds_up_and_never_shrinks(w, expected):
    assert T.main_rot_stretch({"x": w}) == pytest.approx(expected)


def test_every_target_has_a_measured_width():
    assert set(T.MAIN_ROT_WIDTH_PT) == {c for c, _ in T.MAIN_CASES}


@pytest.mark.parametrize("w,nrow,expected", [(44.1, 3, 10.0), (36.6, 3, 6.5), (34.0, 4, 0.0),
                                             (35.9, 4, 1.0), (44.1, 9, 0.0)])
def test_rot_pad_keeps_the_margin_around_a_short_block(w, nrow, expected):
    # (w + 2 margin - nrow rows) / 2, rounded up to 0.5 pt, 0 when the block is tall enough
    pad = C.rot_pad_pt(w, nrow)
    assert pad == expected
    assert nrow * C.ROT_ROW_PT + 2 * pad >= w + 2 * C.ROT_MARGIN_PT


def test_rot_label_smash_and_vmove():
    lab = C.rot_label(["Trimodal", "(1D)"], 12, inner_space=2, smash=True)
    assert lab.startswith(r"\multirow{12}{*}[-1\defaultaddspace]{\smash{\rotatebox[origin=c]{90}")
    assert r"Trimodal\\(1D)" in lab
    # tab_main keeps its label without \smash and without the vertical move
    assert T.main_rot_label("trimodal") == (
        r"\multirow{4}{*}{\rotatebox[origin=c]{90}{\begin{tabular}{@{}c@{}}"
        r"Trimodal\\(1D)\end{tabular}}}")


def test_every_rotated_line_has_a_measured_width():
    for case in T.MAIN_ROT_LABEL:
        C.rot_width(T.rot_lines(case))


# ---------------------------------------------------------------- tab_refine_1d / _2d

def _npz(path, ns, **arrays):
    np.savez(path, ns=np.array(ns), sets=np.arange(2), **arrays)


def _vals(v):
    return np.array([[v, v + 0.01]] * 3)


def _complete_refine(root, cases=("trimodal", "inhom", "circle", "checkerboard", "spirals",
                                  "gauss8")):
    """Every file tab_refine_1d / _2d reads, every N, every m, every penalty."""
    ns = list(SIZES)
    one = [c for c in cases if c in C.DATASETS]
    two = [c for c in cases if c not in C.DATASETS]
    for reg in ("h2", "h1", "l2"):
        sfx = "" if reg == "h2" else f"_{reg}"
        if one:
            d = {f"{c}/ms": np.array(T.REFINE_M[1]) for c in one}
            d.update({f"{c}/m{m}/holdout/{f}": _vals(0.1) for c in one
                      for m in T.REFINE_M[1] for f in ("tv_pos", "tv_pos_se")})
            _npz(root / f"controls_1d{sfx}.npz", ns, **d)
        for c in two:
            d = {f"{c}/ms": np.array(T.REFINE_M[2])}
            d.update({f"{c}/m{m}/holdout/{f}": _vals(0.2) for m in T.REFINE_M[2]
                      for f in ("tv_pos", "tv_pos_se")})
            _npz(root / f"controls_{c}{sfx}.npz", ns, **d)
            m = T.REFINE_M_EXTRA
            _npz(root / f"controls_{c}{sfx}_m{m}.npz", ns,
                 **{f"{c}/ms": np.array([m]), f"{c}/m{m}/holdout/tv_pos": _vals(0.3),
                    f"{c}/m{m}/holdout/tv_pos_se": _vals(0.001)})
    if one:
        _npz(root / "controls_iid_1d.npz", ns,
             **{f"{c}/m{m}/lam0/{f}": _vals(0.9) for c in one for m in T.REFINE_M[1]
                for f in ("tv_iid", "tv_iid_se")})
    for c in two:
        _npz(root / f"controls_iid_{c}.npz", ns,
             **{f"{c}/m{m}/lam0/{f}": _vals(0.9) for m in T.REFINE_M[2]
                for f in ("tv_iid", "tv_iid_se")})
        m = T.REFINE_M_EXTRA
        _npz(root / f"controls_iid_{c}_m{m}.npz", ns,
             **{f"{c}/ms": np.array([m]), f"{c}/m{m}/lam0/tv_iid": _vals(0.95),
                f"{c}/m{m}/lam0/tv_iid_se": _vals(0.001)})


HEAD_RE = re.compile(r"\\multicolumn\{(\d+)\}\{l\}\{\\emph\{(.+?)\}\}")


LAM0_SFX, REG_SFX = r", $\lambda \to 0$", r", $\nabla^k$ regularizer"


def _pair_rows(tex):
    """[(case name, kind, m, cells)] of a side-by-side refine table: per pair two in-line heading rows, '<Name>, $\\lambda \\to 0$' and
    '<Name>, $\\nabla^k$ regularizer', each naming the left and right target, then rows
    k | m | 3 N cells per half. kind is KS[0] in the lambda -> 0 block (its k column is
    empty) and the carried k ($2$, $1$, $0$, printed on the first row of its group) in the
    regularizer block."""
    names, ks, rows = [None, None], ["", ""], []
    for ln in tex.splitlines():
        if ln.startswith("%") or "&" not in ln or not ln.rstrip().endswith("\\\\"):
            continue
        heads = HEAD_RE.findall(ln)
        if heads:
            assert [int(w) for w, _ in heads] == [5, 5], ln
            for h, (_, head) in enumerate(heads):
                lam0 = head.endswith(LAM0_SFX)
                assert lam0 or head.endswith(REG_SFX), head
                names[h] = head.removesuffix(LAM0_SFX if lam0 else REG_SFX)
                ks[h] = KS[0] if lam0 else ""
            continue
        cs = [c.strip() for c in ln.rstrip().removesuffix("\\\\").split("&")]
        assert len(cs) == 10, ln
        for h, part in enumerate((cs[:5], cs[5:])):
            if part[0]:
                assert ks[h] != KS[0], ("k printed in the lambda -> 0 block", ln)
                ks[h] = part[0]
            if part[1].isdigit():
                rows.append((names[h], ks[h], int(part[1]), part[2:]))
    return rows


# row kinds of a refine target: lambda -> 0 and k = 2, 1, 0; table order: the regularizer
# block (k = 2, 1, 0), then the lambda -> 0 block
KS = [r"$\lambda \to 0$", "$2$", "$1$", "$0$"]
KS_ORDER = KS[1:] + KS[:1]


@pytest.mark.parametrize("which,pairs,ms", [
    ("1d", [("Trimodal", "Inhomogeneous")], [256, 512, 1024]),
    ("2d", [("Noisy circle", "Checkerboard"), ("Two spirals", "8 Gaussians")],
     [64, 128, 256, 512])])
def test_refine_tables_complete_inputs(results, which, pairs, ms):
    _complete_refine(results)
    tex = getattr(T, f"tab_refine_{which}")(T.Sources())
    body = "\n".join(ln for ln in tex.splitlines() if not ln.startswith("%"))
    assert r"\rotatebox" not in body and "(a)" not in body and "(b)" not in body
    # targets as in-line headers, two side by side, a lambda -> 0 and a regularizer heading
    assert [tuple(h for _, h in HEAD_RE.findall(ln)) for ln in body.splitlines()
            if HEAD_RE.search(ln)] == [tuple(c + sfx for c in p) for p in pairs
                                       for sfx in (REG_SFX, LAM0_SFX)]
    rows = _pair_rows(tex)
    for case in [c for p in pairs for c in p]:
        got = [(k, m) for c, k, m, _ in rows if c == case]
        assert got == [(k, m) for k in KS_ORDER for m in ms], (case, got)
    for case, k, m, cells in rows:
        assert len(cells) == len(SIZES)
        for c in cells:
            assert c and c not in ("--", r"\tbd") and re.search(r"\d", c), (case, k, m, c)
    assert "MISSING" not in tex and "pending" not in tex
    # k only in the first row of its group; empty in the lambda -> 0 block (its heading
    # names it)
    lines = [ln for ln in body.splitlines() if "&" in ln and not HEAD_RE.search(ln)]
    data = [ln.split("&")[0].strip() for ln in lines if ln.split("&")[1].strip().isdigit()]
    per_target = len(KS) * len(ms)
    for j in range(0, len(data), per_target):
        block = data[j:j + per_target]
        assert block == [k if i == 0 and k != KS[0] else "" for k in KS_ORDER
                         for i in range(len(ms))], block
    assert "iid latents" in tex
    spec = next(ln for ln in body.splitlines() if ln.startswith(r"\begin{tabular}"))
    assert spec.count("ll" + "c" * len(SIZES)) == 2


def test_refine_2d_reads_the_extra_resolution_of_every_target(results):
    _complete_refine(results)
    rows = _pair_rows(T.tab_refine_2d(T.Sources()))
    for case in ("Noisy circle", "Checkerboard", "Two spirals", "8 Gaussians"):
        a = [cells for c, k, m, cells in rows if c == case and m == 512 and k == KS[0]]
        assert a == [["$95.5 \\pm 0.5$"] * 3], (case, a)
        b = [cells for c, k, m, cells in rows if c == case and m == 512 and k != KS[0]]
        assert b == [["$30.5 \\pm 0.5$"] * 3] * 3, (case, b)


def test_refine_tables_without_inputs_print_tbd(results):
    """Missing sources print \\tbd, and so does a present source without an N that config
    requires (M3, per cell; config.CONTROLS_M_EXTRA asks every N of the circle k = 0 row)."""
    rows = _pair_rows(T.tab_refine_2d(T.Sources()))
    assert rows and all(c == r"\tbd" for *_, cells in rows for c in cells)
    _complete_refine(results, cases=("circle",))
    m = T.REFINE_M_EXTRA
    _npz(results / f"controls_circle_l2_m{m}.npz", [1000],
         **{"circle/ms": np.array([m]), f"circle/m{m}/holdout/tv_pos": np.array([[0.3, 0.3]]),
            f"circle/m{m}/holdout/tv_pos_se": np.array([[0.001, 0.001]])})
    C.use(results)  # a new report run: the loader caches the absent files of the first one
    rows = _pair_rows(T.tab_refine_2d(T.Sources()))
    l2 = [cells for c, k, mm, cells in rows if c == "Noisy circle" and k == "$0$" and mm == m]
    assert len(l2) == 1 and l2[0][0] == r"\tbd" and l2[0][2] == r"\tbd" and "30.0" in l2[0][1]


# ---------------------------------------------------------------- tab_twod_full

def _twod_blocks(tex):
    """{target name: [(row label, [7 - 1 cells after expanding \\multicolumn{2} spans])]}."""
    blocks, cur = {}, None
    for ln in tex.splitlines():
        if ln.startswith("%") or "&" not in ln and not HEAD_RE.search(ln):
            continue
        h = HEAD_RE.search(ln)
        if h and ln.strip().startswith(r"\multicolumn{7}{l}"):
            cur = blocks.setdefault(h.group(2), [])
            continue
        if cur is None or not ln.rstrip().endswith("\\\\"):
            continue
        cs = [c.strip() for c in ln.rstrip().removesuffix("\\\\").split("&")]
        cells = []
        for c in cs[1:]:
            sp = re.fullmatch(r"\\multicolumn\{2\}\{c\}\{(.*)\}", c)
            cells += ["<span>" + sp.group(1)] * 2 if sp else [c]
        assert len(cells) == 2 * len(SIZES), ln
        cur.append((cs[0], cells))
    return blocks


def test_tab_twod_one_row_per_method(results):
    tex = T.tab_twod_full(T.Sources())
    body = "\n".join(ln for ln in tex.splitlines() if not ln.startswith("%"))
    assert r"\rotatebox" not in body and r"\rowcolor" not in body
    spec = next(ln for ln in body.splitlines() if ln.startswith(r"\begin{tabular}"))
    assert spec == r"\begin{tabular}{@{}l" + "cc" * len(SIZES) + "@{}}"
    for n in SIZES:
        assert fr"\multicolumn{{2}}{{c}}{{$N = {n}$}}" in body
    head = [ln for ln in body.splitlines() if "data-driven & oracle" in ln]
    assert len(head) == 1 and head[0].count("data-driven & oracle") == len(SIZES)
    blocks = _twod_blocks(tex)
    assert list(blocks) == ["Noisy circle", "Checkerboard", "Two spirals", "8 Gaussians"]
    base = ["FM", "KDE", "Abramson", "4th-order KDE", "4th-order product KDE"]
    aniso = ["aniso. KDE", "4th-order aniso. KDE"]
    # the RK4 step count of the exact field is in a % line for the caption, not in the label
    tail = [r"exact $v_N$", r"FM, $\lambda \to 0$, $m = 128$"]
    for name, rows in blocks.items():
        labs = [lab for lab, _ in rows]
        exp = (base + (aniso if name in ("Noisy circle", "Two spirals") else [])
               + (["Gaussian mixture, EM"] if name == "8 Gaussians" else []) + tail)
        assert labs == exp, (name, labs)
        for lab, (_, cells) in zip(labs, rows):
            spans = [c.startswith("<span>") for c in cells]
            assert all(spans) == (lab in exp[-2:] or lab.startswith("Gaussian mixture")), lab
            assert not any(spans) or all(spans)
    assert "exact tangent" in tex and "LSCV" in tex  # the classification comment


def test_tab_twod_shades_cells_not_rows(results):
    tex = T.tab_twod_full(T.Sources())
    body = "\n".join(ln for ln in tex.splitlines() if not ln.startswith("%"))
    assert r"\cellcolor{black!8}" in body
    # FM: both cells of every N shaded (oracle and hold-out are tab:main rows)
    for name, rows in _twod_blocks(tex).items():
        fm = dict(rows)["FM"]
        assert all(c.startswith(r"\cellcolor{black!8}") for c in fm), (name, fm)
        lam0 = [cells for lab, cells in rows if r"\to 0" in lab][0]
        assert all(r"\cellcolor{black!8}" in c for c in lam0), (name, lam0)


def test_tab_oned_keeps_rotated_labels_with_white_label_cells(results):
    """tab_oned_full has rotated labels (label in the last row of its block,
    white label cell beside shaded rows)."""
    tex = T.tab_oned_full(T.Sources())
    lines = [ln for ln in tex.splitlines() if not ln.startswith("%")]
    assert r"\multicolumn" not in "\n".join(lines)
    shaded = [lines[i + 1] for i, ln in enumerate(lines) if ln == T.SHADE]
    assert shaded and all(ln.startswith(r"\cellcolor{white}") for ln in shaded)
    labs = [ln for ln in lines if r"\rotatebox" in ln]
    assert len(labs) == 2 and all(re.search(r"\\multirow\{-\d+\}", ln) for ln in labs)
    assert all(lines[lines.index(ln) + 1] in (r"\midrule", r"\bottomrule") for ln in labs)


# ---------------------------------------------------------------- hold-out constant row

def _sweep(path, target, n, sel_tv, ref_tv):
    """A lam_sweep npz with the keys lam_sweep_values reads: 3 constants, selection index 1
    in every set, so the hold-out constant TV of set i is sel_tv[i]."""
    S = len(sel_tv)
    const = np.stack([np.full(S, 0.9), np.asarray(sel_tv, float), np.full(S, 0.8)], 1)
    np.savez(path, n=n, target=target, reg="h2",
             **{f"{target}/scale/s": np.array([1.0]),
                f"{target}/scale/tv_pos": np.asarray(ref_tv, float)[:, None],
                f"{target}/const/c": np.array([0.1, 1.0, 10.0]),
                f"{target}/const/c0": 1.0,
                f"{target}/const/tv_pos": const,
                f"{target}/const/tv_pos_se": np.full(const.shape, 1e-3),
                f"{target}/const/sel_idx": np.ones(S, int),
                f"{target}/holdout_ref/tv_pos": np.asarray(ref_tv, float)})


@pytest.fixture
def results(tmp_path):
    C.use(tmp_path)
    return tmp_path


def test_result_names():
    assert T.ls_name("circle", "h2") == "lam_sweep_circle"
    assert T.ls_name("circle", "l2") == "lam_sweep_circle_l2"
    assert T.ls_name("circle", "h2", LS_N) == "lam_sweep_circle"
    assert T.main_const_name("circle", LS_N) == "lam_sweep_circle"
    for n in SIZES:
        if n != LS_N:
            assert T.main_const_name("circle", n) == f"lam_sweep_circle_n{n}"


def test_main_const_reads_every_n_from_its_file(results):
    vals = {}
    for k, n in enumerate(SIZES):
        name = T.main_const_name("trimodal", n)
        vals[n] = [0.1 + k, 0.2 + k, 0.3 + k]
        _sweep(results / f"{name}.npz", "trimodal", n, vals[n], [0.5, 0.5, 0.5])
    S = T.Sources()
    for n in SIZES:
        v = T.main_const(S, "trimodal", n)
        assert v is not None and v is not T.PENDING
        np.testing.assert_allclose(np.asarray(v), vals[n])
        np.testing.assert_allclose(v.se, 1e-3)


def test_main_const_missing_file_is_pending_at_every_n(results):
    _sweep(results / "lam_sweep_trimodal.npz", "trimodal", LS_N, [0.1, 0.2], [0.5, 0.5])
    S = T.Sources()
    for n in SIZES:
        v = T.main_const(S, "trimodal", n)
        assert (v is T.PENDING) == (n != LS_N), n


def test_main_const_file_of_another_n_prints_dash(results):
    other = [n for n in SIZES if n != LS_N][0]
    # a file named for N = other but recording N = LS_N: never shown as N = other
    _sweep(results / f"lam_sweep_circle_n{other}.npz", "circle", LS_N, [0.1], [0.5])
    assert T.main_const(T.Sources(), "circle", other) is None


def test_main_const_smoke_full_sweep_at_its_own_n(results):
    """A smoke directory: the full sweep records the smoke N; without lam_sweep_<T>_n<N> it
    fills the column of that N (and is '--' at LS_N)."""
    small = min(SIZES)
    _sweep(results / "lam_sweep_inhom.npz", "inhom", small, [0.3], [0.5])
    S = T.Sources()
    assert T.main_const_source(S, "inhom", small)[0] == "lam_sweep_inhom"
    np.testing.assert_allclose(np.asarray(T.main_const(S, "inhom", small)), [0.3])
    assert T.main_const(S, "inhom", LS_N) is None


def _const_row(tex, label):
    """Cells of the hold-out constant row in the block whose rotated label contains label."""
    rows = [ln for ln in tex.splitlines() if not ln.startswith("%")]
    start = next(i for i, ln in enumerate(rows) if "multirow" in ln and label in ln)
    row = next(ln for ln in rows[start:] if "hold-out constant" in ln)
    return [c.strip() for c in row.rstrip("\\ ").split("&")[2:]]


def test_tab_main_row_and_header_name_the_files(results):
    """Files at two N: the row prints both, \\tbd at the third; the header names each file
    or its absence; the stretch line opens the tabular."""
    missing = [n for n in SIZES if n != LS_N][-1]
    for n in SIZES:
        if n != missing:
            _sweep(results / f"{T.main_const_name('trimodal', n)}.npz", "trimodal", n,
                   [0.11, 0.13], [0.5, 0.5])
    tex = T.tab_main(T.Sources())
    cells = _const_row(tex, "Trimodal")
    for n, c in zip(SIZES, cells):
        if n == missing:
            assert c == r"\tbd", (n, c)
        else:
            assert re.fullmatch(r"\$(\\mathbf\{)?12\.0 \\pm 1\.0\}?\$", c), (n, c)
    for n in SIZES:
        name = T.main_const_name("trimodal", n)
        line = next(ln for ln in tex.splitlines()
                    if ln.startswith(f"% trimodal: hold-out constant at N = {n} "))
        assert f"{name}.npz" in line
        assert ("pending" in line) == (n == missing)
    assert f"lam_sweep_trimodal_n{missing}.npz" in next(
        ln for ln in tex.splitlines() if ln.startswith("% pending (missing) sources"))
    body = [ln for ln in tex.splitlines() if not ln.startswith("%")]
    k = next(i for i, ln in enumerate(body) if ln.startswith(r"\begin{tabular}"))
    assert body[k - 1] == fr"\renewcommand{{\arraystretch}}{{{T.main_rot_stretch():.2f}}}"
    assert "lam_sweep.py --n N --const-only" in tex


# ================================================================ caption, column spacing, blocks
# tab_twod_full: RK4 step count in the caption, narrower \tabcolsep; tab_refine_1d / _2d: two
# blocks per target (the regularizers, then lambda -> 0). The tests below read the output
# through a generic tabular parser, so that they do not depend on the line layout of the
# generator, only on what LaTeX would typeset.

def _split_top(s, sep):
    """Split s at sep outside braces (a \\\\ or & inside {...} belongs to a nested cell,
    e.g. the rotated two-line labels of tab_oned_full)."""
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and s.startswith(sep, i) and depth == 0:
            out.append("".join(cur))
            cur, i = [], i + len(sep)
            continue
        if ch == "\\" and i + 1 < len(s):
            # escaped characters (\{, \}, \&, \\) never change the depth
            cur.append(s[i:i + 2])
            i += 2
            continue
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur, i = [], i + 1
            continue
        depth += ch == "{"
        depth -= ch == "}"
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


_RULE_RE = re.compile(r"\\(?:toprule|midrule|bottomrule|addlinespace|hline)(?:\[[^\]]*\])?"
                      r"|\\cmidrule(?:\([^)]*\))?\{[^}]*\}"
                      r"|\\rowcolor\{[^}]*\}")
_MC_RE = re.compile(r"\\multicolumn\{(\d+)\}\{([^}]*)\}\{(.*)\}", re.S)


def _strip_comments(tex):
    """Drop LaTeX comments: from an unescaped % to the end of its line."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in tex.splitlines())


def _spec_ncols(spec):
    """Number of columns of a tabular spec: l, c, r and p/m/b{..}; @{..}, !{..} and | are
    not columns."""
    spec = re.sub(r"[@!]\{(?:[^{}]|\{[^{}]*\})*\}", "", spec)
    spec = re.sub(r"[pmb]\{[^}]*\}", "c", spec)
    return sum(spec.count(ch) for ch in "lcr")


def _tabular(tex):
    """(spec, rows) of the outer tabular of tex (one per table); rows = [[(cell text, span)]] with comments,
    rules (\\toprule, \\midrule, \\cmidrule, \\addlinespace) and \\rowcolor removed, and the
    \\multicolumn wrapper resolved to its content and span."""
    body = _strip_comments(tex)
    m = re.search(r"\\begin\{tabular\}\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body)
    assert m, "no tabular"
    spec = m.group(1)
    inner = body[m.end():body.rindex(r"\end{tabular}")]  # nested tabulars sit in braces
    rows = []
    for raw in _split_top(inner, "\\\\"):
        raw = _RULE_RE.sub("", raw).strip()
        if not raw:
            continue
        row = []
        for c in _split_top(raw, "&"):
            c = c.strip()
            mc = _MC_RE.fullmatch(c)
            row.append((mc.group(3).strip(), int(mc.group(1))) if mc else (c, 1))
        rows.append(row)
    return spec, rows


def _expand(row):
    """Cells of a row, one per column (a \\multicolumn{w} span repeated w times)."""
    return [c for c, w in row for _ in range(w)]


def test_tabular_parser_on_a_known_table():
    tex = "\n".join([
        "% a comment & with \\\\ separators",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l@{\hspace{1em}}cc@{}}", r"\toprule",
        r"a & \multicolumn{2}{c}{$N = 1$} \\", r"\cmidrule(lr){2-3}",
        r"\midrule", r"\multicolumn{3}{l}{\emph{Name, $\lambda \to 0$}} \\",
        r"\addlinespace",
        r"\rowcolor{black!8}",
        r"\multirow{2}{*}{\rotatebox{90}{\begin{tabular}{@{}c@{}}A\\B\end{tabular}}} & "
        r"$1.0 \pm 0.1$ & \tbd \\ % trailing comment",
        r"\bottomrule", r"\end{tabular}"])
    spec, rows = _tabular(tex)
    assert _spec_ncols(spec) == 3
    assert [[w for _, w in r] for r in rows] == [[1, 2], [3], [1, 1, 1]]
    assert rows[1][0][0] == r"\emph{Name, $\lambda \to 0$}"
    assert rows[2][1:] == [(r"$1.0 \pm 0.1$", 1), (r"\tbd", 1)]


# ---------------------------------------------------------------- tab_twod_full (b)

NSTEP_FIX = 77  # a step count no config default uses, so the test sees the file's value


@pytest.fixture
def nstep_results(results):
    """twod_eval_<T> and oned_eval files that record only the production step count: every
    TV cell stays empty or pending, the RK4 label text is what the tests read."""
    for target, _ in T.TWOD_CASES:
        np.savez(results / f"{T.tname('twod_eval', target)}.npz", ns=np.array(SIZES),
                 nstep=NSTEP_FIX)
    np.savez(results / "oned_eval.npz", ns=np.array(SIZES), nstep=NSTEP_FIX)
    C.use(results)
    return results


def _twod_labels(tex):
    """First cells of the tab_twod_full body rows (method rows and spanning rows)."""
    _, rows = _tabular(tex)
    return [r[0][0] for r in rows if len(r) > 1 and r[0][1] == 1]


@pytest.mark.parametrize("with_files", [False, True])
def test_tab_twod_row_labels_carry_no_step_count(results, request, with_files):
    if with_files:
        request.getfixturevalue("nstep_results")
    tex = T.tab_twod_full(T.Sources())
    labs = _twod_labels(tex)
    assert labs and not [x for x in labs if "RK4 (" in x], labs
    exact = [x for x in labs if "v_N" in x]
    assert exact == [r"exact $v_N$"] * len(T.TWOD_CASES), exact


def test_tab_twod_step_count_moves_to_a_comment(nstep_results):
    tex = T.tab_twod_full(T.Sources())
    com = [ln for ln in tex.splitlines() if ln.lstrip().startswith("%")]
    assert any("RK4" in ln and re.search(fr"\b{NSTEP_FIX}\b", ln) for ln in com), \
        "the RK4 step count of twod_eval must survive in a % line for the caption"


def test_tab_oned_keeps_the_step_count_in_its_label(nstep_results):
    """The {rk} placeholder of the other tables still expands to 'RK4 (<k> steps)'."""
    _, rows = _tabular(T.tab_oned_full(T.Sources()))
    labs = [c for r in rows for c, _ in r if "v_N" in c]
    assert fr"exact $v_N$, RK4 ({NSTEP_FIX} steps)" in labs, labs


def test_tab_twod_narrows_tabcolsep_before_the_tabular(results):
    tex = _strip_comments(T.tab_twod_full(T.Sources()))
    k = tex.index(r"\begin{tabular}")
    m = list(re.finditer(r"\\setlength\{\\tabcolsep\}\{([0-9.]+)pt\}", tex[:k]))
    assert m, r"\setlength{\tabcolsep}{...pt} missing before \begin{tabular}"
    assert 0 < float(m[-1].group(1)) < 6.0  # LaTeX default: 6pt


def test_tab_twod_rows_fill_the_column_spec(results):
    spec, rows = _tabular(T.tab_twod_full(T.Sources()))
    n = _spec_ncols(spec)
    assert n == 1 + 2 * len(SIZES)
    for r in rows:
        assert sum(w for _, w in r) == n, r


# ---------------------------------------------------------------- tab_refine_1d / _2d (b)

REFINE_SPEC = {
    "1d": (["Trimodal", "Inhomogeneous"], [256, 512, 1024]),
    "2d": (["Noisy circle", "Checkerboard", "Two spirals", "8 Gaussians"], [64, 128, 256, 512]),
}
REFINE_KEY = {"Trimodal": "trimodal", "Inhomogeneous": "inhom", "Noisy circle": "circle",
              "Checkerboard": "checkerboard", "Two spirals": "spirals", "8 Gaussians": "gauss8"}
LAM0_HEAD, REG_HEAD = r"{}, $\lambda \to 0$", r"{}, $\nabla^k$ regularizer"
K_CELLS = {"", "$2$", "$1$", "$0$"}
EMPH_RE = re.compile(r"\\emph\{(.*)\}", re.S)


def _refine_blocks(tex):
    """(ncols, headings in reading order, {heading: [(k, m, [N cells])]}) of a refine table.
    A heading is an \\emph cell spanning one half; the rows below it, in the same half, belong
    to it until the next heading of that half. k is the first column of a half, m the second;
    a half with an empty m cell is padding. Rows above the first heading (column header,
    power row) are skipped."""
    spec, rows = _tabular(tex)
    ncol = _spec_ncols(spec)
    half = ncol // 2
    order, blocks, cur = [], {}, [None, None]
    for r in rows:
        heads, col = [], 0
        for c, w in r:
            e = EMPH_RE.fullmatch(c)
            if e:
                heads.append((col // half, e.group(1)))
            col += w
        if heads:
            for h, name in heads:
                cur[h] = name
                order.append(name)
                blocks.setdefault(name, [])
            continue
        if cur == [None, None]:
            continue
        cells = _expand(r)
        for h in range(2):
            part = cells[h * half:(h + 1) * half]
            if cur[h] is None or len(part) < 2 or not part[1]:
                continue
            blocks[cur[h]].append((part[0], part[1], part[2:]))
    return ncol, order, blocks


def _refine_tex(which, results=None, distinct=False):
    if results is not None:
        (_distinct_refine if distinct else _complete_refine)(results)
    return getattr(T, f"tab_refine_{which}")(T.Sources())


@pytest.mark.parametrize("which", ["1d", "2d"])
@pytest.mark.parametrize("filled", [False, True])
def test_refine_two_headings_per_target_regularizer_first(results, which, filled):
    """The author's order: '<Target>, nabla^k regularizer' (k rows), then '<Target>,
    lambda -> 0'."""
    names, _ = REFINE_SPEC[which]
    _, order, _ = _refine_blocks(_refine_tex(which, results if filled else None))
    for nm in names:
        a, b = LAM0_HEAD.format(nm), REG_HEAD.format(nm)
        assert order.count(a) == 1 and order.count(b) == 1, (nm, order)
        assert order.index(b) < order.index(a), (nm, order)
    assert len(order) == 2 * len(names), order


@pytest.mark.parametrize("which", ["1d", "2d"])
@pytest.mark.parametrize("filled", [False, True])
def test_refine_k_column_holds_only_orders(results, which, filled):
    """lambda -> 0 lives in the heading, never in the k column; the k column of either half
    holds $2$, $1$, $0$ or nothing."""
    _, _, blocks = _refine_blocks(_refine_tex(which, results if filled else None))
    ks = [k for rows in blocks.values() for k, _, _ in rows]
    assert ks and set(ks) <= K_CELLS, sorted(set(ks) - K_CELLS)
    # every row, header rows included: only a heading (an \emph span) may name lambda -> 0
    spec, rows = _tabular(_refine_tex(which))
    half = _spec_ncols(spec) // 2
    for r in rows:
        if EMPH_RE.fullmatch(r[0][0]):
            continue
        cells = _expand(r)
        assert r"\lambda \to 0" not in cells[0] + cells[half], r


@pytest.mark.parametrize("which", ["1d", "2d"])
def test_refine_blocks_order_k_and_m(results, which):
    """lambda -> 0 block: one row per m, k empty; regularizer block: groups k = 2, 1, 0, each
    with one row per m, k on the first row of its group only."""
    names, ms = REFINE_SPEC[which]
    _, _, blocks = _refine_blocks(_refine_tex(which, results))
    for nm in names:
        lam0 = blocks[LAM0_HEAD.format(nm)]
        assert [(k, int(m)) for k, m, _ in lam0] == [("", m) for m in ms], (nm, lam0)
        reg = blocks[REG_HEAD.format(nm)]
        exp = [(k if i == 0 else "", m) for k in ("$2$", "$1$", "$0$")
               for i, m in enumerate(ms)]
        assert [(k, int(m)) for k, m, _ in reg] == exp, (nm, reg)


@pytest.mark.parametrize("which", ["1d", "2d"])
@pytest.mark.parametrize("filled", [False, True])
def test_refine_rows_fill_the_column_spec(results, which, filled):
    spec, rows = _tabular(_refine_tex(which, results if filled else None))
    n = _spec_ncols(spec)
    assert n == 2 * (2 + len(SIZES))
    for r in rows:
        assert sum(w for _, w in r) == n, r


@pytest.mark.parametrize("which", ["1d", "2d"])
def test_refine_pending_sources_print_tbd(results, which):
    names, ms = REFINE_SPEC[which]
    _, _, blocks = _refine_blocks(_refine_tex(which))
    for nm in names:
        for head in (LAM0_HEAD, REG_HEAD):
            rows = blocks[head.format(nm)]
            assert len(rows) == len(ms) * (1 if head is LAM0_HEAD else 3), (nm, head)
            assert all(c == r"\tbd" for _, _, cells in rows for c in cells), (nm, head)


# Distinct value per (target, row kind, m): a misplaced or duplicated cell changes the
# multiset or the placement. TV of set s is v + 0.01 s (2 sets), so the cell reads
# 100 (v + 0.005) +- 0.5 in units of 10^-2.
_REFINE_KINDS = ("lam0", "h2", "h1", "l2")


def _refine_code(case, kind, m):
    cases = list(REFINE_KEY.values())
    ms = sorted(set(T.REFINE_M[1]) | set(T.REFINE_M[2]) | {T.REFINE_M_EXTRA})
    return (cases.index(case) * 4 + _REFINE_KINDS.index(kind)) * len(ms) + ms.index(m) + 1


def _refine_v(case, kind, m):
    return 0.001 * _refine_code(case, kind, m)


def _refine_cell(case, kind, m):
    return f"${100 * (_refine_v(case, kind, m) + 0.005):.1f} \\pm 0.5$"


def _distinct_refine(root):
    """_complete_refine with a value of its own per (target, kind, m)."""
    ns = list(SIZES)
    one = [c for c in REFINE_KEY.values() if c in C.DATASETS]
    two = [c for c in REFINE_KEY.values() if c not in C.DATASETS]
    for reg in ("h2", "h1", "l2"):
        sfx = "" if reg == "h2" else f"_{reg}"
        d = {f"{c}/ms": np.array(T.REFINE_M[1]) for c in one}
        d.update({f"{c}/m{m}/holdout/{f}": _vals(_refine_v(c, reg, m)) for c in one
                  for m in T.REFINE_M[1] for f in ("tv_pos", "tv_pos_se")})
        _npz(root / f"controls_1d{sfx}.npz", ns, **d)
        for c in two:
            d = {f"{c}/ms": np.array(T.REFINE_M[2])}
            d.update({f"{c}/m{m}/holdout/{f}": _vals(_refine_v(c, reg, m))
                      for m in T.REFINE_M[2] for f in ("tv_pos", "tv_pos_se")})
            _npz(root / f"controls_{c}{sfx}.npz", ns, **d)
            m = T.REFINE_M_EXTRA
            _npz(root / f"controls_{c}{sfx}_m{m}.npz", ns,
                 **{f"{c}/ms": np.array([m]),
                    f"{c}/m{m}/holdout/tv_pos": _vals(_refine_v(c, reg, m)),
                    f"{c}/m{m}/holdout/tv_pos_se": _vals(0.001)})
    _npz(root / "controls_iid_1d.npz", ns,
         **{f"{c}/m{m}/lam0/{f}": _vals(_refine_v(c, "lam0", m)) for c in one
            for m in T.REFINE_M[1] for f in ("tv_iid", "tv_iid_se")})
    for c in two:
        _npz(root / f"controls_iid_{c}.npz", ns,
             **{f"{c}/m{m}/lam0/{f}": _vals(_refine_v(c, "lam0", m)) for m in T.REFINE_M[2]
                for f in ("tv_iid", "tv_iid_se")})
        m = T.REFINE_M_EXTRA
        _npz(root / f"controls_iid_{c}_m{m}.npz", ns,
             **{f"{c}/ms": np.array([m]),
                f"{c}/m{m}/lam0/tv_iid": _vals(_refine_v(c, "lam0", m)),
                f"{c}/m{m}/lam0/tv_iid_se": _vals(0.001)})


def _expected_refine_cells(which):
    """The numeric cells the inputs of _distinct_refine determine, independent of layout."""
    names, ms = REFINE_SPEC[which]
    return sorted(_refine_cell(REFINE_KEY[nm], kind, m) for nm in names
                  for kind in _REFINE_KINDS for m in ms for _ in SIZES)


@pytest.mark.parametrize("which", ["1d", "2d"])
def test_refine_numeric_cells_unchanged_by_layout(results, which):
    """The multiset of '$x \\pm y$' cells equals the one the inputs determine (the layout
    change moves cells, it never adds, drops or reformats one)."""
    tex = _refine_tex(which, results, distinct=True)
    _, rows = _tabular(tex)
    got = sorted(c for r in rows for c in _expand(r) if re.fullmatch(r"\$.* \\pm .*\$", c))
    assert got == _expected_refine_cells(which)


@pytest.mark.parametrize("which", ["1d", "2d"])
def test_refine_numeric_cells_sit_in_their_block(results, which):
    names, ms = REFINE_SPEC[which]
    _, _, blocks = _refine_blocks(_refine_tex(which, results, distinct=True))
    for nm in names:
        c = REFINE_KEY[nm]
        lam0 = blocks[LAM0_HEAD.format(nm)]
        assert [cells for *_, cells in lam0] == \
            [[_refine_cell(c, "lam0", m)] * len(SIZES) for m in ms], nm
        reg = blocks[REG_HEAD.format(nm)]
        assert [cells for *_, cells in reg] == \
            [[_refine_cell(c, r, m)] * len(SIZES) for r in ("h2", "h1", "l2") for m in ms], nm


def test_refine_2d_sets_tabcolsep_before_the_tabular(results):
    """tab_refine_2d is set at \\tabcolsep 5.5pt, a value below
    LaTeX's 6pt default so the two halves fit the text width."""
    tex = _strip_comments(T.tab_refine_2d(T.Sources()))
    k = tex.index(r"\begin{tabular}")
    m = re.findall(r"\\setlength\{\\tabcolsep\}\{([0-9.]+)pt\}", tex[:k])
    assert m and float(m[-1]) == 5.5, m
