"""Layout of tab_tend (experiments/report/tables.py): rows = estimator, columns = N x shown
1 - t_end.

Specification under test:
- config.TEND["table_one_minus_t_end"] lists the end times the table shows (a subset of the
  results' one_minus_t_end, in config order); the results hold all of
  config.TEND["one_minus_t_end"].
- One block per target: the two 1D targets of tend.npz (always, \\tbd while missing), then
  every 2D target whose tend_<T>.npz exists, in TWOD_CASES order. A block opens with
  \\midrule and \\multicolumn{ncol}{l}{\\emph{<label>}} \\\\ and holds one body row per
  estimator of TEND_ROWS (4 rows).
- Columns: l + |SIZES| |shown| c columns, N-major. Header: 'Estimator' and one
  \\multicolumn{|shown|}{c}{$N = n$} per N; one \\cmidrule(lr){a-b} per N group; a row
  starting '$1 - t_{\\mathrm{end}}$' with the shown $10^{p}$ per N; a units row.
- Cell (target d, estimator (kind, key), N n, shown end time e) = mean \\pm SE over sets of
  d/kind/key at n and the result index j with one_minus_t_end[j] == e, in units of
  10^TV_E; values at end times not shown do not appear.
- A shown end time absent from the results raises ValueError; a 2D file whose
  1 - t_end differs from tend.npz raises ValueError.
- A comment line of the note names the shown end times and those only in the results.

Unit level: synthetic npz stores in a temporary results directory, no compute."""

from __future__ import annotations

import re
import subprocess

import numpy as np
import pytest

from release_helpers import REPORT, ROOT, load_isolated

pytestmark = pytest.mark.report

C, T = load_isolated(REPORT, "tables", "tend_layout")
CFG = T.config
SIZES = list(C.SIZES)
OM = [float(e) for e in CFG.TEND["one_minus_t_end"]]   # what the results hold
NSET = 5
TWOD_PRESENT = ["circle", "spirals"]          # checkerboard and gauss8 absent on purpose
KINDS = ("exact", "oracle", "holdout")
SET_OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 3.0]) * 1e-4


def shown():
    """The end times the table shows (config; read at call time so monkeypatch applies)."""
    return [float(e) for e in CFG.TEND["table_one_minus_t_end"]]


def ncol():
    return 1 + len(SIZES) * len(shown())


def _p(e):
    return int(round(np.log10(e)))


# ---------------------------------------------------------------- synthetic stores

def _mean(b, g, i, k):
    """Distinct mean per (block b, TEND_ROWS row g, N index i, result t_end index k):
    consecutive codes 0.002 apart, i.e. 0.2 apart in units of 10^-2, so a swapped or
    wrongly selected cell shows."""
    code = ((b * len(T.TEND_ROWS) + g) * len(SIZES) + i) * len(OM) + k
    return 0.002 * (code + 1)


def _arrays(b, d):
    """Keys d/kind/key (+ _se) of block b: shape (len(SIZES), NSET, len(OM)). Keys that no
    row reads get a constant, so that reading one by mistake shows."""
    out = {}
    for kind in KINDS:
        for key in ("tv_iid", "tv_pos", "tv_law"):
            out[f"{d}/{kind}/{key}"] = np.full((len(SIZES), NSET, len(OM)), 0.987)
            out[f"{d}/{kind}/{key}_se"] = np.full((len(SIZES), NSET, len(OM)), 1e-3)
    for g, (kind, key, _) in enumerate(T.TEND_ROWS):
        v = np.empty((len(SIZES), NSET, len(OM)))
        for i in range(len(SIZES)):
            for k in range(len(OM)):
                v[i, :, k] = _mean(b, g, i, k) + SET_OFFSETS * (1 + i + k)
        out[f"{d}/{kind}/{key}"] = v
        out[f"{d}/{kind}/{key}_se"] = np.full_like(v, 5e-4)
    return out


def _expected(b, g, i, k):
    """Formatted cell from the module's own helpers (mean, SE over sets; unit 10^TV_E)."""
    v = _mean(b, g, i, k) + SET_OFFSETS * (1 + i + k)
    m, se = C.mean_se(v)
    return C.fmt_sci(m, se, T.TV_E)


def _write(root, one_d=True, two_d=TWOD_PRESENT, om_2d=None):
    common = dict(ns=np.array(SIZES), sets=np.arange(NSET))
    if one_d:
        d = {}
        for b, ds in enumerate(C.DATASETS):
            d.update(_arrays(b, ds))
        np.savez(root / "tend.npz", one_minus_t_end=np.array(OM), **common, **d)
    for j, t in enumerate(two_d):
        np.savez(root / f"tend_{t}.npz", one_minus_t_end=np.array(om_2d or OM), **common,
                 **_arrays(len(C.DATASETS) + j, t))


def _labels(two_d):
    lab = dict(T.TWOD_CASES)
    return ([C.DATASET_LABEL[d] for d in C.DATASETS]
            + [T.dim_label(t, lab[t]) for t, _ in T.TWOD_CASES if t in two_d])


# ---------------------------------------------------------------- parsing

RULE_RE = re.compile(r"\\(toprule|midrule|bottomrule|addlinespace|cmidrule\(\w*\)\{\d+-\d+\}"
                     r"|rowcolor\{[^}]*\})")
BLOCK_RE = re.compile(r"^\\multicolumn\{(\d+)\}\{l\}\{\\emph\{(.+)\}\}$")
MC_RE = re.compile(r"^\\multicolumn\{(\d+)\}")


def _tabular(tex):
    """(column spec, tabular content) of the one tabular in tex."""
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", tex, re.S)
    assert m, tex
    return m.group(1), m.group(2)


def _rows(content):
    """Rows of a tabular content: text between row ends, rule commands removed."""
    out = []
    for chunk in content.split("\\\\"):
        s = RULE_RE.sub("", chunk).strip()
        if s:
            out.append(s)
    return out


def _cells(row):
    return [c.strip() for c in row.split("&")]


def _span(row):
    """Number of columns a row spans (\\multicolumn{n} counts n)."""
    tot = 0
    for c in _cells(row):
        m = MC_RE.match(c)
        tot += int(m.group(1)) if m else 1
    return tot


def _parse(tex):
    """(header rows, [(block label, [body rows as cell lists])])."""
    _, content = _tabular(tex)
    head, blocks = [], []
    for r in _rows(content):
        m = BLOCK_RE.match(r)
        if m:
            blocks.append((m.group(2), []))
        elif blocks:
            blocks[-1][1].append(_cells(r))
        else:
            head.append(r)
    return head, blocks


def _rendered(root):
    C.use(root)
    return T.tab_tend(T.Sources())


@pytest.fixture
def results(tmp_path):
    C.use(tmp_path)
    return tmp_path


def _check_values(tex, two_d):
    """Every body cell equals the expected cell of its (N, shown end time) column."""
    sh = shown()
    _, blocks = _parse(tex)
    assert [lab for lab, _ in blocks] == _labels(two_d)
    bad = []
    for b, (lab, rows) in enumerate(blocks):
        assert len(rows) == len(T.TEND_ROWS), (lab, rows)
        for g, row in enumerate(rows):
            assert row[0] == T.TEND_ROWS[g][2], (lab, g, row[0])
            assert len(row) == ncol(), (lab, g, row)
            for i in range(len(SIZES)):
                for s, e in enumerate(sh):
                    got = row[1 + i * len(sh) + s]
                    want = _expected(b, g, i, OM.index(e))
                    if got != want:
                        bad.append((lab, T.TEND_ROWS[g][:2], SIZES[i], e, got, want))
    assert not bad, bad[:10]


# ---------------------------------------------------------------- config

def test_config_shows_a_subset_of_the_result_end_times():
    sh = shown()
    assert sh and all(e in OM for e in sh), (sh, OM)
    assert len(set(sh)) == len(sh), sh


# ---------------------------------------------------------------- 1. values

def test_body_cells_equal_mean_se_of_their_n_and_shown_t_end(results):
    _write(results)
    _check_values(_rendered(results), TWOD_PRESENT)


def test_end_times_not_shown_leave_no_value_in_the_table(results):
    """The distinct per-cell values of an omitted end time appear nowhere in the body."""
    _write(results)
    _, content = _tabular(_rendered(results))
    omitted = [k for k, e in enumerate(OM) if e not in shown()]
    found = [(b, g, i, OM[k]) for b in range(len(C.DATASETS) + len(TWOD_PRESENT))
             for g in range(len(T.TEND_ROWS)) for i in range(len(SIZES)) for k in omitted
             if _expected(b, g, i, k) in content]
    assert not found, found[:10]


@pytest.mark.parametrize("sel", ["all", "reversed"])
def test_config_selects_columns(results, monkeypatch, sel):
    """table_one_minus_t_end = every result end time gives the |SIZES| |OM| column layout;
    a reversed selection is shown in config order."""
    monkeypatch.setitem(CFG.TEND, "table_one_minus_t_end",
                        tuple(OM) if sel == "all" else tuple(reversed(shown())))
    _write(results)
    tex = _rendered(results)
    spec, _ = _tabular(tex)
    assert spec == "l" + "c" * (ncol() - 1), spec
    if sel == "all":
        assert ncol() == 1 + len(SIZES) * len(OM)
    head, _ = _parse(tex)
    assert _cells(head[1])[1:] == [f"$10^{{{_p(e)}}}$" for e in shown()] * len(SIZES), head[1]
    _check_values(tex, TWOD_PRESENT)


def _head_tables(tmp_path):
    """tables.py and common.py of git HEAD loaded as separate modules."""
    d = tmp_path / "head_report"
    d.mkdir()
    for f in ("tables.py", "common.py"):
        r = subprocess.run(["git", "show", f"HEAD:experiments/report/{f}"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"git show HEAD:{f} failed: {r.stderr.strip()}")
        (d / f).write_text(r.stdout)
    return load_isolated(d, "tables", "tend_layout_head")


def test_cells_match_head_restricted_to_the_shown_columns(results, tmp_path):
    """Each new cell (block, row g, N i, shown e) equals HEAD's cell in the all-end-times
    layout (d9f2ca0: l + |SIZES| |OM| columns), column 1 + i |OM| + index of e."""
    _write(results)
    new = _parse(_rendered(results))[1]
    C0, T0 = _head_tables(tmp_path)
    C0.use(results)
    try:
        old_tex = T0.tab_tend(T0.Sources())
    except Exception as exc:              # HEAD already reads the new config key, etc.
        pytest.skip(f"HEAD tab_tend does not render the reference layout: {exc!r}")
    if _tabular(old_tex)[0] != "l" + "c" * (len(SIZES) * len(OM)):
        pytest.skip("HEAD tab_tend is not the all-end-times layout")
    old = _parse(old_tex)[1]
    sh = shown()
    assert [lab for lab, _ in old] == [lab for lab, _ in new]
    for (lab, orows), (_, nrows) in zip(old, new):
        assert len(orows) == len(nrows) == len(T.TEND_ROWS), lab
        for g in range(len(T.TEND_ROWS)):
            for i in range(len(SIZES)):
                for s, e in enumerate(sh):
                    assert (nrows[g][1 + i * len(sh) + s]
                            == orows[g][1 + i * len(OM) + OM.index(e)]), (lab, g, i, e)


# ---------------------------------------------------------------- 2. structure

def test_column_spec_and_header(results):
    _write(results)
    tex = _rendered(results)
    sh, nc = shown(), ncol()
    spec, content = _tabular(tex)
    assert spec == "l" + "c" * (nc - 1), spec
    cmid = "".join(fr"\cmidrule(lr){{{2 + len(sh) * j}-{1 + len(sh) * (j + 1)}}}"
                   for j in range(len(SIZES)))
    assert cmid in content, (cmid, content[:600])
    head, _ = _parse(tex)
    assert len(head) == 3, head
    h1, h2, h3 = (_cells(r) for r in head)
    assert h1 == ["Estimator"] + [fr"\multicolumn{{{len(sh)}}}{{c}}{{$N = {n}$}}"
                                  for n in SIZES], h1
    assert h2 == [r"$1 - t_{\mathrm{end}}$"] + [f"$10^{{{_p(e)}}}$" for e in sh] * len(SIZES), h2
    assert h3 == [""] + [C.power_label(T.TV_E)] * (nc - 1), h3
    i1, ic, i2 = (content.index(s) for s in ("Estimator", cmid, r"$1 - t_{\mathrm{end}}$"))
    assert i1 < ic < i2


@pytest.mark.parametrize("two_d", [[], ["circle"], TWOD_PRESENT,
                                   [t for t, _ in T.TWOD_CASES]])
def test_one_block_per_target_four_rows_each(results, two_d):
    _write(results, two_d=two_d)
    tex = _rendered(results)
    _, content = _tabular(tex)
    _, blocks = _parse(tex)
    assert len(blocks) == len(C.DATASETS) + len(two_d)
    assert [lab for lab, _ in blocks] == _labels(two_d)
    assert all(len(rows) == len(T.TEND_ROWS) for _, rows in blocks), blocks
    for lab in _labels(two_d):
        assert re.search(r"\\midrule\s*\\multicolumn\{%d\}\{l\}\{\\emph\{%s\}\} \\\\"
                         % (ncol(), re.escape(lab)), content), lab


def _names(e):
    """Spellings of an end time a note may use: 1e-06, 1e-6, 10^-6, 10^{-6}, 0.0001."""
    p = -_p(e)
    return [f"{e:g}", f"1e-{p}", f"1e-0{p}", f"10^-{p}", f"10^{{-{p}}}", f"1e{-p}"]


def test_note_names_shown_and_omitted_end_times(results):
    _write(results)
    tex = _rendered(results)
    omitted = [e for e in OM if e not in shown()]
    comments = [ln for ln in tex.splitlines() if ln.lstrip().startswith("%")]

    def names(ln, e):
        return any(re.search(re.escape(s) + r"(?![0-9])", ln) for s in _names(e))

    hits = [ln for ln in comments if all(names(ln, e) for e in shown() + omitted)]
    assert hits, ("no comment line names all of", shown(), omitted, comments)


# ---------------------------------------------------------------- 3. robustness

def test_no_tend_npz_renders_tbd(results):
    _, blocks = _parse(_rendered(results))
    assert [lab for lab, _ in blocks] == _labels([])
    for lab, rows in blocks:
        assert len(rows) == len(T.TEND_ROWS)
        for r in rows:
            assert len(r) == ncol() and all(c == C.tbd() for c in r[1:]), (lab, r)


def test_tend_npz_without_2d_files_gives_the_1d_blocks(results):
    _write(results, two_d=[])
    _, blocks = _parse(_rendered(results))
    assert [lab for lab, _ in blocks] == _labels([])
    assert all(C.tbd() not in c for _, rows in blocks for r in rows for c in r)


def test_mismatching_t_end_in_2d_file_raises(results):
    _write(results, two_d=["circle"], om_2d=[1e-2, 1e-4, 1e-6])
    with pytest.raises(ValueError):
        _rendered(results)


def test_shown_end_time_absent_from_results_raises(results, monkeypatch):
    monkeypatch.setitem(CFG.TEND, "table_one_minus_t_end", (OM[0], 1e-5))
    _write(results)
    with pytest.raises(ValueError):
        _rendered(results)


# ---------------------------------------------------------------- 4. separators

@pytest.mark.parametrize("case", ["full", "empty", "one_d_only"])
def test_every_row_spans_the_column_spec(results, case):
    if case == "full":
        _write(results)
    elif case == "one_d_only":
        _write(results, two_d=[])
    spec, content = _tabular(_rendered(results))
    nc = len(spec)
    assert nc == ncol()
    bad = [(r, _span(r)) for r in _rows(content) if _span(r) != nc]
    assert not bad, bad
    for r in _rows(content):
        if r"\multicolumn" not in r:
            assert r.count("&") == nc - 1, r
