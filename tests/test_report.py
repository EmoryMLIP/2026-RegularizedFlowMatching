"""Report stage: paper outputs, compact results, the unresolved rule, the cost table.

Marker ``report`` (the CI report-regression job: ``make fetch-results``, then
``pytest -m report``). The tests that need results read them from
``$SMOOTHFM_REPORT_RESULTS``, the unpacked release archive
``RESULTS_ROOT/<name>/paper/results`` or ``results/compact/`` and skip with that reason when
none exists; the report from the smoke results is tested in test_scripts.py (marker ``script``).

Interface the tests assume (see also tests/release_helpers.py):
- ``experiments/report/OUTPUTS.txt`` lists the generated paper files, one path per line
  (``tables/tab_main.tex``); it equals the \\input targets of the paper plus the cost
  table, ``tables/tab_cost*.tex``, which the paper may not \\input yet.
- "Report from full results equals report from compact results" compares the files after
  removing LaTeX comment lines that name the results directory or a source file.
- The unresolved rule is the function ``unresolved`` of experiments/report/common.py with
  signature (a, se_a, b, se_b[, spread=...]) that
  is true when |a - b| < 2 sqrt(se_a^2 + se_b^2) or |a - b| < spread.
- The cost table names the experiments and gives a wall time or core-hours.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import pathlib
import re
import sys

import pytest

from release_helpers import (COST_TABLE, PAPER_INPUTS, REFINE_TABLES, REPORT, ROOT,
                             committed_out, paper_inputs, read_outputs_txt, run_report,
                             strip_tex_comments, tree_files)

pytestmark = pytest.mark.report


# ---------------------------------------------------------------- output names (no results)

def test_report_outputs_are_the_paper_inputs():
    """The paper \\inputs exactly PAPER_INPUTS."""
    got = paper_inputs()
    if got is None:
        pytest.skip("paper sources not found (set SMOOTHFM_PAPER)")
    assert got == PAPER_INPUTS, (sorted(got - PAPER_INPUTS), sorted(PAPER_INPUTS - got))


def test_outputs_txt_lists_paper_outputs_and_cost_table():
    got = read_outputs_txt()
    assert got is not None, "experiments/report/OUTPUTS.txt missing"
    extra = got - PAPER_INPUTS
    assert PAPER_INPUTS <= got, sorted(PAPER_INPUTS - got)
    cost = [p for p in extra if COST_TABLE.match(p)]
    assert len(cost) == 1, f"one cost table tables/tab_cost*.tex expected, got {cost}"
    assert extra == set(cost), f"outputs that the paper does not \\input: {sorted(extra)}"


def test_outputs_txt_matches_paper_inputs():
    got, paper = read_outputs_txt(), paper_inputs()
    if paper is None:
        pytest.skip("paper sources not found (set SMOOTHFM_PAPER)")
    assert got is not None
    assert paper <= got, sorted(paper - got)
    assert all(COST_TABLE.match(p) for p in got - paper), sorted(got - paper)


# ---------------------------------------------------------------- unresolved rule

def _common():
    p = REPORT / "common.py"
    assert p.is_file(), "experiments/report/common.py missing"
    sys.path.insert(0, str(REPORT))
    try:
        spec = importlib.util.spec_from_file_location("report_common", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(REPORT))
    if callable(getattr(mod, "unresolved", None)):
        return mod.unresolved
    pytest.fail("no unresolved(a, se_a, b, se_b) in experiments/report/common.py")


@pytest.mark.parametrize("ratio,expected", [(0.5, True), (1.9, True), (2.1, False),
                                            (5.0, False)])
def test_unresolved_two_combined_se(ratio, expected):
    f = _common()
    se_a, se_b = 0.003, 0.004
    comb = math.sqrt(se_a ** 2 + se_b ** 2)
    a = 0.135
    for b in (a + ratio * comb, a - ratio * comb):
        assert bool(f(a, se_a, b, se_b)) is expected, (ratio, b)


def test_unresolved_below_spread():
    f = _common()
    if "spread" not in inspect.signature(f).parameters:
        pytest.skip("rule has no spread argument ('or below the set-to-set spread')")
    assert f(0.10, 1e-4, 0.12, 1e-4, spread=0.03)
    assert not f(0.10, 1e-4, 0.12, 1e-4, spread=0.01)


# ---------------------------------------------------------------- report from results

# The comment lines that name the results directory or a source file (LaTeX '%', numbers.txt
# '#'): the only lines in which reports from the full and from the compact results differ.
_PROVENANCE = re.compile(rb"^[%#]\s*(results directory|source|results)\b.*$", re.M | re.I)


def _release_results():
    """RESULTS_ROOT/<name>/paper/results, where make fetch-results unpacks the release archive
    (<name> from experiments/report/results_archive.txt)."""
    name = ""
    with open(REPORT / "results_archive.txt") as f:
        for ln in f:
            w = ln.split("#", 1)[0].split()
            if len(w) >= 2 and w[0] == "name":
                name = w[1]
    root = pathlib.Path(os.environ.get("RESULTS_ROOT") or ROOT / "results")
    return root / name / "paper" / "results"


def _results():
    """$SMOOTHFM_REPORT_RESULTS, else the unpacked release archive (make fetch-results), else
    results/compact (make compact); without them the tests skip, unless
    $SMOOTHFM_REQUIRE_RESULTS is set (CI once the archive is published)."""
    env = os.environ.get("SMOOTHFM_REPORT_RESULTS")
    rel = _release_results()
    p = pathlib.Path(env) if env else rel if rel.is_dir() else ROOT / "results" / "compact"
    if not p.is_dir() or not any(p.glob("*.npz")):
        msg = f"no results at {p} (make fetch-results or $SMOOTHFM_REPORT_RESULTS)"
        if os.environ.get("SMOOTHFM_REQUIRE_RESULTS"):
            pytest.fail(msg)
        pytest.skip(msg)
    return p


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    res = _results()
    outs = []
    for k in range(2):
        out = tmp_path_factory.mktemp(f"report{k}")
        r = run_report(res, out)
        assert r.returncode == 0, r.stderr[-3000:]
        outs.append(out)
    return res, outs


def test_report_generates_every_output(generated):
    _, (a, _) = generated
    files = tree_files(a)
    expected = read_outputs_txt() or PAPER_INPUTS
    missing = sorted(p for p in expected if p not in files or not files[p].strip())
    assert not missing, missing
    top = {p for p in files if p.count("/") == 1 and p.endswith(".tex")}
    extra = top - expected
    assert not extra, f"generated .tex files not in OUTPUTS.txt: {sorted(extra)}"
    paper = paper_inputs()
    if paper is not None:
        assert paper <= set(files), sorted(paper - set(files))


def test_figure_data_files_exist(generated):
    _, (a, _) = generated
    files = tree_files(a)
    missing = []
    for p, body in files.items():
        if not (p.startswith("figures/") and p.endswith(".tex")):
            continue
        txt = body.decode(errors="replace")
        txt = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in txt.splitlines())
        for m in re.finditer(r"\{([^{}\s]+\.(?:dat|csv|png|jpg|pdf))\}", txt):
            ref = m.group(1)
            ref = re.sub(r"^\\figdir/", "", ref)
            ref = ref[len("figures/"):] if ref.startswith("figures/") else ref
            if "figures/" + ref not in files:
                missing.append((p, m.group(1)))
    assert not missing, missing[:20]


def test_report_twice_byte_identical(generated):
    _, (a, b) = generated
    fa, fb = tree_files(a), tree_files(b)
    assert set(fa) == set(fb), sorted(set(fa) ^ set(fb))
    diff = sorted(k for k in fa if fa[k] != fb[k])
    assert not diff, diff[:20]


def test_report_matches_committed_files(generated):
    """The report from the results equals the committed experiments/paper files up to the
    lines that name the results directory or a source (_PROVENANCE)."""
    _, (a, _) = generated
    ref = committed_out()
    fa, fc = tree_files(a), tree_files(ref)
    if not fc:
        pytest.skip(f"no committed outputs under {ref}")
    assert set(fa) == set(fc), (f"only generated: {sorted(set(fa) - set(fc))[:10]}; "
                                f"only committed (stale): {sorted(set(fc) - set(fa))[:10]}")
    diff = sorted(k for k in fa if _PROVENANCE.sub(b"", fa[k]) != _PROVENANCE.sub(b"", fc[k]))
    assert not diff, f"regenerated files differ from the committed ones: {diff[:20]}"


def test_report_has_no_missing_sources(generated):
    _, (a, _) = generated
    tbd = [k for k, v in tree_files(a).items() if k.endswith(".tex") and b"\\tbd" in v]
    assert not tbd, tbd


def test_cost_table(generated):
    _, (a, _) = generated
    files = tree_files(a)
    cost = [p for p in files if COST_TABLE.match(p)]
    assert len(cost) == 1, f"cost table missing: {cost}"
    txt = files[cost[0]].decode(errors="replace")
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in txt.splitlines()).lower()
    body = body.replace("\\_", "_")
    for name in ("oned", "twod", "stability", "integration_error", "lam_sweep"):
        assert name.replace("_", " ") in body.replace("_", " "), name
    assert re.search(r"wall|core[- ]?h", body), "no wall time or core-hours column"
    assert re.search(r"\d", body)


def test_full_results_give_the_compact_report(generated, tmp_path):
    res, (a, _) = generated
    full = ROOT / "results"
    if not (full / "oned.npz").exists() or full.resolve() == res.resolve():
        pytest.skip("full results not present in results/")
    r = run_report(full, tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    fa, ff = tree_files(a), tree_files(tmp_path)
    assert set(fa) == set(ff), sorted(set(fa) ^ set(ff))
    diff = sorted(k for k in fa if _PROVENANCE.sub(b"", fa[k]) != _PROVENANCE.sub(b"", ff[k]))
    assert not diff, diff[:20]


# ============================================================================ committed outputs
# Content checks on the committed experiments/paper files, which
# test_report_matches_committed_files ties to the report code: no W2, no selection rules other
# than oracle and hold-out, regularizers labelled by their derivative order k, the bold rule.

import collections
import json

TEXT_SUFFIXES = (".tex", ".dat", ".csv", ".txt", ".tsv")

# W2 in any spelling ($W_2$, W2, w2 in a key name, Wasserstein); W_1 is not matched.
W2_RE = re.compile(r"wasserstein|(?<![A-Za-z0-9])W_?\{?2(?![0-9])", re.I)
# Selection rules other than oracle and hold-out.
OTHER_RULE_RE = re.compile(r"\bLOO\b|\bGCV\b|discrepancy|Morozov|plug-in|bootstrap")
# Penalty labels by norm ($L^2$, $H^1$, $H^2$, bare L2/H1/H2, Tikhonov). A norm column such as
# "$L^2$ field" (tab_stability_refine) and norm subscripts \|.\|_{H^2(\Omega)} are not penalty
# labels and do not match.
NORM_LABEL_RE = re.compile(r"\$(?:H\^\{?[12]\}?|L\^\{?2\}?)\$(?!\s*field)"
                           r"|(?<![\w\\{])(?:H[12]|L2)(?![\w])|Tikhonov")
PENALTY_WORD = {"0": "smallness", "1": "gradient", "2": "hessian"}
LONG_FORMS = ("smallness ($k = 0$)", "gradient ($k = 1$)", "Hessian ($k = 2$)")
ORDER_RE = r"\$k = {k}\$"
# Outputs that label regularizers, with the orders each one labels (fig_hero only the Hessian).
ORDER_LABELLED = {"tables/tab_reg_full.tex": "012", "tables/tab_lam_sweep_full.tex": "012",
                  "figures/fig_accuracy.tex": "012", "figures/fig_twod_ops.tex": "012",
                  "figures/fig_hero.tex": "2"}

DEC_RE = re.compile(r"-?\d+\.\d+")
ROT_TEXT_RE = re.compile(r"\\begin\{tabular\}\{@\{\}c@\{\}\}(.*?)\\end\{tabular\}")
# the rotated labels abbreviate two targets (tables.MAIN_ROT_LABEL)
_ABBREV = {"inhom": "inhomogeneous", "checkerb": "checkerboard"}


@pytest.fixture(scope="module")
def committed():
    files = tree_files(committed_out())
    if not files:
        pytest.skip(f"no committed outputs under {committed_out()}")
    return files


def _text(files, p):
    assert p in files, f"{p} not generated"
    return files[p].decode(errors="replace")


def _body(txt):
    """Non-comment content, blank lines removed."""
    return "\n".join(ln for ln in strip_tex_comments(txt).splitlines() if ln.strip())


def _tex_bodies(files):
    return {p: _body(b.decode(errors="replace")) for p, b in files.items()
            if p.endswith(".tex") and p.count("/") == 1}


def _raw(c):
    """A cell with its bold kept as *x* (so bold errors show), $ removed, spaces collapsed."""
    c = re.sub(r"\\mathbf\{([^{}]*)\}", r"*\1*", c.strip())
    return re.sub(r"\s+", " ", c.replace("$", "")).strip()


def _case_key(name):
    """'Noisy circle (2D)' -> 'circle'; one key per target, rotated \\multirow labels (their
    first line) included."""
    m = ROT_TEXT_RE.search(name)
    if m:
        name = m.group(1).split("\\\\")[0]
    name = re.sub(r"\((?:1|2)D\)", "", name).replace("\\emph", "")
    words = re.findall(r"[A-Za-z]+", name)
    key = words[-1].lower() if words else ""
    return _ABBREV.get(key, key)


def _blocks(txt):
    r"""[(block name, [(label, [raw cells])])] of a blocked table (tab_oned_full): \midrule
    opens a block, a rotated \multirow label in the first cell of one of its rows names it,
    the row label is the second cell."""
    blocks, cur = [], None
    for ln in _body(txt).splitlines():
        s = ln.strip()
        if s == r"\midrule":
            cur = [None, []]
            blocks.append(cur)
            continue
        if cur is None or "&" not in s or not s.endswith("\\\\"):
            continue
        parts = s[:-2].split("&")
        if ROT_TEXT_RE.search(parts[0]):
            cur[0] = _case_key(parts[0])
        parts = parts[1:]
        cur[1].append((parts[0].strip(), [_raw(c) for c in parts[1:]]))
    return [(n, rows) for n, rows in blocks if rows]


# Data-driven rows of tab_oned_full / tab_twod_full (the tables' own bold notes): hold-out flows
# and Gabriel rows, SJ, LSCV, Abramson-LSCV, Silverman, 4th-order LSCV. Oracle rows, the
# anisotropic rows (exact tangent), exact v_N, lambda -> 0 and the EM mixture never are.
def _data_driven(label):
    if re.search(r"oracle|aniso|exact|\\to 0|EM", label):
        return False
    return bool(re.search(r"hold-out|LSCV|Sheather|Silverman", label))


def test_no_w2_in_any_output(committed):
    hits = []
    for p, b in sorted(committed.items()):
        if not p.endswith(TEXT_SUFFIXES):
            continue
        for k, ln in enumerate(b.decode(errors="replace").splitlines(), 1):
            if W2_RE.search(ln):
                hits.append(f"{p}:{k}: {ln.strip()[:100]}")
    assert not hits, f"{len(hits)} lines name W2: {hits[:10]}"


def test_no_selection_rule_other_than_oracle_and_hold_out(committed):
    hits = [f"{p}: {m.group(0)}" for p, body in sorted(_tex_bodies(committed).items())
            for m in OTHER_RULE_RE.finditer(body)]
    assert not hits, hits[:10]


def test_tab_oned_bold_marks_lowest_data_driven_mean(committed):
    """Per block and column the lowest data-driven mean is bold, and no row that is not
    data-driven is bold. (Further bold entries are allowed: the unresolved rule needs the SEs,
    which the table does not print.)"""
    bad, blocks = [], _blocks(_text(committed, "tables/tab_oned_full.tex"))
    assert blocks, "no blocks parsed in tab_oned_full"
    for name, rows in blocks:
        ncol = max(len(c) for _, c in rows)
        for j in range(ncol):
            vals = []
            for lab, cells in rows:
                c = cells[j] if j < len(cells) else ""
                if not DEC_RE.fullmatch(c.strip("*")):
                    continue
                isbold = c.startswith("*")
                if isbold and not _data_driven(lab):
                    bad.append(f"{name}, col {j}: bold on a row that is not data-driven: {lab}")
                if _data_driven(lab):
                    vals.append((float(c.strip("*")), isbold, lab))
            if vals:
                lo = min(v for v, _, _ in vals)
                bad += [f"{name}, col {j}: lowest data-driven mean {v} not bold ({lab})"
                        for v, b, lab in vals if v == lo and not b]
    assert not bad, bad


def test_tab_oned_keeps_flows_kdes_and_four_gabriel_rows(committed):
    blocks = _blocks(_text(committed, "tables/tab_oned_full.tex"))
    assert sorted(n for n, _ in blocks) == ["inhomogeneous", "trimodal"], [n for n, _ in blocks]
    for name, rows in blocks:
        labels = [lab for lab, _ in rows]
        assert sum(lab.startswith("Gabriel et al.") for lab in labels) == 4, (name, labels)
        assert "FM, oracle $\\lambda$" in labels and "FM, hold-out" in labels, (name, labels)
        assert any(lab.startswith("KDE,") for lab in labels), (name, labels)


def test_readme_table_lists_exactly_the_outputs():
    """The item table of docs/REPRODUCE.md (README.md points to it) names exactly the outputs."""
    readme = (ROOT / "docs" / "REPRODUCE.md").read_text()
    rows = [ln for ln in readme.splitlines() if ln.startswith("| ")]
    named = set(re.findall(r"(?:tables|figures)/\w+\.tex", "\n".join(rows)))
    outs = {p for p in (read_outputs_txt() or set()) if not COST_TABLE.match(p)}
    assert named == outs, (sorted(named - outs), sorted(outs - named))


# ---------------------------------------------------------------- tab_refine_1d / _2d

def test_refine_tables_in_outputs_txt():
    got = read_outputs_txt()
    assert got is not None
    assert set(REFINE_TABLES) <= got, sorted(set(REFINE_TABLES) - got)


def _order_token(txt):
    """'smallness ($k = 0$)' and '$k = 0$' -> '<B0>'."""
    txt = re.sub(r"(?:smallness|gradient|Hessian)\s*\(\$k = (\d)\$\)", r"<B\1>", txt)
    return re.sub(r"\$k = (\d)\$", r"<B\1>", txt)


def _pair_records(txt):
    """[(case key, k, m, cells)] of a side-by-side refine table (in-line header row with the
    left and right target, then k | m | 3 N cells per half; k carried over its group).
    k: 'lam0' for lambda -> 0 (in the heading '<target>, $\\lambda \\to 0$' or the k column),
    '<Bk>' for the hold-out rows of regularizer k."""
    out, names, ks = [], None, ["", ""]
    for ln in _order_token(_body(txt)).splitlines():
        s = ln.strip()
        heads = re.findall(r"\\multicolumn\{\d+\}\{l\}\{\\emph\{(.+?)\}\}", s)
        if heads:
            names = [_case_key(h.split(",")[0]) for h in heads]
            ks = ["lam0" if "\\to 0" in h else "" for h in heads]
            continue
        if names is None or "&" not in s or not s.endswith("\\\\"):
            continue
        cs = [_raw(c) for c in s[:-2].split("&")]
        for h, part in enumerate((cs[:5], cs[5:])):
            if part[0]:
                ks[h] = ("lam0" if "\\to 0" in part[0]
                         else "<B" + part[0].strip("$ ").removeprefix("<B").rstrip(">") + ">")
            if part[1].isdigit():
                out.append((names[h], ks[h], part[1], part[2:]))
    return out


def test_refine_tables_hold_their_targets(committed):
    want = {"tables/tab_refine_1d.tex": ["trimodal", "inhomogeneous"],
            "tables/tab_refine_2d.tex": ["circle", "checkerboard", "spirals", "gaussians"]}
    for p, cases in want.items():
        recs = _pair_records(_text(committed, p))
        assert sorted({c for c, *_ in recs}) == sorted(cases), p
        for c in cases:
            ks = list(dict.fromkeys(k for cc, k, _, _ in recs if cc == c))
            # the regularizer block first, then lambda -> 0
            assert ks == ["<B2>", "<B1>", "<B0>", "lam0"], (p, c, ks)


def test_refine_tables_have_no_stopped_law(committed):
    for p in REFINE_TABLES:
        body = _body(_text(committed, p))
        assert "\\rho_N" not in body and "t_{\\mathrm{end}}" not in body, \
            "stopped-law entries belong to tab_tend"


# ---------------------------------------------------------------- regularizer labels

def test_no_norm_penalty_labels(committed):
    hits = [f"{p}: {m.group(0)!r}" for p, body in sorted(_tex_bodies(committed).items())
            for m in NORM_LABEL_RE.finditer(body)]
    assert not hits, f"{len(hits)} penalty labels by norm: {hits[:10]}"


@pytest.mark.parametrize("p", list(ORDER_LABELLED))
def test_order_labels_present(committed, p):
    """Each order k as '$k = k$', or as the first cell '$k$' of a row under a column '$k$'."""
    body = _body(_text(committed, p))
    column = re.search(r"^\$k\$ &", body, re.M)
    for k in ORDER_LABELLED[p]:
        assert re.search(ORDER_RE.format(k=k), body) or (
            column and re.search(rf"^\${k}\$ &", body, re.M)), f"order k = {k} absent from {p}"


def test_no_operator_labels(committed):
    """The regularizers are labelled by k, not by operators B_k."""
    hits = [f"{p}: {m.group(0)}" for p, body in sorted(_tex_bodies(committed).items())
            for m in re.finditer(r"\$B_\{?\d\}?\$", body)]
    assert not hits, hits[:10]


def test_long_forms_pair_words_with_the_right_order(committed):
    bad = []
    for p, body in _tex_bodies(committed).items():
        for m in re.finditer(r"([A-Za-z]+)\s*\(\s*\$k = (\d)\$\s*\)", body):
            if m.group(1).lower() != PENALTY_WORD.get(m.group(2)):
                bad.append(f"{p}: {m.group(0)}")
    assert not bad, bad


def test_long_forms_used(committed):
    allb = "\n".join(_tex_bodies(committed).values())
    missing = [f for f in LONG_FORMS if f not in allb]
    assert not missing, f"long labels never used: {missing}"


# ---------------------------------------------------------------- tab_stability

def test_tab_stability_reports_constants_over_eps(committed):
    """The panels report TV and the H^2 velocity difference divided by epsilon, with one
    rotated label per block instead of Target and N columns."""
    body = _body(_text(committed, "tables/tab_stability.tex"))
    assert "Target" not in body and "$N$ &" not in body
    assert "(a) $\\mathrm{TV}(p_\\varepsilon, p_0)/\\varepsilon$" in body
    assert "(b) $\\sup_t \\|v_t - v_t'\\|_{H^2(\\Omega)}/\\varepsilon$" in body


def test_tab_stability_refine_has_rotated_target_labels(committed):
    body = _body(_text(committed, "tables/tab_stability_refine.tex"))
    assert "Target" not in body and "\\rotatebox" in body


# ---------------------------------------------------------------- config of the release results

#: Top-level config keys whose current value differs from that in some release manifest. Keys
#: and dict entries present on one side only are not compared: the release results were
#: computed with more configured outputs than the current code writes (LOO, GCV, discrepancy and
#: K-fold rules, W2, Gaussian sets, the second N of GABRIEL_HERO), and none of those is read by
#: the report.
#: - ONED_RULES: the release oned results also hold the LOO and GCV rules; the current list is
#:   oracle and hold-out, the two the report reads.
#: - CONTROLS_IID, CONTROLS_M_EXTRA: the current values add N and targets (every N of every 2D
#:   target, m = 512 for every 2D target). The results they govern come from the run directory
#:   v7, whose manifests carry the current values; the manifests of the other run directories
#:   carry the smaller ones.
CONFIG_CHANGED = frozenset({"ONED_RULES", "CONTROLS_IID", "CONTROLS_M_EXTRA"})


def _same(a, b):
    """Equality of JSON values on the dict keys present in both; floats to 1e-14 relative
    (TIMES is computed by numpy, whose last bit differs between platforms)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return all(_same(a[k], b[k]) for k in a.keys() & b.keys())
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            math.isclose(a, b, rel_tol=1e-14, abs_tol=0.0)
    return a == b


def _release_manifests():
    """Full-mode manifests under $RESULTS_ROOT (every run directory) or results/."""
    roots = [pathlib.Path(os.environ["RESULTS_ROOT"])] if os.environ.get("RESULTS_ROOT") else []
    roots.append(ROOT / "results")
    out = []
    for r in roots:
        for p in sorted(r.glob("**/*.manifest.json")):
            if "smoke" in p.parts or "example" in p.parts:
                continue
            m = json.loads(p.read_text())
            if m.get("mode") == "full" and "config" in m:
                out.append((p, m["config"]))
    return out


def test_config_matches_release_manifests():
    """config.as_dict() equals the 'config' block of every release manifest on the keys both
    hold, up to CONFIG_CHANGED: the constants that computed the published results are
    unchanged."""
    mans = _release_manifests()
    if not mans:
        pytest.skip("no release manifests (set RESULTS_ROOT, see docs/REPRODUCE.md)")
    sys.path.insert(0, str(ROOT / "src"))
    from smoothfm import config
    cur = json.loads(json.dumps(config.as_dict()))
    bad = collections.Counter()
    for p, recorded in mans:
        for k in sorted(recorded.keys() & cur.keys() - CONFIG_CHANGED):
            if not _same(recorded[k], cur[k]):
                bad[f"{p.parent.parent.name}: {k}"] += 1
    assert not bad, f"config differences (run directory: key): {sorted(bad)[:20]}"
