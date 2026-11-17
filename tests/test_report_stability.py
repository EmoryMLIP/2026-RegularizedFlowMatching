"""Report stage, experiments/report/stability.py: fig_stability and tab_stability with and
without stability_circle_l2 (the smallness regularizer on the noisy circle).

Unit level: synthetic stability files (keys epsilon, <target>/ns, <target>/<flow>/tv and
dnorm2) in a temporary results directory; the outputs go to a temporary OUT."""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from release_helpers import REPORT, load_isolated

pytestmark = pytest.mark.report

C, R = load_isolated(REPORT, "stability", "rep_stability")
EPS = np.array([1e-6, 1e-4, 1e-2])
S_SETS, DRAWS, K = 2, 2, 4


def _write(path, targets_flows, scale):
    """npz with per target (N = ns[target]) and flow: tv = scale[flow] * epsilon and a field
    difference whose sup_t H^2 norm is 10 * scale[flow] * epsilon."""
    d = {"epsilon": EPS}
    for target, (n, flows) in targets_flows.items():
        d[f"{target}/ns"] = np.array([n])
        for f in flows:
            tv = np.broadcast_to(scale[f] * EPS[None, None, :, None], (1, S_SETS, len(EPS), DRAWS))
            d[f"{target}/{f}/tv"] = np.array(tv)
            if f != "exact":
                dn = np.zeros((1, S_SETS, len(EPS), DRAWS, K, 3))
                dn[..., 2] = (10 * scale[f] * EPS[None, None, :, None, None]) ** 2
                d[f"{target}/{f}/dnorm2"] = dn
    np.savez(path, **d)


SCALE = {"exact": 1.0, "h2_fixed": 2.0, "h2_resel": 3.0, "l2_fixed": 5.0, "l2_resel": 7.0}
ALL = ["exact", "h2_fixed", "h2_resel", "l2_fixed", "l2_resel"]


@pytest.fixture
def dirs(tmp_path):
    res, out = tmp_path / "results", tmp_path / "out"
    res.mkdir()
    _write(res / "stability_1d.npz", {"trimodal": (100, ALL), "inhom": (100, ALL)}, SCALE)
    _write(res / "stability_circle.npz", {"circle": (1000, ["exact", "h2_fixed", "h2_resel"])},
           SCALE)
    C.set_out(str(out))
    return res, out


def _build(res):
    C.use(str(res))
    return R.build()


def _rows(tex):
    return [ln for ln in tex.splitlines() if not ln.startswith("%") and "&" in ln]


def _blocks(tex):
    """{(panel, label text): [row cells]} of tab_stability: the rows under each rotated
    label (target and N), panel 'a' or 'b'; the label cell is dropped."""
    out, panel, key = {}, None, None
    for ln in tex.splitlines():
        if ln.startswith("%"):
            continue
        if ln.startswith("\\multicolumn"):
            panel = "a" if "(a)" in ln else "b"
            continue
        if ln.startswith("\\midrule"):
            key = None
            continue
        if panel is None or "&" not in ln:
            continue
        cells = [c.strip() for c in ln.rstrip().removesuffix("\\\\").split("&")]
        if cells[0].startswith("\\multirow"):
            lab = re.search(r"\\begin\{tabular\}\{@\{\}c@\{\}\}(.*?)\\end", cells[0]).group(1)
            key = (panel, lab)
            out[key] = []
        assert key is not None, ln
        out[key].append(cells[1:])
    return out


CIRCLE = "Circle\\\\$N = 1000$"


def test_with_circle_l2(dirs):
    res, out = dirs
    _write(res / "stability_circle_l2.npz", {"circle": (1000, ["l2_fixed", "l2_resel"])}, SCALE)
    written = _build(res)
    fig = (out / "figures" / "fig_stability.tex").read_text()
    for f in ("l2_fixed", "l2_resel"):
        # panel (b) TV and panel (c) H^2 for the circle, panel (c) for the trimodal target
        for rel in (f"fig_stability/tv_circle_n1000_{f}.dat",
                    f"fig_stability/h2_circle_n1000_{f}.dat",
                    f"fig_stability/h2_trimodal_n100_{f}.dat"):
            assert f"figures/{rel}" in written and rel in fig, rel
            assert (out / "figures" / rel).is_file()
    m = np.loadtxt(out / "figures" / "fig_stability" / "h2_circle_n1000_l2_resel.dat",
                   skiprows=2)[:, 1]
    np.testing.assert_allclose(m, 10 * SCALE["l2_resel"] * EPS)
    assert "not in the results directory" not in fig
    tab = (out / "tables" / "tab_stability.tex").read_text()
    b = _blocks(tab)
    # (a) TV and (b) H^2: exact (a only), two k = 2 rows, two k = 0 rows
    flows = [R.FLOW_STYLE[f][0] for f in ("h2_fixed", "h2_resel", "l2_fixed", "l2_resel")]
    assert [r[0] for r in b[("a", CIRCLE)]] == ["exact $v_N$"] + flows
    assert [r[0] for r in b[("b", CIRCLE)]] == flows
    assert r"\tbd" not in "\n".join(_rows(tab))


def test_without_circle_l2(dirs):
    res, out = dirs
    written = _build(res)
    fig = (out / "figures" / "fig_stability.tex").read_text()
    assert not any("circle_n1000_l2" in w for w in written)
    assert "circle_n1000_l2" not in fig
    # the 1D smallness curves of panel (c) do not depend on the circle file
    assert "fig_stability/h2_trimodal_n100_l2_fixed.dat" in fig
    note = "% stability_circle_l2.npz not in the results directory"
    assert note in fig
    tab = (out / "tables" / "tab_stability.tex").read_text()
    assert note in tab
    b = _blocks(tab)
    circ = b[("a", CIRCLE)] + b[("b", CIRCLE)]
    assert len(circ) == 3 + 2 and not any("k = 0" in r[0] for r in circ)
    assert r"\tbd" not in "\n".join(_rows(tab))


def test_legend_and_rows_use_the_order_labels(dirs):
    res, out = dirs
    _build(res)
    for p in ("figures/fig_stability.tex", "tables/tab_stability.tex"):
        body = "\n".join(ln for ln in (out / p).read_text().splitlines()
                         if not ln.startswith("%"))
        assert "B_" not in body, p
        for k in (0, 2):
            assert f"$k = {k}$, fixed $\\lambda$" in body, (p, k)
            assert f"$k = {k}$, re-selected $\\lambda$" in body, (p, k)


def test_missing_circle_files_draw_missing_panel(dirs):
    res, out = dirs
    os.remove(res / "stability_circle.npz")
    _build(res)
    fig = (out / "figures" / "fig_stability.tex").read_text()
    panels = fig.split("\\nextgroupplot")[1:]
    assert "missing" in panels[1] and "missing" not in panels[0]


# ---------------------------------------------------------------- table layout

def test_table_entries_are_the_means_over_epsilon(dirs):
    """(a) TV / eps and (b) sup_t H^2 / eps per epsilon column: the synthetic files have
    TV = scale * eps and H^2 = 10 * scale * eps, so every column prints scale (10 scale);
    the slope stays that of the undivided means (1)."""
    res, out = dirs
    _build(res)
    b = _blocks((out / "tables" / "tab_stability.tex").read_text())
    for (panel, lab), rows in b.items():
        for r in rows:
            f = next(k for k, (txt, _) in R.FLOW_STYLE.items() if txt == r[0])
            want = C.fmt_e(SCALE[f] * (1 if panel == "a" else 10))
            assert r[1:-1] == [want] * len(EPS), (panel, lab, r)
            assert r[-1] == "$1.00$", (panel, lab, r)


def test_table_has_one_rotated_label_per_block_and_no_target_column(dirs):
    res, out = dirs
    _build(res)
    tab = (out / "tables" / "tab_stability.tex").read_text()
    body = "\n".join(_rows(tab))
    assert "Target" not in body and "$N$ &" not in body
    assert "(a) $\\mathrm{TV}(p_\\varepsilon, p_0)/\\varepsilon$" in tab
    assert "(b) $\\sup_t \\|v_t - v_t'\\|_{H^2(\\Omega)}/\\varepsilon$" in tab
    b = _blocks(tab)
    assert sorted(b) == sorted((p, f"{n}\\\\$N = {N}$") for p in "ab" for n, N in
                               [("Trimodal", 100), ("Inhom.", 100), ("Circle", 1000)])
    assert tab.count("\\rotatebox") == len(b)
    # \multirow spans exactly the rows of its block
    spans = [int(x) for x in re.findall(r"\\multirow\{(\d+)\}", tab)]
    assert spans == [len(rows) for rows in b.values()]


def test_refine_table_rotated_label_per_target(dirs):
    res, out = dirs
    for m in R.REF_MS:
        _write(res / f"stability_1d_m{m}.npz", {"trimodal": (100, ALL), "inhom": (100, ALL)},
               SCALE)
    C.use(str(res))
    R.write_refine_table()
    tab = (out / "tables" / "tab_stability_refine.tex").read_text()
    body = "\n".join(_rows(tab))
    assert "Target" not in body
    # per target: 4 estimators x (3 m rows + growth), 3 \addlinespace between them
    assert re.findall(r"\\multirow\{(\d+)\}\{\*\}\[(-?[\d.]+)\\defaultaddspace\]", tab) == \
        [("16", "-1.5")] * 2
    assert "Trimodal\\\\$N = 100$" in tab and "Inhom.\\\\$N = 100$" in tab
