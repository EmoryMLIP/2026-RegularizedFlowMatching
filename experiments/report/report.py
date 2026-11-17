"""Report stage (seconds): every table and figure the paper \\inputs, plus the
cost table, from ONE results directory. Paper: all of docs/REPRODUCE.md.

Usage:
  uv run python experiments/report/report.py [--results DIR] [--out DIR] [--compact-out DIR]
                                             [--only MODULE ...]

--results      results directory with the result names of experiments/runs.txt
               (default results/); smoke results are just another directory
               (results/smoke). No fallbacks: a missing file prints \\tbd in its table
               cells and 'missing' in its figure panels, and is listed on stdout.
--out          writes OUT/tables and OUT/figures (default experiments/paper).
--only         runs only these modules of MODULES (`make hero`: hero, the hero figure
               fig_hero); skips the OUTPUTS.txt check.
--compact-out  after the report, writes DIR/<name>.npz with exactly the keys the report read
               (images float32, compressed) and copies every manifest
               (`make compact`); the report from DIR equals the report from --results up to
               the '% results directory' / '% source' comment lines.

The outputs are deterministic (no timestamps, no absolute paths, sorted iteration): two runs
on the same results give byte-identical files. experiments/report/OUTPUTS.txt lists the
.tex outputs (the paper's \\input targets and tables/tab_cost.tex).
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common as C  # noqa: E402

# report modules in run order; each has build() -> list of written paths relative to OUT
MODULES = ["tables", "figures", "hero", "stability", "prose_numbers", "cost"]


def outputs_txt():
    with open(os.path.join(HERE, "OUTPUTS.txt")) as f:
        return [ln.split("#", 1)[0].strip() for ln in f if ln.split("#", 1)[0].strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--results", default=C.DEFAULT_RESULTS, metavar="DIR")
    ap.add_argument("--out", default=C.DEFAULT_OUT, metavar="DIR")
    ap.add_argument("--compact-out", default=None, metavar="DIR")
    ap.add_argument("--only", nargs="+", choices=MODULES, metavar="MODULE")
    a = ap.parse_args(argv)
    res = C.use(a.results)
    C.set_out(a.out)
    print(f"results: {res.label()}")
    written = []
    for name in a.only or MODULES:
        mod = __import__(name)
        written += mod.build()
    expected = [] if a.only else outputs_txt()
    missing_out = [p for p in expected if not os.path.exists(os.path.join(C.OUT, p))]
    if missing_out:
        raise SystemExit(f"outputs of OUTPUTS.txt not written: {missing_out}")
    print(f"wrote {len(written)} files under {a.out}")
    if res.missing():
        print("missing results (\\tbd cells, 'missing' panels): " + ", ".join(res.missing()))
    absent = sorted(res.absent_keys)
    if absent:
        print(f"keys looked up but absent ({len(absent)}; optional keys included):")
        for name, key in absent:
            print(f"  {name}: {key}")
    if a.compact_out:
        res.write_compact(a.compact_out)
        print(f"compact results: {a.compact_out}")


if __name__ == "__main__":
    main()
