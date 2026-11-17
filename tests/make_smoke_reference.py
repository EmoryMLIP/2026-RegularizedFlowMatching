"""Write tests/fixtures/smoke_reference.json from a directory of smoke results.

Usage: python tests/make_smoke_reference.py SMOKE_DIR [--out FILE]

SMOKE_DIR holds <name>.npz for every result name of experiments/runs.txt (`make smoke` writes
results/smoke/). For every numeric array outside the timing and provenance keys the reference
stores shape, dtype, sum, sum of squares, min, max and the count of non-finite entries
(release_helpers.npz_summaries), and for the outcome metrics and their SEs also the entries;
tests/test_scripts.py compares smoke reruns with it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from release_helpers import SMOKE_REFERENCE, npz_summaries, run_names  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("smoke_dir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=SMOKE_REFERENCE)
    a = ap.parse_args(argv)
    names = run_names()
    missing = [n for n in names if not (a.smoke_dir / f"{n}.npz").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} results missing in {a.smoke_dir}: {missing[:10]}")
    ref = {n: npz_summaries(a.smoke_dir / f"{n}.npz") for n in names}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # one line per array, so that a change shows as a line diff
    with open(a.out, "w") as f:
        f.write("{\n")
        for i, n in enumerate(names):
            f.write(f"{json.dumps(n)}: {{\n")
            keys = list(ref[n])
            for j, k in enumerate(keys):
                sep = "," if j < len(keys) - 1 else ""
                f.write(f"  {json.dumps(k)}: {json.dumps(ref[n][k])}{sep}\n")
            f.write("}" + ("," if i < len(names) - 1 else "") + "\n")
        f.write("}\n")
    print(f"{a.out}: {len(names)} results, {sum(len(v) for v in ref.values())} arrays")


if __name__ == "__main__":
    main()
