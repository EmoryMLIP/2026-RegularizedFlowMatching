"""Compose the results directory of the paper from several release runs (`make paper-report`).
Paper: the inputs of every generated table and figure (docs/REPRODUCE.md).

Usage:
  python experiments/report/compose_results.py [--spec FILE] [--root DIR] [--dest DIR]
         [--sources FILE]

The spec (default experiments/report/paper_results.txt) lists result directories relative to
the results root (default env RESULTS_ROOT, else results/ in the repository), one per line, '#' comments. Each file of a listed directory
(subdirectories such as smoke/ are not entered) becomes a symlink DEST/<file name>; a later
directory overrides an earlier one per file name, so a directory that holds a few results
replaces only those. An absent directory is skipped with a note. DEST (default
ROOT/paper/results) is recreated each time; it may hold only symlinks, so a composition never
deletes a result file. SOURCES (default ROOT/paper/SOURCES.txt) lists, per file, the directory
it came from. The report run on DEST names it ('paper/results') in every header.
Standard library only, so that the Makefile can call it before anything else.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC = os.path.join(HERE, "paper_results.txt")
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")


def read_spec(path) -> list:
    """The directories of the spec in order (relative to the results root)."""
    with open(path) as f:
        return [ln.split("#", 1)[0].strip() for ln in f if ln.split("#", 1)[0].strip()]


def compose(spec_dirs, root, dest) -> tuple:
    """Link every file of root/<dir> for dir in spec_dirs into dest, later dirs winning.

    Returns ({file name: source dir as listed}, [skipped dirs]). Raises RuntimeError, before
    touching anything, if dest holds an entry that is not a symlink."""
    root, dest = os.path.abspath(os.path.expanduser(root)), os.path.abspath(dest)
    if os.path.lexists(dest) and not os.path.isdir(dest):
        raise RuntimeError(f"{dest} exists and is not a directory")
    if os.path.isdir(dest):
        real = sorted(e for e in os.listdir(dest) if not os.path.islink(os.path.join(dest, e)))
        if real:
            raise RuntimeError(f"{dest} holds entries that are not symlinks ({', '.join(real[:5])}"
                               f"{', ...' if len(real) > 5 else ''}); refusing to recreate it")
    chosen, skipped = {}, []
    for rel in spec_dirs:
        src = os.path.join(root, rel)
        if not os.path.isdir(src):
            skipped.append(rel)
            continue
        for e in sorted(os.listdir(src)):
            if os.path.isfile(os.path.join(src, e)):
                chosen[e] = rel
    os.makedirs(dest, exist_ok=True)
    for e in os.listdir(dest):
        os.unlink(os.path.join(dest, e))
    for e, rel in sorted(chosen.items()):
        # relative links survive moving the whole results root
        target = os.path.relpath(os.path.join(root, rel, e), dest)
        os.symlink(target, os.path.join(dest, e))
    return chosen, skipped


def sources_text(spec_dirs, chosen, skipped, dest) -> str:
    lines = ["# Composition of " + dest + " (experiments/report/compose_results.py);",
             "# spec order, later directories override earlier ones per file name.",
             "# spec: " + ", ".join(spec_dirs)]
    lines += [f"# skipped (absent): {rel}" for rel in skipped]
    lines += [f"{e}\t{rel}" for e, rel in sorted(chosen.items())]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--root", default=os.environ.get("RESULTS_ROOT") or DEFAULT_ROOT)
    ap.add_argument("--dest", default=None, help="default ROOT/paper/results")
    ap.add_argument("--sources", default=None, help="default ROOT/paper/SOURCES.txt")
    a = ap.parse_args(argv)
    root = os.path.abspath(os.path.expanduser(a.root))
    dest = a.dest or os.path.join(root, "paper", "results")
    sources = a.sources or os.path.join(root, "paper", "SOURCES.txt")
    spec = read_spec(a.spec)
    try:
        chosen, skipped = compose(spec, root, dest)
    except RuntimeError as e:
        sys.exit(f"compose_results: {e}")
    for rel in skipped:
        print(f"compose_results: {rel} absent, skipped")
    with open(sources, "w") as f:
        f.write(sources_text(spec, chosen, skipped, dest))
    counts = {rel: sum(1 for r in chosen.values() if r == rel) for rel in spec}
    print(f"compose_results: {len(chosen)} files in {dest} ("
          + ", ".join(f"{rel} {k}" for rel, k in counts.items()) + f"); {sources}")


if __name__ == "__main__":
    main()
