"""Build the release archive of the results: the files the report stage reads, with sanitized
manifests, a README, SHA256SUMS and one deterministic tar.gz. Paper: the inputs of every
generated table, figure and numbers.txt line (docs/REPRODUCE.md).

Usage:
  uv run python experiments/report/release_results.py --composed DIR --sources FILE
         --readme FILE --dest DIR

--composed  the composed results directory of the paper (`make paper-report`:
            RESULTS_ROOT/paper/results, symlinks are followed)
--sources   its SOURCES.txt (file name -> run directory)
--readme    the README of the archive (experiments/report/RESULTS_README.md)
--dest      output directory; writes DEST/<name>/ and DEST/<name>.tar.gz(.sha256), with <name>
            from experiments/report/results_archive.txt. DEST/<name> must not exist.

Which files: the report runs once on --composed under an audit hook that records every file
it opens there. The archive holds every .npz it opened, unchanged (byte copies), and the
manifest of every run of experiments/runs.txt (the cost table reads them; the report opens
the manifests of the directory, and the others change no output), and the run log <name>.log
of every run of experiments/runs.txt (the paper quotes some numbers from logs only). A file
other than .npz or .manifest.json opened by the report stops the build.

Sanitized manifests and logs (sanitize_text): every absolute path is replaced, up to its last
'/results' component, by '<results>', up to its 'src' or 'experiments' component by '<code>',
else by '<path>'. An absolute path starts after whitespace, a quote, '=', '(', '[', '{', ','
or ':' and has at least two components, so that no number or ratio such as 'lam/lam_s' or
'(t)/3' is touched. Manifests lose the key 'host'; the header line of a log records
'host unrecorded' (sanitize_log). A manifest or log of a dirty tree stops the build. Commit,
tree state, command, parameters, versions, CPU model, timing, every number and the output
checksums stay. Arrays are never rewritten.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compact_archive  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
CONF = os.path.join(HERE, "results_archive.txt")
RUNS = os.path.join(ROOT, "experiments", "runs.txt")
SUBDIR = os.path.join("paper", "results")   # the report names it 'paper/results' in headers
DROP_KEYS = {"host"}
# an absolute path: after start, whitespace, a quote or = ( [ { , : ; two or more components
_ABS = re.compile(r"(?<![^\s'\"=(\[{,:])/[A-Za-z0-9_.~-]+(?:/[^\s'\",;)\]}]*)+")
# the host field of a log header ('... pot 0.9.7  host <name> (<cpu>, ...')
_LOG_HOST = re.compile(r"^(# python .*?\bhost )(\S+)( \()", re.M)
_LOG_DIRTY = re.compile(r"^# \S+  commit \S+ dirty ", re.M)


def read_conf(path=CONF) -> dict:
    """The 'key value' lines of results_archive.txt ('#' comments)."""
    out = {}
    with open(path) as f:
        for ln in f:
            w = ln.split("#", 1)[0].split()
            if len(w) >= 2:
                out[w[0]] = w[1]
    return out


def run_names() -> list:
    with open(RUNS) as f:
        return [ln.split()[0] for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def _path_token(m) -> str:
    p = m.group(0)
    k = p.rfind("/results")
    if k >= 0 and (k + 8 == len(p) or p[k + 8] == "/"):
        return "<results>" + p[k + 8:]
    for comp in ("/src/", "/experiments/"):
        k = p.find(comp)
        if k >= 0:
            return "<code>" + p[k:]
    return "<path>"


def sanitize_text(s: str) -> str:
    """s with every absolute path replaced by <results>/..., <code>/... or <path>."""
    return _ABS.sub(_path_token, s)


def sanitize_log(text: str) -> str:
    """A run log without absolute paths and with 'host unrecorded' in its header; raises
    ValueError for the log of a dirty tree."""
    if _LOG_DIRTY.search(text):
        raise ValueError("log of a dirty tree")
    return _LOG_HOST.sub(r"\1unrecorded\3", sanitize_text(text))


def sanitize(x):
    """Copy of a manifest (dict/list/str/number) without DROP_KEYS and absolute paths."""
    if isinstance(x, dict):
        return {k: sanitize(v) for k, v in x.items() if k not in DROP_KEYS}
    if isinstance(x, list):
        return [sanitize(v) for v in x]
    if isinstance(x, str):
        return sanitize_text(x)
    return x


def traced_report(composed) -> list:
    """Run the report on composed (outputs to a temporary directory); return the names of the
    files it opened in composed, sorted."""
    composed = os.path.abspath(composed)
    seen = set()

    def hook(ev, args):
        if ev in ("open", "os.listdir", "os.scandir") and args and \
                isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fspath(args[0])
            p = os.path.abspath(p.decode() if isinstance(p, bytes) else p)
            if os.path.dirname(p) == composed:
                seen.add(os.path.basename(p))

    sys.addaudithook(hook)   # stays installed; it only records paths under composed
    import report
    with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
        report.main(["--results", composed, "--out", out])
    return sorted(seen)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--composed", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--readme", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    name = read_conf()["name"]
    top = os.path.join(os.path.abspath(a.dest), name)
    if os.path.lexists(top):
        sys.exit(f"release_results: {top} exists; remove it first")

    opened = traced_report(a.composed)
    other = [f for f in opened if not f.endswith((".npz", ".manifest.json"))]
    if other:
        sys.exit(f"release_results: the report opened files other than results: {other}")
    npz = [f for f in opened if f.endswith(".npz")]
    runs = run_names()
    manifests = [n + ".manifest.json" for n in runs]
    logs = [n + ".log" for n in runs]
    absent = [f for f in manifests + logs if not os.path.exists(os.path.join(a.composed, f))]
    if absent:
        sys.exit(f"release_results: manifests or logs of the run list missing: {absent}")
    with open(a.sources) as f:
        src = dict(ln.rstrip("\n").split("\t") for ln in f if ln.strip() and not ln.startswith("#"))

    res = os.path.join(top, SUBDIR)
    os.makedirs(res)
    commits = {}
    for f in manifests:
        with open(os.path.join(a.composed, f)) as g:
            m = json.load(g)
        if m.get("clean") is not True:
            sys.exit(f"release_results: {f} records a dirty tree (clean {m.get('clean')!r})")
        commits[f[:-len(".manifest.json")]] = str(m.get("commit", "?"))
        with open(os.path.join(res, f), "w") as g:
            json.dump(sanitize(m), g, indent=1)
            g.write("\n")
    for f in logs:
        with open(os.path.join(a.composed, f), encoding="utf-8", newline="") as g:
            text = g.read()
        try:
            text = sanitize_log(text)
        except ValueError as e:
            sys.exit(f"release_results: {f}: {e}")
        with open(os.path.join(res, f), "w", encoding="utf-8", newline="") as g:
            g.write(text)
    for f in npz:
        shutil.copyfile(os.path.join(a.composed, f), os.path.join(res, f))   # follows links

    files = sorted(npz + manifests + logs)
    with open(os.path.join(top, "SOURCES.txt"), "w") as g:
        g.write("# file\trun directory of the original result store\tcommit of the authors' "
                "development repository (manifest field 'commit')\n")
        for f in files:
            stem = f.split(".", 1)[0]
            g.write(f"{SUBDIR}/{f}\t{src.get(f, '?')}\t{commits.get(stem, '?')}\n")
    shutil.copyfile(a.readme, os.path.join(top, "README.md"))

    entries = []
    for dirpath, _, fns in os.walk(top):
        for fn in fns:
            rel = os.path.relpath(os.path.join(dirpath, fn), top).replace(os.sep, "/")
            if rel != "SHA256SUMS":
                entries.append(rel)
    with open(os.path.join(top, "SHA256SUMS"), "w") as g:
        for rel in sorted(entries):
            g.write(f"{sha256(os.path.join(top, rel))}  {rel}\n")

    archive = top + ".tar.gz"
    digest = compact_archive.pack(top, archive, top=name)
    with open(archive + ".sha256", "w") as g:
        g.write(f"{digest}  {os.path.basename(archive)}\n")
    print(f"{len(npz)} .npz, {len(manifests)} manifests and {len(logs)} logs -> {top}")
    print(f"{archive}: {os.path.getsize(archive) / 1e6:.1f} MB, sha256 {digest}")
    print(f"enter the sha256 in {os.path.relpath(CONF, ROOT)}")


if __name__ == "__main__":
    main()
