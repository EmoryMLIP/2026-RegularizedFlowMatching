"""Publish the release archive of the results (release_results.py) to a web server, verify
the public copy, and record its URL and sha256 in experiments/report/results_archive.txt.
Paper: none (distribution of the results behind every table and figure).

Usage:
  uv run python experiments/report/upload_results.py --archive FILE [--dest DEST] [--url URL]
         [--yes] [--verify-only] [--conf FILE]

--archive      the local <name>.tar.gz; <name>.tar.gz.sha256 must lie beside it
               (env RESULTS_ARCHIVE)
--dest         rsync destination of the directory that serves the archive, e.g.
               user@server:/path/ or a local directory (env RESULTS_DEST)
--url          public URL of the archive, https://.../<name>.tar.gz; file:// works for tests
               (env RESULTS_URL)
--yes          upload; without it the script verifies the local archive and prints the
               commands it would run (dry run, nothing leaves the machine)
--verify-only  skip the upload, verify the public copy and record it (after a manual upload)
--conf         the file to update (default experiments/report/results_archive.txt)

Steps: (1) the archive's sha256 equals its .sha256 file, and every member of the archive equals
the SHA256SUMS inside it; (2) rsync the archive and its .sha256 to DEST; (3) download URL to a
temporary file and compare its sha256 with the local one; (4) only then write 'url' and
'sha256' into the config. No destination, host or credential is stored here: rsync uses the
caller's ssh configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(HERE, "results_archive.txt")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_local(archive) -> str:
    """The archive's sha256 after checking it against <archive>.sha256 and its members against
    the SHA256SUMS inside; raises SystemExit on a mismatch."""
    side = archive + ".sha256"
    if not os.path.isfile(side):
        sys.exit(f"upload_results: {side} missing")
    with open(side) as f:
        want = f.read().split()[0]
    got = sha256_file(archive)
    if got != want:
        sys.exit(f"upload_results: {archive} sha256 {got} != {want} ({side})")
    sums, hashes = None, {}
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            top, _, rel = m.name.partition("/")
            data = tf.extractfile(m).read()
            if rel == "SHA256SUMS":
                sums = data.decode()
            else:
                hashes[rel] = hashlib.sha256(data).hexdigest()
    if sums is None:
        sys.exit("upload_results: no SHA256SUMS in the archive")
    listed = dict(reversed(ln.split("  ", 1)) for ln in sums.splitlines() if ln.strip())
    if listed != hashes:
        bad = sorted(k for k in set(listed) | set(hashes) if listed.get(k) != hashes.get(k))
        sys.exit(f"upload_results: SHA256SUMS does not match the archive: {bad[:5]}")
    print(f"local archive OK: {len(hashes)} files match SHA256SUMS, sha256 {got}")
    return got


def verify_remote(url, want) -> None:
    """Download url to a temporary file and compare its sha256 with want."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "download")
        with urllib.request.urlopen(url) as r, open(p, "wb") as f:
            shutil.copyfileobj(r, f)
        got = sha256_file(p)
    if got != want:
        sys.exit(f"upload_results: the copy at {url} has sha256 {got}, expected {want}; "
                 "the config is unchanged")
    print(f"public copy OK: {url}")


def write_conf(conf, url, sha) -> None:
    with open(conf) as f:
        lines = f.read().splitlines()
    out, seen = [], set()
    for ln in lines:
        w = ln.split()
        if w and w[0] in ("url", "sha256") and not ln.startswith("#"):
            out.append(f"{w[0]} {url if w[0] == 'url' else sha}")
            seen.add(w[0])
        elif re.match(r"#\s*TODO\(author\): replace by the public URL", ln):
            continue                      # the placeholder note is obsolete once url is set
        else:
            out.append(ln)
    out += [f"{k} {url if k == 'url' else sha}" for k in ("url", "sha256") if k not in seen]
    with open(conf, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"recorded url and sha256 in {conf}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--archive", default=os.environ.get("RESULTS_ARCHIVE"))
    ap.add_argument("--dest", default=os.environ.get("RESULTS_DEST"))
    ap.add_argument("--url", default=os.environ.get("RESULTS_URL"))
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--conf", default=CONF)
    a = ap.parse_args(argv)
    if not a.archive or not a.url or not (a.dest or a.verify_only):
        ap.error("--archive, --url and --dest (or --verify-only) are required "
                 "(or RESULTS_ARCHIVE, RESULTS_URL, RESULTS_DEST)")
    archive = os.path.abspath(a.archive)
    if os.path.basename(a.url.rstrip("/")) != os.path.basename(archive):
        sys.exit(f"upload_results: the URL must end in {os.path.basename(archive)}")
    sha = verify_local(archive)
    cmd = ["rsync", "-v", "-p", "--checksum", archive, archive + ".sha256", a.dest or ""]
    if not a.verify_only:
        print("upload: " + shlex.join(cmd))
    print(f"verify: download {a.url}, expect sha256 {sha}")
    print(f"record: url {a.url} and sha256 {sha} in {a.conf}")
    if not (a.yes or a.verify_only):
        print("dry run: nothing uploaded or written (pass --yes to upload)")
        return
    if not a.verify_only:
        if subprocess.run(cmd).returncode != 0:
            sys.exit("upload_results: rsync failed; the config is unchanged")
    verify_remote(a.url, sha)
    write_conf(a.conf, a.url, sha)


if __name__ == "__main__":
    main()
