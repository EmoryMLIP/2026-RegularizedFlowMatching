"""Pack a compact results directory into a deterministic tar.gz (`make compact`).

Usage: uv run python experiments/report/compact_archive.py DIR ARCHIVE

The archive holds DIR's files under compact/, in sorted order, with zero timestamps, owners
and group, modes 0644 (files) and 0755 (directories), and a gzip header without name or time,
so the same compact results give the same bytes on every platform. Prints the sha256 of the
archive. release_results.py packs the release archive of the results the same way.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import sys
import tarfile

TOP = "compact"


def _info(name, size=0, isdir=False):
    ti = tarfile.TarInfo(name)
    ti.size = size
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.type = tarfile.DIRTYPE if isdir else tarfile.REGTYPE
    ti.mode = 0o755 if isdir else 0o644
    return ti


def pack(src, archive, top=TOP):
    """Pack src under top/ (default compact/); returns the archive's sha256."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        tf.addfile(_info(top, isdir=True))
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames.sort()
            rel = os.path.relpath(dirpath, src)
            base = top if rel == "." else f"{top}/{rel.replace(os.sep, '/')}"
            if rel != ".":
                tf.addfile(_info(base, isdir=True))
            for fn in sorted(filenames):
                if fn == ".DS_Store":
                    continue
                with open(os.path.join(dirpath, fn), "rb") as f:
                    data = f.read()
                tf.addfile(_info(f"{base}/{fn}", len(data)), io.BytesIO(data))
    with open(archive, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                   mtime=0) as gz:
        gz.write(buf.getvalue())
    with open(archive, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__.split("\n\n", 2)[1])
    src, archive = argv
    if not os.path.isdir(src):
        raise SystemExit(f"{src}: no such directory")
    print(f"{archive}: sha256 {pack(src, archive)}")


if __name__ == "__main__":
    main()
