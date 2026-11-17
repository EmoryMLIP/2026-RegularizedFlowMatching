"""The release archive of the results (experiments/report/release_results.py,
results_archive.txt, `make fetch-results`). Paper: the inputs of every generated table,
figure and numbers.txt line.

Marker ``report``. The archive tests read the unpacked archive from
``$SMOOTHFM_RELEASE_DIR`` (the directory <name>/ with SHA256SUMS) or from
``RESULTS_ROOT/<name>`` (``$RESULTS_ROOT``, default results/ in the repository), where
``make fetch-results`` unpacks it, and skip when it is absent; the downloaded
``<name>.tar.gz`` beside it is checked against results_archive.txt when present. The config,
sanitizer and placeholder tests need no archive.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from release_helpers import REPORT, ROOT, env, read_outputs_txt

pytestmark = pytest.mark.report

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _release_module():
    sys.path.insert(0, str(REPORT))
    try:
        spec = importlib.util.spec_from_file_location("release_results",
                                                      REPORT / "release_results.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(REPORT))
    return mod


REL = _release_module()
CONF = REL.read_conf()


def _sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _release_dir() -> pathlib.Path:
    env_dir = os.environ.get("SMOOTHFM_RELEASE_DIR")
    root = pathlib.Path(os.environ.get("RESULTS_ROOT") or ROOT / "results")
    p = pathlib.Path(env_dir) if env_dir else root / CONF["name"]
    if not (p / "SHA256SUMS").is_file():
        msg = f"no unpacked release archive at {p} (make fetch-results or $SMOOTHFM_RELEASE_DIR)"
        if os.environ.get("SMOOTHFM_REQUIRE_RESULTS"):
            pytest.fail(msg)
        pytest.skip(msg)
    return p


# ---------------------------------------------------------------- no archive needed

def test_config_names_archive_url_and_sha256():
    assert CONF.get("name"), "results_archive.txt: no name"
    assert CONF.get("url"), "results_archive.txt: no url"
    sha = CONF.get("sha256", "")
    assert HEX64.match(sha) or "TODO" in sha, sha


def test_fetch_refuses_placeholder_url(tmp_path):
    if "TODO" not in CONF["url"] + CONF["sha256"]:
        pytest.skip("results_archive.txt names a published archive")
    e = env()
    e.pop("RESULTS_URL", None)
    e.pop("RESULTS_SHA256", None)
    r = subprocess.run(["make", "-s", "-C", str(ROOT), "fetch-results",
                        f"RESULTS_ROOT={tmp_path}"], env=e, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode != 0
    assert "placeholder" in r.stdout + r.stderr
    assert not any(tmp_path.iterdir()), "nothing may be written before the URL is known"


def test_sanitize_drops_host_and_paths_keeps_values():
    m = {"host": "node1", "commit": "abc", "clean": True, "wall_s": 1.5, "workers": 3,
         "command": "experiments/compute/oned.py --results /x/y/run/results --workers 3",
         "argv": ["experiments/compute/oned.py", "--results", "/x/y/run/results"],
         "args": {"results_dir": "/x/y/run/results", "ns": [100, 1000]},
         "setup": {"source": "/x/y/run/results/oned.npz sha256 0123", "other": "/opt/data"},
         "jobs": {"a N=100": {"nsolves": 7, "metrics": {"1e-04/tv": 0.25, "ratio": "a/b"}}}}
    s = REL.sanitize(m)
    assert "host" not in s
    assert s["command"] == "experiments/compute/oned.py --results <results> --workers 3"
    assert s["argv"][2] == s["args"]["results_dir"] == "<results>"
    assert s["setup"] == {"source": "<results>/oned.npz sha256 0123", "other": "<path>"}
    assert s["jobs"] == m["jobs"] and s["args"]["ns"] == [100, 1000]
    assert (s["commit"], s["clean"], s["wall_s"], s["workers"]) == ("abc", True, 1.5, 3)


def test_sanitize_text_replaces_paths_only():
    f = REL.sanitize_text
    assert f("Outputs: /a/b/run/results/oned.npz (this file), /a/b/run/results/oned.log") == \
        "Outputs: <results>/oned.npz (this file), <results>/oned.log"
    assert f("source: /a/b/run/results/x.npz sha256 0123") == "source: <results>/x.npz sha256 0123"
    assert f("{'twod': '/a/b/results/t.npz', 'c': '/a/b/results/c.npz'}") == \
        "{'twod': '<results>/t.npz', 'c': '<results>/c.npz'}"
    assert f("module /a/b/repo/src/smoothfm/x.py") == "module <code>/src/smoothfm/x.py"
    assert f("cache /opt/data") == "cache <path>"
    # ratios, units and numbers are not paths
    for s in ("lam/lam_s 3.1e+06", "1e-04/tv_pos 0.25", "(flow - KDE) / KDE", "zeta(u)/3",
              "N(0, I)/2", "q(u) = 2u / (3 pi)^2", "x in [-4.5, 4.5]^2", "1/2"):
        assert f(s) == s, s


def test_sanitize_log_header_and_dirty_tree():
    head = ("# oned  commit 0123456789ab clean  source_sha256 abc  2026-01-01T00:00:00  mode full\n"
            "# python 3.12.1 numpy 2.0 scipy 1.1 cvxopt 1.3 pot 0.9  host node7 (Intel CPU @ 2GHz, "
            "40 cores)  workers 15\n"
            "# command: experiments/compute/oned.py --results /x/y/results --workers 15\n"
            "t 0.5 tv 0.123 lam/lam_s 1e-3\n")
    out = REL.sanitize_log(head)
    assert "host unrecorded (Intel CPU @ 2GHz, 40 cores)" in out and "node7" not in out
    assert "--results <results> --workers 15" in out
    assert out.splitlines()[0] == head.splitlines()[0]
    assert out.splitlines()[-1] == "t 0.5 tv 0.123 lam/lam_s 1e-3"
    with pytest.raises(ValueError):
        REL.sanitize_log(head.replace(" clean ", " dirty "))


# ---------------------------------------------------------------- the unpacked archive

def test_archive_file_matches_config_sha256():
    d = _release_dir()
    tgz = d.parent / f"{CONF['name']}.tar.gz"
    if not tgz.is_file():
        pytest.skip(f"no downloaded archive at {tgz}")
    if not HEX64.match(CONF["sha256"]):
        pytest.skip("results_archive.txt holds no sha256")
    assert _sha256(tgz) == CONF["sha256"]


def test_sha256sums_cover_and_match_every_file():
    d = _release_dir()
    listed = {}
    for ln in (d / "SHA256SUMS").read_text().splitlines():
        h, rel = ln.split("  ", 1)
        listed[rel] = h
    present = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}
    present.discard("SHA256SUMS")
    assert set(listed) == present, sorted(set(listed) ^ present)[:10]
    bad = [rel for rel, h in sorted(listed.items()) if _sha256(d / rel) != h]
    assert not bad, bad[:10]


def test_report_finds_every_result_it_reads(tmp_path):
    """The report from the archive reads no missing result, prints no \\tbd and writes every
    output of OUTPUTS.txt; every run of experiments/runs.txt has a manifest (cost table)."""
    res = _release_dir() / "paper" / "results"
    r = subprocess.run([sys.executable, str(REPORT / "report.py"), "--results", str(res),
                        "--out", str(tmp_path)], cwd=ROOT, env=env(), capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "missing results" not in r.stdout, r.stdout[-3000:]
    for p in read_outputs_txt():
        assert (tmp_path / p).is_file(), p
    tbd = [p.name for p in tmp_path.rglob("*.tex") if b"\\tbd" in p.read_bytes()]
    assert not tbd, tbd
    absent = [n for n in REL.run_names() if not (res / f"{n}.manifest.json").is_file()]
    assert not absent, absent


def test_archive_holds_arrays_and_sanitized_manifests_and_logs():
    res = _release_dir() / "paper" / "results"
    names = sorted(p.name for p in res.iterdir())
    assert all(n.endswith((".npz", ".manifest.json", ".log")) for n in names), names
    abs_path = re.compile(r"(?<![\w.<>~])/[^\s'\"]+")

    def strings(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)
        elif isinstance(x, str):
            yield x

    for n in names:
        if n.endswith(".manifest.json"):
            m = json.loads((res / n).read_text())
            assert "host" not in m, n
            assert m.get("clean") is True, n
            leaks = [s for s in strings(m) if abs_path.search(s)]
            assert not leaks, (n, leaks[:3])
        elif n.endswith(".log"):
            txt = (res / n).read_text(encoding="utf-8")
            assert re.search(r"^# python .*\bhost unrecorded \(", txt, re.M), n
            assert not re.search(r"(?<![^\s'\"=(\[{,:])/[A-Za-z0-9_.~-]+/", txt), n


# ---------------------------------------------------------------- upload_results.py

def _upload_module():
    spec = importlib.util.spec_from_file_location("upload_results", REPORT / "upload_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_archive(tmp_path) -> pathlib.Path:
    """A two-file release archive with SHA256SUMS and .sha256, packed like the real one."""
    sys.path.insert(0, str(REPORT))
    try:
        import compact_archive
    finally:
        sys.path.remove(str(REPORT))
    top = tmp_path / "build" / "tiny-results"
    (top / "paper" / "results").mkdir(parents=True)
    (top / "README.md").write_text("tiny\n")
    (top / "paper" / "results" / "a.manifest.json").write_text("{}\n")
    rels = ["README.md", "paper/results/a.manifest.json"]
    (top / "SHA256SUMS").write_text("".join(f"{_sha256(top / r)}  {r}\n" for r in rels))
    arc = tmp_path / "build" / "tiny-results.tar.gz"
    digest = compact_archive.pack(str(top), str(arc), top="tiny-results")
    (tmp_path / "build" / "tiny-results.tar.gz.sha256").write_text(f"{digest}  {arc.name}\n")
    return arc


def test_upload_dry_run_writes_nothing(tmp_path, capsys):
    up, arc = _upload_module(), _tiny_archive(tmp_path)
    conf = tmp_path / "conf.txt"
    conf.write_text("name tiny-results\nurl TODO_RESULTS_URL\nsha256 TODO\n")
    dest = tmp_path / "www"
    dest.mkdir()
    up.main(["--archive", str(arc), "--dest", f"{dest}/", "--url",
             f"file://{dest}/{arc.name}", "--conf", str(conf)])
    assert "dry run" in capsys.readouterr().out
    assert not any(dest.iterdir())
    assert "TODO_RESULTS_URL" in conf.read_text()


def test_upload_verifies_public_copy_before_recording(tmp_path):
    if shutil.which("rsync") is None:
        pytest.skip("rsync not installed")
    up, arc = _upload_module(), _tiny_archive(tmp_path)
    conf = tmp_path / "conf.txt"
    conf.write_text("name tiny-results\nurl TODO_RESULTS_URL\nsha256 TODO\n")
    # a wrong public copy: verification fails, the config stays unchanged
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / arc.name).write_bytes(b"not the archive")
    with pytest.raises(SystemExit):
        up.main(["--archive", str(arc), "--url", f"file://{bad}/{arc.name}", "--verify-only",
                 "--conf", str(conf)])
    assert "TODO_RESULTS_URL" in conf.read_text()
    # a local directory as the web server: upload, verify, record
    www = tmp_path / "www"
    www.mkdir()
    url = f"file://{www}/{arc.name}"
    up.main(["--archive", str(arc), "--dest", f"{www}/", "--url", url, "--yes",
             "--conf", str(conf)])
    assert (www / arc.name).read_bytes() == arc.read_bytes()
    assert (www / (arc.name + ".sha256")).is_file()
    sha = _sha256(arc)
    assert conf.read_text().splitlines() == ["name tiny-results", f"url {url}", f"sha256 {sha}"]


def test_upload_refuses_archive_with_wrong_sha256_file(tmp_path):
    up, arc = _upload_module(), _tiny_archive(tmp_path)
    pathlib.Path(str(arc) + ".sha256").write_text("0" * 64 + f"  {arc.name}\n")
    with pytest.raises(SystemExit):
        up.verify_local(str(arc))
