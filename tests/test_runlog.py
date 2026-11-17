"""runlog.py: header, worked example, histories, manifest with sha256, cost record, lam / lam_s
per time and the fraction of samples leaving Omega.

Interface the tests assume:
- ``RunLog(name, args, setup, mode)`` accepts ``results_dir=`` for the root that holds
  ``<name>.log``, ``<name>.manifest.json`` and the ``smoke/`` and ``example/``
  subdirectories; without that keyword the tests run in a temporary working directory
  and expect ``results/`` relative to it (adapter ``open_run``);
- ``args`` is an argparse Namespace (workers, overwrite, smoke, example); ``setup`` a dict;
- ``run.save(path, **arrays)`` takes a file name that the run places in its directory
  (results/, results/smoke/); in example mode it writes no npz (no-op or error);
- ``Selection(lam, score, table, at_limit, nsolves)`` constructs a selection record;
- the manifest lists every written file except itself; the listed sha256 and size must
  match the file on disk after ``close()`` (the log, if listed, is hashed after its last
  write).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np
import pytest

from conftest import latent

runlog = pytest.importorskip("smoothfm.runlog")
selection = pytest.importorskip("smoothfm.selection")
sampler = pytest.importorskip("smoothfm.sampler")

ROOT = pathlib.Path(__file__).resolve().parents[1]

MANIFEST_KEYS = ["schema", "name", "command", "argv", "commit", "clean", "source_sha256",
                 "python", "numpy", "scipy", "cvxopt", "platform", "host", "workers",
                 "start", "end", "wall_s", "config", "example", "jobs", "outputs"]


def args(**kw):
    base = dict(workers=1, overwrite=True, smoke=False, example=False)
    base.update(kw)
    return argparse.Namespace(**base)


def open_run(tmp_path, monkeypatch, name, mode, **kw):
    a = args(smoke=mode == "smoke", example=mode == "example", **kw)
    setup = {"target": "TrimodalBeta", "N": 100, "set": 0}
    try:
        return runlog.RunLog(name, a, setup, mode, results_dir=tmp_path / "results")
    except TypeError:
        monkeypatch.chdir(tmp_path)
        return runlog.RunLog(name, a, setup, mode)


class Lin:
    def __init__(self):
        self.latent = latent(1, 5.0)

    def __call__(self, x, t):
        return 0.1 * np.asarray(x, float).reshape(-1, 1)

    def div(self, x, t):
        return np.full(np.asarray(x).reshape(-1, 1).shape[0], 0.1)


def _find(tmp_path, pattern):
    return sorted(tmp_path.rglob(pattern))


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _drive(run, tmp_path, npz=True):
    run.example({"target": "TrimodalBeta 0.30 B(8,24) + ...", "N": 100, "set": 0, "m": 64})
    tab = np.array([[0.0, 0.12], [1e-3, -0.3], [10 ** -2.5, -0.34], [1e-2, -0.33]])
    fields = dict(lam=10 ** -2.5, score=-0.34, table=tab, at_limit=False, nsolves=4,
                  lam_s=1.0, limit_score=float("nan"))
    names = getattr(selection.Selection, "_fields", None)
    s = (selection.Selection(**{k: fields.get(k) for k in names}) if names
         else selection.Selection(10 ** -2.5, -0.34, tab, False, 4))
    run.selection("trimodal_N100_s0", 0.999, "holdout", s, 2.1e-15)
    z = np.asarray(latent(1, 5.0).quantile_midpoints(64), float).reshape(-1, 1)
    _, _, summ = sampler.sample(Lin(), z, nstep=8)
    run.flow("trimodal_N100_s0", "holdout", summ, 1.25)
    run.metrics("trimodal_N100_s0", {"tv_pos": 0.1352, "tv_pos_se": 0.004})
    if npz:
        run.save("unit_run.npz", tables=tab[None], at_limit=np.array([False]))
    run.close()


def test_smoke_run_writes_log_and_manifest(tmp_path, monkeypatch):
    run = open_run(tmp_path, monkeypatch, "unit_run", "smoke")
    _drive(run, tmp_path)
    logs = _find(tmp_path, "unit_run.log")
    mans = _find(tmp_path, "unit_run.manifest.json")
    assert len(logs) == 1 and len(mans) == 1
    assert "smoke" in logs[0].parts
    man = json.loads(mans[0].read_text())
    assert man["schema"] == 1 and man["name"] == "unit_run"
    for k in MANIFEST_KEYS:
        assert k in man, k
    # A3: host, CPU model and core count are recorded
    flat = json.dumps(man).lower()
    assert "cpu" in flat and "core" in flat
    for k in ("MU", "L", "NSTEP", "T_END", "TIMES"):
        assert k in man["config"], k
    outs = man["outputs"]
    assert any(str(o["path"]).endswith("unit_run.npz") for o in outs)
    for o in outs:
        p = pathlib.Path(o["path"])
        p = p if p.is_absolute() else next(iter(_find(tmp_path, p.name)), p)
        assert p.exists(), o
        assert o["bytes"] == p.stat().st_size, o
        assert o["sha256"] == _sha(p), o
    jobs = man["jobs"]
    job = jobs["trimodal_N100_s0"] if isinstance(jobs, dict) else \
        next(j for j in jobs if "trimodal_N100_s0" in json.dumps(j))
    assert "wall" in json.dumps(job)


def test_log_contents(tmp_path, monkeypatch):
    run = open_run(tmp_path, monkeypatch, "unit_run", "smoke")
    _drive(run, tmp_path)
    txt = _find(tmp_path, "unit_run.log")[0].read_text()
    head = "\n".join(txt.splitlines()[:12])
    for k in ("commit", "source_sha256", "python", "numpy", "scipy", "cvxopt", "host",
              "workers", "command", "MU", "NSTEP", "T_END"):
        assert k in head, k
    assert "worked example" in txt
    assert "lam/lam_s" in txt                 # A7: selected lam / lam_s per time
    assert "0.999" in txt
    assert "outside" in txt.lower()           # A7: fraction of samples leaving Omega
    assert "tv_pos" in txt


def test_example_mode_writes_no_npz(tmp_path, monkeypatch):
    run = open_run(tmp_path, monkeypatch, "unit_run", "example")
    try:
        _drive(run, tmp_path)
    except Exception:
        pass
    assert not _find(tmp_path, "*.npz")
    logs = _find(tmp_path, "unit_run.log")
    assert logs and "example" in logs[0].parts


def _dirty():
    try:
        out = subprocess.run(["git", "status", "--porcelain", "src", "experiments"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.strip())


def test_full_run_refuses_dirty_tree(tmp_path, monkeypatch):
    d = _dirty()
    if not d:
        pytest.skip("tree is clean (or no git); the refusal cannot be exercised")
    with pytest.raises(Exception):
        open_run(tmp_path, monkeypatch, "unit_run_full", "full")


def test_full_run_refuses_existing_output(tmp_path, monkeypatch):
    if _dirty() is not False:
        pytest.skip("needs a clean tree")
    run = open_run(tmp_path, monkeypatch, "unit_full", "full", overwrite=False)
    _drive(run, tmp_path)
    with pytest.raises(Exception):
        open_run(tmp_path, monkeypatch, "unit_full", "full", overwrite=False)
