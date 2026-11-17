"""The stability report reads the fine-grid circle smallness run when it exists."""
import numpy as np

from release_helpers import REPORT, load_isolated

C, R = load_isolated(REPORT, "stability", "rep_stability_src")


def _use(tmp_path, monkeypatch, names):
    for n in names:
        np.savez(tmp_path / f"{n}.npz", x=np.zeros(1))
    monkeypatch.setattr(C, "RES", C.Results(str(tmp_path)))


def test_prefers_fine_grid(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, ["stability_circle_l2", "stability_circle_l2_dx005"])
    assert R.circle_l2_source() == "stability_circle_l2_dx005"


def test_falls_back_to_coarse_grid(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, ["stability_circle_l2"])
    assert R.circle_l2_source() == "stability_circle_l2"


def test_absent_names_coarse(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, [])
    assert R.circle_l2_source() == "stability_circle_l2"
