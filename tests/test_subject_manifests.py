"""Selected producer/consumer contracts without reconstruction or containers."""
import json
import shutil
import warnings

import nibabel as nib
import numpy as np
import pytest

from parrot_neuro import Subject


def save(root, relative, value):
    path = root / "derivatives" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, value)
    elif path.suffix == ".npz":
        np.savez(path, **value)
    else:
        path.write_text(value)
    return path


def test_inventory_requires_registration_and_relocates(tmp_path, register):
    path = save(tmp_path, "atlas/sub-010002/atlas100_labels.txt", "0 Unknown\n1 Cortex")
    s = Subject(tmp_path, "010002")
    with pytest.raises(FileNotFoundError, match="manifest population"):
        s.path.atlas_labels(100)
    manifest = register(tmp_path, "atlas")
    assert json.loads(manifest.read_text())["producer_tag"] == "test/producer:1"
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.move(tmp_path / "derivatives", moved / "derivatives")
    s = Subject(moved, "010002")
    assert s.path.atlas_labels(100).read_text() == "0 Unknown\n1 Cortex"
    assert len(s.outputs.find("atlas")) == 1
    assert s.surface_backend is None and s.qc is None
    with pytest.raises(FileNotFoundError):
        s.path.cortex("lh")
    s.path.atlas_labels(100).unlink()
    with pytest.raises(FileNotFoundError, match="Registered output is missing"):
        s.path.atlas_labels(100)


def test_model_selection_cache_and_recording_independence(tmp_path, register):
    save(tmp_path, "electrodes/sub-010002/landmarks_10-5-full.csv",
         "A,0,0,0\nB,1,0,0\nC,2,0,0\nD,3,0,0\n")
    save(tmp_path, "electrodes/sub-010002/selected_landmarks_10-5-full.json", '["A", "C", "D"]')
    original = np.arange(24, dtype=float).reshape(4, 6)
    path = save(tmp_path, "leadfields/sub-010002/processed_duneuroCGAL-2.0mm-leadfield.npy", original)
    save(tmp_path, "dipoles/sub-010002/spacing2.0mm/100Parcels_dipole_labels.npy", np.array([1, 2]))
    save(tmp_path, "EEG/sub-010002/sub-010002_task-eyesclosed_eeg.npz", {"seg_000": original})
    save(tmp_path, "EEG/sub-010002/sub-010002_task-eyesclosed_eeg.json", '{"channel_names": ["B", "A"]}')
    for step, kwargs in [("electrodes", {}), ("dipoles", {"spacing": 2}),
                         ("forwardsolvers-cgal", {"spacing": 2}), ("eeg", {"task": "eyesclosed"})]:
        register(tmp_path, step, **kwargs)
    s = Subject(tmp_path, "010002")
    key = "duneuroCGAL-2.0mm"
    labels = s.load.dipole_labels(100, 2)
    full = s.load.leadfield(key)
    np.testing.assert_array_equal(full, original[[0, 2, 3]] - original[[0, 2, 3]].mean(0))
    assert s.load.leadfield(key) is full
    s.set_electrodes(["D", "A"])
    subset = s.load.leadfield(key)
    np.testing.assert_array_equal(subset, original[[3, 0]] - original[[3, 0]].mean(0))
    np.testing.assert_allclose(subset.mean(0), 0)
    assert list(s.load.electrodes()) == ["D", "A"]
    assert s.load.electrodes()["D"][0] == 3
    assert s.load.dipole_labels(100, 2) is labels
    for invalid in [[], ["A", "A"], ["B"], ["unknown"], "A"]:
        with pytest.raises(ValueError):
            s.set_electrodes(invalid)
        assert s.electrode_names == ["D", "A"]
        assert s.load.leadfield(key) is subset
    np.testing.assert_array_equal(s.load.eeg()["seg_000"], original)
    assert s.load.eeg_metadata()["channel_names"] == ["B", "A"]
    assert s.load.eeg() is not s.load.eeg()
    s.set_electrodes(None)
    np.testing.assert_array_equal(s.load.leadfield(key), full)
    np.testing.assert_array_equal(np.load(path), original)
    assert s.path.leadfield(key) == path


def test_volume_cache_bypass_and_clear(tmp_path, register):
    path = tmp_path / "derivatives/raw/sub-010002/T1.nii.gz"
    path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), path)
    register(tmp_path, "ingest")
    s = Subject(tmp_path, "010002")
    cached = s.load.t1()
    assert s.load.t1() is cached
    assert s.load.t1(use_cache=False) is not cached
    assert s.load.t1() is cached
    s.clear_cache()
    bypass = s.load.t1(use_cache=False)
    assert s.load.t1() is not bypass


def test_two_tasks_preserve_node_identity(tmp_path, register, monkeypatch):
    from parrot_neuro.optimization import connectivity
    monkeypatch.setattr(connectivity, "jnp", np)
    prefix = "connectivity/sub-010002/"
    W = np.arange(9).reshape(3, 3)
    for name in ["weights_100", "weights_invnodevol_100", "distances_100"]:
        save(tmp_path, prefix + name + ".txt", "\n".join(" ".join(map(str, row)) for row in W))
    save(tmp_path, prefix + "labels_100.txt", "Unknown\nA\nB\nC\n")
    save(tmp_path, prefix + "full_to_reduced_100.npy", np.arange(4))
    save(tmp_path, "dipoles/sub-010002/spacing2.0mm/100Parcels_dipole_labels.npy", np.arange(4))
    for step in ["connectivity", "connectivity-atlas"]:
        register(tmp_path, step)
    register(tmp_path, "dipoles", spacing=2)
    for task, keep in [("rest", [True, False, True]), ("active", [False, True, True])]:
        keep = np.array(keep)
        inverse = np.full(3, -1)
        inverse[keep] = np.arange(2)
        base = f"fMRI/sub-010002/sub-010002_task-{task}_atlas-schaefer_desc-"
        save(tmp_path, base + "optim_nodes.npz", {
            "keep_100": keep, "optim_to_conn_100": np.flatnonzero(keep), "conn_to_optim_100": inverse})
        save(tmp_path, base + "conn_timeseries.npz", {
            "ts_100": W, "labels_100": np.array(["A", "B", "C"]), "ids_100": np.arange(1, 4)})
        register(tmp_path, "fmri", task=task)
    s = Subject(tmp_path, "010002")
    for task in ["rest", "active"]:
        keep = s.load.fmri_keep(100, task)
        np.testing.assert_array_equal(s.load.weights(100, fmri_aligned=True, task=task), W[np.ix_(keep, keep)])
        np.testing.assert_array_equal(s.load.distances(100, fmri_aligned=True, task=task), W[np.ix_(keep, keep)])
        assert s.load.connectivity_labels(100, True, task) == list(np.array(["A", "B", "C"])[keep])
        np.testing.assert_array_equal(s.load.dipole_node_labels(100, 2, task)[1:], s.load.fmri_nodes(100, task).from_conn)
        np.testing.assert_array_equal(s.load.fmri_timeseries("conn", task)["ts_100"][keep], W[keep])
        aligned = connectivity.load_structural_connectivity(s, atlas=100, fmri_task=task)
        np.testing.assert_array_equal(aligned.empirical_bold, W[keep].T)
    bad = s.load.fmri_timeseries("conn", "rest")
    bad["labels_100"] = np.array(["B", "A", "C"])
    np.savez(s.path.fmri_timeseries("conn", "rest"), **bad)
    with pytest.raises(ValueError, match="ordering"):
        connectivity.load_structural_connectivity(s, atlas=100, fmri_task="rest")


def test_qc_summary_once_and_missing_qc(tmp_path):
    save(tmp_path, "qc/sub-010002/qc_report.json", json.dumps({"stages": [
        {"name": "atlas", "status": "warn"}, {"name": "leadfields", "status": "fail"}]}))
    with pytest.warns(UserWarning, match="atlas: warn.*leadfields: fail.*index.html") as caught:
        s = Subject(tmp_path, "010002")
        assert s.qc_status("leadfields") == "fail"
    assert len(caught) == 1
    with warnings.catch_warnings(record=True) as caught:
        Subject(tmp_path, "010002", warn_on_qc=False)
        other = Subject(tmp_path, "missing")
    assert not caught and other.qc is None and other.qc_status("atlas") is None
