"""NPZ loading and cache behavior without a reconstructed dataset."""
import numpy as np
import pytest

from parrot_neuro import Subject


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("kind", ["eeg", "fmri_timeseries", "fmri_nodes"])
def test_npz_loads_arrays_and_closes_archive(tmp_path, monkeypatch, cache, kind, register):
    (tmp_path / "derivatives").mkdir()
    if kind == "fmri_nodes":
        path = tmp_path / "derivatives/fMRI/sub-010002/sub-010002_task-rest_atlas-schaefer_desc-optim_nodes.npz"
        arrays = {
            "keep_100": np.array([True, False, True]),
            "optim_to_conn_100": np.array([0, 2]),
            "conn_to_optim_100": np.array([0, -1, 1]),
        }
    else:
        filename = ("EEG/sub-010002/sub-010002_task-eyesclosed_eeg.npz" if kind == "eeg" else
                    "fMRI/sub-010002/sub-010002_task-rest_atlas-schaefer_desc-full_timeseries.npz")
        path = tmp_path / "derivatives" / filename
        arrays = {"seg_000" if kind == "eeg" else "ts_100": np.arange(12).reshape(3, 4)}
    path.parent.mkdir(parents=True)
    np.savez(path, **arrays)
    register(tmp_path, "eeg" if kind == "eeg" else "fmri",
             task="eyesclosed" if kind == "eeg" else "rest")
    subject = Subject(tmp_path, "010002")
    load = (lambda: subject.load.fmri_nodes(100)) if kind == "fmri_nodes" else (
        lambda: getattr(subject.load, kind)(use_cache=cache))

    original_load = np.load
    archives = []

    def track_load(*args, **kwargs):
        archive = original_load(*args, **kwargs)
        archives.append(archive)
        return archive

    monkeypatch.setattr(np, "load", track_load)
    first = load()
    second = load()

    assert len(archives) == (1 if cache or kind == "fmri_nodes" else 2)
    assert all(archive.zip is None for archive in archives)
    if kind == "fmri_nodes":
        np.testing.assert_array_equal(first.keep, arrays["keep_100"])
        np.testing.assert_array_equal(first.to_conn, arrays["optim_to_conn_100"])
        np.testing.assert_array_equal(first.from_conn, arrays["conn_to_optim_100"])
    else:
        assert isinstance(first, dict)
        assert (first is second) == cache
        for key, expected in arrays.items():
            np.testing.assert_array_equal(first[key], expected)
