"""Path regressions that do not require reconstructed data."""
import numpy as np
import pytest

from parrot_neuro import Subject


def test_root_from_environment_is_read_at_construction(tmp_path, monkeypatch):
    for name in ("first", "second"):
        root = tmp_path / name
        (root / "derivatives").mkdir(parents=True)
        monkeypatch.setenv("PARROT_BIDS_ROOT", str(root))
        assert Subject(subject_id="010002").bids_root == root


def test_explicit_root_overrides_environment(tmp_path, monkeypatch):
    (tmp_path / "derivatives").mkdir()
    monkeypatch.setenv("PARROT_BIDS_ROOT", str(tmp_path / "missing"))
    assert Subject(tmp_path, "010002").bids_root == tmp_path


@pytest.mark.parametrize("value", [None, ""])
def test_missing_root_has_clear_error(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PARROT_BIDS_ROOT", raising=False)
    else:
        monkeypatch.setenv("PARROT_BIDS_ROOT", value)
    with pytest.raises(ValueError, match="PARROT_BIDS_ROOT"):
        Subject(subject_id="010002")


def test_subject_id_is_required(tmp_path):
    with pytest.raises(TypeError, match="subject_id is required"):
        Subject(tmp_path)


@pytest.mark.parametrize("subject_id", ["010002", "sub-010002"])
def test_subject_id_normalization(tmp_path, subject_id, register):
    (tmp_path / "derivatives").mkdir()
    path = tmp_path / "derivatives/raw/sub-010002/T1.nii.gz"
    path.parent.mkdir(parents=True)
    path.touch()
    register(tmp_path, "ingest")
    subject = Subject(tmp_path, subject_id)

    assert subject.subject == "010002"
    assert subject.path.t1() == tmp_path / "derivatives/raw/sub-010002/T1.nii.gz"


@pytest.mark.parametrize("spacing", [2, 2.0, 2.5])
def test_dipole_labels_use_reconstruction_spacing_format(tmp_path, spacing, register):
    folder = tmp_path / "derivatives/dipoles/sub-010002" / f"spacing{float(spacing)}mm"
    folder.mkdir(parents=True)
    labels = np.array([0, 1, 2])
    np.save(folder / "100Parcels_dipole_labels.npy", labels)
    register(tmp_path, "dipoles", spacing=spacing)

    subject = Subject(tmp_path, "010002")

    assert subject.path.dipole_dir(spacing) == folder
    np.testing.assert_array_equal(subject.load.dipole_labels(100, spacing), labels)
