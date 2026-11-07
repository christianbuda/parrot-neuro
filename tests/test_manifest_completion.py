"""Completion hooks publish only outputs produced by a successful invocation."""
import importlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from parrot_neuro import Subject

ROOT = Path(__file__).resolve().parents[1]


def test_standalone_eeg_completion_and_skip(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "utils/staging"))
    module = importlib.import_module("lemon_eeg")
    monkeypatch.setattr(module, "SRC_ROOT", tmp_path)
    monkeypatch.setattr(module, "EEG_DERIV", tmp_path / "derivatives/EEG")
    monkeypatch.setenv("PARROT_PRODUCER_TAG", "producer/eeg:7")
    (tmp_path / "sub-010002_EC.set").touch()
    monkeypatch.setattr(module, "read_splicefree_segments",
                        lambda *_: (250, ["Cz"], [np.ones((1, 250))]))
    module.stage_recording("sub-010002", "eyesclosed", ("EC",), 0.5, False)
    s = Subject(tmp_path, "010002")
    assert s.load.eeg_metadata()["channel_names"] == ["Cz"]
    assert s.load.eeg()["seg_000"].shape == (1, 250)
    assert {e["producer_tag"] for e in s.outputs} == {"producer/eeg:7"}
    manifest = tmp_path / "derivatives/manifests/sub-010002/eeg-eyesclosed.json"
    before = manifest.read_bytes()
    monkeypatch.setenv("PARROT_PRODUCER_TAG", "producer/eeg:8")
    module.stage_recording("sub-010002", "eyesclosed", ("EC",), 0.5, False)
    assert manifest.read_bytes() == before


def test_standalone_staging_requires_actual_producer_before_writing(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "utils/staging"))
    module = importlib.import_module("lemon_eeg")
    monkeypatch.delenv("PARROT_PRODUCER_TAG", raising=False)
    monkeypatch.setattr(module, "EEG_DERIV", tmp_path / "derivatives/EEG")
    with pytest.raises(ValueError, match="PARROT_PRODUCER_TAG"):
        module.stage_recording("sub-010002", "eyesclosed", ("EC",), 0.5, False)
    assert not (tmp_path / "derivatives").exists()


def test_staging_independent_variants_do_not_claim_old_files(tmp_path, register):
    directory = tmp_path / "derivatives/fMRI/sub-010002"
    directory.mkdir(parents=True)
    full = directory / "sub-010002_task-rest_atlas-schaefer_desc-full_timeseries.npz"
    conn = directory / "sub-010002_task-rest_atlas-schaefer_desc-conn_timeseries.npz"
    np.savez(full, ts_100=np.ones((2, 3)))
    np.savez(conn, ts_100=np.ones((2, 3)))
    manifest = register(tmp_path, "fmri", task="rest", paths=[full], variant="full")
    assert [e["path"] for e in json.loads(manifest.read_text())["outputs"]] == [
        full.relative_to(tmp_path / "derivatives").as_posix()]
    register(tmp_path, "fmri", task="rest", paths=[conn], variant="conn")
    s = Subject(tmp_path, "010002")
    assert s.path.fmri_timeseries("full") == full
    assert s.path.fmri_timeseries("conn") == conn
    # BOLD remains independently readable when structural manifests are absent.
    assert s.load.fmri_timeseries("conn")["ts_100"].shape == (2, 3)


@pytest.mark.parametrize("rc", [0, 1])
def test_solver_completion_uses_producer_tag_only_on_success(tmp_path, rc):
    script = (ROOT / "bin/run_reconstruction.sh").read_text()
    function = re.search(r"^run_in_docker_SOLVER\(\) \{.*?^\}", script, re.M | re.S).group()
    harness = f"""
{function}
container_exec() {{ return {rc}; }}
register_outputs() {{ printf '%s\\n' "$@" > "$OUTPUT_DIR/published"; }}
OUTPUT_DIR="$1"
WORK_DIR_DOCKER=/work
LOG_FILE="$1/summary"
SUBJECT=010002
spacing=2.0
run_in_docker_SOLVER forwardsolvers-cgal "$1/solver_log.txt" producer/solver:7 ignored collect
"""
    result = subprocess.run(["bash", "-c", harness, "test", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == rc
    published = tmp_path / "published"
    assert published.exists() == (rc == 0)
    if rc == 0:
        assert published.read_text().splitlines() == ["forwardsolvers-cgal", "producer/solver:7", "--spacing", "2.0"]


def test_block_distance_bypass_and_cached_bundle(tmp_path, register):
    directory = tmp_path / "derivatives/dipoles/sub-010002/spacing2.0mm"
    block = directory / "volumetric"
    block.mkdir(parents=True)
    for name in ["dipole_positions", "dipole_directions", "dipole_volume", "dipole_neural_density", "orient_type"]:
        np.save(directory / f"{name}.npy", np.ones((2, 3)))
    np.save(block / "distance_matrix.npy", np.eye(2))
    np.save(block / "dipole_traceback.npy", np.array([True, True]))
    register(tmp_path, "dipoles", spacing=2)
    s = Subject(tmp_path, "010002")
    assert s.dipole_blocks(2) == ["volumetric"]
    assert s.load.dipoles(2) is s.load.dipoles(2.0)
    first = s.load.dipole_block(2, "volumetric", "distance_matrix")
    first[:] = 0
    np.testing.assert_array_equal(s.load.dipole_block(2, "volumetric", "distance_matrix"), np.eye(2))


def test_recorded_channel_mapping_rejects_missing_model_names():
    from parrot_neuro.optimization.data import find_indices
    np.testing.assert_array_equal(find_indices(["C", "A"], ["A", "B", "C"]), [2, 0])
    with pytest.raises(ValueError, match="absent from the model montage"):
        find_indices(["B"], ["A", "C"])


def test_artifact_completion_distinguishes_solve_fallback_and_positions(tmp_path, register):
    directory = tmp_path / "derivatives/leadfields/sub-010002"
    directory.mkdir(parents=True)
    for key in ["duneuro_artifact-eyes-CGAL", "duneuro_artifact-muscle-CGAL", "hartmut_muscle"]:
        np.save(directory / f"processed_{key}-leadfield.npy", np.ones((3, 6)))
    register(tmp_path, "artifacts-leadfields", variant="eyes")
    muscle_manifest = register(tmp_path, "artifacts-leadfields", variant="muscle")
    assert Subject(tmp_path, "010002").available_leadfields() == [
        "duneuro_artifact-eyes-CGAL", "duneuro_artifact-muscle-CGAL"]
    # A fallback completion replaces the muscle registration without claiming
    # the stale solved file (or rewriting the eyes' producer record).
    fallback_manifest = register(tmp_path, "artifacts-leadfields", variant="muscle-fallback")
    assert fallback_manifest == muscle_manifest
    assert Subject(tmp_path, "010002").available_leadfields() == [
        "duneuro_artifact-eyes-CGAL", "hartmut_muscle"]
    assert not Subject(tmp_path, "010002").outputs.find("artifacts")


def test_surface_loading_is_lazy_and_cached(tmp_path, register, monkeypatch):
    import trimesh
    from parrot_neuro import _loaders

    path = tmp_path / "derivatives/surfaces/sub-010002/charm_scalp.ply"
    path.parent.mkdir(parents=True)
    trimesh.creation.icosphere(subdivisions=0).export(path)
    register(tmp_path, "surfaces")
    original = _loaders._read_mesh
    calls = []

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(_loaders, "_read_mesh", read)
    s = Subject(tmp_path, "010002")
    assert not calls
    assert s.load.scalp() is s.load.surface("charm_scalp")
    assert calls == [path]
