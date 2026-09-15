"""Keep bin/legend_of_files.txt honest about the derivatives layout.

The layout is encoded in three places -- the legend (docs), parrot_neuro._layout
(what this package reads) and the orchestrator (what actually creates the dirs).
Only the last two are exercised by code, so a rename that skips the legend breaks
nothing and goes unnoticed. These tests close that gap from both directions:
a stage the pipeline writes but the legend omits, and a folder the legend still
describes after the pipeline stopped writing it.

Purely static -- parses the repo's own files, needs no derivatives tree.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LEGEND = REPO / "bin" / "legend_of_files.txt"
ORCHESTRATOR = REPO / "bin" / "run_reconstruction.sh"
LAYOUT_PY = REPO / "src" / "parrot_neuro" / "_layout.py"

# Orchestrator steps that do NOT own a folder of their own: they write into
# another stage's directory. Mapped here to the folder the legend documents them
# under; a step missing from this map is expected to own the folder it names.
STEP_FOLDER = {
    "ingest": "raw",
    # These three run FreeSurfer tools against the recon SUBJECTS_DIR, so their
    # outputs land inside the surface backend's folder (fastsurfer/ by default).
    "mne": "fastsurfer",
    "schaefer": "fastsurfer",
    "freesurfersubcortical": "fastsurfer",
    # Carries the DWI products into T1 space; writes next to the tensor fit.
    "dwi2t1": "dwitensor",
}

# Top-level entries that are not pipeline stages and so have no layout constant.
NON_STAGE_FOLDERS = {"logs"}


def _read(path: Path) -> str:
    if not path.is_file():
        pytest.fail(f"missing {path.relative_to(REPO)} -- this test parses it directly")
    return path.read_text()


def stage_sections() -> set[str]:
    """Folders the legend documents as pipeline stages.

    These are its section headers: a tree connector at column 0, e.g.
    "├── tissuelabels/: ...".
    """
    return set(re.findall(r"(?m)^[├└]── ([A-Za-z0-9_]+)/", _read(LEGEND)))


def aside_entries() -> set[str]:
    """Folders in the closing "not produced by this pipeline" block.

    Indented and without a tree connector, e.g. "    EEG/sub-<ID>/". Documented
    because they show up in real derivatives trees, but staged separately -- so
    they are expected to have no layout constant and no orchestrator step.
    """
    return set(re.findall(r"(?m)^\s+([A-Za-z0-9_]+)/sub-<ID>/", _read(LEGEND)))


def documented_folders() -> set[str]:
    """Every folder the legend describes, in either shape."""
    return stage_sections() | aside_entries()


def layout_stages() -> set[str]:
    """Stage directory names declared by parrot_neuro._layout.

    Loaded straight from the file rather than imported as ``parrot_neuro._layout``:
    the package __init__ pulls in numpy/nibabel/trimesh, and this test checks text
    against text -- it should run on a bare checkout with nothing but pytest.
    _layout.py is dependency-free constants, so importing it in isolation is safe.
    """
    spec = importlib.util.spec_from_file_location("_parrot_layout", LAYOUT_PY)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable in-repo
        pytest.fail(f"could not load {LAYOUT_PY.relative_to(REPO)}")
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    return {
        value
        for name, value in vars(layout).items()
        if name.isupper() and isinstance(value, str)
    }


def orchestrator_folders() -> set[str]:
    """Folders bin/run_reconstruction.sh writes, derived from its NAME= steps."""
    steps = re.findall(r'(?m)^\s*NAME="([a-z0-9_-]+)"', _read(ORCHESTRATOR))
    return {STEP_FOLDER.get(step, step) for step in steps}


def test_layout_constants_are_documented():
    missing = layout_stages() - documented_folders()
    assert not missing, (
        f"parrot_neuro._layout declares {sorted(missing)}, which bin/legend_of_files.txt "
        "does not describe. Add a section for each (or drop the constant if the stage is gone)."
    )


def test_orchestrator_stages_are_documented():
    missing = orchestrator_folders() - documented_folders()
    assert not missing, (
        f"run_reconstruction.sh writes {sorted(missing)}, which bin/legend_of_files.txt "
        "does not describe. Document the new stage, or -- if the step writes into another "
        "stage's folder -- map it in STEP_FOLDER above."
    )


def test_legend_describes_no_stale_folders():
    known = layout_stages() | orchestrator_folders() | NON_STAGE_FOLDERS
    # Only the stage sections make a claim about what the pipeline writes; the
    # closing aside documents folders it explicitly does NOT produce.
    stale = stage_sections() - known
    assert not stale, (
        f"bin/legend_of_files.txt still describes {sorted(stale)}, which neither "
        "parrot_neuro._layout nor run_reconstruction.sh produces. Remove the section, or "
        "move it to the closing 'not produced by this pipeline' block if it is staged "
        "separately (utils/staging/)."
    )


def test_legend_states_the_stage_major_rule():
    # The layout's one invariant. It was the single biggest error in the pre-BIDS
    # version of this file (stages were documented without their sub-<ID>/ level),
    # so pin it: a rewrite that drops it is a regression.
    assert "<output_dir>/<stage>/sub-<ID>/" in _read(LEGEND)


def test_orchestrator_ships_the_legend():
    # The legend is copied into every output dir so a derivatives tree stays
    # self-documenting once detached from this checkout. Deleting that copy is
    # easy to do by accident while editing the startup block.
    text = _read(ORCHESTRATOR)
    assert re.search(r"cp\b.*legend_of_files\.txt.*\$OUTPUT_DIR", text), (
        "run_reconstruction.sh no longer copies bin/legend_of_files.txt into $OUTPUT_DIR; "
        "outputs would ship without their layout documentation."
    )
