#!/usr/bin/env python3
"""Stage LEMON subject(s) into a flattened, cleaned BIDS dataset for Parrot.

Parrot's orchestrator is session-blind (it globs sub-<ID>/anat and sub-<ID>/dwi,
with no ses-* level), so LEMON's sub-<ID>/ses-01/... layout must be flattened.
Per subject this script:

  1. Flattens + copies only the files Parrot consumes (T1w/UNI + INV1/INV2, T2w,
     DWI+bval/bvec, DWI fieldmap pair), stripping the `_ses-01` entity from names.
  2. Snaps the float32 voxel-size artifact in the source headers (1.0000009... -> 1.0)
     so FastSurfer's surf-stage conform doesn't reject vox_size > 1.0 (see
     common.clean_voxel_size). Geometry-preserving; no data resampling.
  3. Fixes the fieldmap `IntendedFor` (the LEMON source has a malformed
     `ses-01/dwi/sub-sub-..._dwi.nii.gz`) so QSIPrep performs PEPOLAR SDC.

All MP2RAGE intensity preprocessing (MPRAGEise) lives in the pipeline's `mp2rage_prep`
step, gated by the participants.tsv `mp2rage` column -- so staging copies the raw UNI
and its INV1/INV2 inversions as-is (only the header voxel-size is cleaned, not the
intensities).

Runs INSIDE the parrot_mri_reconstruction image (neuro env), with the LEMON MRI_Raw
root mounted at /src (ro) and the target BIDS dir at /dst. Launch via bin/stage.sh.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from common import (
    copy_with_json,
    write_dataset_description,
    write_participants_tsv,
)

SRC_ROOT = Path("/src")  # LEMON .../MRI_Raw mounted here
DST_ROOT = Path("/dst")  # target BIDS dataset (e.g. BIDS_LEMON)

# Orchestrator override columns, IN POSITIONAL ORDER (the orchestrator parses them
# by position): col4=skip-T2-registration, col5=no-neck, col6=mp2rage.
OVERRIDE_COLS = ["skip_t2_registration", "no_neck", "mp2rage"]
DEFAULT_OVERRIDE = {"skip_t2_registration": False, "no_neck": False, "mp2rage": False}

# Per-subject overrides. sub-010002: T1/T2 same-session (register normally), no-neck
# not asserted up front (flip to True if charm's neck fit fails), mp2rage True (raw
# UNI -> mp2rage_prep MPRAGEises it).
SUBJECT_OVERRIDES: dict[str, dict[str, bool]] = {
    "sub-010002": {"skip_t2_registration": False, "no_neck": False, "mp2rage": True},
}


def find_session_dir(sub_dir: Path) -> Path:
    sessions = sorted(sub_dir.glob("ses-*"))
    if not sessions:
        raise FileNotFoundError(f"No ses-* directory under {sub_dir}")
    return sessions[0]


def stage_subject(sub: str) -> None:
    print(f"\n=== staging {sub} ===")
    ses = find_session_dir(SRC_ROOT / sub).name
    src = SRC_ROOT / sub / ses
    prefix = f"{sub}_{ses}"          # source filename prefix
    out = DST_ROOT / sub             # flattened destination (no ses level)

    # --- T1w (raw MP2RAGE UNI) + INV1/INV2 -------------------------------------
    # Copied verbatim; the pipeline's mp2rage_prep does the MPRAGEise.
    copy_with_json(
        src / "anat" / f"{prefix}_acq-mp2rage_T1w.nii.gz",
        out / "anat" / f"{sub}_acq-mp2rage_T1w.nii.gz",
    )
    copy_with_json(
        src / "anat" / f"{prefix}_inv-1_mp2rage.nii.gz",
        out / "anat" / f"{sub}_inv-1_MP2RAGE.nii.gz",
    )
    copy_with_json(
        src / "anat" / f"{prefix}_inv-2_mp2rage.nii.gz",
        out / "anat" / f"{sub}_inv-2_MP2RAGE.nii.gz",
    )
    print("  T1w (raw UNI) + INV1/INV2 copied")

    # --- T2w -------------------------------------------------------------------
    copy_with_json(
        src / "anat" / f"{prefix}_T2w.nii.gz",
        out / "anat" / f"{sub}_T2w.nii.gz",
    )
    print("  T2w copied")

    # --- DWI (nii + sidecars) --------------------------------------------------
    (out / "dwi").mkdir(parents=True, exist_ok=True)
    copy_with_json(
        src / "dwi" / f"{prefix}_dwi.nii.gz",
        out / "dwi" / f"{sub}_dwi.nii.gz",
    )
    for ext in (".bval", ".bvec"):
        shutil.copyfile(src / "dwi" / f"{prefix}_dwi{ext}", out / "dwi" / f"{sub}_dwi{ext}")
    print("  DWI + bval/bvec copied")

    # --- DWI fieldmaps: fix IntendedFor to the flattened DWI -------------------
    intended = f"dwi/{sub}_dwi.nii.gz"  # BIDS: path relative to the subject folder

    def fix_intended(meta: dict) -> dict:
        meta["IntendedFor"] = intended
        return meta

    for pe in ("AP", "PA"):
        copy_with_json(
            src / "fmap" / f"{prefix}_acq-SEfmapDWI_dir-{pe}_epi.nii.gz",
            out / "fmap" / f"{sub}_acq-SEfmapDWI_dir-{pe}_epi.nii.gz",
            json_edit=fix_intended,
        )
    print(f"  fmap AP/PA copied (IntendedFor -> {intended})")


def main() -> None:
    subjects = sys.argv[1:] or ["sub-010002"]
    for sub in subjects:
        stage_subject(sub)
    write_dataset_description(
        DST_ROOT,
        "LEMON (flattened subset for Parrot)",
        source_url="MPI-Leipzig Mind-Brain-Body LEMON",
    )
    write_participants_tsv(
        DST_ROOT,
        subjects,
        override_cols=OVERRIDE_COLS,
        subject_overrides=SUBJECT_OVERRIDES,
        default_override=DEFAULT_OVERRIDE,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
