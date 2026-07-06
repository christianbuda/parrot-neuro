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
# LEMON is a *uniformly* MP2RAGE cohort, so mp2rage defaults to True here: the raw
# UNI ships with a high-intensity background that breaks FastSurfer's conform unless
# the pipeline's mp2rage_prep MPRAGEises it first. Override per-subject below only for
# genuine exceptions.
DEFAULT_OVERRIDE = {"skip_t2_registration": False, "no_neck": False, "mp2rage": True}

# Per-subject exceptions to DEFAULT_OVERRIDE (key by sub-<ID>); empty = all subjects
# take the defaults. Add an entry e.g. to assert no_neck=True up front if charm's neck
# fit fails for a subject, or skip_t2_registration if the T2 is a different session.
SUBJECT_OVERRIDES: dict[str, dict[str, bool]] = {}


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

    # --- T2w (OPTIONAL) --------------------------------------------------------
    # A few LEMON subjects have no T2 (e.g. sub-010012, sub-010052). T2 is optional
    # for Parrot (auto-discovered; used for pial refinement / charm), so skip it when
    # absent -- charm/pial then run T1-only.
    t2_src = src / "anat" / f"{prefix}_T2w.nii.gz"
    if t2_src.exists():
        copy_with_json(t2_src, out / "anat" / f"{sub}_T2w.nii.gz")
        print("  T2w copied")
    else:
        print("  T2w ABSENT -> skipped (charm/pial run T1-only)")

    # --- DWI + fieldmaps (OPTIONAL) --------------------------------------------
    # DWI drives the connectivity/anisotropy branch; when absent those stages fall
    # back to the group template (QSIPrep/QSIRecon just skip). The SEfmapDWI AP/PA
    # pair is optional even when DWI is present (e.g. sub-010083 has DWI but no
    # fieldmap) -- without it QSIPrep runs without PEPOLAR SDC.
    dwi_src = src / "dwi" / f"{prefix}_dwi.nii.gz"
    if dwi_src.exists():
        (out / "dwi").mkdir(parents=True, exist_ok=True)
        copy_with_json(dwi_src, out / "dwi" / f"{sub}_dwi.nii.gz")
        for ext in (".bval", ".bvec"):
            shutil.copyfile(src / "dwi" / f"{prefix}_dwi{ext}", out / "dwi" / f"{sub}_dwi{ext}")
        print("  DWI + bval/bvec copied")

        intended = f"dwi/{sub}_dwi.nii.gz"  # BIDS: path relative to the subject folder

        def fix_intended(meta: dict) -> dict:
            meta["IntendedFor"] = intended
            return meta

        ap = src / "fmap" / f"{prefix}_acq-SEfmapDWI_dir-AP_epi.nii.gz"
        pa = src / "fmap" / f"{prefix}_acq-SEfmapDWI_dir-PA_epi.nii.gz"
        if ap.exists() and pa.exists():
            for pe in ("AP", "PA"):
                copy_with_json(
                    src / "fmap" / f"{prefix}_acq-SEfmapDWI_dir-{pe}_epi.nii.gz",
                    out / "fmap" / f"{sub}_acq-SEfmapDWI_dir-{pe}_epi.nii.gz",
                    json_edit=fix_intended,
                )
            print(f"  fmap AP/PA copied (IntendedFor -> {intended})")
        else:
            print("  DWI fieldmaps ABSENT -> QSIPrep runs without PEPOLAR SDC")
    else:
        print("  DWI ABSENT -> skipped (template connectivity fallback)")


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
