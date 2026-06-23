#!/usr/bin/env python3
"""Stage HCP Young Adult subject(s) into a Parrot-ready BIDS dataset.

HCP-YA ships already-minimally-preprocessed data in its own layout
(``<HCPID>/T1w/...``), with the diffusion co-registered to the structural T1w
(AC-PC) space and a full FreeSurfer recon. Parrot consumes that by:

  1. **Anatomy -> BIDS anat/**: HCP's ``T1w_acpc_dc_restore`` becomes the Parrot
     T1 (this is the space the DWI is already in, so dwi2t1 is identity), and
     ``T2w_acpc_dc_restore`` becomes the T2 (SimNIBS charm / recon-all T2pial).
  2. **Diffusion -> sourcedata/hcp/<HCPID>/** (native HCP layout): the pipeline
     runs QSIRecon with ``--input-type hcpya`` against this tree, and the DTI fit
     reads ``data.nii.gz`` directly (already in T1 space).
  3. **FreeSurfer reuse**: HCP's recon is copied to ``derivatives/freesurfer/
     sub-<HCPID>/`` and a placeholder ``freesurfer_log.txt`` is dropped so the
     pipeline's recon-all stage is skipped (surfaces reused; FastSurfer still runs
     --seg_only for the CerebNet/HypVINN CNN subsegs). Requires running Parrot with
     ``--recon freesurfer`` (or letting the auto-detect pick up the staged recon).

IMPORTANT: because the FreeSurfer recon + placeholder log are written under
``<bids_out>/derivatives/``, the Parrot run must use that as its output_dir:

    ./bin/stage.sh hcp /path/to/HCP_1200 /path/to/BIDS_HCP 100206
    ./bin/run_reconstruction.sh /path/to/BIDS_HCP /path/to/BIDS_HCP/derivatives \
        participant --participant-label 100206 --dwi-preprocessed hcp --recon freesurfer

Runs INSIDE the parrot_mri_reconstruction image (neuro env); launch via
bin/stage.sh, which mounts the HCP root at /src (ro) and the BIDS dir at /dst.
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

SRC_ROOT = Path("/src")  # HCP cohort root (contains <HCPID>/ dirs) mounted here
DST_ROOT = Path("/dst")  # target BIDS dataset

# Orchestrator override columns, IN POSITIONAL ORDER (col4=skip-T2-registration,
# col5=no-neck, col6=mp2rage). HCP T1w is a plain (MPRAGE) restore, not MP2RAGE.
OVERRIDE_COLS = ["skip_t2_registration", "no_neck", "mp2rage"]
DEFAULT_OVERRIDE = {"skip_t2_registration": False, "no_neck": False, "mp2rage": False}

# Minimal HCP diffusion files QSIRecon's hcpya ingress needs (bvals/bvecs/data are
# required; the brain images improve recon; the large MNI warps are not needed since
# we stay in T1 space). Paths are relative to <HCPID>/.
HCP_DWI_FILES = [
    "T1w/Diffusion/data.nii.gz",
    "T1w/Diffusion/bvals",
    "T1w/Diffusion/bvecs",
    "T1w/Diffusion/nodif_brain_mask.nii.gz",
    "T1w/T1w_acpc_dc_restore_brain.nii.gz",
    "T1w/brainmask_fs.nii.gz",
]


def stage_subject(arg: str) -> str:
    """Stage one HCP subject; returns the BIDS subject label (sub-<HCPID>)."""
    hcpid = arg[4:] if arg.startswith("sub-") else arg
    sub = f"sub-{hcpid}"
    print(f"\n=== staging {sub} (HCP {hcpid}) ===")
    src = SRC_ROOT / hcpid
    if not (src / "T1w").is_dir():
        raise FileNotFoundError(f"HCP subject {hcpid} not found under {SRC_ROOT} (no {src}/T1w)")

    # --- 1. anatomy -> BIDS anat/ (header voxel-size cleaned, like other cohorts) --
    copy_with_json(src / "T1w" / "T1w_acpc_dc_restore.nii.gz",
                   DST_ROOT / sub / "anat" / f"{sub}_T1w.nii.gz")
    copy_with_json(src / "T1w" / "T2w_acpc_dc_restore.nii.gz",
                   DST_ROOT / sub / "anat" / f"{sub}_T2w.nii.gz")
    print("  T1w + T2w copied")

    # --- 2. diffusion -> sourcedata/hcp/<HCPID>/ (native layout, for hcpya) --------
    # Copied verbatim (no header rewrite): QSIRecon's ingress expects HCP-native data.
    dwi_dst = DST_ROOT / "sourcedata" / "hcp" / hcpid
    for rel in HCP_DWI_FILES:
        s = src / rel
        if not s.exists():
            raise FileNotFoundError(f"expected HCP file missing: {s}")
        d = dwi_dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
    print(f"  HCP diffusion tree staged -> {dwi_dst}")

    # --- 3. FreeSurfer reuse -> derivatives/freesurfer/ + placeholder skip-log -----
    fs_src = src / "T1w" / hcpid          # HCP's FreeSurfer subject dir
    if not (fs_src / "surf" / "lh.white").is_file():
        raise FileNotFoundError(f"HCP FreeSurfer recon not found at {fs_src} (no surf/lh.white)")
    fs_dst = DST_ROOT / "derivatives" / "freesurfer" / sub
    shutil.copytree(fs_src, fs_dst, dirs_exist_ok=True)
    log_dir = DST_ROOT / "derivatives" / "logs" / sub
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "freesurfer_log.txt").write_text(
        f"Placeholder: FreeSurfer recon reused from HCP for {sub} "
        f"(staged by utils/staging/hcp.py). recon-all skipped.\n"
    )
    print(f"  FreeSurfer recon reused -> {fs_dst} (+ placeholder freesurfer_log.txt)")

    return sub


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: hcp.py <HCPID|sub-HCPID> [more ...]")
    subjects = [stage_subject(a) for a in args]
    write_dataset_description(
        DST_ROOT,
        "HCP-YA (staged subset for Parrot)",
        source_url="Human Connectome Project Young Adult (1200 subjects release)",
    )
    write_participants_tsv(
        DST_ROOT,
        subjects,
        override_cols=OVERRIDE_COLS,
        subject_overrides={},
        default_override=DEFAULT_OVERRIDE,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
