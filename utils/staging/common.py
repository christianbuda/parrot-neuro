#!/usr/bin/env python3
"""Cohort-agnostic helpers for staging raw datasets into Parrot-ready BIDS.

Shared by the per-cohort staging scripts (``lemon.py``, future ``hcp.py``). These
run INSIDE the ``parrot_mri_reconstruction`` image (the host has no nibabel); launch
them via ``bin/stage.sh``.

What lives here vs. in a cohort script: anything that is the *same* regardless of
source dataset -- NIfTI header hygiene and the dataset_description/participants.tsv
writers. Cohort scripts own the source layout, file map, and override values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib


def clean_voxel_size(src_nii: Path, dst_nii: Path) -> None:
    """Copy a NIfTI, snapping float32 voxel-size noise in the header to clean values.

    Some source headers carry a float32 artifact: voxel size 1.0000009..., not
    exactly 1.0. recon-all's conform tolerates it, but FastSurfer's surf-stage
    conform.py rejects vox_size > 1.0 (argparse requires it in (0, 1]) and silently
    dies. Fix it at the source so every downstream tool gets clean geometry: rescale
    each spatial affine column to its voxel size rounded to 4 decimals, preserving the
    direction cosines exactly (sub-micron change, no resampling of the data).
    """
    img = nib.load(src_nii)
    aff = img.affine.copy()
    zooms = list(img.header.get_zooms())
    for i in range(3):
        col = aff[:3, i]
        norm = float(np.linalg.norm(col))
        if norm > 0:
            clean = round(norm, 4)
            aff[:3, i] = col * (clean / norm)  # exact unit scale, e.g. -1.0000009 -> -1.0
            zooms[i] = clean
    out = nib.Nifti1Image(img.dataobj, aff, img.header)  # dataobj proxy: dtype/scaling preserved
    # Passing the original header leaves its stale srow/quaternion in the written file
    # (the affine arg updates .affine but not the saved sform/qform). FastSurfer reads
    # voxel size from the sform column norms, so force the cleaned affine into BOTH.
    out.set_sform(aff, code=int(img.header["sform_code"]))
    out.set_qform(aff, code=int(img.header["qform_code"]))
    out.header.set_zooms(tuple(zooms))
    nib.save(out, dst_nii)


def copy_with_json(src_nii: Path, dst_nii: Path, *, json_edit=None) -> None:
    """Copy a .nii.gz (with header voxel-size cleanup) and its sidecar .json.

    ``json_edit`` is an optional ``dict -> dict`` callback to rewrite the sidecar
    metadata (e.g. fixing a malformed ``IntendedFor``).
    """
    dst_nii.parent.mkdir(parents=True, exist_ok=True)
    clean_voxel_size(src_nii, dst_nii)  # not a byte copy: rewrites the header (see above)
    src_json = src_nii.with_name(src_nii.name.replace(".nii.gz", ".json"))
    if src_json.exists():
        meta = json.loads(src_json.read_text())
        if json_edit is not None:
            meta = json_edit(meta)
        dst_json = dst_nii.with_name(dst_nii.name.replace(".nii.gz", ".json"))
        dst_json.write_text(json.dumps(meta, indent=2))


def write_dataset_description(
    dst_root: Path, name: str, *, source_url: str | None = None, bids_version: str = "1.8.0"
) -> None:
    """Write a minimal BIDS dataset_description.json for the staged dataset."""
    desc: dict = {"Name": name, "BIDSVersion": bids_version, "DatasetType": "raw"}
    if source_url is not None:
        desc["SourceDatasets"] = [{"URL": source_url}]
    (Path(dst_root) / "dataset_description.json").write_text(json.dumps(desc, indent=2))


def write_participants_tsv(
    dst_root: Path,
    subjects: list[str],
    *,
    override_cols: list[str],
    subject_overrides: dict[str, dict[str, bool]],
    default_override: dict[str, bool],
) -> None:
    """Write participants.tsv carrying the orchestrator's per-subject override columns.

    The Parrot orchestrator parses override columns POSITIONALLY, so column order is
    significant -- ``override_cols`` defines that order. The leading
    participant_id/age/sex columns are BIDS padding the orchestrator ignores.
    """
    header = "participant_id\tage\tsex\t" + "\t".join(override_cols) + "\n"
    rows = []
    for sub in subjects:
        ov = subject_overrides.get(sub, default_override)
        vals = "\t".join(str(ov.get(c, default_override[c])).lower() for c in override_cols)
        rows.append(f"{sub}\tn/a\tn/a\t{vals}\n")
    (Path(dst_root) / "participants.tsv").write_text(header + "".join(rows))
    print("\nWrote dataset_description.json and participants.tsv")
