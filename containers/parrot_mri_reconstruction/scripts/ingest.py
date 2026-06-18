#!/usr/bin/env python3
"""Ingest (stage 0) for the Parrot pipeline.

Validates each input anatomical volume and writes the standardized working inputs the
rest of the pipeline reads, under ``derivatives/raw/sub-<ID>/``, each with a JSON
provenance sidecar recording its origin and the preprocessing this stage applied.

Per volume:
  1. VALIDATE
       - loadable ``.nii.gz``                              -> hard error if not
       - genuinely 3D: a ``[X, Y, Z, 1]`` singleton is squeezed *only* with
         ``--fix-inputs`` (else flagged + hard error); a true multi-volume 4D is
         always a hard error (a 4D anatomical is wrong)
       - near-integer voxel-size artifact (e.g. ``1.0000009`` instead of ``1.0``,
         which silently breaks FastSurfer's surf-stage conform): flagged, and cleaned
         *only* with ``--fix-inputs``
  2. STANDARDIZE -> ``raw/<name>.nii.gz``
       - T1: MPRAGEised for MP2RAGE (N3-bias-correct INV2, soft-weight the UNI),
         otherwise copied verbatim
       - T2: copied
  3. PROVENANCE -> ``raw/<name>.json`` : ``{Sources, GeneratedBy, Operations}``

Runs under FreeSurfer's fspython: ``mri_nu_correct.mni`` (N3, for MPRAGEise) comes
from the sourced FreeSurfer env and nibabel is available there.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import nibabel as nib

TOOL = {"Name": "parrot ingest", "Description": "Parrot pipeline input ingest (stage 0)"}


class InputError(RuntimeError):
    """A fatal, non-fixable problem with an input volume."""


def _clean_voxel_size(img: nib.Nifti1Image, ops: list[str]) -> nib.Nifti1Image:
    """Snap float32 voxel-size noise: rescale each affine column to its rounded norm.

    Forces the cleaned affine into both sform and qform (the constructor alone leaves
    the header's stale srow on save). FastSurfer reads voxel size from the sform.
    """
    aff = img.affine.copy()
    zooms = list(img.header.get_zooms())
    for i in range(3):
        col = aff[:3, i]
        norm = float(np.linalg.norm(col))
        if norm > 0:
            aff[:3, i] = col * (round(norm, 4) / norm)
            zooms[i] = round(norm, 4)
    out = nib.Nifti1Image(img.dataobj, aff, img.header)
    out.set_sform(aff, code=int(img.header["sform_code"]))
    out.set_qform(aff, code=int(img.header["qform_code"]))
    out.header.set_zooms(tuple(zooms))
    ops.append("voxel-size header snapped to clean values")
    return out


def load_and_validate(path: str, fix: bool, ops: list[str]) -> nib.Nifti1Image:
    """Load a NIfTI and enforce the input contract; returns a (possibly fixed) image."""
    if not (path.endswith(".nii.gz") or path.endswith(".nii")):
        raise InputError(f"{path}: not a NIfTI (.nii/.nii.gz)")
    try:
        img = nib.load(path)
        img.header  # touch header to force a real read/parse
    except Exception as e:  # noqa: BLE001 -- any load failure is fatal here
        raise InputError(f"{path}: not a loadable NIfTI ({e})")

    # --- shape: must be genuinely 3D ----------------------------------------
    shape = img.shape
    if len(shape) == 4 and shape[3] == 1:
        if not fix:
            raise InputError(
                f"{path}: 4D with a singleton volume {shape}. Re-run with --fix-inputs "
                "to squeeze it to 3D."
            )
        img = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine, img.header)
        ops.append(f"squeezed singleton 4th dim {shape} -> {img.shape}")
    elif len(shape) != 3:
        raise InputError(f"{path}: expected a 3D anatomical, got shape {shape}")

    # --- voxel size: flag (and optionally clean) float32 artifacts ----------
    # A "near-integer but not exact" voxel size (e.g. 1.0000009) is a float32 header
    # artifact that breaks FastSurfer's surf conform. Flag only that band: ignore
    # negligible noise (< 1e-7, sub-float32-eps) and legitimate non-integer sizes
    # (e.g. 0.8 mm, which is far from any integer).
    def _is_artifact(n: float) -> bool:
        return 1e-7 < abs(n - round(n)) < 1e-3
    norms = [float(np.linalg.norm(img.affine[:3, i])) for i in range(3)]
    if any(_is_artifact(n) for n in norms):
        if fix:
            img = _clean_voxel_size(img, ops)
        else:
            print(
                f"[ingest] WARNING {os.path.basename(path)}: non-integer voxel size "
                f"{tuple(norms)} (float32 artifact). This can break FastSurfer's surf "
                "stage; re-run with --fix-inputs to snap it.",
                flush=True,
            )
    return img


def mprageise(uni: nib.Nifti1Image, inv2_path: str, ops: list[str], percentile: float = 99.0) -> nib.Nifti1Image:
    """presurfer-style MPRAGEise: N3-bias-correct INV2, scale to [0,1], soft-weight UNI."""
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    try:
        subprocess.run(["mri_nu_correct.mni", "--i", inv2_path, "--o", tmp, "--n", "2"], check=True)
        inv2 = np.asarray(nib.load(tmp).get_fdata(), dtype=np.float32)
        uni_data = np.asarray(uni.get_fdata(), dtype=np.float32)
        scale = float(np.percentile(inv2[inv2 > 0], percentile))
        weight = np.clip(inv2 / scale, 0.0, 1.0)
        t1 = uni_data * weight
        ops.append(f"MPRAGEised (N3-corrected INV2, p{percentile:g} soft weight)")
        return nib.Nifti1Image(t1.astype(uni.get_data_dtype()), uni.affine, uni.header)
    finally:
        os.remove(tmp)


def write_sidecar(out_nii: str, sources: list[str], ops: list[str]) -> None:
    meta = {
        "Sources": [f"bids:{os.path.basename(s)}" for s in sources],
        "GeneratedBy": [TOOL],
        "Operations": ops or ["copied verbatim (no preprocessing)"],
    }
    with open(out_nii.replace(".nii.gz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)


def emit(img: nib.Nifti1Image, out_nii: str, sources: list[str], ops: list[str]) -> None:
    os.makedirs(os.path.dirname(out_nii), exist_ok=True)
    nib.save(img, out_nii)
    write_sidecar(out_nii, sources, ops)
    print(f"[ingest] -> {out_nii}  ({'; '.join(ops) or 'copied'})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Parrot ingest: validate + standardize inputs into raw/.")
    p.add_argument("--out-dir", required=True, help="derivatives/raw/sub-<ID> output dir")
    p.add_argument("--t1", required=True, help="input T1w (MP2RAGE UNI if --mp2rage)")
    p.add_argument("--t2", help="input T2w (optional)")
    p.add_argument("--inv2", help="MP2RAGE INV2 (required with --mp2rage)")
    p.add_argument("--mp2rage", action="store_true", help="T1 is an MP2RAGE UNI -> MPRAGEise it")
    p.add_argument("--fix-inputs", action="store_true",
                   help="auto-fix flagged issues (squeeze singleton 4D, snap voxel size); "
                        "default off = flag only, never mutate")
    args = p.parse_args()

    # --- T1 -----------------------------------------------------------------
    ops: list[str] = []
    t1_img = load_and_validate(args.t1, args.fix_inputs, ops)
    if args.mp2rage:
        if not args.inv2:
            raise InputError("--mp2rage requires --inv2")
        load_and_validate(args.inv2, args.fix_inputs, ops)  # validate INV2 too
        t1_out = mprageise(t1_img, args.inv2, ops)
        sources = [args.t1, args.inv2]
    else:
        t1_out, sources = t1_img, [args.t1]
    emit(t1_out, os.path.join(args.out_dir, "T1.nii.gz"), sources, ops)

    # --- T2 -----------------------------------------------------------------
    if args.t2:
        ops2: list[str] = []
        t2_img = load_and_validate(args.t2, args.fix_inputs, ops2)
        emit(t2_img, os.path.join(args.out_dir, "T2.nii.gz"), [args.t2], ops2)


if __name__ == "__main__":
    main()
