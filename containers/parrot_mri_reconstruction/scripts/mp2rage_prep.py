#!/usr/bin/env python3
"""MP2RAGE preconditioning for the Parrot pipeline.

Turns a raw MP2RAGE UNI (+ INV2) into the two images the pipeline needs:

  --out-t1    : MPRAGEised UNI (presurfer-style). N3-bias-correct INV2, scale it to
                [0, 1], and use it as a SOFT weight on the UNI. Air -> ~0 (INV2 ~ 0
                there) so the high-intensity salt-and-pepper background is suppressed,
                while CSF inside the head is preserved (INV2 has real signal there) --
                unlike a hard INV2 threshold, which also drops dark interior CSF.
                Full head; this becomes the subject's working T1 for every stage
                (FastSurfer, HippUnfold, charm, MNE BEM, ...).
  --out-recon : the MPRAGEised T1 divided by SAMSEG's estimated bias field, guarded
                to identity outside the brain (where the field is 0). The full head is
                kept while the brain is intensity-homogenized. ONLY recon-all consumes
                this: the raw UNI breaks recon-all's mri_normalize ("could not find
                enough control points"), and SAMSEG's own brain-masked output would
                break the MNE BEM watershed (no skull/scalp).

Runs inside the parrot_mri_reconstruction image under FreeSurfer's fspython: both
mri_nu_correct.mni (N3) and run_samseg come from the sourced FreeSurfer env, and
nibabel is available there. (antspyx's N4 lives only in the separate `neuro` env,
whose numpy/OpenBLAS clashes with FreeSurfer's run_samseg -> SIGSEGV. Staying in one
env and using N3 avoids that; N3 is plenty for a soft suppression weight.)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np
import nibabel as nib


def mprageise(uni: str, inv2: str, out_t1: str, percentile: float = 99.0) -> None:
    """MPRAGEise an MP2RAGE UNI (presurfer's method, reimplemented natively).

    Suppresses the UNI's high-intensity background WITHOUT a hard threshold: N3-bias-
    correct INV2, scale it to [0, 1], and use it as a soft weight on UNI. Correcting
    INV2's bias field first stops the weight from dimming superficial cortex near the
    skull (where the receive field falls off).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    try:
        # N3 nonuniformity correction on INV2 via FreeSurfer (keeps us in fspython;
        # antspyx's N4 is only in the `neuro` env). The .nii.gz suffix makes it emit
        # NIfTI on the input voxel grid, so it stays co-registered with the UNI.
        subprocess.run(
            ["mri_nu_correct.mni", "--i", inv2, "--o", tmp, "--n", "2"],
            check=True,
        )
        inv2_corr = np.asarray(nib.load(tmp).get_fdata(), dtype=np.float32)
        uni_img = nib.load(uni)
        uni_data = np.asarray(uni_img.get_fdata(), dtype=np.float32)

        # Robust scale to [0, 1]: divide by a high percentile (not the max, which a
        # single bright voxel would blow out), then clip. This is the soft weight.
        scale = float(np.percentile(inv2_corr[inv2_corr > 0], percentile))
        weight = np.clip(inv2_corr / scale, 0.0, 1.0)

        t1 = uni_data * weight
        nib.save(
            nib.Nifti1Image(t1.astype(uni_img.get_data_dtype()), uni_img.affine, uni_img.header),
            out_t1,
        )
        print(f"[mp2rage_prep] MPRAGEised (N3 INV2, p{percentile:g} scale) -> {out_t1}")
    finally:
        os.remove(tmp)


def samseg_correct(t1: str, samseg_dir: str, out_recon: str, threads: int) -> None:
    """Run SAMSEG, then divide the T1 by its bias field (identity outside the brain)."""
    subprocess.run(
        ["run_samseg", "-i", t1, "-o", samseg_dir, "--threads", str(threads)],
        check=True,
    )
    field = nib.load(os.path.join(samseg_dir, "mode01_bias_field.mgz")).get_fdata()
    t1_img = nib.load(t1)
    # SAMSEG's bias field is 0 outside its brain segmentation; guard to 1.0 there so
    # the skull/scalp/neck pass through unchanged (kept for the MNE BEM watershed).
    corrected = t1_img.get_fdata() / np.where(field > 0, field, 1.0)
    nib.save(
        nib.Nifti1Image(corrected.astype(np.float32), t1_img.affine, t1_img.header),
        out_recon,
    )
    print(f"[mp2rage_prep] SAMSEG bias-field corrected -> {out_recon}")


def main() -> None:
    p = argparse.ArgumentParser(description="MP2RAGE preconditioning (MPRAGEise + SAMSEG for recon-all).")
    p.add_argument("--uni", required=True, help="MP2RAGE UNI image")
    p.add_argument("--inv2", required=True, help="MP2RAGE INV2 image")
    p.add_argument("--out-t1", required=True, help="MPRAGEised full-head T1 (working T1 for all stages)")
    p.add_argument("--out-recon", required=True, help="SAMSEG-corrected full-head T1 (recon-all input)")
    p.add_argument("--samseg-dir", required=True, help="SAMSEG output directory")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    mprageise(args.uni, args.inv2, args.out_t1)
    samseg_correct(args.out_t1, args.samseg_dir, args.out_recon, args.threads)


if __name__ == "__main__":
    main()
