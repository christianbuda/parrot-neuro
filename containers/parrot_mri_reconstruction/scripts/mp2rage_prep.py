#!/usr/bin/env python3
"""MP2RAGE preconditioning for the Parrot pipeline.

MPRAGEises a raw MP2RAGE UNI into the working T1 that every stage consumes:

  --out-t1 : MPRAGEised UNI (presurfer-style). N3-bias-correct INV2, scale it to
             [0, 1], and use it as a SOFT weight on the UNI. Air -> ~0 (INV2 ~ 0
             there) so the high-intensity salt-and-pepper background is suppressed,
             while CSF inside the head is preserved (INV2 has real signal there) --
             unlike a hard INV2 threshold, which also drops dark interior CSF.
             Full head; becomes the subject's T1 for every stage (FastSurfer,
             HippUnfold, charm, MNE BEM, ...).

Runs inside the parrot_mri_reconstruction image under FreeSurfer's fspython:
mri_nu_correct.mni (N3) comes from the sourced FreeSurfer env and nibabel is available
there. (antspyx's N4 lives only in the separate `neuro` env; N3 is plenty for a soft
suppression weight.)
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


def main() -> None:
    p = argparse.ArgumentParser(description="MP2RAGE preconditioning (MPRAGEise).")
    p.add_argument("--uni", required=True, help="MP2RAGE UNI image")
    p.add_argument("--inv2", required=True, help="MP2RAGE INV2 image")
    p.add_argument("--out-t1", required=True, help="MPRAGEised full-head T1 (working T1 for all stages)")
    args = p.parse_args()

    mprageise(args.uni, args.inv2, args.out_t1)


if __name__ == "__main__":
    main()
