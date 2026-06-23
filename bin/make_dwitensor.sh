#!/bin/bash
#
# Fit a diffusion tensor from the QSIPrep-preprocessed DWI, for downstream
# white-matter anisotropic-conductivity modelling.
#
# Runs INSIDE the QSIRecon image (not a Parrot image) so the fit uses the *same*
# MRtrix3 that preprocessed the DWI (zero version drift). Operates entirely on
# the mounted /derivatives tree.
#
#   Usage (inside the container):  make_dwitensor.sh <subject_id>
#
# SCOPE / SPACE: this stage stays in QSIPrep's ACPC (T1w-aligned) space -- the
# space the preprocessed DWI lives in -- which is NOT the Parrot FEM-mesh space
# (FastSurfer/charm-conformed, different grid + orientation, ~rigid offset).
# Resampling onto the mesh grid AND reorienting the tensor eigenvectors by that
# transform is a separate, not-yet-built step; do it there, not here. Fitting in
# native diffusion space keeps this step accurate and self-contained.
#
# Outputs (derivatives/dwitensor/sub-<ID>/), all space-ACPC:
#   *_model-dti_tensor.nii.gz        6-vol MRtrix tensor (D11 D22 D33 D12 D13 D23);
#                                    canonical artefact -- everything below derives from it
#   *_model-dti_param-eigvecs.nii.gz 9-vol eigenframe v1|v2|v3 (orthotropic axes)
#   *_model-dti_param-eigvals.nii.gz 3-vol eigenvalues (lambda1>=2>=3)
#   *_model-dti_param-fa.nii.gz      fractional anisotropy (QC + fixed-ratio fallback)

set -euo pipefail

SUB="$1"
FORMAT="${2:-}"            # "" = QSIPrep ACPC (default); "hcp" = HCP-YA native tree
OUT="/derivatives/dwitensor/sub-${SUB}"
mkdir -p "$OUT"

if [ "$FORMAT" = "hcp" ]; then
    # HCP-YA's preprocessed DWI already lives in the subject T1w (ACPC) space, which
    # Parrot takes as the mesh space (identity) -- so we fit directly and label the
    # output space-T1, and the separate dwi2t1 step is skipped for HCP. No gradient
    # rotation is needed (no resampling happens here, unlike the qsiprep->T1 path).
    DDIR="/hcp_in/${SUB}/T1w/Diffusion"
    DWI="$DDIR/data.nii.gz"
    BVEC="$DDIR/bvecs"
    BVAL="$DDIR/bvals"
    MASK="$DDIR/nodif_brain_mask.nii.gz"
    PRE="$OUT/sub-${SUB}_space-T1_model-dti"
else
    DWI_DIR="/derivatives/qsiprep/sub-${SUB}/dwi"
    DWI=$(ls "$DWI_DIR"/*space-ACPC_desc-preproc_dwi.nii.gz | head -n 1)
    BVEC="${DWI%.nii.gz}.bvec"
    BVAL="${DWI%.nii.gz}.bval"
    MASK=$(ls "$DWI_DIR"/*space-ACPC_desc-brain_mask.nii.gz | head -n 1)
    PRE="$OUT/sub-${SUB}_space-ACPC_model-dti"
fi

echo "DWI  : $DWI"
echo "bvec : $BVEC"
echo "bval : $BVAL"
echo "mask : $MASK"

# Iteratively-reweighted least-squares tensor fit (MRtrix default). Works on a
# single non-zero shell + b0, which is all DTI needs.
dwi2tensor "$DWI" "${PRE}_tensor.nii.gz" \
    -fslgrad "$BVEC" "$BVAL" \
    -mask "$MASK" \
    -quiet -force

# Derive the full eigen-frame + FA the conductivity model consumes. -num is
# shared by -vector and -value, so 1,2,3 yields all three eigenvectors (9 vols:
# v1|v2|v3) and all three eigenvalues -- the orthotropic axes + magnitudes.
tensor2metric "${PRE}_tensor.nii.gz" \
    -mask "$MASK" \
    -num 1,2,3 \
    -vector "${PRE}_param-eigvecs.nii.gz" \
    -value "${PRE}_param-eigvals.nii.gz" \
    -fa "${PRE}_param-fa.nii.gz" \
    -quiet -force

echo "DWI tensor + eigen maps written for sub-${SUB}."
