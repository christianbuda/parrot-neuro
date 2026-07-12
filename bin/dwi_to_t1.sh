#!/bin/bash
#
# Bring the QSIPrep/QSIRecon DWI derivatives into Parrot T1/mesh space
# (raw/sub-<ID>/T1.nii.gz -- the grid the mesh, atlas, dipoles and electrodes all
# live in). Step [B] of the white-matter anisotropy feature; also closes the
# connectome registration gap (tracts then share the atlas's space).
#
# Runs INSIDE the QSIRecon image (ANTs + MRtrix + python/numpy/nibabel).
#
#   Usage (inside the container):  dwi_to_t1.sh <subject_id> [scratch_base_dir]
#
# REGISTRATION: we do NOT re-register. QSIPrep already aligned the DWI to the
# subject T1 and AC-PC'd it; its `from-ACPC_to-anat` transform maps ACPC -> the
# original T1 grid, which (verified) is bit-identical to Parrot's raw/T1 space
# (qsiprep `from-orig_to-anat` is identity). We reuse that transform; applying it
# natively with ANTs gives NCC ~0.86. (mrregister diverged here; reusing the
# existing, validated transform is both correct and robust.)
#
# TENSOR, qsiprep-style: resample the DWI into T1 with ANTs, rotate the DW
# gradients by the transform's rotation (done in MRtrix WORLD frame, so the
# ACPC-LAS vs T1-RAS voxel flip never enters), then re-fit dwi2tensor in T1.
# Validated: FA corr 0.979 vs the ACPC fit; brainstem V1 is S-I (z) dominant and
# the colour-FA is canonical (CC red, CST blue) -- the rotation sense is R^T.
#
# TRACTS: tcktransform via a deformation field built from the (rigid) transform.
# Validated: 100% of transformed streamlines fall in the brain, TDI-vs-FA corr
# 0.84 (streamlines follow WM).
#
# The DWI->T1 transform and all T1-space products are written under dwitensor/;
# this stage has no output folder of its own. Heavy scratch goes to a swept work
# dir (2nd arg; the orchestrator passes its WORK_DIR), never the derivatives tree.

set -euo pipefail

SUB="$1"
WORK_BASE="${2:-/derivatives}"                   # orchestrator passes its swept WORK_DIR
DWI_DIR="/derivatives/qsiprep/sub-${SUB}/dwi"
ANAT_DIR="/derivatives/qsiprep/sub-${SUB}/anat"
QSIREC_DIR="/derivatives/qsirecon/sub-${SUB}/dwi"
RAW_T1="/derivatives/raw/sub-${SUB}/T1.nii.gz"
TEN_DIR="/derivatives/dwitensor/sub-${SUB}"; mkdir -p "$TEN_DIR"
SCRATCH="$WORK_BASE/.dwi2t1_scratch_${SUB}"; mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

ACPC_DWI=$(ls "$DWI_DIR"/*space-ACPC_desc-preproc_dwi.nii.gz | head -n 1)
BVEC="${ACPC_DWI%.nii.gz}.bvec"
BVAL="${ACPC_DWI%.nii.gz}.bval"
ACPC_MASK=$(ls "$DWI_DIR"/*space-ACPC_desc-brain_mask.nii.gz | head -n 1)
ACPC2ANAT="$ANAT_DIR/sub-${SUB}_from-ACPC_to-anat_mode-image_xfm.mat"

RAS="$TEN_DIR/sub-${SUB}_from-ACPC_to-T1_ras.txt"   # provenance: the applied transform
MASK_T1="$TEN_DIR/sub-${SUB}_space-T1_desc-brain_mask.nii.gz"
PRE_T1="$TEN_DIR/sub-${SUB}_space-T1_model-dti"
DWI_T1="$SCRATCH/dwi_space-T1.nii.gz"

echo "ACPC DWI : $ACPC_DWI"
echo "raw T1   : $RAW_T1"
echo "transform: $ACPC2ANAT (reused from QSIPrep)"

# --- 0. Pull the qsiprep transform into a RAS 4x4 matrix (numpy-friendly) ------
ConvertTransformFile 3 "$ACPC2ANAT" "$RAS" --hm --RAS

# --- 1. TENSOR: resample DWI (ANTs) + rotate gradients (R^T, world frame) + refit
echo "== [1] DWI -> T1 (ANTs resample) + gradient rotation + dwi2tensor =="
antsApplyTransforms -d 3 -e 3 -i "$ACPC_DWI" -r "$RAW_T1" -t "$ACPC2ANAT" -o "$DWI_T1" -n Linear
antsApplyTransforms -d 3 -i "$ACPC_MASK" -r "$RAW_T1" -t "$ACPC2ANAT" -o "$MASK_T1" -n NearestNeighbor
# Erode the (ANTs-resampled) brain mask by one voxel for the TENSOR FIT ONLY.
# Linear resampling undershoots to <=0 on the mask rim, and dwi2tensor's
# log-domain fit turns that into whole-voxel NaN tensors (327 rim voxels on
# sub-010024) that later poison the anisotropy stage's batched eigh. The written
# brain-mask artifact stays the full mask; only the fit + metrics below use the
# eroded one, so no degenerate rim voxel is ever estimated in the first place.
MASK_FIT="$SCRATCH/sub-${SUB}_space-T1_desc-fit_mask.mif"
maskfilter "$MASK_T1" erode -npass 1 "$MASK_FIT" -quiet -force
mrconvert "$ACPC_DWI" -fslgrad "$BVEC" "$BVAL" -export_grad_mrtrix "$SCRATCH/grad_acpc.b" "$SCRATCH/dwi_acpc.mif" -quiet -force
python3 - "$RAS" "$SCRATCH/grad_acpc.b" "$SCRATCH/grad_T1.b" <<'PY'
import sys, numpy as np
ras, gin, gout = sys.argv[1:4]
R = np.loadtxt(ras)[:3, :3]
U, _, Vt = np.linalg.svd(R); R = U @ Vt          # nearest pure rotation
g = np.loadtxt(gin)
g[:, :3] = g[:, :3] @ R                            # rotate world gradients by R^T (validated sense)
np.savetxt(gout, g)
PY
mrconvert "$DWI_T1" -grad "$SCRATCH/grad_T1.b" "$SCRATCH/dwi_T1.mif" -quiet -force
dwi2tensor "$SCRATCH/dwi_T1.mif" "${PRE_T1}_tensor.nii.gz" -mask "$MASK_FIT" -quiet -force
# Belt-and-suspenders: guarantee a finite canonical tensor artifact even if the
# fit still emits a non-finite voxel (log-domain singularities the erosion above
# doesn't catch). Whole-voxel zero -> the anisotropy stage's iso fallback; also
# keeps tensor2metric's FA/eigval/eigvec maps clean.
python3 - "${PRE_T1}_tensor.nii.gz" <<'PY'
import sys, numpy as np, nibabel as nib
f = sys.argv[1]
img = nib.load(f)
d = np.asarray(img.dataobj, dtype=np.float32)         # writable copy, native tensor dtype
bad = ~np.isfinite(d).all(axis=-1)                    # any non-finite component -> whole voxel
if bad.any():
    d[bad] = 0.0
    nib.Nifti1Image(d, img.affine, img.header).to_filename(f)
    print(f"[sanitize] zeroed {int(bad.sum())} non-finite tensor voxels")
PY
tensor2metric "${PRE_T1}_tensor.nii.gz" -mask "$MASK_FIT" \
    -num 1,2,3 \
    -vector "${PRE_T1}_param-eigvecs.nii.gz" \
    -value "${PRE_T1}_param-eigvals.nii.gz" \
    -fa "${PRE_T1}_param-fa.nii.gz" \
    -quiet -force

# --- 2. TRACTS: deformation field from the rigid transform, then tcktransform ---
TCKGZ=$(ls "$QSIREC_DIR"/*space-ACPC*streamlines.tck.gz 2>/dev/null | head -n 1 || true)
if [ -n "$TCKGZ" ]; then
    echo "== [2] tractogram -> T1 =="
    warpinit "$RAW_T1" "$SCRATCH/idwarp.nii.gz" -quiet -force
    python3 - "$RAS" "$SCRATCH/idwarp.nii.gz" "$SCRATCH/warp_def.nii.gz" <<'PY'
import sys, numpy as np, nibabel as nib
ras, idw, out = sys.argv[1:4]
A = np.linalg.inv(np.loadtxt(ras))                 # ACPC->raw applied to raw-grid positions (validated)
w = nib.load(idw); P = np.asarray(w.dataobj).astype(np.float64)
flat = P.reshape(-1, 3).T
defm = (A[:3, :3] @ flat + A[:3, 3:4]).T.reshape(P.shape)
nib.Nifti1Image(defm.astype(np.float32), w.affine, w.header).to_filename(out)
PY
    TCK_ACPC="$SCRATCH/tracts_space-ACPC.tck"
    TCK_T1="$QSIREC_DIR/sub-${SUB}_space-T1_model-ifod2_streamlines.tck"
    gunzip -c "$TCKGZ" > "$TCK_ACPC"               # MRtrix can't read .tck.gz
    tcktransform "$TCK_ACPC" "$SCRATCH/warp_def.nii.gz" "$TCK_T1" -quiet -force
    gzip -f "$TCK_T1"
else
    echo "== [2] no ACPC tractogram found -- skipping streamline transform =="
fi

echo "DWI derivatives mapped to T1 space for sub-${SUB}."
