#!/bin/bash
###############################################################################
# Phase A of the two-phase .sif build (for memory-limited LEONARDO login nodes).
#
# `prepull_sifs.sh` does fetch + extract + mksquashfs in one shot; the squashfs
# step OOM-kills on the login node for the big ~20GB image. This script does only
# fetch + extract into a SANDBOX directory (no mksquashfs -> memory-light, runs
# fine on the login node), staged under <sif_dir>/.staging/. Then submit
# `build_sif.sbatch` (Phase B) to convert the sandbox(es) to .sif inside a
# budget-free serial job that has real allocated memory.
#
# Run on a LOGIN node (needs internet). Skips images that already have a .sif.
#
#   bash hpc/leonardo/build_sif_fallback.sh /leonardo_work/<ACCT>/parrot_sif
#   bash hpc/leonardo/build_sif_fallback.sh /leonardo_work/<ACCT>/parrot_sif \
#        christianbuda/parrot_mri_reconstruction:latest      # one image only
###############################################################################
set -euo pipefail

APP="$(command -v apptainer || command -v singularity || true)"
[ -n "$APP" ] || { echo "ERROR: no apptainer/singularity on PATH (expected /usr/bin/singularity)."; exit 1; }

SIF="${1:?usage: build_sif_fallback.sh <sif_dir> [image ...]}"; shift || true
mkdir -p "$SIF"

# Same disk-redirect as prepull: keep cache/tmp off the RAM-backed login /tmp.
: "${APPTAINER_CACHEDIR:=$SIF/.cache}"
: "${APPTAINER_TMPDIR:=$SIF/.tmp}"
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR" SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

STAGE="$SIF/.staging"
mkdir -p "$STAGE"

# Default to the full set (keep in sync with prepull_sifs.sh); already-built .sif
# are skipped, so in practice only the image(s) that failed get re-fetched.
IMAGES=( "$@" )
if [ ${#IMAGES[@]} -eq 0 ]; then
  IMAGES=(
    christianbuda/parrot_mri_reconstruction:latest
    christianbuda/parrot_forward_model:latest
    christianbuda/parrot_forward_solvers:latest
    christianbuda/parrot_qc:latest
    deepmi/fastsurfer:latest
    khanlab/hippunfold:latest
    pennlinc/qsiprep:latest
    pennlinc/qsirecon:latest
  )
fi

echo "stage=$STAGE  cache=$APPTAINER_CACHEDIR  tmp=$APPTAINER_TMPDIR"
for img in "${IMAGES[@]}"; do
  base="${img##*/}"; base="${base//:/_}"
  sb="$STAGE/$base.sandbox"
  if [ -f "$SIF/$base.sif" ]; then
    echo "  have .sif       $base.sif (skip)"
  elif [ -d "$sb" ]; then
    echo "  have sandbox    $base.sandbox (skip; run Phase B to make .sif)"
  else
    echo "  fetch sandbox   $base  <- docker://$img"
    "$APP" build --sandbox "$sb" "docker://$img"
  fi
done
echo
echo "Phase A done. Now submit Phase B to build the .sif(s):"
echo "  sbatch hpc/leonardo/build_sif.sbatch $SIF"