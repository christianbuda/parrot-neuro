#!/bin/bash
###############################################################################
# Pre-pull all Parrot .sif images into $WORK/parrot_sif.
#
# Run this on a LOGIN or DATA-MOVER node (data.leonardo.cineca.it) -- NOT inside
# a batch job: compute nodes on LEONARDO generally have no outbound internet, so
# `apptainer pull docker://...` only works from the login/data-mover side. Do it
# once; the GPU pilot/array jobs then reuse the cached .sif files.
#
# NOTE: the Parrot images must already be published to Docker Hub as :latest
# (the rootless build). `./bin/build.sh --push` from a machine that has Docker.
#
# On LEONARDO, Singularity is a SYSTEM command (/usr/bin/singularity), not a module --
# no `module load` needed. (No `apptainer`; singularity is the binary here.)
#
#   bash hpc/leonardo/prepull_sifs.sh /leonardo_work/<ACCT>/parrot_sif
###############################################################################
set -euo pipefail

APP="$(command -v apptainer || command -v singularity || true)"
[ -n "$APP" ] || { echo "ERROR: no apptainer/singularity on PATH (expected /usr/bin/singularity). Ask CINECA."; exit 1; }

SIF="${1:?usage: prepull_sifs.sh <sif_dir>, e.g. /leonardo_work/<ACCT>/parrot_sif}"
mkdir -p "$SIF"

# Keep this list in sync with bin/images.sh. .sif name = <image-without-registry>
# with ':' -> '_', matching sif_path() in bin/run_reconstruction.sh.
IMAGES=(
  christianbuda/parrot_mri_reconstruction:latest
  christianbuda/parrot_forward_model:latest
  christianbuda/parrot_forward_solvers:latest
  deepmi/fastsurfer:latest
  khanlab/hippunfold:latest
  pennlinc/qsiprep:latest
  pennlinc/qsirecon:latest
)

for img in "${IMAGES[@]}"; do
  base="${img##*/}"; base="${base//:/_}"
  sif="$SIF/${base}.sif"
  if [ -f "$sif" ]; then
    echo "  have  $(basename "$sif")"
  else
    echo "  pull  docker://$img -> $(basename "$sif")"
    "$APP" pull "$sif" "docker://$img"
  fi
done
echo "Done. .sif cache: $SIF"
