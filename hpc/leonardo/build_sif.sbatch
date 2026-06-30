#!/bin/bash
###############################################################################
# Phase B of the two-phase .sif build: convert the local archive(s) that
# build_sif_fallback.sh downloaded into .sif files. The extract + mksquashfs
# needs real memory (the login node OOM-kills it), so we run it here in a job
# with --mem allocated. No internet needed -- it reads the local archive.
#
# Uses lrd_all_serial: CPU-only, BUDGET-FREE, no GPU. Idempotent: skips any .sif
# already built, so if it hits the 4h wall just re-submit and it resumes.
#
#   sbatch hpc/leonardo/build_sif.sbatch /leonardo_work/<ACCT>/parrot_sif
###############################################################################
#SBATCH --account=<YOUR_ACCOUNT>      # <<EDIT>> required even on the budget-free serial partition
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=parrot-mksif
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # serial QoS: max 4 physical cores
#SBATCH --mem=32G                     # the lever: extract+mksquashfs get real RAM (raise if a big image is killed)
#SBATCH --time=04:00:00
#SBATCH --output=parrot-mksif-%j.out
#SBATCH --error=parrot-mksif-%j.err

set -euo pipefail

SIF="${1:?usage: sbatch build_sif.sbatch <sif_dir>}"
APP="$(command -v apptainer || command -v singularity || true)"
[ -n "$APP" ] || { echo "ERROR: no apptainer/singularity on PATH."; exit 1; }

# squashfs scratch on the disk-backed work FS (not RAM/tmpfs).
: "${APPTAINER_TMPDIR:=$SIF/.tmp}"
export APPTAINER_TMPDIR SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_TMPDIR"

STAGE="$SIF/.staging"
shopt -s nullglob
archives=( "$STAGE"/*.oci.tar "$STAGE"/*.docker.tar )
[ ${#archives[@]} -gt 0 ] || { echo "no archives in $STAGE -- run build_sif_fallback.sh (Phase A) first."; exit 1; }

echo "=== Phase B on $(hostname): ${#archives[@]} archive(s), mem=${SLURM_MEM_PER_NODE:-?}M, $(date) ==="
for arc in "${archives[@]}"; do
  case "$arc" in
    *.oci.tar)    base="$(basename "$arc" .oci.tar)";    src="oci-archive:$arc:latest" ;;
    *.docker.tar) base="$(basename "$arc" .docker.tar)"; src="docker-archive:$arc" ;;
    *) continue ;;
  esac
  out="$SIF/$base.sif"
  if [ -f "$out" ]; then echo "  have   $base.sif (skip)"; continue; fi
  echo "  build  $base.sif  <- $(basename "$arc")  [$(date +%H:%M:%S)]"
  "$APP" build "$out" "$src"
done

echo "=== Phase B done $(date) ==="
echo "Verify with check_leonardo.sh, then reclaim space:"
echo "  rm -rf $STAGE/*.tar"