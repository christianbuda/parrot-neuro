#!/bin/bash
###############################################################################
# Phase B of the two-phase .sif build: convert the sandbox(es) staged by
# build_sif_fallback.sh into .sif files. mksquashfs needs real memory, which the
# login node doesn't give us -- so run it here, in a job, with --mem allocated.
#
# Uses lrd_all_serial: CPU-only, BUDGET-FREE, no GPU. No internet needed (it
# reads the local sandbox dir, not the registry). 4h cap is plenty.
#
#   sbatch hpc/leonardo/build_sif.sbatch /leonardo_work/<ACCT>/parrot_sif
###############################################################################
#SBATCH --account=<YOUR_ACCOUNT>      # <<EDIT>> required even on the budget-free serial partition
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=parrot-mksif
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # serial QoS: max 4 physical cores
#SBATCH --mem=32G                     # the lever: mksquashfs gets real RAM here (lower if rejected)
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
sandboxes=( "$STAGE"/*.sandbox )
[ ${#sandboxes[@]} -gt 0 ] || { echo "no sandboxes in $STAGE -- run build_sif_fallback.sh (Phase A) first."; exit 1; }

echo "=== Phase B on $(hostname): $((${#sandboxes[@]})) sandbox(es), mem=${SLURM_MEM_PER_NODE:-?}M ==="
for sb in "${sandboxes[@]}"; do
  base="$(basename "$sb" .sandbox)"
  out="$SIF/$base.sif"
  if [ -f "$out" ]; then
    echo "  have   $base.sif (skip)"
    continue
  fi
  echo "  build  $base.sif  <- $base.sandbox"
  "$APP" build "$out" "$sb"
done

echo "Done. Remove sandboxes to reclaim space (they're big):"
echo "  rm -rf $STAGE/*.sandbox"