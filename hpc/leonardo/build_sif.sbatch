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
#SBATCH --account=${ACCT}             # required even on budget-free serial; set via config.local.sh
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=parrot-mksif
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # serial QoS cap is cpu=8; more procs = faster squashfs = lower peak RAM
#SBATCH --mem=30000M                  # just under the serial QoS cap (mem=30800M). Singularity can't cap mksquashfs
                                      # memory (it auto-sizes from node RAM), so we give it the whole allowance.
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

# SANDBOX=1 -> build read-write sandbox DIRECTORIES instead of .sif. The .sif path
# runs mksquashfs (memory-heavy, uncapped, may OOM in the 30G serial cgroup); the
# sandbox path is extract-only (memory-light, always fits) at the cost of slower
# container startup and lots of small files on Lustre. Use it only if the .sif
# build OOMs here. The orchestrator auto-detects a <name> sandbox dir if no .sif.
SANDBOX="${SANDBOX:-0}"

echo "=== Phase B on $(hostname): ${#archives[@]} archive(s), mem=${SLURM_MEM_PER_NODE:-?}M, sandbox=$SANDBOX, $(date) ==="
for arc in "${archives[@]}"; do
  case "$arc" in
    *.oci.tar)    base="$(basename "$arc" .oci.tar)";    src="oci-archive:$arc:latest" ;;
    *.docker.tar) base="$(basename "$arc" .docker.tar)"; src="docker-archive:$arc" ;;
    *) continue ;;
  esac
  if [ "$SANDBOX" = 1 ]; then
    out="$SIF/$base"            # sandbox dir (no .sif extension)
    [ -d "$out" ] && { echo "  have   $base/ sandbox (skip)"; continue; }
    echo "  sandbox $base/  <- $(basename "$arc")  [$(date +%H:%M:%S)]"
    "$APP" build --sandbox "$out" "$src"
  else
    out="$SIF/$base.sif"
    [ -f "$out" ] && { echo "  have   $base.sif (skip)"; continue; }
    echo "  build  $base.sif  <- $(basename "$arc")  [$(date +%H:%M:%S)]"
    "$APP" build "$out" "$src"
  fi
done

echo "=== Phase B done $(date) ==="
echo "Verify with check_leonardo.sh, then reclaim space:"
echo "  rm -rf $STAGE/*.tar"