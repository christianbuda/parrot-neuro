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
# Re-pulls only images that CHANGED on the registry: it compares each remote
# manifest digest (via skopeo/crane, if available) against a `<sif>.digest`
# sidecar and re-pulls just the stale ones. If no digest tool is present it
# leaves existing .sif files alone -- set FORCE=1 to re-pull everything anyway.
#
#   bash hpc/leonardo/prepull_sifs.sh /leonardo_work/<ACCT>/parrot_sif
#   FORCE=1 bash hpc/leonardo/prepull_sifs.sh /leonardo_work/<ACCT>/parrot_sif
###############################################################################
set -euo pipefail

APP="$(command -v apptainer || command -v singularity || true)"
[ -n "$APP" ] || { echo "ERROR: no apptainer/singularity on PATH (expected /usr/bin/singularity). Ask CINECA."; exit 1; }

SIF="${1:?usage: prepull_sifs.sh <sif_dir>, e.g. /leonardo_work/<ACCT>/parrot_sif}"
FORCE="${FORCE:-0}"
mkdir -p "$SIF"

# Echo the remote manifest digest (sha256:...) for a docker image, or nothing if
# we have no tool to query it without downloading the whole image.
remote_digest() {
  local img=$1
  if command -v skopeo >/dev/null 2>&1; then
    skopeo inspect --format '{{.Digest}}' "docker://$img" 2>/dev/null && return 0
  fi
  if command -v crane >/dev/null 2>&1; then
    crane digest "$img" 2>/dev/null && return 0
  fi
  return 0   # no tool -> empty (caller falls back to keep-or-FORCE)
}
if ! command -v skopeo >/dev/null 2>&1 && ! command -v crane >/dev/null 2>&1; then
  echo "NOTE: no skopeo/crane found -> cannot auto-detect updates; existing .sif kept (use FORCE=1 to re-pull)."
fi

# Keep this list in sync with bin/images.sh. .sif name = <image-without-registry>
# with ':' -> '_', matching sif_path() in bin/run_reconstruction.sh.
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

for img in "${IMAGES[@]}"; do
  base="${img##*/}"; base="${base//:/_}"
  sif="$SIF/${base}.sif"
  dgf="$sif.digest"
  remote="$(remote_digest "$img")"

  if [ -f "$sif" ] && [ "$FORCE" != 1 ]; then
    if [ -n "$remote" ] && [ -f "$dgf" ] && [ "$(cat "$dgf")" = "$remote" ]; then
      echo "  up-to-date  $base.sif"
      continue
    elif [ -z "$remote" ]; then
      echo "  have        $base.sif  (no digest tool; FORCE=1 to re-pull)"
      continue
    else
      echo "  UPDATE      $base.sif  (registry changed -> re-pulling)"
    fi
  else
    echo "  pull        $base.sif  <- docker://$img"
  fi

  "$APP" pull --force "$sif" "docker://$img"
  [ -n "$remote" ] && printf '%s\n' "$remote" > "$dgf"
done
echo "Done. .sif cache: $SIF"
