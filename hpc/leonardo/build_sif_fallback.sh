#!/bin/bash
###############################################################################
# Phase A of the two-phase .sif build (for memory-limited LEONARDO login nodes).
#
# Even fetch+EXTRACT (`singularity build --sandbox` / `singularity pull`) gets
# OOM/arbiter-killed on the login node for the big multi-GB images. So here we
# only DOWNLOAD each image into a single local archive file -- no extraction,
# memory-light, survives the login node. Phase B (build_sif.sbatch) then does the
# heavy extract+squashfs from that local archive inside a compute job that has
# real memory (and needs no internet).
#
# singularity can't download-without-extracting, so we use skopeo or crane. If
# neither is installed we fetch a static `crane` binary (login node has internet).
#
# Run on a LOGIN node (needs internet), ideally in tmux. Skips anything that
# already has a .sif or a staged archive.
#
#   bash hpc/leonardo/build_sif_fallback.sh /leonardo_work/<ACCT>/parrot_sif
#   bash hpc/leonardo/build_sif_fallback.sh /leonardo_work/<ACCT>/parrot_sif \
#        christianbuda/parrot_mri_reconstruction:latest      # one image only
###############################################################################
set -euo pipefail

SIF="${1:?usage: build_sif_fallback.sh <sif_dir> [image ...]}"; shift || true
mkdir -p "$SIF"
STAGE="$SIF/.staging"; mkdir -p "$STAGE"

# Remove leftovers from any killed `singularity build` attempts (frees inodes).
rm -rf "$STAGE"/build-temp-* "$STAGE"/*.sandbox 2>/dev/null || true

# Pick a download-only tool. Prefer skopeo, then crane; else fetch static crane.
DL=""; CRANE=""
if command -v skopeo >/dev/null 2>&1; then
  DL=skopeo
elif command -v crane >/dev/null 2>&1; then
  DL=crane; CRANE=crane
else
  CB="$STAGE/bin"; mkdir -p "$CB"
  echo "no skopeo/crane found -> fetching a static crane binary into $CB ..."
  url="https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz"
  if curl -fsSL "$url" | tar -xz -C "$CB" crane 2>/dev/null && [ -x "$CB/crane" ]; then
    DL=crane; CRANE="$CB/crane"; echo "  ok: $CRANE"
  else
    echo "ERROR: no skopeo/crane and could not fetch crane. Install one and re-run."; exit 1
  fi
fi
echo "downloader: $DL"

# Keep in sync with prepull_sifs.sh / bin/images.sh. Already-built .sif are skipped.
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

for img in "${IMAGES[@]}"; do
  base="${img##*/}"; base="${base//:/_}"
  if [ -f "$SIF/$base.sif" ]; then
    echo "  have .sif     $base.sif (skip)"; continue
  fi
  if [ "$DL" = skopeo ]; then
    arc="$STAGE/$base.oci.tar"
    [ -f "$arc" ] && { echo "  have archive  $base.oci.tar (skip)"; continue; }
    echo "  download      $base.oci.tar  <- docker://$img"
    # write to a .part then rename, so a killed download doesn't look complete
    skopeo copy "docker://$img" "oci-archive:$arc.part:latest" && mv "$arc.part" "$arc"
  else
    arc="$STAGE/$base.docker.tar"
    [ -f "$arc" ] && { echo "  have archive  $base.docker.tar (skip)"; continue; }
    echo "  download      $base.docker.tar  <- $img"
    "$CRANE" pull "$img" "$arc.part" && mv "$arc.part" "$arc"
  fi
done
echo
echo "Phase A done. Build the .sif(s) in a budget-free serial job:"
echo "  sbatch hpc/leonardo/build_sif.sbatch $SIF"