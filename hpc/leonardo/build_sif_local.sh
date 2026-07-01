#!/bin/bash
###############################################################################
# Build the Parrot .sif images LOCALLY (on a workstation with Docker) and stage
# them for transfer to LEONARDO. This is the deterministic alternative to the
# two-phase login+serial-job build (build_sif_fallback.sh + build_sif.sbatch),
# which gambles on fitting mksquashfs inside the login node / serial QoS memory.
# A workstation has real RAM, so it never OOMs and produces proper single-file
# .sif (better than a Lustre sandbox for a 1000+ subject run).
#
# WHY sudo (see SUDO= below): building a .sif extracts the image into a full
# root-owned filesystem tree (root:root files, setuid bits) before squashing it.
# Creating files under a uid that isn't yours needs either real root or
# unprivileged user namespaces. Ubuntu 23.10+/24.04 blocks unprivileged userns
# (kernel.apparmor_restrict_unprivileged_userns=1), so we build as real root.
# Building as root does NOT make the image need root at runtime -- the .sif runs
# rootless on LEONARDO regardless.
#
# Needs `apptainer` (or `singularity`) installed locally:
#   sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt install -y apptainer
#
#   # build all 8 into ./parrot_sif_local
#   bash hpc/leonardo/build_sif_local.sh
#   # build a subset into a chosen dir
#   OUT=/data/parrot_sif bash hpc/leonardo/build_sif_local.sh parrot_mri_reconstruction qsiprep
#   # build AND push straight to the LEONARDO login node (host/path kept out of git)
#   DEST=user@login.leonardo.cineca.it:/leonardo_work/<ACCT>/parrot/parrot_sif/ \
#       bash hpc/leonardo/build_sif_local.sh
###############################################################################
set -uo pipefail   # NOT -e: one image failing must not abort the rest

APP="$(command -v apptainer || command -v singularity || true)"
[ -n "$APP" ] || {
  echo "ERROR: no apptainer/singularity found locally. Install it, e.g.:"
  echo "  sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt update && sudo apt install -y apptainer"
  exit 1; }

OUT="${OUT:-$PWD/parrot_sif_local}"      # where the .sif land locally
FORCE="${FORCE:-0}"                      # 1 = rebuild even if the .sif exists
DEST="${DEST:-}"                         # optional rsync target (host:path); empty = just print the cmd
# Build as real root by default (rootless userns is AppArmor-blocked on Ubuntu).
# Set SUDO= (empty) if your box allows unprivileged userns and you'd rather not sudo.
SUDO="${SUDO:-sudo}"
[ "$(id -u)" -eq 0 ] && SUDO=""          # already root -> no sudo needed
mkdir -p "$OUT"

# Keep this list in sync with prepull_sifs.sh / check_leonardo.sh / bin/images.sh.
# .sif name = <image-without-registry> with ':' -> '_', matching sif_path().
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

# Optional subset: positional args match against the image basenames, e.g.
# `... parrot_mri_reconstruction qsiprep` builds only those two.
if [ "$#" -gt 0 ]; then
  want=(); miss=()
  for a in "$@"; do
    hit=""
    for img in "${IMAGES[@]}"; do
      b="${img##*/}"; b="${b%%:*}"
      [ "$b" = "$a" ] && { want+=( "$img" ); hit=1; break; }
    done
    [ -n "$hit" ] || miss+=( "$a" )
  done
  [ "${#miss[@]:-0}" -eq 0 ] || { echo "ERROR: unknown image(s): ${miss[*]}"; echo "known: ${IMAGES[*]##*/}"; exit 2; }
  IMAGES=( "${want[@]}" )
fi

echo "== local .sif build -> $OUT  (runtime: $APP, root: ${SUDO:-yes(already)}) =="
built=(); failed=()
for img in "${IMAGES[@]}"; do
  base="${img##*/}"; base="${base//:/_}"
  sif="$OUT/${base}.sif"
  if [ -f "$sif" ] && [ "$FORCE" != 1 ]; then
    echo "  have    $base.sif (skip; FORCE=1 to rebuild)"; built+=( "$sif" ); continue
  fi

  # Prefer the LOCAL docker image (fast, no re-download) when present -- these are
  # the images you just built/pushed. Fall back to Docker Hub for anything not on
  # this machine (typically the external fastsurfer/hippunfold/qsiprep/qsirecon).
  if docker image inspect "$img" >/dev/null 2>&1; then
    src="docker-daemon://$img"; echo "  build   $base.sif  <- local docker  ($img)"
  else
    src="docker://$img";        echo "  build   $base.sif  <- Docker Hub    ($img)"
  fi

  # Build to a temp name, then rename, so an interrupted build never leaves a
  # truncated .sif that later looks "done".
  tmp="$sif.part"
  if $SUDO "$APP" build --force "$tmp" "$src"; then
    $SUDO mv -f "$tmp" "$sif"
    # Built as root -> hand ownership back so you can rsync it without sudo.
    [ -n "$SUDO" ] && $SUDO chown "$(id -u):$(id -g)" "$sif"
    built+=( "$sif" )
  else
    $SUDO rm -f "$tmp"
    echo "  FAILED  $base.sif"
    failed+=( "$img" )
  fi
done

echo
echo "built ${#built[@]}/${#IMAGES[@]} .sif in $OUT"
if [ "${#failed[@]:-0}" -gt 0 ]; then
  echo "FAILURES (${#failed[@]}): ${failed[*]}"
fi

# --- transfer to LEONARDO ----------------------------------------------------
# Compute nodes have no internet, so the .sif must be pushed from your side to
# the login node (you have no interactive data-mover login). rsync -P resumes
# partial transfers, so a dropped SSH is safe to re-run.
if [ -n "$DEST" ]; then
  echo "== rsync -> $DEST =="
  rsync -avP "$OUT"/*.sif "$DEST"
else
  echo "To copy them to LEONARDO (fill in your login host + account path):"
  echo "  rsync -avP $OUT/*.sif <USER>@login.leonardo.cineca.it:/leonardo_work/<ACCT>/parrot/parrot_sif/"
  echo "or set DEST=... and re-run to push automatically."
fi

[ "${#failed[@]:-0}" -eq 0 ] || exit 1
