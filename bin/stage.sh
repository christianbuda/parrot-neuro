#!/usr/bin/env bash
#
# Stage a raw cohort dataset into a flattened, Parrot-ready BIDS dataset.
#
# Thin host-side launcher: it runs the per-cohort Python stager (utils/staging/
# <cohort>.py) INSIDE the parrot_mri_reconstruction image, because the staging
# code needs nibabel (not installed on the host). The image, cohort scripts, and
# mounts are wired up here so the gnarly `docker run` invocation lives in one place.
#
#   Usage:  ./bin/stage.sh <cohort> <src_dir> <bids_out_dir> [subject ...]
#
#   <cohort>        name of a stager in utils/staging/ (e.g. "lemon")
#   <src_dir>       raw source root, mounted read-only at /src
#   <bids_out_dir>  target BIDS dataset, mounted read-write at /dst
#   [subject ...]   optional subject IDs (default: the stager's own default)
#
# Example:
#   ./bin/stage.sh lemon /srv/.../MRI_MPILMBB_LEMON/MRI_Raw /srv/.../BIDS_LEMON sub-010002

set -euo pipefail

PARROT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

# Docker image definitions (single source of truth, shared with run_reconstruction.sh)
source "$PARROT_SCRIPT_DIR/bin/images.sh"

if [ "$#" -lt 3 ]; then
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
fi

COHORT="$1"
SRC_DIR="$2"
BIDS_DIR="$3"
shift 3
SUBJECTS=("$@")  # may be empty -> stager uses its built-in default

STAGER="$PARROT_SCRIPT_DIR/utils/staging/${COHORT}.py"
if [ ! -f "$STAGER" ]; then
    echo "[ERROR] no stager for cohort '$COHORT' (expected utils/staging/${COHORT}.py)" >&2
    echo "        available: $(cd "$PARROT_SCRIPT_DIR/utils/staging" && ls *.py 2>/dev/null | grep -v '^common.py$' | sed 's/\.py$//' | tr '\n' ' ')" >&2
    exit 1
fi
if [ ! -d "$SRC_DIR" ]; then
    echo "[ERROR] source dir not found: $SRC_DIR" >&2
    exit 1
fi
mkdir -p "$BIDS_DIR"

echo "Staging cohort '$COHORT': $SRC_DIR -> $BIDS_DIR ${SUBJECTS[*]:-(default subjects)}"

# --user keeps output ownership as the caller; HOME/XDG point at /tmp because the
# host UID has no home inside the image. utils/staging is mounted read-only at /work
# (both <cohort>.py and its common.py import partner live there, so the sibling
# `import common` resolves). The cohort script reads /src and writes /dst.
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e XDG_CACHE_HOME=/tmp \
    -v "$SRC_DIR":/src:ro \
    -v "$BIDS_DIR":/dst \
    -v "$PARROT_SCRIPT_DIR/utils/staging":/work:ro \
    --entrypoint micromamba \
    "$IMG_MRI_RECONSTRUCTION" \
    run -n neuro python "/work/${COHORT}.py" "${SUBJECTS[@]}"

echo "Done. Staged BIDS dataset at: $BIDS_DIR"
