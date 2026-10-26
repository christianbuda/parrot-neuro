#!/usr/bin/env bash
#
# Single source of truth for Parrot's Docker images.
#
# This file is *sourced* (by bin/run_reconstruction.sh and bin/build.sh); it is
# not meant to be executed directly. Define an image once here and every script
# picks it up.

# --- Runtime pins ------------------------------------------------------------
# Reference these named variables wherever a specific image is needed.
# Digests are immutable; version comments are for humans. See hpc/leonardo/README.md
# before updating a pin (CLI, output layout and custom recon specs must agree).

# External images: pulled as-is from their upstream registries, never built here.
IMG_FASTSURFER="deepmi/fastsurfer@sha256:11d594ae6fea7b1dde5b66ba5124b6295c4eedf1f401e21c7319c22eefd55c50" # 2.5.4
IMG_HIPPUNFOLD="khanlab/hippunfold@sha256:b14d6593839d09de56c21c585df2c02fe78e045d53f445f9d52727dd766a408a" # 1.5.3; v2 changes surface paths
IMG_QSIPREP="pennlinc/qsiprep@sha256:b578687c4667ae43f4229212b3ae91c2ff842d49a29d42e64c013aaae9600373" # 26.0.0; 26.1 removes CLI flags we use
IMG_QSIRECON="pennlinc/qsirecon@sha256:5f95877eeea494c5fd82be4b8dca657be63d132f0425f71d82b0448d20ff5d69" # 26.0.1

# Parrot images track the repo and are built/published by bin/build.sh.
IMG_MRI_RECONSTRUCTION="christianbuda/parrot_mri_reconstruction:latest"
IMG_FORWARD_MODEL="christianbuda/parrot_forward_model:latest"
IMG_FORWARD_SOLVERS="christianbuda/parrot_forward_solvers:latest"
IMG_QC="christianbuda/parrot_qc:latest"

# --- Derived collections -----------------------------------------------------
# Used to pull (run_reconstruction.sh) and build (build.sh) in bulk.

# External images have no build context.
EXTERNAL_IMAGES=(
    "$IMG_FASTSURFER"
    "$IMG_HIPPUNFOLD"
    "$IMG_QSIPREP"
    "$IMG_QSIRECON"
)

# Parrot images as "image_tag|build_context"; the Dockerfile is taken from
# <build_context>/Dockerfile and the path is relative to the repository root.
# Listed in build order.
PARROT_IMAGES=(
    "$IMG_MRI_RECONSTRUCTION|containers/parrot_mri_reconstruction"
    "$IMG_FORWARD_MODEL|containers/parrot_forward_model"
    "$IMG_FORWARD_SOLVERS|containers/parrot_forward_solvers"
    "$IMG_QC|containers/parrot_qc"
)

ALL_IMAGES=( "$IMG_MRI_RECONSTRUCTION" "$IMG_FORWARD_MODEL" "$IMG_FORWARD_SOLVERS" "$IMG_QC" "${EXTERNAL_IMAGES[@]}" )

# Include the digest in cache names so a changed pin cannot reuse an older SIF.
image_cache_name() {
    local base="${1##*/}"
    base="${base//@/_}"
    printf '%s\n' "${base//:/_}"
}
