#!/usr/bin/env bash
#
# Single source of truth for Parrot's Docker images.
#
# This file is *sourced* (by bin/run_reconstruction.sh and bin/build.sh); it is
# not meant to be executed directly. Define an image once here and every script
# picks it up.

# --- Image tags --------------------------------------------------------------
# Reference these named variables wherever a specific image is needed.

# External images: pulled as-is from their upstream registries, never built here.
IMG_FASTSURFER="deepmi/fastsurfer:latest"
IMG_HIPPUNFOLD="khanlab/hippunfold:latest"
IMG_QSIPREP="pennlinc/qsiprep:latest"
IMG_QSIRECON="pennlinc/qsirecon:latest"

# Parrot images: built and published by bin/build.sh.
IMG_MRI_RECONSTRUCTION="christianbuda/parrot_mri_reconstruction:latest"
IMG_FORWARD_MODEL="christianbuda/parrot_forward_model:latest"
IMG_FORWARD_SOLVERS="christianbuda/parrot_forward_solvers:latest"

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
)
