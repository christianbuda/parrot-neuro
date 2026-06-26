#!/bin/bash
#
# Build (and optionally push) the Parrot Docker images.
#
# The image list and build contexts live in bin/images.sh, the same manifest
# bin/run_reconstruction.sh uses to pull, so building and pulling never drift apart.
#
# Usage:
#   ./bin/build.sh [OPTIONS] [FILTER ...]
#
# Options:
#   --push           Push each image to its registry after a successful build.
#   --version X.Y.Z  Also tag (and, with --push, push) each image as :X.Y.Z in
#                    addition to :latest. The same version is applied to every
#                    image built in this run. --tag is an alias.
#   --no-cache       Build without using Docker's layer cache.
#   -h, --help       Show this help and exit.
#
# Arguments:
#   FILTER        Optional substring(s); only images whose tag contains a filter
#                 are built (e.g. "solvers" builds parrot_forward_solvers only).
#                 With no filter, all Parrot images are built.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
source "$SCRIPT_DIR/images.sh"

usage() {
    cat <<'EOF'
Build (and optionally push) the Parrot Docker images.

Usage:
  ./bin/build.sh [OPTIONS] [FILTER ...]

Options:
  --push           Push each image to its registry after a successful build.
  --version X.Y.Z  Also tag (and, with --push, push) each image as :X.Y.Z in
                   addition to :latest. The same version is applied to every
                   image built in this run. --tag is an alias.
  --no-cache       Build without using Docker's layer cache.
  -h, --help       Show this help and exit.

Arguments:
  FILTER        Optional substring(s); only images whose tag contains a filter
                are built (e.g. "solvers" builds parrot_forward_solvers only).
                With no filter, all Parrot images are built.
EOF
    exit "${1:-0}"
}

PUSH=false
NO_CACHE=()
FILTERS=()
VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)       PUSH=true; shift ;;
        --version|--tag)
            [[ $# -ge 2 ]] || { echo "$1 needs a value (e.g. --version 1.0.1)" >&2; usage 1; }
            VERSION="$2"; shift 2 ;;
        --version=*|--tag=*)
            VERSION="${1#*=}"; shift ;;
        --no-cache)   NO_CACHE=(--no-cache); shift ;;
        -h|--help)    usage 0 ;;
        -*)           echo "Unknown option: $1" >&2; usage 1 ;;
        *)            FILTERS+=("$1"); shift ;;
    esac
done

# Decide whether an image tag should be built given the active filters.
matches_filter() {
    local tag="$1"
    [[ ${#FILTERS[@]} -eq 0 ]] && return 0
    for f in "${FILTERS[@]}"; do
        [[ "$tag" == *"$f"* ]] && return 0
    done
    return 1
}

built=()
for entry in "${PARROT_IMAGES[@]}"; do
    tag="${entry%%|*}"
    context="${entry##*|}"

    matches_filter "$tag" || continue

    # When a version is requested, derive the same repo:VERSION ref by replacing
    # the :latest suffix so the single build gets both tags (same image ID).
    tags=("$tag")
    [[ -n "$VERSION" ]] && tags+=("${tag%:*}:$VERSION")

    build_args=()
    for t in "${tags[@]}"; do build_args+=(-t "$t"); done

    echo "====================================================================="
    echo "Building ${tags[*]}"
    echo "  context: $REPO_ROOT/$context"
    echo "====================================================================="
    docker build "${NO_CACHE[@]}" "${build_args[@]}" "$REPO_ROOT/$context"

    if [ "$PUSH" = true ]; then
        for t in "${tags[@]}"; do
            echo "Pushing $t..."
            docker push "$t"
        done
    fi

    built+=("${tags[@]}")
done

if [ ${#built[@]} -eq 0 ]; then
    echo "No images matched filter(s): ${FILTERS[*]}" >&2
    exit 1
fi

echo ""
echo "Done. Built ${#built[@]} image(s):"
printf '  %s\n' "${built[@]}"
if [ "$PUSH" = true ]; then
    echo "All pushed to their registries."
fi
