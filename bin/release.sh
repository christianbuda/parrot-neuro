#!/bin/bash
#
# Release Parrot: bump the image version, build + push the images, commit, tag, push git.
#
# Usage:
#   ./bin/release.sh X.Y.Z
#
# Steps (stops at the first failure):
#   1. Checks: valid version, on main, clean working tree, tag vX.Y.Z does not exist yet.
#   2. Sets PARROT_VERSION in bin/images.sh.
#   3. Builds and pushes all Parrot images (./bin/build.sh --push). If this fails,
#      bin/images.sh is restored and nothing is committed or tagged.
#   4. Commits the version bump, creates the annotated tag vX.Y.Z, pushes main and the tag.
#
# Run it from a machine with Docker, logged in to the registry, and push access to git.

set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$REPO_ROOT"

VERSION="${1:-}"
case "$VERSION" in
    -h|--help|"") sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Version must look like X.Y.Z, got '$VERSION'" >&2; exit 1; }

[[ "$(git branch --show-current)" == "main" ]] || { echo "Release from the main branch." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Working tree is not clean; commit or stash first." >&2; exit 1; }
git rev-parse -q --verify "refs/tags/v$VERSION" > /dev/null && { echo "Tag v$VERSION already exists." >&2; exit 1; }

sed -i -E "s/^PARROT_VERSION=\"[^\"]*\"/PARROT_VERSION=\"$VERSION\"/" bin/images.sh
grep -q "^PARROT_VERSION=\"$VERSION\"" bin/images.sh || { git checkout -- bin/images.sh; echo "Could not set PARROT_VERSION in bin/images.sh" >&2; exit 1; }

if ! ./bin/build.sh --push; then
    git checkout -- bin/images.sh
    echo "Build or push failed; bin/images.sh restored, nothing committed or tagged." >&2
    exit 1
fi

git commit -q -m "chore: release $VERSION" bin/images.sh
git tag -a "v$VERSION" -m "v$VERSION"
git push origin main "v$VERSION"

echo "Released v$VERSION (images pushed as :$VERSION and :latest)."
