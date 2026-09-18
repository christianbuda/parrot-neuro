"""Code shared by more than one Parrot container image.

Each image's Docker build context is its own containers/<image>/ directory, so a
Dockerfile cannot COPY from outside it. bin/build.sh therefore passes this directory as
a BuildKit *named build context* (`--build-context shared=containers/_shared`) and each
Dockerfile does:

    COPY --from=shared parrot_common /opt/parrot_common
    ENV PYTHONPATH="/opt/parrot_common:${PYTHONPATH}"

Every Parrot image gets it, whether or not it imports from it today, so sharing a helper
later never needs a Dockerfile change.

Consequence to know: nothing detects which images depend on this code. After editing it,
rebuild every image that imports it (in practice: ./bin/build.sh).

Submodules are split by dependency so importing one never drags in deps an image lacks;
this package imports nothing itself for the same reason.

    thresholds  stdlib only     -- numbers two images must agree on
    geometry    numpy, scipy    -- point/surface measurements
    meshops     + trimesh       -- trimesh surface primitives
"""
