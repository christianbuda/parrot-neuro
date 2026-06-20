#!/bin/bash
#
# Build per-subject structural connectomes from the QSIRecon tractogram and the
# per-subject Parrot connectivity atlases, using MRtrix3 `tck2connectome`.
#
# This is run INSIDE the QSIRecon image (not a Parrot image) so the connectome
# is built with the *same* MRtrix3 that generated the tracks (zero version
# drift). It operates entirely on the mounted /derivatives tree.
#
#   Usage (inside the container):  make_connectomes.sh <subject_id>
#
# Canonical Parrot connectome definition (agreed; templates regenerated to match):
#   - assignment: -assignment_radial_search 2  (explicit, reproducible; safe for
#                 fine Schaefer-1000 parcels)
#   - weights_{N}            : raw SIFT2-weighted streamline count (TVB coupling)
#   - weights_invnodevol_{N} : SIFT2 + -scale_invnodevol      (graph-theory variant)
#   - distances_{N}          : mean streamline length          (TVB conduction delays)
#   - assignments_{N}        : per-streamline node assignment  (re-derive later w/o re-tracking)
# All matrices symmetric, zero-diagonal. Node 0 (Unknown) is dropped by default.

set -euo pipefail

SUB="$1"
CONN="/derivatives/connectivity/sub-${SUB}"
DWI_DIR="/derivatives/qsirecon/sub-${SUB}/dwi"

TCK_GZ=$(ls "$DWI_DIR"/*streamlines.tck.gz | head -n 1)
WEIGHTS=$(ls "$DWI_DIR"/*streamlineweights.csv | head -n 1)
echo "Tractogram   : $TCK_GZ"
echo "SIFT2 weights: $WEIGHTS"

# QSIRecon stores the tractogram gzipped, but MRtrix `tck2connectome` cannot
# read a .tck.gz: it locates the binary track data via a byte-offset recorded in
# the .tck header and then seeks to it, which a gzip stream can't satisfy (hence
# "not a valid track file"). So decompress ONCE to a temporary plain .tck that
# every resolution/metric below reuses, and remove it on exit -- the file is
# large (~15 GB; the gz barely compresses high-entropy streamline coordinates).
# It lives under $CONN (already created by the orchestrator) on the roomy
# derivatives volume, not in a possibly RAM-backed container /tmp.
TCK="$CONN/.streamlines_tmp.tck"
trap 'rm -f "$TCK"' EXIT INT TERM
echo "Decompressing tractogram -> $TCK ..."
gunzip -c "$TCK_GZ" > "$TCK"

for N in 100 200 300 400 500 600 700 800 900 1000; do
    NODES="$CONN/atlas${N}_connectivity.nii.gz"
    echo "=== resolution ${N} ==="

    # Raw SIFT2-weighted streamline count (biophysical coupling for TVB).
    # -out_assignments is saved here once per resolution (it depends only on the
    # tracks, atlas and assignment radius, not on the edge scaling below).
    tck2connectome "$TCK" "$NODES" "$CONN/weights_${N}.txt" \
        -tck_weights_in "$WEIGHTS" \
        -assignment_radial_search 2 \
        -symmetric -zero_diagonal \
        -out_assignments "$CONN/assignments_${N}.txt" \
        -force

    # SIFT2-weighted, node-volume normalised (graph-theory variant).
    tck2connectome "$TCK" "$NODES" "$CONN/weights_invnodevol_${N}.txt" \
        -tck_weights_in "$WEIGHTS" \
        -assignment_radial_search 2 \
        -symmetric -zero_diagonal -scale_invnodevol \
        -force

    # Mean streamline length per edge (TVB conduction delays; not Euclidean).
    tck2connectome "$TCK" "$NODES" "$CONN/distances_${N}.txt" \
        -tck_weights_in "$WEIGHTS" \
        -assignment_radial_search 2 \
        -symmetric -zero_diagonal -scale_length -stat_edge mean \
        -force
done

echo "All connectomes generated for sub-${SUB}."
