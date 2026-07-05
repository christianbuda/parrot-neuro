#!/bin/bash
###############################################################################
# Pre-populate a TemplateFlow cache LOCALLY, to rsync up to the cluster's
# persistent cache (the orchestrator binds <output_dir>/.templateflow ->
# /templateflow with TEMPLATEFLOW_HOME for QSIPrep/QSIRecon; see
# bin/run_reconstruction.sh:1254-1255).
#
# WHY: QSIPrep/QSIRecon fetch standard templates from templateflow.s3.amazonaws.com
# at runtime (e.g. tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz). LEONARDO's COMPUTE
# NODES have no egress to S3 ("[Errno 101] Network is unreachable"), so the fetch
# can never succeed on the node -- we ship a pre-warmed cache. (3rd such cache,
# alongside HippUnfold/OSF and the MNI/artifact template.)
#
# TWO MODES:
#  (default) COPY an existing local cache. A real local QSIPrep run leaves a
#            minimal, exact cache at <its output_dir>/.templateflow -- TemplateFlow
#            pulls only the files it touches, so this is precisely what the cluster
#            needs (~17 MB). This is the robust path: ship what actually worked.
#  (BUILD=1) FETCH a template superset via the QSIPrep image's bundled templateflow
#            (for a machine with no prior local run). Larger (whole templates), but
#            self-contained. Needs docker + internet.
#
# Run on the WORKSTATION (needs internet; BUILD mode also needs docker).
#
#   # copy your existing cache and print the rsync command:
#   bash hpc/leonardo/prewarm_templateflow.sh
#   # copy a specific source cache:
#   SRC=/path/to/derivatives/.templateflow bash hpc/leonardo/prewarm_templateflow.sh
#   # copy AND ship up in one go:
#   DEST=<USER>@data.leonardo.cineca.it:/leonardo_work/<ACCT>/parrot/bids/derivatives/.templateflow/ \
#       bash hpc/leonardo/prewarm_templateflow.sh
#   # no local cache -> build a fresh one from the image:
#   BUILD=1 bash hpc/leonardo/prewarm_templateflow.sh
###############################################################################
set -uo pipefail

CACHE="${1:-$PWD/templateflow_cache}"     # staging dir we assemble/ship
SRC="${SRC:-}"                            # existing local .templateflow to copy (auto-detected if empty)
BUILD="${BUILD:-}"                        # non-empty => fetch via the image instead of copying SRC
IMG="${QSIPREP_IMAGE:-pennlinc/qsiprep:latest}"
# Template superset for BUILD mode: QSIPrep outputs to MNI152NLin2009cAsym and skull-strips via
# MNI152NLin6Asym/OASIS30ANTs; QSIRecon surface work uses fsLR/fsaverage. Override with TEMPLATES=.
TEMPLATES="${TEMPLATES:-MNI152NLin2009cAsym MNI152NLin6Asym OASIS30ANTs fsLR fsaverage}"
DEST="${DEST:-}"                          # optional rsync target; empty = just print the command

# The one file the QSIPrep failure named -- used to sanity-check the cache is real.
CANARY="tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"

mkdir -p "$CACHE"

if [ -n "$BUILD" ]; then
  # --- BUILD: fetch a template superset via the image's templateflow ----------
  command -v docker >/dev/null || { echo "ERROR: BUILD mode needs docker (this runs on your workstation)."; exit 1; }
  echo "== BUILD: fetching {$TEMPLATES} via $IMG =="
  docker run --rm -e TEMPLATEFLOW_HOME=/tf -v "$CACHE:/tf" --entrypoint bash "$IMG" -c '
    python - '"$TEMPLATES"' <<PY
import sys, templateflow.api as tf
for t in sys.argv[1:]:
    print("  fetch  tpl-%s" % t, flush=True)
    tf.get(t)   # whole template (over-fetches vs a real run, but self-contained)
print("done")
PY' || { echo "ERROR: BUILD fetch failed."; exit 1; }
else
  # --- COPY: ship an existing local cache -------------------------------------
  if [ -z "$SRC" ]; then
    echo "== auto-detecting a populated local .templateflow =="
    # Pick the largest non-empty candidate from known local derivative trees.
    best=""; best_n=0
    while IFS= read -r d; do
      [ -d "$d" ] || continue
      n=$(find "$d" -type f 2>/dev/null | wc -l)
      [ "$n" -gt "$best_n" ] && { best="$d"; best_n="$n"; }
    done < <(find /srv/nfs-data/sisko/christian -maxdepth 4 -type d -name .templateflow 2>/dev/null)
    SRC="$best"
    [ -n "$SRC" ] && echo "  found: $SRC  ($best_n files)"
  fi
  if [ -z "$SRC" ] || [ ! -d "$SRC" ] || [ -z "$(ls -A "$SRC" 2>/dev/null)" ]; then
    echo "ERROR: no populated local .templateflow found."
    echo "       Point SRC at one from a prior local QSIPrep run (e.g. <output_dir>/.templateflow),"
    echo "       or re-run with BUILD=1 to fetch a fresh superset from the image."
    exit 1
  fi
  echo "== copy $SRC -> $CACHE =="
  rsync -a "$SRC"/ "$CACHE"/     # trailing slashes = merge contents (avoids nesting)
fi

# --- sanity check ------------------------------------------------------------
echo
echo "cache at: $CACHE"
du -sh "$CACHE" 2>/dev/null | awk '{print "  size:  "$1}'
echo "  files: $(find "$CACHE" -type f | wc -l)"
if [ -f "$CACHE/$CANARY" ]; then
  echo "  OK:    canary present ($CANARY)"
else
  echo "  WARN:  canary MISSING ($CANARY) -- QSIPrep will still fail. Check SRC / BUILD templates."
fi

# --- ship it to the cluster --------------------------------------------------
# Destination = the cluster's <output_dir>/.templateflow. Trailing slash both sides = merge.
if [ -n "$DEST" ]; then
  echo "== rsync -> $DEST =="
  rsync -avP "$CACHE"/ "$DEST"
else
  echo "To ship it up (dest = the cluster's <output_dir>/.templateflow):"
  echo "  rsync -avP $CACHE/ <USER>@data.leonardo.cineca.it:/leonardo_work/<ACCT>/parrot/bids/derivatives/.templateflow/"
fi
