#!/bin/bash
###############################################################################
# Populate a TemplateFlow cache IN PLACE at <output_dir>/.templateflow -- the
# path the orchestrator binds to /templateflow (TEMPLATEFLOW_HOME) for QSIPrep/
# QSIRecon (bin/run_reconstruction.sh:1254-1255). No rsync: run it wherever the
# cache should live (the LOGIN node writes straight onto the work filesystem).
#
# WHY: QSIPrep/QSIRecon fetch standard templates from templateflow.s3.amazonaws.com
# at runtime (e.g. tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz). LEONARDO's COMPUTE
# NODES have no S3 egress ("[Errno 101] Network is unreachable"), so the fetch dies
# on the node. LOGIN nodes do have egress -- warm the cache there. (3rd such cache,
# alongside HippUnfold/OSF and the MNI/artifact template.)
#
# TWO MODES:
#  (default) BUILD -- fetch a template superset via the QSIPrep container's bundled
#            templateflow. RUNTIME=apptainer uses the .sif in SIF_DIR (login node,
#            no docker); RUNTIME=docker uses the Hub/local image (workstation).
#  (SRC=...) COPY -- seed from an existing local cache (e.g. a prior workstation
#            QSIPrep run's minimal ~17 MB set); no container/egress needed.
#
#   # on a LOGIN node (has egress + singularity + the .sif cache):
#   RUNTIME=apptainer SIF_DIR=$WORKDIR/parrot/parrot_sif \
#     bash hpc/leonardo/prewarm_templateflow.sh $WORKDIR/parrot/bids/derivatives/.templateflow
#   # on the workstation, seed from a prior local run:
#   SRC=/srv/.../derivatives_e2e/.templateflow bash hpc/leonardo/prewarm_templateflow.sh ./tf_cache
###############################################################################
set -uo pipefail

CACHE="${1:-$PWD/templateflow_cache}"     # the cache dir to populate (point at <output_dir>/.templateflow on the cluster)
SRC="${SRC:-}"                            # existing cache to COPY from; empty => BUILD via the container
RUNTIME="${RUNTIME:-docker}"              # docker (workstation) | apptainer (login node, uses SIF_DIR)
SIF_DIR="${SIF_DIR:-}"                    # .sif cache dir (apptainer only)
IMG="${QSIPREP_IMAGE:-pennlinc/qsiprep:latest}"   # docker image ref (docker runtime)
# Template superset for BUILD: QSIPrep outputs to MNI152NLin2009cAsym and skull-strips via
# MNI152NLin6Asym/OASIS30ANTs; QSIRecon surface work uses fsLR/fsaverage. Override with TEMPLATES=.
TEMPLATES="${TEMPLATES:-MNI152NLin2009cAsym MNI152NLin6Asym OASIS30ANTs fsLR fsaverage}"

# The one file the QSIPrep failure named -- used to sanity-check the cache is real.
CANARY="tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"

mkdir -p "$CACHE"

# Run a bash -c command inside the QSIPrep container with the cache bound at /tf and
# TEMPLATEFLOW_HOME=/tf, under whichever runtime is selected.
run_in_qsiprep() {   # $1 = bash -c command string
  case "$RUNTIME" in
    docker)
      command -v docker >/dev/null || { echo "ERROR: RUNTIME=docker but docker not found."; return 1; }
      docker run --rm -e TEMPLATEFLOW_HOME=/tf -v "$CACHE:/tf" --entrypoint bash "$IMG" -c "$1" ;;
    apptainer)
      local app sif
      app="$(command -v apptainer || command -v singularity || true)"
      [ -n "$app" ] || { echo "ERROR: RUNTIME=apptainer but no apptainer/singularity on PATH."; return 1; }
      sif="$SIF_DIR/qsiprep_latest.sif"
      [ -f "$sif" ] || { echo "ERROR: $sif not found (set SIF_DIR to your .sif cache)."; return 1; }
      "$app" exec --env TEMPLATEFLOW_HOME=/tf --bind "$CACHE:/tf" "$sif" bash -c "$1" ;;
    *) echo "ERROR: RUNTIME must be 'docker' or 'apptainer' (got '$RUNTIME')."; return 1 ;;
  esac
}

if [ -n "$SRC" ]; then
  # --- COPY: seed from an existing local cache (no container / no egress) ------
  if [ ! -d "$SRC" ] || [ -z "$(ls -A "$SRC" 2>/dev/null)" ]; then
    echo "ERROR: SRC is not a populated cache: $SRC"; exit 1
  fi
  echo "== copy $SRC -> $CACHE =="
  cp -R "$SRC"/. "$CACHE"/       # merge contents into CACHE (no -a: NFS rejects perm-preserve)
else
  # --- BUILD: fetch a template superset via the container's templateflow -------
  echo "== BUILD ($RUNTIME): fetching {$TEMPLATES} into $CACHE =="
  # Pixi image: python lives in the qsiprep env; a bare exec shell may not have it on
  # PATH (activation normally happens in the entrypoint we bypass). Prepend the env bin
  # (path taken from the QSIPrep traceback) so this works under docker exec / singularity exec.
  run_in_qsiprep '
    export PATH=/app/.pixi/envs/qsiprep/bin:$PATH
    python - '"$TEMPLATES"' <<PY
import sys, templateflow.api as tf
for t in sys.argv[1:]:
    print("  fetch  tpl-%s" % t, flush=True)
    tf.get(t)   # whole template (over-fetches vs a real run, but self-contained)
print("done")
PY' || { echo "ERROR: BUILD fetch failed."; exit 1; }
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
  exit 1
fi
