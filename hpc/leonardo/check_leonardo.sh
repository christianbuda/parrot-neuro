#!/bin/bash
###############################################################################
# Preflight check for the LEONARDO Parrot pilot. Run on a LOGIN node BEFORE you
# `sbatch pilot.sbatch`. It catches the "queued for hours, then died in 10s on a
# typo / missing file" failure mode by verifying everything the job assumes:
#   - a container runtime is on PATH (and new enough for --env/--home)
#   - the .sif cache is complete (all images pre-pulled)
#   - the BIDS dataset, FreeSurfer license, and target subject exist
#   - the repo (orchestrator) is present and executable
#   - the work area is writable
#   - the account is known to saldo
#
# Configure via env vars (or edit the defaults) -- KEEP THESE IN SYNC with the
# four `# <<EDIT>>` values in pilot.sbatch:
#   ACCT=<YOUR_ACCOUNT> SUBJECT=010002 bash hpc/leonardo/check_leonardo.sh
#
# Pass --fix to POPULATE a missing HippUnfold/TemplateFlow cache in place by running
# the prewarm scripts (they run on the LOGIN node via singularity + the .sif cache).
# Without --fix the preflight is read-only: it reports the gap and prints the command.
#
# Exits non-zero if any [FAIL] is printed.
###############################################################################
set -uo pipefail   # NOT -e: we want every check to run and report, not abort early

PREWARM=0
for a in "$@"; do case "$a" in --fix) PREWARM=1 ;; esac; done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # for locating the prewarm_*.sh siblings

# --- personal config (gitignored): set ACCT/WORKDIR/REPO/SUBJECT/SIF once -----
# First match wins; call-time env vars still override it (config uses ${VAR:-...}).
for _c in "${PARROT_CONFIG:-}" "$HERE/config.local.sh" \
          "${SLURM_SUBMIT_DIR:-}/hpc/leonardo/config.local.sh" \
          "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[config] loaded $_c"; break; }
done

ACCT="${ACCT:-<YOUR_ACCOUNT>}"            # SLURM BILLING account (saldo -b); the -A / --account value
# Storage area, INDEPENDENT of the billing account (you may bill one allocation but store on
# another). Default to $WORK for convenience -- but $WORK is AMBIGUOUS if you belong to several
# accounts (it may resolve to the wrong allocation), so override WORKDIR explicitly in that case.
WORKDIR="${WORKDIR:-${WORK:-}}"
: "${WORKDIR:?set WORKDIR (or export \$WORK) -- e.g. /leonardo_work/<ACCT>}"
REPO="${REPO:-$HOME/parrot-neuro}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
SUBJECT="${SUBJECT:-010002}"
SIF="${SIF:-$WORKDIR/parrot/parrot_sif}"
OUTPUT_DIR="${OUTPUT_DIR:-$BIDS/derivatives}"       # matches pilot.sbatch's OUT
HU_CACHE="${HU_CACHE:-$OUTPUT_DIR/.hippunfold_cache}" # matches the orchestrator's HIPPUNFOLD_CACHE_HOST default
TF_CACHE="${TF_CACHE:-$OUTPUT_DIR/.templateflow}"     # matches the orchestrator's TEMPLATEFLOW_DIR

# Keep this list in sync with prepull_sifs.sh / bin/images.sh.
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

fail=0
ok()   { printf '  [ OK ]  %s\n' "$1"; }
warn() { printf '  [WARN]  %s\n' "$1"; }
bad()  { printf '  [FAIL]  %s\n' "$1"; fail=1; }

echo "== container runtime =="
APP="$(command -v apptainer || command -v singularity || true)"
if [ -z "$APP" ]; then
  bad "no apptainer/singularity on PATH (expected /usr/bin/singularity)"
else
  ver="$("$APP" --version 2>/dev/null | awk '{print $NF}')"
  case "$APP" in
    *apptainer) ok "apptainer $ver ($APP)" ;;
    *)  # singularity: need >=3.6 for --env and --home host:ctr (used by container_exec)
        major="${ver%%.*}"; rest="${ver#*.}"; minor="${rest%%.*}"
        if [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]] && \
           { [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 6 ]; }; }; then
          ok "singularity $ver ($APP) -- supports --env / --home host:ctr"
        else
          warn "singularity $ver ($APP) -- verify it supports --env / --home host:ctr (need CE >= 3.6)"
        fi ;;
  esac
fi

echo "== .sif cache: $SIF =="
if [ ! -d "$SIF" ]; then
  bad "sif dir missing: $SIF  (run prepull_sifs.sh)"
else
  for img in "${IMAGES[@]}"; do
    base="${img##*/}"; base="${base//:/_}"
    if [ -f "$SIF/$base.sif" ]; then ok "$base.sif"
    elif [ -d "$SIF/$base" ]; then ok "$base/ (sandbox)"
    else bad "missing $base.sif  (run prepull_sifs.sh, or build_sif_fallback.sh + build_sif.sbatch)"; fi
  done
fi

echo "== BIDS dataset: $BIDS =="
[ -d "$BIDS" ] && ok "dataset dir present" || bad "BIDS dir missing: $BIDS"
[ -f "$BIDS/license.txt" ] && ok "FreeSurfer license.txt at root" || bad "license.txt missing at BIDS root (every FreeSurfer/QSIPrep stage needs it)"
[ -f "$BIDS/participants.tsv" ] && ok "participants.tsv present" || bad "participants.tsv missing"
[ -f "$BIDS/dataset_description.json" ] && ok "dataset_description.json present" || warn "dataset_description.json missing (BIDS indexer may complain)"
[ -d "$BIDS/sub-$SUBJECT" ] && ok "subject sub-$SUBJECT present" || bad "subject dir missing: $BIDS/sub-$SUBJECT"
[ -d "$BIDS/sub-$SUBJECT/anat" ] && ok "sub-$SUBJECT/anat present" || bad "sub-$SUBJECT/anat missing (T1 required)"
if ls "$BIDS/sub-$SUBJECT/dwi/"*.nii* >/dev/null 2>&1; then
  warn "sub-$SUBJECT/dwi present -> QSIPrep + QSIRecon WILL run (adds hours; fine for a DWI pilot)"
else
  ok "no dwi/ -> fast anat-only pilot (recon -> forward -> solvers -> QC)"
fi

# --- runtime-fetch caches (must be prewarmed; compute nodes have no egress) ---
# HippUnfold (OSF) and TemplateFlow (S3) both download at runtime, and LEONARDO compute
# nodes can't reach either. Catch an empty/incomplete cache HERE, not at hour 3. With
# --fix, populate it in place via the prewarm_*.sh siblings (login node: singularity +
# the .sif cache). Runtime for the fix = apptainer when present, else docker.
FIX_RT="$([ -n "${APP:-}" ] && echo apptainer || echo docker)"

check_hu() {   # returns 0 if the HippUnfold cache is complete, 1 otherwise
  local miss=0 r
  for r in atlas/multihist7 template/upenn template/CITI168; do
    if [ -d "$HU_CACHE/$r" ] && [ -n "$(ls -A "$HU_CACHE/$r" 2>/dev/null)" ]; then ok "$r"
    else printf '  [ -- ]  %s (missing)\n' "$r"; miss=1; fi
  done
  [ -n "$(ls -A "$HU_CACHE/model" 2>/dev/null)" ] && ok "model/ present" \
    || warn "model/ empty (Zenodo reachable on-node -> self-heals; ships better prewarmed)"
  return $miss
}

check_tf() {   # returns 0 if the TemplateFlow cache has the canary QSIPrep needs
  local canary="tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
  if [ -f "$TF_CACHE/$canary" ]; then ok "canary tpl-MNI152NLin2009cAsym_res-01_T1w"; return 0
  else printf '  [ -- ]  TemplateFlow empty/incomplete (%s missing)\n' "$canary"; return 1; fi
}

run_cache_check() {   # $1=title $2=check-fn $3=prewarm-script $4=cache-dir $5=egress-hint
  echo "== $1 =="
  "$2" && return
  if [ "$PREWARM" -eq 1 ]; then
    warn "$1 incomplete -> running $3 (RUNTIME=$FIX_RT) ..."
    if RUNTIME="$FIX_RT" SIF_DIR="$SIF" bash "$HERE/$3" "$4"; then
      "$2" && ok "$1 populated" || bad "$1 still incomplete after prewarm"
    else
      bad "$3 failed (see output above)"
    fi
  else
    bad "$1 not prewarmed -- $5. Fix: re-run with --fix on a login node, or 'bash $HERE/$3 $4'"
  fi
}

run_cache_check "HippUnfold cache: $HU_CACHE" check_hu prewarm_hippunfold.sh "$HU_CACHE" \
  "compute nodes can't reach OSF"
run_cache_check "TemplateFlow cache: $TF_CACHE" check_tf prewarm_templateflow.sh "$TF_CACHE" \
  "compute nodes can't reach templateflow S3"

echo "== repo / orchestrator =="
[ -x "$REPO/bin/run_reconstruction.sh" ] && ok "run_reconstruction.sh present + executable" \
  || bad "run_reconstruction.sh missing or not executable in $REPO/bin"

echo "== work area =="
if t="$(mktemp "$WORKDIR/.parrot_check.XXXXXX" 2>/dev/null)"; then
  rm -f "$t"; ok "writable: $WORKDIR"
else
  bad "cannot write to $WORKDIR (wrong account/path, or quota?)"
fi

echo "== account / budget =="
if command -v saldo >/dev/null 2>&1; then
  if saldo -b 2>/dev/null | grep -q "$ACCT"; then ok "$ACCT found in 'saldo -b'"; else warn "$ACCT not in 'saldo -b' -- check the account name"; fi
else
  warn "saldo not on PATH -- skipping budget check"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "PREFLIGHT PASSED -- safe to: sbatch hpc/leonardo/pilot.sbatch"
else
  echo "PREFLIGHT FAILED -- fix the [FAIL] items above before submitting."
  echo "  (cache gaps: re-run with --fix on a login node to prewarm them in place.)"
  exit 1
fi
