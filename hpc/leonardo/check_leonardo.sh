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
# Exits non-zero if any [FAIL] is printed.
###############################################################################
set -uo pipefail   # NOT -e: we want every check to run and report, not abort early

ACCT="${ACCT:-<YOUR_ACCOUNT>}"            # your SLURM account (saldo -b)
WORKDIR="${WORKDIR:-/leonardo_work/$ACCT}"
REPO="${REPO:-$HOME/parrot-neuro}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
SUBJECT="${SUBJECT:-010002}"
SIF="${SIF:-$WORKDIR/parrot_sif}"

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
  exit 1
fi
