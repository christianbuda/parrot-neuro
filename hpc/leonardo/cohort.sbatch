#!/bin/bash
###############################################################################
# Parrot cohort chunk runner for CINECA LEONARDO.
#
# A THIN, resource-agnostic runner: it reconstructs ONE subject and runs only
# the pipeline stages it is told to (`$PARROT_STAGES`), on whatever GPU policy
# it is told (`$PARROT_GPUS`). It does NOT know about chunks, partitions, cores
# or walltime -- all of that (the A1/A2/B/C/D chunk map + the SLURM resource
# matrix + the dependency chain) lives in submit_cohort.sh, the single source
# of truth. This file should almost never need editing.
#
# It is submitted three ways, all by submit_cohort.sh:
#   - as a job ARRAY over the cohort  -> subject = line ($SLURM_ARRAY_TASK_ID+1)
#                                        of $PARROT_SUBJECTS_FILE
#   - as a single SMOKE job           -> subject = $PARROT_SUBJECT
# Never `sbatch` this directly; you'd get no --account, stages or resources.
#
# Per-stage idempotency (log-guards) + `--stages` mean a chunk only executes its
# own stages; re-running a completed chunk is a cheap no-op. That is what makes
# the dependency chain safe to resubmit.
###############################################################################

# Resource #SBATCH directives are intentionally ABSENT here: they are parsed
# before the shell runs, so they cannot be per-chunk. submit_cohort.sh passes
# them all as `sbatch` CLI options (which override any #SBATCH anyway). We keep
# only the invariant plumbing here.
#SBATCH --job-name=parrot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=parrot-%x-%A_%a.out
#SBATCH --error=parrot-%x-%A_%a.err

set -euo pipefail

# --- personal config (gitignored): ACCT/WORKDIR/REPO/BIDS/OUTPUT_DIR/SIF ------
# sbatch spools this script, so $BASH_SOURCE is not the repo copy -- resolve via
# $SLURM_SUBMIT_DIR (where you ran sbatch) or the default clone at $HOME.
for _c in "${PARROT_CONFIG:-}" \
          "${SLURM_SUBMIT_DIR:-}/hpc/leonardo/config.local.sh" \
          "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[config] loaded $_c"; break; }
done

REPO="${REPO:-$HOME/parrot-neuro}"
: "${WORKDIR:?set WORKDIR in config.local.sh (e.g. /leonardo_work/<ACCT>)}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
OUTPUT_DIR="${OUTPUT_DIR:-$BIDS/derivatives}"
SIF="${SIF:-$WORKDIR/parrot/parrot_sif}"

# --- what to run (set by submit_cohort.sh via --export) -----------------------
: "${PARROT_STAGES:?PARROT_STAGES not set -- submit via submit_cohort.sh, not raw sbatch}"
PARROT_GPUS="${PARROT_GPUS:-none}"

# --- which subject: array element, or an explicit smoke subject ---------------
if [ -n "${PARROT_SUBJECT:-}" ]; then
    SUBJECT="$PARROT_SUBJECT"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    : "${PARROT_SUBJECTS_FILE:?PARROT_SUBJECTS_FILE not set for an array run}"
    # +1: SLURM_ARRAY_TASK_ID is 0-based, sed lines are 1-based.
    SUBJECT="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARROT_SUBJECTS_FILE")"
    [ -n "$SUBJECT" ] || { echo "ERROR: no subject at line $((SLURM_ARRAY_TASK_ID + 1)) of $PARROT_SUBJECTS_FILE"; exit 1; }
else
    echo "ERROR: neither PARROT_SUBJECT nor SLURM_ARRAY_TASK_ID set."; exit 1
fi

CPUS="${SLURM_CPUS_PER_TASK:-8}"

# --- modules -----------------------------------------------------------------
# CUDA for --nv (harmless on the CPU dcgp chunks). Singularity is a SYSTEM
# command on LEONARDO (/usr/bin/singularity), so it survives `module purge`.
module purge
module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
command -v apptainer >/dev/null || command -v singularity >/dev/null || {
    echo "ERROR: no apptainer/singularity on PATH (expected /usr/bin/singularity)."; exit 1; }

mkdir -p "$SIF" "$OUTPUT_DIR"

# Anchor the process CWD on the robust work FS. Apptainer lstat()s the current directory at
# container creation; SLURM starts batch scripts in the submit dir (the quota'd $HOME on
# LEONARDO's LDAP/SSSD-backed home FS), which intermittently denies that lstat during
# identity/home-FS blips -> "FATAL: could not obtain current directory path". Anchoring on
# $WORKDIR (/leonardo_work) keeps container startup off the home FS entirely -- the `--home`
# override already does this for the container HOME; this covers the host-side CWD too.
cd "$WORKDIR" || { echo "ERROR: cannot cd to WORKDIR=$WORKDIR"; exit 1; }

echo "=== Parrot cohort chunk: job ${SLURM_JOB_ID:-?} (array ${SLURM_ARRAY_JOB_ID:--}_${SLURM_ARRAY_TASK_ID:--}) on $(hostname), $(date) ==="
echo "subject=$SUBJECT  stages=$PARROT_STAGES  gpus=$PARROT_GPUS  cpus=$CPUS"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible -- expected for the CPU/dcgp chunks)"

# --- the run -----------------------------------------------------------------
# Don't let `set -e` abort on a stage failure: capture rc, still print the
# summary, then exit with rc so `aftercorr` correctly withholds the dependents.
SECONDS=0
set +e
"$REPO/bin/run_reconstruction.sh" \
    "$BIDS" "$OUTPUT_DIR" participant \
    --participant-label "$SUBJECT" \
    --runtime apptainer --sif-dir "$SIF" \
    --threads "$CPUS" \
    --stages "$PARROT_STAGES" \
    --gpus "$PARROT_GPUS"
rc=$?
set -e
echo "=== chunk finished in $((SECONDS/60)) min, rc=$rc ($(date)) ==="

# --- per-stage wall-clock (the numbers we actually care about) ---------------
LOGDIR="$OUTPUT_DIR/logs/sub-$SUBJECT"
echo "--- per-stage wall-clock (this + prior runs on sub-$SUBJECT) ---"
grep -E "completed in [0-9]+ minutes" "$LOGDIR/parrot-reconstruction_log.txt" 2>/dev/null \
    || echo "(see $LOGDIR/parrot-reconstruction_log.txt)"

exit $rc
