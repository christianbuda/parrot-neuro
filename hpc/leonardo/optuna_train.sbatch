#!/bin/bash
###############################################################################
# Optuna hyperparameter-search trial for CINECA LEONARDO -- ONE trial, ALL of
# $OPTUNA_SUBJECTS, fully offline (compute nodes have no internet -- see
# README.md).
#
# Unlike hpc/leonardo/sweep_train.sbatch (the wandb-agent-dispatched sweep
# this replaces -- see README.md's "Hyperparameter sweep" section for the
# full rationale), this needs NO login-node orchestrator process: the search
# state lives in one shared Optuna JournalStorage file on NFS
# ($OPTUNA_STORAGE), safe for many concurrent processes to read/write.
# Submit this DIRECTLY as a SLURM job ARRAY (see submit_optuna_sweep.sh) --
# one array task = one trial. SLURM's own scheduler handles concurrency and
# throttling; nothing needs to stay alive on a login node for hours.
#
# A THIN, resource-agnostic runner (mirrors optim_cohort.sbatch /
# sweep_train.sbatch): resources (partition/qos/gres/cpus/time/mem) are
# supplied by whoever calls `sbatch` (submit_optuna_sweep.sh), not hardcoded
# here as #SBATCH. This file should almost never need editing.
###############################################################################

#SBATCH --job-name=parrot-optuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=parrot-optuna-%x-%A_%a.out
#SBATCH --error=parrot-optuna-%x-%A_%a.err

set -euo pipefail

for _c in "${PARROT_CONFIG:-}" \
          "${SLURM_SUBMIT_DIR:-}/hpc/leonardo/config.local.sh" \
          "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[config] loaded $_c"; break; }
done

REPO="${REPO:-$HOME/parrot-neuro}"
: "${WORKDIR:?set WORKDIR in config.local.sh (e.g. /leonardo_work/<ACCT>)}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
OPTUNA_OUTPUT_ROOT="${OPTUNA_OUTPUT_ROOT:-$WORKDIR/parrot/eeg_bold_fit_optuna_res}"
: "${OPTUNA_SUBJECTS:?set OPTUNA_SUBJECTS in config.local.sh, e.g. 010002,010003,010004,010005,010006}"
: "${OPTUNA_STUDY_NAME:?set via submit_optuna_sweep.sh (or OPTUNA_STUDY_NAME in config.local.sh)}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-$WORKDIR/parrot/optuna/$OPTUNA_STUDY_NAME.log}"

# --- non-swept fit hyperparameters (fixed across the whole search) ----------
OPTUNA_ATLAS="${OPTUNA_ATLAS:-1000}"
OPTUNA_SPACING="${OPTUNA_SPACING:-2.0}"
OPTUNA_LEADFIELD_LABEL="${OPTUNA_LEADFIELD_LABEL:-duneuroCGAL}"
OPTUNA_OPTIMIZE="${OPTUNA_OPTIMIZE:-both}"
# bold_model: "hrf" (default, linear HRF-kernel convolution) | "balloon"
# (Friston/Deco Balloon-Windkessel hemodynamic ODE) -- see eeg_bold_fit_cli.py
# --bold-model.
OPTUNA_BOLD_MODEL="${OPTUNA_BOLD_MODEL:-hrf}"
# schedule: "alternating" (default, original interleaved fit) | "phased"
# (splits OPTUNA_NUM_EPOCHS in half: BOLD-only then EEG-only) | "joint" -- see
# eeg_bold_fit_cli.py --schedule. joint_*_weight only matters for "joint" --
# harmless to always pass it regardless of which schedule is active.
OPTUNA_SCHEDULE="${OPTUNA_SCHEDULE:-alternating}"
OPTUNA_JOINT_EEG_WEIGHT="${OPTUNA_JOINT_EEG_WEIGHT:-1e5}"
OPTUNA_JOINT_BOLD_WEIGHT="${OPTUNA_JOINT_BOLD_WEIGHT:-1.0}"
OPTUNA_NUM_EPOCHS="${OPTUNA_NUM_EPOCHS:-300}"
OPTUNA_BOLD_EVERY="${OPTUNA_BOLD_EVERY:-2}"
OPTUNA_EEG_TASK="${OPTUNA_EEG_TASK:-eyesclosed}"
OPTUNA_FMRI_TASK="${OPTUNA_FMRI_TASK:-rest}"
# Same GPU-memory-OOM fixes as optim_cohort.sbatch/sweep_train.sbatch (atlas=
# 1000 + a long BOLD horizon reliably OOMs without both of these).
OPTUNA_T1_WARMUP="${OPTUNA_T1_WARMUP:-30000}"
OPTUNA_SOLVER_BLOCK_SIZE="${OPTUNA_SOLVER_BLOCK_SIZE:-1400}"
OPTUNA_SKIP_DIAGNOSTICS="${OPTUNA_SKIP_DIAGNOSTICS:-0}"

# --- pixi ---------------------------------------------------------------
PIXI="$(command -v pixi || true)"
[ -z "$PIXI" ] && [ -x "$HOME/.pixi/bin/pixi" ] && PIXI="$HOME/.pixi/bin/pixi"
[ -n "$PIXI" ] || { echo "ERROR: pixi not found (on PATH or at \$HOME/.pixi/bin) -- run setup_optim_env.sh"; exit 1; }
[ -d "$REPO/.pixi" ] || { echo "ERROR: $REPO/.pixi missing -- run setup_optim_env.sh on a login node"; exit 1; }

# One JAX cache / wandb-offline-dir subdir per array task -- many concurrent
# trials hitting ONE shared cache/dir risks lock contention or two tasks
# writing into the same offline-run directory (same rationale as
# sweep_train.sbatch's per-run_id cache, optim_cohort.sbatch's per-subject
# one).
TASK_TAG="${SLURM_ARRAY_TASK_ID:-$SLURM_JOB_ID}"
export PARROT_JAX_CACHE_DIR="$WORKDIR/parrot/.jax_cache/optuna-$OPTUNA_STUDY_NAME-$TASK_TAG"
mkdir -p "$PARROT_JAX_CACHE_DIR" "$OPTUNA_OUTPUT_ROOT" "$(dirname "$OPTUNA_STORAGE")"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Offline: this compute node has no internet (see README.md) -- wandb here is
# a local logger only, synced in bulk later from a login node
# (sync_orphaned_runs.sh). Optuna's study.ask()/tell() (over OPTUNA_STORAGE,
# a plain NFS file) is what coordinates the search -- it needs no network at
# all, online or offline.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$WORKDIR/parrot/wandb_offline/optuna-$OPTUNA_STUDY_NAME-$TASK_TAG}"
mkdir -p "$WANDB_DIR"

cd "$WORKDIR" || { echo "ERROR: cannot cd to WORKDIR=$WORKDIR"; exit 1; }

echo "=== Parrot Optuna trial: job ${SLURM_JOB_ID:-?} (array ${SLURM_ARRAY_JOB_ID:--}_${SLURM_ARRAY_TASK_ID:--}) on $(hostname), $(date) ==="
echo "study=$OPTUNA_STUDY_NAME storage=$OPTUNA_STORAGE subjects=$OPTUNA_SUBJECTS atlas=$OPTUNA_ATLAS optimize=$OPTUNA_OPTIMIZE bold_model=$OPTUNA_BOLD_MODEL schedule=$OPTUNA_SCHEDULE epochs=$OPTUNA_NUM_EPOCHS"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible -- this stage always needs one)"

ARGS=(
    --bids-root "$BIDS" --subjects "$OPTUNA_SUBJECTS" --output-root "$OPTUNA_OUTPUT_ROOT"
    --atlas "$OPTUNA_ATLAS" --spacing "$OPTUNA_SPACING" --leadfield-label "$OPTUNA_LEADFIELD_LABEL"
    --optimize "$OPTUNA_OPTIMIZE" --bold-model "$OPTUNA_BOLD_MODEL" --schedule "$OPTUNA_SCHEDULE"
    --joint-eeg-weight "$OPTUNA_JOINT_EEG_WEIGHT" --joint-bold-weight "$OPTUNA_JOINT_BOLD_WEIGHT"
    --num-epochs "$OPTUNA_NUM_EPOCHS" --bold-every "$OPTUNA_BOLD_EVERY"
    --eeg-task "$OPTUNA_EEG_TASK" --fmri-task "$OPTUNA_FMRI_TASK"
    --solver-block-size "$OPTUNA_SOLVER_BLOCK_SIZE" --t1-warmup "$OPTUNA_T1_WARMUP"
    --optuna-study-name "$OPTUNA_STUDY_NAME" --optuna-storage "$OPTUNA_STORAGE"
)
[ "$OPTUNA_SKIP_DIAGNOSTICS" = 1 ] && ARGS+=( --skip-diagnostics )
# Parallel (whole-node) mode -- see submit_optuna_sweep.sh's OPTUNA_GPUS
# handling for the matching --gres=gpu:<count>/QoS/walltime it computes to
# go with this (same mechanism as sweep_dispatch.sh's SWEEP_GPUS, just
# computed once at array-submission time instead of per-trial, since there's
# no per-trial dispatcher any more).
[ -n "${OPTUNA_GPUS:-}" ] && ARGS+=( --gpus "$OPTUNA_GPUS" )

SECONDS=0
set +e
# -u: unbuffered stdout -- without it, prints can sit invisible in the buffer
# for the whole job once stdout isn't a TTY (redirected to the .out file).
(cd "$REPO" && "$PIXI" run python -u examples/eeg_bold_fit_optuna.py "${ARGS[@]}")
rc=$?
set -e
echo "=== Optuna trial finished in $((SECONDS/60)) min, rc=$rc ($(date)) ==="
exit $rc
