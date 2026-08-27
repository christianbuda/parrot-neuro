#!/bin/bash
###############################################################################
# W&B sweep trial trainer for CINECA LEONARDO -- ONE trial, ALL of
# $SWEEP_SUBJECTS, offline (compute nodes have no internet -- see README.md).
#
# A THIN, resource-agnostic runner (mirrors optim_cohort.sbatch): resources
# (partition/qos/gres/cpus/time/mem) are supplied by whoever calls `sbatch`,
# i.e. sweep_dispatch.sh, not hardcoded here as #SBATCH. This file should
# almost never need editing.
#
# Only ever submitted (via `sbatch --wait ... --export=ALL,WANDB_MODE=offline,...`)
# by sweep_dispatch.sh, which runs under `wandb agent` on the LOGIN node --
# never `sbatch` this directly; you'd get no --account/resources/hyperparameters.
###############################################################################

#SBATCH --job-name=parrot-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=parrot-sweep-%x-%j.out
#SBATCH --error=parrot-sweep-%x-%j.err

set -euo pipefail

for _c in "${PARROT_CONFIG:-}" \
          "${SLURM_SUBMIT_DIR:-}/hpc/leonardo/config.local.sh" \
          "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[config] loaded $_c"; break; }
done

REPO="${REPO:-$HOME/parrot-neuro}"
: "${WORKDIR:?set WORKDIR in config.local.sh (e.g. /leonardo_work/<ACCT>)}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-$WORKDIR/parrot/eeg_bold_fit_sweep_res}"
: "${SWEEP_SUBJECTS:?set SWEEP_SUBJECTS in config.local.sh, e.g. 010002,010003,010004,010005,010006}"
: "${WANDB_RUN_ID:?expected WANDB_RUN_ID from sweep_dispatch.sh}"

# --- non-swept fit hyperparameters (fixed across the whole sweep) -----------
SWEEP_ATLAS="${SWEEP_ATLAS:-1000}"
SWEEP_SPACING="${SWEEP_SPACING:-2.0}"
SWEEP_LEADFIELD_LABEL="${SWEEP_LEADFIELD_LABEL:-duneuroCGAL}"
SWEEP_OPTIMIZE="${SWEEP_OPTIMIZE:-both}"
SWEEP_NUM_EPOCHS="${SWEEP_NUM_EPOCHS:-300}"
SWEEP_BOLD_EVERY="${SWEEP_BOLD_EVERY:-2}"
SWEEP_EEG_TASK="${SWEEP_EEG_TASK:-eyesclosed}"
SWEEP_FMRI_TASK="${SWEEP_FMRI_TASK:-rest}"
# Same GPU-memory-OOM fixes as optim_cohort.sbatch (atlas=1000 + a long BOLD
# horizon reliably OOMs without both of these -- see their docstrings).
# solver_block_size also drives the BOLD simulator's streaming HRF
# convolution block size (train.build_simulators) -- must be an exact
# multiple of the BOLD period in raw steps (tr_ms/dt); 1400 = one TR/block.
SWEEP_T1_WARMUP="${SWEEP_T1_WARMUP:-30000}"
SWEEP_SOLVER_BLOCK_SIZE="${SWEEP_SOLVER_BLOCK_SIZE:-1400}"
SWEEP_SKIP_DIAGNOSTICS="${SWEEP_SKIP_DIAGNOSTICS:-0}"

# --- pixi -------------------------------------------------------------------
PIXI="$(command -v pixi || true)"
[ -z "$PIXI" ] && [ -x "$HOME/.pixi/bin/pixi" ] && PIXI="$HOME/.pixi/bin/pixi"
[ -n "$PIXI" ] || { echo "ERROR: pixi not found (on PATH or at \$HOME/.pixi/bin) -- run setup_optim_env.sh"; exit 1; }
[ -d "$REPO/.pixi" ] || { echo "ERROR: $REPO/.pixi missing -- run setup_optim_env.sh on a login node"; exit 1; }

export PARROT_JAX_CACHE_DIR="$WORKDIR/parrot/.jax_cache/sweep-$WANDB_RUN_ID"
mkdir -p "$PARROT_JAX_CACHE_DIR" "$SWEEP_OUTPUT_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# WANDB_MODE=offline and WANDB_DIR are set by sweep_dispatch.sh's
# --export=ALL,... -- default them here too in case this is ever run outside
# that flow (e.g. manual debugging).
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$WORKDIR/parrot/wandb_offline/$WANDB_RUN_ID}"
mkdir -p "$WANDB_DIR"

cd "$WORKDIR" || { echo "ERROR: cannot cd to WORKDIR=$WORKDIR"; exit 1; }

echo "=== Parrot sweep trial: job ${SLURM_JOB_ID:-?} run_id=$WANDB_RUN_ID on $(hostname), $(date) ==="
echo "subjects=$SWEEP_SUBJECTS atlas=$SWEEP_ATLAS optimize=$SWEEP_OPTIMIZE epochs=$SWEEP_NUM_EPOCHS"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible -- this stage always needs one)"

ARGS=(
    --bids-root "$BIDS" --subjects "$SWEEP_SUBJECTS" --output-root "$SWEEP_OUTPUT_ROOT"
    --atlas "$SWEEP_ATLAS" --spacing "$SWEEP_SPACING" --leadfield-label "$SWEEP_LEADFIELD_LABEL"
    --optimize "$SWEEP_OPTIMIZE" --num-epochs "$SWEEP_NUM_EPOCHS" --bold-every "$SWEEP_BOLD_EVERY"
    --eeg-task "$SWEEP_EEG_TASK" --fmri-task "$SWEEP_FMRI_TASK"
    --solver-block-size "$SWEEP_SOLVER_BLOCK_SIZE" --t1-warmup "$SWEEP_T1_WARMUP"
)
[ -n "${SWEEP_LEARNING_RATE:-}" ] && ARGS+=( --learning-rate "$SWEEP_LEARNING_RATE" )
[ -n "${SWEEP_LEARNING_RATE_BOLD:-}" ] && ARGS+=( --learning-rate-bold "$SWEEP_LEARNING_RATE_BOLD" )
[ -n "${SWEEP_BOLD_FC_WEIGHT:-}" ] && ARGS+=( --bold-fc-weight "$SWEEP_BOLD_FC_WEIGHT" )
[ -n "${SWEEP_BOLD_DFC_WEIGHT:-}" ] && ARGS+=( --bold-dfc-weight "$SWEEP_BOLD_DFC_WEIGHT" )
[ -n "${SWEEP_BOLD_PSD_WEIGHT:-}" ] && ARGS+=( --bold-psd-weight "$SWEEP_BOLD_PSD_WEIGHT" )
[ -n "${SWEEP_DFC_WINDOW_TRS:-}" ] && ARGS+=( --dfc-window-trs "$SWEEP_DFC_WINDOW_TRS" )
[ -n "${SWEEP_DFC_STEP_TRS:-}" ] && ARGS+=( --dfc-step-trs "$SWEEP_DFC_STEP_TRS" )
[ "$SWEEP_SKIP_DIAGNOSTICS" = 1 ] && ARGS+=( --skip-diagnostics )
# Parallel (whole-node) mode -- see sweep_dispatch.sh's SWEEP_GPUS handling
# for the matching --gres=gpu:<count>/QoS/walltime it computes to go with this.
[ -n "${SWEEP_GPUS:-}" ] && ARGS+=( --gpus "$SWEEP_GPUS" )

SECONDS=0
set +e
(cd "$REPO" && "$PIXI" run python -u examples/eeg_bold_fit_sweep.py "${ARGS[@]}")
rc=$?
set -e
echo "=== sweep trial finished in $((SECONDS/60)) min, rc=$rc ($(date)) ==="
exit $rc
