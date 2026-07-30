#!/bin/bash
###############################################################################
# EEG+BOLD optimization runner for CINECA LEONARDO -- ONE subject, ONE fit.
#
# A THIN, resource-agnostic runner (mirrors hpc/leonardo/cohort.sbatch for the
# reconstruction pipeline): it just runs examples/eeg_bold_fit_cli.py for one
# subject with whatever fit hyperparameters it's told (`$OPTIM_*`). It does
# NOT know about partitions/QoS/walltime -- all of that lives in
# submit_optim.sh, the single source of truth. This file should almost never
# need editing.
#
# It is submitted three ways, all by submit_optim.sh:
#   - as a job ARRAY over the cohort -> subject = line ($SLURM_ARRAY_TASK_ID+1)
#                                        of $OPTIM_SUBJECTS_FILE
#   - as a single SMOKE/PILOT job     -> subject = $OPTIM_SUBJECT
# Never `sbatch` this directly; you'd get no --account or resources.
###############################################################################

#SBATCH --job-name=parrot-optim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=parrot-optim-%x-%A_%a.out
#SBATCH --error=parrot-optim-%x-%A_%a.err

set -euo pipefail

# --- personal config (gitignored): ACCT/WORKDIR/REPO/BIDS shared with the ----
# reconstruction scripts. sbatch spools this script, so $BASH_SOURCE isn't the
# repo copy -- resolve via $SLURM_SUBMIT_DIR or the default clone at $HOME.
for _c in "${PARROT_CONFIG:-}" \
          "${SLURM_SUBMIT_DIR:-}/hpc/leonardo/config.local.sh" \
          "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[config] loaded $_c"; break; }
done

REPO="${REPO:-$HOME/parrot-neuro}"
: "${WORKDIR:?set WORKDIR in config.local.sh (e.g. /leonardo_work/<ACCT>)}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
OPTIM_OUTPUT_DIR="${OPTIM_OUTPUT_DIR:-$WORKDIR/parrot/eeg_bold_fit_res}"

# --- fit hyperparameters (set by submit_optim.sh via --export) --------------
OPTIM_ATLAS="${OPTIM_ATLAS:-1000}"
OPTIM_SPACING="${OPTIM_SPACING:-2.0}"
OPTIM_LEADFIELD_LABEL="${OPTIM_LEADFIELD_LABEL:-duneuroCGAL}"
OPTIM_OPTIMIZE="${OPTIM_OPTIMIZE:-both}"
OPTIM_BOLD_LOSS="${OPTIM_BOLD_LOSS:-fc}"
OPTIM_NUM_EPOCHS="${OPTIM_NUM_EPOCHS:-300}"
OPTIM_BOLD_EVERY="${OPTIM_BOLD_EVERY:-2}"
OPTIM_EEG_TASK="${OPTIM_EEG_TASK:-eyesclosed}"
OPTIM_FMRI_TASK="${OPTIM_FMRI_TASK:-rest}"
OPTIM_LEARNING_RATE="${OPTIM_LEARNING_RATE:-1e-2}"
OPTIM_SKIP_DIAGNOSTICS="${OPTIM_SKIP_DIAGNOSTICS:-0}"
# GPU-memory fixes for atlas=1000 + the long default BOLD horizon (validated
# locally on GPU 2026-07-29: without these, even an 80G card OOMs; with both,
# peak usage is ~31GiB -- comfortable headroom under the A100's 64G). See
# train.build_simulators' and network.build_network's docstrings for why each
# is needed (they fix two DIFFERENT OOM sites -- the one-time warm-up solve
# vs. the actual training gradient step -- neither alone is sufficient).
OPTIM_T1_WARMUP="${OPTIM_T1_WARMUP:-30000}"
OPTIM_SOLVER_BLOCK_SIZE="${OPTIM_SOLVER_BLOCK_SIZE:-565}"
# Early stopping (train.is_loss_stalled): stop once every actively-optimized
# loss's trend has stayed flat/increasing for OPTIM_EARLY_STOP_PATIENCE
# consecutive OPTIM_EARLY_STOP_WINDOW-sized checks. Empty (default) = off,
# matching config.BoldFitConfig's own default -- set an int to opt in.
OPTIM_EARLY_STOP_PATIENCE="${OPTIM_EARLY_STOP_PATIENCE:-}"
OPTIM_EARLY_STOP_WINDOW="${OPTIM_EARLY_STOP_WINDOW:-20}"
OPTIM_EARLY_STOP_MIN_DELTA="${OPTIM_EARLY_STOP_MIN_DELTA:-1e-3}"

# --- which subject: array element, or an explicit smoke/pilot subject -------
if [ -n "${OPTIM_SUBJECT:-}" ]; then
    SUBJECT="$OPTIM_SUBJECT"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    : "${OPTIM_SUBJECTS_FILE:?OPTIM_SUBJECTS_FILE not set for an array run}"
    SUBJECT="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$OPTIM_SUBJECTS_FILE")"
    [ -n "$SUBJECT" ] || { echo "ERROR: no subject at line $((SLURM_ARRAY_TASK_ID + 1)) of $OPTIM_SUBJECTS_FILE"; exit 1; }
else
    echo "ERROR: neither OPTIM_SUBJECT nor SLURM_ARRAY_TASK_ID set."; exit 1
fi

# --- pixi ---------------------------------------------------------------
PIXI="$(command -v pixi || true)"
[ -z "$PIXI" ] && [ -x "$HOME/.pixi/bin/pixi" ] && PIXI="$HOME/.pixi/bin/pixi"
[ -n "$PIXI" ] || { echo "ERROR: pixi not found (on PATH or at \$HOME/.pixi/bin) -- run setup_optim_env.sh"; exit 1; }
[ -d "$REPO/.pixi" ] || { echo "ERROR: $REPO/.pixi missing -- run setup_optim_env.sh on a login node"; exit 1; }

# Keep the JAX compilation cache off LEONARDO's small, quota'd $HOME, and give
# each subject its own subdir -- a job array writing one shared cache
# concurrently risks lock contention / a corrupt shared entry.
export PARROT_JAX_CACHE_DIR="$WORKDIR/parrot/.jax_cache/sub-$SUBJECT"
mkdir -p "$PARROT_JAX_CACHE_DIR" "$OPTIM_OUTPUT_DIR"

# JAX's default preallocates ~75% of the GPU upfront regardless of actual
# need, which inflates nvidia-smi readings and (on a tighter-memory node)
# needlessly risks OOM before the real working set is even known. Validated
# locally: disabling this doesn't change results (bit-identical losses),
# just reports/uses memory honestly (measured ~31GiB actual peak vs. ~70GiB
# with default preallocation on the same run).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Anchor CWD on the robust work FS (same rationale as cohort.sbatch: avoid any
# lstat/identity blip on the home FS at process start).
cd "$WORKDIR" || { echo "ERROR: cannot cd to WORKDIR=$WORKDIR"; exit 1; }

echo "=== Parrot optim: job ${SLURM_JOB_ID:-?} (array ${SLURM_ARRAY_JOB_ID:--}_${SLURM_ARRAY_TASK_ID:--}) on $(hostname), $(date) ==="
echo "subject=$SUBJECT  atlas=$OPTIM_ATLAS  optimize=$OPTIM_OPTIMIZE  bold_loss=$OPTIM_BOLD_LOSS  epochs=$OPTIM_NUM_EPOCHS  early_stop_patience=${OPTIM_EARLY_STOP_PATIENCE:-off}"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible -- this stage always needs one)"

# --- optional GPU-utilisation sampling (submit_optim.sh sets this for `pilot`
# only -- it's the measurement run whose job is to answer "is the A100 idle
# most of the time"; the cohort array skips the overhead once that's known).
GPULOG=""
if [ "${OPTIM_GPU_UTIL_LOG:-0}" = 1 ]; then
    GPULOG="${SLURM_SUBMIT_DIR:-$WORKDIR}/gpu_util-${SLURM_JOB_ID:-$$}.csv"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
               --format=csv -l 10 > "$GPULOG" 2>/dev/null &
    SMI_PID=$!
    trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT
fi

ARGS=(
    --bids-root "$BIDS" --subject "$SUBJECT" --output-root "$OPTIM_OUTPUT_DIR"
    --atlas "$OPTIM_ATLAS" --spacing "$OPTIM_SPACING" --leadfield-label "$OPTIM_LEADFIELD_LABEL"
    --optimize "$OPTIM_OPTIMIZE" --bold-loss "$OPTIM_BOLD_LOSS"
    --num-epochs "$OPTIM_NUM_EPOCHS" --bold-every "$OPTIM_BOLD_EVERY"
    --eeg-task "$OPTIM_EEG_TASK" --fmri-task "$OPTIM_FMRI_TASK"
    --learning-rate "$OPTIM_LEARNING_RATE"
)
[ "$OPTIM_SKIP_DIAGNOSTICS" = 1 ] && ARGS+=( --skip-diagnostics )
[ -n "$OPTIM_SOLVER_BLOCK_SIZE" ] && ARGS+=( --solver-block-size "$OPTIM_SOLVER_BLOCK_SIZE" )
[ -n "$OPTIM_T1_WARMUP" ] && ARGS+=( --t1-warmup "$OPTIM_T1_WARMUP" )
if [ -n "$OPTIM_EARLY_STOP_PATIENCE" ]; then
    ARGS+=( --early-stop-patience "$OPTIM_EARLY_STOP_PATIENCE"
            --early-stop-window "$OPTIM_EARLY_STOP_WINDOW"
            --early-stop-min-delta "$OPTIM_EARLY_STOP_MIN_DELTA" )
fi

SECONDS=0
set +e
# -u: unbuffered stdout. Without it, Python block-buffers stdout once it's not
# a TTY (i.e. redirected to the .out file), so the epoch-by-epoch progress
# prints can sit invisible for the whole job instead of showing up live.
(cd "$REPO" && "$PIXI" run python -u examples/eeg_bold_fit_cli.py "${ARGS[@]}")
rc=$?
set -e
echo "=== optim finished in $((SECONDS/60)) min, rc=$rc ($(date)) ==="

if [ -n "$GPULOG" ]; then
    echo "--- GPU-idle fraction (rows at 0% util / total samples in $GPULOG) ---"
    awk -F', ' 'NR>1{t++; if($2 ~ /^0 ?%/) z++} END{if(t)printf "    %d/%d = %.0f%% idle\n", z, t, 100*z/t}' "$GPULOG" 2>/dev/null || true
fi

exit $rc
