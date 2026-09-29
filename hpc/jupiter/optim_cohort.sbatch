#!/bin/bash
###############################################################################
# EEG+BOLD optimization runner for JUPITER (JSC) -- ONE subject, ONE fit.
#
# A THIN, resource-agnostic runner (mirrors hpc/leonardo/optim_cohort.sbatch):
# it just runs examples/eeg_bold_fit_cli.py for one subject with whatever fit
# hyperparameters it's told (`$OPTIM_*`). It does NOT know about
# partitions/QoS/walltime -- all of that lives in submit_optim.sh, the single
# source of truth. This file should almost never need editing.
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
# rest of the JUPITER scripts. sbatch spools this script, so $BASH_SOURCE isn't
# the repo copy -- resolve via $SLURM_SUBMIT_DIR or the default clone at $HOME.
for _c in "${PARROT_CONFIG:-}" \
          "${SLURM_SUBMIT_DIR:-}/hpc/jupiter/config.local.sh" \
          "$HOME/parrot-neuro/hpc/jupiter/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[config] loaded $_c"; break; }
done

REPO="${REPO:-$HOME/parrot-neuro}"
: "${WORKDIR:?set WORKDIR in config.local.sh (e.g. /p/project1/<PROJECT_DIR>)}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
OPTIM_OUTPUT_DIR="${OPTIM_OUTPUT_DIR:-$WORKDIR/parrot/eeg_bold_fit_res}"

# --- fit hyperparameters (set by submit_optim.sh via --export) --------------
OPTIM_ATLAS="${OPTIM_ATLAS:-1000}"
OPTIM_SPACING="${OPTIM_SPACING:-2.0}"
OPTIM_LEADFIELD_LABEL="${OPTIM_LEADFIELD_LABEL:-duneuroCGAL}"
OPTIM_OPTIMIZE="${OPTIM_OPTIMIZE:-both}"
# bold_model: "hrf" (default, linear HRF-kernel convolution) | "balloon"
# (Friston/Deco Balloon-Windkessel hemodynamic ODE) -- see eeg_bold_fit_cli.py
# --bold-model.
OPTIM_BOLD_MODEL="${OPTIM_BOLD_MODEL:-hrf}"
# schedule: "alternating" (default, original interleaved fit) | "phased"
# (splits OPTIM_NUM_EPOCHS in half: BOLD-only then EEG-only) | "joint" (one
# combined EEG+BOLD loss/step per epoch instead of two separate ones) -- see
# eeg_bold_fit_cli.py --schedule. joint_*_weight only matters for "joint".
OPTIM_SCHEDULE="${OPTIM_SCHEDULE:-alternating}"
OPTIM_JOINT_EEG_WEIGHT="${OPTIM_JOINT_EEG_WEIGHT:-1e5}"
OPTIM_JOINT_BOLD_WEIGHT="${OPTIM_JOINT_BOLD_WEIGHT:-1.0}"
# BOLD loss is a weighted combination of static FC + dFC/FCD (both computed
# from the same simulated trajectory -- see train.make_bold_loss_fn); either
# weight at 0 recovers a single-mode fit.
OPTIM_BOLD_FC_WEIGHT="${OPTIM_BOLD_FC_WEIGHT:-0.5}"
OPTIM_BOLD_DFC_WEIGHT="${OPTIM_BOLD_DFC_WEIGHT:-0.5}"
# dFC sliding-window length/stride in TRs (config.BoldFitConfig.dfc_window_trs/
# dfc_step_trs) -- two of the 7 Optuna/wandb-swept fields.
OPTIM_DFC_WINDOW_TRS="${OPTIM_DFC_WINDOW_TRS:-6}"
OPTIM_DFC_STEP_TRS="${OPTIM_DFC_STEP_TRS:-1}"
OPTIM_NUM_EPOCHS="${OPTIM_NUM_EPOCHS:-300}"
OPTIM_BOLD_EVERY="${OPTIM_BOLD_EVERY:-2}"
OPTIM_EEG_TASK="${OPTIM_EEG_TASK:-eyesclosed}"
OPTIM_FMRI_TASK="${OPTIM_FMRI_TASK:-rest}"
OPTIM_LEARNING_RATE="${OPTIM_LEARNING_RATE:-1e-2}"
# Empty (default) = reuse OPTIM_LEARNING_RATE for the BOLD step too -- EEG and
# BOLD each get their own Adam state, so they can also use different rates.
OPTIM_LEARNING_RATE_BOLD="${OPTIM_LEARNING_RATE_BOLD:-}"
OPTIM_BOLD_PSD_WEIGHT="${OPTIM_BOLD_PSD_WEIGHT:-0}"
OPTIM_GAMMA_WEIGHT="${OPTIM_GAMMA_WEIGHT:-0}"
OPTIM_SKIP_DIAGNOSTICS="${OPTIM_SKIP_DIAGNOSTICS:-0}"
# GPU-memory fixes carried over from LEONARDO's A100 tuning (see
# hpc/leonardo/README.md's OOM notes) -- UNVERIFIED on GH200's unified-memory
# architecture; re-measure with `pilot` before trusting these numbers here.
OPTIM_T1_WARMUP="${OPTIM_T1_WARMUP:-30000}"
OPTIM_SOLVER_BLOCK_SIZE="${OPTIM_SOLVER_BLOCK_SIZE:-1400}"
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
[ -d "$REPO/.pixi" ] || { echo "ERROR: $REPO/.pixi missing -- run setup_optim_env.sh on the login node"; exit 1; }

# Keep the JAX compilation cache off JUPITER's small, quota'd $HOME, and give
# each subject its own subdir -- a job array writing one shared cache
# concurrently risks lock contention / a corrupt shared entry.
export PARROT_JAX_CACHE_DIR="$WORKDIR/parrot/.jax_cache/sub-$SUBJECT"
mkdir -p "$PARROT_JAX_CACHE_DIR" "$OPTIM_OUTPUT_DIR"

# JAX's default preallocates ~75% of the GPU upfront regardless of actual
# need -- same rationale as hpc/leonardo/optim_cohort.sbatch. Unverified
# whether this matters the same way on GH200's unified CPU/GPU memory, but
# harmless either way (config.apply_jax_env() already sets these via
# setdefault, so this is belt-and-suspenders).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Anchor CWD on the robust work FS (same rationale as the LEONARDO script).
cd "$WORKDIR" || { echo "ERROR: cannot cd to WORKDIR=$WORKDIR"; exit 1; }

echo "=== Parrot optim: job ${SLURM_JOB_ID:-?} (array ${SLURM_ARRAY_JOB_ID:--}_${SLURM_ARRAY_TASK_ID:--}) on $(hostname), $(date) ==="
echo "subject=$SUBJECT  atlas=$OPTIM_ATLAS  optimize=$OPTIM_OPTIMIZE  bold_model=$OPTIM_BOLD_MODEL  schedule=$OPTIM_SCHEDULE  bold_fc_weight=$OPTIM_BOLD_FC_WEIGHT  bold_dfc_weight=$OPTIM_BOLD_DFC_WEIGHT  dfc_window_trs=$OPTIM_DFC_WINDOW_TRS  dfc_step_trs=$OPTIM_DFC_STEP_TRS  epochs=$OPTIM_NUM_EPOCHS  early_stop_patience=${OPTIM_EARLY_STOP_PATIENCE:-off}"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible -- this stage always needs one)"

# --- optional GPU-utilisation sampling (submit_optim.sh sets this for `pilot`
# only -- see hpc/leonardo/optim_cohort.sbatch for the same rationale).
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
    --optimize "$OPTIM_OPTIMIZE" --bold-model "$OPTIM_BOLD_MODEL" --schedule "$OPTIM_SCHEDULE"
    --joint-eeg-weight "$OPTIM_JOINT_EEG_WEIGHT" --joint-bold-weight "$OPTIM_JOINT_BOLD_WEIGHT"
    --bold-fc-weight "$OPTIM_BOLD_FC_WEIGHT" --bold-dfc-weight "$OPTIM_BOLD_DFC_WEIGHT"
    --dfc-window-trs "$OPTIM_DFC_WINDOW_TRS" --dfc-step-trs "$OPTIM_DFC_STEP_TRS"
    --num-epochs "$OPTIM_NUM_EPOCHS" --bold-every "$OPTIM_BOLD_EVERY"
    --eeg-task "$OPTIM_EEG_TASK" --fmri-task "$OPTIM_FMRI_TASK"
    --learning-rate "$OPTIM_LEARNING_RATE"
)
[ "$OPTIM_SKIP_DIAGNOSTICS" = 1 ] && ARGS+=( --skip-diagnostics )
[ -n "$OPTIM_SOLVER_BLOCK_SIZE" ] && ARGS+=( --solver-block-size "$OPTIM_SOLVER_BLOCK_SIZE" )
[ -n "$OPTIM_T1_WARMUP" ] && ARGS+=( --t1-warmup "$OPTIM_T1_WARMUP" )
[ -n "$OPTIM_LEARNING_RATE_BOLD" ] && ARGS+=( --learning-rate-bold "$OPTIM_LEARNING_RATE_BOLD" )
[ -n "$OPTIM_BOLD_PSD_WEIGHT" ] && [ "$OPTIM_BOLD_PSD_WEIGHT" != 0 ] && ARGS+=( --bold-psd-weight "$OPTIM_BOLD_PSD_WEIGHT" )
[ -n "$OPTIM_GAMMA_WEIGHT" ] && [ "$OPTIM_GAMMA_WEIGHT" != 0 ] && ARGS+=( --gamma-weight "$OPTIM_GAMMA_WEIGHT" )
if [ -n "$OPTIM_EARLY_STOP_PATIENCE" ]; then
    ARGS+=( --early-stop-patience "$OPTIM_EARLY_STOP_PATIENCE"
            --early-stop-window "$OPTIM_EARLY_STOP_WINDOW"
            --early-stop-min-delta "$OPTIM_EARLY_STOP_MIN_DELTA" )
fi

SECONDS=0
set +e
# -u: unbuffered stdout, same rationale as hpc/leonardo/optim_cohort.sbatch.
# -e optim: the optimization-stage pixi environment/feature (see
# setup_optim_env.sh's header -- pixi.toml needs restructuring for this to
# resolve on linux-aarch64; update this flag if the feature ends up named
# differently).
(cd "$REPO" && "$PIXI" run -e optim python -u examples/eeg_bold_fit_cli.py "${ARGS[@]}")
rc=$?
set -e
echo "=== optim finished in $((SECONDS/60)) min, rc=$rc ($(date)) ==="

if [ -n "$GPULOG" ]; then
    echo "--- GPU-idle fraction (rows at 0% util / total samples in $GPULOG) ---"
    awk -F', ' 'NR>1{t++; if($2 ~ /^0 ?%/) z++} END{if(t)printf "    %d/%d = %.0f%% idle\n", z, t, 100*z/t}' "$GPULOG" 2>/dev/null || true
fi

exit $rc
