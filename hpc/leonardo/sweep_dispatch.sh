#!/bin/bash
###############################################################################
# W&B sweep trial dispatcher -- runs on the LEONARDO LOGIN node (has internet),
# launched once per trial by `wandb agent` (see sweep_eeg_bold.yaml's `command:`
# and submit_sweep.sh's `start`).
#
# Compute nodes have NO internet (see README.md), so `wandb agent` cannot run
# there directly -- it runs here instead, and this script hands each trial off
# to a compute node: submit an `sbatch --wait` job that trains fully offline
# (WANDB_MODE=offline, same WANDB_RUN_ID the agent already assigned), then
# `wandb sync` the result once the job finishes and we have egress again.
#
# `wandb agent` calls this with the trial's sampled hyperparameters as
# `--key=value` argv (the ${args} macro in sweep_eeg_bold.yaml's `command:`)
# and ALSO already exports WANDB_RUN_ID / WANDB_SWEEP_ID / WANDB_PROJECT /
# WANDB_ENTITY / WANDB_API_KEY into this process's environment -- nothing here
# needs to import wandb or talk to the API itself.
#
# Never invoke directly; `wandb agent` is what calls this.
###############################################################################
set -euo pipefail

: "${WANDB_RUN_ID:?wandb agent did not set WANDB_RUN_ID -- run this via 'wandb agent', not directly}"
: "${WANDB_SWEEP_ID:?wandb agent did not set WANDB_SWEEP_ID}"

# --- personal config (gitignored): ACCT/WORKDIR/REPO/SWEEP_* shared with the
# other hpc/leonardo/ scripts. sbatch/agent spool this script, so resolve the
# repo copy the same way optim_cohort.sbatch does.
for _c in "${PARROT_CONFIG:-}" \
          "${SLURM_SUBMIT_DIR:-}/hpc/leonardo/config.local.sh" \
          "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[sweep_dispatch] loaded $_c"; break; }
done

REPO="${REPO:-$HOME/parrot-neuro}"
: "${WORKDIR:?set WORKDIR in config.local.sh (e.g. /leonardo_work/<ACCT>)}"
: "${ACCT:?set ACCT in config.local.sh}"

BOOST_PART="${BOOST_PART:-boost_usr_prod}"
# SWEEP_GPUS (e.g. "0,1,2,3"), unset by default, switches eeg_bold_fit_sweep.py
# to its --gpus parallel mode -- see README.md's "Hyperparameter sweep" ->
# "Parallel (whole-node) mode" section for the full rationale/numbers. This
# changes BOTH the requested GPU count AND the sensible default QoS:
#   - sequential (SWEEP_GPUS unset): --gres=gpu:1, one subject at a time.
#     Fitting all of SWEEP_SUBJECTS sequentially at 300 epochs each is easily
#     several times a single-subject fit's walltime -- default to the 4-day
#     QoS (same one `pilot` uses for its own unmeasured first run), NOT the
#     24h "normal" QoS.
#   - parallel (SWEEP_GPUS set): --gres=gpu:<count>, one whole node's worth of
#     subjects fit SIMULTANEOUSLY per round. Measured on this project
#     (2026-08): ~6.6h/subject: len(SWEEP_SUBJECTS)==len(SWEEP_GPUS) (e.g. 4
#     subjects on 4 GPUs -- the case with ZERO idle-GPU billing, see README)
#     gives a ~6.6h trial, comfortably inside `normal`'s 24h -- which has no
#     documented account-wide GPU cap, unlike boost_qos_lprod's shared 32.
# Either way, SWEEP_QOS/SWEEP_TIME below are still just defaults -- override
# in config.local.sh once you've measured YOUR actual trial walltime (see
# submit_sweep.sh smoke), especially if SWEEP_SUBJECTS doesn't divide evenly
# by SWEEP_GPUS (extra rounds add to the trial's total walltime).
if [ -n "${SWEEP_GPUS:-}" ]; then
    export SWEEP_GPUS
    SWEEP_GPU_COUNT=$(( $(printf '%s' "$SWEEP_GPUS" | tr -cd ',' | wc -c) + 1 ))
    SWEEP_QOS="${SWEEP_QOS:-normal}"
else
    SWEEP_GPU_COUNT=1
    SWEEP_QOS="${SWEEP_QOS:-${PILOT_QOS:-boost_qos_lprod}}"
fi
# 8 cores / 64G per GPU -- matches the single-GPU default exactly, scaled by
# SWEEP_GPU_COUNT so `len(SWEEP_GPUS)` concurrent worker processes each still
# get their own fair share, not all fighting over one GPU's worth of cores.
# For a full-node parallel job this reaches the node's own 32-core/512G specs
# (4 x 8 = 32, 4 x 64 = 256) -- free from a billing perspective too, since
# R (the billing formula's max-reserved-fraction) is already at 1.0 from the
# GPU request alone once SWEEP_GPU_COUNT=4; more CPUs/mem up to the node's
# actual capacity costs nothing extra. See README.md's billing-formula note.
SWEEP_CPUS="${SWEEP_CPUS:-$((8 * SWEEP_GPU_COUNT))}"
SWEEP_MEM="${SWEEP_MEM:-$((64 * SWEEP_GPU_COUNT))G}"
# NOT the same default for both modes: `normal` QoS hard-caps walltime at 24h,
# so parallel mode's default MUST stay under that or sbatch rejects the job
# outright at submission (not a graceful timeout -- an immediate error).
# 20h leaves ~4h margin over the measured ~6.6h/subject x however many rounds
# SWEEP_SUBJECTS needs -- widen it (still <24h) if you run more rounds.
if [ -n "${SWEEP_GPUS:-}" ]; then
    SWEEP_TIME="${SWEEP_TIME:-20:00:00}"
else
    SWEEP_TIME="${SWEEP_TIME:-2-00:00:00}"
fi

# --- map this trial's --key=value argv (wandb's ${args}) to SWEEP_<KEY> env
# vars for sweep_train.sbatch/eeg_bold_fit_sweep.py to pick up. ---
for arg in "$@"; do
    case "$arg" in
        --*=*)
            key="${arg#--}"; key="${key%%=*}"
            val="${arg#*=}"
            export "SWEEP_$(printf '%s' "$key" | tr '[:lower:]' '[:upper:]')=$val"
            ;;
        *) echo "[sweep_dispatch] ignoring unrecognized arg: $arg" >&2 ;;
    esac
done

OFFLINE_ROOT="$WORKDIR/parrot/wandb_offline/$WANDB_RUN_ID"
mkdir -p "$OFFLINE_ROOT"

echo "[sweep_dispatch] trial run_id=$WANDB_RUN_ID sweep_id=$WANDB_SWEEP_ID"
echo "[sweep_dispatch] learning_rate=${SWEEP_LEARNING_RATE:-?} bold_fc_weight=${SWEEP_BOLD_FC_WEIGHT:-?} bold_dfc_weight=${SWEEP_BOLD_DFC_WEIGHT:-?} dfc_window_trs=${SWEEP_DFC_WINDOW_TRS:-?}"
echo "[sweep_dispatch] dispatching to compute node: qos=$SWEEP_QOS time=$SWEEP_TIME mem=$SWEEP_MEM cpus=$SWEEP_CPUS gpus=$SWEEP_GPU_COUNT"

set +e
sbatch --wait --account="$ACCT" --job-name=parrot-sweep \
    --partition="$BOOST_PART" --qos="$SWEEP_QOS" --gres=gpu:"$SWEEP_GPU_COUNT" \
    --cpus-per-task="$SWEEP_CPUS" --time="$SWEEP_TIME" --mem="$SWEEP_MEM" \
    --export=ALL,WANDB_MODE=offline,WANDB_DIR="$OFFLINE_ROOT" \
    "$REPO/hpc/leonardo/sweep_train.sbatch"
rc=$?
set -e

echo "[sweep_dispatch] training job rc=$rc; syncing $OFFLINE_ROOT -> wandb"
# Offline runs land in $WANDB_DIR/wandb/offline-run-<timestamp>-<id>/ -- glob
# for it rather than hardcoding the timestamp.
run_dir=$(compgen -G "$OFFLINE_ROOT/wandb/offline-run-*-$WANDB_RUN_ID" | head -n1 || true)
if [ -n "$run_dir" ]; then
    wandb sync --id "$WANDB_RUN_ID" "$run_dir"
else
    echo "[sweep_dispatch] WARNING: no offline run dir found under $OFFLINE_ROOT/wandb -- nothing to sync (training likely failed before wandb.init)" >&2
fi

exit $rc
