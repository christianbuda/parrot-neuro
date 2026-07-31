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
# 5 subjects x up to 300 epochs each, sequentially, is easily several times a
# single-subject fit's walltime -- default to the 4-day QoS (same one `pilot`
# uses for its own unmeasured first run), NOT the 24h "normal" QoS. Override
# SWEEP_QOS in config.local.sh once you've actually measured a trial's walltime
# (see submit_sweep.sh smoke).
SWEEP_QOS="${SWEEP_QOS:-${PILOT_QOS:-boost_qos_lprod}}"
SWEEP_CPUS="${SWEEP_CPUS:-8}"
SWEEP_MEM="${SWEEP_MEM:-64G}"
SWEEP_TIME="${SWEEP_TIME:-2-00:00:00}"

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
echo "[sweep_dispatch] dispatching to compute node: qos=$SWEEP_QOS time=$SWEEP_TIME mem=$SWEEP_MEM cpus=$SWEEP_CPUS"

set +e
sbatch --wait --account="$ACCT" --job-name=parrot-sweep \
    --partition="$BOOST_PART" --qos="$SWEEP_QOS" --gres=gpu:1 \
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
