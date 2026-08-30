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

# EXPERIMENTAL (2026-08-30, see hpc/leonardo/README.md's "duplicate run_id"
# gotcha): the training job's own wandb.init() doesn't happen until deep
# inside the offline sbatch job, hours from now, and even then it never
# touches the network (WANDB_MODE=offline) -- so the wandb SWEEP CONTROLLER
# never learns this run_id/config was claimed until the final `wandb sync`.
# In the meantime, any other idle agent that asks the server for "next run"
# can be handed this SAME still-apparently-unclaimed run_id/config again,
# duplicating hours of real GPU compute (confirmed via sacct + agent log
# cross-referencing -- see the README section for the full writeup).
#
# Fix attempt: claim the run with the server RIGHT NOW, from the login node
# (which has real internet, unlike the compute node), by creating it online
# and immediately finishing it with zero data logged. This is a real,
# empty run -- not a no-op -- so the sweep controller should stop treating
# this run_id as available to hand out again. `eeg_bold_fit_sweep.py`
# already calls wandb.init(id=run_id, resume="allow", ...) whenever
# WANDB_RUN_ID is set (see its main()), so the real offline run later
# RESUMES this same run_id and appends its actual data/history to it --
# resuming a run that already exists (even one already marked "finished")
# is the normal, supported wandb resume path, not a special case.
# Single-threaded BLAS for the same reason as the sync call below (wandb's
# import chain pulls in numpy). Non-fatal: if the claim ping itself fails
# (network hiccup, wandb API error), log it and dispatch anyway -- worst
# case we're back to the pre-existing duplicate-run-id risk, not worse.
echo "[sweep_dispatch] claiming run_id=$WANDB_RUN_ID online before offline dispatch"
set +e
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python - <<'PY'
import os
import sys

try:
    import wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "parrot-eeg-bold-sweep"),
        entity=os.environ.get("WANDB_ENTITY"),
        id=os.environ["WANDB_RUN_ID"],
        resume="allow",
        mode="online",
    )
    wandb.finish()
    print("[sweep_dispatch] claim ok")
except Exception as exc:  # noqa: BLE001 -- best-effort, must never block dispatch
    print(f"[sweep_dispatch] WARNING: claim ping failed ({exc!r}) -- dispatching anyway", file=sys.stderr)
PY
claim_rc=$?
set -e
[ "$claim_rc" -eq 0 ] || echo "[sweep_dispatch] WARNING: claim step exited $claim_rc -- dispatching anyway" >&2

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
    # `wandb sync`'s import chain pulls in numpy transitively -- OpenBLAS then
    # sizes its threadpool to the login node's full core count (128) on every
    # invocation. In parallel (SWEEP_GPUS) mode especially, every agent's
    # trial takes about the same wall time, so a whole batch of agents tends
    # to finish and call this within moments of each other -- dozens of
    # simultaneous 128-thread spin-ups blow through the login node's shared
    # RLIMIT_NPROC and this segfaults outright (not just the soft "EOF"
    # warning `wandb sync` prints on a merely-truncated log -- an actual
    # SIGSEGV, losing the sync entirely), silently dropping that trial's data.
    # Sync needs zero linear algebra, so force single-threaded BLAS -- same
    # fix already applied to sync_orphaned_runs.sh for the equivalent
    # overhead there. Guarded with set +e/-e (not the top-level set -e) so a
    # sync failure here is logged but still lets the trial's OWN rc (from
    # training, not sync) reach `exit $rc` below -- sync_orphaned_runs.sh
    # picks up anything that still fails to sync.
    set +e
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        wandb sync --id "$WANDB_RUN_ID" "$run_dir"
    sync_rc=$?
    set -e
    [ "$sync_rc" -eq 0 ] || echo "[sweep_dispatch] WARNING: wandb sync exited $sync_rc for $run_dir -- sync_orphaned_runs.sh will retry it" >&2
else
    echo "[sweep_dispatch] WARNING: no offline run dir found under $OFFLINE_ROOT/wandb -- nothing to sync (training likely failed before wandb.init)" >&2
fi

exit $rc
