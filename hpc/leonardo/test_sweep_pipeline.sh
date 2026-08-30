#!/bin/bash
###############################################################################
# Cheap end-to-end concurrency test for the sweep agent/dispatch pipeline.
#
# `submit_sweep.sh start N COUNT` should log exactly N*COUNT runs to wandb.
# If it doesn't (e.g. `start 24 3` finishing but only 23/72 runs showing up),
# this drives the EXACT SAME agent -> sweep_dispatch.sh -> sbatch --wait ->
# offline-train -> wandb-sync path, just cheap enough to iterate on: 1
# subject, 2 epochs, its own SWEEP_NAME + WANDB_PROJECT so it never touches
# your real sweep's state, history, or Bayesian search.
#
# It then counts trials at each stage (SLURM jobs actually dispatched vs.
# local offline run dirs written vs. what synced to wandb) so a gap tells
# you WHERE trials are being lost:
#   - dispatched < expected      -> sbatch submissions themselves are failing
#                                    (QoS/account/resource request rejected)
#   - offline dirs < dispatched  -> training is crashing before wandb.init()
#                                    (real failure -- read the .out/.err below)
#   - wandb UI count < offline dirs -> the `wandb sync` step is the problem
#                                    (sync_orphaned_runs.sh run below should
#                                    already have caught/fixed this)
#
# Usage:
#   bash hpc/leonardo/test_sweep_pipeline.sh create           # once
#   bash hpc/leonardo/test_sweep_pipeline.sh run [N] [COUNT]  # default 6 3 = 18 trials
#
# Override the fast-test defaults if needed:
#   TEST_SWEEP_SUBJECT=010002 TEST_SWEEP_QOS=normal TEST_SWEEP_TIME=00:30:00 \
#     bash hpc/leonardo/test_sweep_pipeline.sh run 6 3
#
# Not auto-cleaned-up: delete the wandb project named by WANDB_PROJECT below
# from the UI when done, and `rm hpc/leonardo/.sweep_id.test
# hpc/leonardo/.sweep_agent_pids.test` plus `rm -r hpc/leonardo/sweep_logs-test`.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] || { echo "ERROR: hpc/leonardo/config.local.sh not found -- cp it from config.local.sh.example"; exit 1; }
. "$SCRIPT_DIR/config.local.sh"
: "${WORKDIR:?set WORKDIR in config.local.sh (e.g. /leonardo_work/<ACCT>)}"

# SWEEP_NAME=test namespaces sweep-ID/agent-PID/log files (see submit_sweep.sh);
# a dedicated WANDB_PROJECT keeps these throwaway 2-epoch runs out of your
# real sweep's dashboard entirely.
export SWEEP_NAME="test"
export WANDB_PROJECT="${TEST_WANDB_PROJECT:-parrot-eeg-bold-sweep-test}"
export SWEEP_NUM_EPOCHS=2
export SWEEP_SKIP_DIAGNOSTICS=1
export SWEEP_SUBJECTS="${TEST_SWEEP_SUBJECT:-${SUBJECT:-010002,010003,010004,010005}}"
# NOT boost_qos_lprod's shared 32-GPU account cap (unnecessary here) and NOT
# boost_qos_dbg (max 1 running-or-pending job -- defeats testing concurrency).
export SWEEP_QOS="${TEST_SWEEP_QOS:-normal}"
export SWEEP_TIME="${TEST_SWEEP_TIME:-00:30:00}"

CMD="${1:-}"
case "$CMD" in
    create)
        echo "[test] registering test sweep: project=$WANDB_PROJECT subject=$SWEEP_SUBJECTS epochs=$SWEEP_NUM_EPOCHS qos=$SWEEP_QOS"
        bash "$SCRIPT_DIR/submit_sweep.sh" create
        ;;

    run)
        n="${2:-6}"
        count="${3:-3}"
        expected=$((n * count))
        start_ts="$(date +%Y-%m-%dT%H:%M:%S)"
        echo "[test] launching $n agents x $count runs = $expected trials (project=$WANDB_PROJECT, subject=$SWEEP_SUBJECTS, epochs=$SWEEP_NUM_EPOCHS, qos=$SWEEP_QOS)"
        bash "$SCRIPT_DIR/submit_sweep.sh" start "$n" "$count"

        echo "[test] waiting for all $n background agents to exit (polling every 60s -- this can take a few minutes)..."
        while :; do
            alive_line="$(bash "$SCRIPT_DIR/submit_sweep.sh" status 2>/dev/null | grep -oE '[0-9]+/[0-9]+ agent' || true)"
            alive="${alive_line%%/*}"
            [ -z "$alive" ] && alive=0
            [ "$alive" -eq 0 ] && break
            echo "  ... $alive_line still running"
            sleep 60
        done
        echo "[test] all agents exited."

        echo "[test] resyncing any orphaned offline runs (dropped-SSH recovery)..."
        bash "$SCRIPT_DIR/sync_orphaned_runs.sh" || true

        dispatched="$(sacct -u "$USER" --name=parrot-sweep -X --starttime="$start_ts" --format=JobID --noheader 2>/dev/null | wc -l)"
        offline_dirs="$(find "$WORKDIR/parrot/wandb_offline" -maxdepth 2 -type d -name 'offline-run-*' -newermt "$start_ts" 2>/dev/null | wc -l)"

        echo ""
        echo "=== [test] summary ==="
        echo "  expected trials (N*COUNT):        $expected"
        echo "  SLURM jobs dispatched since start: $dispatched   (sacct -u \$USER --name=parrot-sweep --starttime=$start_ts)"
        echo "  local offline run dirs written:    $offline_dirs   (== wandb.init() was reached, regardless of sync)"
        echo "  now check the wandb UI project '$WANDB_PROJECT' -- its run count is ground truth for what's actually logged."
        echo ""
        echo "  dispatched < expected      -> sbatch submission itself is failing; check the [sweep_dispatch] output"
        echo "                                 in sweep_logs-test/agent-*.log for the sbatch error"
        echo "  offline_dirs < dispatched  -> training crashed before wandb.init(); check parrot-sweep-*.out/.err"
        echo "                                 (written to the directory you ran 'start' from) for the real error"
        echo "  wandb UI count < offline_dirs -> sync failures; sync_orphaned_runs.sh above should have caught these --"
        echo "                                 rerun it once more to confirm nothing's left unsynced"
        ;;

    *)
        cat >&2 <<'EOF'
usage: test_sweep_pipeline.sh <command>

  create              register the isolated test sweep (once)
  run [N] [COUNT]     N agents x COUNT runs (default 6 3 = 18 trials) at
                      2 epochs / 1 subject, then report expected vs.
                      dispatched vs. actually-logged trial counts.

This drives the real submit_sweep.sh/sweep_dispatch.sh pipeline under
SWEEP_NAME=test + a dedicated WANDB_PROJECT, so it never touches your real
sweep's state. See the header comment for what each summary line means.
EOF
        exit 1 ;;
esac
