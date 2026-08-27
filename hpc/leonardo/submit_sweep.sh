#!/bin/bash
###############################################################################
# W&B hyperparameter sweep operator for CINECA LEONARDO.
#
# `wandb agent` needs live access to api.wandb.ai; LEONARDO's compute nodes
# have none (see README.md), so the agent runs HERE, on the login node, and
# hands each trial off to a compute node via sweep_dispatch.sh (sbatch --wait,
# offline training, wandb sync afterward) -- see README.md's "Hyperparameter
# sweep" section for the full picture.
#
# Usage (run each step in order the first time you use this):
#   ./submit_sweep.sh create              # register the sweep, print/save its ID
#   ./submit_sweep.sh smoke               # ONE trial, 1 subject, 2 epochs, foreground --
#                                         #   validates the whole agent->dispatch->sbatch
#                                         #   --wait->offline-train->sync round trip
#   ./submit_sweep.sh start [N] [COUNT]   # N background agents (default 8), COUNT runs
#                                         #   each (default 5) -> N*COUNT total trials
#   ./submit_sweep.sh status              # squeue + how many agents are still running
#   ./submit_sweep.sh stop                # kill this sweep's background agents
###############################################################################
set -euo pipefail

# BASH_SOURCE, not $0 -- $0 is the invoking shell (e.g. "-bash"), not this
# script's path, if this is run with `source`/`.` instead of executed directly
# (dirname would then choke on "-bash", misreading "-b" as an option flag).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] || { echo "ERROR: hpc/leonardo/config.local.sh not found -- cp it from config.local.sh.example"; exit 1; }
. "$SCRIPT_DIR/config.local.sh"

: "${WANDB_API_KEY:?set WANDB_API_KEY in config.local.sh (from https://wandb.ai/authorize)}"
WANDB_PROJECT="${WANDB_PROJECT:-parrot-eeg-bold-sweep}"
export WANDB_API_KEY WANDB_PROJECT
[ -n "${WANDB_ENTITY:-}" ] && export WANDB_ENTITY

SWEEP_ID_FILE="$SCRIPT_DIR/.sweep_id"
AGENT_PID_FILE="$SCRIPT_DIR/.sweep_agent_pids"
AGENT_LOG_DIR="${SWEEP_AGENT_LOG_DIR:-$SCRIPT_DIR/sweep_logs}"

# Array, not a string -- "pixi run wandb" is 3 words, and quoting a string
# variable containing spaces makes bash look for one file with that literal
# name (exactly the "No such file or directory" bug this used to hit).
if command -v wandb >/dev/null 2>&1; then
    WANDB_BIN=(wandb)
else
    PIXI="$(command -v pixi || true)"
    [ -z "$PIXI" ] && [ -x "$HOME/.pixi/bin/pixi" ] && PIXI="$HOME/.pixi/bin/pixi"
    [ -n "$PIXI" ] || { echo "ERROR: neither 'wandb' nor 'pixi' found on PATH"; exit 1; }
    WANDB_BIN=("$PIXI" run wandb)
fi

sweep_id() {
    [ -f "$SWEEP_ID_FILE" ] || { echo "ERROR: no sweep registered yet -- run '$0 create' first" >&2; exit 1; }
    cat "$SWEEP_ID_FILE"
}

CMD="${1:-}"
case "$CMD" in
    create)
        [ -f "$SWEEP_ID_FILE" ] && { echo "ERROR: $SWEEP_ID_FILE already exists (sweep $(cat "$SWEEP_ID_FILE")) -- rm it to register a new one" >&2; exit 1; }
        entity_flag=(); [ -n "${WANDB_ENTITY:-}" ] && entity_flag=( --entity "$WANDB_ENTITY" )
        out="$("${WANDB_BIN[@]}" sweep --project "$WANDB_PROJECT" "${entity_flag[@]}" "$SCRIPT_DIR/sweep_eeg_bold.yaml" 2>&1 | tee /dev/stderr)"
        # Save the FULLY-QUALIFIED "entity/project/sweep_id" path (from wandb's own
        # "Run sweep agent with: wandb agent entity/project/id" line), not just the
        # bare ID -- `wandb agent <bare_id>` has to resolve a default entity via the
        # API, which fails ("entityName required for project query") for accounts
        # without one (e.g. team/org accounts). The qualified path never needs that.
        id="$(printf '%s\n' "$out" | grep -oE 'Run sweep agent with: wandb agent .*' | awk '{print $NF}')"
        [ -n "$id" ] || id="$(printf '%s\n' "$out" | grep -oE 'Creating sweep with ID: [A-Za-z0-9]+' | awk '{print $NF}')"
        [ -n "$id" ] || { echo "ERROR: could not parse sweep ID from wandb output above" >&2; exit 1; }
        printf '%s\n' "$id" > "$SWEEP_ID_FILE"
        echo "[create] sweep ID $id saved to $SWEEP_ID_FILE"
        ;;

    smoke)
        id="$(sweep_id)"
        if [ -n "${SWEEP_GPUS:-}" ]; then
            # Parallel mode: keep the SWEEP_SUBJECTS already configured in
            # config.local.sh (sized to match SWEEP_GPUS) instead of forcing
            # it down to 1 -- a 1-subject smoke test would never actually
            # exercise the round-based parallel launch (subprocess-per-GPU,
            # CUDA_VISIBLE_DEVICES pinning, per-GPU JAX cache, result replay).
            : "${SWEEP_SUBJECTS:?SWEEP_GPUS is set but SWEEP_SUBJECTS is not -- set both in config.local.sh}"
            echo "[smoke] PARALLEL foreground trial: gpus=$SWEEP_GPUS subjects=$SWEEP_SUBJECTS epochs=2, no diagnostics"
            export SWEEP_NUM_EPOCHS=2 SWEEP_SKIP_DIAGNOSTICS=1
        else
            subject="${2:-${SUBJECT:-010002}}"
            echo "[smoke] ONE foreground trial: subject=$subject epochs=2, no diagnostics -- sanity check only"
            export SWEEP_SUBJECTS="$subject" SWEEP_NUM_EPOCHS=2 SWEEP_SKIP_DIAGNOSTICS=1
        fi
        "${WANDB_BIN[@]}" agent --count 1 "$id"
        ;;

    start)
        id="$(sweep_id)"
        n_agents="${2:-8}"
        runs_per_agent="${3:-5}"
        mkdir -p "$AGENT_LOG_DIR"
        : > "$AGENT_PID_FILE"
        echo "[start] launching $n_agents background agents x $runs_per_agent runs each ($((n_agents * runs_per_agent)) total trials)"
        echo "[start] these are long-lived LOGIN-NODE processes (hours-to-days) -- run this under tmux/screen,"
        echo "        not a plain shell that dies on logout."
        for i in $(seq 1 "$n_agents"); do
            log="$AGENT_LOG_DIR/agent-$i.log"
            nohup "${WANDB_BIN[@]}" agent --count "$runs_per_agent" "$id" > "$log" 2>&1 < /dev/null &
            pid=$!
            disown "$pid" 2>/dev/null || true
            echo "$pid" >> "$AGENT_PID_FILE"
            echo "  agent $i: pid=$pid log=$log"
        done
        ;;

    status)
        echo "--- SLURM (this user's parrot-sweep jobs) ---"
        squeue --me -o '%.10i %.9P %.20j %.8T %.10M %.6D %R' 2>/dev/null | { head -1; grep parrot-sweep || echo "  (none)"; }
        echo "--- background wandb agents ---"
        if [ -f "$AGENT_PID_FILE" ]; then
            alive=0
            while read -r pid; do
                [ -n "$pid" ] || continue
                if kill -0 "$pid" 2>/dev/null; then echo "  pid $pid: running"; alive=$((alive + 1)); fi
            done < "$AGENT_PID_FILE"
            echo "  $alive/$(wc -l < "$AGENT_PID_FILE") agent(s) still running"
        else
            echo "  (no agents started via '$0 start' in this checkout)"
        fi
        ;;

    stop)
        [ -f "$AGENT_PID_FILE" ] || { echo "no agents to stop (no $AGENT_PID_FILE)"; exit 0; }
        while read -r pid; do
            [ -n "$pid" ] || continue
            kill "$pid" 2>/dev/null && echo "stopped pid $pid" || echo "pid $pid already gone"
        done < "$AGENT_PID_FILE"
        : > "$AGENT_PID_FILE"
        echo "[stop] agents killed. In-flight sbatch --wait dispatches (if any) will finish their compute job"
        echo "       and exit on their own -- cancel those separately with: scancel -u \$USER --name=parrot-sweep"
        ;;

    *)
        cat >&2 <<'EOF'
usage: submit_sweep.sh <command>

  create                  register the sweep from sweep_eeg_bold.yaml, save its ID
  smoke [subject]         ONE foreground trial, 2 epochs, no diagnostics --
                          validates the full agent->dispatch->sbatch->sync round trip.
                          If SWEEP_GPUS is set in config.local.sh, uses the
                          already-configured (parallel-sized) SWEEP_SUBJECTS
                          instead of forcing 1 subject, to exercise the
                          parallel launch too; [subject] only applies otherwise.
  start [N] [COUNT]       N background agents (default 8) x COUNT runs each
                          (default 5) = N*COUNT total trials. Run under tmux/screen.
  status                  squeue + how many agents are still running
  stop                     kill this sweep's background agents

Config (account/paths/WANDB_*/SWEEP_* resources) is read from
hpc/leonardo/config.local.sh.
EOF
        exit 1 ;;
esac
