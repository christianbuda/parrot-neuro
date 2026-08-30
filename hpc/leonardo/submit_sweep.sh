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
#   ./submit_sweep.sh list                # show every known sweep (name + ID) in this checkout
#   ./submit_sweep.sh status              # squeue + how many agents are still running
#   ./submit_sweep.sh stop                # kill this sweep's background agents
#
# Running a SECOND, independent search alongside one that's already going:
# prefix every command with SWEEP_NAME=<tag> (and optionally SWEEP_YAML=<path>
# for a genuinely different search space) -- this namespaces the sweep-ID/
# agent-PID/log files so the two never share state or interfere with each
# other. wandb supports concurrent sweeps natively; the only thing this
# script CAN'T partition for you is the Leonardo account's real GPU/node/
# core-hour budget, which both searches still draw from together.
#   SWEEP_NAME=explore2 ./submit_sweep.sh create
#   SWEEP_NAME=explore2 ./submit_sweep.sh smoke
#   SWEEP_NAME=explore2 ./submit_sweep.sh start 8 5
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

# SWEEP_NAME (unset by default) namespaces the sweep-ID/agent-PID/log files so
# a NEW search can be created and run without touching an already-running
# one's state -- e.g. SWEEP_NAME=explore2 ./submit_sweep.sh create. Unset
# (the default) keeps the exact original filenames, so any already-running
# sweep (created before this existed, or run without SWEEP_NAME) is completely
# untouched by a namespaced invocation, and vice versa. Two sweeps this way
# CAN run concurrently -- wandb supports that natively -- the only shared,
# non-namespaced resource is the Leonardo account's real GPU/node/core-hour
# budget, which this script has no way to partition for you.
SWEEP_NAME="${SWEEP_NAME:-}"
if [ -n "$SWEEP_NAME" ]; then
    SWEEP_ID_FILE="$SCRIPT_DIR/.sweep_id.$SWEEP_NAME"
    AGENT_PID_FILE="$SCRIPT_DIR/.sweep_agent_pids.$SWEEP_NAME"
    AGENT_LOG_DIR="${SWEEP_AGENT_LOG_DIR:-$SCRIPT_DIR/sweep_logs-$SWEEP_NAME}"
else
    SWEEP_ID_FILE="$SCRIPT_DIR/.sweep_id"
    AGENT_PID_FILE="$SCRIPT_DIR/.sweep_agent_pids"
    AGENT_LOG_DIR="${SWEEP_AGENT_LOG_DIR:-$SCRIPT_DIR/sweep_logs}"
fi
# SWEEP_YAML lets a namespaced sweep use a genuinely different search space
# (not just a fresh Bayesian search over the same one) -- defaults to the
# usual file, so plain `create` behaves exactly as before.
SWEEP_YAML="${SWEEP_YAML:-$SCRIPT_DIR/sweep_eeg_bold.yaml}"

# Activate the pixi env ONCE in this shell (eval its activation hook) instead
# of wrapping every `wandb` call in `pixi run` -- critically, `start` launches
# up to N agents in a tight loop, and `pixi run wandb agent ...` re-initializes
# pixi's own internal thread pool (rayon) on EVERY invocation. Dozens of those
# starting within the same second or two can blow through the login node's
# RLIMIT_NPROC, causing `pixi` itself to panic ("failed to initialize global
# rayon pool: ... Resource temporarily unavailable") and crash BEFORE `wandb
# agent` ever starts -- an agent that dies at that exact line looks, from the
# outside, exactly like "start requested N agents but far fewer ever actually
# ran," which is exactly the failure this was designed around (same fix
# already applied to sync_orphaned_runs.sh for the equivalent per-run overhead
# there). One shell-hook eval, then every `wandb` call below is direct.
REPO="${REPO:-$HOME/parrot-neuro}"
if ! command -v wandb >/dev/null 2>&1; then
    PIXI="$(command -v pixi || true)"
    [ -z "$PIXI" ] && [ -x "$HOME/.pixi/bin/pixi" ] && PIXI="$HOME/.pixi/bin/pixi"
    [ -n "$PIXI" ] || { echo "ERROR: neither 'wandb' nor 'pixi' found on PATH"; exit 1; }
    # set +u around the hook: its activation script (bash-completion files it
    # sources, e.g. hwloc's) isn't `set -u`-safe -- references $ZSH_VERSION
    # unconditionally, an unbound-variable error under our own -u.
    set +u
    eval "$(cd "$REPO" && "$PIXI" shell-hook)"
    set -u
    command -v wandb >/dev/null 2>&1 || { echo "ERROR: wandb still not on PATH after pixi shell-hook"; exit 1; }
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
        out="$(wandb sweep --project "$WANDB_PROJECT" "${entity_flag[@]}" "$SWEEP_YAML" 2>&1 | tee /dev/stderr)"
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
        wandb agent --count 1 "$id"
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
            nohup wandb agent --count "$runs_per_agent" "$id" > "$log" 2>&1 < /dev/null &
            pid=$!
            disown "$pid" 2>/dev/null || true
            echo "$pid" >> "$AGENT_PID_FILE"
            echo "  agent $i: pid=$pid log=$log"
            # Stagger, not a bare burst of N -- each `wandb agent` startup
            # spins up wandb's own internal asyncio "service" thread (plus
            # Python import overhead) immediately, and launching dozens within
            # the same few seconds risks the login node's RLIMIT_NPROC ceiling
            # -- same class of failure `pixi run` per-agent used to cause
            # outright (see the note above WANDB_BIN's resolution), but a
            # DIFFERENT source of it: confirmed 2026-08-30, N=24 with the old
            # 0.2s stagger still lost 4 agents at launch to "RuntimeError:
            # can't start new thread" / a bare SIGSEGV, all from wandb's own
            # thread spin-up, not pixi/rayon. 2s x 128 = ~4.3min total to reach
            # full concurrency -- negligible against hours-long trials, and
            # gives each agent's thread pool room to settle before the next.
            sleep 2
        done
        ;;

    list)
        echo "--- known sweeps (this checkout) ---"
        found=0
        for f in "$SCRIPT_DIR"/.sweep_id "$SCRIPT_DIR"/.sweep_id.*; do
            [ -f "$f" ] || continue
            found=1
            name="default"
            case "$f" in *.sweep_id.*) name="${f##*.sweep_id.}" ;; esac
            echo "  name=$name  id=$(cat "$f")  file=$f"
        done
        [ "$found" = 1 ] || echo "  (none -- run 'create', optionally with SWEEP_NAME=<name> set, first)"
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
  list                    show every known sweep (name + ID) in this checkout
  status                  squeue + how many agents are still running
  stop                     kill this sweep's background agents

Prefix any command with SWEEP_NAME=<tag> to create/run a SECOND, independent
search without touching an already-running one's state (sweep-ID/agent-PID/
log files are namespaced per name; unset = the original, unnamed files).
Add SWEEP_YAML=<path> alongside SWEEP_NAME=<tag> on 'create' for a genuinely
different search space, not just a fresh Bayesian search over the same one.
Both searches still share the same Leonardo account GPU/node/core-hour
budget -- this script has no way to partition that for you.

Config (account/paths/WANDB_*/SWEEP_* resources) is read from
hpc/leonardo/config.local.sh.
EOF
        exit 1 ;;
esac
