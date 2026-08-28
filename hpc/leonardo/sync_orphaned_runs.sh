#!/bin/bash
###############################################################################
# Re-sync any sweep trial results that finished training but never got
# uploaded to wandb -- recovery for the case where the LOGIN-NODE
# sweep_dispatch.sh process (blocked in `sbatch --wait`) died before its
# compute job finished, e.g. an SSH session getting killed/dropped (Leonardo
# has multiple login nodes behind one round-robin alias -- see README.md).
#
# SLURM jobs don't depend on the submitting process staying alive, so the
# training itself completes fine and writes a full offline wandb run
# directory -- but with sweep_dispatch.sh gone, nothing is left to run the
# `wandb sync` step that would normally follow `sbatch --wait` returning.
# The data isn't lost, just unsynced; this re-syncs everything found.
#
# Safe to run anytime, over everything, not just orphaned runs -- `wandb
# sync` is idempotent (already-synced runs are a harmless no-op). Each
# offline run directory already has its run ID embedded from when the
# training job called wandb.init(id=..., mode="offline"), so no --id needed.
#
# Usage: bash hpc/leonardo/sync_orphaned_runs.sh
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _c in "${PARROT_CONFIG:-}" \
          "$SCRIPT_DIR/config.local.sh" \
          "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[sync_orphaned_runs] loaded $_c"; break; }
done
: "${WORKDIR:?set WORKDIR in config.local.sh (e.g. /leonardo_work/<ACCT>)}"
REPO="${REPO:-$HOME/parrot-neuro}"

if ! command -v wandb >/dev/null 2>&1; then
    PIXI="$(command -v pixi || true)"
    [ -z "$PIXI" ] && [ -x "$HOME/.pixi/bin/pixi" ] && PIXI="$HOME/.pixi/bin/pixi"
    [ -n "$PIXI" ] || { echo "ERROR: neither 'wandb' nor 'pixi' found on PATH"; exit 1; }
    # Activate the pixi env ONCE in this shell (eval its activation hook), not
    # `pixi run wandb sync ...` per offline run in the loop below -- `pixi run`
    # pays full environment resolution (including a repodata-cache check on a
    # network filesystem, per the WARN it prints) on EVERY invocation. With
    # dozens/hundreds of orphaned runs that overhead compounds into looking
    # like the loop is stuck, when it's actually just re-paying pixi startup
    # cost from scratch each time. One shell-hook eval, then plain `wandb`
    # calls for the rest of this script's life.
    echo "[sync_orphaned_runs] activating pixi env once (not per-run)"
    # set +u around the hook: its activation script (bash-completion files it
    # sources, e.g. hwloc's) isn't `set -u`-safe -- references $ZSH_VERSION
    # unconditionally, an unbound-variable error under our own -u. Not our
    # script's bug to fix, just needs tolerating around this one eval.
    set +u
    eval "$(cd "$REPO" && "$PIXI" shell-hook)"
    set -u
    command -v wandb >/dev/null 2>&1 || { echo "ERROR: wandb still not on PATH after pixi shell-hook"; exit 1; }
fi

n=0
failed=0
for run_dir in "$WORKDIR"/parrot/wandb_offline/*/wandb/offline-run-*; do
    [ -d "$run_dir" ] || continue
    n=$((n + 1))
    echo "=== [$n] syncing $run_dir ==="
    wandb sync "$run_dir" || { echo "  FAILED: $run_dir" >&2; failed=$((failed + 1)); }
done

if [ "$n" -eq 0 ]; then
    echo "No offline run directories found under $WORKDIR/parrot/wandb_offline/"
else
    echo "Done: $((n - failed))/$n synced successfully."
    [ "$failed" -eq 0 ] || { echo "$failed failed -- see output above." >&2; exit 1; }
fi
