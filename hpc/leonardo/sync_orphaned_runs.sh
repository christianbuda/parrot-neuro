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
# Works unchanged for BOTH the Optuna pipeline (examples/eeg_bold_fit_optuna.py
# / optuna_train.sbatch) and the legacy wandb-sweep one (sweep_train.sbatch)
# -- this globs by directory STRUCTURE (<dir>/wandb/offline-run-*), which
# both write, not by any pipeline-specific naming.
#
# Usage:
#   bash hpc/leonardo/sync_orphaned_runs.sh            # everything
#   bash hpc/leonardo/sync_orphaned_runs.sh 'optuna-*'  # Optuna runs only --
#     matches optuna_train.sbatch's WANDB_DIR naming (wandb_offline/optuna-
#     <study_name>-<task_tag>/); the legacy path names dirs after the bare
#     wandb run ID instead, so this glob naturally excludes those. Quote it
#     so the shell doesn't expand the glob itself before this script sees it.
###############################################################################
set -euo pipefail

DIR_GLOB="${1:-*}"

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

# `wandb`'s import chain pulls in numpy transitively (via pydantic's docstring
# introspection, not anything sync actually needs numpy FOR) -- numpy's
# OpenBLAS backend then tries to auto-detect the node's full core count and
# spin up a matching thread pool (128 on a Booster node) on every single
# invocation. With however many other processes you already have running
# (background sweep agents, active job processes, etc.), that can exhaust
# RLIMIT_NPROC and fail outright. Sync needs zero linear algebra, so just
# stop OpenBLAS from trying -- no downside, this loop never touches numpy for
# real work.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

n=0
failed=0
corrupt=0
for run_dir in "$WORKDIR"/parrot/wandb_offline/$DIR_GLOB/wandb/offline-run-*; do
    [ -d "$run_dir" ] || continue
    n=$((n + 1))
    echo "=== [$n] syncing $run_dir ==="
    # `wandb sync` exits 0 even when the local transaction log is
    # truncated (the training job got SIGKILL'd -- OOM/walltime/scancel/
    # node failure -- before the wandb writer flushed+closed cleanly): it
    # just uploads whatever was readable up to the break and prints
    # "ERROR ... transactionlog: error getting next record: EOF" as a
    # non-fatal warning. Checking only $? (as this used to) miscounts
    # these partial-data runs as full successes -- grep the output too.
    output="$(wandb sync "$run_dir" 2>&1)"
    rc=$?
    printf '%s\n' "$output"
    if [ "$rc" -ne 0 ]; then
        echo "  FAILED (exit $rc): $run_dir" >&2
        failed=$((failed + 1))
    elif printf '%s\n' "$output" | grep -q 'transactionlog: error getting next record'; then
        echo "  PARTIAL (truncated local log -- run likely killed mid-training, some data missing on wandb): $run_dir" >&2
        corrupt=$((corrupt + 1))
    fi
done

if [ "$n" -eq 0 ]; then
    echo "No offline run directories found under $WORKDIR/parrot/wandb_offline/$DIR_GLOB/"
else
    echo "Done: $((n - failed - corrupt))/$n fully synced, $corrupt partial (truncated log), $failed hard failures."
    if [ "$failed" -ne 0 ] || [ "$corrupt" -ne 0 ]; then
        echo "See PARTIAL/FAILED lines above. A PARTIAL run's SLURM job was probably killed mid-training --" >&2
        echo "cross-check its run ID (the offline-run-*-<id> directory name) against 'sacct -u \$USER --name=parrot-sweep --format=JobID,State,ExitCode' for OOM/TIMEOUT/CANCELLED." >&2
        exit 1
    fi
fi
