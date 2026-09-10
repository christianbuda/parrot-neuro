#!/bin/bash
###############################################################################
# Optuna hyperparameter-search operator for CINECA LEONARDO.
#
# Replaces submit_sweep.sh/sweep_dispatch.sh (wandb Sweeps) -- see README.md's
# "Hyperparameter sweep" section for the full rationale. wandb Sweeps needs a
# continuously-online agent process to hand out hyperparameters, which on
# LEONARDO can only run on a login node (compute nodes have no internet);
# keeping that alive for hours-to-days is what caused the RLIMIT_NPROC /
# duplicate-run-id saga documented there. Optuna coordinates the search
# through ONE shared file (JournalStorage) on NFS instead of a server -- no
# process needs to stay alive anywhere. This script's `start` is therefore
# just ONE `sbatch --array=...` submission, not N background agents: SLURM's
# own scheduler handles concurrency/throttling, and there's no login-node
# process-count ceiling to fight at all.
#
# Usage (run each step in order the first time you use this):
#   ./submit_optuna_sweep.sh create              # register the study (once)
#   ./submit_optuna_sweep.sh smoke                # ONE trial, 1 subject, 2 epochs, foreground --
#                                                  #   validates the whole ask->train->tell round trip
#   ./submit_optuna_sweep.sh start [N] [COUNT]    # N*COUNT total trials as ONE SLURM array
#                                                  #   job, throttled to N concurrent (default N=8,
#                                                  #   COUNT=5) -- see README for what N to pick.
#   ./submit_optuna_sweep.sh list                 # show every known study (name + trial counts)
#   ./submit_optuna_sweep.sh status                # squeue + per-state trial counts for this study
#   ./submit_optuna_sweep.sh stop                  # scancel this study's array job
#
# Running a SECOND, independent search alongside one that's already going:
# prefix every command with OPTUNA_STUDY_NAME=<tag> -- this namespaces the
# study file (hence the SLURM job name too, parrot-optuna-<tag>), so the two
# never share state. Optuna supports concurrent independent studies natively;
# the only thing this script CAN'T partition for you is the Leonardo
# account's real GPU/node/core-hour budget, which both searches still draw
# from together.
#   OPTUNA_STUDY_NAME=explore2 ./submit_optuna_sweep.sh create
#   OPTUNA_STUDY_NAME=explore2 ./submit_optuna_sweep.sh start 8 5
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] || { echo "ERROR: hpc/leonardo/config.local.sh not found -- cp it from config.local.sh.example"; exit 1; }
. "$SCRIPT_DIR/config.local.sh"

: "${ACCT:?set ACCT in config.local.sh}"
: "${WORKDIR:?set WORKDIR in config.local.sh}"

BOOST_PART="${BOOST_PART:-boost_usr_prod}"
DEBUG_QOS="${DEBUG_QOS:-boost_qos_dbg}"
PILOT_QOS="${PILOT_QOS:-boost_qos_lprod}"

# OPTUNA_STUDY_NAME (default "default") namespaces the study file / job name --
# a NEW search can be created and run without touching an already-running
# one's state. $WORKDIR is shared NFS across every Leonardo login node, so
# the study file created here is visible everywhere without copying anything.
OPTUNA_STUDY_NAME="${OPTUNA_STUDY_NAME:-default}"
export OPTUNA_STUDY_NAME
OPTUNA_STORAGE="${OPTUNA_STORAGE:-$WORKDIR/parrot/optuna/$OPTUNA_STUDY_NAME.log}"
export OPTUNA_STORAGE
JOB_NAME="parrot-optuna-$OPTUNA_STUDY_NAME"

# --- fixed (non-swept) fit hyperparameters -- override any in config.local.sh
# (OPTUNA_ATLAS=..., etc.) or as a call-time env var. Same fields/defaults as
# sweep_train.sbatch; kept here too so `smoke`/`start` can echo them and so
# `--export=ALL` has something to propagate. ---------------------------------
OPTUNA_ATLAS="${OPTUNA_ATLAS:-1000}"
OPTUNA_SPACING="${OPTUNA_SPACING:-2.0}"
OPTUNA_LEADFIELD_LABEL="${OPTUNA_LEADFIELD_LABEL:-duneuroCGAL}"
OPTUNA_OPTIMIZE="${OPTUNA_OPTIMIZE:-both}"
OPTUNA_BOLD_MODEL="${OPTUNA_BOLD_MODEL:-hrf}"
OPTUNA_SCHEDULE="${OPTUNA_SCHEDULE:-alternating}"
OPTUNA_JOINT_EEG_WEIGHT="${OPTUNA_JOINT_EEG_WEIGHT:-1e5}"
OPTUNA_JOINT_BOLD_WEIGHT="${OPTUNA_JOINT_BOLD_WEIGHT:-1.0}"
OPTUNA_NUM_EPOCHS="${OPTUNA_NUM_EPOCHS:-300}"
OPTUNA_BOLD_EVERY="${OPTUNA_BOLD_EVERY:-2}"
OPTUNA_EEG_TASK="${OPTUNA_EEG_TASK:-eyesclosed}"
OPTUNA_FMRI_TASK="${OPTUNA_FMRI_TASK:-rest}"
OPTUNA_T1_WARMUP="${OPTUNA_T1_WARMUP:-30000}"
OPTUNA_SOLVER_BLOCK_SIZE="${OPTUNA_SOLVER_BLOCK_SIZE:-1400}"
: "${OPTUNA_SUBJECTS:?set OPTUNA_SUBJECTS in config.local.sh, e.g. 010002,010003,010004,010005,010006}"

# Exports every OPTUNA_* fixed hyperparam + study/GPU vars so --export=ALL
# propagates them into the array job's environment.
export_run_vars() {
    export OPTUNA_ATLAS OPTUNA_SPACING OPTUNA_LEADFIELD_LABEL OPTUNA_OPTIMIZE OPTUNA_BOLD_MODEL \
           OPTUNA_SCHEDULE OPTUNA_JOINT_EEG_WEIGHT OPTUNA_JOINT_BOLD_WEIGHT \
           OPTUNA_NUM_EPOCHS OPTUNA_BOLD_EVERY OPTUNA_EEG_TASK OPTUNA_FMRI_TASK \
           OPTUNA_T1_WARMUP OPTUNA_SOLVER_BLOCK_SIZE OPTUNA_SUBJECTS
    [ -n "${OPTUNA_GPUS:-}" ] && export OPTUNA_GPUS
}

# Activate the pixi env ONCE in this shell (eval its activation hook) so the
# small python snippets below (create/list/status) run as plain `python`, not
# `pixi run python` -- consistent with submit_sweep.sh's own rationale
# (avoids repeated pixi-env-resolution overhead per invocation), even though
# this script never loops over dozens of launches the way that one did.
REPO="${REPO:-$HOME/parrot-neuro}"
if ! command -v python >/dev/null 2>&1 || ! python -c "import optuna" >/dev/null 2>&1; then
    PIXI="$(command -v pixi || true)"
    [ -z "$PIXI" ] && [ -x "$HOME/.pixi/bin/pixi" ] && PIXI="$HOME/.pixi/bin/pixi"
    [ -n "$PIXI" ] || { echo "ERROR: neither a pixi 'python' with optuna nor 'pixi' itself found on PATH"; exit 1; }
    set +u
    eval "$(cd "$REPO" && "$PIXI" shell-hook)"
    set -u
    python -c "import optuna" >/dev/null 2>&1 || { echo "ERROR: optuna still not importable after pixi shell-hook -- run 'pixi install' (adds it to pixi.toml)"; exit 1; }
fi

# Login-node RLIMIT_NPROC pressure from OpenBLAS spinning up a thread per
# core (see README.md) applies here too -- these are quick one-shot calls,
# not a persistent process, but no reason to risk it.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CMD="${1:-}"
case "$CMD" in
    create)
        mkdir -p "$(dirname "$OPTUNA_STORAGE")"
        python - "$OPTUNA_STUDY_NAME" "$OPTUNA_STORAGE" <<'PY'
import sys
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

study_name, storage_path = sys.argv[1], sys.argv[2]
storage = JournalStorage(JournalFileBackend(storage_path))
try:
    optuna.create_study(study_name=study_name, storage=storage, direction="minimize", load_if_exists=False)
except optuna.exceptions.DuplicatedStudyError:
    print(f"ERROR: study '{study_name}' already exists at {storage_path} -- rm it (or use a "
          f"different OPTUNA_STUDY_NAME) to register a new one", file=sys.stderr)
    sys.exit(1)
print(f"[create] study '{study_name}' registered at {storage_path}")
PY
        ;;

    smoke)
        export_run_vars
        subject="${2:-${SUBJECT:-010002}}"
        export OPTUNA_SUBJECTS="$subject" OPTUNA_NUM_EPOCHS=2 OPTUNA_SKIP_DIAGNOSTICS=1
        echo "[smoke] ONE foreground trial: study=$OPTUNA_STUDY_NAME subject=$subject epochs=2, no diagnostics -- sanity check only"
        sbatch --wait --account="$ACCT" --job-name="$JOB_NAME-smoke" \
            --partition="$BOOST_PART" --qos="$DEBUG_QOS" --gres=gpu:1 \
            --cpus-per-task=8 --time=00:30:00 --mem=32G --export=ALL \
            "$SCRIPT_DIR/optuna_train.sbatch"
        ;;

    start)
        n_concurrent="${2:-8}"
        count="${3:-5}"
        total=$((n_concurrent * count))
        export_run_vars

        if [ -n "${OPTUNA_GPUS:-}" ]; then
            gpu_count=$(( $(printf '%s' "$OPTUNA_GPUS" | tr -cd ',' | wc -c) + 1 ))
            qos="${OPTUNA_QOS:-normal}"
            time="${OPTUNA_TIME:-20:00:00}"
        else
            gpu_count=1
            qos="${OPTUNA_QOS:-$PILOT_QOS}"
            time="${OPTUNA_TIME:-2-00:00:00}"
        fi
        cpus="${OPTUNA_CPUS:-$((8 * gpu_count))}"
        mem="${OPTUNA_MEM:-$((64 * gpu_count))G}"

        echo "[start] study=$OPTUNA_STUDY_NAME  $total trials total, throttled to $n_concurrent concurrent"
        echo "[start] resources per trial: gpu:$gpu_count  ${cpus}c  time=$time  mem=$mem  qos=$qos"
        echo "[start] this is ONE sbatch array submission -- no login-node process to keep alive,"
        echo "        safe to submit from a plain shell and log out; 'status'/'stop' work from anywhere."
        jid=$(sbatch --account="$ACCT" --job-name="$JOB_NAME" \
            --partition="$BOOST_PART" --qos="$qos" --gres=gpu:"$gpu_count" \
            --cpus-per-task="$cpus" --time="$time" --mem="$mem" \
            --array="1-${total}%${n_concurrent}" --export=ALL --parsable \
            "$SCRIPT_DIR/optuna_train.sbatch")
        echo "[start] submitted: $jid   Watch: squeue --me   Cancel: scancel -u \$USER --name=$JOB_NAME"
        ;;

    list)
        echo "--- known studies (this checkout) ---"
        found=0
        for f in "$WORKDIR"/parrot/optuna/*.log; do
            [ -f "$f" ] || continue
            found=1
            name="$(basename "$f" .log)"
            python - "$name" "$f" <<'PY' 2>/dev/null || echo "  name=$name  file=$f  (could not read trial counts)"
import sys
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

name, path = sys.argv[1], sys.argv[2]
study = optuna.load_study(study_name=name, storage=JournalStorage(JournalFileBackend(path)))
states = {}
for t in study.trials:
    states[t.state.name] = states.get(t.state.name, 0) + 1
best = None
try:
    best = study.best_value
except ValueError:
    pass
print(f"  name={name}  trials={len(study.trials)} {states}  best={best}")
PY
        done
        [ "$found" = 1 ] || echo "  (none -- run 'create', optionally with OPTUNA_STUDY_NAME=<name> set, first)"
        ;;

    status)
        echo "--- SLURM (this user's $JOB_NAME jobs) ---"
        squeue --me -o '%.10i %.9P %.20j %.8T %.10M %.6D %R' 2>/dev/null | { head -1; grep -F "$JOB_NAME" || echo "  (none)"; }
        echo "--- study trial states (study=$OPTUNA_STUDY_NAME) ---"
        python - "$OPTUNA_STUDY_NAME" "$OPTUNA_STORAGE" <<'PY' 2>/dev/null || echo "  (study not found -- run 'create' first)"
import sys
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

name, path = sys.argv[1], sys.argv[2]
study = optuna.load_study(study_name=name, storage=JournalStorage(JournalFileBackend(path)))
states = {}
for t in study.trials:
    states[t.state.name] = states.get(t.state.name, 0) + 1
print(f"  {len(study.trials)} trial(s): {states}")
try:
    print(f"  best combined_loss so far: {study.best_value}  (trial #{study.best_trial.number})")
except ValueError:
    print("  no completed trials yet")
PY
        ;;

    stop)
        echo "[stop] cancelling all $JOB_NAME (array) jobs for this user..."
        scancel -u "$USER" --name="$JOB_NAME" 2>/dev/null || true
        echo "[stop] done. Trials already RUNNING when cancelled will show up as RUNNING forever in"
        echo "       the study (Optuna has no external liveness check) -- harmless to the search"
        echo "       itself (TPE just ignores stale RUNNING trials when proposing new points), but"
        echo "       if you want them cleaned up: prune manually via optuna's study.trials API."
        ;;

    *)
        cat >&2 <<'EOF'
usage: submit_optuna_sweep.sh <command>

  create                  register the study (once)
  smoke [subject]         ONE foreground trial, 2 epochs, no diagnostics --
                          validates the full ask->train->tell round trip
  start [N] [COUNT]       N*COUNT total trials (default 8 5 = 40) as ONE
                          SLURM array job, throttled to N concurrent
  list                    show every known study (name + trial-state counts)
                          in this checkout's $WORKDIR/parrot/optuna/
  status                  squeue + this study's trial-state counts
  stop                    cancel this study's array job (scancel by job name)

Prefix any command with OPTUNA_STUDY_NAME=<tag> to create/run a SECOND,
independent search without touching an already-running one's state (the
study file, hence the SLURM job name, is namespaced by it; unset = "default").
Both searches still share the same Leonardo account GPU/node/core-hour
budget -- this script has no way to partition that for you.

Config (account/paths/OPTUNA_* resources) is read from
hpc/leonardo/config.local.sh.
EOF
        exit 1 ;;
esac
