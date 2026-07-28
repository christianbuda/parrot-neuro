#!/bin/bash
###############################################################################
# EEG+BOLD optimization submitter for CINECA LEONARDO.
#
# Single source of truth for the SLURM resources (partition / QoS / GPU /
# cores / walltime / mem) of the optimization stage, and for building the
# subject-index job array. optim_cohort.sbatch is a thin runner that just
# does what it's told (same split as submit_cohort.sh / cohort.sbatch for the
# reconstruction pipeline).
#
# This is a SEPARATE stage from reconstruction: it runs in a `pixi` env (no
# container), reads a subject's EEG+fMRI+leadfield derivatives, and writes
# fitted-parameter results -- it does not touch the .sif cache.
#
# Usage (run each step in order the first time you use this):
#   ./submit_optim.sh smoke [subject]     # ~2 epochs, debug QoS, no diagnostics
#                                          #   -- "does the env + pipeline even run on a GPU node"
#   ./submit_optim.sh pilot [subject]     # full hyperparameters, ONE subject, timed + GPU-util logged
#                                          #   -- "how long does a real fit take" (read this before `run`)
#   ./submit_optim.sh run                 # full job array over the cohort
#   ./submit_optim.sh list                # show the array + resource matrix, submit nothing
#
#   PARROT_DRYRUN=1 ./submit_optim.sh run   # print the sbatch command, submit nothing
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] || { echo "ERROR: hpc/leonardo/config.local.sh not found -- cp it from config.local.sh.example"; exit 1; }
. "$SCRIPT_DIR/config.local.sh"

: "${ACCT:?set ACCT in config.local.sh}"
: "${WORKDIR:?set WORKDIR in config.local.sh}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
PARTICIPANTS="${PARTICIPANTS:-$BIDS/participants.tsv}"
# Reuses the SAME subject list as the reconstruction cohort (per-account
# choice: only reconstructed subjects are candidates for the fit anyway).
SUBJ_FILE="${SUBJ_FILE:-$WORKDIR/parrot/cohort_subjects.txt}"
OPTIM_OUTPUT_DIR="${OPTIM_OUTPUT_DIR:-$WORKDIR/parrot/eeg_bold_fit_res}"

BOOST_PART="${BOOST_PART:-boost_usr_prod}"
BOOST_QOS="${BOOST_QOS:-normal}"
DEBUG_QOS="${DEBUG_QOS:-boost_qos_dbg}"
PILOT_QOS="${PILOT_QOS:-boost_qos_lprod}"    # 4-day wall: unmeasured, so start generous
ARRAY_THROTTLE="${ARRAY_THROTTLE:-%40}"
MAX_SUBMIT="${MAX_SUBMIT:-1000}"

# --- fit hyperparameters -- defaults mirror examples/eeg_bold_fit_new.py ----
# (and eeg_bold_fit_cli.py's own argparse defaults); override any of these in
# config.local.sh (OPTIM_ATLAS=..., etc.) or as a call-time env var.
OPTIM_ATLAS="${OPTIM_ATLAS:-1000}"
OPTIM_SPACING="${OPTIM_SPACING:-2.0}"
OPTIM_LEADFIELD_LABEL="${OPTIM_LEADFIELD_LABEL:-duneuroCGAL}"
OPTIM_OPTIMIZE="${OPTIM_OPTIMIZE:-both}"
OPTIM_BOLD_LOSS="${OPTIM_BOLD_LOSS:-fc}"
OPTIM_NUM_EPOCHS="${OPTIM_NUM_EPOCHS:-300}"
OPTIM_BOLD_EVERY="${OPTIM_BOLD_EVERY:-2}"
OPTIM_EEG_TASK="${OPTIM_EEG_TASK:-eyesclosed}"
OPTIM_FMRI_TASK="${OPTIM_FMRI_TASK:-rest}"
OPTIM_LEARNING_RATE="${OPTIM_LEARNING_RATE:-1e-2}"
# Unset (default) = monolithic scan, O(n_steps) backward-pass GPU memory --
# dominated by the long BOLD horizon. Set to an int (K ~ sqrt(n_steps), e.g.
# ~565 for the default t1_bold=320000ms @ dt=1.0ms) to checkpoint the scan and
# fit within the A100's 64G if you're hitting OOM (~1.3-1.7x more compute,
# exact gradient either way -- see network.build_network's docstring).
OPTIM_SOLVER_BLOCK_SIZE="${OPTIM_SOLVER_BLOCK_SIZE:-}"

# Cohort-array resources. TIME/MEM/CPUS are UNMEASURED defaults -- run `pilot`
# first and set these (in config.local.sh) from what you actually observe;
# see the README "read the pilot" section. Billing is per-GPU on Booster, so
# this always takes exactly one.
OPTIM_CPUS="${OPTIM_CPUS:-8}"
OPTIM_MEM="${OPTIM_MEM:-64G}"
OPTIM_TIME="${OPTIM_TIME:-08:00:00}"

DRYRUN="${PARROT_DRYRUN:-0}"

build_subjects() {
    mkdir -p "$(dirname "$SUBJ_FILE")"
    if [ "$#" -gt 0 ]; then
        printf '%s\n' "$@" | sed 's#^sub-##' > "$SUBJ_FILE"
    else
        [ -f "$PARTICIPANTS" ] || { echo "ERROR: participants.tsv not found at $PARTICIPANTS" >&2; exit 1; }
        awk -F'\t' 'NR>1 && $1!="" { id=$1; sub(/^sub-/,"",id); print id }' "$PARTICIPANTS" > "$SUBJ_FILE"
    fi
    N=$(wc -l < "$SUBJ_FILE")
    [ "$N" -gt 0 ] || { echo "ERROR: no subjects to run" >&2; exit 1; }
    echo "$N"
}

# Exports every OPTIM_* + resource var so `--export=ALL` propagates them.
export_run_vars() {
    export OPTIM_ATLAS OPTIM_SPACING OPTIM_LEADFIELD_LABEL OPTIM_OPTIMIZE OPTIM_BOLD_LOSS \
           OPTIM_NUM_EPOCHS OPTIM_BOLD_EVERY OPTIM_EEG_TASK OPTIM_FMRI_TASK OPTIM_LEARNING_RATE \
           OPTIM_OUTPUT_DIR OPTIM_SOLVER_BLOCK_SIZE
}

CMD="${1:-}"
case "$CMD" in
    smoke)
        subject="${2:-${SUBJECT:-010002}}"
        export_run_vars
        export OPTIM_SUBJECT="$subject" OPTIM_NUM_EPOCHS=2 OPTIM_SKIP_DIAGNOSTICS=1 OPTIM_GPU_UTIL_LOG=0
        unset OPTIM_SUBJECTS_FILE || true
        echo "[smoke] subject=$subject  qos=$DEBUG_QOS  epochs=2 (diagnostics skipped) -- sanity check only"
        cmd=( sbatch --account="$ACCT" --job-name=parrot-optim-smoke
              --partition="$BOOST_PART" --qos="$DEBUG_QOS" --gres=gpu:1
              --cpus-per-task=4 --time=00:30:00 --mem=32G --export=ALL
              "$SCRIPT_DIR/optim_cohort.sbatch" )
        if [ "$DRYRUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
        ;;

    pilot)
        subject="${2:-${SUBJECT:-010002}}"
        export_run_vars
        export OPTIM_SUBJECT="$subject" OPTIM_SKIP_DIAGNOSTICS=0 OPTIM_GPU_UTIL_LOG=1
        unset OPTIM_SUBJECTS_FILE || true
        echo "[pilot] subject=$subject  qos=$PILOT_QOS  epochs=$OPTIM_NUM_EPOCHS  atlas=$OPTIM_ATLAS  optimize=$OPTIM_OPTIMIZE"
        echo "[pilot] this is a MEASUREMENT run -- read its walltime + GPU-idle before sizing 'run'"
        cmd=( sbatch --account="$ACCT" --job-name=parrot-optim-pilot
              --partition="$BOOST_PART" --qos="$PILOT_QOS" --gres=gpu:1
              --cpus-per-task="$OPTIM_CPUS" --time=2-00:00:00 --mem="$OPTIM_MEM" --export=ALL
              "$SCRIPT_DIR/optim_cohort.sbatch" )
        if [ "$DRYRUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
        ;;

    run)
        shift || true
        subjects=( "$@" )

        # Same live-array footgun guard as submit_cohort.sh: a targeted retry
        # racing a still-draining full-cohort array can corrupt the shared
        # subjects-file / double-process a subject.
        if [ "${#subjects[@]}" -gt 0 ] && command -v squeue >/dev/null 2>&1; then
            live=$(squeue --me -h -o '%j' 2>/dev/null | grep -c '^parrot-optim' || true)
            if [ "${live:-0}" -gt 0 ] && [ "${PARROT_FORCE:-0}" != 1 ]; then
                echo "ERROR: $live parrot-optim* task(s) still queued/running; refusing a targeted retry" >&2
                echo "       until the cohort drains, or set PARROT_FORCE=1 to override." >&2
                exit 1
            fi
        fi

        N=$(build_subjects "${subjects[@]+"${subjects[@]}"}")
        if [ "$N" -gt "$MAX_SUBMIT" ]; then
            echo "ERROR: $N subjects > MAX_SUBMIT=$MAX_SUBMIT (QOS submit cap). Submit a subset, or raise MAX_SUBMIT in config.local.sh if the real cap is higher." >&2
            exit 1
        fi
        ARR="0-$((N - 1))${ARRAY_THROTTLE}"

        SUBJ_FILE_ACTIVE="${SUBJ_FILE%.txt}.optim.$(date +%Y%m%d-%H%M%S)-$$.txt"
        cp "$SUBJ_FILE" "$SUBJ_FILE_ACTIVE"
        echo "[run] subject snapshot: $SUBJ_FILE_ACTIVE"

        export_run_vars
        export OPTIM_SUBJECTS_FILE="$SUBJ_FILE_ACTIVE" OPTIM_SKIP_DIAGNOSTICS=0 OPTIM_GPU_UTIL_LOG=0
        unset OPTIM_SUBJECT || true

        src=$([ "${#subjects[@]}" -gt 0 ] && echo "subset (${subjects[*]})" || echo "$PARTICIPANTS")
        echo "[run] $N subjects from $src  ->  --array=$ARR"
        echo "[run] resources: gpu:1  ${OPTIM_CPUS}c  time=$OPTIM_TIME  mem=$OPTIM_MEM  qos=$BOOST_QOS"
        cmd=( sbatch --account="$ACCT" --job-name=parrot-optim
              --partition="$BOOST_PART" --qos="$BOOST_QOS" --gres=gpu:1
              --cpus-per-task="$OPTIM_CPUS" --time="$OPTIM_TIME" --mem="$OPTIM_MEM"
              --array="$ARR" --export=ALL --parsable
              "$SCRIPT_DIR/optim_cohort.sbatch" )
        if [ "$DRYRUN" = 1 ]; then
            printf '%q ' "${cmd[@]}"; echo
        else
            jid=$("${cmd[@]}")
            echo "[run] submitted: $jid  Watch: squeue --me ; cancel: scancel -u \$USER --name=parrot-optim"
        fi
        ;;

    list)
        N=$(build_subjects)
        echo "$N subjects -> --array=0-$((N-1))${ARRAY_THROTTLE}  (file: $SUBJ_FILE)"
        printf '  gpu:1  %sc  time=%s  mem=%s  qos=%s (part=%s)\n' "$OPTIM_CPUS" "$OPTIM_TIME" "$OPTIM_MEM" "$BOOST_QOS" "$BOOST_PART"
        printf '  atlas=%s  optimize=%s  bold_loss=%s  epochs=%s  bold_every=%s  solver_block_size=%s\n' \
            "$OPTIM_ATLAS" "$OPTIM_OPTIMIZE" "$OPTIM_BOLD_LOSS" "$OPTIM_NUM_EPOCHS" "$OPTIM_BOLD_EVERY" \
            "${OPTIM_SOLVER_BLOCK_SIZE:-off}"
        echo "  output: $OPTIM_OUTPUT_DIR"
        ;;

    *)
        cat >&2 <<'EOF'
usage: submit_optim.sh <command>

  smoke [subject]   ONE subject, 2 epochs, debug QoS, no diagnostics
                     ("does the env + pipeline run on a GPU node") -- subject
                     defaults to $SUBJECT (config.local.sh)
  pilot [subject]    ONE subject, full hyperparameters, timed + GPU-util
                     logged -- read this before `run` (see README)
  run [subject ...]  full dependency-free job array over the cohort. No args
                     = every subject in participants.tsv; positional args =
                     just those subjects (small pilot / targeted retry)
  list               print the cohort array + resource matrix; submit nothing

  PARROT_DRYRUN=1 ...   print the sbatch command instead of submitting

Config (account/paths/partitions/fit hyperparameters) is read from
hpc/leonardo/config.local.sh.
EOF
        exit 1 ;;
esac
