#!/bin/bash
###############################################################################
# EEG+BOLD optimization submitter for JUPITER (JSC).
#
# Single source of truth for the SLURM resources (partition / QoS / GPU /
# cores / walltime / mem) of the optimization stage, and for building the
# subject-index job array. Mirrors hpc/leonardo/submit_optim.sh -- same
# commands/semantics -- with JUPITER-specific differences:
#   - typed gres (--gres=gpu:gh200:N, not LEONARDO's untyped --gres=gpu:N)
#   - a single confirmed QoS ("normal", 12h wall cap) instead of LEONARDO's
#     normal(24h)/boost_qos_dbg(30m)/boost_qos_lprod(4d) trio -- see
#     config.local.sh.example's GPU_QOS comment before assuming a longer QoS
#     is usable on your account
#   - `pixi run -e optim` (the optim feature/environment -- see
#     setup_optim_env.sh's header for why: pixi.toml needs restructuring
#     before this resolves on JUPITER's linux-aarch64 nodes)
#
# This is a SEPARATE stage from reconstruction: it runs in a `pixi` env (no
# container), reads a subject's EEG+fMRI+leadfield derivatives (assumed
# already staged onto JUPITER -- there is no JUPITER port of the
# reconstruction pipeline here), and writes fitted-parameter results.
#
# Usage (run each step in order the first time you use this):
#   ./submit_optim.sh smoke [subject]     # ~2 epochs, no diagnostics -- quick sanity check
#   ./submit_optim.sh pilot [subject]     # full hyperparameters, ONE subject, timed + GPU-util logged
#                                          #   -- "how long does a real fit take" (read this before `run`)
#   ./submit_optim.sh run                 # full job array over the cohort
#   ./submit_optim.sh run <subj ...>      # job array over an explicit subgroup (see subj_id.txt)
#   ./submit_optim.sh list                # show the array + resource matrix, submit nothing
#
#   PARROT_DRYRUN=1 ./submit_optim.sh run   # print the sbatch command, submit nothing
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] || { echo "ERROR: hpc/jupiter/config.local.sh not found -- cp it from config.local.sh.example"; exit 1; }
. "$SCRIPT_DIR/config.local.sh"

: "${ACCT:?set ACCT in config.local.sh}"
: "${WORKDIR:?set WORKDIR in config.local.sh}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
PARTICIPANTS="${PARTICIPANTS:-$BIDS/participants.tsv}"
SUBJ_FILE="${SUBJ_FILE:-$WORKDIR/parrot/cohort_subjects.txt}"
OPTIM_OUTPUT_DIR="${OPTIM_OUTPUT_DIR:-$WORKDIR/parrot/eeg_bold_fit_res}"

GPU_PART="${GPU_PART:-booster}"
GPU_GRES_TYPE="${GPU_GRES_TYPE:-gh200}"
GPU_QOS="${GPU_QOS:-normal}"
DEBUG_QOS="${DEBUG_QOS:-$GPU_QOS}"     # no dedicated debug QoS confirmed on this account -- see config example
PILOT_QOS="${PILOT_QOS:-$GPU_QOS}"     # no dedicated long QoS confirmed either -- normal's 12h cap applies
ARRAY_THROTTLE="${ARRAY_THROTTLE:-%40}"
MAX_SUBMIT="${MAX_SUBMIT:-1000}"

# --- fit hyperparameters -- IDENTICAL semantics/defaults to
# hpc/leonardo/submit_optim.sh (same eeg_bold_fit_cli.py); override any of
# these in config.local.sh (OPTIM_ATLAS=..., etc.) or as a call-time env var.
OPTIM_ATLAS="${OPTIM_ATLAS:-1000}"
OPTIM_SPACING="${OPTIM_SPACING:-2.0}"
OPTIM_LEADFIELD_LABEL="${OPTIM_LEADFIELD_LABEL:-duneuroCGAL}"
OPTIM_OPTIMIZE="${OPTIM_OPTIMIZE:-both}"
# bold_model: "hrf" (default, linear HRF-kernel convolution) | "balloon"
# (Friston/Deco Balloon-Windkessel hemodynamic ODE) -- see eeg_bold_fit_cli.py
# --bold-model.
OPTIM_BOLD_MODEL="${OPTIM_BOLD_MODEL:-hrf}"
# schedule: "alternating" (default, original interleaved fit) | "phased"
# (splits OPTIM_NUM_EPOCHS in half: BOLD-only then EEG-only) | "joint" (one
# combined EEG+BOLD loss/step per epoch instead of two separate ones) -- see
# eeg_bold_fit_cli.py --schedule. joint_*_weight only matters for "joint".
OPTIM_SCHEDULE="${OPTIM_SCHEDULE:-alternating}"
OPTIM_JOINT_EEG_WEIGHT="${OPTIM_JOINT_EEG_WEIGHT:-1e5}"
OPTIM_JOINT_BOLD_WEIGHT="${OPTIM_JOINT_BOLD_WEIGHT:-1.0}"
# BOLD loss is a weighted combination of static FC + dFC/FCD (both computed
# from the same simulated trajectory -- see train.make_bold_loss_fn); either
# weight at 0 recovers a single-mode fit.
OPTIM_BOLD_FC_WEIGHT="${OPTIM_BOLD_FC_WEIGHT:-0.5}"
OPTIM_BOLD_DFC_WEIGHT="${OPTIM_BOLD_DFC_WEIGHT:-0.5}"
# dFC sliding-window length/stride in TRs (config.BoldFitConfig.dfc_window_trs/
# dfc_step_trs) -- two of the 7 Optuna/wandb-swept fields.
OPTIM_DFC_WINDOW_TRS="${OPTIM_DFC_WINDOW_TRS:-6}"
OPTIM_DFC_STEP_TRS="${OPTIM_DFC_STEP_TRS:-1}"
OPTIM_NUM_EPOCHS="${OPTIM_NUM_EPOCHS:-300}"
OPTIM_BOLD_EVERY="${OPTIM_BOLD_EVERY:-2}"
OPTIM_EEG_TASK="${OPTIM_EEG_TASK:-eyesclosed}"
OPTIM_FMRI_TASK="${OPTIM_FMRI_TASK:-rest}"
OPTIM_LEARNING_RATE="${OPTIM_LEARNING_RATE:-1e-2}"
# Empty (default) = reuse OPTIM_LEARNING_RATE for the BOLD step too -- EEG and
# BOLD each get their own Adam state, so they can also use different rates.
OPTIM_LEARNING_RATE_BOLD="${OPTIM_LEARNING_RATE_BOLD:-}"
# Optional BOLD spectral-shape term -- 0 (default) = off.
OPTIM_BOLD_PSD_WEIGHT="${OPTIM_BOLD_PSD_WEIGHT:-0}"
# Optional EEG gamma-band term -- 0 (default) = off.
OPTIM_GAMMA_WEIGHT="${OPTIM_GAMMA_WEIGHT:-0}"
# GPU-memory fixes carried over from LEONARDO's tuning (see
# hpc/leonardo/README.md's OOM notes) -- UNVERIFIED on GH200's unified-memory
# architecture specifically; re-measure with `pilot` before trusting these.
OPTIM_T1_WARMUP="${OPTIM_T1_WARMUP:-30000}"
OPTIM_SOLVER_BLOCK_SIZE="${OPTIM_SOLVER_BLOCK_SIZE:-1400}"
# Early stopping (train.is_loss_stalled) -- empty (default) = off. Worth
# enabling here more than on LEONARDO: JUPITER's confirmed QoS caps at 12h
# wall, so a fit that would exceed it needs either this, fewer epochs, or a
# confirmed longer QoS (see config.local.sh.example's GPU_QOS comment).
OPTIM_EARLY_STOP_PATIENCE="${OPTIM_EARLY_STOP_PATIENCE:-}"
OPTIM_EARLY_STOP_WINDOW="${OPTIM_EARLY_STOP_WINDOW:-20}"
OPTIM_EARLY_STOP_MIN_DELTA="${OPTIM_EARLY_STOP_MIN_DELTA:-1e-3}"

# Cohort-array resources. TIME/MEM/CPUS are UNMEASURED defaults -- run `pilot`
# first and set these (in config.local.sh) from what you actually observe.
# Sized conservatively against one GH200 GPU-share's ceiling (~72 cores /
# ~219G out of the node's 288 cores / ~878G across 4 GPUs).
OPTIM_CPUS="${OPTIM_CPUS:-16}"
OPTIM_MEM="${OPTIM_MEM:-128G}"
OPTIM_TIME="${OPTIM_TIME:-10:00:00}"    # MUST stay under GPU_QOS's 12h cap -- see config example

# Noise / initialization seeds (eeg_bold_fit_cli.py --noise-seed / --init-seed).
# Empty = the CLI defaults (noise seed 69, homogeneous LearnableParam.init).
OPTIM_NOISE_SEED="${OPTIM_NOISE_SEED:-}"
OPTIM_INIT_SEED="${OPTIM_INIT_SEED:-}"
OPTIM_INIT_SPREAD="${OPTIM_INIT_SPREAD:-0.5}"
OPTIM_INIT_KEEP="${OPTIM_INIT_KEEP:-}"       # space-separated params kept at default init, e.g. "G"
OPTIM_RUN_TAG="${OPTIM_RUN_TAG:-}"           # suffixes the per-subject JAX cache dir

# `replicates` design (stability across initialization AND noise): one job
# array per (init seed x noise seed) cell, each into its own output root
# ${REPLICATE_OUTPUT_PREFIX}_init-<i>_noise-<n>. "default" = homogeneous init.
# The (default, 69) cell is the plain `run` output, so it is skipped unless
# REPLICATE_INCLUDE_BASE=1. Full crossing separates init from noise variance.
REPLICATE_INIT_SEEDS="${REPLICATE_INIT_SEEDS:-default 1 2 3}"
REPLICATE_NOISE_SEEDS="${REPLICATE_NOISE_SEEDS:-69 1001}"
REPLICATE_INCLUDE_BASE="${REPLICATE_INCLUDE_BASE:-0}"
REPLICATE_OUTPUT_PREFIX="${REPLICATE_OUTPUT_PREFIX:-$OPTIM_OUTPUT_DIR}"
# Whole-node packing: each array element takes a node and runs this many fits
# at once, one per GPU (optim_pack.sbatch). CPUs/mem default to that many
# times the single-fit OPTIM_CPUS/OPTIM_MEM; override if the node is smaller.
REPLICATE_GPUS_PER_NODE="${REPLICATE_GPUS_PER_NODE:-4}"
REPLICATE_NODE_CPUS="${REPLICATE_NODE_CPUS:-}"
REPLICATE_NODE_MEM="${REPLICATE_NODE_MEM:-}"
# Subjects when none are given on the command line (one per line or
# tab-separated; see hpc/pick_replicate_subjects.py).
REPLICATE_SUBJECTS_FILE="${REPLICATE_SUBJECTS_FILE:-}"

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
    export OPTIM_ATLAS OPTIM_SPACING OPTIM_LEADFIELD_LABEL OPTIM_OPTIMIZE OPTIM_BOLD_MODEL \
           OPTIM_SCHEDULE OPTIM_JOINT_EEG_WEIGHT OPTIM_JOINT_BOLD_WEIGHT \
           OPTIM_BOLD_FC_WEIGHT OPTIM_BOLD_DFC_WEIGHT OPTIM_DFC_WINDOW_TRS OPTIM_DFC_STEP_TRS \
           OPTIM_NUM_EPOCHS OPTIM_BOLD_EVERY OPTIM_EEG_TASK OPTIM_FMRI_TASK OPTIM_LEARNING_RATE \
           OPTIM_LEARNING_RATE_BOLD OPTIM_BOLD_PSD_WEIGHT OPTIM_GAMMA_WEIGHT \
           OPTIM_OUTPUT_DIR OPTIM_SOLVER_BLOCK_SIZE OPTIM_T1_WARMUP \
           OPTIM_EARLY_STOP_PATIENCE OPTIM_EARLY_STOP_WINDOW OPTIM_EARLY_STOP_MIN_DELTA \
           OPTIM_NOISE_SEED OPTIM_INIT_SEED OPTIM_INIT_SPREAD OPTIM_INIT_KEEP OPTIM_RUN_TAG
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
              --partition="$GPU_PART" --qos="$DEBUG_QOS" --gres=gpu:"$GPU_GRES_TYPE":1
              --cpus-per-task=8 --time=00:30:00 --mem=32G --export=ALL
              "$SCRIPT_DIR/optim_cohort.sbatch" )
        if [ "$DRYRUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
        ;;

    pilot)
        subject="${2:-${SUBJECT:-010002}}"
        export_run_vars
        export OPTIM_SUBJECT="$subject" OPTIM_SKIP_DIAGNOSTICS=0 OPTIM_GPU_UTIL_LOG=1
        unset OPTIM_SUBJECTS_FILE || true
        echo "[pilot] subject=$subject  qos=$PILOT_QOS  epochs=$OPTIM_NUM_EPOCHS  atlas=$OPTIM_ATLAS  optimize=$OPTIM_OPTIMIZE  schedule=$OPTIM_SCHEDULE"
        echo "[pilot] this is a MEASUREMENT run -- read its walltime + GPU-idle before sizing 'run'."
        echo "[pilot] NOTE: $PILOT_QOS caps at 12h wall on this account -- if the job hits the time"
        echo "        limit before finishing, that IS the measurement: shorten via early stopping /"
        echo "        fewer epochs, or confirm a longer QoS is actually usable first."
        cmd=( sbatch --account="$ACCT" --job-name=parrot-optim-pilot
              --partition="$GPU_PART" --qos="$PILOT_QOS" --gres=gpu:"$GPU_GRES_TYPE":1
              --cpus-per-task="$OPTIM_CPUS" --time=12:00:00 --mem="$OPTIM_MEM" --export=ALL
              "$SCRIPT_DIR/optim_cohort.sbatch" )
        if [ "$DRYRUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
        ;;

    run)
        shift || true
        subjects=( "$@" )

        # Same live-array footgun guard as hpc/leonardo/submit_optim.sh: a
        # targeted retry racing a still-draining full-cohort array can
        # corrupt the shared subjects-file / double-process a subject.
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
            echo "ERROR: $N subjects > MAX_SUBMIT=$MAX_SUBMIT (submit cap). Submit a subset, or raise MAX_SUBMIT in config.local.sh if the real cap is higher." >&2
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
        echo "[run] resources: gpu:gh200:1  ${OPTIM_CPUS}c  time=$OPTIM_TIME  mem=$OPTIM_MEM  qos=$GPU_QOS"
        cmd=( sbatch --account="$ACCT" --job-name=parrot-optim
              --partition="$GPU_PART" --qos="$GPU_QOS" --gres=gpu:"$GPU_GRES_TYPE":1
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

    replicates)
        shift || true
        subjects=( "$@" )
        if [ "${#subjects[@]}" -eq 0 ] && [ -n "$REPLICATE_SUBJECTS_FILE" ]; then
            [ -f "$REPLICATE_SUBJECTS_FILE" ] || { echo "ERROR: REPLICATE_SUBJECTS_FILE=$REPLICATE_SUBJECTS_FILE not found" >&2; exit 1; }
            # one id per line or tab/space-separated (subj_id.txt style); "sub-" optional
            mapfile -t subjects < <(tr '\t ' '\n\n' < "$REPLICATE_SUBJECTS_FILE" | sed '/^$/d')
        fi
        [ "${#subjects[@]}" -gt 0 ] || { echo "ERROR: no subjects -- pass them (./submit_optim.sh replicates <subj ...>) or set REPLICATE_SUBJECTS_FILE (e.g. from hpc/pick_replicate_subjects.py)" >&2; exit 1; }
        G="$REPLICATE_GPUS_PER_NODE"
        TASKS="${SUBJ_FILE%.txt}.replicates.$(date +%Y%m%d-%H%M%S)-$$.tasks"
        mkdir -p "$(dirname "$TASKS")"
        : > "$TASKS"
        roots=()
        for init in $REPLICATE_INIT_SEEDS; do
            for noise in $REPLICATE_NOISE_SEEDS; do
                if [ "$init" = default ] && [ "$noise" = 69 ] && [ "$REPLICATE_INCLUDE_BASE" != 1 ]; then continue; fi
                roots+=( "${REPLICATE_OUTPUT_PREFIX}_init-${init}_noise-${noise}/atlas-$OPTIM_ATLAS" )
            done
        done
        # subject-major: a node mostly runs one subject's refits (similar runtimes, less idle GPU)
        for subj in "${subjects[@]}"; do
            subj="${subj#sub-}"
            for init in $REPLICATE_INIT_SEEDS; do
                for noise in $REPLICATE_NOISE_SEEDS; do
                    if [ "$init" = default ] && [ "$noise" = 69 ] && [ "$REPLICATE_INCLUDE_BASE" != 1 ]; then continue; fi
                    echo "$subj $init $noise ${REPLICATE_OUTPUT_PREFIX}_init-${init}_noise-${noise}" >> "$TASKS"
                done
            done
        done
        NT=$(wc -l < "$TASKS")
        NN=$(( (NT + G - 1) / G ))
        CPUS="${REPLICATE_NODE_CPUS:-$(( G * OPTIM_CPUS ))}"
        MEM="${REPLICATE_NODE_MEM:-$(( G * ${OPTIM_MEM%G} ))G}"
        echo "[replicates] ${#subjects[@]} subjects x $(( NT / ${#subjects[@]} )) refits = $NT fits; $G per node -> $NN node jobs"
        echo "[replicates] init seeds {$REPLICATE_INIT_SEEDS} x noise seeds {$REPLICATE_NOISE_SEEDS}; tasks: $TASKS"
        echo "[replicates] per node: gpu x$G  ${CPUS}c  mem=$MEM  time=$OPTIM_TIME"
        export_run_vars
        export OPTIM_TASKS_FILE="$TASKS" OPTIM_GPUS_PER_NODE="$G" OPTIM_RUNNER="$SCRIPT_DIR/optim_cohort.sbatch" \
               OPTIM_SKIP_DIAGNOSTICS=0 OPTIM_GPU_UTIL_LOG=0
        unset OPTIM_SUBJECT OPTIM_SUBJECTS_FILE || true
        cmd=( sbatch --account="$ACCT" --job-name=parrot-optim-rep
              --partition="$GPU_PART" --qos="$GPU_QOS" --gres=gpu:"$GPU_GRES_TYPE":$G
              --cpus-per-task="$CPUS" --time="$OPTIM_TIME" --mem="$MEM"
              --array="0-$((NN - 1))${ARRAY_THROTTLE}" --export=ALL --parsable
              "$SCRIPT_DIR/optim_pack.sbatch" )
        if [ "$DRYRUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; head -n "$G" "$TASKS" | sed 's/^/    node 0: /'
        else echo "[replicates] submitted: $("${cmd[@]}")  (per-fit logs: ./parrot-optim-rep-logs/)"; fi
        echo "[replicates] when done, add to the population report:"
        echo "    --replicate-roots ${roots[*]}"
        ;;

    list)
        N=$(build_subjects)
        echo "$N subjects -> --array=0-$((N-1))${ARRAY_THROTTLE}  (file: $SUBJ_FILE)"
        printf '  gpu:gh200:1  %sc  time=%s  mem=%s  qos=%s (part=%s)\n' "$OPTIM_CPUS" "$OPTIM_TIME" "$OPTIM_MEM" "$GPU_QOS" "$GPU_PART"
        printf '  atlas=%s  optimize=%s  bold_model=%s  schedule=%s  bold_fc_weight=%s  bold_dfc_weight=%s  dfc_window_trs=%s  dfc_step_trs=%s  epochs=%s  bold_every=%s  t1_warmup=%s  solver_block_size=%s  early_stop_patience=%s\n' \
            "$OPTIM_ATLAS" "$OPTIM_OPTIMIZE" "$OPTIM_BOLD_MODEL" "$OPTIM_SCHEDULE" "$OPTIM_BOLD_FC_WEIGHT" "$OPTIM_BOLD_DFC_WEIGHT" "$OPTIM_DFC_WINDOW_TRS" "$OPTIM_DFC_STEP_TRS" "$OPTIM_NUM_EPOCHS" "$OPTIM_BOLD_EVERY" \
            "${OPTIM_T1_WARMUP:-off}" "${OPTIM_SOLVER_BLOCK_SIZE:-off}" "${OPTIM_EARLY_STOP_PATIENCE:-off}"
        if [ "$OPTIM_SCHEDULE" = "joint" ]; then
            printf '  joint_eeg_weight=%s  joint_bold_weight=%s\n' "$OPTIM_JOINT_EEG_WEIGHT" "$OPTIM_JOINT_BOLD_WEIGHT"
        fi
        printf '  learning_rate_bold=%s  bold_psd_weight=%s  gamma_weight=%s\n' \
            "${OPTIM_LEARNING_RATE_BOLD:-off}" "$OPTIM_BOLD_PSD_WEIGHT" "$OPTIM_GAMMA_WEIGHT"
        echo "  output: $OPTIM_OUTPUT_DIR"
        printf '  seeds: noise=%s  init=%s (spread %s, keep: %s)\n' "${OPTIM_NOISE_SEED:-69}" "${OPTIM_INIT_SEED:-default}" "$OPTIM_INIT_SPREAD" "${OPTIM_INIT_KEEP:-none}"
        printf '  replicates design: init {%s} x noise {%s} -> %s_init-<i>_noise-<n>; %s fits/node; subjects file: %s\n' "$REPLICATE_INIT_SEEDS" "$REPLICATE_NOISE_SEEDS" "$REPLICATE_OUTPUT_PREFIX" "$REPLICATE_GPUS_PER_NODE" "${REPLICATE_SUBJECTS_FILE:-none}"
        ;;

    *)
        cat >&2 <<'EOF'
usage: submit_optim.sh <command>

  smoke [subject]   ONE subject, 2 epochs, no diagnostics
                     ("does the env + pipeline run on a GPU node") -- subject
                     defaults to $SUBJECT (config.local.sh)
  pilot [subject]    ONE subject, full hyperparameters, timed + GPU-util
                     logged -- read this before `run` (see README)
  run [subject ...]  full dependency-free job array over the cohort. No args
                     = every subject in participants.tsv; positional args =
                     just those subjects (small pilot / targeted retry,
                     e.g. `run $(cat subj_id.txt)`)
  replicates [subj ...]
                     stability refits: every subject x (REPLICATE_INIT_SEEDS x
                     REPLICATE_NOISE_SEEDS) cell, packed REPLICATE_GPUS_PER_NODE
                     (default 4) fits per node, one per GPU; each cell writes to
                     ${REPLICATE_OUTPUT_PREFIX}_init-<i>_noise-<n>. Subjects from
                     the args, else REPLICATE_SUBJECTS_FILE
  list               print the cohort array + resource matrix; submit nothing

  PARROT_DRYRUN=1 ...   print the sbatch command instead of submitting

Config (account/paths/partitions/fit hyperparameters) is read from
hpc/jupiter/config.local.sh.
EOF
        exit 1 ;;
esac
