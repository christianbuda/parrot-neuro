#!/bin/bash
###############################################################################
# Parrot cohort submitter for CINECA LEONARDO.
#
# Single source of truth for the A1/A2/B/C/D chunk map: which stages each chunk
# runs, and the SLURM resources (partition / QoS / GPU / cores / walltime / mem)
# each gets. It generates the job array from participants.tsv and wires the
# per-subject dependency chain with `--dependency=aftercorr` (element-wise:
# array task i of one chunk waits on task i of its predecessor -- NOT the whole
# array). cohort.sbatch is a thin runner that just does what it's told.
#
# Chunk DAG (per subject):
#
#     A1 --aftercorr--> A2 --aftercorr--> +--> B (DWI) --+
#     (GPU DL)          (ANTs)            |              +--aftercorr--> D
#                                         +--> C (mesh) -+   (solvers)
#
# EVERYTHING runs on Booster (boost_usr_prod/normal): AIFAC accounts are Booster-only
# allocations with NO dcgp/CPU budget (a dcgp submit -> "invalid account or expired
# budget"), so the CPU chunks run on GPU nodes. Billing is by allocated CORES (no
# TRESBillingWeights), so this fits budget; the cost is idle/stranded GPUs on the 32c
# CPU chunks -- see the citizenship note by set_chunk.
#
#   A1  ingest,fastsurfer,hippunfold        gpu:1  8c   (only chunk that uses the GPU)
#   A2  mne..tissuelabels (+bigbrain)       32c ~16h    (ANTs poles; bigbrain ~6h local)
#   B   qsiprep,qsirecon,connectivity,      32c ~4h     (DWI; qsiprep is CPU-eddy, GPU
#       dwitensor,dwi2t1                                 confirmed 100% idle -> no GPU)
#   C   electrodes,dipoles,tetmesh          4c ~10h     (dipoles single-threaded ->
#                                                        min cores; billing=cores)
#   D   anisotropy,forwardsolvers,          32c ~4h     (join of B and C)
#       artifacts,qc
#
# Usage:
#   ./submit_cohort.sh smoke <chunk> [subject]   # one subject, one chunk, no deps
#   ./submit_cohort.sh run                        # full dependency-chained array
#   ./submit_cohort.sh list                       # show the array + chunk matrix, submit nothing
#
#   PARROT_DRYRUN=1 ./submit_cohort.sh run        # print the sbatch commands, submit nothing
#
# Every chunk stays under the 24h `normal` QoS (NOT the concurrency-capped lprod).
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] || { echo "ERROR: hpc/leonardo/config.local.sh not found -- cp it from config.local.sh.example"; exit 1; }
. "$SCRIPT_DIR/config.local.sh"

: "${ACCT:?set ACCT in config.local.sh}"
: "${WORKDIR:?set WORKDIR in config.local.sh}"
BIDS="${BIDS:-$WORKDIR/parrot/bids}"
PARTICIPANTS="${PARTICIPANTS:-$BIDS/participants.tsv}"
SUBJ_FILE="${SUBJ_FILE:-$WORKDIR/parrot/cohort_subjects.txt}"

# Partition / QoS names are config-overridable: dcgp is untested for this account,
# so if the names differ set BOOST_PART/DCGP_PART/*_QOS in config.local.sh.
BOOST_PART="${BOOST_PART:-boost_usr_prod}"
DCGP_PART="${DCGP_PART:-dcgp_usr_prod}"
BOOST_QOS="${BOOST_QOS:-normal}"
DCGP_QOS="${DCGP_QOS:-normal}"
# Be a good cluster citizen: cap how many array elements run at once.
ARRAY_THROTTLE="${ARRAY_THROTTLE:-%40}"
# QOS submit cap (pending+running tasks per user). SLURM counts each ARRAY TASK,
# not each array, so the full a1..d chain = 5*N tasks. LEONARDO's cap is ~1000
# (measured: 4*227=908 accepted, the 5th array rejected). We refuse to submit
# rather than trip QOSMaxSubmitJobPerUserLimit mid-chain (which leaves a headless,
# partially-submitted DAG). Override if `sacctmgr show qos` reveals the real number.
MAX_SUBMIT="${MAX_SUBMIT:-1000}"

DRYRUN="${PARROT_DRYRUN:-0}"

# --- chunk dependency DAG (per subject) --------------------------------------
# Predecessors of each chunk. submit_cohort.sh wires `aftercorr` ONLY between
# chunks present in the current --chunks selection; a predecessor NOT selected is
# assumed already-complete on disk (the pipeline's per-stage log-guards enforce
# that its outputs exist), so the successor is submitted with NO dependency. That
# is what lets you PHASE the DAG under the submit cap: e.g. `run --chunks a1,a2,b,c`
# (908 tasks), wait for B+C to finish, then `run --chunks d` (D finds its inputs).
CHUNK_ORDER=(a1 a2 b c d)
declare -A DEP=( [a1]="" [a2]="a1" [b]="a2" [c]="a2" [d]="b c" )

# --- chunk matrix -------------------------------------------------------------
# Sets STAGES/GPUS/PART/QOS/GRES/CPUS/TIME/MEM for a chunk. Walltimes are grounded
# in the DWI-run timings (see cineca-leonardo memory). Core counts are the FEWEST
# that fit 24h: billing == allocated cores (no TRESBillingWeights), so with sublinear
# scaling more cores only costs more. A2 is pinned at 32c (boost node max) because
# bigbrain won't fit 24h at fewer.
# CITIZENSHIP: the 32c CPU chunks (A2/B/D) take a whole Booster node's cores while
# using no GPU -> 4 idle A100s per job. Unavoidable while AIFAC has no dcgp/CPU budget.
# If a CPU allocation appears, set PART/QOS back to $DCGP_PART/$DCGP_QOS for these.
set_chunk() {
    GRES=""
    case "$1" in
        a1) STAGES="ingest,fastsurfer,hippunfold"
            GPUS="all";  PART="$BOOST_PART"; QOS="$BOOST_QOS"; GRES="gpu:1"
            CPUS=8;   TIME="03:00:00"; MEM="64G" ;;                      # ~1.4h measured
        a2) STAGES="mne,schaefer,freesurfersubcortical,simnibscharm,fslfirst,synthstrip,cerebellum,bigbrain,surfaces,atlas,tissuelabels"
            GPUS="none"; PART="$BOOST_PART"; QOS="$BOOST_QOS"
            CPUS=32;  TIME="24:00:00"; MEM="240G" ;;                     # ANTs poles ~16h@32c (bigbrain ~6h local); billing=cores so 32c is cheapest that fits 24h
        b)  STAGES="qsiprep,qsirecon,connectivity,dwitensor,dwi2t1"
            GPUS="none"; PART="$BOOST_PART"; QOS="$BOOST_QOS"
            CPUS=32;  TIME="10:00:00"; MEM="240G" ;;                     # ~4.2h measured @32c, GPU-idle
        c)  STAGES="electrodes,dipoles,tetmesh"
            GPUS="none"; PART="$BOOST_PART"; QOS="$BOOST_QOS"
            CPUS=4;   TIME="16:00:00"; MEM="32G" ;;                      # ~9.8h, dipoles single-threaded -> min cores (dcgp shares nodes)
        d)  STAGES="anisotropy,forwardsolvers,artifacts,qc"
            GPUS="none"; PART="$BOOST_PART"; QOS="$BOOST_QOS"
            CPUS=32;  TIME="12:00:00"; MEM="240G" ;;                     # forwardsolvers ~3.9h @32c
        *)  echo "ERROR: unknown chunk '$1' (want a1|a2|b|c|d)"; exit 1 ;;
    esac
}

# --- build the subjects list -------------------------------------------------
# One participant label per line (leading `sub-` stripped). Array index -> line.
# With explicit args -> that SUBSET (for a small pilot run or a targeted retry);
# otherwise the full cohort from participants.tsv. Errors -> stderr so they don't
# get swallowed into the captured count ($(build_subjects)).
build_subjects() {
    mkdir -p "$(dirname "$SUBJ_FILE")"
    if [ "$#" -gt 0 ]; then
        printf '%s\n' "$@" | sed 's#^sub-##' > "$SUBJ_FILE"       # explicit subset
    else
        [ -f "$PARTICIPANTS" ] || { echo "ERROR: participants.tsv not found at $PARTICIPANTS" >&2; exit 1; }
        awk -F'\t' 'NR>1 && $1!="" { id=$1; sub(/^sub-/,"",id); print id }' "$PARTICIPANTS" > "$SUBJ_FILE"
    fi
    N=$(wc -l < "$SUBJ_FILE")
    [ "$N" -gt 0 ] || { echo "ERROR: no subjects to run" >&2; exit 1; }
    echo "$N"
}

# --- submit one chunk. Extra args (--array, --dependency, --parsable) forwarded.
# Exports the run vars into THIS shell so `--export=ALL` propagates them (SLURM's
# --export=VAR=val splits on commas, which our comma-separated STAGES would break).
submit_chunk() {
    local chunk="$1"; shift
    set_chunk "$chunk"
    export PARROT_STAGES="$STAGES" PARROT_GPUS="$GPUS" PARROT_SUBJECTS_FILE="${SUBJ_FILE_ACTIVE:-$SUBJ_FILE}"
    local cmd=( sbatch
        --account="$ACCT" --job-name="parrot-$chunk"
        --partition="$PART" --qos="$QOS"
        --cpus-per-task="$CPUS" --time="$TIME" --mem="$MEM"
        --export=ALL )
    [ -n "$GRES" ] && cmd+=( --gres="$GRES" )
    cmd+=( "$@" "$SCRIPT_DIR/cohort.sbatch" )
    if [ "$DRYRUN" = 1 ]; then
        # Print to STDERR so stdout carries only the fake jobid (mirrors real
        # `sbatch --parsable`, whose stdout is just the id we capture into $J*).
        { printf '  PARROT_STAGES=%q\n  ' "$STAGES"; printf '%q ' "${cmd[@]}"; echo; } >&2
        echo "DRYRUN-$chunk"      # fake jobid so the dependency chain still prints
    else
        "${cmd[@]}"
    fi
}

CMD="${1:-}"
case "$CMD" in
    smoke)
        chunk="${2:?usage: smoke <chunk> [subject]}"
        subject="${3:-${SUBJECT:-010002}}"
        set_chunk "$chunk"
        # Export what cohort.sbatch reads (--export=ALL only propagates EXPORTED vars).
        export PARROT_STAGES="$STAGES" PARROT_GPUS="$GPUS" PARROT_SUBJECT="$subject"
        unset PARROT_SUBJECTS_FILE || true
        echo "[smoke] chunk=$chunk subject=$subject part=$PART qos=$QOS gres=${GRES:-none} cpus=$CPUS time=$TIME stages=$STAGES"
        cmd=( sbatch --account="$ACCT" --job-name="parrot-$chunk-smoke"
              --partition="$PART" --qos="$QOS" --cpus-per-task="$CPUS"
              --time="$TIME" --mem="$MEM" --export=ALL )
        [ -n "$GRES" ] && cmd+=( --gres="$GRES" )
        cmd+=( "$SCRIPT_DIR/cohort.sbatch" )
        if [ "$DRYRUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
        ;;

    run)
        shift || true
        # Parse `--chunks a1,a2,...` (which chunks to submit this phase); any other
        # positional args = optional subject subset (small pilot / targeted retry).
        CHUNKS_CSV=""; subjects=()
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --chunks)   CHUNKS_CSV="${2:?--chunks needs a value}"; shift 2 ;;
                --chunks=*) CHUNKS_CSV="${1#*=}"; shift ;;
                *)          subjects+=( "$1" ); shift ;;
            esac
        done
        # Selection: default is the full chain. Validate against the known chunks.
        if [ -n "$CHUNKS_CSV" ]; then IFS=',' read -r -a sel <<< "$CHUNKS_CSV"; else sel=( "${CHUNK_ORDER[@]}" ); fi
        declare -A SELECTED=()
        for ch in "${sel[@]}"; do
            case "$ch" in a1|a2|b|c|d) SELECTED[$ch]=1 ;; *) echo "ERROR: unknown chunk '$ch' in --chunks (want a1|a2|b|c|d)" >&2; exit 1 ;; esac
        done

        # Footgun guard: cohort.sbatch resolves its subject from the subjects file at RUNTIME,
        # so a targeted retry (subject subset) submitted WHILE the original full-cohort arrays are
        # still draining used to silently re-map their still-pending tasks (or abort them) when it
        # overwrote the shared cohort_subjects.txt. The per-run snapshot below removes the
        # corruption, but re-running a subject a live array may still process is its own hazard
        # (double work / racing outputs), so refuse a subset submission while parrot-* tasks are
        # live unless explicitly forced.
        if [ "${#subjects[@]}" -gt 0 ] && command -v squeue >/dev/null 2>&1; then
            live=$(squeue --me -h -o '%j' 2>/dev/null | grep -c '^parrot-' || true)
            if [ "${live:-0}" -gt 0 ] && [ "${PARROT_FORCE:-0}" != 1 ]; then
                {
                  echo "ERROR: $live parrot-* cohort task(s) still queued/running; refusing a targeted"
                  echo "       retry (subjects: ${subjects[*]}) until the cohort drains -- re-running a"
                  echo "       subject a live array may still process risks double work / racing outputs."
                  echo "       Wait for 'squeue --me' to clear, or set PARROT_FORCE=1 to override."
                } >&2
                exit 1
            fi
        fi

        N=$(build_subjects "${subjects[@]+"${subjects[@]}"}")
        ARR="0-$((N - 1))${ARRAY_THROTTLE}"

        # Snapshot the subject list to an IMMUTABLE per-run file and point every chunk of THIS run
        # at it (via SUBJ_FILE_ACTIVE, exported by submit_chunk). This makes each submission immune
        # to a later `run` overwriting the shared cohort_subjects.txt: still-pending tasks keep
        # reading their own snapshot. aftercorr indexing stays consistent because all chunks of this
        # run share the one snapshot.
        SUBJ_FILE_ACTIVE="${SUBJ_FILE%.txt}.$(date +%Y%m%d-%H%M%S)-$$.txt"
        cp "$SUBJ_FILE" "$SUBJ_FILE_ACTIVE"
        echo "[run] subject snapshot: $SUBJ_FILE_ACTIVE  (immune to later cohort_subjects.txt churn)"

        # Submit-cap guard: co-resident tasks = (#selected chunks) * N. Refuse rather
        # than trip QOSMaxSubmitJobPerUserLimit mid-chain (headless partial submit).
        nsel=${#SELECTED[@]}
        if [ $(( nsel * N )) -gt "$MAX_SUBMIT" ]; then
            maxN=$(( MAX_SUBMIT / nsel ))
            sel_csv=$(IFS=,; echo "${sel[*]}")
            {
              echo "ERROR: $nsel chunks x $N subjects = $(( nsel * N )) tasks > MAX_SUBMIT=$MAX_SUBMIT (QOS submit cap)."
              echo "       Options:"
              echo "         - phase the DAG:   run --chunks a1,a2,b,c   then (after B+C COMPLETE)  run --chunks d"
              echo "         - fewer subjects:  run --chunks $sel_csv  with <= $maxN subject labels"
              echo "       (override MAX_SUBMIT in config.local.sh if the real QOS cap is higher.)"
            } >&2
            exit 1
        fi

        src=$([ "${#subjects[@]}" -gt 0 ] && echo "subset (${subjects[*]})" || echo "$PARTICIPANTS")
        echo "[run] $N subjects from $src  ->  --array=$ARR"
        echo "[run] chunks: ${sel[*]}  ($(( nsel * N )) tasks; aftercorr wired only within this selection)"

        # Submit in DAG order; each chunk's dependency = aftercorr on its SELECTED
        # predecessors (unselected predecessors are assumed already done on disk).
        declare -A JID=()
        for ch in "${CHUNK_ORDER[@]}"; do
            [ -n "${SELECTED[$ch]:-}" ] || continue
            deps=""
            for pred in ${DEP[$ch]}; do
                [ -n "${SELECTED[$pred]:-}" ] && deps="${deps:+$deps:}${JID[$pred]}"
            done
            args=( --array="$ARR" --parsable )
            [ -n "$deps" ] && args+=( --dependency="aftercorr:$deps" )
            JID[$ch]=$(submit_chunk "$ch" "${args[@]}")
            if [ -n "$deps" ]; then echo "  $ch = ${JID[$ch]}  (aftercorr:$deps)"; else echo "  $ch = ${JID[$ch]}  (no dep)"; fi
        done
        names=$(printf 'parrot-%s,' "${sel[@]}"); names="${names%,}"
        echo "[run] submitted. Watch: squeue --me ; cancel this phase: scancel -u \$USER --name=$names"
        ;;

    list)
        N=$(build_subjects)
        echo "$N subjects -> --array=0-$((N-1))${ARRAY_THROTTLE}  (file: $SUBJ_FILE)"
        for ch in a1 a2 b c d; do
            set_chunk "$ch"
            printf '  %-3s %-9s %-6s gres=%-6s %3sc %-9s %-5s  %s\n' \
                "$ch" "$PART" "$QOS" "${GRES:-none}" "$CPUS" "$TIME" "$MEM" "$STAGES"
        done
        ;;

    *)
        cat >&2 <<'EOF'
usage: submit_cohort.sh <command>

  smoke <chunk> [subject]   submit ONE subject, ONE chunk, no dependencies
                            chunk = a1|a2|b|c|d  (subject defaults to $SUBJECT)
  run [--chunks LIST] [subject ...]
                            submit dependency-chained arrays (A1 -> A2 -> {B,C} -> D,
                            per-subject aftercorr). No args = full cohort from
                            participants.tsv; positional args = just those subjects
                            (small pilot / targeted retry).
                            --chunks a1,a2,b,c  submits ONLY those chunks (aftercorr
                            wired within the selection; unselected predecessors are
                            assumed already complete on disk). Use it to PHASE the DAG
                            under the QOS submit cap -- see below.
  list                      print the cohort array + chunk/resource matrix; submit nothing

  # Phasing a big cohort under the ~1000-task submit cap (5*227 > 1000):
  #   run --chunks a1,a2,b,c          # phase 1 (908 tasks)
  #   # ...wait until B and C are COMPLETE for all subjects...
  #   run --chunks d                  # phase 2 (D finds its inputs via log-guards)

  PARROT_DRYRUN=1 ...        print the sbatch commands instead of submitting

Config (account/paths/partitions) is read from hpc/leonardo/config.local.sh.
EOF
        exit 1 ;;
esac
