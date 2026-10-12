#!/bin/bash
###############################################################################
# Packed EEG+BOLD fits for LEONARDO -- several fits on ONE node, one per GPU.
#
# Thin runner, submitted by `submit_optim.sh replicates` (never sbatch it
# directly). Array element k runs lines [k*G+1, k*G+G] of $OPTIM_TASKS_FILE
# (G = $OPTIM_GPUS_PER_NODE) concurrently, each pinned to its own GPU via
# CUDA_VISIBLE_DEVICES, each through the normal single-fit runner
# optim_cohort.sbatch (same args, env, JAX cache handling). Task line format:
#     <subject> <init seed|default> <noise seed> <output dir>
# Per-fit logs go to $OPTIM_PACK_LOGDIR; the job fails if any fit failed.
###############################################################################

#SBATCH --job-name=parrot-optim-rep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=parrot-optim-rep-%A_%a.out
#SBATCH --error=parrot-optim-rep-%A_%a.err

set -uo pipefail

: "${OPTIM_TASKS_FILE:?OPTIM_TASKS_FILE not set -- submit via submit_optim.sh replicates}"
: "${OPTIM_RUNNER:?OPTIM_RUNNER (path to optim_cohort.sbatch) not set}"
G="${OPTIM_GPUS_PER_NODE:-4}"
LOGDIR="${OPTIM_PACK_LOGDIR:-${SLURM_SUBMIT_DIR:-$PWD}/parrot-optim-rep-logs}"
mkdir -p "$LOGDIR"
first=$(( ${SLURM_ARRAY_TASK_ID:-0} * G ))

echo "=== packed fits: job ${SLURM_JOB_ID:-?} (array ${SLURM_ARRAY_JOB_ID:--}_${SLURM_ARRAY_TASK_ID:--}) on $(hostname), $(date) ==="
nvidia-smi -L 2>/dev/null || echo "(no GPU visible)"

pids=(); names=(); logs=()
for g in $(seq 0 $((G - 1))); do
    line="$(sed -n "$((first + g + 1))p" "$OPTIM_TASKS_FILE")"
    [ -n "$line" ] || break
    read -r subj init noise outdir <<< "$line"
    tag="init-${init}_noise-${noise}"
    log="$LOGDIR/${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}_sub-${subj}_${tag}.log"
    (
        export CUDA_VISIBLE_DEVICES="$g"
        export OPTIM_TASK_SUBJECT="$subj" OPTIM_TASK_RUN_TAG="$tag" OPTIM_TASK_NOISE_SEED="$noise" \
               OPTIM_TASK_OUTPUT_DIR="$outdir" OPTIM_TASK_INIT_SEED="$([ "$init" = default ] || echo "$init")"
        unset SLURM_ARRAY_TASK_ID
        exec bash "$OPTIM_RUNNER"
    ) > "$log" 2>&1 &
    pids+=( $! ); names+=( "sub-$subj $tag (GPU $g)" ); logs+=( "$log" )
    echo "started sub-$subj $tag on GPU $g -> $log"
done
[ "${#pids[@]}" -gt 0 ] || { echo "ERROR: no task lines from $((first + 1)) in $OPTIM_TASKS_FILE"; exit 1; }

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "OK    ${names[$i]}"
    else echo "FAIL  ${names[$i]}  (see ${logs[$i]})"; rc=1; fi
done
echo "=== packed fits finished ($(date)), rc=$rc ==="
exit $rc
