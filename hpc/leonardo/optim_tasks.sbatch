#!/bin/bash
###############################################################################
# Packed article GPU tasks for LEONARDO -- several on ONE node, one per GPU.
#
# Thin runner, submitted by `submit_optim.sh article` (never sbatch it
# directly). Array element k runs lines [k*G+1, k*G+G] of $OPTIM_TASKS_FILE
# (G = $OPTIM_GPUS_PER_NODE) concurrently, each pinned to its own GPU via
# CUDA_VISIBLE_DEVICES. Task lines are what `gpu_fit.py tasks` writes:
#     pixi run python examples/article/gpu_fit.py <fit|recovery|fresh-noise> ...
# run here as `pixi run --as-is python -u ...` from $REPO (compute nodes are
# offline). A task whose log already ends in a success line is skipped, so
# resubmitting the same task file only reruns what failed or never finished.
# Per-task logs go to $OPTIM_PACK_LOGDIR/task_<NNN>.log; the job fails if any
# task failed.
###############################################################################

#SBATCH --job-name=parrot-optim-art
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=parrot-optim-art-%A_%a.out
#SBATCH --error=parrot-optim-art-%A_%a.err

set -uo pipefail

: "${OPTIM_TASKS_FILE:?OPTIM_TASKS_FILE not set -- submit via submit_optim.sh article}"
: "${WORKDIR:?WORKDIR not set -- submit via submit_optim.sh article}"
REPO="${REPO:-$HOME/parrot-neuro}"
G="${OPTIM_GPUS_PER_NODE:-4}"
LOGDIR="${OPTIM_PACK_LOGDIR:-${SLURM_SUBMIT_DIR:-$PWD}/parrot-optim-art-logs}"
mkdir -p "$LOGDIR"
first=$(( ${SLURM_ARRAY_TASK_ID:-0} * G ))

PIXI="$(command -v pixi || true)"
[ -z "$PIXI" ] && [ -x "$HOME/.pixi/bin/pixi" ] && PIXI="$HOME/.pixi/bin/pixi"
[ -n "$PIXI" ] || { echo "ERROR: pixi not found (on PATH or at \$HOME/.pixi/bin) -- run setup_optim_env.sh"; exit 1; }
[ -d "$REPO/.pixi" ] || { echo "ERROR: $REPO/.pixi missing -- run setup_optim_env.sh on a login node"; exit 1; }

# same memory settings as optim_cohort.sbatch
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "=== packed article tasks: job ${SLURM_JOB_ID:-?} (array ${SLURM_ARRAY_JOB_ID:--}_${SLURM_ARRAY_TASK_ID:--}) on $(hostname), $(date) ==="
nvidia-smi -L 2>/dev/null || echo "(no GPU visible)"

mapfile -t LINES < <(grep -v '^\s*\(#\|$\)' "$OPTIM_TASKS_FILE")
pids=(); names=(); logs=()
for g in $(seq 0 $((G - 1))); do
    i=$((first + g))
    [ "$i" -lt "${#LINES[@]}" ] || break
    line="${LINES[$i]}"
    log="$LOGDIR/task_$(printf %03d "$i").log"
    if [ -f "$log" ] && grep -q -E '^\[(fit .*|recovery|fresh-noise)\] (done|->)' "$log"; then
        echo "SKIP  task $i (already done: $log)"; continue
    fi
    # gpu_fit.py args after the `pixi run python` prefix (paths have no spaces)
    read -ra argv <<< "${line#pixi run python }"
    # own JAX cache per task: concurrent compiles must not share one
    tag="$(printf %03d "$i")"
    (
        export CUDA_VISIBLE_DEVICES="$g"
        export PARROT_JAX_CACHE_DIR="$WORKDIR/parrot/.jax_cache/article-task-$tag"
        mkdir -p "$PARROT_JAX_CACHE_DIR"
        echo "# $line"
        cd "$REPO" && exec "$PIXI" run --as-is python -u "${argv[@]}"
    ) > "$log" 2>&1 &
    pids+=( $! ); names+=( "task $i (GPU $g): ${line#pixi run python examples/article/gpu_fit.py }" ); logs+=( "$log" )
    echo "started task $i on GPU $g -> $log"
done
[ "${#pids[@]}" -gt 0 ] || { echo "nothing to run for lines $((first + 1))-$((first + G)) (all done or past the end)"; exit 0; }

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "OK    ${names[$i]}"
    else echo "FAIL  ${names[$i]}  (see ${logs[$i]})"; rc=1; fi
done
echo "=== packed article tasks finished ($(date)), rc=$rc ==="
exit $rc
