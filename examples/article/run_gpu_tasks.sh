#!/bin/bash
###############################################################################
# Run a task file (one command per line, from `gpu_fit.py tasks --write ...`)
# on a pool of local GPUs: tasks are dealt round-robin into one queue per GPU,
# each queue runs its tasks one after another with CUDA_VISIBLE_DEVICES set.
# A task whose log already ends in a success line is skipped, so the same
# file can be re-run after an interruption.
#
#   examples/article/run_gpu_tasks.sh article_tasks.txt 2,3,4,5,6,7 [log_dir]
#
# Each full fit takes ~9-12 h on one H100 (300 epochs, atlas 100); a
# fresh-noise job ~20 min.
###############################################################################
set -euo pipefail
TASKS="${1:?task file}"
GPUS="${2:?comma-separated GPU ids, e.g. 2,3,4,5}"
LOG_DIR="${3:-article_fits/logs}"
mkdir -p "$LOG_DIR"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
mapfile -t LINES < <(grep -v '^\s*\(#\|$\)' "$TASKS")
echo "[run_gpu_tasks] ${#LINES[@]} tasks on GPUs ${GPU_LIST[*]} -- logs in $LOG_DIR"

run_queue() {
  local gpu="$1"; shift
  for i in "$@"; do
    local log="$LOG_DIR/task_$(printf %03d "$i").log"
    if [ -f "$log" ] && grep -q -E '^\[(fit .*|recovery|fresh-noise)\] (done|->)' "$log"; then
      echo "[gpu $gpu] task $i already done -- skipped"; continue
    fi
    echo "[gpu $gpu] task $i: ${LINES[$i]}"
    echo "# ${LINES[$i]}" > "$log"
    if CUDA_VISIBLE_DEVICES="$gpu" bash -c "${LINES[$i]}" >> "$log" 2>&1; then
      echo "[gpu $gpu] task $i OK"
    else
      echo "[gpu $gpu] task $i FAILED (see $log)"
    fi
  done
}

for g in "${!GPU_LIST[@]}"; do
  idx=()
  for ((i = g; i < ${#LINES[@]}; i += ${#GPU_LIST[@]})); do idx+=("$i"); done
  [ "${#idx[@]}" -gt 0 ] && run_queue "${GPU_LIST[$g]}" "${idx[@]}" &
done
wait
echo "[run_gpu_tasks] all queues finished"
