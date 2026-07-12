#!/bin/bash
###############################################################################
# Parrot cohort error triage for CINECA LEONARDO.
#
# Scans the SLURM accounting record for the cohort's chunk arrays
# (parrot-a1..parrot-d) and reports every array TASK that ended in a non-success
# terminal state -- mapped back to its (subject, chunk, jobid) and pointed at the
# logs to drill into.
#
# WHY sacct (not the pipeline logs) is the source of truth: cohort.sbatch does
# `exit $rc` with run_reconstruction's exit code, so a real stage failure surfaces
# as SLURM FAILED -- AND sacct additionally catches what a pipeline log never can
# (OOM kills, walltime TIMEOUT, NODE_FAIL: the job dies before it can log). The
# per-subject pipeline log is for the *why*, printed on request via --tail.
#
# Subject mapping: read from each task's .out (`subject=<ID>`, echoed by the
# runner) so it's tied to the job that actually ran -- robust against the
# cohort_subjects.txt churn (that file is overwritten by the last `run`, so a
# targeted-retry would corrupt an index-based mapping). Falls back to the index
# only when the .out is missing.
#
# Usage:
#   ./check_errors.sh                      # errors since midnight today
#   ./check_errors.sh --since now-3days    # widen the window (sacct --starttime syntax)
#   ./check_errors.sh --min-jobid 48730241 # ignore anything below this id (drops earlier
#                                          #   cancelled smoke/headless jobs from the report)
#   ./check_errors.sh --chunks b,c         # only these chunks
#   ./check_errors.sh --tail 40            # also dump the last 40 lines of each failure
#   ./check_errors.sh --logdir ~/parrot-neuro/hpc/leonardo   # where the .out/.err live
#
# Exit status: 0 = no errored tasks, 1 = at least one (usable in a watch/CI loop).
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] && . "$SCRIPT_DIR/config.local.sh"

WORKDIR="${WORKDIR:-}"
BIDS="${BIDS:-${WORKDIR:+$WORKDIR/parrot/bids}}"
OUTPUT_DIR="${OUTPUT_DIR:-${BIDS:+$BIDS/derivatives}}"
SUBJ_FILE="${SUBJ_FILE:-${WORKDIR:+$WORKDIR/parrot/cohort_subjects.txt}}"

# --- defaults / args ---------------------------------------------------------
SINCE="$(date +%Y-%m-%d)"      # midnight today (sacct --starttime); override with --since
LOGDIR="$PWD"                  # where sbatch dropped parrot-*.out/.err (= the submit dir)
TAIL=0                         # >0: also print that many trailing lines of each failure
CHUNKS="a1,a2,b,c,d"
MINJOB=""                      # drop tasks whose array-master jobid is below this (clock-
                               # independent way to exclude earlier cancelled/smoke jobs)

while [ "$#" -gt 0 ]; do
    case "$1" in
        --since)      SINCE="${2:?}"; shift 2 ;;
        --logdir)     LOGDIR="${2:?}"; shift 2 ;;
        --tail)       TAIL="${2:?}"; shift 2 ;;
        --chunks)     CHUNKS="${2:?}"; shift 2 ;;
        --min-jobid)  MINJOB="${2:?}"; shift 2 ;;
        -h|--help)  sed -n '2,34p' "$0"; exit 0 ;;
        *)          echo "ERROR: unknown arg '$1' (see --help)" >&2; exit 2 ;;
    esac
done

# parrot-a1,parrot-b,... for sacct --name
NAMES="$(printf 'parrot-%s,' ${CHUNKS//,/ })"; NAMES="${NAMES%,}"

# States that are NOT errors: done-ok or still-in-flight. Everything else
# (FAILED/TIMEOUT/OUT_OF_MEMORY/NODE_FAIL/BOOT_FAIL/DEADLINE/CANCELLED/...) is flagged.
is_ok_state() {
    case "$1" in
        COMPLETED|RUNNING|PENDING|REQUEUED|RESIZING|SUSPENDED) return 0 ;;
        *) return 1 ;;
    esac
}

# --- gather ------------------------------------------------------------------
# -X: one row per array task (no .batch/.extern substeps). -P|-n: pipe-delimited, headerless.
declare -A TALLY=()
errors=()                      # "jobname|jobid|state|exitcode"
running=0; pending=0; total=0

while IFS='|' read -r jobid jobname state exitcode; do
    [ -n "$jobid" ] || continue
    # --min-jobid: drop tasks from earlier submissions (cancelled smoke/headless runs).
    if [ -n "$MINJOB" ]; then
        master=${jobid%_*}
        case "$master" in ''|*[!0-9]*) : ;; *) [ "$master" -lt "$MINJOB" ] && continue ;; esac
    fi
    # Un-expanded pending array range, e.g. 49277376_[20-226] or _[20-226%40]:
    # sacct emits ONE row for all not-yet-scheduled tasks. Don't error-analyse it
    # (it's not a task that ran), but DO count its members toward the pending tally
    # -- otherwise "0 pending" wrongly implies the run has (nearly) drained.
    if [[ "$jobid" == *"["* ]]; then
        range=${jobid#*[}; range=${range%]}; range=${range%%%*}   # strip [ ] and %throttle
        n=0
        IFS=',' read -ra _parts <<< "$range"
        for p in "${_parts[@]}"; do
            case "$p" in
                *-*) lo=${p%-*}; hi=${p#*-}; n=$(( n + hi - lo + 1 )) ;;
                *)   n=$(( n + 1 )) ;;
            esac
        done
        state=${state%% *}
        TALLY[$state]=$(( ${TALLY[$state]:-0} + n ))
        total=$(( total + n ))
        [ "$state" = PENDING ] && pending=$(( pending + n ))
        continue
    fi
    state=${state%% *}                            # "CANCELLED by 12345" -> CANCELLED
    TALLY[$state]=$(( ${TALLY[$state]:-0} + 1 ))
    total=$((total+1))
    [ "$state" = RUNNING ] && running=$((running+1))
    [ "$state" = PENDING ] && pending=$((pending+1))
    is_ok_state "$state" || errors+=( "$jobname|$jobid|$state|$exitcode" )
done < <(sacct -X -n -P --starttime "$SINCE" --name="$NAMES" \
             --format=JobID,JobName,State,ExitCode 2>/dev/null || true)

echo "=== Parrot cohort error check (chunks: $CHUNKS  since: $SINCE) ==="
if [ "$total" -eq 0 ]; then
    echo "No parrot-{$CHUNKS} tasks found in the window. Widen with --since (e.g. --since now-3days),"
    echo "or check the chunk names / that jobs were actually submitted."
    exit 0
fi

# --- tally -------------------------------------------------------------------
echo "State tally ($total tasks):"
for st in $(printf '%s\n' "${!TALLY[@]}" | sort); do
    printf '  %6d  %s\n' "${TALLY[$st]}" "$st"
done
[ $((running+pending)) -gt 0 ] && \
    echo "  (note: $running running + $pending pending -- not final; re-run when they drain)"

# --- errors ------------------------------------------------------------------
if [ "${#errors[@]}" -eq 0 ]; then
    echo
    echo "OK: no errored tasks."
    exit 0
fi

echo
echo "!! ${#errors[@]} task(s) with errors:"
printf '%-4s %-5s %-9s %-16s %-14s %-6s %s\n' CHUNK IDX SUBJECT JOBID STATE EXIT ERRLOG
for e in "${errors[@]}"; do
    IFS='|' read -r jobname jobid state exitcode <<< "$e"
    chunk=${jobname#parrot-}
    idx=${jobid#*_}; master=${jobid%_*}
    out="$LOGDIR/parrot-${jobname}-${master}_${idx}.out"
    err="$LOGDIR/parrot-${jobname}-${master}_${idx}.err"

    subj="?"
    if [ -f "$out" ]; then
        subj=$(grep -m1 -oE 'subject=[^ ]+' "$out" 2>/dev/null | cut -d= -f2 || true)
    fi
    if [ -z "$subj" ] || [ "$subj" = "?" ]; then
        [ -n "$SUBJ_FILE" ] && [ -f "$SUBJ_FILE" ] && subj="$(sed -n "$((idx+1))p" "$SUBJ_FILE" 2>/dev/null)"
        subj="${subj:-?}"
    fi

    errshow="$err"; [ -f "$err" ] || errshow="(missing: $err)"
    printf '%-4s %-5s %-9s %-16s %-14s %-6s %s\n' \
        "$chunk" "$idx" "$subj" "$jobid" "$state" "$exitcode" "$errshow"

    if [ "$TAIL" -gt 0 ]; then
        echo "  ---- tail -$TAIL $err ----"
        [ -f "$err" ] && tail -n "$TAIL" "$err" | sed 's/^/  /' || echo "  (no .err file)"
        plog="${OUTPUT_DIR:+$OUTPUT_DIR/logs/sub-$subj/parrot-reconstruction_log.txt}"
        if [ -n "$plog" ] && [ -f "$plog" ]; then
            echo "  ---- tail -$TAIL $plog ----"
            tail -n "$TAIL" "$plog" | sed 's/^/  /'
        fi
        echo
    fi
done

echo
echo "Drill down:  tail -60 <ERRLOG>"
[ -n "$OUTPUT_DIR" ] && echo "     stage:  tail -60 $OUTPUT_DIR/logs/sub-<ID>/parrot-reconstruction_log.txt"
echo "     retry a chunk for just the failed subjects:"
echo "             bash $SCRIPT_DIR/submit_cohort.sh run --chunks <chunk> sub-<ID> ..."
exit 1
