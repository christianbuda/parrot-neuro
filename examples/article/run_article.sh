#!/bin/bash
###############################################################################
# Every article result from one population of fits (run from the repo root).
#
#   examples/article/run_article.sh cpu     # everything computable from the saved fits
#                                           # (+ whatever GPU outputs already exist)
#   examples/article/run_article.sh gpu     # write + run the extra fits on local GPUs
#   examples/article/run_article.sh quick   # cpu with few permutations (minutes)
#
# Environment (defaults in brackets):
#   ROOT [new_leadfields/1/atlas-100]   population of joint fits
#   BIDS [/srv/nfs-data/sisko/christian/parrot_LEMON]
#   OUT  [$ROOT/article]                figures/, tables/, numbers.json
#   FITS [article_fits]                 where the GPU fits land
#   GPUS [2,3,4,5,6,7]                  local GPUs for `gpu`
#   REPLICATE_ROOTS                     refit roots (space-separated) for the
#                                       refit-stability + age-across-refits sections
#   NEUROMAPS_PATH                      PYTHONPATH entry holding neuromaps (if it
#                                       is not installed in the pixi env)
###############################################################################
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="${ROOT:-new_leadfields/1/atlas-100}"
BIDS="${BIDS:-/srv/nfs-data/sisko/christian/parrot_LEMON}"
OUT="${OUT:-$ROOT/article}"
FITS="${FITS:-article_fits}"
GPUS="${GPUS:-2,3,4,5,6,7}"
PY="pixi run python examples/article"
MODE="${1:-cpu}"
common=(--root "$ROOT" --bids "$BIDS" --out "$OUT")

if [ "$MODE" = "quick" ]; then
  PERM=(--n-perm 50); AGE=(--n-perm 50 --n-boot 50 --n-null 10 --n-boot-null 20); SPIN=(--n-spin 100)
else
  PERM=(); AGE=(); SPIN=()
fi

case "$MODE" in
  gpu)
    mkdir -p "$FITS"
    $PY/gpu_fit.py tasks --root "$ROOT" --bids "$BIDS" --out "$FITS" --write "$FITS/tasks.txt"
    examples/article/run_gpu_tasks.sh "$FITS/tasks.txt" "$GPUS" "$FITS/logs"
    ;;
  cpu|quick)
    echo "== checks / figure 1";  $PY/checks.py "${common[@]}"; $PY/schematic.py --root "$ROOT" --out "$OUT"

    echo "== section 2: fit quality"
    HO=(); [ -d "$FITS/heldout/atlas-100" ] && HO=(--heldout-root "$FITS/heldout/atlas-100")
    $PY/fit_quality.py "${common[@]}" "${PERM[@]}" "${HO[@]}"

    echo "== section 3: joint vs single modality"
    J=()
    [ -d "$FITS/single_modality/atlas-100" ] && J+=(--eeg-only-root "$FITS/single_modality/atlas-100" \
                                                    --bold-only-root "$FITS/single_modality/atlas-100")
    [ -d "$FITS/eeg_weight/atlas-100" ] && J+=(--weight-roots "1e4=$FITS/eeg_weight/atlas-100" "1e6=$FITS/eeg_weight/atlas-100")
    if [ "${#J[@]}" -gt 0 ]; then $PY/joint_vs_single.py "${common[@]}" "${J[@]}"; else echo "   (no GPU fits yet -- run: $0 gpu)"; fi

    echo "== section 4: identifiability"
    I=()
    [ -d "$FITS/recovery/atlas-100" ] && I+=(--recovery-root "$FITS/recovery/atlas-100")
    [ -d "$FITS/fresh_noise" ] && I+=(--fresh-noise-root "$FITS/fresh_noise")
    $PY/identifiability.py "${common[@]}" "${I[@]}"

    echo "== section 5: regional organization"
    EXT=()
    if PYTHONPATH="${NEUROMAPS_PATH:-}" pixi run python -c "import neuromaps" 2>/dev/null; then EXT=(--external-maps "${SPIN[@]}"); \
    else echo "   (neuromaps not importable -- external maps skipped)"; fi
    PYTHONPATH="${NEUROMAPS_PATH:-}" $PY/regional.py "${common[@]}" "${EXT[@]}"

    echo "== section 7: aging"
    $PY/aging.py "${common[@]}" "${AGE[@]}"

    echo "== population report (regional maps, refit stability, age across refits) + section 8 behaviour"
    REP=(); [ -n "${REPLICATE_ROOTS:-}" ] && read -ra REP_LIST <<< "$REPLICATE_ROOTS" && REP=(--replicate-roots "${REP_LIST[@]}")
    QP=(); [ "$MODE" = "quick" ] && QP=(--n-perm 20 --no-brains)
    pixi run python examples/population_params_report_cli.py --results-root "$ROOT" --bids-dir "$BIDS" \
        --out "$OUT/population_report" "${REP[@]}" "${QP[@]}"
    pixi run python examples/behaviour_params_report_cli.py --report-dir "$OUT/population_report" \
        --behaviour-dir Behavioural_Data_MPILMBB_LEMON --bids-dir "$BIDS" --results-root "$ROOT" \
        --out "$OUT/behaviour" $([ "$MODE" = "quick" ] && echo "--n-perm 20 --no-brains")
    echo "== done: $OUT/figures, $OUT/tables, $OUT/numbers.json"
    ;;
  *) echo "usage: $0 cpu|gpu|quick"; exit 1 ;;
esac
