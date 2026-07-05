#!/bin/bash
###############################################################################
# Wrapper: load config.local.sh, then submit build_sif.sbatch with the correct
# SLURM account. Forwards <sif_dir> as the JOB script's positional arg.
#
#   ./hpc/leonardo/submit_build_sif.sh <sif_dir>
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] || { echo "ERROR: hpc/leonardo/config.local.sh not found -- cp it from config.local.sh.example"; exit 1; }
. "$SCRIPT_DIR/config.local.sh"

# "$@" is <sif_dir>, a positional arg of the JOB script (build_sif.sbatch reads $1)
# -> it must come AFTER the script name. (Opposite of submit_pilot.sh, where the
# extra args are sbatch options.) Don't "align" these two -- the placement differs by design.
sbatch --account="$ACCT" "$SCRIPT_DIR/build_sif.sbatch" "$@"