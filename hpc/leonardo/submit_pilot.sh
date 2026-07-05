#!/bin/bash
###############################################################################
# Wrapper: load config.local.sh, then submit pilot.sbatch with the correct
# SLURM account. Extra args are forwarded to sbatch (e.g. --parsable).
#
#   ./hpc/leonardo/submit_pilot.sh [--parsable | any sbatch flag]
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] || { echo "ERROR: hpc/leonardo/config.local.sh not found -- cp it from config.local.sh.example"; exit 1; }
. "$SCRIPT_DIR/config.local.sh"

# "$@" are sbatch OPTIONS -> they must come BEFORE the script name (sbatch stops
# parsing options at the script; anything after it goes to the job script instead).
sbatch --account="$ACCT" "$@" "$SCRIPT_DIR/pilot.sbatch"