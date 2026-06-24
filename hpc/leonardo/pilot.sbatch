#!/bin/bash
###############################################################################
# Parrot pilot job for CINECA LEONARDO (Booster / GPU partition).
#
# Runs ONE subject end-to-end on a single GPU under Apptainer, and instruments
# it (per-stage wall-clock + GPU-utilisation samples). This is a *measurement*
# run: its numbers (total walltime vs the 24h cap, and the GPU-idle fraction of
# the recon container) decide whether the cohort runs monolithic-on-Booster or
# need the GPU/CPU split. Do NOT scale to the array until this has been read.
#
# Booster node = 1x Xeon 8358 (32 cores) + 4x A100-64GB. We deliberately take
# only ONE GPU (--gres=gpu:1) and a 1/4 slice of cores: Booster allows node
# sharing and bills per-GPU, so this is the cheapest honest pilot.
#
# EDIT the four things marked `# <<EDIT>>` below, then:
#   sbatch hpc/leonardo/pilot.sbatch
###############################################################################

#SBATCH --account=<YOUR_ACCOUNT>           # <<EDIT>> your SLURM account (find it with: saldo -b)
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=parrot-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                  # 1/4 of a 32-core node, matching --gres=gpu:1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G                         # ~1/4 of 512G
#SBATCH --time=20:00:00                    # under the 24h boost_usr_prod cap
#SBATCH --output=parrot-pilot-%j.out
#SBATCH --error=parrot-pilot-%j.err

set -euo pipefail

# --- account / paths / subject (EDIT) ----------------------------------------
# $WORK/$SCRATCH are ambiguous on LEONARDO if you belong to several accounts -- use explicit
# /leonardo_work/<ACCT> paths. The per-account work area is large (tens of TB), so data +
# derivatives live there; see `cindata` for your areas.
ACCT="<YOUR_ACCOUNT>"                       # <<EDIT>> same account as --account above
WORKDIR="/leonardo_work/$ACCT"             # large per-account work area
REPO="$HOME/parrot-neuro"                  # <<EDIT>> clone location of this repo
BIDS="$WORKDIR/parrot/bids"                # <<EDIT>> staged BIDS dataset on the cluster
SUBJECT="010002"                           # <<EDIT>> participant label to pilot
OUT="$BIDS/derivatives"                    # derivatives go next to the data
SIF="$WORKDIR/parrot_sif"                  # persistent .sif cache (pull once, reuse)

# --- modules -----------------------------------------------------------------
# CUDA for --nv. Apptainer/Singularity is NOT in LEONARDO's default module profile -- it is
# usually a system command (or under another profile). Verify on a login node first:
#   command -v apptainer singularity ; module spider apptainer singularity
module purge
module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true   # harmless if absent
command -v apptainer >/dev/null || command -v singularity >/dev/null || {
    echo "ERROR: no apptainer/singularity on PATH. Find it: 'module spider apptainer' or ask CINECA."; exit 1; }

mkdir -p "$SIF" "$OUT"

echo "=== Parrot pilot: job $SLURM_JOB_ID on $(hostname), $(date) ==="
echo "subject=$SUBJECT  cpus=$SLURM_CPUS_PER_TASK  gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi -L || echo "WARNING: no GPU visible"

# --- GPU-utilisation sampling (for the GPU-idle-fraction measurement) --------
GPULOG="$SLURM_SUBMIT_DIR/gpu_util-$SLURM_JOB_ID.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
           --format=csv -l 10 > "$GPULOG" 2>/dev/null &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

# --- the run -----------------------------------------------------------------
# One subject, whole pipeline, Apptainer runtime, single GPU. The orchestrator
# pulls any missing .sif from Docker Hub into $SIF on first use (slow once; see
# prepull note in README to do it off the GPU clock). Modest --threads avoids
# the BLAS oversubscription that makes place_dipoles crawl.
SECONDS=0
"$REPO/bin/run_reconstruction.sh" \
    "$BIDS" "$OUT" participant \
    --participant-label "$SUBJECT" \
    --runtime apptainer --sif-dir "$SIF" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --gpus all
echo "=== pilot finished in $((SECONDS/60)) min, rc=$? ($(date)) ==="

# --- quick post-run summary --------------------------------------------------
LOGDIR="$OUT/logs/sub-$SUBJECT"
echo "--- per-stage wall-clock ---"
grep -E "completed in [0-9]+ minutes" "$LOGDIR/parrot-reconstruction_log.txt" 2>/dev/null \
    || echo "(see $LOGDIR/parrot-reconstruction_log.txt)"
echo "--- GPU-idle fraction (rows at 0% util / total samples in $GPULOG) ---"
awk -F', ' 'NR>1{t++; if($2 ~ /^0 ?%/) z++} END{if(t)printf "    %d/%d = %.0f%% idle\n", z, t, 100*z/t}' "$GPULOG" 2>/dev/null || true
