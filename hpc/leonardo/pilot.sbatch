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
#SBATCH --qos=boost_qos_lprod              # 4-day wall: the recon CPU tail (ANTs) blows past 24h at 8 cores
#SBATCH --job-name=parrot-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # WHOLE node's cores: cerebellum+bigbrain are ANTs SyN registration
                                           # (ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS) -- CPU-bound, GPU idle. At 8
                                           # cores bigbrain alone ran >11.5h and timed out; 32 cores ~4x that.
                                           # NB: 32 cores + gpu:1 strands the node's other 3 GPUs -> you may be
                                           # billed ~the whole node. Fine for a one-off pilot; the cohort split
                                           # (Deliverable 2) runs these steps on a CPU-only partition instead.
#SBATCH --gres=gpu:1                       # still needed for FastSurfer / QSIPrep(-if-DWI); harmless otherwise
#SBATCH --mem=240G                         # headroom for ANTs 100um bigbrain registration at 32 threads
#SBATCH --time=2-00:00:00                  # 48h; ample once the ANTs tail is on 32 cores (lprod allows up to 4d)
#SBATCH --output=parrot-pilot-%j.out
#SBATCH --error=parrot-pilot-%j.err

set -euo pipefail

# --- personal config (gitignored): set ACCT/WORKDIR/REPO/SUBJECT once ---------
# sbatch spools this script, so $BASH_SOURCE is not the repo copy -- resolve via
# $SLURM_SUBMIT_DIR (where you ran sbatch) or the default clone at $HOME/parrot-neuro.
# Fill hpc/leonardo/config.local.sh once and you can stop editing the <<EDIT>>s below.
for _c in "${PARROT_CONFIG:-}" \
          "${SLURM_SUBMIT_DIR:-}/hpc/leonardo/config.local.sh" \
          "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[config] loaded $_c"; break; }
done

# --- account / paths / subject (EDIT, or set them once in config.local.sh) ----
# $WORK/$SCRATCH are ambiguous on LEONARDO if you belong to several accounts -- use explicit
# /leonardo_work/<ACCT> paths. The per-account work area is large (tens of TB), so data +
# derivatives live there; see `cindata` for your areas.
ACCT="${ACCT:-<YOUR_ACCOUNT>}"             # <<EDIT>> SLURM billing account; keep in sync with --account above
# Storage area, INDEPENDENT of the billing account. Defaults to $WORK for convenience, but $WORK is
# AMBIGUOUS if you belong to several accounts (may resolve to the wrong allocation) -> set WORKDIR then.
WORKDIR="${WORKDIR:-${WORK:-}}"            # e.g. /leonardo_work/<ACCT>
: "${WORKDIR:?set WORKDIR (or export \$WORK) -- e.g. /leonardo_work/<ACCT>}"
REPO="$HOME/parrot-neuro"                  # <<EDIT>> clone location of this repo
BIDS="$WORKDIR/parrot/bids"                # <<EDIT>> staged BIDS dataset on the cluster
SUBJECT="010002"                           # <<EDIT>> participant label to pilot
OUT="$BIDS/derivatives"                    # derivatives go next to the data
SIF="$WORKDIR/parrot/parrot_sif"           # persistent .sif cache (pull once, reuse)

# --- modules -----------------------------------------------------------------
# CUDA for --nv. On LEONARDO, Singularity is a SYSTEM command (/usr/bin/singularity) -- NOT a
# module -- so it survives `module purge`. (No `apptainer`; our runtime helper falls back to
# singularity.) Confirmed 2026-06-25: `command -v singularity` -> /usr/bin/singularity.
module purge
module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
command -v apptainer >/dev/null || command -v singularity >/dev/null || {
    echo "ERROR: no apptainer/singularity on PATH (expected /usr/bin/singularity). Ask CINECA."; exit 1; }

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
