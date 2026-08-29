#!/bin/bash
###############################################################################
# One-time setup of the `pixi` environment for the EEG+BOLD optimization stage
# (parrot_neuro.optimization) on CINECA LEONARDO.
#
# Unlike the reconstruction pipeline (four Docker/Apptainer images), the
# optimization stage has no container -- it's a real Python+JAX environment,
# built the same way as local dev via `pixi.toml` (which already declares
# linux-64 as "your workstation + LEONARDO"). `jax[cuda12]` ships its own CUDA
# runtime as pip wheels, so it only needs the compute node's NVIDIA driver --
# no system CUDA module/toolkit dependency.
#
# Run ONCE on a LOGIN node (needs internet for conda-forge/PyPI resolution).
# The resulting .pixi/envs/default is a self-contained env directory -- fully
# offline-usable afterward on compute nodes, same "build once with egress,
# reuse without" shape as the .sif cache / hippunfold+templateflow prewarm.
#
#   bash hpc/leonardo/setup_optim_env.sh
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for _c in "${PARROT_CONFIG:-}" "$SCRIPT_DIR/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; echo "[config] loaded $_c"; break; }
done
REPO="${REPO:-$HOME/parrot-neuro}"

if ! command -v pixi >/dev/null 2>&1; then
  if [ -x "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
  else
    echo "[pixi] not found -- installing to \$HOME/.pixi/bin ..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
  fi
fi
command -v pixi >/dev/null || {
  echo "ERROR: pixi install failed / not on PATH. Open a new shell (the installer appends to ~/.bashrc) and re-run."
  exit 1
}
echo "[pixi] $(pixi --version)"

[ -d "$REPO" ] || { echo "ERROR: repo not found at $REPO (set REPO in config.local.sh)"; exit 1; }
cd "$REPO"

echo "[pixi] resolving + installing environment from pixi.toml (needs internet -- run on a LOGIN node) ..."
pixi install

echo "[pixi] sanity-checking imports (CPU-only here; GPU devices are only visible inside a GPU job) ..."
# Login nodes cap per-user thread/process counts well below the node's full
# core count; OpenBLAS otherwise sizes its threadpool to nproc (128 here) and
# pthread_create() starts failing partway through. This is just an import
# check, so force single-threaded BLAS for it.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
pixi run python -c "
import jax, tvboptim, optax, equinox
print('jax', jax.__version__, '-- import OK')
"

echo "[setup_optim_env] done -- environment ready at $REPO/.pixi"
echo "Next: bash hpc/leonardo/check_optim.sh   (preflight before smoke/pilot/run)"
