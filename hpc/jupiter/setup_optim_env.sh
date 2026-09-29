#!/bin/bash
###############################################################################
# One-time setup of the `pixi` environment for the EEG+BOLD optimization stage
# (parrot_neuro.optimization) on JUPITER (JSC).
#
# Mirrors hpc/leonardo/setup_optim_env.sh -- same idea (a real Python+JAX env
# via pixi, no container, built once with login-node internet and reused
# offline on compute nodes afterward) -- but JUPITER's GH200 nodes are
# linux-aarch64, not linux-64 like LEONARDO/a typical workstation.
#
# *** KNOWN BLOCKER, unresolved as of this script's creation ***
# The repo's pixi.toml currently declares `platforms = ["linux-64",
# "osx-arm64"]` only -- linux-aarch64 was tried before and dropped because
# conda-forge has no `marimo` build for it (see pixi.toml's own comment).
# `marimo`/`ipykernel` are dev-only (local notebooks under development/) --
# examples/eeg_bold_fit_cli.py never imports them. The fix is to split
# pixi.toml into a dev feature (marimo/ipykernel, linux-64/osx-arm64 only)
# and an optim feature (jax/optax/equinox/tvboptim/mne/etc. -- everything the
# fit actually needs) that ALSO targets linux-aarch64, via pixi's
# [feature.*]/[environments] tables, then running `pixi install -e optim`
# HERE (this script) to resolve/lock the aarch64 env against conda-forge/PyPI
# (needs real internet -- login nodes have it, same assumption as LEONARDO).
# That pixi.toml restructuring has NOT been done yet -- `pixi install` below
# will fail to resolve on this node's aarch64 platform until it is. Do that
# first (edit pixi.toml, ideally validated with a `pixi install -e optim` in
# THIS script), then re-run this script.
#
#   bash hpc/jupiter/setup_optim_env.sh
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
echo "[pixi] platform: $(uname -m) -- must be linux-aarch64 on JUPITER's GH200 nodes"

[ -d "$REPO" ] || { echo "ERROR: repo not found at $REPO (set REPO in config.local.sh)"; exit 1; }
cd "$REPO"

if ! grep -q '"linux-aarch64"' pixi.toml 2>/dev/null; then
  echo "ERROR: pixi.toml does not declare linux-aarch64 yet -- this is the known blocker"
  echo "       documented at the top of this script. Restructure pixi.toml first (split"
  echo "       marimo/ipykernel into a dev-only feature so an optim feature can target"
  echo "       linux-aarch64), then re-run this script."
  exit 1
fi

echo "[pixi] resolving + installing environment from pixi.toml (needs internet -- run on a LOGIN node) ..."
# -e optim: once pixi.toml is restructured (see header), the optim feature/
# environment should exclude marimo/ipykernel so it can actually resolve on
# aarch64. If pixi.toml ends up using a different environment name, update
# this flag to match.
pixi install -e optim

echo "[pixi] sanity-checking imports (CPU-only here; GPU devices are only visible inside a GPU job) ..."
# Login nodes cap per-user thread/process counts well below the node's full
# core count; OpenBLAS otherwise sizes its threadpool to nproc and
# pthread_create() starts failing partway through. This is just an import
# check, so force single-threaded BLAS for it.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
pixi run -e optim python -c "
import jax, tvboptim, optax, equinox
print('jax', jax.__version__, '-- import OK')
print('jax devices (CPU-only expected here, no GPU on a login node):', jax.devices())
"

echo "[setup_optim_env] done -- environment ready at $REPO/.pixi"
echo "Next: bash hpc/jupiter/check_optim.sh   (preflight before smoke/pilot/run)"
