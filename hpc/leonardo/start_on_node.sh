#!/bin/bash
###############################################################################
# Launch a batch of sweep agents on THIS login node, pointed at an already-
# registered sweep, so the SAME shared Bayesian search can be worked by
# agents running concurrently on Leonardo's OTHER login nodes too.
#
# Why this exists: RLIMIT_NPROC (and the wandb-agent-startup/OpenBLAS-sync
# thread pressure that trips it -- see README.md's "start N ... launching
# far fewer than N agents" gotcha) is enforced PER LOGIN NODE, not
# cluster-wide -- Leonardo has (at least) four independent ones:
# login01/02/05/07-ext.leonardo.cineca.it, behind the round-robin
# login.leonardo.cineca.it alias. A plain `ssh` puts you on whichever one
# the alias picks; running everything from a single persistent session
# concentrates ALL your agents' process load onto that one machine's quota.
# Splitting batches across nodes multiplies the sustainable total instead
# of fighting one node's ceiling. (There is no CINECA-native "online sweep
# agent" node/mechanism -- see README.md's "as intended by CINECA" note;
# this is the practical lever that actually exists.)
#
# Usage: run on EACH login node you want to add capacity from, with a
# DIFFERENT <node-tag> per node so their PID/log bookkeeping (status/stop)
# stays independent -- but all pointed at the SAME underlying sweep_id, so
# they contribute to one shared search rather than fragmenting into
# separate ones. $HOME is shared NFS across all login nodes, so the
# sweep-ID file written on the node where you ran `create` is already
# visible here.
#
#   ssh bgambosi@login01-ext.leonardo.cineca.it
#   MASTER_SWEEP_NAME=parallel-2 bash hpc/leonardo/start_on_node.sh nodeA 15 5
#
#   ssh bgambosi@login02-ext.leonardo.cineca.it
#   MASTER_SWEEP_NAME=parallel-2 bash hpc/leonardo/start_on_node.sh nodeB 15 5
#
# MASTER_SWEEP_NAME defaults to "parallel-2" (the sweep this was written
# for) -- override it, or the SWEEP_NAME you used with `submit_sweep.sh
# create`, if that's not the one you're extending. Run each of these
# inside tmux/screen -- see submit_sweep.sh's own warning, still applies
# per node.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_TAG="${1:?usage: start_on_node.sh <node-tag> [N] [COUNT]}"
N="${2:-8}"
COUNT="${3:-5}"
MASTER_SWEEP_NAME="${MASTER_SWEEP_NAME:-parallel-2}"

MASTER_ID_FILE="$SCRIPT_DIR/.sweep_id.$MASTER_SWEEP_NAME"
NODE_ID_FILE="$SCRIPT_DIR/.sweep_id.$NODE_TAG"

[ -f "$MASTER_ID_FILE" ] || {
    echo "ERROR: $MASTER_ID_FILE not found -- set MASTER_SWEEP_NAME to whatever" >&2
    echo "       SWEEP_NAME you used with 'submit_sweep.sh create' for the sweep" >&2
    echo "       you want to extend across nodes (unset SWEEP_NAME at create time" >&2
    echo "       means the file is just hpc/leonardo/.sweep_id, no suffix)." >&2
    exit 1
}

if [ -f "$NODE_ID_FILE" ]; then
    # Already seeded (e.g. re-running `start` again on this same node) --
    # confirm it still points at the same sweep rather than silently
    # diverging into a different search.
    if ! diff -q "$MASTER_ID_FILE" "$NODE_ID_FILE" >/dev/null; then
        echo "ERROR: $NODE_ID_FILE already exists and points at a DIFFERENT sweep" >&2
        echo "       than $MASTER_ID_FILE -- resolve manually (rm it if it's stale)." >&2
        exit 1
    fi
else
    cp "$MASTER_ID_FILE" "$NODE_ID_FILE"
    echo "[start_on_node] seeded $NODE_ID_FILE from $MASTER_ID_FILE (sweep $(cat "$NODE_ID_FILE"))"
fi

echo "[start_on_node] on $(hostname): launching $N agents x $COUNT runs under SWEEP_NAME=$NODE_TAG (sweep $(cat "$NODE_ID_FILE"))"
SWEEP_NAME="$NODE_TAG" bash "$SCRIPT_DIR/submit_sweep.sh" start "$N" "$COUNT"
