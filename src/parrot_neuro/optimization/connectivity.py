"""Structural connectivity + empirical BOLD, aligned to the fMRI-usable node set.

Node alignment -- which connectome nodes have usable BOLD, and how dipoles map
onto them -- is owned by ``parrot_neuro.Subject``: the ``fmri_aligned=True``
loaders and ``dipole_node_labels`` apply one precomputed ``fmri_keep`` mask
consistently across the SC weights/delays, the dipole->node map, and the BOLD
target. This module just loads the aligned SC and the matching BOLD; it no longer
re-derives a missing-region mask from NaN rows (that was a second, divergable copy
of the same information).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from . import config


@dataclass
class StructuralConnectivity:
    """SC graph + BOLD target, reduced to the fMRI-usable node set."""

    weights: jnp.ndarray          # (N, N) normalized coupling
    delays: jnp.ndarray           # (N, N) ms
    num_nodes: int
    keep: np.ndarray              # (M,) bool over the full connectome axis -> the N kept nodes
    empirical_bold: jnp.ndarray   # (T, N)


def load_structural_connectivity(
    subject,
    atlas=config.ATLAS,
    conduction_speed=config.CONDUCTION_SPEED,
    fmri_task="rest",
) -> StructuralConnectivity:
    """Subject SC weights/delays + empirical BOLD, aligned to the fMRI-usable node set.

    Weights/distances are loaded already restricted to the fMRI-usable connectome
    nodes via the Subject's ``fmri_aligned=True`` flag (the ``fmri_keep`` mask is
    applied *before* the ``/max`` normalization below, so the coupling scale stays
    self-consistent). The empirical BOLD target is the connectome-node-numbered
    ``desc-conn`` Schaefer time series with its unusable rows dropped by the *same*
    mask, so the SC graph and BOLD share one ``num_nodes``-length node axis. The
    dipole->node map (``subject.load.dipole_node_labels``) indexes into this exact
    same set. ``keep`` (over the full connectome axis) is returned so callers can
    bring other per-node quantities (e.g. the cortical mask) onto this same set.
    """
    print(f"Loading fMRI-aligned SC weights/delays for atlas {atlas}...")
    W = subject.load.weights(atlas, normalized=True, fmri_aligned=True)
    L = subject.load.distances(atlas, fmri_aligned=True)
    keep = np.asarray(subject.load.fmri_keep(atlas, fmri_task))
    print(f"  SC weights: {W.shape}  delays: {L.shape}  "
          f"(num_nodes = {W.shape[0]} of {keep.size} connectome nodes)")

    ts = subject.load.fmri_timeseries(variant="conn", task=fmri_task)[f"ts_{atlas}"]
    X_emp = ts[keep].T  # (n_conn_nodes, T)[keep] -> (T, n_kept)

    if X_emp.shape[1] != W.shape[0]:
        raise ValueError(
            f"Empirical BOLD has {X_emp.shape[1]} kept regions but fMRI-aligned SC "
            f"has {W.shape[0]}; check the fMRI 'conn' timeseries and the connectivity "
            f"weights/distances agree on atlas {atlas}'s node count/order for {subject.subj}."
        )

    weights = jnp.array(W, dtype=jnp.float64)
    weights = weights / jnp.max(weights)
    delays = jnp.array(L, dtype=jnp.float64) / conduction_speed

    return StructuralConnectivity(
        weights=weights,
        delays=delays,
        num_nodes=int(weights.shape[0]),
        keep=keep,
        empirical_bold=jnp.array(X_emp, dtype=jnp.float64),
    )


# --- BOLD feature helpers (used by both the loss and diagnostic plots) -----

def zscore_time(X, eps=1e-8):
    """Z-score each column (region) of a (T, N) array over the time axis."""
    mu = jnp.mean(X, axis=0, keepdims=True)
    sd = jnp.std(X, axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def fc_vector(X, skip_t=0, eps=1e-8):
    """Upper-triangle functional-connectivity vector from a (T, N) BOLD array."""
    X = zscore_time(X[skip_t:], eps=eps)
    C = jnp.corrcoef(X, rowvar=False)
    iu = jnp.triu_indices(C.shape[0], k=1)
    return C[iu]


def extract_bold_2d(bold_result):
    """(T, n_voi, N) Bold-monitor output with a single voi -> (T, N)."""
    return bold_result.ys[:, 0, :]
