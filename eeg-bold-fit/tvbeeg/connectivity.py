"""Structural connectivity, empirical BOLD, and the missing-label bookkeeping
that ties a subject's forward-model dipoles to a connectome parcellation.

Parcellations occasionally drop regions (no dipoles, no BOLD coverage, ...).
Everything that must be re-indexed when regions are dropped lives here so it
only needs to be gotten right once: SC weights/delays, the empirical BOLD
target, and the dipole -> region label map used by the EEG forward model.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from . import config


def sc_weights_and_delays(subject, atlas=config.ATLAS, conduction_speed=config.CONDUCTION_SPEED):
    """Subject-specific (DWI-derived) structural connectivity as normalized
    (weights, delays).

    weights are divided by their max (unitless coupling strength); delays are
    tract length / conduction_speed (ms, if lengths are mm and speed is m/s* ).
    """
    W = subject.load.weights(atlas, normalized=True)
    L = subject.load.distances(atlas)
    return W, L


def drop_labels_square(M, missing):
    """Drop matching rows and columns from a square matrix."""
    idx = np.array([i for i in range(M.shape[0]) if i not in missing])
    return M[np.ix_(idx, idx)]


def drop_labels_vector(v, missing):
    """Drop matching entries from a 1D array (e.g. a per-region mask)."""
    idx = np.array([i for i in range(len(v)) if i not in missing])
    return v[idx]


def remap_dipole_labels(dipole_labels, missing_labels, n_full_regions):
    """Re-index dipole->region labels after dropping ``missing_labels``.

    Regions in ``missing_labels`` have no valid target index; dipoles that
    belonged to them are dropped (returned ``valid`` mask marks which of the
    input dipoles survive).

    Returns (remapped_labels, valid_mask) where remapped_labels has length
    ``valid_mask.sum()`` and indexes into the post-drop region ordering.
    """
    print(f"Re-indexing {len(dipole_labels)} dipoles after dropping "
          f"{len(missing_labels)} missing labels...")
    old_to_new = np.full(n_full_regions, -1, dtype=int)
    kept = np.array([i for i in range(n_full_regions) if i not in missing_labels])
    old_to_new[kept] = np.arange(len(kept))

    remapped = old_to_new[np.asarray(dipole_labels)]
    valid = remapped >= 0
    return remapped[valid], valid


@dataclass
class StructuralConnectivity:
    """SC graph + BOLD target, already reduced to the shared region set."""

    weights: jnp.ndarray          # (N, N) normalized coupling
    delays: jnp.ndarray           # (N, N) ms
    num_nodes: int
    missing_labels: np.ndarray    # region indices dropped from the full atlas
    n_full_regions: int           # atlas region count before dropping missing_labels
    empirical_bold: jnp.ndarray   # (T, N)


def load_structural_connectivity(
    subject,
    atlas=config.ATLAS,
    conduction_speed=config.CONDUCTION_SPEED,
    fmri_task="rest",
) -> StructuralConnectivity:
    """Subject SC weights/delays + empirical BOLD, aligned to the same region set.

    Empirical BOLD comes from the subject's own fMRI derivatives (the
    connectome-node-numbered ``desc-conn`` Schaefer time series, where row *i*
    already is connectome node *i*). Regions with no BOLD coverage come back as
    an all-NaN row there -- that NaN mask *is* the missing-labels list (no
    external missing-ROI file needed), and those regions are dropped from the
    SC matrices too so everything shares one ``num_nodes``-length region axis.
    """
    print(f"Loading SC weights/delays for atlas {atlas}...")
    W, L = sc_weights_and_delays(subject, atlas, conduction_speed)
    print(f"  SC weights: {W.shape}  delays: {L.shape}  (num_nodes = {W.shape[0]})")
    ts = subject.load.fmri_timeseries(variant="conn", task=fmri_task)[f"ts_{atlas}"]
    n_full_regions = ts.shape[0]
    missing = np.flatnonzero(np.isnan(ts).all(axis=1))
    X_emp = np.delete(ts, missing, axis=0).T  # (n_regions, T) -> (T, n_kept_regions)

    W = drop_labels_square(W, missing)
    L = drop_labels_square(L, missing)
    if X_emp.shape[1] != W.shape[0]:
        raise ValueError(
            f"Empirical BOLD has {X_emp.shape[1]} regions but SC (after "
            f"dropping missing labels) has {W.shape[0]}; check the fMRI 'conn' "
            "atlas variant and the SC weights/distances files agree on atlas "
            f"{atlas}'s region count/order for {subject.subj}."
        )

    weights = jnp.array(W, dtype=jnp.float64)
    weights = weights / jnp.max(weights)
    delays = jnp.array(L, dtype=jnp.float64) / conduction_speed

    return StructuralConnectivity(
        weights=weights,
        delays=delays,
        num_nodes=int(weights.shape[0]),
        missing_labels=missing,
        n_full_regions=n_full_regions,
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
