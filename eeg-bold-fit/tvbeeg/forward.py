"""Forward model: leadfield loading, dipole orientation, source->scalp projection.

Loads a subject's Parrot forward-model derivatives (leadfield, dipole
positions/orientations, per-parcel structure) and assembles everything the
optimizer needs to turn per-region source activity into scalp EEG.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax.scipy import linalg

from . import config


def sample_spherical(npoints, key):
    """Uniform random unit vectors, shape (npoints, 3)."""
    vec = jax.random.normal(key, shape=(npoints * 3,)).reshape((-1, 3))
    norms = jnp.maximum(jnp.linalg.norm(vec, axis=-1, keepdims=True), 1e-12)
    return vec / norms


@jax.jit
def apply_gaussian_smoothing(distance_matrix, dipole_volume, signals, sigma):
    """Volume-weighted Gaussian spatial smoothing of dipole signals.

    - distance_matrix: (N, N) inter-dipole distances.
    - dipole_volume: (N,) tissue volume per dipole.
    - signals: (N, T) signals to smooth.
    - sigma: Gaussian spread (same units as distances, e.g. mm).
    Returns (N, T) smoothed signals.
    """
    weights = jnp.square(distance_matrix) / (-2.0 * sigma**2)
    weights = jnp.exp(weights) * dipole_volume[None, :]
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights @ signals


@jax.jit
def compute_cholesky_matrix(distance_matrix, lambda_decay, eps=1e-5):
    """Lower-triangular spatial-mixing matrix L (C = L @ L.T) for colored noise."""
    covariance_matrix = jnp.exp(distance_matrix / (-1.0 * lambda_decay))
    diag = jnp.diag_indices(covariance_matrix.shape[0])
    covariance_matrix = covariance_matrix.at[diag].add(eps)
    return linalg.cholesky(covariance_matrix, lower=True)


def set_directions(leadfield, dipoles, key):
    """Collapse the (N_elec, 3*N_dip) leadfield onto per-dipole orientations.

    Fixed-orientation dipoles use their stored normal; free ('R') dipoles get a
    random unit orientation. ``dipoles`` is a ``parrot_neuro`` ``DipoleSet``
    (``subject.load.dipoles(spacing)``). Returns (N_elec, N_dip).
    """
    # Copy: dipoles.directions may be a cached array on the Subject (cache=True)
    # and we mutate it in place below.
    dipole_directions = np.array(dipoles.directions, copy=True)
    orient_type = dipoles.orient_type
    orient_type_mask = orient_type == "R"

    random_dirs = sample_spherical(np.count_nonzero(orient_type_mask), key)
    dipole_directions[orient_type_mask] = random_dirs

    leadfield = leadfield * dipole_directions.reshape((1, -1))
    leadfield = leadfield.reshape((leadfield.shape[0], -1, 3)).sum(axis=-1)
    return leadfield


def get_electric_signals(
    subject,
    spacing=config.SPACING,
    atlas=config.ATLAS,
    leadfield_label=config.LEADFIELD_LABEL,
    verbose=False,
    compute_representative_dipole=True,
):
    """Assemble the forward-model tensors for one subject.

    With tens of thousands of dipoles this is the slow part of a fit: most
    of the cost is loading dozens of per-block ``distance_matrix.npy`` files
    (I/O-bound) and exponentiating them into Gaussian weights (CPU-bound,
    but numpy releases the GIL for large arrays) — both are independent
    per-block work, so both are parallelized across blocks with a thread
    pool. ``compute_representative_dipole=False`` additionally skips
    building a second dense (N_dip, N_dip) distance matrix — pass it when
    you don't need ``representative_dipole`` (most callers don't).

    ``subject`` is a ``parrot_neuro.Subject``. The per-block ``surfaces/<block>/``
    and ``volumetric/`` distance/volume files have no curated Subject accessor
    (only the aggregated top-level arrays do), so those are still read via raw
    path composition off ``subject.path.dipole_dir(spacing)``.

    Returns:
        leadfield: (N_elec, N_dip) orientation-collapsed leadfield.
        weights: (N_dip, N_dip) volume-weighted Gaussian source-smoothing matrix.
        dipole_labels: (N_dip,) reduced-atlas parcel index per dipole (0-based).
        orient_atlas: (N_parcels,) orientation type per parcel ('N/G/P' cortical).
        representative_dipole: (N_parcels,) index of the most-central dipole per
            parcel, or ``None`` if ``compute_representative_dipole=False``.
    """
    assert isinstance(spacing, str), 'spacing must be a string (e.g. "2.0")'
    assert float(spacing) in subject.dipole_spacings(), f"input spacing of {spacing} not found"
    dipoles_path = subject.path.dipole_dir(float(spacing))

    all_blocks = [
        item for item in (dipoles_path / "surfaces").iterdir() if item.is_dir()
    ] + [dipoles_path / "volumetric"]

    print(f"Loading leadfield and dipoles from {dipoles_path}...")
    dipoles = subject.load.dipoles(float(spacing))
    dipole_labels = subject.load.dipole_labels(atlas, float(spacing))
    label_converter = subject.load.npy("connectivity", f"full_to_reduced_{atlas}.npy")
    orient_type = dipoles.orient_type

    # Map to reduced-atlas labels; -1 because the connectome has no label 0.
    dipole_labels = label_converter[dipole_labels] - 1

    # 'G','N','P' ~ pyramidal (kept as cortical sources); 'R','U' discarded.
    orient_atlas = np.array(["U"] * (label_converter.max()))
    for val in np.unique(dipole_labels):
        orient_atlas[val] = np.unique(orient_type[dipole_labels == val])[0]

    # # Per-block file reads are independent I/O — parallelize across blocks
    # # (measured ~4x on NFS: dominant cost of this whole function otherwise).
    # with ThreadPoolExecutor(max_workers=min(8, len(all_blocks))) as pool:
    #     dipole_volume = np.concatenate(
    #         list(pool.map(lambda block: np.load(block / "dipole_volume.npy"), all_blocks))
    #     )
    #     dist_mats = list(pool.map(lambda block: np.load(block / "distance_matrix.npy"), all_blocks))

    # # Volume-weighted Gaussian smoothing matrix (block-diagonal across
    # # regions). Elementwise exp() over ~500M+ values is the other dominant
    # # cost; numpy releases the GIL for large arrays so threading this across
    # # blocks helps too (measured ~2.5x), unlike most pure-Python CPU work.
    # sigma = float(spacing) * 1.5
    # with ThreadPoolExecutor(max_workers=min(8, len(dist_mats))) as pool:
    #     weights_blocks = list(pool.map(lambda m: np.exp(m**2 / (-2.0 * sigma**2)), dist_mats))

    # # Volume-weight and row-normalize PER BLOCK before assembling into the
    # # full (N_dip, N_dip) matrix, not after: block_diag fills everything
    # # outside a dipole's own block with exact zeros, so a full-row sum over
    # # the assembled matrix is mathematically identical to summing just that
    # # dipole's own block — but touches ~4x fewer elements (blocks are ~27%
    # # of the dense N_dip^2 total here), and the blocks are independent so
    # # this is thread-parallelizable the same way the exp() above is.
    # def _volume_normalize(block_w, vol):
    #     block_w = block_w * vol[None, :]
    #     return block_w / block_w.sum(axis=1, keepdims=True)

    # volume_slices = []
    # idx = 0
    # for mat in dist_mats:
    #     volume_slices.append(dipole_volume[idx : idx + len(mat)])
    #     idx += len(mat)
    # with ThreadPoolExecutor(max_workers=min(8, len(weights_blocks))) as pool:
    #     weights_blocks = list(pool.map(_volume_normalize, weights_blocks, volume_slices))

    # weights = scipy.linalg.block_diag(*weights_blocks)

    # representative_dipole = None
    # if compute_representative_dipole:
    #     # Full distance matrix (inf across blocks) to pick central dipoles
    #     # per parcel. A second dense (N_dip, N_dip) array — skip this whole
    #     # block (compute_representative_dipole=False) if you don't need it.
    #     n = weights.shape[0]
    #     distances = np.full((n, n), np.inf)
    #     block_id = np.empty(n, dtype=int)
    #     current_idx = 0
    #     for b, mat in enumerate(dist_mats):
    #         sl = slice(current_idx, current_idx + len(mat))
    #         distances[sl, sl] = mat
    #         block_id[sl] = b
    #         current_idx += len(mat)

    #     representative_dipole = -np.ones(dipole_labels.max() + 1, dtype=int)
    #     for i in range(len(representative_dipole)):
    #         idx = np.flatnonzero(dipole_labels == i)
    #         if len(idx) > 0:
    #             # Dipoles are only mutually comparable within one block (each
    #             # surface/volumetric block has its own independent distance
    #             # matrix; cross-block entries are inf). A parcel can straddle
    #             # more than one block (e.g. surface/volumetric boundary), so
    #             # represent it from whichever block holds most of its dipoles
    #             # rather than requiring a single shared block for all of them.
    #             blocks, counts = np.unique(block_id[idx], return_counts=True)
    #             idx = idx[block_id[idx] == blocks[np.argmax(counts)]]
    #             best_dip = np.argmin(distances[np.ix_(idx, idx)].mean(axis=0))
    #             representative_dipole[i] = idx[best_dip]
    #             if verbose:
    #                 print(f"lab {i}, best dip {best_dip}, val {representative_dipole[i]}")
    # if representative_dipole is not None:
    #     print(f"Found {np.count_nonzero(representative_dipole >= 0)} representative dipoles.")

    # key = jax.random.PRNGKey(42)
    # leadfield = np.load(
    #     leadfields_path / f"processed_{leadfield_label}-{spacing}mm-leadfield.npy"
    # )
    # leadfield = set_directions(leadfield, dipoles_path, key)
    dipole_volume = np.concatenate([np.load(block / 'dipole_volume.npy') for block in all_blocks])

    sigma = float(spacing) * 1.5
    weights_matrices = [np.load(block / 'distance_matrix.npy') for block in all_blocks]
    idx = 0
    for mat in weights_matrices:
        np.square(mat, out=mat)
        mat *= -1.0 / (2.0 * sigma**2)
        np.exp(mat, out=mat)
        mat *= dipole_volume[None, idx:idx + len(mat)]
        mat /= mat.sum(axis=1, keepdims=True)
        idx += len(mat)
    
    key = jax.random.PRNGKey(42)
    leadfield = subject.load.leadfield(f"{leadfield_label}-{spacing}mm")
    leadfield = set_directions(leadfield, dipoles, key)
    representative_dipole = None
    return leadfield, weights_matrices, dipole_labels, orient_atlas, representative_dipole


def project_to_scalp(source_activity, channel_indices, leadfield, smoothing_weights, dipole_labels):
    """Per-region source activity -> scalp EEG.

    Broadcasts region activity to dipoles, spatially smooths, then applies the
    leadfield for the selected channels. Returns (n_channels, T).
    """
    source_activity = source_activity[dipole_labels]
    source_activity = smoothing_weights @ source_activity
    return leadfield[channel_indices] @ source_activity
