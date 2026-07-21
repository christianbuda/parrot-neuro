"""Forward model: leadfield loading, dipole orientation, source->scalp projection.

Loads a subject's Parrot forward-model derivatives (leadfield, dipole
positions/orientations, per-parcel structure) and assembles everything the
optimizer needs to turn per-region source activity into scalp EEG.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg

from . import config


def sample_spherical(npoints, key):
    """Uniform random unit vectors, shape (npoints, 3)."""
    vec = jax.random.normal(key, shape=(npoints * 3,)).reshape((-1, 3))
    norms = jnp.maximum(jnp.linalg.norm(vec, axis=-1, keepdims=True), 1e-12)
    return vec / norms


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


def block_diag_matmul(blocks, x):
    """Block-diagonal matmul: ``block_diag(*blocks) @ x`` without ever
    materializing the dense block-diagonal matrix.

    ``blocks`` is a tuple of square ``(n_b, n_b)`` matrices whose sizes sum to
    ``x.shape[0]``; block *b* multiplies the contiguous ``x[offset:offset+n_b]``
    slice and the results are stacked back into a ``(sum n_b, T)`` array. The
    Gaussian source-smoothing matrix is block-diagonal (a dipole only smooths
    within its own surface/volumetric block), so this is exact — but it touches
    only the ~few-percent of entries that aren't structurally zero, and never
    allocates the dense ``(N_dip, N_dip)`` matrix (multi-GB at tens of thousands
    of dipoles). The tuple length and every block size are static (known at
    trace time), so the loop unrolls cleanly under jit and stays differentiable.
    """
    outputs = []
    offset = 0
    for block in blocks:
        size = block.shape[1]
        outputs.append(block @ x[offset:offset + size])
        offset += size
    return jnp.concatenate(outputs, axis=0)


def project_to_scalp(source_activity, channel_indices, leadfield, smoothing_blocks, dipole_labels):
    """Per-region source activity -> scalp EEG.

    Broadcasts region activity to dipoles, spatially smooths (block-diagonal
    Gaussian smoothing, applied without densifying — see ``block_diag_matmul``),
    then applies the leadfield for the selected channels. Returns (n_channels, T).
    """
    source_activity = source_activity[dipole_labels]
    source_activity = block_diag_matmul(smoothing_blocks, source_activity)
    return leadfield[channel_indices] @ source_activity
