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

import jax
import jax.numpy as jnp
import numpy as np

from . import config
from .signal import bandpass_filter, welch_psd


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
    """(T, n_voi, N) BOLD output with a single voi -> (T, N).

    Accepts either a plain ``(T, n_voi, N)`` array (``simulator_bold``'s
    streaming-reduced return -- see ``train.build_simulators``) or, for
    backward compatibility, anything with a ``.ys`` attribute of that shape
    (a ``NativeSolution``-like object, e.g. from a direct post-hoc
    ``HRFBold(...)`` call).
    """
    ys = getattr(bold_result, "ys", bold_result)
    return ys[:, 0, :]


def filter_sim_bold(X, tr_ms, low=config.BOLD_BANDPASS_LOW, high=config.BOLD_BANDPASS_HIGH,
                     order=config.BOLD_BANDPASS_ORDER):
    """Zero-phase Butterworth-bandpass a ``(T, N)`` *simulated* BOLD series to
    the same band the empirical BOLD arrives pre-filtered to (typical
    resting-state fMRI preprocessing: 0.01-0.1 Hz). Only the simulated side
    needs this -- ``StructuralConnectivity.empirical_bold`` is already
    bandpassed upstream in the fMRI derivatives, so filtering it again here
    would double-filter it. Call this on simulated BOLD right before any
    FC/FCD computation (``fc_vector``, ``fcd_matrix``, ``dfc_histogram``)."""
    fs = 1000.0 / tr_ms
    return bandpass_filter(X, fs, low, high, order=order)


def bold_psd_band(X, tr_ms, nperseg, noverlap, skip_t=0,
                   low=config.BOLD_BANDPASS_LOW, high=config.BOLD_BANDPASS_HIGH, eps=1e-8):
    """Welch PSD of a ``(T, N)`` BOLD array, restricted + per-node normalized
    (each node's band power sums to 1) to ``[low, high]`` Hz.

    Welch's fixed ``nperseg`` fixes the frequency-bin grid independent of how
    long ``X`` is -- so a much-shorter simulated horizon and a much-longer
    empirical recording, computed with the SAME ``nperseg``/``noverlap``/
    ``tr_ms``, land on identical frequency bins and are directly comparable
    band-for-band despite the very different lengths. This is the mechanism
    the optional BOLD spectral-shape loss term (``train.make_bold_loss_fn``'s
    ``psd_weight``) relies on -- ``fc_vector``'s time-averaged correlation has
    no sensitivity at all to each signal's own temporal/spectral shape, only
    to which regions co-fluctuate.
    """
    fs = 1000.0 / tr_ms
    psd = welch_psd(X[skip_t:].T, fs=fs, nperseg=nperseg, noverlap=noverlap)  # (N, n_freq)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    idx_min = int(np.searchsorted(freqs, low))
    idx_max = int(np.searchsorted(freqs, high))
    band = psd[:, idx_min:idx_max]
    return band / (jnp.sum(band, axis=-1, keepdims=True) + eps)


# --- dynamic FC (FCD) -- the windowed alternative to the static fc_vector ---

def sliding_windows(X, window_size, step):
    """(T, N) -> (n_windows, window_size, N) overlapping windows (a gather, not
    a copy loop -- window_size/step must be static Python ints, not traced)."""
    n_windows = (X.shape[0] - window_size) // step + 1
    starts = step * jnp.arange(n_windows)
    idx = starts[:, None] + jnp.arange(window_size)[None, :]
    return X[idx]


def fcd_matrix(X, window_trs, step_trs, skip_t=0, eps=1e-8):
    """Dynamic-FC (FCD) matrix: correlation between per-window FC vectors.

    X: (T, N) BOLD. Returns (n_windows, n_windows) -- entry (i, j) is how
    similar window i's and window j's FC pattern are (the standard
    Hansen/Deco-style "FCD" used to characterize whole-brain-model dynamics,
    as opposed to fc_vector's single time-averaged FC).

    ``window_trs``/``step_trs`` are in TR units, not ms -- how many TRs that
    is in real time depends on ``tr_ms``, and how many TRs are even available
    depends on ``t1_bold``/``skip_t``, so this combination can go infeasible
    silently (e.g. raising ``tr_ms`` for a dataset with a longer TR shrinks
    the *simulated* TR count for the same t1_bold). Checked explicitly here
    (T, window_trs, step_trs are static Python ints at trace time -- this
    check costs nothing at trace/compile time) rather than left to fail deep
    inside JAX's gather lowering with a cryptic "slice size out of range".
    """
    T = X.shape[0] - skip_t
    n_windows = (T - window_trs) // step_trs + 1
    if n_windows < 2:
        raise ValueError(
            f"dfc: only {T} BOLD TRs available after skip_t={skip_t} (of {X.shape[0]} total), "
            f"not enough for even 2 windows of window_trs={window_trs} at step_trs={step_trs} "
            f"(got n_windows={n_windows}). This usually means t1_bold/tr_ms/bold_skip_trs no "
            "longer leaves enough simulated TRs for the configured window -- e.g. raising tr_ms "
            "to match your dataset's real TR shrinks the simulated TR count for the same t1_bold. "
            "Fix by lowering dfc_window_trs, lowering bold_skip_trs, or raising t1_bold."
        )
    windows = sliding_windows(X[skip_t:], window_trs, step_trs)  # (n_windows, window_trs, N)

    def window_fc_vec(w):
        wz = zscore_time(w, eps=eps)
        C = jnp.corrcoef(wz, rowvar=False)
        iu = jnp.triu_indices(C.shape[0], k=1)
        return C[iu]

    fc_vecs = jax.vmap(window_fc_vec)(windows)  # (n_windows, n_pairs)
    return jnp.corrcoef(fc_vecs, rowvar=True)   # (n_windows, n_windows)


def fcd_values(fcd, k_min=1):
    """Off-diagonal (``k_min`` and above) upper-triangle values of an
    ``(n_windows, n_windows)`` FCD matrix, as a flat vector."""
    iu = jnp.triu_indices(fcd.shape[0], k=k_min)
    return fcd[iu]


def soft_histogram(vals, centers, sigma=0.05, eps=1e-8):
    """Differentiable histogram via Gaussian kernels: each value's mass spreads
    across ``centers`` by a Gaussian of width ``sigma`` rather than falling into
    one hard bin, so (unlike ``jnp.histogram``, whose bin assignment has zero
    gradient almost everywhere) this stays usable inside a jax.grad'd loss.
    ``vals``: (P,), ``centers``: (B,) -> normalized (B,) histogram."""
    d = vals[:, None] - centers[None, :]
    w = jnp.exp(-0.5 * (d / sigma) ** 2)
    h = jnp.mean(w, axis=0)
    return h / (jnp.sum(h) + eps)


def wasserstein_1d_from_hist(p, q):
    """1-Wasserstein distance between two histograms sharing the same
    (equally-spaced) bin grid -- the mean absolute difference of their CDFs."""
    return jnp.mean(jnp.abs(jnp.cumsum(p) - jnp.cumsum(q)))


def dfc_histogram(X, window_trs, step_trs, centers, skip_t=0, k_min=1, sigma=0.05, eps=1e-8):
    """Soft-histogram-summarized distribution of FCD upper-triangle values.

    The simulated BOLD horizon (t1_bold, kept short to stay cheap per gradient
    step) is much shorter than the empirical recording, so their FCD matrices
    have different numbers of windows -- comparing entries pairwise isn't
    possible. Comparing the *distribution* of FCD values instead is
    window-count agnostic; the standard way to do that is a KS-distance
    between the two empirical CDFs, but a KS statistic (max of a step-function
    difference) has zero gradient almost everywhere and is a poor fit for
    gradient-based optimization. Binning both onto the same ``centers`` grid
    via ``soft_histogram`` and comparing those with ``wasserstein_1d_from_hist``
    is a smooth, shape-independent stand-in that can be used directly inside a
    jax.grad'd loss (see train.make_bold_loss_fn's dFC term).
    """
    fcd = fcd_matrix(X, window_trs, step_trs, skip_t=skip_t, eps=eps)
    values = fcd_values(fcd, k_min=k_min)
    return soft_histogram(values, centers, sigma=sigma, eps=eps)
