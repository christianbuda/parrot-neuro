"""Per-channel signal-quality metrics, pure numpy/scipy (no jax -- this never
needs to be differentiated, unlike ``optimization.signal``).

Each metric is computed once across all of a task's splice-free segments and
returned as one array per channel, bundled in :class:`ChannelMetrics`. Kept
independent of any single "bad channel" opinion -- :mod:`.flags` turns these
into pass/warn/fail calls, so thresholds can change without recomputing stats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal
import scipy.stats

# Mirrors optimization.config.FMIN/FMAX (1-40 Hz is this project's EEG band of
# interest for the LEMON data); HF_BAND is the top of that band, where muscle/
# electrode noise typically dominates over genuine cortical rhythms.
BAND = (1.0, 40.0)
HF_BAND = (30.0, 40.0)
LINE_FREQ = 50.0


@dataclass
class ChannelMetrics:
    channel_names: list[str]
    sfreq: float
    freqs: np.ndarray
    psd: np.ndarray  # (n_channels, n_freqs), Welch, averaged over segments

    rms: np.ndarray
    flatline_fraction: np.ndarray
    kurtosis: np.ndarray
    hf_noise_ratio: np.ndarray
    line_noise_ratio: np.ndarray
    neighbor_corr: np.ndarray
    segment_std_cv: np.ndarray


def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD), falling back to std when MAD collapses
    (e.g. more than half the channels tied at the same value)."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - med))
    if scale < 1e-12:
        scale = np.std(x) + 1e-12
    return (x - med) / scale


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-12
    return (x - mu) / sd


def _neighbor_corr_global(segments: list[np.ndarray]) -> np.ndarray:
    """Each channel's correlation with the (leave-one-out) mean of every
    other channel, per segment, averaged. Fallback for when electrode
    positions aren't available -- see :func:`_neighbor_corr_spatial` for why
    this is otherwise the wrong comparison to make."""
    n_channels = segments[0].shape[0]
    per_seg = []
    for seg in segments:
        others_mean = (seg.sum(axis=0, keepdims=True) - seg) / (n_channels - 1)
        a = _zscore_rows(seg)
        b = _zscore_rows(others_mean)
        per_seg.append((a * b).mean(axis=1))  # both unit-variance -> mean product = Pearson r
    return np.mean(per_seg, axis=0)


def _spatial_neighbor_indices(channel_names, positions, k):
    """Index (into ``channel_names``) of each positioned channel's ``k``
    nearest neighbors by 3D scalp distance, and the list of positioned
    indices themselves."""
    positioned = [i for i, n in enumerate(channel_names) if n in positions]
    coords = np.stack([positions[channel_names[i]] for i in positioned])
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    kk = min(k, len(positioned) - 1)
    order_local = np.argsort(dist, axis=1)[:, 1 : kk + 1]  # column 0 is self (dist 0)
    neighbor_idx = np.asarray(positioned)[order_local]  # (n_positioned, kk) -> full-array indices
    return positioned, neighbor_idx


def _neighbor_corr_spatial(segments: list[np.ndarray], channel_names, positions, k: int = 6) -> np.ndarray:
    """Each channel's correlation with the mean of its ``k`` spatially
    *nearest* electrodes (not the whole-scalp average). A global average
    reference systematically under-correlates whole regions when the signal
    isn't spatially uniform (e.g. eyes-closed posterior alpha dominance makes
    frontal channels look "bad" purely from anatomy/physiology, not a
    hardware fault) -- comparing against local neighbors instead keeps the
    baseline regionally fair, so a low score is actually evidence of a bad
    electrode. Channels missing a position fall back to the whole-scalp
    (positioned) average, since they have no defined neighborhood; they're
    already flagged separately for lacking a position."""
    n_channels = len(channel_names)
    positioned, neighbor_idx = _spatial_neighbor_indices(channel_names, positions, k)
    unpositioned = [i for i in range(n_channels) if i not in positioned]

    per_seg = []
    for seg in segments:
        a = _zscore_rows(seg)
        corr = np.zeros(n_channels)
        for local_i, ch_i in enumerate(positioned):
            neighbor_mean = seg[neighbor_idx[local_i]].mean(axis=0, keepdims=True)
            b = _zscore_rows(neighbor_mean)[0]
            corr[ch_i] = (a[ch_i] * b).mean()
        if unpositioned:
            global_mean = seg[positioned].mean(axis=0, keepdims=True)
            b = _zscore_rows(global_mean)[0]
            for ch_i in unpositioned:
                corr[ch_i] = (a[ch_i] * b).mean()
        per_seg.append(corr)
    return np.mean(per_seg, axis=0)


def _welch_psd(segments: list[np.ndarray], sfreq: float):
    """Welch PSD per segment (own boundaries -- never pooled across the
    splice points), averaged. A fixed ``nperseg`` (capped by the shortest
    segment) keeps every segment's frequency grid identical so the average is
    well defined."""
    min_len = min(seg.shape[1] for seg in segments)
    nperseg = int(min(4 * sfreq, min_len))
    psds = []
    freqs = None
    for seg in segments:
        freqs, p = scipy.signal.welch(seg, fs=sfreq, nperseg=nperseg, axis=1)
        psds.append(p)
    return freqs, np.mean(psds, axis=0)


def _line_noise_ratio(psd, freqs, line_freq, halfwidth=1.0, guard=1.5, baseline_width=3.0, eps=1e-20):
    """Power right at ``line_freq`` vs. a nearby baseline just outside it --
    a sharp peak relative to its own neighbourhood, regardless of a channel's
    overall power, is the powerline-contamination signature."""
    peak_mask = np.abs(freqs - line_freq) <= halfwidth
    baseline_mask = (np.abs(freqs - line_freq) > guard) & (np.abs(freqs - line_freq) <= guard + baseline_width)
    if not peak_mask.any() or not baseline_mask.any():
        return np.zeros(psd.shape[0])
    peak = psd[:, peak_mask].mean(axis=1)
    baseline = psd[:, baseline_mask].mean(axis=1)
    return peak / (baseline + eps)


def compute_channel_metrics(
    segments: list[np.ndarray],
    channel_names: list[str],
    sfreq: float,
    positions: dict[str, np.ndarray] | None = None,
    n_neighbors: int = 6,
    band=BAND,
    hf_band=HF_BAND,
    line_freq: float = LINE_FREQ,
    eps: float = 1e-20,
) -> ChannelMetrics:
    concat = np.concatenate(segments, axis=1)  # (n_channels, total_samples)

    rms = np.sqrt(np.mean(concat**2, axis=1))

    diffs = np.abs(np.diff(concat, axis=1))
    nonzero = diffs[diffs > 0]
    flat_thresh = 0.01 * np.median(nonzero) if nonzero.size else 0.0
    flatline_fraction = np.mean(diffs <= flat_thresh, axis=1)

    kurtosis = scipy.stats.kurtosis(concat, axis=1, fisher=True, bias=False)

    freqs, psd = _welch_psd(segments, sfreq)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])
    total_power = psd[:, band_mask].sum(axis=1)
    hf_power = psd[:, hf_mask].sum(axis=1)
    hf_noise_ratio = hf_power / (total_power + eps)

    line_noise_ratio = _line_noise_ratio(psd, freqs, line_freq)
    if positions:
        neighbor_corr = _neighbor_corr_spatial(segments, channel_names, positions, k=n_neighbors)
    else:
        neighbor_corr = _neighbor_corr_global(segments)

    seg_stds = np.stack([seg.std(axis=1) for seg in segments], axis=0)  # (n_segments, n_channels)
    segment_std_cv = seg_stds.std(axis=0) / (seg_stds.mean(axis=0) + eps)

    return ChannelMetrics(
        channel_names=list(channel_names),
        sfreq=sfreq,
        freqs=freqs,
        psd=psd,
        rms=rms,
        flatline_fraction=flatline_fraction,
        kurtosis=kurtosis,
        hf_noise_ratio=hf_noise_ratio,
        line_noise_ratio=line_noise_ratio,
        neighbor_corr=neighbor_corr,
        segment_std_cv=segment_std_cv,
    )
