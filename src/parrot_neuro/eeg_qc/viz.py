"""Diagnostic plots for channel QC: raw traces, PSDs, scalp topographies,
inter-channel correlation, and a sorted summary — every function takes its
inputs explicitly and returns the created ``Figure`` (same convention as
``optimization.viz``), so they compose in a notebook without re-running
anything.

Color is used deliberately, not decoratively: pass/warn/fail always maps to
the same three colors (green/amber/red, never reused for anything else),
per-channel magnitude maps (topomaps) use a single-hue sequential ramp, and
the one diverging quantity (correlation, signed -1..1) uses the same
blue/red diverging pair the rest of the codebase's FC/correlation plots use
(``optimization.viz``'s ``RdBu_r``) for visual consistency across the project.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import griddata

from .checks import FAIL, PASS, WARN
from .metrics import ChannelMetrics, robust_z

STATUS_COLOR = {PASS: "#0ca30c", WARN: "#fab219", FAIL: "#d03b3b"}
MISSING_COLOR = "#898781"  # muted gray -- an expected-but-absent electrode is a fact, not a severity
SEQUENTIAL_CMAP = "Blues"


def _status_of(flags, name: str) -> str:
    for f in flags:
        if f.name == name:
            return f.status
    return PASS


def _project_positions(channel_names, positions: dict[str, np.ndarray]):
    """Azimuthal-equidistant projection of each channel's 3D scalp position
    onto 2D (vertex = origin, nose/anterior = +y, right ear = +x) -- the
    standard EEG topomap layout. Centered on the centroid of the channels
    actually being plotted (a reasonable stand-in for the head center, since
    the raw coordinates are in subject/world space, not head-centered)."""
    names = [n for n in channel_names if n in positions]
    pts = np.stack([positions[n] for n in names])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    r = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.clip(r, 1e-9, None)
    theta = np.arccos(np.clip(unit[:, 2], -1, 1))  # angle from vertex (z = superior)
    phi = np.arctan2(unit[:, 1], unit[:, 0])  # y = anterior, x = right
    xy = np.stack([theta * np.cos(phi), theta * np.sin(phi)], axis=1)
    return names, xy


def _draw_head_outline(ax, radius):
    ax.add_patch(Circle((0, 0), radius, fill=False, linewidth=1.2, color="#52514e", zorder=3))
    nose = radius * 0.12
    ax.plot([-nose, 0, nose], [radius, radius + nose * 1.4, radius], color="#52514e", linewidth=1.2, zorder=3)


def plot_channel_timeseries(segment, sfreq, channel_names, flags, seconds=10.0, title=None):
    """Stacked per-channel traces (first ``seconds`` of one segment), colored
    by QC status -- the fastest sanity check of "does this look like EEG"."""
    n_samples = min(int(seconds * sfreq), segment.shape[1])
    t = np.arange(n_samples) / sfreq
    x = segment[:, :n_samples]

    scale = np.median(np.std(x, axis=1)) + 1e-12
    offset_step = 6 * scale
    n_channels = len(channel_names)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.18 * n_channels)))
    for i, name in enumerate(channel_names):
        offset = (n_channels - 1 - i) * offset_step
        status = _status_of(flags, name)
        color = STATUS_COLOR[status]
        lw, alpha, z = (1.6, 0.95, 2) if status != PASS else (0.8, 0.6, 1)
        ax.plot(t, x[i] + offset, color=color, linewidth=lw, alpha=alpha, zorder=z)
        ax.text(-0.01 * t[-1], offset, name, ha="right", va="center", fontsize=6, color="#52514e")

    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, t[-1])
    ax.set_title(title or f"Channel traces (first {seconds:.0f}s) — bad channels in color")
    for status in (WARN, FAIL):
        ax.plot([], [], color=STATUS_COLOR[status], label=status)
    if any(_status_of(flags, n) != PASS for n in channel_names):
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def plot_channel_psd(metrics: ChannelMetrics, flags, fmax=60.0, line_freq=50.0):
    """Every channel's PSD (log-power), good channels in translucent gray and
    flagged channels in their status color + name label -- makes noisy
    channels' spectral signature (broadband HF energy, a line-noise spike)
    visually obvious against the "normal" spread."""
    freqs, psd = metrics.freqs, metrics.psd
    fmask = freqs <= fmax
    log_psd = 10 * np.log10(psd + 1e-20)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, name in enumerate(metrics.channel_names):
        status = _status_of(flags, name)
        if status == PASS:
            ax.plot(freqs[fmask], log_psd[i, fmask], color="#c3c2b7", linewidth=0.8, alpha=0.6, zorder=1)
    for i, name in enumerate(metrics.channel_names):
        status = _status_of(flags, name)
        if status != PASS:
            ax.plot(freqs[fmask], log_psd[i, fmask], color=STATUS_COLOR[status], linewidth=1.4,
                     label=name, zorder=2)

    mean_psd = log_psd[:, fmask].mean(axis=0)
    ax.plot(freqs[fmask], mean_psd, color="#0b0b0b", linewidth=1.6, linestyle="--", label="mean", zorder=3)
    ax.axvline(line_freq, color="#eb6834", linewidth=1.0, linestyle=":", alpha=0.8, zorder=0)
    ax.text(line_freq, ax.get_ylim()[1], f" {line_freq:.0f} Hz", fontsize=7, color="#eb6834", va="top")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Per-channel PSD — flagged channels labeled")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    return fig


def plot_montage_status(channel_names, positions, flags, missing=None):
    """Scalp map of every recorded channel colored by QC status; electrodes
    the montage has a position for but that weren't recorded (``missing``,
    e.g. from :func:`~.flags.missing_channels`) are drawn as muted gray
    "missing" markers -- the single figure that answers both "which channels
    are noisy" and "which channels are we not even recording" at a glance."""
    all_names, all_xy = _project_positions(list(positions.keys()), positions)
    xy_by_name = dict(zip(all_names, all_xy))

    rec_names, rec_xy = _project_positions(channel_names, positions)
    radius = max(np.linalg.norm(rec_xy, axis=1).max() * 1.15, 1.0) if len(rec_xy) else 1.0

    missing = missing or []
    unpositioned = [n for n in channel_names if n not in positions]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _draw_head_outline(ax, radius)

    for name in missing:
        if name in xy_by_name:
            x, y = xy_by_name[name]
            ax.scatter(x, y, s=70, facecolor="none", edgecolor=MISSING_COLOR, linewidth=1.4,
                       marker="x", zorder=2)
            ax.text(x, y - radius * 0.045, name, ha="center", va="top", fontsize=6, color=MISSING_COLOR)

    for name, (x, y) in zip(rec_names, rec_xy):
        status = _status_of(flags, name)
        ax.scatter(x, y, s=90, facecolor=STATUS_COLOR[status], edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(x, y + radius * 0.045, name, ha="center", va="bottom", fontsize=6, color="#0b0b0b")

    for row, name in enumerate(unpositioned):
        ax.text(0.02, 0.02 - 0.03 * row, f"✕ {name}: no position",
                 transform=ax.transAxes, fontsize=7, color=STATUS_COLOR[FAIL])

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c, markeredgecolor="white",
                          markersize=8, label=lbl)
               for lbl, c in (("pass", STATUS_COLOR[PASS]), ("warn", STATUS_COLOR[WARN]),
                              ("fail", STATUS_COLOR[FAIL]))]
    if missing:
        handles.append(plt.Line2D([0], [0], marker="x", linestyle="", markeredgecolor=MISSING_COLOR,
                                  markersize=8, label="missing (not recorded)"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_xlim(-radius * 1.3, radius * 1.3)
    ax.set_ylim(-radius * 1.3, radius * 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Electrode status (top-down, nose up)")
    plt.tight_layout()
    return fig


def plot_metric_topomaps(metrics: ChannelMetrics, positions, flags):
    """Small-multiples scalp topography of the metrics that vary meaningfully
    in space (amplitude, HF-noise ratio, neighbor correlation, line-noise
    ratio) -- a single glance for whether bad channels cluster (e.g. one
    noisy region of the cap) or are scattered (independent bad contacts)."""
    names, xy = _project_positions(metrics.channel_names, positions)
    idx = [metrics.channel_names.index(n) for n in names]
    radius = max(np.linalg.norm(xy, axis=1).max() * 1.15, 1.0)

    grid_x, grid_y = np.mgrid[-radius:radius:200j, -radius:radius:200j]
    mask = grid_x**2 + grid_y**2 <= radius**2

    panels = [
        ("RMS amplitude (robust z)", np.abs(robust_z(metrics.rms))[idx], "Blues"),
        ("HF-noise ratio", metrics.hf_noise_ratio[idx], "Blues"),
        ("Neighbor correlation", metrics.neighbor_corr[idx], "Blues_r"),
        ("Line-noise ratio", metrics.line_noise_ratio[idx], "Blues"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, (title, values, cmap) in zip(axes, panels):
        grid_z = griddata(xy, values, (grid_x, grid_y), method="cubic")
        grid_z = np.where(mask, grid_z, np.nan)
        im = ax.contourf(grid_x, grid_y, grid_z, levels=12, cmap=cmap)
        ax.scatter(xy[:, 0], xy[:, 1], s=14, color="#0b0b0b", zorder=3)
        _draw_head_outline(ax, radius)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlim(-radius * 1.3, radius * 1.3)
        ax.set_ylim(-radius * 1.3, radius * 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    return fig


def plot_correlation_matrix(segments, channel_names, flags):
    """Mean inter-channel correlation matrix across segments (RdBu_r, matching
    the diverging convention ``optimization.viz`` uses for FC/correlation
    plots elsewhere in this project), with flagged channels' tick labels
    colored by status so a bad channel's off-pattern row/column is easy to spot."""
    corrs = [np.corrcoef(seg) for seg in segments]
    mean_corr = np.mean(corrs, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mean_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(channel_names)))
    ax.set_yticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names, fontsize=5, rotation=90)
    ax.set_yticklabels(channel_names, fontsize=5)
    for tick_labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for label in tick_labels:
            status = _status_of(flags, label.get_text())
            if status != PASS:
                label.set_color(STATUS_COLOR[status])
                label.set_fontweight("bold")

    ax.set_title("Mean inter-channel correlation (segment-averaged)")
    plt.tight_layout()
    return fig


def plot_summary_bar(metrics: ChannelMetrics, flags):
    """One composite badness score per channel (count of triggered criteria),
    sorted worst-first -- the executive-summary figure: which channels to
    actually go look at."""
    n_criteria = np.array([len(f.reasons) for f in flags])
    order = np.argsort(-n_criteria, kind="stable")

    names = [flags[i].name for i in order]
    counts = n_criteria[order]
    colors = [STATUS_COLOR[flags[i].status] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.18 * len(names))))
    ax.barh(range(len(names)), counts, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Number of QC criteria triggered")
    ax.set_title("Channels ranked by QC criteria triggered")
    ax.set_xlim(0, max(1, counts.max()) + 1)

    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLOR[s], label=s) for s in (PASS, WARN, FAIL)]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig
