"""Diagnostic plots for the EEG+BOLD fit: node activity, PSD, FC, correlation.

Every function takes its inputs explicitly (simulator outputs, targets,
metadata) and returns the created ``Figure`` — none of them read module-level
state, so they can be called mid-fit, after the fact from saved arrays, or in
a notebook cell without re-running anything.
"""
from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .connectivity import fcd_matrix, fcd_values, filter_sim_bold, soft_histogram, wasserstein_1d_from_hist, zscore_time
from .forward import project_to_scalp
from .signal import compute_psd


def _zs(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def plot_node_activity(sim_result, mask_cortical, dt, settle_ms=500.0, stride_ms=4.0, n_show=5):
    """JR cortical output, WC subcortical E, and the shared network_out
    channel for a handful of representative nodes of each type."""
    settle, stride = int(settle_ms / dt), int(stride_ms / dt)
    t_plot = np.arange(sim_result.ys[settle::stride].shape[0]) * stride * dt

    jr_output = np.asarray(
        (sim_result.ys[settle::stride, 1] - sim_result.ys[settle::stride, 2]).T
    ) * np.asarray(mask_cortical)[:, None]
    wc_E = np.asarray(sim_result.ys[settle::stride, 6].T) * (1 - np.asarray(mask_cortical))[:, None]
    network_out = np.asarray(sim_result.ys[settle::stride, 8].T)

    cortical = np.where(np.asarray(mask_cortical) == 1)[0]
    subcortical = np.where(np.asarray(mask_cortical) == 0)[0]
    n = min(n_show, len(cortical), len(subcortical))
    cx, sx = cortical[:n], subcortical[:n]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    for node in cx:
        ax.plot(t_plot, jr_output[node], label=f"Cortex node {node}", alpha=0.8)
    ax.set_ylabel("JR output\n(y1 - y2) [mV]")
    ax.set_title("Cortical nodes — Jansen-Rit pyramidal output")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    for node in sx:
        ax.plot(t_plot, wc_E[node], label=f"Subcortex node {node}", alpha=0.8)
    ax.set_ylabel("WC excitatory\npopulation E [a.u.]")
    ax.set_title("Subcortical nodes — Wilson-Cowan E activity")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[2]
    for node in cx:
        ax.plot(t_plot, network_out[node], color="steelblue", alpha=0.5, linewidth=0.8)
    for node in sx:
        ax.plot(t_plot, network_out[node], color="tomato", alpha=0.5, linewidth=0.8)
    ax.plot([], [], color="steelblue", label="Cortex (JR normalized)")
    ax.plot([], [], color="tomato", label="Subcortex (WC E)")
    ax.set_ylabel("network_out [0, 1]")
    ax.set_xlabel("Time (ms)")
    ax.set_title("network_out — unified hemodynamic/coupling input (all nodes)")
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    return fig


def plot_bold_timeseries(sim_bold_2d, empirical_bold, mask_cortical, tr_ms, skip_t=0, n_show=4):
    """Simulated vs. empirical BOLD, z-scored, for a few cortical/subcortical nodes."""
    Xs = np.asarray(sim_bold_2d)[skip_t:, :]
    Xe = np.asarray(empirical_bold)[skip_t:, :]
    t_sim = np.arange(Xs.shape[0]) * (tr_ms / 1000.0)
    t_emp = np.arange(Xe.shape[0]) * (tr_ms / 1000.0)

    cortical = np.where(np.asarray(mask_cortical) == 1)[0]
    subcortical = np.where(np.asarray(mask_cortical) == 0)[0]
    n = min(n_show, len(cortical), len(subcortical))
    show_nodes = np.concatenate([cortical[:n], subcortical[:n]])

    fig, axes = plt.subplots(len(show_nodes), 1, figsize=(12, 2 * len(show_nodes)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, node in zip(axes, show_nodes):
        kind = "cortex" if mask_cortical[node] == 1 else "subcortex"
        ax.plot(t_sim, _zs(Xs[:, node]), color="steelblue", label="sim")
        if node < Xe.shape[1]:
            ax.plot(t_emp, _zs(Xe[:, node]), color="tomato", alpha=0.7, label="emp")
        ax.set_ylabel(f"node {node}\n({kind})")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Simulated vs empirical BOLD (z-scored)")
    plt.tight_layout()
    return fig


def plot_eeg_psd_comparison(sim_psd, target_psd, freqs, idx_min, idx_max, n_electrodes_show=5, log_scale=True):
    """Per-electrode and mean±std normalized PSD, simulated vs. empirical.

    ``log_scale=True`` (default) plots the y-axis (PSD) on a log scale;
    ``log_scale=False`` plots the same data on a linear y-axis instead.
    """
    sim_psd, target_psd = np.asarray(sim_psd), np.asarray(target_psd)
    sim_norm = sim_psd / (sim_psd[:, idx_min:idx_max].sum(keepdims=True) + 1e-8)
    target_norm = target_psd / (target_psd[:, idx_min:idx_max].sum(keepdims=True) + 1e-8)

    electrode_indices = np.linspace(0, sim_psd.shape[0] - 1, min(n_electrodes_show, sim_psd.shape[0]), dtype=int)
    xlim = (freqs[idx_min], freqs[idx_max - 1])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot = (lambda ax: ax.semilogy) if log_scale else (lambda ax: ax.plot)
    scale_label = "log" if log_scale else "linear"

    ax = axes[0]
    for i, ch in enumerate(electrode_indices):
        color = plt.cm.tab10(i / len(electrode_indices))
        plot(ax)(freqs[idx_min:idx_max], sim_norm[ch, idx_min:idx_max],
                 color=color, linewidth=1.5, label=f"Sim ch{ch}")
        plot(ax)(freqs[idx_min:idx_max], target_norm[ch, idx_min:idx_max],
                 color=color, linewidth=1.5, linestyle="--", alpha=0.6, label=f"Emp ch{ch}")
    ax.set_xlim(xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Normalized PSD ({scale_label})")
    ax.set_title("EEG PSD — simulated (solid) vs empirical (dashed)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    ax = axes[1]
    mean_sim, std_sim = sim_norm.mean(axis=0), sim_norm.std(axis=0)
    mean_target, std_target = target_norm.mean(axis=0), target_norm.std(axis=0)
    plot(ax)(freqs[idx_min:idx_max], mean_sim[idx_min:idx_max], color="steelblue", linewidth=2, label="Sim (mean)")
    ax.fill_between(freqs[idx_min:idx_max], (mean_sim - std_sim)[idx_min:idx_max],
                     (mean_sim + std_sim)[idx_min:idx_max], color="steelblue", alpha=0.2)
    plot(ax)(freqs[idx_min:idx_max], mean_target[idx_min:idx_max], color="tomato", linewidth=2,
             linestyle="--", label="Emp (mean)")
    ax.fill_between(freqs[idx_min:idx_max], (mean_target - std_target)[idx_min:idx_max],
                     (mean_target + std_target)[idx_min:idx_max], color="tomato", alpha=0.2)
    ax.set_xlim(xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Normalized PSD ({scale_label})")
    ax.set_title("EEG PSD — mean ± std across all electrodes")
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_fc_comparison(sim_bold_2d, empirical_bold, tr_ms, skip_t=0, eps=1e-8):
    """Empirical FC, simulated FC, and their upper-triangle scatter/correlation.

    ``sim_bold_2d`` is bandpassed (``connectivity.filter_sim_bold``) to match
    the band the empirical BOLD was already preprocessed with before FC is
    computed -- only the simulated side needs it.
    """
    Xs_raw = jnp.array(np.asarray(sim_bold_2d)[skip_t:, :])
    Xs_z = zscore_time(filter_sim_bold(Xs_raw, tr_ms), eps=eps)
    fc_sim = np.asarray(jnp.corrcoef(Xs_z, rowvar=False))
    Xe_z = np.asarray(zscore_time(jnp.array(np.asarray(empirical_bold)[skip_t:, :]), eps=eps))
    fc_emp = np.corrcoef(Xe_z, rowvar=False)

    iu = np.triu_indices(fc_sim.shape[0], k=1)
    fc_sim_vec, fc_emp_vec = fc_sim[iu], fc_emp[iu]
    fc_corr = np.corrcoef(fc_sim_vec, fc_emp_vec)[0, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title in ((axes[0], fc_emp, "Empirical FC"), (axes[1], fc_sim, "Simulated FC")):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("Region")
        ax.set_ylabel("Region")

    ax = axes[2]
    ax.scatter(fc_emp_vec, fc_sim_vec, alpha=0.15, s=3, color="steelblue", rasterized=True)
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Empirical FC")
    ax.set_ylabel("Simulated FC")
    ax.set_title(f"FC scatter\nPearson r = {fc_corr:.3f}")

    plt.tight_layout()
    return fig, fc_corr


def plot_fcd_comparison(sim_bold_2d, empirical_bold, tr_ms, window_trs, step_trs, skip_t=0,
                         k_min=1, n_bins=25, sigma=0.05, eps=1e-8):
    """Empirical FCD, simulated FCD matrices, and their dFC-value distribution
    comparison -- the dynamic-FC counterpart to plot_fc_comparison. Sim and
    empirical FCD matrices are generally different sizes (different BOLD
    horizons -> different window counts), so unlike plot_fc_comparison's
    scatter this compares the two as value histograms, reporting the same
    1-Wasserstein distance the "dfc" loss optimizes (see
    connectivity.dfc_histogram/wasserstein_1d_from_hist). ``sim_bold_2d`` is
    sliced to ``skip_t:`` and bandpassed (``connectivity.filter_sim_bold``)
    before windowing -- matching the order ``make_bold_dfc_loss_fn`` uses --
    the empirical side is already filtered upstream."""
    Xs = filter_sim_bold(jnp.array(np.asarray(sim_bold_2d))[skip_t:, :], tr_ms)
    Xe = jnp.array(np.asarray(empirical_bold))
    fcd_sim = np.asarray(fcd_matrix(Xs, window_trs, step_trs, skip_t=0, eps=eps))
    fcd_emp = np.asarray(fcd_matrix(Xe, window_trs, step_trs, skip_t=skip_t, eps=eps))

    vals_sim = np.asarray(fcd_values(jnp.array(fcd_sim), k_min=k_min))
    vals_emp = np.asarray(fcd_values(jnp.array(fcd_emp), k_min=k_min))

    centers = jnp.linspace(-1.0, 1.0, n_bins)
    hist_sim = soft_histogram(jnp.array(vals_sim), centers, sigma=sigma, eps=eps)
    hist_emp = soft_histogram(jnp.array(vals_emp), centers, sigma=sigma, eps=eps)
    w_dist = float(wasserstein_1d_from_hist(hist_sim, hist_emp))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title in ((axes[0], fcd_emp, f"Empirical FCD ({fcd_emp.shape[0]} windows)"),
                            (axes[1], fcd_sim, f"Simulated FCD ({fcd_sim.shape[0]} windows)")):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("Window")
        ax.set_ylabel("Window")

    ax = axes[2]
    bins = np.linspace(-1, 1, 60)
    ax.hist(vals_emp, bins=bins, density=True, alpha=0.5, color="tomato", label="Empirical")
    ax.hist(vals_sim, bins=bins, density=True, alpha=0.5, color="steelblue", label="Simulated")
    ax.set_xlabel("dFC (correlation)")
    ax.set_ylabel("Density")
    ax.set_title(f"dFC value distribution\nWasserstein-1 = {w_dist:.4f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig, w_dist


def plot_fcd_learning(sim_bold_2d_before, sim_bold_2d_after, empirical_bold, tr_ms, window_trs, step_trs,
                       skip_t=0, k_min=1, n_bins=25, sigma=0.05, eps=1e-8):
    """Before/after/empirical FCD matrices, plus their dFC-value distribution
    comparison -- the dfc counterpart to plot_bold_learning, extending
    plot_fcd_comparison's sim-vs-emp with the pre-training FCD too. Reports
    the same 1-Wasserstein distance the "dfc" loss optimizes. Both simulated
    (before/after) series are sliced to ``skip_t:`` and bandpassed
    (``connectivity.filter_sim_bold``) before windowing -- matching the order
    ``make_bold_dfc_loss_fn`` uses -- the empirical side is already filtered
    upstream."""
    Xb = filter_sim_bold(jnp.array(np.asarray(sim_bold_2d_before))[skip_t:, :], tr_ms)
    Xa = filter_sim_bold(jnp.array(np.asarray(sim_bold_2d_after))[skip_t:, :], tr_ms)
    Xe = jnp.array(np.asarray(empirical_bold))

    fcd_before = np.asarray(fcd_matrix(Xb, window_trs, step_trs, skip_t=0, eps=eps))
    fcd_after = np.asarray(fcd_matrix(Xa, window_trs, step_trs, skip_t=0, eps=eps))
    fcd_emp = np.asarray(fcd_matrix(Xe, window_trs, step_trs, skip_t=skip_t, eps=eps))

    vals_before = np.asarray(fcd_values(jnp.array(fcd_before), k_min=k_min))
    vals_after = np.asarray(fcd_values(jnp.array(fcd_after), k_min=k_min))
    vals_emp = np.asarray(fcd_values(jnp.array(fcd_emp), k_min=k_min))

    centers = jnp.linspace(-1.0, 1.0, n_bins)
    hist_before = soft_histogram(jnp.array(vals_before), centers, sigma=sigma, eps=eps)
    hist_after = soft_histogram(jnp.array(vals_after), centers, sigma=sigma, eps=eps)
    hist_emp = soft_histogram(jnp.array(vals_emp), centers, sigma=sigma, eps=eps)
    w_dist_before = float(wasserstein_1d_from_hist(hist_before, hist_emp))
    w_dist_after = float(wasserstein_1d_from_hist(hist_after, hist_emp))

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, mat, title in (
        (axes[0], fcd_before, f"Before FCD ({fcd_before.shape[0]} windows)"),
        (axes[1], fcd_after, f"After FCD ({fcd_after.shape[0]} windows)"),
        (axes[2], fcd_emp, f"Empirical FCD ({fcd_emp.shape[0]} windows)"),
    ):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("Window")
        ax.set_ylabel("Window")

    ax = axes[3]
    bins = np.linspace(-1, 1, 60)
    ax.hist(vals_emp, bins=bins, density=True, alpha=0.5, color="tomato", label="Empirical target")
    ax.hist(vals_before, bins=bins, density=True, histtype="step", color="gray",
            linestyle=":", linewidth=2, label="Before (init params)")
    ax.hist(vals_after, bins=bins, density=True, histtype="step", color="steelblue",
            linewidth=2, label="After (fitted params)")
    ax.set_xlabel("dFC (correlation)")
    ax.set_ylabel("Density")
    ax.set_title(f"dFC value distribution\nWasserstein-1: before={w_dist_before:.4f}, after={w_dist_after:.4f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig, w_dist_before, w_dist_after


def plot_eeg_psd_learning(psd_before, psd_after, target_psd, freqs, idx_min, idx_max, log_scale=True):
    """Mean +/- std normalized PSD across electrodes: before vs after
    training, against the empirical target -- shows whether the fit actually
    moved the simulated spectrum toward the target.

    ``log_scale=True`` (default) plots the y-axis (PSD) on a log scale;
    ``log_scale=False`` plots the same data on a linear y-axis instead.
    """
    def _norm(p):
        p = np.asarray(p)
        return p / (p[:, idx_min:idx_max].sum(keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot = ax.semilogy if log_scale else ax.plot
    scale_label = "log" if log_scale else "linear"
    for norm, color, ls, label in (
        (_norm(psd_before), "gray", ":", "Before (init params)"),
        (_norm(psd_after), "steelblue", "-", "After (fitted params)"),
        (_norm(target_psd), "tomato", "--", "Empirical target"),
    ):
        mean, std = norm.mean(axis=0), norm.std(axis=0)
        plot(freqs[idx_min:idx_max], mean[idx_min:idx_max], color=color, linestyle=ls,
             linewidth=2, label=label)
        ax.fill_between(freqs[idx_min:idx_max], (mean - std)[idx_min:idx_max],
                         (mean + std)[idx_min:idx_max], color=color, alpha=0.15)

    ax.set_xlim(freqs[idx_min], freqs[idx_max - 1])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Normalized PSD ({scale_label})")
    ax.set_title("EEG PSD — before vs after training (mean ± std across electrodes)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_bold_learning(sim_bold_2d_before, sim_bold_2d_after, empirical_bold, mask_cortical,
                        tr_ms, skip_t=0, n_show=4):
    """Simulated BOLD before vs after training, vs empirical, z-scored, per node --
    the BOLD-side counterpart to plot_eeg_psd_learning."""
    Xb = np.asarray(sim_bold_2d_before)[skip_t:, :]
    Xa = np.asarray(sim_bold_2d_after)[skip_t:, :]
    Xe = np.asarray(empirical_bold)[skip_t:, :]
    t_b = np.arange(Xb.shape[0]) * (tr_ms / 1000.0)
    t_a = np.arange(Xa.shape[0]) * (tr_ms / 1000.0)
    t_e = np.arange(Xe.shape[0]) * (tr_ms / 1000.0)

    cortical = np.where(np.asarray(mask_cortical) == 1)[0]
    subcortical = np.where(np.asarray(mask_cortical) == 0)[0]
    n = min(n_show, len(cortical), len(subcortical))
    show_nodes = np.concatenate([cortical[:n], subcortical[:n]])

    fig, axes = plt.subplots(len(show_nodes), 1, figsize=(12, 2 * len(show_nodes)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, node in zip(axes, show_nodes):
        kind = "cortex" if mask_cortical[node] == 1 else "subcortex"
        ax.plot(t_b, _zs(Xb[:, node]), color="gray", linestyle=":", alpha=0.8, label="before")
        ax.plot(t_a, _zs(Xa[:, node]), color="steelblue", label="after")
        if node < Xe.shape[1]:
            ax.plot(t_e, _zs(Xe[:, node]), color="tomato", alpha=0.7, linestyle="--", label="emp")
        ax.set_ylabel(f"node {node}\n({kind})")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Simulated BOLD before vs after training (z-scored)")
    plt.tight_layout()
    return fig


def plot_eeg_corr_comparison(simulated_eeg, empirical_chunks):
    """Empirical vs. simulated inter-electrode correlation matrices."""
    sim_corr = np.asarray(jnp.corrcoef(jnp.array(simulated_eeg)))
    emp_corr = np.asarray(np.stack(list(map(jnp.corrcoef, empirical_chunks))).mean(axis=0))

    iu = np.triu_indices(sim_corr.shape[0], k=1)
    r = np.corrcoef(sim_corr[iu], emp_corr[iu])[0, 1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, mat, title in ((axes[0], emp_corr, "Empirical EEG correlation"),
                            (axes[1], sim_corr, "Simulated EEG correlation")):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")

    ax = axes[2]
    ax.scatter(emp_corr[iu], sim_corr[iu], alpha=0.15, s=3, color="steelblue", rasterized=True)
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Empirical")
    ax.set_ylabel("Simulated")
    ax.set_title(f"EEG correlation scatter\nPearson r = {r:.3f}")

    plt.tight_layout()
    return fig, r
