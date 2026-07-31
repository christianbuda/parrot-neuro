"""Post-fit diagnostic plots: BOLD FC/timeseries/learning + EEG PSD/correlation/learning.

Owns the "which plots, saved where" orchestration on top of ``viz.py``'s plot
primitives -- shared between the training CLI (``examples/eeg_bold_fit_cli.py``,
called right after ``pipeline.fit``) and a diagnostics-only script that
re-simulates from a previously saved ``optimized_params.npz`` without
re-training (``examples/postfit_diagnostics_cli.py``).
"""
from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

from . import connectivity, train, viz
from .forward import project_to_scalp
from .signal import compute_psd


def run_and_save(ctx, diff_params, static_params, dataset, out_dir) -> None:
    """Simulate with ``diff_params``/``static_params`` and save every diagnostic plot.

    ``ctx`` is a ``pipeline.ExperimentContext`` (from ``pipeline.build_context``).
    ``diff_params``/``static_params`` need not be ``ctx``'s own fit result --
    e.g. ``postfit_diagnostics_cli.py`` reconstructs ``diff_params`` from a
    saved ``optimized_params.npz`` instead of actually re-fitting.
    ``dataset`` must already be loaded (the subject's EEG chunks): the EEG
    diagnostics always need it, regardless of whether EEG was a fit target.
    """
    cfg = ctx.cfg
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- simulation + BOLD (always available) ---
    combined = eqx.combine(diff_params, static_params)
    sim_result_eeg = ctx.simulators.simulator_eeg(combined)
    sim_result_bold = ctx.simulators.simulator_bold(combined)

    fig = viz.plot_node_activity(sim_result_eeg, ctx.mask_cortical, cfg.dt)
    fig.savefig(out_dir / "node_activity.png", dpi=150)

    sim_bold_2d = connectivity.extract_bold_2d(ctx.simulators.bold_monitor(sim_result_bold))
    # Free the raw per-ms trajectory (~23GiB at atlas=1000/t1_bold=320s) now that
    # only the much smaller TR-downsampled sim_bold_2d is needed -- this call is
    # forward-only (no jax.grad), so solver_block_size doesn't shrink it, and
    # leaving it referenced through the rest of this function would keep it
    # resident alongside sim_result_bold_init's own ~23GiB trajectory below.
    # MUST happen before the first np.asarray()/materializing call below --
    # jax's async dispatch means sim_result_bold's buffer isn't reclaimable
    # until its last reference is actually dropped, so a `del` placed after a
    # forcing call (as an earlier version of this code mistakenly did) is too
    # late to help that exact call.
    del sim_result_bold

    fig = viz.plot_bold_timeseries(sim_bold_2d, ctx.sc.empirical_bold, ctx.mask_cortical,
                                    cfg.tr_ms, skip_t=cfg.bold_skip_trs)
    fig.savefig(out_dir / "bold_timeseries.png", dpi=150)

    if cfg.bold_dfc_weight > 0:
        fig, dfc_w_dist = viz.plot_fcd_comparison(sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms,
                                                    cfg.dfc_window_trs, cfg.dfc_step_trs,
                                                    skip_t=cfg.bold_skip_trs, k_min=cfg.dfc_kmin,
                                                    n_bins=cfg.dfc_n_bins, sigma=cfg.dfc_sigma)
        fig.savefig(out_dir / "fcd_comparison.png", dpi=150)
        print(f"dFC Wasserstein-1 distance (sim vs emp): {dfc_w_dist:.5f}")
    fig, fc_corr = viz.plot_fc_comparison(sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms, skip_t=cfg.bold_skip_trs)
    fig.savefig(out_dir / "fc_comparison.png", dpi=150)
    print(f"FC Pearson correlation (sim vs emp): {fc_corr:.4f}")

    # --- BOLD learning (first iteration vs last) ---
    combined_init = eqx.combine(ctx.diff_params_init, ctx.static_params)
    sim_result_eeg_init = ctx.simulators.simulator_eeg(combined_init)
    sim_result_bold_init = ctx.simulators.simulator_bold(combined_init)
    sim_bold_2d_init = connectivity.extract_bold_2d(ctx.simulators.bold_monitor(sim_result_bold_init))
    del sim_result_bold_init  # same reasoning + ordering as the del above

    fig = viz.plot_bold_learning(sim_bold_2d_init, sim_bold_2d, ctx.sc.empirical_bold,
                                  ctx.mask_cortical, cfg.tr_ms, skip_t=cfg.bold_skip_trs)
    fig.savefig(out_dir / "bold_learning.png", dpi=150)

    if cfg.bold_dfc_weight > 0:
        fig, dfc_w_dist_before, dfc_w_dist_after = viz.plot_fcd_learning(
            sim_bold_2d_init, sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms,
            cfg.dfc_window_trs, cfg.dfc_step_trs, skip_t=cfg.bold_skip_trs,
            k_min=cfg.dfc_kmin, n_bins=cfg.dfc_n_bins, sigma=cfg.dfc_sigma,
        )
        fig.savefig(out_dir / "fcd_learning.png", dpi=150)
        print(f"dFC Wasserstein-1 distance: before={dfc_w_dist_before:.5f}  after={dfc_w_dist_after:.5f}")

    # --- EEG ---
    target_psd = ctx.target_psd if ctx.target_psd is not None else train.compute_target_psd(dataset)

    source_activity = (
        sim_result_eeg.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 1].T
        - sim_result_eeg.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 2].T
    ) * jnp.atleast_2d(ctx.mask_cortical).T
    simulated_eeg = project_to_scalp(
        source_activity, dataset.channel_indices, ctx.leadfield, ctx.smoothing_blocks, ctx.dipole_labels
    )
    sim_psd = compute_psd(simulated_eeg)

    fig = viz.plot_eeg_psd_comparison(sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max)
    fig.savefig(out_dir / "eeg_psd_comparison.png", dpi=150)

    fig = viz.plot_eeg_psd_comparison(sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max, log_scale=False)
    fig.savefig(out_dir / "eeg_psd_comparison_linear.png", dpi=150)

    fig, r_eeg = viz.plot_eeg_corr_comparison(simulated_eeg, dataset._chunks)
    fig.savefig(out_dir / "eeg_corr_comparison.png", dpi=150)
    print(f"EEG correlation matrix Pearson r: {r_eeg:.4f}")

    source_activity_init = (
        sim_result_eeg_init.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 1].T
        - sim_result_eeg_init.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 2].T
    ) * jnp.atleast_2d(ctx.mask_cortical).T
    simulated_eeg_init = project_to_scalp(
        source_activity_init, dataset.channel_indices, ctx.leadfield, ctx.smoothing_blocks, ctx.dipole_labels
    )
    sim_psd_init = compute_psd(simulated_eeg_init)

    fig = viz.plot_eeg_psd_learning(sim_psd_init, sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max)
    fig.savefig(out_dir / "eeg_psd_learning.png", dpi=150)

    fig = viz.plot_eeg_psd_learning(sim_psd_init, sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max,
                                     log_scale=False)
    fig.savefig(out_dir / "eeg_psd_learning_linear.png", dpi=150)

    print(f"diagnostics saved to {out_dir}")
