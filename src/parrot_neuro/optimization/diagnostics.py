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


def run_and_save(ctx, diff_params, static_params, dataset, out_dir) -> dict:
    """Simulate with ``diff_params``/``static_params`` and save every diagnostic plot.

    ``ctx`` is a ``pipeline.ExperimentContext`` (from ``pipeline.build_context``).
    ``diff_params``/``static_params`` need not be ``ctx``'s own fit result --
    e.g. ``postfit_diagnostics_cli.py`` reconstructs ``diff_params`` from a
    saved ``optimized_params.npz`` instead of actually re-fitting.
    ``dataset`` must already be loaded (the subject's EEG chunks): the EEG
    diagnostics always need it, regardless of whether EEG was a fit target.

    Returns ``{"metrics": {...}, "figures": {...}}`` -- the same scalar
    comparison metrics and figure paths this already prints/saves, handed
    back for a caller that wants to log them elsewhere (e.g. wandb) without
    duplicating this function's plotting logic. Existing callers that ignore
    the return value are unaffected.
    """
    cfg = ctx.cfg
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}
    figures: dict = {}

    # --- simulation + BOLD (always available) ---
    combined = eqx.combine(diff_params, static_params)
    sim_result_eeg = ctx.simulators.simulator_eeg(combined)
    sim_result_bold = ctx.simulators.simulator_bold(combined)

    fig = viz.plot_node_activity(sim_result_eeg, ctx.mask_cortical, cfg.dt)
    figures["node_activity"] = out_dir / "node_activity.png"
    fig.savefig(figures["node_activity"], dpi=150)

    # simulator_bold already streams the BOLD forward model (HRF convolution
    # or Balloon-Windkessel ODE integration, per cfg.bold_model -- see
    # train.build_simulators), so sim_result_bold is already the small
    # [n_bold, n_voi, n_nodes] buffer, not the full raw per-ms trajectory --
    # no separate bold_monitor(...) call, and no need to `del` it early for
    # memory (the old ~23GiB-per-call trajectory this used to guard against
    # doesn't get materialized at all anymore).
    sim_bold_2d = connectivity.extract_bold_2d(sim_result_bold)

    fig = viz.plot_bold_timeseries(sim_bold_2d, ctx.sc.empirical_bold, ctx.mask_cortical,
                                    cfg.tr_ms, skip_t=cfg.bold_skip_trs)
    figures["bold_timeseries"] = out_dir / "bold_timeseries.png"
    fig.savefig(figures["bold_timeseries"], dpi=150)

    if cfg.bold_dfc_weight > 0:
        fig, dfc_w_dist = viz.plot_fcd_comparison(sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms,
                                                    cfg.dfc_window_trs, cfg.dfc_step_trs,
                                                    skip_t=cfg.bold_skip_trs, k_min=cfg.dfc_kmin,
                                                    n_bins=cfg.dfc_n_bins, sigma=cfg.dfc_sigma)
        figures["fcd_comparison"] = out_dir / "fcd_comparison.png"
        fig.savefig(figures["fcd_comparison"], dpi=150)
        metrics["dfc_w_dist"] = dfc_w_dist
        print(f"dFC Wasserstein-1 distance (sim vs emp): {dfc_w_dist:.5f}")
    fig, fc_corr = viz.plot_fc_comparison(sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms, skip_t=cfg.bold_skip_trs)
    figures["fc_comparison"] = out_dir / "fc_comparison.png"
    fig.savefig(figures["fc_comparison"], dpi=150)
    metrics["fc_corr"] = fc_corr
    print(f"FC Pearson correlation (sim vs emp): {fc_corr:.4f}")

    # --- BOLD learning (first iteration vs last) ---
    combined_init = eqx.combine(ctx.diff_params_init, ctx.static_params)
    sim_result_eeg_init = ctx.simulators.simulator_eeg(combined_init)
    sim_result_bold_init = ctx.simulators.simulator_bold(combined_init)
    sim_bold_2d_init = connectivity.extract_bold_2d(sim_result_bold_init)

    fig = viz.plot_bold_learning(sim_bold_2d_init, sim_bold_2d, ctx.sc.empirical_bold,
                                  ctx.mask_cortical, cfg.tr_ms, skip_t=cfg.bold_skip_trs)
    figures["bold_learning"] = out_dir / "bold_learning.png"
    fig.savefig(figures["bold_learning"], dpi=150)

    if cfg.bold_dfc_weight > 0:
        fig, dfc_w_dist_before, dfc_w_dist_after = viz.plot_fcd_learning(
            sim_bold_2d_init, sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms,
            cfg.dfc_window_trs, cfg.dfc_step_trs, skip_t=cfg.bold_skip_trs,
            k_min=cfg.dfc_kmin, n_bins=cfg.dfc_n_bins, sigma=cfg.dfc_sigma,
        )
        figures["fcd_learning"] = out_dir / "fcd_learning.png"
        fig.savefig(figures["fcd_learning"], dpi=150)
        metrics["dfc_w_dist_before"] = dfc_w_dist_before
        metrics["dfc_w_dist_after"] = dfc_w_dist_after
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
    figures["eeg_psd_comparison"] = out_dir / "eeg_psd_comparison.png"
    fig.savefig(figures["eeg_psd_comparison"], dpi=150)

    fig = viz.plot_eeg_psd_comparison(sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max, log_scale=False)
    figures["eeg_psd_comparison_linear"] = out_dir / "eeg_psd_comparison_linear.png"
    fig.savefig(figures["eeg_psd_comparison_linear"], dpi=150)

    fig, r_eeg = viz.plot_eeg_corr_comparison(simulated_eeg, dataset._chunks)
    figures["eeg_corr_comparison"] = out_dir / "eeg_corr_comparison.png"
    fig.savefig(figures["eeg_corr_comparison"], dpi=150)
    metrics["eeg_corr"] = r_eeg
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
    figures["eeg_psd_learning"] = out_dir / "eeg_psd_learning.png"
    fig.savefig(figures["eeg_psd_learning"], dpi=150)

    fig = viz.plot_eeg_psd_learning(sim_psd_init, sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max,
                                     log_scale=False)
    figures["eeg_psd_learning_linear"] = out_dir / "eeg_psd_learning_linear.png"
    fig.savefig(figures["eeg_psd_learning_linear"], dpi=150)

    print(f"diagnostics saved to {out_dir}")
    return {"metrics": metrics, "figures": figures}
