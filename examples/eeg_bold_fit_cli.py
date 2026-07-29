#!/usr/bin/env python
"""CLI driver for the EEG+BOLD alternating fit -- cluster-runnable twin of
eeg_bold_fit_new.py.

Same pipeline as the notebook driver (data loading, forward model, network
assembly, alternating fit, diagnostics all live in ``parrot_neuro.optimization``
-- this file only exposes the run as command-line arguments instead of the
notebook's "edit these for your run" constants, so one SLURM array task can
point at each subject without editing a file per run:

    python examples/eeg_bold_fit_cli.py --bids-root <BIDS> --subject 010005

Defaults mirror eeg_bold_fit_new.py (atlas=1000, optimize=both, bold_loss=fc,
num_epochs=300, bold_every=2). See --help for the full set of overridable
BoldFitConfig fields. For interactive/exploratory edits, use
eeg_bold_fit_new.py directly -- this file is the batch entry point.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bids-root", required=True, help="Parrot dataset root (dir containing 'derivatives/')")
    p.add_argument("--subject", required=True, help="participant label, with or without 'sub-' prefix")
    p.add_argument("--output-root", default="eeg_bold_fit_res",
                    help="results land under <output-root>/atlas-<atlas>/<subject>_<optimize>_<bold-loss>")
    p.add_argument("--atlas", type=int, default=1000, choices=(100, 1000))
    p.add_argument("--spacing", default="2.0", help="dipole spacing in mm (string)")
    p.add_argument("--leadfield-label", default="duneuroCGAL")
    p.add_argument("--optimize", default="both", choices=("eeg", "bold", "both"))
    p.add_argument("--bold-loss", default="fc", choices=("fc", "dfc"))
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--bold-every", type=int, default=2)
    p.add_argument("--eeg-task", default="eyesclosed", help="subject.load.eeg(...) recording to fit")
    p.add_argument("--fmri-task", default="rest")
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--noise-seed", type=int, default=69)
    p.add_argument("--t1-warmup", type=float, default=None,
                    help="duration (ms) of the one-time BOLD warm-up solve, separate from "
                         "--num-epochs's t1_bold -- default None reuses t1_bold (slow/OOM-prone "
                         "for a long horizon at a large atlas). Set e.g. 30000 for a short "
                         "warm-up with margin over both settling time and the HRF kernel's 20s "
                         "duration; does not change how much BOLD signal the loss sees.")
    p.add_argument("--solver-block-size", type=int, default=None,
                    help="checkpoint the integration scan in blocks of this many steps -- "
                         "trades ~1.3-1.7x compute for O(n_steps/K + K) instead of O(n_steps) "
                         "backward-pass GPU memory (exact gradient either way). Default None = "
                         "off (monolithic scan). K ~ sqrt(n_steps) is a good starting point -- "
                         "e.g. ~565 for the default t1_bold=320000ms at dt=1.0ms (320k steps). "
                         "Use this if you're hitting GPU OOM.")
    p.add_argument("--skip-diagnostics", action="store_true",
                    help="fit + save params/losses only -- skip the plotting section "
                         "(faster; useful for a smoke test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # config.apply_jax_env() must run before any jax import -- it sets
    # CUDA/JAX env vars that only take effect pre-import.
    from parrot_neuro.optimization import config
    config.apply_jax_env()

    import jax
    jax.config.update("jax_enable_x64", True)

    import matplotlib
    matplotlib.use("Agg")  # headless: no DISPLAY on compute nodes

    import equinox as eqx
    import numpy as np

    from parrot_neuro import Subject
    from parrot_neuro.optimization import connectivity, data, pipeline, train, viz
    from parrot_neuro.optimization.forward import project_to_scalp
    from parrot_neuro.optimization.signal import compute_psd
    import jax.numpy as jnp

    subject = Subject(args.bids_root, args.subject)

    output_root = os.path.join(args.output_root, f"atlas-{args.atlas}")
    output_dir = os.path.join(output_root, f"{subject.subject}_{args.optimize}_{args.bold_loss}")
    os.makedirs(output_dir, exist_ok=True)

    cfg = config.BoldFitConfig(
        subject=subject,
        atlas=args.atlas,
        spacing=args.spacing,
        leadfield_label=args.leadfield_label,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        bold_every=args.bold_every,
        optimize=args.optimize,
        bold_loss=args.bold_loss,
        eeg_task=args.eeg_task,
        fmri_task=args.fmri_task,
        learning_rate=args.learning_rate,
        noise_seed=args.noise_seed,
        solver_block_size=args.solver_block_size,
        t1_warmup=args.t1_warmup,
    )

    load_eeg = args.optimize != "bold"
    dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length) if load_eeg else None
    if dataset is not None:
        print(f"Fitting {subject.subj}: {len(dataset)} chunks of {cfg.chunk_length} samples")
    else:
        print(f"optimize={args.optimize!r}: EEG not loaded (not a fit target).")

    ctx = pipeline.build_context(cfg, dataset)
    result = pipeline.fit(ctx)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "loss_history_eeg.npy", np.array(result.loss_history_eeg))
    np.save(out_dir / "loss_history_bold.npy", np.array(result.loss_history_bold))

    optimized = train.extract_learnable_values(result.diff_params, cfg.learnable_params)
    np.savez(out_dir / "optimized_params.npz", **optimized,
              loss_eeg=np.array(result.loss_history_eeg), loss_bold=np.array(result.loss_history_bold))

    if result.loss_history_eeg:
        print(f"Final EEG loss:  {result.loss_history_eeg[-1]:.5f}")
    if result.loss_history_bold:
        print(f"Final BOLD loss: {result.loss_history_bold[-1]:.5f}")
    for name, values in optimized.items():
        print(f"{name:6s} -- mean {values.mean():.4f}  std {values.std():.4f}")
    print(f"Saved to {out_dir}")

    if args.skip_diagnostics:
        print("--skip-diagnostics set: done.")
        return

    # --- diagnostics: simulation + BOLD (always available) -------------------
    combined = eqx.combine(result.diff_params, result.static_params)
    sim_result_eeg = ctx.simulators.simulator_eeg(combined)
    sim_result_bold = ctx.simulators.simulator_bold(combined)

    fig = viz.plot_node_activity(sim_result_eeg, ctx.mask_cortical, cfg.dt)
    fig.savefig(out_dir / "node_activity.png", dpi=150)

    sim_bold_2d = connectivity.extract_bold_2d(ctx.simulators.bold_monitor(sim_result_bold))
    sim_bold_2d_filt = connectivity.filter_sim_bold(np.asarray(sim_bold_2d), cfg.tr_ms)

    fig = viz.plot_bold_timeseries(sim_bold_2d_filt, ctx.sc.empirical_bold, ctx.mask_cortical,
                                    cfg.tr_ms, skip_t=cfg.bold_skip_trs)
    fig.savefig(out_dir / "bold_timeseries.png", dpi=150)

    if cfg.bold_loss == "dfc":
        fig, dfc_w_dist = viz.plot_fcd_comparison(sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms,
                                                    cfg.dfc_window_trs, cfg.dfc_step_trs,
                                                    skip_t=cfg.bold_skip_trs, k_min=cfg.dfc_kmin,
                                                    n_bins=cfg.dfc_n_bins, sigma=cfg.dfc_sigma)
        fig.savefig(out_dir / "fcd_comparison.png", dpi=150)
        print(f"dFC Wasserstein-1 distance (sim vs emp): {dfc_w_dist:.5f}")
    fig, fc_corr = viz.plot_fc_comparison(sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms, skip_t=cfg.bold_skip_trs)
    fig.savefig(out_dir / "fc_comparison.png", dpi=150)
    print(f"FC Pearson correlation (sim vs emp): {fc_corr:.4f}")

    # --- diagnostics: BOLD learning (first iteration vs last) ----------------
    combined_init = eqx.combine(ctx.diff_params_init, ctx.static_params)
    sim_result_eeg_init = ctx.simulators.simulator_eeg(combined_init)
    sim_result_bold_init = ctx.simulators.simulator_bold(combined_init)
    sim_bold_2d_init = connectivity.extract_bold_2d(ctx.simulators.bold_monitor(sim_result_bold_init))
    sim_bold_2d_init_filt = connectivity.filter_sim_bold(np.asarray(sim_bold_2d_init), cfg.tr_ms)

    fig = viz.plot_bold_learning(sim_bold_2d_init_filt, sim_bold_2d_filt, ctx.sc.empirical_bold,
                                  ctx.mask_cortical, cfg.tr_ms, skip_t=cfg.bold_skip_trs)
    fig.savefig(out_dir / "bold_learning.png", dpi=150)

    if cfg.bold_loss == "dfc":
        fig, dfc_w_dist_before, dfc_w_dist_after = viz.plot_fcd_learning(
            sim_bold_2d_init, sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms,
            cfg.dfc_window_trs, cfg.dfc_step_trs, skip_t=cfg.bold_skip_trs,
            k_min=cfg.dfc_kmin, n_bins=cfg.dfc_n_bins, sigma=cfg.dfc_sigma,
        )
        fig.savefig(out_dir / "fcd_learning.png", dpi=150)
        print(f"dFC Wasserstein-1 distance: before={dfc_w_dist_before:.5f}  after={dfc_w_dist_after:.5f}")

    # --- diagnostics: EEG (needs the subject's real EEG; load if skipped) ----
    if dataset is None:
        dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
        print(f"Loaded {subject.subj} EEG for visualization only (still not used as a fit target)")

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

    print("diagnostics saved.")


if __name__ == "__main__":
    main()
