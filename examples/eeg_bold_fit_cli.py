#!/usr/bin/env python
"""CLI driver for the EEG+BOLD alternating fit -- cluster-runnable twin of
eeg_bold_fit_new.py.

Same pipeline as the notebook driver (data loading, forward model, network
assembly, alternating fit, diagnostics all live in ``parrot_neuro.optimization``
-- this file only exposes the run as command-line arguments instead of the
notebook's "edit these for your run" constants, so one SLURM array task can
point at each subject without editing a file per run:

    python examples/eeg_bold_fit_cli.py --bids-root <BIDS> --subject 010005

Defaults mirror eeg_bold_fit_new.py (atlas=1000, optimize=both, BOLD loss =
0.5*fc + 0.5*dfc, num_epochs=300, bold_every=2). See --help for the full set
of overridable BoldFitConfig fields. For interactive/exploratory edits, use
eeg_bold_fit_new.py directly -- this file is the batch entry point.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bids-root", required=True, help="Parrot dataset root (dir containing 'derivatives/')")
    p.add_argument("--subject", required=True, help="participant label, with or without 'sub-' prefix")
    p.add_argument("--output-root", default="eeg_bold_fit_res",
                    help="results land under <output-root>/atlas-<atlas>/<subject>_<optimize>")
    p.add_argument("--atlas", type=int, default=1000, choices=(100, 1000))
    p.add_argument("--spacing", default="2.0", help="dipole spacing in mm (string)")
    p.add_argument("--leadfield-label", default="duneuroCGAL")
    p.add_argument("--optimize", default="both", choices=("eeg", "bold", "both"))
    p.add_argument("--bold-fc-weight", type=float, default=0.5,
                    help="weight of the static-FC term in the combined BOLD loss -- 0 drops it "
                         "(dfc-only fit).")
    p.add_argument("--bold-dfc-weight", type=float, default=0.5,
                    help="weight of the dynamic-FC (FCD) term in the combined BOLD loss -- 0 drops "
                         "it (fc-only fit, and also disables the FCD diagnostic plots).")
    p.add_argument("--dfc-window-trs", type=int, default=6,
                    help="dFC sliding-window length in TRs (see config.BoldFitConfig.dfc_window_trs)")
    p.add_argument("--dfc-step-trs", type=int, default=1,
                    help="dFC sliding-window stride in TRs (see config.BoldFitConfig.dfc_step_trs)")
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--bold-every", type=int, default=2)
    p.add_argument("--eeg-task", default="eyesclosed", help="subject.load.eeg(...) recording to fit")
    p.add_argument("--fmri-task", default="rest")
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--learning-rate-bold", type=float, default=None,
                    help="learning rate for the BOLD step -- default None reuses --learning-rate "
                         "(EEG and BOLD each get their own Adam state, so they can also use "
                         "different step sizes).")
    p.add_argument("--bold-psd-weight", type=float, default=0.0,
                    help="weight of an optional Welch-PSD spectral-shape term (restricted to the "
                         "0.01-0.1Hz BOLD bandpass) added to the combined BOLD loss -- 0 (default) "
                         "= off. fc_vector's time-averaged correlation has no sensitivity at all "
                         "to each signal's own temporal/spectral shape; this adds a gradient for it.")
    p.add_argument("--gamma-weight", type=float, default=0.0,
                    help="weight of an optional log(PSD) MSE term over 15-40Hz added to the EEG "
                         "loss, alongside the existing normalized-linear PSD MSE over 1-15Hz -- "
                         "0 (default) = off.")
    p.add_argument("--noise-seed", type=int, default=69)
    p.add_argument("--t1-warmup", type=float, default=30_000.0,
                    help="duration (ms) of the one-time BOLD warm-up solve, separate from "
                         "--num-epochs's t1_bold. Defaults to a short warm-up (comfortable margin "
                         "over both settling time and the HRF kernel's 20s duration) rather than "
                         "reusing the full t1_bold, which is slow/OOM-prone at a large atlas -- "
                         "does not change how much BOLD signal the loss sees. Pass --t1-warmup=-1 "
                         "to get the old behaviour (reuse t1_bold) instead.")
    p.add_argument("--solver-block-size", type=int, default=1400,
                    help="checkpoint the integration scan in blocks of this many steps -- "
                         "trades ~1.3-1.7x compute for O(n_steps/K + K) instead of O(n_steps) "
                         "backward-pass GPU memory (exact gradient either way). The BOLD simulator "
                         "also streams its HRF convolution through this same block scan (see "
                         "train.build_simulators), so K must be an exact multiple of the BOLD "
                         "period in raw steps (tr_ms/dt -- 1400 for the defaults), not just close "
                         "to sqrt(n_steps); 1400 (one TR per block) is the smallest valid choice. "
                         "Pass --solver-block-size=0 for the old unblocked behaviour (also drops "
                         "the streaming BOLD monitor's memory win -- only sensible at a much "
                         "shorter t1_bold/smaller atlas where OOM isn't a risk).")
    p.add_argument("--skip-diagnostics", action="store_true",
                    help="fit + save params/losses only -- skip the plotting section "
                         "(faster; useful for a smoke test)")
    p.add_argument("--early-stop-patience", type=int, default=None,
                    help="stop once every actively-optimized loss's trend has stayed "
                         "flat/increasing for this many consecutive --early-stop-window "
                         "checks -- default None keeps the old behaviour (always run all "
                         "--num-epochs).")
    p.add_argument("--early-stop-window", type=int, default=20,
                    help="epochs (EEG) / bold-steps (BOLD) of loss history the trend is "
                         "fit over for --early-stop-patience.")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-3,
                    help="minimum relative per-step loss decrease to NOT count as stalled "
                         "for --early-stop-patience.")
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

    import numpy as np

    from parrot_neuro import Subject
    from parrot_neuro.optimization import data, diagnostics, pipeline, train

    subject = Subject(args.bids_root, args.subject)

    output_root = os.path.join(args.output_root, f"atlas-{args.atlas}")
    output_dir = os.path.join(output_root, f"{subject.subject}_{args.optimize}")
    os.makedirs(output_dir, exist_ok=True)

    # Sentinels for "give me the old, unblocked/full-t1_bold-warmup behaviour"
    # (see their --help text) -- BoldFitConfig itself wants None for that.
    solver_block_size = None if args.solver_block_size == 0 else args.solver_block_size
    t1_warmup = None if args.t1_warmup < 0 else args.t1_warmup

    cfg = config.BoldFitConfig(
        subject=subject,
        atlas=args.atlas,
        spacing=args.spacing,
        leadfield_label=args.leadfield_label,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        bold_every=args.bold_every,
        optimize=args.optimize,
        bold_fc_weight=args.bold_fc_weight,
        bold_dfc_weight=args.bold_dfc_weight,
        dfc_window_trs=args.dfc_window_trs,
        dfc_step_trs=args.dfc_step_trs,
        eeg_task=args.eeg_task,
        fmri_task=args.fmri_task,
        learning_rate=args.learning_rate,
        learning_rate_bold=args.learning_rate_bold,
        bold_psd_weight=args.bold_psd_weight,
        gamma_weight=args.gamma_weight,
        noise_seed=args.noise_seed,
        solver_block_size=solver_block_size,
        t1_warmup=t1_warmup,
        early_stop_patience=args.early_stop_patience,
        early_stop_window=args.early_stop_window,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    cfg_path = cfg.save()
    print(f"Saved run config to {cfg_path}")

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

    # relative_final_loss (final/first-logged) puts EEG (~1e-6) and BOLD
    # (~1e-1) on a comparable ~1.0-scale "fraction of initial loss remaining"
    # -- same combined-loss definition examples/eeg_bold_fit_sweep.py uses for
    # its wandb sweep objective, so a single fit here and a sweep trial report
    # the same number for the same fit.
    eeg_ratio = train.relative_final_loss(result.loss_history_eeg)
    bold_ratio = train.relative_final_loss(result.loss_history_bold)
    combined_ratio = (eeg_ratio or 0.0) + (bold_ratio or 0.0)

    optimized = train.extract_learnable_values(result.diff_params, cfg.learnable_params)
    np.savez(out_dir / "optimized_params.npz", **optimized,
              loss_eeg=np.array(result.loss_history_eeg), loss_bold=np.array(result.loss_history_bold),
              eeg_loss_ratio=np.nan if eeg_ratio is None else eeg_ratio,
              bold_loss_ratio=np.nan if bold_ratio is None else bold_ratio,
              combined_loss_ratio=combined_ratio)

    if result.loss_history_eeg:
        print(f"Final EEG loss:  {result.loss_history_eeg[-1]:.5f}  (ratio to first: {eeg_ratio:.4f})")
    if result.loss_history_bold:
        print(f"Final BOLD loss: {result.loss_history_bold[-1]:.5f}  (ratio to first: {bold_ratio:.4f})")
    if eeg_ratio is not None or bold_ratio is not None:
        print(f"Combined loss (sum of ratios): {combined_ratio:.4f}")
    for name, values in optimized.items():
        print(f"{name:6s} -- mean {values.mean():.4f}  std {values.std():.4f}")
    print(f"Saved to {out_dir}")

    if args.skip_diagnostics:
        print("--skip-diagnostics set: done.")
        return

    # Diagnostics always need the subject's real EEG (to visualize simulated-
    # vs-real, regardless of whether EEG was a fit target) -- reload if skipped.
    if dataset is None:
        dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
        print(f"Loaded {subject.subj} EEG for visualization only (still not used as a fit target)")

    diag = diagnostics.run_and_save(ctx, result.diff_params, result.static_params, dataset, out_dir)
    # Saved (not just printed) so a caller that ran this as a subprocess --
    # e.g. eeg_bold_fit_sweep.py's --gpus parallel-worker mode -- can read the
    # metrics back after this process exits, without this script needing to
    # know anything about wandb itself.
    (out_dir / "diagnostics_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in diag["metrics"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
