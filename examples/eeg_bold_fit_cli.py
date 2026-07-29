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

    import numpy as np

    from parrot_neuro import Subject
    from parrot_neuro.optimization import data, diagnostics, pipeline, train

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

    # Diagnostics always need the subject's real EEG (to visualize simulated-
    # vs-real, regardless of whether EEG was a fit target) -- reload if skipped.
    if dataset is None:
        dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
        print(f"Loaded {subject.subj} EEG for visualization only (still not used as a fit target)")

    diagnostics.run_and_save(ctx, result.diff_params, result.static_params, dataset, out_dir)


if __name__ == "__main__":
    main()
