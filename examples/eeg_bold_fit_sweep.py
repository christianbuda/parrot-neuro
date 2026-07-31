#!/usr/bin/env python
"""wandb sweep trial: fit the EEG+BOLD model for a FIXED list of subjects with
ONE sampled hyperparameter set, logging per-subject curves/plots plus an
aggregated objective wandb's Bayesian search optimizes.

Structurally a twin of eeg_bold_fit_cli.py (same BoldFitConfig fields, same
data/pipeline/diagnostics calls) except it loops over --subjects (plural)
instead of a single --subject, and every metric/plot goes to wandb instead of
just stdout/disk.

Meant to be launched by examples/eeg_bold_fit_cli.py's sibling infrastructure
under hpc/leonardo/ (see hpc/leonardo/README.md's "Hyperparameter sweep"
section): a `wandb agent` on the LEONARDO *login* node (has internet) resolves
each trial's sampled hyperparameters and dispatches an `sbatch --wait` job
that runs THIS script on a *compute* node (no internet) with
WANDB_MODE=offline and WANDB_RUN_ID/WANDB_SWEEP_ID pinned to the ids the agent
already assigned; the dispatcher `wandb sync`s the result afterward. This
split exists ONLY because Leonardo's compute nodes have no network egress --
run directly with WANDB_MODE unset (defaults to "online") for a normal
workstation sweep trial or manual smoke test, e.g.:

    WANDB_RUN_ID=<id> python examples/eeg_bold_fit_sweep.py \\
        --bids-root <BIDS> --subjects 010002,010003 --num-epochs 5

The 7 swept fields (--learning-rate, --learning-rate-bold, --bold-fc-weight,
--bold-dfc-weight, --bold-psd-weight, --dfc-window-trs, --dfc-step-trs) each
default from a SWEEP_* environment variable (set by hpc/leonardo/
sweep_dispatch.sh -> sweep_train.sbatch) so this script also runs standalone
with just --subjects/--num-epochs overridden, for local testing.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bids-root", default=os.environ.get("SWEEP_BIDS_ROOT"),
                    required=os.environ.get("SWEEP_BIDS_ROOT") is None,
                    help="Parrot dataset root (dir containing 'derivatives/')")
    p.add_argument("--subjects", default=os.environ.get("SWEEP_SUBJECTS"),
                    required=os.environ.get("SWEEP_SUBJECTS") is None,
                    help="comma-separated participant labels, with or without 'sub-' prefix "
                         "(e.g. 010002,010003,010004,010005,010006) -- every subject is fit "
                         "sequentially, in ONE wandb run, with the SAME sampled hyperparameters")
    p.add_argument("--output-root", default=os.environ.get("SWEEP_OUTPUT_ROOT", "eeg_bold_fit_sweep_res"),
                    help="results land under <output-root>/atlas-<atlas>/<wandb-run-id>/<subject>_<optimize>")
    p.add_argument("--atlas", type=int, default=1000, choices=(100, 1000))
    p.add_argument("--spacing", default="2.0", help="dipole spacing in mm (string)")
    p.add_argument("--leadfield-label", default="duneuroCGAL")
    p.add_argument("--optimize", default="both", choices=("eeg", "bold", "both"))
    p.add_argument("--num-epochs", type=int, default=_env_int("SWEEP_NUM_EPOCHS", 300))
    p.add_argument("--bold-every", type=int, default=2)
    p.add_argument("--eeg-task", default="eyesclosed")
    p.add_argument("--fmri-task", default="rest")
    p.add_argument("--noise-seed", type=int, default=69)
    p.add_argument("--t1-warmup", type=float, default=30_000.0,
                    help="see eeg_bold_fit_cli.py --t1-warmup; -1 = old full-t1_bold warm-up")
    p.add_argument("--solver-block-size", type=int, default=565,
                    help="see eeg_bold_fit_cli.py --solver-block-size; 0 = unblocked")
    p.add_argument("--gamma-weight", type=float, default=0.0)
    p.add_argument("--skip-diagnostics", action="store_true",
                    help="fit + save params/losses only -- skip plots (faster smoke test)")

    # --- swept hyperparameters -- default from the SWEEP_* env vars sweep_train.sbatch sets,
    # so a sweep trial needs no CLI overrides at all; still overridable for manual testing. ---
    p.add_argument("--learning-rate", type=float, default=_env_float("SWEEP_LEARNING_RATE", 1e-2))
    p.add_argument("--learning-rate-bold", type=float,
                    default=_env_float("SWEEP_LEARNING_RATE_BOLD", 1e-2))
    p.add_argument("--bold-fc-weight", type=float, default=_env_float("SWEEP_BOLD_FC_WEIGHT", 0.5))
    p.add_argument("--bold-dfc-weight", type=float, default=_env_float("SWEEP_BOLD_DFC_WEIGHT", 0.5))
    p.add_argument("--bold-psd-weight", type=float, default=_env_float("SWEEP_BOLD_PSD_WEIGHT", 0.0))
    p.add_argument("--dfc-window-trs", type=int, default=_env_int("SWEEP_DFC_WINDOW_TRS", 6))
    p.add_argument("--dfc-step-trs", type=int, default=_env_int("SWEEP_DFC_STEP_TRS", 1))

    # --- wandb ---
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "parrot-eeg-bold-sweep"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
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
    import wandb

    from parrot_neuro import Subject
    from parrot_neuro.optimization import data, diagnostics, pipeline, train

    # WANDB_RUN_ID/WANDB_MODE come from the SLURM job env (set by
    # sweep_dispatch.sh -> sweep_train.sbatch) when this is a real sweep
    # trial; both are unset for a normal standalone/manual run, in which case
    # wandb.init just creates a fresh online run as usual.
    run_id = os.environ.get("WANDB_RUN_ID")
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        id=run_id, resume="allow" if run_id else None,
        config=vars(args),
    )

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    print(f"Sweep trial {run.id}: fitting {len(subjects)} subjects: {subjects}")

    per_subject_combined = []
    per_subject_eeg = []
    per_subject_bold = []

    for subject_id in subjects:
        subject = Subject(args.bids_root, subject_id)

        output_root = os.path.join(args.output_root, f"atlas-{args.atlas}", run.id)
        output_dir = os.path.join(output_root, f"{subject.subject}_{args.optimize}")
        os.makedirs(output_dir, exist_ok=True)

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
            bold_psd_weight=args.bold_psd_weight,
            dfc_window_trs=args.dfc_window_trs,
            dfc_step_trs=args.dfc_step_trs,
            eeg_task=args.eeg_task,
            fmri_task=args.fmri_task,
            learning_rate=args.learning_rate,
            learning_rate_bold=args.learning_rate_bold,
            gamma_weight=args.gamma_weight,
            noise_seed=args.noise_seed,
            solver_block_size=solver_block_size,
            t1_warmup=t1_warmup,
        )
        cfg_path = cfg.save()
        print(f"[{subject_id}] saved run config to {cfg_path}")

        load_eeg = args.optimize != "bold"
        dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length) if load_eeg else None
        if dataset is not None:
            print(f"[{subject_id}] fitting {len(dataset)} chunks of {cfg.chunk_length} samples")

        ctx = pipeline.build_context(cfg, dataset)

        def on_epoch(epoch, loss_eeg, loss_bold, bold_stepped, _subject_id=subject_id):
            log = {f"subj_{_subject_id}/epoch": epoch}
            if loss_eeg is not None:
                log[f"subj_{_subject_id}/eeg_loss"] = loss_eeg
            if bold_stepped:
                log[f"subj_{_subject_id}/bold_loss"] = loss_bold
            wandb.log(log)

        result = pipeline.fit(ctx, on_epoch=on_epoch)

        out_dir = Path(cfg.output_dir)
        np.save(out_dir / "loss_history_eeg.npy", np.array(result.loss_history_eeg))
        np.save(out_dir / "loss_history_bold.npy", np.array(result.loss_history_bold))

        optimized = train.extract_learnable_values(result.diff_params, cfg.learnable_params)
        np.savez(out_dir / "optimized_params.npz", **optimized,
                  loss_eeg=np.array(result.loss_history_eeg), loss_bold=np.array(result.loss_history_bold))

        final_eeg = result.loss_history_eeg[-1] if result.loss_history_eeg else None
        final_bold = result.loss_history_bold[-1] if result.loss_history_bold else None
        summary = {f"subj_{subject_id}/final_eeg_loss": final_eeg,
                   f"subj_{subject_id}/final_bold_loss": final_bold}

        if not args.skip_diagnostics:
            if dataset is None:
                dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
                print(f"[{subject_id}] loaded EEG for visualization only (not a fit target)")
            diag = diagnostics.run_and_save(ctx, result.diff_params, result.static_params, dataset, out_dir)
            for name, value in diag["metrics"].items():
                summary[f"subj_{subject_id}/{name}"] = float(value)
            for name, path in diag["figures"].items():
                summary[f"subj_{subject_id}/plots/{name}"] = wandb.Image(str(path))

        wandb.log(summary)

        combined = (final_eeg or 0.0) + (final_bold or 0.0)
        per_subject_combined.append(combined)
        if final_eeg is not None:
            per_subject_eeg.append(final_eeg)
        if final_bold is not None:
            per_subject_bold.append(final_bold)
        print(f"[{subject_id}] final EEG loss: {final_eeg}  final BOLD loss: {final_bold}  saved to {out_dir}")

    aggregate = {"aggregate/combined_loss": float(np.mean(per_subject_combined))}
    if per_subject_eeg:
        aggregate["aggregate/eeg_loss_mean"] = float(np.mean(per_subject_eeg))
    if per_subject_bold:
        aggregate["aggregate/bold_loss_mean"] = float(np.mean(per_subject_bold))
    wandb.log(aggregate)
    print(f"Aggregate over {len(subjects)} subjects: {aggregate}")

    wandb.finish()


if __name__ == "__main__":
    main()
