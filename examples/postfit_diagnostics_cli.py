#!/usr/bin/env python
"""Re-run ONLY the post-fit diagnostic plots from a saved optimized_params.npz
-- no re-training.

Useful when a fit already completed and you just want the plots (re)made
without paying for another full training run -- e.g. the original run used
``--skip-diagnostics``, or its diagnostics section OOM'd/crashed, or you
tweaked ``--solver-block-size``/rendering and want fresh figures for the same
fitted parameters.

Rebuilds the same network/simulators as the original fit (pass MATCHING
--atlas/--spacing/--leadfield-label/--bold-loss/--eeg-task/--fmri-task/
--noise-seed -- these determine what gets built, and must agree with however
the npz was produced), then reconstructs the fitted parameters from the npz
(saved in natural post-sigmoid units -- see
``parrot_neuro.optimization.train.extract_learnable_values``) and calls
``parrot_neuro.optimization.diagnostics.run_and_save``, the same function
``eeg_bold_fit_cli.py`` calls right after ``pipeline.fit``.

    python examples/postfit_diagnostics_cli.py \\
        --bids-root <BIDS> --subject 010005 \\
        --optimized-params eeg_bold_fit_res/atlas-1000/010005_both_fc/optimized_params.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bids-root", required=True, help="Parrot dataset root (dir containing 'derivatives/')")
    p.add_argument("--subject", required=True, help="participant label, with or without 'sub-' prefix")
    p.add_argument("--optimized-params", required=True,
                    help="path to a saved optimized_params.npz from a prior fit run")
    p.add_argument("--output-dir", default=None,
                    help="where to save the plots -- default: same directory as --optimized-params")
    # Forward-model / data selection -- MUST match the original fit (they determine
    # num_nodes and the exact network the npz's values were fitted for).
    p.add_argument("--atlas", type=int, default=1000, choices=(100, 1000))
    p.add_argument("--spacing", default="2.0", help="dipole spacing in mm (string)")
    p.add_argument("--leadfield-label", default="duneuroCGAL")
    p.add_argument("--bold-loss", default="fc", choices=("fc", "dfc"),
                    help="must match the original fit -- selects whether the dFC/FCD plots also run")
    p.add_argument("--eeg-task", default="eyesclosed", help="must match the original fit")
    p.add_argument("--fmri-task", default="rest", help="must match the original fit")
    p.add_argument("--noise-seed", type=int, default=69,
                    help="must match the original fit -- affects the stochastic warm-up/network dynamics")
    # Memory/compute knobs -- safe to set independently of the original fit
    # (they don't change results, see their --help in eeg_bold_fit_cli.py),
    # except t1_warmup also affects the reconstructed warm-up state. Default to
    # the same values eeg_bold_fit_cli.py now defaults to: the diagnostics
    # section's own forward passes (simulator_bold called directly, twice) are
    # just as OOM-prone at atlas=1000 as training is -- see diagnostics.py.
    p.add_argument("--t1-warmup", type=float, default=30_000.0,
                    help="should match the original fit for a bit-identical warm-up state; "
                         "harmless to change otherwise (still a settled state either way). "
                         "Pass --t1-warmup=-1 for the old behaviour (reuse t1_bold).")
    p.add_argument("--solver-block-size", type=int, default=565,
                    help="pure memory/compute trade, doesn't affect results. Pass "
                         "--solver-block-size=0 for the old unblocked behaviour.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from parrot_neuro.optimization import config
    config.apply_jax_env()

    import jax
    jax.config.update("jax_enable_x64", True)

    import matplotlib
    matplotlib.use("Agg")  # headless: no DISPLAY on compute nodes

    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    from parrot_neuro import Subject
    from parrot_neuro.optimization import data, diagnostics, pipeline

    subject = Subject(args.bids_root, args.subject)
    npz_path = Path(args.optimized_params)
    out_dir = Path(args.output_dir) if args.output_dir else npz_path.parent

    # Sentinels for "old behaviour" (see their --help text) -- BoldFitConfig wants None.
    solver_block_size = None if args.solver_block_size == 0 else args.solver_block_size
    t1_warmup = None if args.t1_warmup < 0 else args.t1_warmup

    cfg = config.BoldFitConfig(
        subject=subject,
        atlas=args.atlas,
        spacing=args.spacing,
        leadfield_label=args.leadfield_label,
        output_dir=out_dir,
        bold_loss=args.bold_loss,
        eeg_task=args.eeg_task,
        fmri_task=args.fmri_task,
        noise_seed=args.noise_seed,
        solver_block_size=solver_block_size,
        t1_warmup=t1_warmup,
    )

    dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
    print(f"Loaded {subject.subj}: {len(dataset)} chunks of {cfg.chunk_length} samples")

    ctx = pipeline.build_context(cfg, dataset)

    # Reconstruct diff_params from the npz: ctx.diff_params_init already has the
    # right structure/bounds (one SigmoidBoundedParameter per learnable leaf,
    # None elsewhere -- see train.learnable_partition) -- swap each leaf's
    # VALUE for the fitted one, in place on the EXISTING parameter object
    # (via eqx.tree_at) rather than constructing a fresh SigmoidBoundedParameter.
    # Constructing fresh ones would build new forward_transform/inverse_transform
    # closures (different Python object identity each call), which breaks
    # eqx.combine's structural match against static_params later -- reusing the
    # original object's closures and only replacing .value avoids that.
    # extract_learnable_values saved NATURAL (post-sigmoid) units;
    # inverse_transform maps them back to the parameter's internal
    # unconstrained representation.
    npz = np.load(npz_path)
    diff_params = ctx.diff_params_init
    for lp in cfg.learnable_params:
        if lp.name not in npz.files:
            raise KeyError(
                f"{lp.name!r} (from cfg.learnable_params) not found in {npz_path} -- "
                f"available: {list(npz.files)}. Does --atlas/--bold-loss/etc match the original fit?"
            )
        natural_value = jnp.asarray(npz[lp.name])
        if lp.location == "dynamics":
            orig = diff_params["dynamics"][lp.name]
        elif lp.location == "coupling":
            orig = diff_params["coupling"]["delayed"][lp.name]
        else:
            raise ValueError(f"Unknown location {lp.location!r} for learnable param {lp.name!r}")
        new_param = eqx.tree_at(lambda p: p.value, orig, orig.inverse_transform(natural_value))
        if lp.location == "dynamics":
            diff_params["dynamics"][lp.name] = new_param
        else:
            diff_params["coupling"]["delayed"][lp.name] = new_param
    print(f"Reconstructed {len(cfg.learnable_params)} learnable parameters from {npz_path}")

    diagnostics.run_and_save(ctx, diff_params, ctx.static_params, dataset, out_dir)


if __name__ == "__main__":
    main()
