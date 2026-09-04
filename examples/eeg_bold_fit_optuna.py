#!/usr/bin/env python
"""Optuna hyperparameter-search trial: fit the EEG+BOLD model for a FIXED
list of subjects with ONE hyperparameter set sampled from a SHARED, locally-
coordinated Optuna study, logging per-subject curves/plots plus an aggregated
objective to wandb (offline, logging only) and reporting the objective back
to Optuna via study.tell().

Replaces the wandb-Sweeps-based examples/eeg_bold_fit_sweep.py for LEONARDO
runs (see hpc/leonardo/README.md's "Hyperparameter sweep" section for the
full rationale): wandb Sweeps needs a continuously-online agent process to
coordinate which hyperparameters to try next, which on LEONARDO can only run
on a login node (compute nodes have no internet) -- forcing a long-lived
process to survive there for hours-to-days is what caused the RLIMIT_NPROC /
duplicate-run-id saga documented in the README. Optuna's JournalStorage
coordinates entirely through a shared file on NFS (no server, no long-lived
process needed anywhere) -- so this script needs NO login-node orchestrator
at all: submit it directly as a SLURM job ARRAY (hpc/leonardo/
submit_optuna_sweep.sh), one task = one trial, SLURM's own scheduler handles
concurrency/throttling.

Reuses eeg_bold_fit_sweep.py's fitting logic verbatim (_run_sequential /
_run_parallel / _log_aggregate) -- only how the 7 swept hyperparameters are
obtained (Optuna's trial.suggest_* instead of wandb-agent-provided argv) and
how a trial's outcome is reported (study.tell() instead of nothing, since
wandb's Sweeps controller isn't in the loop) differ. --gpus (parallel mode)
works identically to eeg_bold_fit_sweep.py -- see its own module docstring.

    OPTUNA_STUDY_NAME=<name> OPTUNA_STORAGE=<path> python examples/eeg_bold_fit_optuna.py \\
        --bids-root <BIDS> --subjects 010002,010003 --num-epochs 5

The 7 swept fields (learning_rate, learning_rate_bold, bold_fc_weight,
bold_dfc_weight, bold_psd_weight, dfc_window_trs, dfc_step_trs) and their
distributions are kept in sync BY HAND with hpc/leonardo/sweep_eeg_bold.yaml
(the legacy wandb-Sweeps search space this replaces) -- wandb sweep configs
and Optuna's define-by-run trial.suggest_* calls are two structurally
different ways of expressing a search space, so there's no single source of
truth to derive one from the other automatically. Change one, change both if
you want them to stay equivalent.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_bold_fit_sweep import _log_aggregate, _run_parallel, _run_sequential  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bids-root", required=True, help="Parrot dataset root (dir containing 'derivatives/')")
    p.add_argument("--subjects", required=True,
                    help="comma-separated participant labels, with or without 'sub-' prefix -- "
                         "every subject is fit with the SAME sampled hyperparameters, in ONE "
                         "wandb run / Optuna trial")
    p.add_argument("--output-root", default="eeg_bold_fit_optuna_res",
                    help="results land under <output-root>/atlas-<atlas>/<wandb-run-id>/<subject>_<optimize> "
                         "(sequential mode) or <output-root>/<wandb-run-id>/atlas-<atlas>/<subject>_<optimize> "
                         "(--gpus parallel mode) -- same layout as eeg_bold_fit_sweep.py")
    p.add_argument("--atlas", type=int, default=1000, choices=(100, 1000))
    p.add_argument("--spacing", default="2.0", help="dipole spacing in mm (string)")
    p.add_argument("--leadfield-label", default="duneuroCGAL")
    p.add_argument("--optimize", default="both", choices=("eeg", "bold", "both"))
    p.add_argument("--bold-model", default="hrf", choices=("hrf", "balloon"),
                    help="see eeg_bold_fit_cli.py --bold-model")
    p.add_argument("--schedule", default="alternating", choices=("alternating", "phased", "joint"),
                    help="see eeg_bold_fit_cli.py --schedule")
    p.add_argument("--joint-eeg-weight", type=float, default=1e5,
                    help="see eeg_bold_fit_cli.py --joint-eeg-weight (--schedule joint only)")
    p.add_argument("--joint-bold-weight", type=float, default=1.0,
                    help="see eeg_bold_fit_cli.py --joint-bold-weight (--schedule joint only)")
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--bold-every", type=int, default=2)
    p.add_argument("--eeg-task", default="eyesclosed")
    p.add_argument("--fmri-task", default="rest")
    p.add_argument("--noise-seed", type=int, default=69)
    p.add_argument("--t1-warmup", type=float, default=30_000.0,
                    help="see eeg_bold_fit_cli.py --t1-warmup; -1 = old full-t1_bold warm-up")
    p.add_argument("--solver-block-size", type=int, default=1400,
                    help="see eeg_bold_fit_cli.py --solver-block-size; 0 = unblocked")
    p.add_argument("--gamma-weight", type=float, default=0.0)
    p.add_argument("--skip-diagnostics", action="store_true",
                    help="fit + save params/losses only -- skip plots (faster smoke test)")
    p.add_argument("--gpus", default=None,
                    help="comma-separated GPU device indices (e.g. '0,1,2,3') -- parallel mode, "
                         "identical to eeg_bold_fit_sweep.py --gpus")

    # --- Optuna: shared search coordination, no server/agent needed ---------
    p.add_argument("--optuna-study-name", default=os.environ.get("OPTUNA_STUDY_NAME"),
                    required=os.environ.get("OPTUNA_STUDY_NAME") is None,
                    help="must already exist (see submit_optuna_sweep.sh create)")
    p.add_argument("--optuna-storage", default=os.environ.get("OPTUNA_STORAGE"),
                    required=os.environ.get("OPTUNA_STORAGE") is None,
                    help="path to the shared Optuna JournalStorage file -- safe for many "
                         "concurrent processes on NFS (see optuna.storages.journal.JournalFileBackend)")

    # --- wandb: LOGGING ONLY here -- no Sweeps controller, no run_id claiming
    # dance (see hpc/leonardo/sweep_dispatch.sh's "claim" step docstring for
    # why that was ever needed -- it isn't, here: Optuna's ask() is what
    # prevents duplicate work, not a server-side run_id). --------------------
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "parrot-eeg-bold-sweep"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"),
                    choices=("offline", "online", "disabled"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    storage = JournalStorage(JournalFileBackend(args.optuna_storage))
    # load_study, not create_study -- the study must already exist (created
    # ONCE via `submit_optuna_sweep.sh create`); every trial process only
    # ever attaches to it. Loading (not creating) here means a typo'd study
    # name fails loudly instead of silently starting a second, empty study.
    study = optuna.load_study(study_name=args.optuna_study_name, storage=storage)
    trial = study.ask()

    # Same 7 fields/distributions as hpc/leonardo/sweep_eeg_bold.yaml -- see
    # module docstring's "kept in sync BY HAND" note.
    args.learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    args.learning_rate_bold = trial.suggest_float("learning_rate_bold", 1e-3, 0.9, log=True)
    args.bold_fc_weight = trial.suggest_float("bold_fc_weight", 0.1, 1.0)
    args.bold_dfc_weight = trial.suggest_float("bold_dfc_weight", 0.0, 1.0)
    args.bold_psd_weight = trial.suggest_float("bold_psd_weight", 0.0, 0.5)
    args.dfc_window_trs = trial.suggest_int("dfc_window_trs", 4, 12)
    args.dfc_step_trs = trial.suggest_int("dfc_step_trs", 1, 3)

    import wandb
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
        config={**vars(args), "optuna_trial_number": trial.number, "optuna_study_name": args.optuna_study_name},
    )

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    print(f"Optuna trial {trial.number} (wandb run {run.id}): fitting {len(subjects)} subjects: {subjects}"
          + (f" (parallel, gpus={args.gpus})" if args.gpus else " (sequential)"))
    print(f"  sampled: learning_rate={args.learning_rate:.4g} learning_rate_bold={args.learning_rate_bold:.4g} "
          f"bold_fc_weight={args.bold_fc_weight:.4g} bold_dfc_weight={args.bold_dfc_weight:.4g} "
          f"bold_psd_weight={args.bold_psd_weight:.4g} dfc_window_trs={args.dfc_window_trs} "
          f"dfc_step_trs={args.dfc_step_trs}")

    try:
        if args.gpus:
            results = _run_parallel(wandb, run, args, subjects)
        else:
            results = _run_sequential(wandb, run, args, subjects)
        objective = _log_aggregate(wandb, *results, n_subjects=len(subjects))
    except Exception:
        # Report the failure to Optuna too, not just SLURM's exit code --
        # otherwise this trial sits stuck in RUNNING state in the shared
        # study forever (Optuna has no external liveness check the way
        # wandb's server does -- it only ever learns a trial's outcome from
        # an explicit tell()), which would silently shrink the effective
        # search budget every time a trial crashes.
        wandb.finish(exit_code=1)
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        raise

    wandb.finish()
    study.tell(trial, objective)
    print(f"Optuna trial {trial.number} done: combined_loss={objective}")


if __name__ == "__main__":
    main()
