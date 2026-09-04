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

--gpus (comma-separated device indices, e.g. "0,1,2,3") switches from the
default sequential single-process loop to a parallel mode: subjects are
chunked into rounds of len(--gpus), each round's subjects are fit
SIMULTANEOUSLY as separate eeg_bold_fit_cli.py subprocesses (one pinned to
each listed GPU via CUDA_VISIBLE_DEVICES), and this process waits for each
round, then replays the finished workers' saved loss histories/diagnostics
into wandb (it can't stream live -- the fit ran in another process). Choosing
a subject count that's an exact multiple of len(--gpus) (e.g. 4 subjects on 4
GPUs, one round) uses every reserved GPU for the whole trial with no idle
time; a remainder subject count leaves some GPUs idle during the last round
(harmless correctness-wise, just billed-but-idle capacity on a cluster that
charges for reserved-not-just-used resources -- see hpc/leonardo/README.md).

Each worker's stdout/stderr goes to its own file under
<output-root>/<wandb-run-id>/worker_logs/<subject>.log (not interleaved into
this process's own output) -- tail that file for a subject that seems stuck,
rather than guessing from 4 processes' output mixed into one stream.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


def _ratio(history):
    """Same definition as train.relative_final_loss -- duplicated (not
    imported) so the --gpus parallel orchestrator never has to import
    parrot_neuro.optimization.train (which imports jax at module load time).
    The orchestrator manages subprocesses/wandb only; it must not itself
    initialize a CUDA context that could collide with a worker subprocess's
    assigned GPU. Keep in sync with train.relative_final_loss if that changes.
    """
    if not history or not history[0]:
        return None
    return history[-1] / history[0]


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
                         "with the SAME sampled hyperparameters, in ONE wandb run")
    p.add_argument("--output-root", default=os.environ.get("SWEEP_OUTPUT_ROOT", "eeg_bold_fit_sweep_res"),
                    help="results land under <output-root>/atlas-<atlas>/<wandb-run-id>/<subject>_<optimize> "
                         "(sequential mode) or <output-root>/<wandb-run-id>/atlas-<atlas>/<subject>_<optimize> "
                         "(--gpus parallel mode, via eeg_bold_fit_cli.py's own atlas-suffixing)")
    p.add_argument("--atlas", type=int, default=1000, choices=(100, 1000))
    p.add_argument("--spacing", default="2.0", help="dipole spacing in mm (string)")
    p.add_argument("--leadfield-label", default="duneuroCGAL")
    p.add_argument("--optimize", default="both", choices=("eeg", "bold", "both"))
    p.add_argument("--bold-model", default="hrf", choices=("hrf", "balloon"),
                    help="see eeg_bold_fit_cli.py --bold-model -- 'hrf' (linear HRF-kernel "
                         "convolution, default) or 'balloon' (Friston/Deco Balloon-Windkessel "
                         "hemodynamic ODE), applied identically to every subject in this sweep "
                         "trial.")
    p.add_argument("--schedule", default="alternating", choices=("alternating", "phased", "joint"),
                    help="see eeg_bold_fit_cli.py --schedule -- same three strategies (phased "
                         "splits --num-epochs in half), applied identically to every subject in "
                         "this sweep trial.")
    p.add_argument("--joint-eeg-weight", type=float, default=1e5,
                    help="see eeg_bold_fit_cli.py --joint-eeg-weight (--schedule joint only)")
    p.add_argument("--joint-bold-weight", type=float, default=1.0,
                    help="see eeg_bold_fit_cli.py --joint-bold-weight (--schedule joint only)")
    p.add_argument("--num-epochs", type=int, default=_env_int("SWEEP_NUM_EPOCHS", 300))
    p.add_argument("--bold-every", type=int, default=2)
    p.add_argument("--eeg-task", default="eyesclosed")
    p.add_argument("--fmri-task", default="rest")
    p.add_argument("--noise-seed", type=int, default=69)
    p.add_argument("--t1-warmup", type=float, default=30_000.0,
                    help="see eeg_bold_fit_cli.py --t1-warmup; -1 = old full-t1_bold warm-up")
    p.add_argument("--solver-block-size", type=int, default=1400,
                    help="see eeg_bold_fit_cli.py --solver-block-size; must be an exact multiple "
                         "of the BOLD period in raw steps (tr_ms/dt); 0 = unblocked")
    p.add_argument("--gamma-weight", type=float, default=0.0)
    p.add_argument("--skip-diagnostics", action="store_true",
                    help="fit + save params/losses only -- skip plots (faster smoke test)")
    p.add_argument("--gpus", default=None,
                    help="comma-separated GPU device indices (e.g. '0,1,2,3') -- switches to "
                         "parallel mode: subjects are chunked into rounds of this many, each "
                         "round's subjects fit SIMULTANEOUSLY as separate eeg_bold_fit_cli.py "
                         "subprocesses, one pinned per listed GPU. Default (unset) = sequential, "
                         "single-process, one subject at a time -- see module docstring.")

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


def _log_subject_summary(wandb, subject_id, loss_eeg, loss_bold, metrics, figures):
    """Log one subject's final metrics/ratios/diagnostics to wandb -- the
    part identical between the sequential (in-process) and --gpus (read back
    from a finished subprocess) paths. Returns (eeg_ratio, bold_ratio,
    final_eeg, final_bold) for the caller's aggregate/combined_loss tally.
    """
    final_eeg = loss_eeg[-1] if len(loss_eeg) else None
    final_bold = loss_bold[-1] if len(loss_bold) else None
    # EEG (~1e-6) and BOLD (~1e-1) losses live on completely different scales
    # -- see train.relative_final_loss's docstring -- so the ratio to each
    # subject's own early loss, not the raw value, is what feeds the combined
    # objective below.
    eeg_ratio = _ratio(loss_eeg)
    bold_ratio = _ratio(loss_bold)
    summary = {f"subj_{subject_id}/final_eeg_loss": final_eeg,
               f"subj_{subject_id}/final_bold_loss": final_bold,
               f"subj_{subject_id}/eeg_loss_ratio": eeg_ratio,
               f"subj_{subject_id}/bold_loss_ratio": bold_ratio}
    for name, value in metrics.items():
        summary[f"subj_{subject_id}/{name}"] = float(value)
    for name, path in figures.items():
        summary[f"subj_{subject_id}/plots/{name}"] = wandb.Image(str(path))
    wandb.log(summary)
    print(f"[{subject_id}] final EEG loss: {final_eeg} (ratio {eeg_ratio})  "
          f"final BOLD loss: {final_bold} (ratio {bold_ratio})")
    return eeg_ratio, bold_ratio, final_eeg, final_bold


def _log_aggregate(wandb, per_subject_combined, per_subject_eeg, per_subject_bold,
                    per_subject_eeg_ratio, per_subject_bold_ratio, n_subjects):
    import numpy as np
    # aggregate/combined_loss (the sweep's minimized metric -- see
    # sweep_eeg_bold.yaml) is the ratio-based combination; the raw *_loss_mean
    # values are logged alongside purely for human-readable context, not
    # optimized directly (see _log_subject_summary's scale-mismatch note).
    aggregate = {"aggregate/combined_loss": float(np.mean(per_subject_combined))}
    if per_subject_eeg:
        aggregate["aggregate/eeg_loss_mean"] = float(np.mean(per_subject_eeg))
    if per_subject_bold:
        aggregate["aggregate/bold_loss_mean"] = float(np.mean(per_subject_bold))
    if per_subject_eeg_ratio:
        aggregate["aggregate/eeg_loss_ratio_mean"] = float(np.mean(per_subject_eeg_ratio))
    if per_subject_bold_ratio:
        aggregate["aggregate/bold_loss_ratio_mean"] = float(np.mean(per_subject_bold_ratio))
    wandb.log(aggregate)
    print(f"Aggregate over {n_subjects} subjects: {aggregate}")


def _run_sequential(wandb, run, args, subjects):
    """One subject at a time, in THIS process -- needs jax, so it's imported
    here rather than at module level (the --gpus path never needs it; see
    _run_parallel)."""
    from parrot_neuro.optimization import config
    config.apply_jax_env()  # must run before any jax import

    import jax
    jax.config.update("jax_enable_x64", True)

    import matplotlib
    matplotlib.use("Agg")

    import numpy as np

    from parrot_neuro import Subject
    from parrot_neuro.optimization import data, diagnostics, pipeline

    per_subject_combined, per_subject_eeg, per_subject_bold = [], [], []
    per_subject_eeg_ratio, per_subject_bold_ratio = [], []

    for subject_id in subjects:
        subject = Subject(args.bids_root, subject_id)

        # Schedule suffix only when non-default -- mirrors eeg_bold_fit_cli.py's
        # own output_dir naming exactly, so _run_parallel's post-hoc out_dir
        # reconstruction (which has no in-process cfg to read it back from)
        # stays in sync with whichever path actually wrote the results.
        schedule_suffix = "" if args.schedule == "alternating" else f"_{args.schedule}"
        output_root = os.path.join(args.output_root, f"atlas-{args.atlas}", run.id)
        output_dir = os.path.join(output_root, f"{subject.subject}_{args.optimize}{schedule_suffix}")
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
            bold_model=args.bold_model,
            schedule=args.schedule,
            joint_eeg_weight=args.joint_eeg_weight,
            joint_bold_weight=args.joint_bold_weight,
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

        from parrot_neuro.optimization import train
        optimized = train.extract_learnable_values(result.diff_params, cfg.learnable_params)
        np.savez(out_dir / "optimized_params.npz", **optimized,
                  loss_eeg=np.array(result.loss_history_eeg), loss_bold=np.array(result.loss_history_bold))

        metrics, figures = {}, {}
        if not args.skip_diagnostics:
            if dataset is None:
                dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
                print(f"[{subject_id}] loaded EEG for visualization only (not a fit target)")
            diag = diagnostics.run_and_save(ctx, result.diff_params, result.static_params, dataset, out_dir)
            metrics, figures = diag["metrics"], diag["figures"]

        eeg_ratio, bold_ratio, final_eeg, final_bold = _log_subject_summary(
            wandb, subject_id, result.loss_history_eeg, result.loss_history_bold, metrics, figures)

        per_subject_combined.append((eeg_ratio or 0.0) + (bold_ratio or 0.0))
        if final_eeg is not None:
            per_subject_eeg.append(final_eeg)
        if final_bold is not None:
            per_subject_bold.append(final_bold)
        if eeg_ratio is not None:
            per_subject_eeg_ratio.append(eeg_ratio)
        if bold_ratio is not None:
            per_subject_bold_ratio.append(bold_ratio)
        print(f"[{subject_id}] saved to {out_dir}")

    return per_subject_combined, per_subject_eeg, per_subject_bold, per_subject_eeg_ratio, per_subject_bold_ratio


def _worker_argv(worker_script, args, subject_id, worker_output_root):
    argv = [
        # -u: unbuffered stdout -- without it, Python fully buffers stdout
        # once it's not a TTY (true here regardless of the orchestrator's own
        # stdout handling: it always ends up redirected to a file, either the
        # SLURM job's .out or this worker's own log), so prints can sit
        # invisible in the buffer for a long time. That looks EXACTLY like a
        # hang from the outside -- same class of gotcha sweep_train.sbatch's
        # own `python -u` already guards against for the orchestrator itself.
        sys.executable, "-u", str(worker_script),
        "--bids-root", args.bids_root,
        "--subject", subject_id,
        "--output-root", worker_output_root,
        "--atlas", str(args.atlas),
        "--spacing", args.spacing,
        "--leadfield-label", args.leadfield_label,
        "--optimize", args.optimize,
        "--bold-model", args.bold_model,
        "--schedule", args.schedule,
        "--joint-eeg-weight", str(args.joint_eeg_weight),
        "--joint-bold-weight", str(args.joint_bold_weight),
        "--num-epochs", str(args.num_epochs),
        "--bold-every", str(args.bold_every),
        "--eeg-task", args.eeg_task,
        "--fmri-task", args.fmri_task,
        "--noise-seed", str(args.noise_seed),
        "--t1-warmup", str(args.t1_warmup),
        "--solver-block-size", str(args.solver_block_size),
        "--gamma-weight", str(args.gamma_weight),
        "--learning-rate", str(args.learning_rate),
        "--learning-rate-bold", str(args.learning_rate_bold),
        "--bold-fc-weight", str(args.bold_fc_weight),
        "--bold-dfc-weight", str(args.bold_dfc_weight),
        "--bold-psd-weight", str(args.bold_psd_weight),
        "--dfc-window-trs", str(args.dfc_window_trs),
        "--dfc-step-trs", str(args.dfc_step_trs),
    ]
    if args.skip_diagnostics:
        argv.append("--skip-diagnostics")
    return argv


def _load_worker_result(out_dir: Path):
    import numpy as np
    loss_eeg = np.load(out_dir / "loss_history_eeg.npy").tolist()
    loss_bold = np.load(out_dir / "loss_history_bold.npy").tolist()
    metrics_path = out_dir / "diagnostics_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    figures = {p.stem: p for p in sorted(out_dir.glob("*.png"))}
    return loss_eeg, loss_bold, metrics, figures


def _log_subject_epochs(wandb, subject_id, loss_eeg, loss_bold, bold_every):
    """Replay a finished subject's per-epoch loss curve into wandb -- the fit
    ran in a separate process, so unlike _run_sequential's live on_epoch
    callback, this happens all at once after the subprocess exits. Each point
    is tagged with its true epoch index (subj_<id>/epoch); view with that as
    a custom x-axis in the wandb UI, not the default Step (which will show
    all EEG points before all BOLD points, not chronologically interleaved --
    harmless for the plotted curve itself, just an odd Step ordering).
    """
    for i, v in enumerate(loss_eeg):
        wandb.log({f"subj_{subject_id}/epoch": i, f"subj_{subject_id}/eeg_loss": v})
    for i, v in enumerate(loss_bold):
        wandb.log({f"subj_{subject_id}/epoch": (i + 1) * bold_every - 1, f"subj_{subject_id}/bold_loss": v})


def _run_parallel(wandb, run, args, subjects):
    """len(--gpus) subjects at a time, each as its own eeg_bold_fit_cli.py
    subprocess pinned to one GPU -- see module docstring. Deliberately never
    imports jax/parrot_neuro.optimization.train: this process only manages
    subprocesses, reads their saved .npy/.json output back, and logs to
    wandb -- it must not itself claim a CUDA context that could collide with
    a worker's assigned GPU.
    """
    from parrot_neuro import Subject  # jax-free (core BIDS API, not optimization)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    worker_script = Path(__file__).resolve().parent / "eeg_bold_fit_cli.py"
    worker_output_root = os.path.join(args.output_root, run.id)
    log_dir = Path(worker_output_root) / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    per_subject_combined, per_subject_eeg, per_subject_bold = [], [], []
    per_subject_eeg_ratio, per_subject_bold_ratio = [], []

    rounds = [subjects[i:i + len(gpus)] for i in range(0, len(subjects), len(gpus))]
    print(f"Parallel mode: {len(subjects)} subjects over {len(gpus)} GPUs {gpus} "
          f"-> {len(rounds)} round(s): {rounds}")

    for round_idx, round_subjects in enumerate(rounds):
        procs = []
        for subject_id, gpu_id in zip(round_subjects, gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            # One subdir per GPU, not one shared cache for the whole job --
            # concurrent workers hitting the SAME jax compilation cache dir
            # risk lock contention / a corrupt shared entry (same hazard
            # optim_cohort.sbatch already avoids for job arrays, one dir per
            # subject there; here it's one dir per GPU since a single worker
            # process owns each GPU for the whole round).
            base_cache = env.get("PARROT_JAX_CACHE_DIR", os.path.expanduser("~/.cache/jax"))
            env["PARROT_JAX_CACHE_DIR"] = os.path.join(base_cache, f"gpu-{gpu_id}")
            argv = _worker_argv(worker_script, args, subject_id, worker_output_root)
            log_path = log_dir / f"{subject_id}.log"
            log_file = open(log_path, "w")
            print(f"[round {round_idx}] launching {subject_id} on GPU {gpu_id} -> {log_path}")
            # stdin=DEVNULL: these workers never need input, and 4 processes
            # all inheriting the SAME live terminal/job stdin is a known way
            # to get an unpredictable hang if anything anywhere ever probes
            # or reads it. stdout/stderr to a per-subject file, not inherited
            # -- 4 processes interleaving raw output into one shared stream
            # is nearly unreadable anyway, and this makes "which subject is
            # actually stuck" a one-command check (tail -f) instead of a
            # guess from interleaved noise.
            proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((subject_id, gpu_id, proc, log_file))

        failed = []
        for subject_id, gpu_id, proc, log_file in procs:
            rc = proc.wait()
            log_file.close()
            print(f"[round {round_idx}] {subject_id} (GPU {gpu_id}) finished rc={rc}")
            if rc != 0:
                failed.append((subject_id, rc))
        if failed:
            raise RuntimeError(f"round {round_idx}: subject(s) failed: {failed} -- see {log_dir}/<subject>.log")

        # Matches eeg_bold_fit_cli.py's own output_dir naming (see
        # _run_sequential's identical schedule_suffix comment) -- this process
        # never sees the worker's in-process cfg, so it must reconstruct the
        # exact same path the subprocess wrote to.
        schedule_suffix = "" if args.schedule == "alternating" else f"_{args.schedule}"
        for subject_id, _gpu_id, _proc, _log_file in procs:
            subject = Subject(args.bids_root, subject_id)
            out_dir = (Path(worker_output_root) / f"atlas-{args.atlas}"
                       / f"{subject.subject}_{args.optimize}{schedule_suffix}")
            loss_eeg, loss_bold, metrics, figures = _load_worker_result(out_dir)
            _log_subject_epochs(wandb, subject_id, loss_eeg, loss_bold, args.bold_every)
            eeg_ratio, bold_ratio, final_eeg, final_bold = _log_subject_summary(
                wandb, subject_id, loss_eeg, loss_bold, metrics, figures)

            per_subject_combined.append((eeg_ratio or 0.0) + (bold_ratio or 0.0))
            if final_eeg is not None:
                per_subject_eeg.append(final_eeg)
            if final_bold is not None:
                per_subject_bold.append(final_bold)
            if eeg_ratio is not None:
                per_subject_eeg_ratio.append(eeg_ratio)
            if bold_ratio is not None:
                per_subject_bold_ratio.append(bold_ratio)

    return per_subject_combined, per_subject_eeg, per_subject_bold, per_subject_eeg_ratio, per_subject_bold_ratio


def main() -> None:
    args = parse_args()

    import wandb

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
    print(f"Sweep trial {run.id}: fitting {len(subjects)} subjects: {subjects}"
          + (f" (parallel, gpus={args.gpus})" if args.gpus else " (sequential)"))

    if args.gpus:
        results = _run_parallel(wandb, run, args, subjects)
    else:
        results = _run_sequential(wandb, run, args, subjects)

    _log_aggregate(wandb, *results, n_subjects=len(subjects))
    wandb.finish()


if __name__ == "__main__":
    main()
