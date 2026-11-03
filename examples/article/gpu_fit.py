#!/usr/bin/env python
"""GPU side of the article analyses: the extra fits / re-simulations the
saved population results cannot answer on their own.

Every subcommand starts from a finished base run (``--base-run``, a
``<root>/<subject>_both_joint`` folder) and rebuilds its configuration from
its ``config.json``, so all hyperparameters match the population fit
exactly; only what the experiment changes is overridden.

  fit --variant eeg-only | bold-only | weight=<w> | half-A | half-B
      Section 3 (joint vs single modality; EEG-weight trade-off) and
      section 2 held-out data. Single-modality fits use the alternating
      schedule restricted to one loss (bold-only steps BOLD every epoch);
      ``half-A`` fits EEG on alternating 15-chunk blocks A and BOLD on the
      first half of the run (``half-B`` the complement) and writes
      ``holdout.json`` for ``fit_quality.py --heldout-root``.
      Output: ``<out-root>/atlas-<n>/<subject>_<variant>/`` with the same
      files ``eeg_bold_fit_cli.py`` writes.

  recovery
      Section 4 parameter recovery: draw "true" per-node parameters (group
      median of each region + ``--variation`` x the between-subject SD,
      from the population under ``--pop-root``), simulate synthetic EEG
      (mean PSD over ``--n-target-seeds`` noise realizations) and BOLD with
      them on this subject's connectome/leadfield, then fit from the default
      start with the fit's own noise seed (a different realization, so no
      inverse crime). Writes ``true_params.npz`` next to the usual outputs.

  fresh-noise
      Section 4: score the fitted (best-epoch) and the starting parameters
      under ``--seeds`` new noise realizations without refitting; first
      re-simulates the original seed to check it reproduces the saved
      ``simulation_results.npz``. Writes ``fresh_noise.json``.

  tasks
      Print one command per job (subjects picked young/old balanced among
      working fits) for ``run_gpu_tasks.sh``. CPU only.

    pixi run python examples/article/gpu_fit.py fit \\
        --base-run new_leadfields/1/atlas-100/010004_both_joint --variant eeg-only \\
        --out-root article_fits/eeg_only
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (DEFAULT_BIDS, DEFAULT_ROOT, Subject, eeg_metrics, fc_r, fcd_ks,  # noqa: E402
                     holdout_blocks, load_runs, participants)


# ---------------------------------------------------------------------------
# context helpers (jax imported lazily: `tasks` must stay CPU-only)

def _jax():
    from parrot_neuro.optimization import config
    config.apply_jax_env()
    import jax
    jax.config.update("jax_enable_x64", True)
    import matplotlib
    matplotlib.use("Agg")


def cfg_from_run(run_dir: Path, bids: Path, output_dir: Path | None = None, **overrides):
    """``BoldFitConfig`` identical to the one ``run_dir`` was fitted with
    (bids root swapped for the local one), plus ``overrides``."""
    from parrot_neuro.optimization import config
    d = json.loads((run_dir / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(config.BoldFitConfig)}
    kw = {k: v for k, v in d.items() if k in fields and k not in ("subject", "output_dir", "learnable_params")}
    kw["learnable_params"] = tuple(config.LearnableParam(**lp) for lp in d["learnable_params"])
    kw["init_keep"] = tuple(kw.get("init_keep", ()))
    kw.update(overrides)
    subject = Subject(str(bids), d["subject"]["subject"])
    return config.BoldFitConfig(subject=subject, output_dir=str(output_dir or run_dir), **kw)


def rebuild_with_seed(ctx, noise_seed: int):
    """Same subject/forward model/targets, new noise realization (and the
    network's default start) -- skips reloading the leadfield."""
    from parrot_neuro.optimization.network import build_network
    from parrot_neuro.optimization.train import build_simulators, learnable_partition
    cfg = dataclasses.replace(ctx.cfg, noise_seed=noise_seed)
    network, solver, _ = build_network(
        ctx.mask_cortical, ctx.sc.weights, ctx.sc.delays, ctx.sc.num_nodes,
        learnable_params=cfg.learnable_params, base_sigma=cfg.base_sigma, noise_seed=noise_seed,
        solver_block_size=cfg.solver_block_size, init_seed=cfg.init_seed, init_spread=cfg.init_spread,
        init_keep=cfg.init_keep,
    )
    sims = build_simulators(
        network, solver, cfg.t0, cfg.dt, cfg.t1_eeg, cfg.t1_bold, cfg.tr_ms, cfg.bold_downsample_ms,
        t1_warmup=cfg.t1_warmup, bold_model=cfg.bold_model,
        balloon_taus=cfg.balloon_taus, balloon_tauf=cfg.balloon_tauf, balloon_tauo=cfg.balloon_tauo,
        balloon_alpha=cfg.balloon_alpha, balloon_Eo=cfg.balloon_Eo, balloon_vo=cfg.balloon_vo,
        balloon_TE=cfg.balloon_TE,
    )
    diff, static = learnable_partition(sims.params, cfg.learnable_params)
    return dataclasses.replace(ctx, cfg=cfg, network=network, solver=solver, simulators=sims,
                               diff_params_init=diff, static_params=static)


def with_values(ctx, values: dict):
    """``ctx.diff_params_init`` with every learnable leaf set to ``values``
    (natural units) -- same in-place ``tree_at`` swap as
    ``postfit_diagnostics_cli.py`` (keeps the bounded-parameter closures)."""
    import equinox as eqx
    import jax.numpy as jnp
    diff = ctx.diff_params_init
    for lp in ctx.cfg.learnable_params:
        holder = diff["dynamics"] if lp.location == "dynamics" else diff["coupling"]["delayed"]
        orig = holder[lp.name]
        holder[lp.name] = eqx.tree_at(lambda p: p.value, orig,
                                      orig.inverse_transform(jnp.asarray(values[lp.name], dtype=jnp.float64)))
    return diff


def simulate(ctx, diff, channel_indices) -> dict:
    """Scalp PSD (channels x freqs) + BOLD (T x N) of one parameter set."""
    import equinox as eqx
    import jax.numpy as jnp
    from parrot_neuro.optimization import connectivity
    from parrot_neuro.optimization.forward import project_to_scalp
    from parrot_neuro.optimization.signal import compute_psd
    cfg = ctx.cfg
    combined = eqx.combine(diff, ctx.static_params)
    ys = ctx.simulators.simulator_eeg(combined).ys
    settle, stride = int(cfg.eeg_settle_ms / cfg.dt), int(cfg.eeg_stride_ms / cfg.dt)
    src = (ys[settle::stride, 1].T - ys[settle::stride, 2].T) * jnp.atleast_2d(ctx.mask_cortical).T
    eeg = project_to_scalp(src, channel_indices, ctx.leadfield, ctx.smoothing_blocks, ctx.dipole_labels)
    bold = connectivity.extract_bold_2d(ctx.simulators.simulator_bold(combined))
    return {"eeg_psd": np.asarray(compute_psd(eeg)), "bold": np.asarray(bold)}


def save_fit(ctx, result, out_dir: Path, dataset=None) -> None:
    """The files ``eeg_bold_fit_cli.py`` writes after a fit (+ diagnostics
    when ``dataset`` is given), so every report reads these runs unchanged."""
    from parrot_neuro.optimization import diagnostics, train
    lp = ctx.cfg.learnable_params
    eeg_ratio = train.relative_final_loss(result.loss_history_eeg)
    bold_ratio = train.relative_final_loss(result.loss_history_bold)
    np.save(out_dir / "loss_history_eeg.npy", np.array(result.loss_history_eeg))
    np.save(out_dir / "loss_history_bold.npy", np.array(result.loss_history_bold))
    np.savez(out_dir / "optimized_params.npz", **train.extract_learnable_values(result.diff_params, lp),
             loss_eeg=np.array(result.loss_history_eeg), loss_bold=np.array(result.loss_history_bold),
             eeg_loss_ratio=np.nan if eeg_ratio is None else eeg_ratio,
             bold_loss_ratio=np.nan if bold_ratio is None else bold_ratio,
             combined_loss_ratio=(eeg_ratio or 0.0) + (bold_ratio or 0.0))
    best = result.best_diff_params
    if best is not None:
        np.savez(out_dir / "best_params.npz", **train.extract_learnable_values(best, lp),
                 loss_eeg=np.array(result.loss_history_eeg), loss_bold=np.array(result.loss_history_bold),
                 best_epoch=result.best_epoch, best_score=result.best_score,
                 best_loss_eeg=np.nan if result.best_loss_eeg is None else result.best_loss_eeg,
                 best_loss_bold=np.nan if result.best_loss_bold is None else result.best_loss_bold)
    if dataset is None:
        return
    params, label = (best, "best") if best is not None else (result.diff_params, "final")
    diag = diagnostics.run_and_save(ctx, params, result.static_params, dataset, out_dir, params_label=label)
    (out_dir / "diagnostics_metrics.json").write_text(json.dumps(
        {k: float(v) for k, v in diag["metrics"].items()}
        | ({"best_epoch": float(result.best_epoch)} if label == "best" else {}), indent=2))


def _out_dir(out_root: Path, cfg, name: str) -> Path:
    d = out_root / f"atlas-{cfg.atlas}" / f"{cfg.subject.subject}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# fit

VARIANTS = {
    "eeg-only": {"optimize": "eeg", "schedule": "alternating"},
    "bold-only": {"optimize": "bold", "schedule": "alternating", "bold_every": 1},
    "half-A": {}, "half-B": {},
}


def cmd_fit(args) -> None:
    _jax()
    from parrot_neuro.optimization import data, pipeline, train
    if args.variant.startswith("weight="):
        over, name = {"joint_eeg_weight": float(args.variant.split("=", 1)[1])}, f"both_joint_w{args.variant.split('=', 1)[1]}"
    elif args.variant in VARIANTS:
        over, name = dict(VARIANTS[args.variant]), args.variant
    else:
        raise SystemExit(f"unknown --variant {args.variant!r}")
    if args.num_epochs:
        over["num_epochs"] = args.num_epochs
    probe = cfg_from_run(args.base_run, args.bids)
    out_dir = _out_dir(args.out_root, probe, name)
    cfg = cfg_from_run(args.base_run, args.bids, output_dir=out_dir, **over)
    cfg.save()

    dataset = data.load_subject_eeg(cfg.subject, cfg.eeg_task, cfg.chunk_length)
    holdout = None
    if args.variant.startswith("half-"):
        half = 0 if args.variant == "half-A" else 1
        lab = holdout_blocks(len(dataset), args.block_chunks)
        dataset._chunks = [c for c, h in zip(dataset._chunks, lab) if h == half]
        holdout = {"eeg_half": half, "block_chunks": args.block_chunks, "n_train_chunks": len(dataset)}
    ctx = pipeline.build_context(cfg, dataset)
    if holdout is not None:
        T = int(ctx.sc.empirical_bold.shape[0])
        h = T // 2
        train_trs, test_trs = ([0, h], [h, T]) if holdout["eeg_half"] == 0 else ([h, T], [0, h])
        ctx.sc.empirical_bold = ctx.sc.empirical_bold[train_trs[0]:train_trs[1]]
        holdout |= {"bold_train_trs": train_trs, "bold_test_trs": test_trs, "base_run": str(args.base_run)}
        (out_dir / "holdout.json").write_text(json.dumps(holdout, indent=2))

    np.savez(out_dir / "init_params.npz", **train.extract_learnable_values(ctx.diff_params_init, cfg.learnable_params))
    result = pipeline.fit(ctx)
    save_fit(ctx, result, out_dir, None if args.skip_diagnostics else dataset)
    print(f"[fit {args.variant}] done -> {out_dir}")


# ---------------------------------------------------------------------------
# recovery

def population_maps(pop_root: Path, bids: Path) -> tuple[dict, dict, list[str]]:
    """Per-param group median / between-subject SD of every connectome
    region (final params scattered onto the connectome axis), + G."""
    from params_report_cli import PER_NODE_PARAMS
    vals, names = {}, None
    for run in load_runs(pop_root, need_sim=False):
        subj = Subject(str(bids), run.sid)
        tc = np.asarray(subj.load.fmri_nodes(run.cfg["atlas"], run.cfg["fmri_task"]).to_conn)
        if names is None:
            names = subj.load.connectivity_labels(run.cfg["atlas"])
        p = run.params("final")
        for k, v in p.items():
            if k == "G":
                vals.setdefault("G", []).append(float(v[0]))
            elif k in PER_NODE_PARAMS:
                row = np.full(len(names), np.nan)
                row[tc] = v
                vals.setdefault(k, []).append(row)
    med, sd = {}, {}
    for k, v in vals.items():
        V = np.asarray(v, float)
        med[k], sd[k] = np.nanmedian(V, axis=0), np.nanstd(V, axis=0)
    return med, sd, names


def cmd_recovery(args) -> None:
    _jax()
    import jax.numpy as jnp
    from parrot_neuro.optimization import connectivity, data, pipeline, train
    from params_report_cli import LIVE_ON

    over = {"noise_seed": args.data_noise_seed}
    if args.num_epochs:
        over["num_epochs"] = args.num_epochs
    probe = cfg_from_run(args.base_run, args.bids)
    out_dir = _out_dir(args.out_root, probe, f"recovery-v{args.variation:g}-s{args.truth_seed}")
    cfg = cfg_from_run(args.base_run, args.bids, output_dir=out_dir, **over)
    fit_seed = probe.noise_seed
    if fit_seed == args.data_noise_seed:
        raise SystemExit("--data-noise-seed must differ from the fit's noise seed")

    # -- truth: group map + subject-like deviation, on this subject's nodes
    med, sd, _ = population_maps(args.pop_root, args.bids)
    tc = np.asarray(cfg.subject.load.fmri_nodes(cfg.atlas, cfg.fmri_task).to_conn)
    rng = np.random.default_rng(args.truth_seed)
    dataset = data.load_subject_eeg(cfg.subject, cfg.eeg_task, cfg.chunk_length)
    ctx = pipeline.build_context(cfg, dataset)
    surface = np.asarray(ctx.mask_cortical) > 0.5
    truth = {}
    for lp in cfg.learnable_params:
        span = lp.high - lp.low
        lo, hi = lp.low + 0.02 * span, lp.high - 0.02 * span
        if lp.location == "coupling":
            truth[lp.name] = np.clip([med[lp.name] + args.variation * sd[lp.name] * rng.standard_normal()], lo, hi)
            continue
        m, s = med[lp.name][tc], sd[lp.name][tc]
        v = m + args.variation * s * rng.standard_normal(len(tc))
        live = surface if LIVE_ON.get(lp.name) == "surface" else ~surface
        v = np.where(live & np.isfinite(v), v, lp.init)          # dead / uncovered nodes keep their start
        truth[lp.name] = np.clip(v, lo, hi)

    # -- synthetic data from the truth (data noise seeds), BOLD from the first
    true_diff = with_values(ctx, truth)
    first = simulate(ctx, true_diff, dataset.channel_indices)
    psds = [first["eeg_psd"]]
    for k in range(1, args.n_target_seeds):
        c = rebuild_with_seed(ctx, args.data_noise_seed + k)
        psds.append(simulate(c, with_values(c, truth), dataset.channel_indices)["eeg_psd"])
    target_psd = np.mean(psds, axis=0)
    bold = np.asarray(connectivity.filter_sim_bold(jnp.asarray(first["bold"]), cfg.tr_ms))
    if not (np.all(np.isfinite(bold)) and np.all(np.isfinite(target_psd))):
        raise SystemExit("synthetic data not finite -- truth outside a stable regime, try another --truth-seed")

    # -- fit: the run's own noise seed, default start, synthetic targets
    fit_ctx = rebuild_with_seed(ctx, fit_seed)
    fit_ctx = dataclasses.replace(fit_ctx, target_psd=jnp.asarray(target_psd),
                                  sc=dataclasses.replace(fit_ctx.sc, empirical_bold=jnp.asarray(bold)))
    dataclasses.replace(cfg, noise_seed=fit_seed).save(out_dir)
    np.savez(out_dir / "true_params.npz", **truth)
    np.savez(out_dir / "init_params.npz",
             **train.extract_learnable_values(fit_ctx.diff_params_init, cfg.learnable_params))
    np.savez(out_dir / "synthetic_data.npz", eeg_psd_target=target_psd, bold=bold,
             mask_cortical=np.asarray(ctx.mask_cortical), to_conn=tc)
    (out_dir / "recovery.json").write_text(json.dumps({
        "base_run": str(args.base_run), "pop_root": str(args.pop_root), "variation": args.variation,
        "truth_seed": args.truth_seed, "data_noise_seeds": [args.data_noise_seed + k for k in range(args.n_target_seeds)],
        "fit_noise_seed": fit_seed}, indent=2))
    result = pipeline.fit(fit_ctx)
    save_fit(fit_ctx, result, out_dir)
    print(f"[recovery] done -> {out_dir}")


# ---------------------------------------------------------------------------
# fresh noise

def _score(sim: dict, z, names, cfg, fc_emp, kmin) -> dict:
    from parrot_neuro.optimization import connectivity
    import jax.numpy as jnp
    m = eeg_metrics(sim["eeg_psd"], z["eeg_psd_target"], z["eeg_freqs"], names, cfg.fmin, cfg.fmax)
    skip = cfg.bold_skip_trs
    fc_sim, _ = connectivity.fc_matrices(sim["bold"], z["bold_emp"], cfg.tr_ms, skip_t=skip)
    out = {"spec_r": m["spec_r"], "alpha_topo_r": m["alpha_topo_r"], "apf_sim": m["apf_sim"],
           "fc_r": fc_r(fc_sim, fc_emp), "finite": bool(np.all(np.isfinite(sim["bold"])))}
    if "fcd_emp" in z.files:
        fcd_sim, _ = connectivity.fcd_matrices(sim["bold"], z["bold_emp"], cfg.tr_ms, cfg.dfc_window_trs,
                                                cfg.dfc_step_trs, skip_t=skip)
        out["fcd_ks"] = fcd_ks(fcd_sim, z["fcd_emp"], kmin)
    # the training EEG loss (numpy twin of train._eeg_psd_loss, gamma off)
    from _common import smooth
    b = slice(int(z["eeg_idx_min"]), int(z["eeg_idx_max"]))
    S, T = smooth(sim["eeg_psd"]), smooth(z["eeg_psd_target"])
    out["eeg_loss"] = float(np.mean((S[:, b] / (S[:, b].sum() + 1e-8) - T[:, b] / (T[:, b].sum() + 1e-8)) ** 2))
    del jnp
    return out


def cmd_fresh_noise(args) -> None:
    _jax()
    from parrot_neuro.optimization import data, pipeline
    from _common import Run, channel_names
    run = Run(args.base_run, json.loads((args.base_run / "config.json").read_text()))
    cfg = cfg_from_run(args.base_run, args.bids)
    z = run.sim
    names = channel_names(run, cfg.subject)
    dataset = data.load_subject_eeg(cfg.subject, cfg.eeg_task, cfg.chunk_length)
    ctx = pipeline.build_context(cfg, dataset)
    fitted, start = run.params(args.params), run.params("init")
    kmin = cfg.dfc_kmin

    ref = simulate(ctx, with_values(ctx, fitted), dataset.channel_indices)
    rel = float(np.max(np.abs(ref["eeg_psd"] - z["eeg_psd_sim"])) / np.max(np.abs(z["eeg_psd_sim"])))
    print(f"[fresh-noise] original seed {cfg.noise_seed} re-simulated: max rel. PSD diff vs saved = {rel:.2e}")
    rows = [{"seed": cfg.noise_seed, "params": "fitted", "original_seed": True,
             **_score(ref, z, names, cfg, z["fc_emp"], kmin)}]
    for s in args.seeds:
        c = rebuild_with_seed(ctx, s)
        for label, vals in (("fitted", fitted), ("start", start)):
            rows.append({"seed": s, "params": label, "original_seed": False,
                         **_score(simulate(c, with_values(c, vals), dataset.channel_indices), z, names, cfg,
                                  z["fc_emp"], kmin)})
        print(f"  seed {s}: fitted fc_r={rows[-2]['fc_r']:.3f} spec_r={rows[-2]['spec_r']:.3f}")
    out_dir = args.out_root / f"atlas-{cfg.atlas}" / run.path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fresh_noise.json").write_text(json.dumps(
        {"subject": run.sid, "base_run": str(args.base_run), "params": args.params,
         "reproduction_max_rel_diff": rel, "rows": rows}, indent=2, default=float))
    print(f"[fresh-noise] -> {out_dir / 'fresh_noise.json'}")


# ---------------------------------------------------------------------------
# tasks

def pick_subjects(root: Path, bids: Path, n: int, seed: int, min_fc: float, max_eeg_loss: float) -> list:
    runs = load_runs(root, need_sim=False)
    p = participants(bids).set_index("subject")
    ok = [r for r in runs if r.sid in p.index and np.isfinite(p.age[r.sid])
          and r.diag.get("fc_corr", np.nan) > min_fc and r.losses["loss_eeg"] < max_eeg_loss]
    rng = np.random.default_rng(seed)
    old = [r for r in ok if p.old[r.sid]]
    young = [r for r in ok if not p.old[r.sid]]
    k_old = min(len(old), n // 2)
    picks = list(rng.choice(old, k_old, replace=False)) + list(rng.choice(young, min(len(young), n - k_old), replace=False))
    return sorted(picks, key=lambda r: r.sid)


def cmd_tasks(args) -> None:
    py = "pixi run python examples/article/gpu_fit.py"
    common = f"--bids {args.bids}" + (f" --num-epochs {args.num_epochs}" if args.num_epochs else "")
    lines = []
    plan = {
        "single": (args.n_single, ["eeg-only", "bold-only"]),
        "weights": (args.n_weights, [f"weight={w}" for w in args.weights]),
        "heldout": (args.n_heldout, ["half-A"]),
    }
    for kind in args.kinds:
        if kind in plan:
            n, variants = plan[kind]
            for r in pick_subjects(args.root, args.bids, n, args.seed, args.min_fc, args.max_eeg_loss):
                for v in variants:
                    dest = {"single": "single_modality", "weights": "eeg_weight", "heldout": "heldout"}[kind]
                    lines.append(f"{py} fit --base-run {r.path} --variant {v} --out-root {args.out}/{dest} {common}")
        elif kind == "recovery":
            for k, r in enumerate(pick_subjects(args.root, args.bids, args.n_recovery, args.seed + 1,
                                                args.min_fc, args.max_eeg_loss)):
                lines.append(f"{py} recovery --base-run {r.path} --pop-root {args.root} --truth-seed {k} "
                             f"--variation {args.variation} --out-root {args.out}/recovery {common}")
        elif kind == "fresh-noise":
            for r in pick_subjects(args.root, args.bids, args.n_fresh, args.seed + 2, args.min_fc, args.max_eeg_loss):
                lines.append(f"{py} fresh-noise --base-run {r.path} --out-root {args.out}/fresh_noise --bids {args.bids}")
        else:
            raise SystemExit(f"unknown kind {kind!r}")
    text = "\n".join(lines) + "\n"
    if args.write:
        args.write.write_text(text)
        print(f"{len(lines)} tasks -> {args.write}")
    else:
        print(text, end="")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def base(p):
        p.add_argument("--base-run", required=True, type=Path, help="finished <subject>_both_joint run folder")
        p.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
        p.add_argument("--out-root", type=Path, required=True)
        p.add_argument("--num-epochs", type=int, default=None, help="override (e.g. 2 for a smoke test)")

    p = sub.add_parser("fit")
    base(p)
    p.add_argument("--variant", required=True, help="eeg-only | bold-only | weight=<w> | half-A | half-B")
    p.add_argument("--block-chunks", type=int, default=15)
    p.add_argument("--skip-diagnostics", action="store_true")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("recovery")
    base(p)
    p.add_argument("--pop-root", type=Path, default=DEFAULT_ROOT, help="population the truth is drawn from")
    p.add_argument("--variation", type=float, default=1.0, help="x between-subject SD added to the group map")
    p.add_argument("--truth-seed", type=int, default=0)
    p.add_argument("--data-noise-seed", type=int, default=1001)
    p.add_argument("--n-target-seeds", type=int, default=4, help="noise realizations averaged into the EEG target")
    p.set_defaults(fn=cmd_recovery)

    p = sub.add_parser("fresh-noise")
    base(p)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(2001, 2011)))
    p.add_argument("--params", choices=("best", "final"), default="best")
    p.set_defaults(fn=cmd_fresh_noise)

    p = sub.add_parser("tasks")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
    p.add_argument("--out", default="article_fits", help="out-root prefix of the generated commands")
    p.add_argument("--kinds", nargs="+", default=["single", "weights", "heldout", "recovery", "fresh-noise"])
    p.add_argument("--n-single", type=int, default=10)
    p.add_argument("--n-weights", type=int, default=5)
    p.add_argument("--weights", nargs="+", default=["1e4", "1e6"], help="1e5 = the base run itself")
    p.add_argument("--n-heldout", type=int, default=20)
    p.add_argument("--n-recovery", type=int, default=8)
    p.add_argument("--variation", type=float, default=1.0)
    p.add_argument("--n-fresh", type=int, default=20)
    p.add_argument("--min-fc", type=float, default=0.05)
    p.add_argument("--max-eeg-loss", type=float, default=1e-4)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", type=Path, default=None)
    p.set_defaults(fn=cmd_tasks)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
