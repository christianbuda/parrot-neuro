"""End-to-end orchestration: subject forward model + connectome -> network ->
alternating EEG+BOLD fit.

The thin entry point the rest of ``tvbeeg`` composes into. A driver script
next to this package does roughly::

    from parrot_neuro import Subject
    from tvbeeg import config, data, pipeline
    config.apply_jax_env()                      # before any jax import!

    subject = Subject(bids_root, subject_id)
    cfg = config.BoldFitConfig(subject=subject, num_epochs=200)
    dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
    ctx = pipeline.build_context(cfg, dataset)
    result = pipeline.fit(ctx)

Everything downstream of "I have a dataset" — SC/BOLD loading, dipole-label
alignment, network construction, the two simulators, and the training loop —
is fully generic and lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import numpy as np

from . import config
from .connectivity import (
    StructuralConnectivity,
    drop_labels_vector,
    load_structural_connectivity,
    remap_dipole_labels,
)
from .forward import get_electric_signals
from .network import build_network
from .train import (
    FitResult,
    Simulators,
    build_simulators,
    compute_target_psd,
    learnable_partition,
    make_bold_loss_fn,
    make_eeg_loss_fn,
    make_optimizer,
    make_update_steps,
    print_learnable_params,
    run_alternating_fit,
)


@dataclass
class ExperimentContext:
    """Everything needed to run (or re-run) the alternating fit once."""

    cfg: config.BoldFitConfig
    mask_cortical: np.ndarray
    leadfield: jnp.ndarray
    smoothing_weights: jnp.ndarray
    dipole_labels: jnp.ndarray
    sc: StructuralConnectivity
    network: object
    solver: object
    simulators: Simulators
    dataset: object  # data.SingleSubjectDataset
    target_psd: jnp.ndarray
    freqs: np.ndarray
    idx_min: int
    idx_max: int
    diff_params_init: object
    static_params: object
    optimizer: object


def build_context(cfg: config.BoldFitConfig, dataset) -> ExperimentContext:
    """Load forward model + connectome, align them, and build the network.

    ``dataset`` must already be sliced to the recordings you want the EEG
    loss fit against (e.g. resting state) — see the module docstring.
    """
    # --- forward model (leadfield, dipole labels, cortical/subcortical type) ---
    # representative_dipole isn't used by this pipeline — skip computing it
    # (saves a second dense (N_dip, N_dip) array on top of the smoothing matrix).
    leadfield, weights_matrices, dipole_labels, orient_atlas, _representative_dipole = (
        get_electric_signals(
            cfg.subject,
            spacing=cfg.spacing,
            atlas=cfg.atlas,
            leadfield_label=cfg.leadfield_label,
            compute_representative_dipole=False,
        )
    )

    # --- structural connectivity + empirical BOLD, aligned to one region set ---
    sc = load_structural_connectivity(
        cfg.subject,
        atlas=cfg.atlas,
        conduction_speed=cfg.conduction_speed,
        fmri_task=cfg.fmri_task,
    )

    # dipole_labels/orient_atlas are in the *pre-missing-labels* connectome
    # region indexing; re-index them onto sc's (post-drop) region set and
    # drop dipoles that belonged to a region BOLD has no coverage for.
    dipole_labels, valid = remap_dipole_labels(
        dipole_labels, sc.missing_labels, sc.n_full_regions
    )
    # float32: these two are dense (N_dip, N_dip) / (N_elec, N_dip) fixed geometric
    # projection matrices, not part of the integrated ODE state -- they don't need
    # x64 precision, and at tens of thousands of dipoles the float64 versions are
    # large enough (multi-GB) to make the host->GPU transfer fail to pin ("could not
    # allocate pinned host memory" / CUDA_ERROR_INVALID_VALUE). Downstream matmuls
    # against the (float64) dynamics state still promote to float64 as usual.
    smoothing_weights = np.asarray(smoothing_weights, dtype=np.float32)[valid][:, valid]
    leadfield = np.asarray(leadfield, dtype=np.float32)[:, valid]

    mask_cortical = np.where(np.isin(orient_atlas, ["N", "G", "P"]), 1.0, 0.0)
    mask_cortical = drop_labels_vector(mask_cortical, sc.missing_labels)

    dipole_labels = jnp.array(dipole_labels)
    smoothing_weights = jnp.array(smoothing_weights)
    leadfield = jnp.array(leadfield)

    # --- network + two simulators (short EEG horizon, long BOLD horizon) ---
    network, solver, _brain_model = build_network(
        mask_cortical, sc.weights, sc.delays, sc.num_nodes,
        learnable_params=cfg.learnable_params, base_sigma=cfg.base_sigma, noise_seed=cfg.noise_seed,
    )
    simulators = build_simulators(
        network, solver, cfg.t0, cfg.dt, cfg.t1_eeg, cfg.t1_bold,
        cfg.tr_ms, cfg.bold_downsample_ms,
    )

    diff_params_init, static_params = learnable_partition(simulators.params, cfg.learnable_params)

    # --- EEG loss target (mean PSD across this subject's chunks) ---
    target_psd = compute_target_psd(dataset)
    freqs = np.fft.rfftfreq(cfg.chunk_length, d=1.0 / cfg.fs)
    idx_min = int(np.searchsorted(freqs, cfg.fmin))
    idx_max = int(np.searchsorted(freqs, cfg.fmax))

    optimizer = make_optimizer(cfg.learning_rate, cfg.grad_clip_norm)

    return ExperimentContext(
        cfg=cfg, mask_cortical=mask_cortical, leadfield=leadfield,
        smoothing_weights=smoothing_weights, dipole_labels=dipole_labels, sc=sc,
        network=network, solver=solver, simulators=simulators, dataset=dataset,
        target_psd=target_psd, freqs=freqs, idx_min=idx_min, idx_max=idx_max,
        diff_params_init=diff_params_init, static_params=static_params,
        optimizer=optimizer,
    )


def fit(ctx: ExperimentContext) -> FitResult:
    """Run the alternating EEG+BOLD training loop against ``ctx``."""
    eeg_loss_fn = make_eeg_loss_fn(
        ctx.simulators.simulator_eeg, ctx.mask_cortical, ctx.idx_min, ctx.idx_max, ctx.cfg.dt,
        settle_ms=ctx.cfg.eeg_settle_ms, stride_ms=ctx.cfg.eeg_stride_ms,
    )
    bold_loss_fn = make_bold_loss_fn(
        ctx.simulators.simulator_bold, ctx.simulators.bold_monitor,
        target_fc_vec=_target_fc_vec(ctx), skip_t=ctx.cfg.bold_skip_trs,
    )
    eeg_update_step, bold_update_step = make_update_steps(eeg_loss_fn, bold_loss_fn, ctx.optimizer)

    return run_alternating_fit(
        ctx.diff_params_init, ctx.static_params, eeg_update_step, bold_update_step,
        ctx.optimizer, ctx.target_psd, ctx.dataset.channel_indices,
        ctx.leadfield, ctx.smoothing_weights, ctx.dipole_labels,
        num_epochs=ctx.cfg.num_epochs, bold_every=ctx.cfg.bold_every,
        print_every=ctx.cfg.print_params_every,
        print_fn=partial(print_learnable_params, learnable_params=ctx.cfg.learnable_params),
    )


def _target_fc_vec(ctx: ExperimentContext):
    from .connectivity import fc_vector
    return fc_vector(ctx.sc.empirical_bold, skip_t=ctx.cfg.bold_skip_trs)


def run(cfg: config.BoldFitConfig, dataset) -> tuple[ExperimentContext, FitResult]:
    """Convenience one-shot: build the context and run the fit."""
    ctx = build_context(cfg, dataset)
    return ctx, fit(ctx)
