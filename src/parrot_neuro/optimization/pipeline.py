"""End-to-end orchestration: subject forward model + connectome -> network ->
alternating EEG+BOLD fit.

The thin entry point the rest of ``parrot_neuro.optimization`` composes into. A driver script
next to this package does roughly::

    from parrot_neuro import Subject
    from parrot_neuro.optimization import config, data, pipeline
    config.apply_jax_env()                      # before any jax import!

    subject = Subject(bids_root, subject_id)
    cfg = config.BoldFitConfig(subject=subject, num_epochs=200)
    dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
    ctx = pipeline.build_context(cfg, dataset)
    result = pipeline.fit(ctx)

Everything downstream of "I have a dataset" — SC/BOLD loading, dipole-label
alignment, network construction, the two simulators, and the training loop —
is fully generic and lives here.

``dataset`` is optional (``build_context(cfg)``, no second arg) when
``cfg.optimize == "bold"`` -- EEG isn't a fit target there, so there's no need
to load it just to run the fit; load it later, whenever you actually want to
look at simulated-vs-real EEG, and pass it straight to the ``viz`` plotting
functions (they don't go through ``ctx``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import numpy as np

from . import config
from .connectivity import StructuralConnectivity, load_structural_connectivity
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
    smoothing_blocks: tuple  # per-block (n_b, n_b) Gaussian smoothing matrices
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
    gamma_idx_min: int
    gamma_idx_max: int
    diff_params_init: object
    static_params: object
    eeg_optimizer: object
    bold_optimizer: object


def build_context(cfg: config.BoldFitConfig, dataset=None) -> ExperimentContext:
    """Load forward model + connectome, align them, and build the network.

    ``dataset`` must already be sliced to the recordings you want the EEG
    loss fit against (e.g. resting state) — see the module docstring.

    ``dataset`` is optional when ``cfg.optimize == "bold"`` -- EEG isn't a fit
    target there, so there's no need to require (or even load) the subject's
    EEG derivatives just to run a BOLD-only fit. It's still required for
    ``optimize in ("eeg", "both")`` (raised below), and can always be loaded
    later, purely for visualization, and passed to the plotting helpers
    directly (they don't go through ``ctx``).
    """
    if dataset is None and cfg.optimize in ("eeg", "both"):
        raise ValueError(
            f"optimize={cfg.optimize!r} needs an EEG dataset (pass one, or set "
            "optimize='bold' if you don't want EEG as a fit target)."
        )

    # --- forward model (leadfield, dipole labels, cortical/subcortical type) ---
    # representative_dipole isn't used by this pipeline — skip computing it
    # (saves a second dense (N_dip, N_dip) array on top of the smoothing matrix).
    leadfield, weights_matrices, _fwd_dipole_labels, orient_atlas, _representative_dipole = (
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

    # dipoles -> optimization-node ids, already indexing sc's (fMRI-aligned) node
    # set; -1 where the dipole's connectome node has no usable BOLD. `valid` drops
    # exactly those dipoles. Same authoritative fmri_keep mask the SC loaders use,
    # so no NaN re-derivation here (and same dipole ordering as the leadfield /
    # smoothing blocks, since all three come from subject.load.dipole_labels).
    dipole_labels = cfg.subject.load.dipole_node_labels(cfg.atlas, float(cfg.spacing), cfg.fmri_task)
    valid = dipole_labels >= 0
    dipole_labels = dipole_labels[valid]

    # Volume-weighted Gaussian source-smoothing is block-diagonal (a dipole only
    # smooths within its own surface/volumetric block), so we keep it as a list
    # of per-block matrices rather than assembling the dense (N_dip, N_dip)
    # block-diagonal matrix -- at tens of thousands of dipoles that dense matrix
    # is multi-GB and almost all zeros. project_to_scalp applies it as a
    # block-diagonal matmul (forward.block_diag_matmul).
    #
    # `valid` drops dipoles whose region has no BOLD coverage; it's in global
    # (block-concatenated) dipole order, so it splits into one contiguous slice
    # per block. Subselect each block's surviving rows AND columns -- keeping
    # every block square and the whole operator block-diagonal in the reduced
    # dipole space. This reproduces the old dense `[valid][:, valid]` exactly
    # (deliberately no re-normalization of the surviving rows, as before).
    #
    # float32: these blocks (and the leadfield) are fixed geometric projections,
    # not part of the integrated ODE state -- they don't need x64 precision, and
    # the float64 versions are large enough to make the host->GPU transfer fail
    # to pin ("could not allocate pinned host memory" / CUDA_ERROR_INVALID_VALUE).
    # Downstream matmuls against the (float64) dynamics state still promote to f64.
    smoothing_blocks = []
    start = 0
    for block in weights_matrices:
        size = block.shape[0]
        vb = valid[start:start + size]
        start += size
        if vb.any():  # a block wholly inside a dropped region contributes nothing
            sub = np.asarray(block, dtype=np.float32)[vb][:, vb]
            smoothing_blocks.append(jnp.asarray(sub))
    assert start == len(valid), "block sizes must tile the full dipole axis"
    smoothing_blocks = tuple(smoothing_blocks)

    leadfield = np.asarray(leadfield, dtype=np.float32)[:, valid]

    mask_cortical = np.where(np.isin(orient_atlas, ["N", "G", "P"]), 1.0, 0.0)
    mask_cortical = mask_cortical[sc.keep]  # full connectome axis -> fMRI-aligned node set

    dipole_labels = jnp.array(dipole_labels)
    leadfield = jnp.array(leadfield)

    # --- network + two simulators (short EEG horizon, long BOLD horizon) ---
    network, solver, _brain_model = build_network(
        mask_cortical, sc.weights, sc.delays, sc.num_nodes,
        learnable_params=cfg.learnable_params, base_sigma=cfg.base_sigma, noise_seed=cfg.noise_seed,
        solver_block_size=cfg.solver_block_size,
    )
    simulators = build_simulators(
        network, solver, cfg.t0, cfg.dt, cfg.t1_eeg, cfg.t1_bold,
        cfg.tr_ms, cfg.bold_downsample_ms, t1_warmup=cfg.t1_warmup,
    )

    diff_params_init, static_params = learnable_partition(simulators.params, cfg.learnable_params)

    # --- EEG loss target (mean PSD across this subject's chunks) ---
    # freqs/idx_min/idx_max only depend on cfg (fs/chunk_length/fmin/fmax), so
    # they're always computed -- handy if EEG gets loaded later just to look.
    target_psd = compute_target_psd(dataset) if dataset is not None else None
    freqs = np.fft.rfftfreq(cfg.chunk_length, d=1.0 / cfg.fs)
    idx_min = int(np.searchsorted(freqs, cfg.fmin))
    idx_max = int(np.searchsorted(freqs, cfg.fmax))
    # Same freqs grid (independent of the band) also gives the optional gamma
    # term's band -- see train.make_eeg_loss_fn's gamma_weight.
    gamma_idx_min = int(np.searchsorted(freqs, cfg.gamma_fmin))
    gamma_idx_max = int(np.searchsorted(freqs, cfg.gamma_fmax))

    # Separate optimizer per loss (see train.run_alternating_fit) -- also lets
    # EEG and BOLD use different learning rates.
    eeg_optimizer = make_optimizer(cfg.learning_rate, cfg.grad_clip_norm)
    bold_optimizer = make_optimizer(
        cfg.learning_rate_bold if cfg.learning_rate_bold is not None else cfg.learning_rate,
        cfg.grad_clip_norm,
    )

    return ExperimentContext(
        cfg=cfg, mask_cortical=mask_cortical, leadfield=leadfield,
        smoothing_blocks=smoothing_blocks, dipole_labels=dipole_labels, sc=sc,
        network=network, solver=solver, simulators=simulators, dataset=dataset,
        target_psd=target_psd, freqs=freqs, idx_min=idx_min, idx_max=idx_max,
        gamma_idx_min=gamma_idx_min, gamma_idx_max=gamma_idx_max,
        diff_params_init=diff_params_init, static_params=static_params,
        eeg_optimizer=eeg_optimizer, bold_optimizer=bold_optimizer,
    )


def fit(ctx: ExperimentContext) -> FitResult:
    """Run the alternating EEG+BOLD training loop against ``ctx``.

    ``ctx.dataset`` (hence ``eeg_loss_fn``) may be ``None`` -- build_context
    already enforced that this is only possible when ``optimize == "bold"``,
    so ``run_alternating_fit`` simply never calls it in that case (see its
    ``optimize`` gating); it's built here only when there's actually a target.
    """
    eeg_loss_fn = None
    if ctx.dataset is not None:
        eeg_loss_fn = make_eeg_loss_fn(
            ctx.simulators.simulator_eeg, ctx.mask_cortical, ctx.idx_min, ctx.idx_max, ctx.cfg.dt,
            settle_ms=ctx.cfg.eeg_settle_ms, stride_ms=ctx.cfg.eeg_stride_ms,
            gamma_weight=ctx.cfg.gamma_weight, gamma_idx_min=ctx.gamma_idx_min, gamma_idx_max=ctx.gamma_idx_max,
        )

    # fc and dfc targets are always built -- the combined BOLD loss always
    # computes both terms (from one simulated trajectory), weighted by
    # cfg.bold_fc_weight/bold_dfc_weight (either at 0 recovers a single-mode fit).
    centers = jnp.linspace(-1.0, 1.0, ctx.cfg.dfc_n_bins)
    target_psd_band = _target_bold_psd_band(ctx) if ctx.cfg.bold_psd_weight > 0 else None
    bold_loss_fn = make_bold_loss_fn(
        ctx.simulators.simulator_bold, ctx.simulators.bold_monitor,
        target_fc_vec=_target_fc_vec(ctx), target_dfc_hist=_target_dfc_hist(ctx, centers),
        skip_t=ctx.cfg.bold_skip_trs, tr_ms=ctx.cfg.tr_ms,
        dfc_window_trs=ctx.cfg.dfc_window_trs, dfc_step_trs=ctx.cfg.dfc_step_trs,
        dfc_centers=centers, dfc_k_min=ctx.cfg.dfc_kmin, dfc_sigma=ctx.cfg.dfc_sigma,
        fc_weight=ctx.cfg.bold_fc_weight, dfc_weight=ctx.cfg.bold_dfc_weight,
        bandpass_low=ctx.cfg.bold_bandpass_low, bandpass_high=ctx.cfg.bold_bandpass_high,
        bandpass_order=ctx.cfg.bold_bandpass_order,
        psd_weight=ctx.cfg.bold_psd_weight, target_psd_band=target_psd_band,
        psd_nperseg=ctx.cfg.bold_psd_nperseg_trs, psd_noverlap=ctx.cfg.bold_psd_noverlap_trs,
    )

    eeg_update_step, bold_update_step = make_update_steps(
        eeg_loss_fn, bold_loss_fn, ctx.eeg_optimizer, ctx.bold_optimizer
    )

    channel_indices = ctx.dataset.channel_indices if ctx.dataset is not None else None
    return run_alternating_fit(
        ctx.diff_params_init, ctx.static_params, eeg_update_step, bold_update_step,
        ctx.eeg_optimizer, ctx.bold_optimizer, ctx.target_psd, channel_indices,
        ctx.leadfield, ctx.smoothing_blocks, ctx.dipole_labels,
        num_epochs=ctx.cfg.num_epochs, bold_every=ctx.cfg.bold_every,
        print_every=ctx.cfg.print_params_every,
        print_fn=partial(print_learnable_params, learnable_params=ctx.cfg.learnable_params),
        optimize=ctx.cfg.optimize,
        early_stop_window=ctx.cfg.early_stop_window,
        early_stop_patience=ctx.cfg.early_stop_patience,
        early_stop_min_delta=ctx.cfg.early_stop_min_delta,
    )


def _target_fc_vec(ctx: ExperimentContext):
    from .connectivity import fc_vector
    return fc_vector(ctx.sc.empirical_bold, skip_t=ctx.cfg.bold_skip_trs)


def _target_dfc_hist(ctx: ExperimentContext, centers):
    from .connectivity import dfc_histogram
    return dfc_histogram(
        ctx.sc.empirical_bold, ctx.cfg.dfc_window_trs, ctx.cfg.dfc_step_trs, centers,
        skip_t=ctx.cfg.bold_skip_trs, k_min=ctx.cfg.dfc_kmin, sigma=ctx.cfg.dfc_sigma,
    )


def _target_bold_psd_band(ctx: ExperimentContext):
    from .connectivity import bold_psd_band
    return bold_psd_band(
        ctx.sc.empirical_bold, ctx.cfg.tr_ms, ctx.cfg.bold_psd_nperseg_trs, ctx.cfg.bold_psd_noverlap_trs,
        skip_t=ctx.cfg.bold_skip_trs, low=ctx.cfg.bold_bandpass_low, high=ctx.cfg.bold_bandpass_high,
    )


def run(cfg: config.BoldFitConfig, dataset=None) -> tuple[ExperimentContext, FitResult]:
    """Convenience one-shot: build the context and run the fit."""
    ctx = build_context(cfg, dataset)
    return ctx, fit(ctx)
