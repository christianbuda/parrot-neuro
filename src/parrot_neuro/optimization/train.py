"""Alternating EEG(PSD) + BOLD(FC) training loop for the JR/WC heterogeneous model.

Two simulators share one parameter pytree: a short one (``t1_eeg``, ~seconds)
for the EEG spectral loss, and a long one (``t1_bold``, ~minutes) for the BOLD
functional-connectivity loss. Every ``bold_every`` epochs both losses take a
gradient step against the *same* differentiable parameters; in between, only
the (much cheaper) EEG step runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tvboptim.experimental.network_dynamics import prepare, solve
from tvboptim.observations.tvb_monitors import HRFBold, SubSampling, streaming_hrf_bold

from .config import BOLD_BANDPASS_HIGH, BOLD_BANDPASS_LOW, BOLD_BANDPASS_ORDER, DEFAULT_LEARNABLE_PARAMS, LearnableParam
from .connectivity import bold_psd_band, dfc_histogram, fc_vector, filter_sim_bold, wasserstein_1d_from_hist
from .forward import project_to_scalp
from .signal import compute_psd, smooth_ts


def _param_node(params, lp: LearnableParam):
    """Navigate to a LearnableParam's leaf within the params/diff_params tree."""
    if lp.location == "dynamics":
        return params["dynamics"][lp.name]
    if lp.location == "coupling":
        return params["coupling"]["delayed"][lp.name]
    raise ValueError(f"Unknown location {lp.location!r} for learnable param {lp.name!r}")


@dataclass
class Simulators:
    """Two prepared simulators (short EEG horizon, long BOLD horizon)
    sharing one parameter pytree, plus the BOLD monitor built on the long
    warm-up's history.

    ``simulator_bold``'s HRF convolution is streamed (folded into
    ``prepare()``'s ``reduce=`` block scan -- see ``build_simulators``), so
    calling it already returns the final BOLD buffer, not a raw solution to
    run ``bold_monitor`` on afterward. ``bold_monitor`` is kept on this
    dataclass only because ``_process_history``'s warm-start buffer and
    ``streaming_hrf_bold``'s kernel/period/voi config live on it -- it is
    never called directly as a function anymore."""

    simulator_eeg: Callable
    simulator_bold: Callable
    bold_monitor: HRFBold
    params: object  # Bunch — feed to learnable_partition() / eqx.combine


def build_simulators(
    network,
    solver,
    t0: float,
    dt: float,
    t1_eeg: float,
    t1_bold: float,
    tr_ms: float,
    bold_downsample_ms: float,
    bold_voi: int = 8,
    t1_warmup: float | None = None,
) -> Simulators:
    """Warm up ``network`` at both horizons and prepare pure solve functions.

    Order matters: the EEG (short) warm-up seeds the delay history first,
    then the BOLD warm-up overwrites it. Both ``prepare()`` calls expose the
    same differentiable leaves (only static per-horizon delay-buffer metadata
    differs, baked into each closure), so the params from the *second* call
    are the ones used for both simulators.

    ``t1_warmup`` (default ``None`` -> falls back to ``t1_bold``, the old
    behaviour) is the duration of the *BOLD warm-up solve only* -- a one-time,
    forward-only simulation whose sole purpose is to seed ``network``'s
    initial state (the warm-up's *last* timestep -- see
    ``Network.initial_state``) and a short delay-coupling history buffer,
    plus give ``HRFBold`` a history tail to convolve against. None of those
    three consumers reads more than a short recent window regardless of how
    long the warm-up ran (delay buffers need ``max_delay`` seconds -- tens of
    ms for physiological conduction delays; ``HRFBold._process_history`` slices
    exactly the kernel's ``duration`` -- 20s by default -- off the *end*).
    Running the warm-up for the *full* ``t1_bold`` (which can be minutes, by
    design, to give the FC/dFC loss a long BOLD signal) computes and holds a
    proportionally huge trajectory just to throw away all but its tail --
    the dominant GPU-memory cost of this function for a long ``t1_bold``.
    ``t1_warmup`` decouples the two: pass something comfortably longer than
    both the settling time of your dynamics and the HRF kernel duration (e.g.
    30s), independent of how long ``t1_bold`` itself is. This does NOT change
    the length of BOLD signal available to the loss -- ``simulator_bold``
    (the one actually called during training) still integrates the full
    ``t1_bold``; only the throwaway pre-roll gets shorter. It does change the
    *exact* initial state training starts from (a different, still-settled,
    point along an equally-valid stochastic trajectory -- not a less-settled
    one), which is why this is opt-in via ``None`` rather than always-on.
    """
    print(f"Preparing simulators: EEG t1={t1_eeg:.1f}s, BOLD t1={t1_bold:.1f}s")
    result_eeg = solve(network, solver, t0=t0, t1=t1_eeg, dt=dt)
    network.update_history(result_eeg)
    simulator_eeg, _ = prepare(network, solver, t0=t0, t1=t1_eeg, dt=dt)

    warmup_t1 = t1_bold if t1_warmup is None else min(t1_warmup, t1_bold)
    if warmup_t1 != t1_bold:
        print(f"  BOLD warm-up solve: t1={warmup_t1:.1f}s (t1_bold={t1_bold:.1f}s used unchanged "
              "for the actual training simulator)")
    result_bold = solve(network, solver, t0=t0, t1=warmup_t1, dt=dt)
    network.update_history(result_bold)

    # SubSampling (pick every downsample_period-th raw sample), NOT HRFBold's
    # own default (TemporalAverage, mean over each window): streaming_hrf_bold
    # below requires a uniform-integer-stride downsampler -- its per-block
    # update() always does a hard-coded "take every Nth sample" slice
    # regardless of what monitor.downsample actually is, so passing anything
    # else here would silently desync the streaming path from what this
    # object's own (now never-called-directly) __call__ would have computed.
    # A deliberate, small numerical difference from the old TemporalAverage
    # default -- see build_simulators' docstring.
    bold_monitor = HRFBold(
        history=result_bold,
        period=tr_ms,
        downsample_period=bold_downsample_ms,
        voi=bold_voi,
        downsample=SubSampling(voi=bold_voi, period=bold_downsample_ms),
    )
    # reduce=streaming_hrf_bold(...) folds the HRF convolution into the same
    # block scan solver_block_size checkpoints, block-by-block, instead of
    # materializing the full raw trajectory and convolving it post-hoc (the
    # dominant GPU-memory cost of this whole pipeline for a long t1_bold --
    # see config.BoldFitConfig.solver_block_size). simulator_bold(combined)
    # therefore returns the final [n_bold, n_voi, n_nodes] BOLD buffer
    # directly, not a raw solution -- there is no post-hoc bold_monitor(sol)
    # call anywhere anymore (make_bold_loss_fn, diagnostics.py). Requires
    # solver.block_size to be an exact multiple of the BOLD period in raw
    # steps (tr_ms/dt); see streaming_hrf_bold's own docstring.
    simulator_bold, params = prepare(
        network, solver, t0=t0, t1=t1_bold, dt=dt,
        reduce=streaming_hrf_bold(bold_monitor, dt),
    )

    return Simulators(simulator_eeg, simulator_bold, bold_monitor, params)


def learnable_partition(params, learnable_params: tuple[LearnableParam, ...] = DEFAULT_LEARNABLE_PARAMS):
    """Split ``params`` into (diff_params, static_params), learnable-only.

    ``eqx.is_inexact_array`` — the obvious filter to reach for — can't tell
    "a value wrapped in ``SigmoidBoundedParameter``" apart from "any float
    array anywhere in the simulator config": the structural connectivity
    graph, the per-node initial state, the noise sigma matrix, and the
    cortex/subcortex masks are all plain arrays too, and silently end up
    learnable (and get real Adam updates) if you filter on dtype alone.

    Instead, build a boolean mask with ``params``' own tree structure, True
    only at the leaves named in ``learnable_params`` (must be the same list
    passed to ``network.build_network`` — those are the only leaves actually
    wrapped as ``SigmoidBoundedParameter``), and let ``eqx.partition`` build
    both (structurally symmetric) halves from it. Building diff/static by
    hand instead breaks ``eqx.combine``'s depth matching (a bare ``None``
    next to a real ``Parameter`` object at the same tree position raises
    "Custom node type mismatch").
    """
    mask = jax.tree_util.tree_map(lambda _: False, params)
    for lp in learnable_params:
        node = _param_node(params, lp)
        if lp.location == "dynamics":
            mask["dynamics"][lp.name] = jax.tree_util.tree_map(lambda _: True, node)
        else:
            mask["coupling"]["delayed"][lp.name] = jax.tree_util.tree_map(lambda _: True, node)
    return eqx.partition(params, mask)


def compute_target_psd(dataset):
    """Mean PSD across a dataset's chunks — the EEG loss's fit target."""
    return jnp.array(np.stack(list(map(compute_psd, dataset._chunks))).mean(axis=0))


def _eeg_psd_loss(source_ys, mask_col, idx_min, idx_max, target_psd, channel_indices,
                   leadfield, smoothing_blocks, dipole_labels, settle, stride,
                   gamma_weight, gamma_idx_min, gamma_idx_max, eps):
    """Linear-PSD MSE between ``source_ys[settle::stride, 1/2]`` (JR pyramidal
    y1 - y2, zeroed on subcortical nodes) projected to scalp and
    ``target_psd``. The numeric core shared by ``make_eeg_loss_fn`` and
    ``make_joint_loss_fn`` -- both feed it ``simulator_eeg``'s own
    short-horizon ``.ys``, so both compute the *exact* same loss.
    """
    source_activity = (
        source_ys[settle::stride, 1].T - source_ys[settle::stride, 2].T
    ) * mask_col

    simulated_eeg = project_to_scalp(
        source_activity, channel_indices, leadfield, smoothing_blocks, dipole_labels
    )

    sim_psd = smooth_ts(compute_psd(simulated_eeg))
    target_psd = smooth_ts(target_psd)

    norm_sim = sim_psd / (jnp.sum(sim_psd[:, idx_min:idx_max], keepdims=True) + eps)
    norm_target = target_psd / (jnp.sum(target_psd[:, idx_min:idx_max], keepdims=True) + eps)

    loss = jnp.mean((norm_sim[:, idx_min:idx_max] - norm_target[:, idx_min:idx_max]) ** 2)

    if gamma_weight > 0:
        log_sim = jnp.log(sim_psd[:, gamma_idx_min:gamma_idx_max] + eps)
        log_target = jnp.log(target_psd[:, gamma_idx_min:gamma_idx_max] + eps)
        loss = loss + gamma_weight * jnp.mean((log_sim - log_target) ** 2)

    return loss


def make_eeg_loss_fn(simulator_eeg, mask_cortical, idx_min, idx_max, dt,
                      settle_ms=500.0, stride_ms=4.0,
                      gamma_weight=0.0, gamma_idx_min=None, gamma_idx_max=None, eps=1e-8):
    """Linear-PSD MSE loss, closing over everything that never changes per-step
    (the mask, frequency-bin window, and timing — mirrors how the simulator
    itself is a closure): only ``(diff, static, target_psd, channel_indices,
    leadfield, smoothing_blocks, dipole_labels)`` vary call to call.

    ``gamma_weight`` > 0 adds an optional second term: MSE between log(PSD)
    (not the normalized-linear comparison the main band uses) over
    ``[gamma_idx_min, gamma_idx_max)`` -- gamma-band power is orders of
    magnitude smaller than the main band's, so a linear/normalized comparison
    would be swamped by it; log-space keeps both bands' errors on comparable
    footing. 0 (default) keeps the old main-band-only behaviour;
    ``gamma_idx_min``/``gamma_idx_max`` must be given when ``gamma_weight`` > 0.
    """
    settle = int(settle_ms / dt)
    stride = int(stride_ms / dt)
    mask_col = jnp.atleast_2d(jnp.asarray(mask_cortical)).T
    if gamma_weight > 0 and (gamma_idx_min is None or gamma_idx_max is None):
        raise ValueError("gamma_idx_min/gamma_idx_max must be given when gamma_weight > 0")

    @eqx.filter_jit
    def eeg_loss_fn(current_diff, current_static, target_psd, channel_indices,
                     leadfield, smoothing_blocks, dipole_labels):
        combined = eqx.combine(current_diff, current_static)
        sim_result = simulator_eeg(combined)
        return _eeg_psd_loss(
            sim_result.ys, mask_col, idx_min, idx_max, target_psd, channel_indices,
            leadfield, smoothing_blocks, dipole_labels, settle, stride,
            gamma_weight, gamma_idx_min, gamma_idx_max, eps,
        )

    return eeg_loss_fn


def _bold_fc_dfc_loss(Xs, target_fc_vec, target_dfc_hist, tr_ms,
                       dfc_window_trs, dfc_step_trs, dfc_centers, dfc_k_min, dfc_sigma,
                       fc_weight, dfc_weight,
                       bandpass_low, bandpass_high, bandpass_order, eps,
                       psd_weight, target_psd_band, psd_nperseg, psd_noverlap):
    """``fc_weight * FC-vector MSE + dfc_weight * dFC/FCD Wasserstein distance``
    (plus an optional ``psd_weight`` spectral-shape term) on an already
    ``skip_t``-trimmed BOLD signal ``Xs`` [n_bold, n_nodes], from ONE filtered
    trajectory (``Xs_filt``). The numeric core shared by ``make_bold_loss_fn``
    (which feeds it its own ``simulator_bold`` call's output) and
    ``make_joint_loss_fn`` (which feeds it the BOLD half of the joint
    simulator's output instead) so both compute the exact same BOLD loss.
    """
    Xs_filt = filter_sim_bold(Xs, tr_ms, low=bandpass_low, high=bandpass_high, order=bandpass_order)

    fc_sim = fc_vector(Xs_filt, skip_t=0, eps=eps)
    fc_loss = jnp.mean((fc_sim - target_fc_vec) ** 2)

    dfc_hist_sim = dfc_histogram(Xs_filt, dfc_window_trs, dfc_step_trs, dfc_centers,
                                  skip_t=0, k_min=dfc_k_min, sigma=dfc_sigma, eps=eps)
    dfc_loss = wasserstein_1d_from_hist(dfc_hist_sim, target_dfc_hist)

    loss = fc_weight * fc_loss + dfc_weight * dfc_loss

    if psd_weight > 0:
        sim_psd_band = bold_psd_band(Xs_filt, tr_ms, psd_nperseg, psd_noverlap, skip_t=0,
                                      low=bandpass_low, high=bandpass_high, eps=eps)
        loss = loss + psd_weight * jnp.mean((sim_psd_band - target_psd_band) ** 2)

    return loss


def make_bold_loss_fn(simulator_bold, target_fc_vec, target_dfc_hist, skip_t, tr_ms,
                       dfc_window_trs, dfc_step_trs, dfc_centers, dfc_k_min=1, dfc_sigma=0.05,
                       fc_weight=0.5, dfc_weight=0.5,
                       bandpass_low=BOLD_BANDPASS_LOW, bandpass_high=BOLD_BANDPASS_HIGH,
                       bandpass_order=BOLD_BANDPASS_ORDER, eps=1e-8, bad_loss=1e3,
                       psd_weight=0.0, target_psd_band=None, psd_nperseg=32, psd_noverlap=16):
    """Combined BOLD loss: ``fc_weight * FC-vector MSE + dfc_weight * dFC/FCD
    Wasserstein distance``, plus an optional ``psd_weight * spectral-shape MSE``
    term -- falls back to a large constant loss (rather than NaN) if the
    simulation blew up, so a single unlucky epoch doesn't poison the gradient
    with NaNs the optimizer can never recover from.

    Computing ``simulator_bold(combined)`` is the single most expensive part
    of this whole pipeline, so all three terms are derived from ONE forward
    pass / one filtered trajectory (``Xs_filt``) rather than three separate
    closures each re-running the simulator -- ``fc_weight``/``dfc_weight`` at
    0 recovers the old single-mode ("fc"-only or "dfc"-only) behaviour without
    any wasted compute on the zeroed-out term (its gradient is simply 0, XLA
    still traces it, but this is cheap relative to the simulator itself).

    ``simulator_bold`` (as built by ``build_simulators``) already returns the
    HRF-convolved, TR-downsampled BOLD buffer directly -- the streaming
    ``reduce=streaming_hrf_bold(...)`` baked into its ``prepare()`` call folds
    the HRF convolution block-by-block during integration instead of
    materializing the full raw trajectory and convolving it post-hoc. There is
    no separate ``bold_monitor(sol)`` call here (unlike the EEG side, which
    still needs its own post-hoc scalp projection).

    Simulated BOLD is bandpassed (``connectivity.filter_sim_bold``, still
    differentiable) to the same band the empirical BOLD (hence
    ``target_fc_vec``/``target_dfc_hist``/``target_psd_band``) was already
    preprocessed with -- otherwise the comparisons would be between
    differently-filtered signals.

    ``psd_weight`` > 0 adds a Welch-PSD shape term (``connectivity.bold_psd_band``,
    restricted to ``[bandpass_low, bandpass_high]``) that ``fc_vector``'s
    time-averaged correlation has no sensitivity to at all -- 0 (default)
    keeps the old FC/dFC-only behaviour; ``target_psd_band`` must be given
    when ``psd_weight`` > 0.
    """
    if psd_weight > 0 and target_psd_band is None:
        raise ValueError("target_psd_band must be given when psd_weight > 0")

    @eqx.filter_jit
    def bold_loss_fn(current_diff, current_static):
        combined = eqx.combine(current_diff, current_static)
        bold_buffer = simulator_bold(combined)  # [n_bold, n_voi, n_nodes], already HRF-convolved
        Xs = bold_buffer[:, 0, :][skip_t:, :]
        ok = jnp.all(jnp.isfinite(Xs))

        def good(_):
            return _bold_fc_dfc_loss(
                Xs, target_fc_vec, target_dfc_hist, tr_ms,
                dfc_window_trs, dfc_step_trs, dfc_centers, dfc_k_min, dfc_sigma,
                fc_weight, dfc_weight, bandpass_low, bandpass_high, bandpass_order, eps,
                psd_weight, target_psd_band, psd_nperseg, psd_noverlap,
            )

        def bad(_):
            return jnp.array(bad_loss, dtype=jnp.float64)

        return jax.lax.cond(ok, good, bad, operand=None)

    return bold_loss_fn


def make_optimizer(learning_rate=1e-3, grad_clip_norm=1.0):
    return optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(learning_rate))


def make_update_steps(eeg_loss_fn, bold_loss_fn, eeg_optimizer, bold_optimizer):
    """Gradient-step closures over the two loss functions, each with its own
    optimizer -- separate transforms (not just separate states), so EEG and
    BOLD can run at different learning rates."""

    @eqx.filter_jit
    def eeg_update_step(current_diff, current_static, current_opt_state,
                         target_psd, channel_indices, leadfield, smoothing_blocks, dipole_labels):
        loss, grads = jax.value_and_grad(eeg_loss_fn, argnums=0)(
            current_diff, current_static, target_psd, channel_indices,
            leadfield, smoothing_blocks, dipole_labels,
        )
        updates, new_opt_state = eeg_optimizer.update(grads, current_opt_state, current_diff)
        new_diff = optax.apply_updates(current_diff, updates)
        return new_diff, new_opt_state, loss

    @eqx.filter_jit
    def bold_update_step(current_diff, current_static, current_opt_state):
        loss, grads = jax.value_and_grad(bold_loss_fn, argnums=0)(current_diff, current_static)
        updates, new_opt_state = bold_optimizer.update(grads, current_opt_state, current_diff)
        new_diff = optax.apply_updates(current_diff, updates)
        return new_diff, new_opt_state, loss

    return eeg_update_step, bold_update_step


def make_joint_loss_fn(simulator_eeg, simulator_bold, mask_cortical, idx_min, idx_max, dt,
                        target_fc_vec, target_dfc_hist, skip_t, tr_ms,
                        dfc_window_trs, dfc_step_trs, dfc_centers, dfc_k_min, dfc_sigma,
                        fc_weight, dfc_weight,
                        bandpass_low=BOLD_BANDPASS_LOW, bandpass_high=BOLD_BANDPASS_HIGH,
                        bandpass_order=BOLD_BANDPASS_ORDER, eps=1e-8, bad_loss=1e3,
                        psd_weight=0.0, target_psd_band=None, psd_nperseg=32, psd_noverlap=16,
                        settle_ms=500.0, stride_ms=4.0,
                        gamma_weight=0.0, gamma_idx_min=None, gamma_idx_max=None,
                        joint_eeg_weight=1.0, joint_bold_weight=1.0):
    """Single combined EEG-PSD + BOLD-FC/dFC loss over the SAME two simulators
    ``make_eeg_loss_fn``/``make_bold_loss_fn`` already use -- the "joint"
    schedule's loss, as opposed to the alternating/phased schedules' two
    separate per-loss *steps* (this is one step, over both losses).

    An earlier version tried to get this from ONE simulator call (a single
    long ``t1_bold``-horizon run whose ``reduce=`` also captured a raw EEG
    window near its start, avoiding a second simulator entirely). That
    blows up backward-pass memory: the block-checkpointed scan
    (``config.BoldFitConfig.solver_block_size``) only saves O(n_steps/K + K)
    memory when its carry is state-sized (see
    ``network.build_network``'s docstring) -- adding a several-hundred-MB
    raw-window buffer to that carry makes JAX retain ONE COPY OF IT PER
    BLOCK BOUNDARY for the backward pass (~500 blocks at the production
    ``t1_bold``/``solver_block_size``), turning a few-hundred-MB buffer into
    tens-to-100+GB of added memory -- observed as the "joint" schedule
    becoming dramatically (not just ~2x) slower than alternating. Calling
    ``simulator_eeg`` and ``simulator_bold`` as two ordinary, independent
    calls (exactly as ``make_eeg_loss_fn``/``make_bold_loss_fn`` already do)
    sidesteps this entirely -- neither one's reduce/carry is touched, so
    both keep their existing, already-proven memory profile. The only
    difference from the alternating schedule is what happens with their
    outputs: ONE combined loss/gradient step instead of two separate ones.

    The two losses are summed as ``joint_eeg_weight * eeg_loss +
    joint_bold_weight * bold_loss`` -- plain scalar weights, not
    auto-balanced, because the two losses live on very different scales
    (EEG's normalized-linear PSD MSE ~1e-6, BOLD's weighted FC+dFC ~1e-1);
    pick weights that roughly counter that gap.

    Returns ``(loss, (eeg_loss, bold_loss))`` (``has_aux``-style) so
    ``run_joint_fit`` can log both raw, unweighted components every epoch
    into the same ``loss_history_eeg``/``loss_history_bold`` fields the
    other two schedules populate -- one combined-loss printout
    (``train.relative_final_loss``) works the same across all three.
    """
    settle = int(settle_ms / dt)
    stride = int(stride_ms / dt)
    mask_col = jnp.atleast_2d(jnp.asarray(mask_cortical)).T
    if gamma_weight > 0 and (gamma_idx_min is None or gamma_idx_max is None):
        raise ValueError("gamma_idx_min/gamma_idx_max must be given when gamma_weight > 0")
    if psd_weight > 0 and target_psd_band is None:
        raise ValueError("target_psd_band must be given when psd_weight > 0")

    @eqx.filter_jit
    def joint_loss_fn(current_diff, current_static, target_psd, channel_indices,
                       leadfield, smoothing_blocks, dipole_labels):
        combined = eqx.combine(current_diff, current_static)

        sim_result = simulator_eeg(combined)
        eeg_loss = _eeg_psd_loss(
            sim_result.ys, mask_col, idx_min, idx_max, target_psd, channel_indices,
            leadfield, smoothing_blocks, dipole_labels, settle, stride,
            gamma_weight, gamma_idx_min, gamma_idx_max, eps,
        )

        bold_buffer = simulator_bold(combined)
        Xs = bold_buffer[:, 0, :][skip_t:, :]
        ok = jnp.all(jnp.isfinite(Xs))

        def good(_):
            return _bold_fc_dfc_loss(
                Xs, target_fc_vec, target_dfc_hist, tr_ms,
                dfc_window_trs, dfc_step_trs, dfc_centers, dfc_k_min, dfc_sigma,
                fc_weight, dfc_weight, bandpass_low, bandpass_high, bandpass_order, eps,
                psd_weight, target_psd_band, psd_nperseg, psd_noverlap,
            )

        def bad(_):
            return jnp.array(bad_loss, dtype=jnp.float64)

        bold_loss = jax.lax.cond(ok, good, bad, operand=None)

        loss = joint_eeg_weight * eeg_loss + joint_bold_weight * bold_loss
        return loss, (eeg_loss, bold_loss)

    return joint_loss_fn


def make_joint_update_step(joint_loss_fn, joint_optimizer):
    """Single gradient-step closure for the "joint" schedule -- one optimizer,
    one combined loss, mirroring ``make_update_steps``'s two per-loss steps
    but fused into one."""

    @eqx.filter_jit
    def joint_update_step(current_diff, current_static, current_opt_state,
                           target_psd, channel_indices, leadfield, smoothing_blocks, dipole_labels):
        (loss, (eeg_loss, bold_loss)), grads = jax.value_and_grad(
            joint_loss_fn, argnums=0, has_aux=True
        )(
            current_diff, current_static, target_psd, channel_indices,
            leadfield, smoothing_blocks, dipole_labels,
        )
        updates, new_opt_state = joint_optimizer.update(grads, current_opt_state, current_diff)
        new_diff = optax.apply_updates(current_diff, updates)
        return new_diff, new_opt_state, loss, eeg_loss, bold_loss

    return joint_update_step


def print_learnable_params(diff_params, learnable_params: tuple[LearnableParam, ...] = DEFAULT_LEARNABLE_PARAMS):
    """Print the current mean±std of every learnable parameter.

    Driven by the same ``learnable_params`` list passed to
    ``network.build_network``/``learnable_partition`` — printing anything
    not in that list would silently rely on it also being learnable, which
    is exactly the mismatch that caused past bugs.
    """
    parts = []
    for lp in learnable_params:
        node = _param_node(diff_params, lp)
        val = node.forward_transform(node.value)
        parts.append(f"{lp.name}={float(jnp.mean(val)):.4f}±{float(jnp.std(val)):.4f}")
    print("  params | " + "  ".join(parts))


def extract_learnable_values(
    diff_params, learnable_params: tuple[LearnableParam, ...] = DEFAULT_LEARNABLE_PARAMS
) -> dict:
    """Final learnable-parameter values in their natural post-sigmoid units, as numpy arrays."""
    out = {}
    for lp in learnable_params:
        node = _param_node(diff_params, lp)
        out[lp.name] = np.asarray(node.forward_transform(node.value))
    return out


@dataclass
class FitResult:
    diff_params: object
    static_params: object
    loss_history_eeg: list = field(default_factory=list)
    loss_history_bold: list = field(default_factory=list)


def relative_final_loss(history):
    """``history[-1] / history[0]`` -- a scale-free "fraction of the early
    loss remaining" (~1.0 at the start, <1 as it improves). ``None`` if
    ``history`` is empty or its first value is falsy (no valid baseline to
    divide by, e.g. a loss that started at exactly 0).

    Exists so EEG (normalized-linear PSD MSE, typically ~1e-6) and BOLD
    (weighted FC+dFC, typically ~1e-1) losses -- which live on completely
    different absolute scales -- can be combined into one meaningful summary
    number instead of one silently swamping the other in a raw sum. Used by
    both ``examples/eeg_bold_fit_cli.py`` (single-subject "combined loss"
    printout) and ``examples/eeg_bold_fit_sweep.py`` (the wandb sweep's
    ``aggregate/combined_loss`` objective) so the two report the exact same
    number for the exact same fit.
    """
    if not history or not history[0]:
        return None
    return history[-1] / history[0]


def is_loss_stalled(history, window, patience, min_delta):
    """True if ``history``'s relative linear trend has stayed >= ``-min_delta``
    (i.e. not meaningfully decreasing) over each of the last ``patience``
    overlapping windows of length ``window``.

    Fits a straight line to each window and normalizes its slope by the
    window's mean (a scale-free "fraction of the loss's own magnitude lost
    per step", comparable across losses/epochs regardless of absolute loss
    scale). Both a flat trend and an increasing one (e.g. the BOLD loss's
    ``bad_loss`` divergence sentinel) count as stalled; any window where the
    loss is still dropping faster than ``min_delta`` resets the check to
    "not stalled" -- ``patience`` consecutive stalled windows are required
    before this reports ``True``, so a single noisy epoch can't trigger it.
    """
    if len(history) < window + patience - 1:
        return False
    x = np.arange(window)
    for k in range(patience):
        end = len(history) - k
        seg = np.asarray(history[end - window:end])
        slope = np.polyfit(x, seg, 1)[0]
        rel_slope = slope / (abs(seg.mean()) + 1e-8)
        if rel_slope <= -min_delta:
            return False
    return True


def run_alternating_fit(
    diff_params_init,
    static_params,
    eeg_update_step,
    bold_update_step,
    eeg_optimizer,
    bold_optimizer,
    target_psd,
    channel_indices,
    leadfield,
    smoothing_blocks,
    dipole_labels,
    num_epochs=200,
    bold_every=1,
    print_every=10,
    print_fn=print_learnable_params,
    optimize="both",
    early_stop_window=20,
    early_stop_patience=None,
    early_stop_min_delta=1e-3,
    on_epoch=None,
) -> FitResult:
    """Run the alternating EEG/BOLD loop. 1 EEG step every epoch, 1 BOLD step
    every ``bold_every`` epochs (BOLD is far more expensive per step).

    ``optimize`` selects which loss(es) actually take a gradient step:
    ``"eeg"`` or ``"bold"`` fits against only that target (the other update
    step is simply never called -- its loss history stays empty and its JAX
    computation is never traced/compiled, so there's no wasted cost either);
    ``"both"`` (default) is the original alternating fit.

    ``early_stop_patience`` (``None`` by default = old behaviour, always run
    all ``num_epochs``) stops the loop once every *actively optimized* loss
    (per ``optimize``) is stalled per ``is_loss_stalled`` with the given
    ``early_stop_window``/``early_stop_min_delta``. In ``"both"`` mode this
    requires BOTH losses to be stalled -- EEG is cheap and plateaus fast, so
    stopping the moment it alone plateaus would cut off BOLD's (typically
    much slower) fit early.

    ``on_epoch`` (``None`` by default), if given, is called after every epoch
    as ``on_epoch(epoch, loss_eeg, loss_bold, bold_stepped)`` -- ``loss_eeg``/
    ``loss_bold`` are ``None`` when that loss wasn't computed this epoch (not
    optimized, or a non-BOLD-step epoch). This is the hook external callers
    (e.g. an experiment-tracking sweep script) use to stream per-epoch metrics
    without this module needing to know anything about how they're logged.
    """
    if optimize not in ("eeg", "bold", "both"):
        raise ValueError(f"optimize must be 'eeg', 'bold', or 'both', got {optimize!r}")
    do_eeg = optimize in ("eeg", "both")
    do_bold = optimize in ("bold", "both")

    diff_params = diff_params_init
    # Separate optimizer state per loss -- not just one shared state -- so
    # Adam's per-parameter moment estimates (and step-count-dependent bias
    # correction) for the EEG PSD loss and the BOLD FC loss don't overwrite
    # each other between interleaved steps. Sharing one state here previously
    # made the alternating "both" fit's BOLD loss barely move even though a
    # BOLD-only fit converged fine at the same epoch count. Separate optimizer
    # TRANSFORMS (not just states) also let EEG and BOLD use different
    # learning rates.
    opt_state_eeg = eeg_optimizer.init(diff_params)
    opt_state_bold = bold_optimizer.init(diff_params)

    loss_history_eeg, loss_history_bold = [], []
    last_eeg_loss = last_bold_loss = float("nan")

    for epoch in range(num_epochs):
        if do_eeg:
            diff_params, opt_state_eeg, loss_eeg = eeg_update_step(
                diff_params, static_params, opt_state_eeg,
                target_psd, channel_indices, leadfield, smoothing_blocks, dipole_labels,
            )
            opt_state_eeg = jax.lax.stop_gradient(opt_state_eeg)
            loss_history_eeg.append(float(loss_eeg))
            last_eeg_loss = float(loss_eeg)

        bold_stepped = False
        if do_bold and (epoch + 1) % bold_every == 0:
            diff_params, opt_state_bold, loss_bold = bold_update_step(diff_params, static_params, opt_state_bold)
            opt_state_bold = jax.lax.stop_gradient(opt_state_bold)
            loss_history_bold.append(float(loss_bold))
            last_bold_loss = float(loss_bold)
            bold_stepped = True

        eeg_str = f"EEG: {last_eeg_loss:.5f}" if do_eeg else "EEG: (not optimized)"
        if do_bold:
            bold_str = f"BOLD FC: {last_bold_loss:.5f}" + ("" if bold_stepped else " (cached)")
        else:
            bold_str = "BOLD FC: (not optimized)"
        print(f"Epoch {epoch + 1:04d} | {eeg_str} | {bold_str}")

        if on_epoch is not None:
            on_epoch(
                epoch,
                last_eeg_loss if do_eeg else None,
                last_bold_loss if bold_stepped else None,
                bold_stepped,
            )

        if print_fn is not None and (epoch + 1) % print_every == 0:
            print_fn(diff_params)

        if early_stop_patience is not None:
            eeg_stalled = (not do_eeg) or is_loss_stalled(
                loss_history_eeg, early_stop_window, early_stop_patience, early_stop_min_delta)
            bold_stalled = (not do_bold) or is_loss_stalled(
                loss_history_bold, early_stop_window, early_stop_patience, early_stop_min_delta)
            if eeg_stalled and bold_stalled:
                print(f"Early stopping at epoch {epoch + 1}: loss trend stalled "
                      f"(window={early_stop_window}, patience={early_stop_patience}).")
                break

    return FitResult(diff_params, static_params, loss_history_eeg, loss_history_bold)


def run_phased_fit(
    diff_params_init,
    static_params,
    eeg_update_step,
    bold_update_step,
    eeg_optimizer,
    bold_optimizer,
    target_psd,
    channel_indices,
    leadfield,
    smoothing_blocks,
    dipole_labels,
    bold_phase_epochs=200,
    eeg_phase_epochs=200,
    print_every=10,
    print_fn=print_learnable_params,
    early_stop_window=20,
    early_stop_patience=None,
    early_stop_min_delta=1e-3,
    on_epoch=None,
) -> FitResult:
    """Two-phase fit: ``bold_phase_epochs`` of BOLD-only steps, then
    ``eeg_phase_epochs`` of EEG-only steps -- as opposed to
    ``run_alternating_fit``'s interleaved schedule. Both phases share ONE
    continuously-updated ``diff_params`` (phase 2 starts from phase 1's
    fitted values); this is literally two back-to-back
    ``run_alternating_fit`` calls with ``optimize="bold"`` then
    ``optimize="eeg"``, not new alternation math -- each phase gets its own
    fresh per-loss Adam state exactly as ``run_alternating_fit`` already
    builds internally (phase 1 never steps EEG, so there is no phase-1 EEG
    optimizer state for phase 2 to inherit anyway).

    ``on_epoch``'s epoch index restarts at 0 for phase 2 (phase-local, not a
    running total across both phases) -- a cosmetic simplification; the
    printed phase-boundary banner disambiguates which phase a given epoch
    number belongs to.
    """
    print(f"=== Phase 1/2: BOLD-only, {bold_phase_epochs} epochs ===")
    phase1 = run_alternating_fit(
        diff_params_init, static_params, eeg_update_step, bold_update_step,
        eeg_optimizer, bold_optimizer, target_psd, channel_indices, leadfield,
        smoothing_blocks, dipole_labels,
        num_epochs=bold_phase_epochs, bold_every=1, print_every=print_every, print_fn=print_fn,
        optimize="bold",
        early_stop_window=early_stop_window, early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        on_epoch=on_epoch,
    )

    print(f"=== Phase 2/2: EEG-only, {eeg_phase_epochs} epochs ===")
    phase2 = run_alternating_fit(
        phase1.diff_params, static_params, eeg_update_step, bold_update_step,
        eeg_optimizer, bold_optimizer, target_psd, channel_indices, leadfield,
        smoothing_blocks, dipole_labels,
        num_epochs=eeg_phase_epochs, bold_every=1, print_every=print_every, print_fn=print_fn,
        optimize="eeg",
        early_stop_window=early_stop_window, early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        on_epoch=on_epoch,
    )

    return FitResult(
        phase2.diff_params, static_params,
        loss_history_eeg=phase2.loss_history_eeg,
        loss_history_bold=phase1.loss_history_bold,
    )


def run_joint_fit(
    diff_params_init,
    static_params,
    joint_update_step,
    joint_optimizer,
    target_psd,
    channel_indices,
    leadfield,
    smoothing_blocks,
    dipole_labels,
    num_epochs=200,
    print_every=10,
    print_fn=print_learnable_params,
    early_stop_window=20,
    early_stop_patience=None,
    early_stop_min_delta=1e-3,
    on_epoch=None,
) -> FitResult:
    """Single combined-loss fit for the "joint" schedule: one gradient step
    per epoch against ``joint_eeg_weight * eeg_loss + joint_bold_weight *
    bold_loss`` (see ``make_joint_loss_fn``) -- both terms every epoch, no
    ``bold_every``-style skipping (a joint loss needs both every step).

    Both raw (unweighted) loss components are still logged into
    ``loss_history_eeg``/``loss_history_bold`` every epoch, so downstream
    reporting (``relative_final_loss``, the CLI's combined-ratio printout)
    works the same across all three schedules.
    """
    diff_params = diff_params_init
    opt_state = joint_optimizer.init(diff_params)

    loss_history_eeg, loss_history_bold = [], []

    for epoch in range(num_epochs):
        diff_params, opt_state, loss, loss_eeg, loss_bold = joint_update_step(
            diff_params, static_params, opt_state,
            target_psd, channel_indices, leadfield, smoothing_blocks, dipole_labels,
        )
        opt_state = jax.lax.stop_gradient(opt_state)
        loss_eeg = float(loss_eeg)
        loss_bold = float(loss_bold)
        loss_history_eeg.append(loss_eeg)
        loss_history_bold.append(loss_bold)

        print(f"Epoch {epoch + 1:04d} | joint: {float(loss):.5f} | "
              f"EEG: {loss_eeg:.5f} | BOLD FC: {loss_bold:.5f}")

        if on_epoch is not None:
            on_epoch(epoch, loss_eeg, loss_bold, True)

        if print_fn is not None and (epoch + 1) % print_every == 0:
            print_fn(diff_params)

        if early_stop_patience is not None:
            eeg_stalled = is_loss_stalled(
                loss_history_eeg, early_stop_window, early_stop_patience, early_stop_min_delta)
            bold_stalled = is_loss_stalled(
                loss_history_bold, early_stop_window, early_stop_patience, early_stop_min_delta)
            if eeg_stalled and bold_stalled:
                print(f"Early stopping at epoch {epoch + 1}: loss trend stalled "
                      f"(window={early_stop_window}, patience={early_stop_patience}).")
                break

    return FitResult(diff_params, static_params, loss_history_eeg, loss_history_bold)
