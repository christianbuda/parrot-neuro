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
from tvboptim.observations.tvb_monitors import Bold

from .config import BOLD_BANDPASS_HIGH, BOLD_BANDPASS_LOW, BOLD_BANDPASS_ORDER, DEFAULT_LEARNABLE_PARAMS, LearnableParam
from .connectivity import dfc_histogram, fc_vector, filter_sim_bold, wasserstein_1d_from_hist
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
    warm-up's history."""

    simulator_eeg: Callable
    simulator_bold: Callable
    bold_monitor: Bold
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
    simulator_bold, params = prepare(network, solver, t0=t0, t1=t1_bold, dt=dt)

    bold_monitor = Bold(
        history=result_bold,
        period=tr_ms,
        downsample_period=bold_downsample_ms,
        voi=bold_voi,
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


def make_eeg_loss_fn(simulator_eeg, mask_cortical, idx_min, idx_max, dt,
                      settle_ms=500.0, stride_ms=4.0):
    """Linear-PSD MSE loss, closing over everything that never changes per-step
    (the mask, frequency-bin window, and timing — mirrors how the simulator
    itself is a closure): only ``(diff, static, target_psd, channel_indices,
    leadfield, smoothing_blocks, dipole_labels)`` vary call to call.
    """
    settle = int(settle_ms / dt)
    stride = int(stride_ms / dt)
    mask_col = jnp.atleast_2d(jnp.asarray(mask_cortical)).T

    @eqx.filter_jit
    def eeg_loss_fn(current_diff, current_static, target_psd, channel_indices,
                     leadfield, smoothing_blocks, dipole_labels):
        combined = eqx.combine(current_diff, current_static)
        sim_result = simulator_eeg(combined)

        # JR pyramidal output (y1 - y2), zeroed on subcortical nodes.
        source_activity = (
            sim_result.ys[settle::stride, 1].T - sim_result.ys[settle::stride, 2].T
        ) * mask_col

        simulated_eeg = project_to_scalp(
            source_activity, channel_indices, leadfield, smoothing_blocks, dipole_labels
        )

        sim_psd = smooth_ts(compute_psd(simulated_eeg))
        target_psd = smooth_ts(target_psd)

        norm_sim = sim_psd / (jnp.sum(sim_psd[:, idx_min:idx_max], keepdims=True) + 1e-8)
        norm_target = target_psd / (jnp.sum(target_psd[:, idx_min:idx_max], keepdims=True) + 1e-8)

        return 10000 *jnp.mean((norm_sim[:, idx_min:idx_max] - norm_target[:, idx_min:idx_max]) ** 2)

    return eeg_loss_fn


def make_bold_loss_fn(simulator_bold, bold_monitor, target_fc_vec, skip_t, tr_ms,
                       bandpass_low=BOLD_BANDPASS_LOW, bandpass_high=BOLD_BANDPASS_HIGH,
                       bandpass_order=BOLD_BANDPASS_ORDER, eps=1e-8, bad_loss=1e3):
    """FC-vector MSE loss; falls back to a large constant loss (rather than
    NaN) if the simulation blew up, so a single unlucky epoch doesn't poison
    the gradient with NaNs the optimizer can never recover from.

    Simulated BOLD is bandpassed (``connectivity.filter_sim_bold``, still
    differentiable) to the same band the empirical BOLD (hence
    ``target_fc_vec``) was already preprocessed with -- otherwise the FC
    comparison would be between differently-filtered signals.
    """

    @eqx.filter_jit
    def bold_loss_fn(current_diff, current_static):
        combined = eqx.combine(current_diff, current_static)
        sol = simulator_bold(combined)
        Xs = bold_monitor(sol).ys[:, 0, :][skip_t:, :]
        ok = jnp.all(jnp.isfinite(Xs))

        def good(_):
            Xs_filt = filter_sim_bold(Xs, tr_ms, low=bandpass_low, high=bandpass_high, order=bandpass_order)
            fc_sim = fc_vector(Xs_filt, skip_t=0, eps=eps)
            return jnp.mean((fc_sim - target_fc_vec) ** 2)

        def bad(_):
            return jnp.array(bad_loss, dtype=jnp.float64)

        return jax.lax.cond(ok, good, bad, operand=None)

    return bold_loss_fn


def make_bold_dfc_loss_fn(simulator_bold, bold_monitor, target_hist, skip_t, tr_ms,
                           window_trs, step_trs, centers, k_min=1, sigma=0.05,
                           bandpass_low=BOLD_BANDPASS_LOW, bandpass_high=BOLD_BANDPASS_HIGH,
                           bandpass_order=BOLD_BANDPASS_ORDER, eps=1e-8, bad_loss=1e3):
    """Dynamic-FC (FCD) Wasserstein loss -- the dfc alternative to
    make_bold_loss_fn's static FC. Compares soft-histogram-summarized
    FCD-value distributions rather than raw FCD matrices (see
    connectivity.dfc_histogram for why), via the 1-Wasserstein distance
    between the two histograms on the shared ``centers`` grid; same
    NaN-guarded shape as make_bold_loss_fn otherwise. Simulated BOLD is
    bandpassed the same way as make_bold_loss_fn before windowing."""

    @eqx.filter_jit
    def bold_dfc_loss_fn(current_diff, current_static):
        combined = eqx.combine(current_diff, current_static)
        sol = simulator_bold(combined)
        Xs = bold_monitor(sol).ys[:, 0, :][skip_t:, :]
        ok = jnp.all(jnp.isfinite(Xs))

        def good(_):
            Xs_filt = filter_sim_bold(Xs, tr_ms, low=bandpass_low, high=bandpass_high, order=bandpass_order)
            sim_hist = dfc_histogram(Xs_filt, window_trs, step_trs, centers, skip_t=0,
                                      k_min=k_min, sigma=sigma, eps=eps)
            return wasserstein_1d_from_hist(sim_hist, target_hist)

        def bad(_):
            return jnp.array(bad_loss, dtype=jnp.float64)

        return jax.lax.cond(ok, good, bad, operand=None)

    return bold_dfc_loss_fn


def make_optimizer(learning_rate=1e-3, grad_clip_norm=1.0):
    return optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(learning_rate))


def make_update_steps(eeg_loss_fn, bold_loss_fn, optimizer):
    """Gradient-step closures over the two loss functions and the optimizer."""

    @eqx.filter_jit
    def eeg_update_step(current_diff, current_static, current_opt_state,
                         target_psd, channel_indices, leadfield, smoothing_blocks, dipole_labels):
        loss, grads = jax.value_and_grad(eeg_loss_fn, argnums=0)(
            current_diff, current_static, target_psd, channel_indices,
            leadfield, smoothing_blocks, dipole_labels,
        )
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_diff)
        new_diff = optax.apply_updates(current_diff, updates)
        return new_diff, new_opt_state, loss

    @eqx.filter_jit
    def bold_update_step(current_diff, current_static, current_opt_state):
        loss, grads = jax.value_and_grad(bold_loss_fn, argnums=0)(current_diff, current_static)
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_diff)
        new_diff = optax.apply_updates(current_diff, updates)
        return new_diff, new_opt_state, loss

    return eeg_update_step, bold_update_step


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
    optimizer,
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
    # BOLD-only fit converged fine at the same epoch count.
    opt_state_eeg = optimizer.init(diff_params)
    opt_state_bold = optimizer.init(diff_params)

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
