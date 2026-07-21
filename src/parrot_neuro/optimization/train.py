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

from .config import DEFAULT_LEARNABLE_PARAMS, LearnableParam
from .connectivity import fc_vector
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
) -> Simulators:
    """Warm up ``network`` at both horizons and prepare pure solve functions.

    Order matters: the EEG (short) warm-up seeds the delay history first,
    then the BOLD (long) warm-up overwrites it — the BOLD monitor's HRF
    convolution needs that longer history. Both ``prepare()`` calls expose
    the same differentiable leaves (only static per-horizon delay-buffer
    metadata differs, baked into each closure), so the params from the
    *second* call are the ones used for both simulators.
    """
    print(f"Preparing simulators: EEG t1={t1_eeg:.1f}s, BOLD t1={t1_bold:.1f}s")
    result_eeg = solve(network, solver, t0=t0, t1=t1_eeg, dt=dt)
    network.update_history(result_eeg)
    simulator_eeg, _ = prepare(network, solver, t0=t0, t1=t1_eeg, dt=dt)

    result_bold = solve(network, solver, t0=t0, t1=t1_bold, dt=dt)
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
    """Log-PSD MSE loss, closing over everything that never changes per-step
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

        sim_log = jnp.log(norm_sim + 1e-8)
        target_log = jnp.log(norm_target + 1e-8)
        return jnp.mean((sim_log[:, idx_min:idx_max] - target_log[:, idx_min:idx_max]) ** 2)

    return eeg_loss_fn


def make_bold_loss_fn(simulator_bold, bold_monitor, target_fc_vec, skip_t, eps=1e-8, bad_loss=1e3):
    """FC-vector MSE loss; falls back to a large constant loss (rather than
    NaN) if the simulation blew up, so a single unlucky epoch doesn't poison
    the gradient with NaNs the optimizer can never recover from."""

    @eqx.filter_jit
    def bold_loss_fn(current_diff, current_static):
        combined = eqx.combine(current_diff, current_static)
        sol = simulator_bold(combined)
        Xs = bold_monitor(sol).ys[:, 0, :][skip_t:, :]
        ok = jnp.all(jnp.isfinite(Xs))

        def good(_):
            fc_sim = fc_vector(Xs, skip_t=0, eps=eps)
            return jnp.mean((fc_sim - target_fc_vec) ** 2)

        def bad(_):
            return jnp.array(bad_loss, dtype=jnp.float64)

        return jax.lax.cond(ok, good, bad, operand=None)

    return bold_loss_fn


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
) -> FitResult:
    """Run the alternating EEG/BOLD loop. 1 EEG step every epoch, 1 BOLD step
    every ``bold_every`` epochs (BOLD is far more expensive per step)."""
    diff_params = diff_params_init
    opt_state = optimizer.init(diff_params)

    loss_history_eeg, loss_history_bold = [], []
    last_bold_loss = float("nan")

    for epoch in range(num_epochs):
        diff_params, opt_state, loss_eeg = eeg_update_step(
            diff_params, static_params, opt_state,
            target_psd, channel_indices, leadfield, smoothing_blocks, dipole_labels,
        )
        opt_state = jax.lax.stop_gradient(opt_state)
        loss_history_eeg.append(float(loss_eeg))

        if (epoch + 1) % bold_every == 0:
            diff_params, opt_state, loss_bold = bold_update_step(diff_params, static_params, opt_state)
            opt_state = jax.lax.stop_gradient(opt_state)
            loss_history_bold.append(float(loss_bold))
            last_bold_loss = float(loss_bold)
            print(f"Epoch {epoch + 1:04d} | EEG: {loss_eeg:.5f} | BOLD FC: {loss_bold:.5f}")
        else:
            print(f"Epoch {epoch + 1:04d} | EEG: {loss_eeg:.5f} | BOLD FC: {last_bold_loss:.5f} (cached)")

        if print_fn is not None and (epoch + 1) % print_every == 0:
            print_fn(diff_params)

    return FitResult(diff_params, static_params, loss_history_eeg, loss_history_bold)
