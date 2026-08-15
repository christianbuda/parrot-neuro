"""Assemble a ready-to-run ``HeterogeneousModel`` network from a config.

Wires together the pieces from ``model``, ``connectivity`` and the bounded
Heun solver, and wraps exactly the parameters named in a
``config.LearnableParam`` list as ``SigmoidBoundedParameter`` — the same
list ``train.learnable_partition`` uses to build its optimizer mask, so the
two can never disagree about what's learnable.
"""
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from tvboptim.experimental.network_dynamics.coupling import DelayedLinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.types import SigmoidBoundedParameter

from .config import DEFAULT_LEARNABLE_PARAMS, LearnableParam
from .model import CustomStateNetwork, HeterogeneousModel

#: Parameters DelayedLinearCoupling accepts (mirrors its own DEFAULT_PARAMS;
#: used only to validate LearnableParam(location="coupling", name=...) early
#: with a clear error instead of a silent typo'd new Bunch key).
_COUPLING_PARAM_NAMES = frozenset(DelayedLinearCoupling.DEFAULT_PARAMS.keys())


def _make_learnable(brain_model, name: str, low: float, high: float, init: float | None, num_nodes: int):
    if name not in HeterogeneousModel.DEFAULT_PARAMS:
        raise ValueError(
            f"Unknown dynamics parameter {name!r}; must be one of "
            f"{sorted(HeterogeneousModel.DEFAULT_PARAMS.keys())}"
        )
    if init is None:
        init = HeterogeneousModel.DEFAULT_PARAMS[name]
    brain_model.params[name] = SigmoidBoundedParameter(
        jnp.full((num_nodes,), init, dtype=jnp.float64), low, high
    )


def build_network(
    mask_cortical: np.ndarray,
    weights: jnp.ndarray,
    delays: jnp.ndarray,
    num_nodes: int,
    learnable_params: Sequence[LearnableParam] = DEFAULT_LEARNABLE_PARAMS,
    base_sigma: float = 0.048,
    noise_seed: int = 69,
    solver_block_size: int | None = None,
):
    """Build the JR-cortex/WC-subcortex network, wrapping each parameter
    named in ``learnable_params`` as a ``SigmoidBoundedParameter``.

    Returns (network, solver, brain_model). Everything not named in
    ``learnable_params`` (masks, connectivity, noise, initial state, and
    every other dynamics/coupling parameter) stays a plain array — see
    ``train.learnable_partition`` for why that distinction matters when
    partitioning the simulator's parameters for gradient descent.

    ``solver_block_size`` (default ``None``, i.e. off) controls the native
    solver's backward-pass memory: with ``None`` the integration is a single
    monolithic ``jax.lax.scan`` and every step's state is kept live for the
    backward pass -- O(n_steps) GPU memory, which is the dominant cost for
    the long BOLD horizon (e.g. t1_bold=320s at dt=1ms is 320k steps). Set it
    to an int ``K`` to checkpoint the scan in blocks of ``K`` steps
    (``jax.checkpoint`` under the hood): backward memory drops to
    ``O(n_steps/K + K)`` at the cost of ~1.3-1.7x gradient wall-time (one
    extra forward recompute per block). Peak memory is U-shaped in ``K``;
    ``K ~ sqrt(n_steps)`` is the rule-of-thumb optimum -- see
    ``NativeSolver.__init__`` in ``tvboptim.experimental.network_dynamics.solvers.native``
    for the full accounting. The forward computation and the gradient are
    unaffected (exact, not truncated) -- this is a pure memory/compute trade,
    unlike ``grad_horizon`` (truncated BPTT, not exposed here since it biases
    the gradient toward fast timescales).

    This ``solver`` is shared by both the EEG and BOLD ``prepare()`` calls in
    ``train.build_simulators`` -- the BOLD one also streams its HRF
    convolution through this same block scan (``reduce=streaming_hrf_bold``),
    which requires ``K`` to be an *exact* multiple of the BOLD period in raw
    steps (``tr_ms / dt``), not just close to ``sqrt(n_steps)``. See
    ``config.BoldFitConfig.solver_block_size`` for the full rationale.
    """
    print(f"Building network with {num_nodes} nodes, {len(learnable_params)} learnable params, "
          f"base_sigma={base_sigma}, noise_seed={noise_seed}, solver_block_size={solver_block_size}")
    brain_model = HeterogeneousModel(mask_cortical=mask_cortical)

    # Each stays a plain float (not learnable) unless a "coupling"
    # LearnableParam below wraps it as a SigmoidBoundedParameter.
    coupling_values = dict(DelayedLinearCoupling.DEFAULT_PARAMS)

    for lp in learnable_params:
        if lp.location == "dynamics":
            _make_learnable(brain_model, lp.name, lp.low, lp.high, lp.init, num_nodes)
        elif lp.location == "coupling":
            if lp.name not in _COUPLING_PARAM_NAMES:
                raise ValueError(
                    f"Unknown coupling parameter {lp.name!r}; must be one of {sorted(_COUPLING_PARAM_NAMES)}"
                )
            init = coupling_values[lp.name] if lp.init is None else lp.init
            coupling_values[lp.name] = SigmoidBoundedParameter(
                jnp.array([init], dtype=jnp.float64), lp.low, lp.high
            )
        else:
            raise ValueError(f"Unknown location {lp.location!r} for learnable param {lp.name!r}")

    # Noise only on the "live" states per node type: JR voltages (0:3) in
    # cortex, WC proportions (6:8) in subcortex. network_out stays noiseless
    # so the shared low-pass channel isn't itself a noise source.
    sigma_matrix = np.zeros((9, num_nodes))
    sigma_matrix[0:3, :] = base_sigma * mask_cortical
    sigma_matrix[6:8, :] = base_sigma * (1 - mask_cortical)

    network = CustomStateNetwork(
        dynamics=brain_model,
        coupling={"delayed": DelayedLinearCoupling(incoming_states="network_out", **coupling_values)},
        graph=DenseDelayGraph(weights, delays),
        noise=AdditiveNoise(sigma=jnp.array(sigma_matrix), key=jax.random.key(noise_seed)),
    )

    low_bounds = jnp.full((9, num_nodes), -jnp.inf)
    high_bounds = jnp.full((9, num_nodes), jnp.inf)
    low_bounds = low_bounds.at[0:3, :].set(-50.0)     # JR voltages
    high_bounds = high_bounds.at[0:3, :].set(50.0)
    low_bounds = low_bounds.at[6:8, :].set(0.0)       # WC proportions
    high_bounds = high_bounds.at[6:8, :].set(1.0)
    low_bounds = low_bounds.at[8, :].set(0.0)         # network_out
    high_bounds = high_bounds.at[8, :].set(1.0)
    solver = BoundedSolver(Heun(block_size=solver_block_size), low=low_bounds, high=high_bounds)

    return network, solver, brain_model
