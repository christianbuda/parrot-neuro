"""Neural-mass model definitions for the TVB->EEG forward simulation.

`HeterogeneousModel` couples Jansen-Rit (cortex) and Wilson-Cowan (subcortex)
masses on one graph, switched per node by a cortical mask. Kept as plain,
hand-written Python (not notebook-serialized) so the dynamics can never be
silently corrupted by a notebook round-trip.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics import Network
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics


class HeterogeneousModel(AbstractDynamics):
    """Heterogeneous neural mass model: Jansen-Rit (cortex) + Wilson-Cowan (subcortex).

    Node type is selected by ``mask_cortical`` (1 = cortical/JR, 0 = subcortical/WC).
    State layout (9 vars): 6 Jansen-Rit (y0..y5), 2 Wilson-Cowan (E, I), and a
    shared low-pass ``network_out`` that both populations project through.
    """

    STATE_NAMES = ("y0", "y1", "y2", "y3", "y4", "y5", "wc_E", "wc_I", "network_out")
    # Sensible resting-state initial conditions.
    INITIAL_STATE = (0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.1, 0.05, 0.0)
    COUPLING_INPUTS = {"delayed": 1}
    EXTERNAL_INPUTS = {"stimulus": 1}

    DEFAULT_PARAMS = Bunch(
        # --- heterogeneity masks ---
        mask_cortical=0.0,
        mask_subcortical=0.0,
        # --- network params ---
        cortex_coupling_scale=1,  # keeps network input from saturating JR
        tau_axon_JR=15.0,         # low-pass tau for JR axonal projection (ms)
        tau_axon_WC=1.0,          # low-pass tau for WC axonal projection (ms)
        # --- Jansen-Rit params (ms scale) ---
        A=3.25,          # max EPSP amplitude [mV]
        B=22.0,          # max IPSP amplitude [mV]
        a=0.1,           # reciprocal membrane time constant [ms^-1]
        b=0.05,          # reciprocal membrane time constant [ms^-1]
        v0=5.52,         # firing threshold [mV]
        nu_max=0.0025,   # max firing rate [ms^-1]
        r=0.56,          # sigmoid steepness [mV^-1]
        J=135.0,         # average number of synapses
        a_1=1.0,         # excitatory feedback probability
        a_2=0.8,         # slow excitatory feedback probability
        a_3=0.25,        # inhibitory feedback probability
        a_4=0.25,        # slow inhibitory feedback probability
        mu=0.22,         # mean background input firing rate
        # --- Wilson-Cowan params ---
        c_ee=12.0, c_ei=4.0, c_ie=13.0, c_ii=11.0,   # local connectivity
        tau_e=10.0, tau_i=10.0,                       # time constants (ms)
        a_e=1.2, b_e=2.8, c_e=1.0, theta_e=0.0,       # excitatory sigmoid
        a_i=1.0, b_i=4.0, c_i=1.0, theta_i=0.0,       # inhibitory sigmoid
        r_e=1.0, r_i=1.0, k_e=1.0, k_i=1.0,           # response modulation
        P=0.0, Q=0.0,                                 # external inputs
        alpha_e=1.0, alpha_i=1.0,                     # input gains
        shift_sigmoid=1.0,  # 1.0 = baseline-corrected sigmoid, 0.0 = standard
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store masks on the instance to avoid global-scope leaks at init.
        self.mask_cortical = kwargs.get("mask_cortical", self.DEFAULT_PARAMS.mask_cortical)
        self.mask_subcortical = 1 - self.mask_cortical
        # `dynamics()` reads `params.mask_subcortical` (the pytree entry set by
        # AbstractDynamics.__init__ from kwargs), not `self.mask_subcortical`.
        # Only `mask_cortical` is ever passed as a kwarg, so without this line
        # `params.mask_subcortical` is stuck at DEFAULT_PARAMS' scalar 0. and
        # `d_wc` below is always zero — Wilson-Cowan (subcortical) states never
        # evolve and c_ee/P silently have no effect on the simulation at all.
        self.params["mask_subcortical"] = self.mask_subcortical

    def get_default_initial_state(self, n_nodes):
        """Broadcast INITIAL_STATE to n_nodes, zeroing inactive populations."""
        states = jnp.tile(jnp.array(self.INITIAL_STATE)[:, None], (1, n_nodes))
        states = states.at[0:6, :].multiply(self.mask_cortical)   # JR only in cortex
        states = states.at[6:8, :].multiply(self.mask_subcortical)  # WC only in subcortex
        init_out = (0.0 * self.mask_cortical) + (0.1 * self.mask_subcortical)
        states = states.at[8, :].set(init_out)
        return states

    def dynamics(self, t, state, params, coupling, external):
        y0, y1, y2, y3, y4, y5, E, I, network_out = state

        global_input = coupling.delayed[0]
        stim = external.stimulus[0] if "stimulus" in external else 0.0

        def jr_sigmoid(v):
            # voltages -> absolute firing rates (ms^-1)
            return (2.0 * params.nu_max) * jax.nn.sigmoid(params.r * (v - params.v0))

        def wc_sigmoid(x, a, b, c):
            std_sig = c * jax.nn.sigmoid(a * (x - b))
            shift = (c * jax.nn.sigmoid(-a * b)) * params.shift_sigmoid
            return std_sig - shift

        # --- Jansen-Rit (cortex) ---
        safe_cortex_input = global_input * params.cortex_coupling_scale
        jr_total_input = params.mu + safe_cortex_input + stim
        dy0 = y3
        dy3 = params.A * params.a * jr_sigmoid(y1 - y2) - 2.0 * params.a * y3 - (params.a**2) * y0
        dy1 = y4
        dy4 = (
            params.A * params.a
            * (jr_total_input + (params.J * params.a_2) * jr_sigmoid((params.J * params.a_1) * y0))
            - 2.0 * params.a * y4 - (params.a**2) * y1
        )
        dy2 = y5
        dy5 = (
            params.B * params.b
            * ((params.J * params.a_4) * jr_sigmoid((params.J * params.a_3) * y0))
            - 2.0 * params.b * y5 - (params.b**2) * y2
        )
        d_jr = jnp.where(
            params.mask_cortical == 1, jnp.stack([dy0, dy1, dy2, dy3, dy4, dy5]), 0.0
        )

        # --- Wilson-Cowan (subcortex) ---
        x_e = params.alpha_e * (
            params.c_ee * E - params.c_ei * I + params.P - params.theta_e + global_input + stim
        )
        x_i = params.alpha_i * (params.c_ie * E - params.c_ii * I + params.Q - params.theta_i)
        S_e = wc_sigmoid(x_e, params.a_e, params.b_e, params.c_e)
        S_i = wc_sigmoid(x_i, params.a_i, params.b_i, params.c_i)
        dE = (-E + (params.k_e - params.r_e * E) * S_e) / params.tau_e
        dI = (-I + (params.k_i - params.r_i * I) * S_i) / params.tau_i
        d_wc = jnp.where(params.mask_subcortical == 1, jnp.stack([dE, dI]), 0.0)

        # --- Axonal projection (shared low-pass output channel) ---
        jr_firing_rate = jr_sigmoid(y1 - y2)
        jr_proportion = jr_firing_rate / (2.0 * params.nu_max)  # normalize to [0, 1]
        target_firing_rate = jnp.where(params.mask_cortical == 1, jr_proportion, E)
        tau_axon = jnp.where(params.mask_cortical == 1, params.tau_axon_JR, params.tau_axon_WC)
        d_network_out = (target_firing_rate - network_out) / tau_axon

        return jnp.concatenate([d_jr, d_wc, jnp.stack([d_network_out])], axis=0)


class CustomStateNetwork(Network):
    """Network that warm-starts from the last simulated state when history exists."""

    @property
    def initial_state(self) -> jnp.ndarray:
        if self._history is not None:
            return self._history.ys[-1]  # [n_states, n_nodes]
        return self.dynamics.get_default_initial_state(self.graph.n_nodes)
