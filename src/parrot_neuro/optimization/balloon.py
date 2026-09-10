"""Block-streaming reducer for ``tvboptim``'s ``BalloonWindkesselBold`` monitor.

``tvboptim`` ships ``BalloonWindkesselBold`` (Friston 2000 / Deco 2014: a
4-state hemodynamic ODE -- vasodilatory signal ``s``, blood flow ``f``, blood
volume ``v``, deoxyhemoglobin ``q``) but only as a post-hoc ``__call__(sol)``
that needs the *entire* raw per-ms trajectory materialized first. That is
exactly what ``tvboptim.observations.tvb_monitors.streaming_hrf_bold`` exists
to avoid for the HRF-convolution path (see ``train.build_simulators``'s
docstring for the OOM this sidesteps), by folding into the same
block-checkpointed scan ``prepare(..., reduce=...)`` already runs. This module
is the same trick for the Balloon-Windkessel ODE.

The Balloon ODE is actually a better fit for streaming than HRF's convolution:
it is already a sequential recurrence (``jax.lax.scan`` over ``bw_step`` in
``BalloonWindkesselBold.__call__``), so its carry is just the tiny 4-tuple
``(s, f, v, q)`` per node -- no kernel-length ring buffer, no history/warm-start
buffer at all. The ODE's own time constants (``taus``~0.65s, ``tauf``~0.41s,
``tauo``~0.98s) settle well inside this pipeline's existing BOLD burn-in
(``BoldFitConfig.bold_skip_trs``, ~11s+), so starting every build from the
standard resting initial condition (``s=0, f=v=q=1``) is standard practice,
not a shortcut -- unlike ``HRFBold``, there is no warm-start parameter here.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from tvboptim.observations.tvb_monitors import BalloonWindkesselBold


def streaming_balloon_bold(monitor: BalloonWindkesselBold, dt: float):
    """Block-level streaming reducer form of :class:`BalloonWindkesselBold`.

    Returns an ``(init, update, finalize)`` triple for the ``reduce=`` kwarg of
    ``prepare()``/``solve()``, integrating the same Balloon-Windkessel ODE as
    ``monitor(full_solution)`` without ever stacking the full neural
    trajectory. Mirrors ``streaming_hrf_bold``'s contract exactly.

    Requirements (so blocks align with the TR grid, same shape of constraint
    as ``streaming_hrf_bold``):

    - ``monitor.downsample`` must be ``None`` and ``monitor.dt_bw`` must equal
      ``dt`` -- this reducer always integrates at the raw simulation step; it
      does not support ``BalloonWindkesselBold``'s optional pre-integration
      downsampling or a ``dt_bw != dt`` repeat/subsample stage.
    - ``block_size`` (or ``n_steps``, when ``solver.block_size`` is ``None``)
      must be a multiple of ``period / dt_bw`` (the TR in raw steps); the
      per-block ``update`` asserts this.
    """
    if monitor.downsample is not None:
        raise ValueError(
            "streaming_balloon_bold does not support monitor.downsample -- "
            "build the BalloonWindkesselBold with downsample=None (integrate "
            "at the raw simulation dt)."
        )
    if not jnp.isclose(monitor.dt_bw, dt):
        raise ValueError(
            f"streaming_balloon_bold requires monitor.dt_bw == dt (raw "
            f"simulation step); got dt_bw={monitor.dt_bw}, dt={dt}."
        )

    voi = monitor.voi
    period = monitor.period
    dt_bw = monitor.dt_bw
    taus, tauf, tauo, alpha = monitor.taus, monitor.tauf, monitor.tauo, monitor.alpha
    Eo, vo = monitor.Eo, monitor.vo
    k1, k2, k3 = monitor.k1, monitor.k2, monitor.k3

    save_every = int(round(period / dt_bw))
    dt_s = dt_bw / 1000.0

    itaus = 1.0 / taus
    itauf = 1.0 / tauf
    itauo = 1.0 / tauo
    ialpha = 1.0 / alpha

    def bw_step(state, r_t):
        s, f, v, q = state

        ds = r_t - itaus * s - itauf * (f - 1.0)
        df = s
        dv = itauo * (f - v**ialpha)
        dq = itauo * (f * (1.0 - (1.0 - Eo) ** (1.0 / f)) / Eo - v ** (ialpha - 1.0) * q)

        s = s + dt_s * ds
        f = f + dt_s * df
        v = v + dt_s * dv
        q = q + dt_s * dq

        bold = vo * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))
        return (s, f, v, q), bold

    def init(template, n_steps):
        n_nodes = template[voi, :].squeeze(0).shape[-1]
        n_bold = n_steps // save_every
        state0 = (
            jnp.zeros(n_nodes),  # s
            jnp.ones(n_nodes),   # f
            jnp.ones(n_nodes),   # v
            jnp.ones(n_nodes),   # q
        )
        bold0 = jnp.zeros((n_bold, 1, n_nodes))
        return (state0, bold0, jnp.array(0))

    def update(acc, block):
        state, bold_buffer, step_count = acc
        r = block[:, voi, :].squeeze(1)  # [block_len, n_nodes] (voi is dimension-preserving, size 1)
        block_len = r.shape[0]
        assert block_len % save_every == 0, (
            "streaming_balloon_bold requires each block length to be a "
            f"multiple of the BOLD period in steps ({save_every} = "
            f"period/dt_bw); got {block_len}. Set block_size and n_steps to "
            "multiples of period/dt_bw."
        )

        new_state, bold_all = jax.lax.scan(bw_step, state, r)  # bold_all: [block_len, n_nodes]

        idx = jnp.arange(save_every - 1, block_len, save_every)
        bold_samples = bold_all[idx][:, None, :]  # [m_b, 1, n_nodes]

        start = step_count // save_every
        bold_buffer = jax.lax.dynamic_update_slice(
            bold_buffer, bold_samples, (start, 0, 0)
        )
        return (new_state, bold_buffer, step_count + block_len)

    def finalize(acc):
        _state, bold_buffer, _step_count = acc
        return bold_buffer

    return (init, update, finalize)
