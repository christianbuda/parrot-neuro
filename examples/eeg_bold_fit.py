# %% [markdown]
# # EEG + BOLD alternating fit — JR cortex / WC subcortex
#
# Thin driver over `parrot_neuro.optimization`: all the actual logic (data loading,
# forward model, network assembly, loss functions, training loop, plots)
# lives there and is reusable outside this notebook. This file only wires a
# subject + hyperparameters together and calls it.
#
# To fit a different subject / atlas / hyperparameters, edit the
# `config.BoldFitConfig(...)` call in the cell below — nothing else should
# need to change.

# %%
from parrot_neuro.optimization import config
config.apply_jax_env()  # must run before any jax import (sets CUDA/JAX env vars)

# %%
import jax
jax.config.update("jax_enable_x64", True)

from parrot_neuro import Subject
from parrot_neuro.optimization import connectivity, data, pipeline, train, viz

# %% [markdown]
# ## Subject + hyperparameters
#
# `subject` is a `parrot_neuro.Subject` over one reconstructed subject's
# Parrot derivatives. `BoldFitConfig` defaults mirror `optimization.config`'s
# module-level atlas/spacing constants; the subject itself has no default —
# point it at whichever subject you're fitting.

# %%
# --- edit these three for your run ---
BIDS_ROOT = "/srv/nfs-data/sisko/christian/parrot_LEMON"  # Parrot derivatives root
subject_id = "010005"
output_root = "eeg_bold_fit_res"  # per-subject results go under <output_root>/<subject_id>

subject = Subject(BIDS_ROOT, subject_id)

import os
output_dir = os.path.join(output_root, subject_id)
os.makedirs(output_dir, exist_ok=True)

cfg = config.BoldFitConfig(
    subject=subject,
    output_dir=output_dir,
    num_epochs=50,
    bold_every=1,
    # `learnable_params` controls exactly which parameters the optimizer can
    # touch — defaults to config.DEFAULT_LEARNABLE_PARAMS (a still-evolving
    # prototyping set spanning JR, WC, and coupling params; see config for the
    # current list). Pass your own tuple to add/remove/rebound parameters, e.g.
    # to also learn the WC excitatory->inhibitory coupling c_ei:
    #
    # learnable_params=config.DEFAULT_LEARNABLE_PARAMS + (
    #     config.LearnableParam("c_ei", low=2.0, high=8.0, location="dynamics"),
    # ),
)

# %% [markdown]
# ## Load this subject's EEG chunks
#
# Reads the subject's own splice-free EEG segments
# (`derivatives/EEG/sub-<ID>/..._task-<eeg_task>_eeg.npz` + its sidecar JSON
# for sampling rate/channel names) — a first-class Parrot output, so no more
# site-specific `all_data.pkl` schema to slice by hand.

# %%
dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
print(f"Fitting {subject.subj}: {len(dataset)} chunks of {cfg.chunk_length} samples")

# %% [markdown]
# ## Build the network + simulators, then run the alternating fit

# %%
ctx = pipeline.build_context(cfg, dataset)

# %%
result = pipeline.fit(ctx)

# %% [markdown]
# ## Save results

# %%
import numpy as np
from pathlib import Path

out_dir = Path(cfg.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "loss_history_eeg.npy", np.array(result.loss_history_eeg))
np.save(out_dir / "loss_history_bold.npy", np.array(result.loss_history_bold))

optimized = train.extract_learnable_values(result.diff_params, cfg.learnable_params)
np.savez(out_dir / "optimized_params.npz", **optimized,
         loss_eeg=np.array(result.loss_history_eeg), loss_bold=np.array(result.loss_history_bold))

print(f"Final EEG loss:  {result.loss_history_eeg[-1]:.5f}")
print(f"Final BOLD loss: {result.loss_history_bold[-1]:.5f}")
for name, values in optimized.items():
    print(f"{name:6s} — mean {values.mean():.4f}  std {values.std():.4f}")
print(f"Saved to {out_dir}")

# %% [markdown]
# ## Diagnostics

# %%
import equinox as eqx

combined = eqx.combine(result.diff_params, result.static_params)
sim_result_eeg = ctx.simulators.simulator_eeg(combined)
sim_result_bold = ctx.simulators.simulator_bold(combined)

# %%
fig = viz.plot_node_activity(sim_result_eeg, ctx.mask_cortical, cfg.dt)
fig.savefig(out_dir / "node_activity.png", dpi=150)

# %%
sim_bold_2d = connectivity.extract_bold_2d(ctx.simulators.bold_monitor(sim_result_bold))
fig = viz.plot_bold_timeseries(sim_bold_2d, ctx.sc.empirical_bold, ctx.mask_cortical,
                                cfg.tr_ms, skip_t=cfg.bold_skip_trs)
fig.savefig(out_dir / "bold_timeseries.png", dpi=150)

# %%
fig, fc_corr = viz.plot_fc_comparison(sim_bold_2d, ctx.sc.empirical_bold, skip_t=cfg.bold_skip_trs)
fig.savefig(out_dir / "fc_comparison.png", dpi=150)
print(f"FC Pearson correlation (sim vs emp): {fc_corr:.4f}")

# %%
from parrot_neuro.optimization.forward import project_to_scalp
from parrot_neuro.optimization.signal import compute_psd
import jax.numpy as jnp

source_activity = (
    sim_result_eeg.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 1].T
    - sim_result_eeg.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 2].T
) * jnp.atleast_2d(ctx.mask_cortical).T
simulated_eeg = project_to_scalp(
    source_activity, dataset.channel_indices, ctx.leadfield, ctx.smoothing_blocks, ctx.dipole_labels
)
sim_psd = compute_psd(simulated_eeg)

fig = viz.plot_eeg_psd_comparison(sim_psd, ctx.target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max)
fig.savefig(out_dir / "eeg_psd_comparison.png", dpi=150)

# %%
fig, r_eeg = viz.plot_eeg_corr_comparison(simulated_eeg, dataset._chunks)
fig.savefig(out_dir / "eeg_corr_comparison.png", dpi=150)
print(f"EEG correlation matrix Pearson r: {r_eeg:.4f}")

# %% [markdown]
# ## Learning: first iteration (init params) vs last iteration (fitted params)
#
# Same simulate-and-extract steps as above, but with `ctx.diff_params_init`
# (the network's parameters before any gradient step) instead of the fitted
# `result.diff_params` — no retraining needed, `diff_params_init` was already
# sitting in `ctx`. Shows whether the fit actually moved sim EEG/BOLD toward
# the empirical targets, not just that the loss went down.

# %%
combined_init = eqx.combine(ctx.diff_params_init, ctx.static_params)
sim_result_eeg_init = ctx.simulators.simulator_eeg(combined_init)
sim_result_bold_init = ctx.simulators.simulator_bold(combined_init)

source_activity_init = (
    sim_result_eeg_init.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 1].T
    - sim_result_eeg_init.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 2].T
) * jnp.atleast_2d(ctx.mask_cortical).T
simulated_eeg_init = project_to_scalp(
    source_activity_init, dataset.channel_indices, ctx.leadfield, ctx.smoothing_blocks, ctx.dipole_labels
)
sim_psd_init = compute_psd(simulated_eeg_init)
sim_bold_2d_init = connectivity.extract_bold_2d(ctx.simulators.bold_monitor(sim_result_bold_init))

# %%
fig = viz.plot_eeg_psd_learning(sim_psd_init, sim_psd, ctx.target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max)
fig.savefig(out_dir / "eeg_psd_learning.png", dpi=150)

# %%
fig = viz.plot_bold_learning(sim_bold_2d_init, sim_bold_2d, ctx.sc.empirical_bold, ctx.mask_cortical,
                              cfg.tr_ms, skip_t=cfg.bold_skip_trs)
fig.savefig(out_dir / "bold_learning.png", dpi=150)
