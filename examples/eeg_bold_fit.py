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
import os

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
# --- edit these for your run ---
BIDS_ROOT = "/srv/nfs-data/sisko/christian/parrot_LEMON"  # Parrot derivatives root
subject_id = "010012"
output_root = "eeg_bold_fit_res"  # per-subject results go under <output_root>/<subject_id>

# Which loss(es) actually take gradient steps: "eeg", "bold", or "both" (the
# alternating fit). The simulator/loss not being optimized is still built (for
# the before/after diagnostics below) but never gets a gradient step.
OPTIMIZE = "both"  # "eeg" | "bold" | "both"

# BOLD loss is always a weighted combination of static FC (time-averaged) and
# dFC/FCD (windowed FC-of-FC, compared via a 1-Wasserstein distance between
# soft-histogram-summarized value distributions, since the sim/empirical BOLD
# horizons have different window counts -- see optimization.connectivity.dfc_histogram),
# both computed from the SAME simulated trajectory (see train.make_bold_loss_fn).
# Set either weight to 0 to recover a single-mode ("fc"-only or "dfc"-only) fit.
# Only matters when BOLD is optimized.
BOLD_FC_WEIGHT = 0.5
BOLD_DFC_WEIGHT = 0.5

GAMMA_WEIGHT = 0.0  # gamma loss is not used in this example
BOLD_PSD_WEIGHT = 0.0  # BOLD PSD loss is not used in this example
LEARNING_RATE = 0.01
LEARNING_RATE_BOLD = 0.01
ATLAS = 100

subject = Subject(BIDS_ROOT, subject_id)
output_root = os.path.join(output_root, f"atlas-{ATLAS}")
os.makedirs(output_root, exist_ok=True)


output_dir = os.path.join(output_root, f"{subject_id}_{OPTIMIZE}")
os.makedirs(output_dir, exist_ok=True)

cfg = config.BoldFitConfig(
    subject=subject,
    atlas = ATLAS,
    output_dir=output_dir,
    num_epochs=300,
    bold_every=2,
    optimize=OPTIMIZE,
    bold_fc_weight=BOLD_FC_WEIGHT,
    bold_dfc_weight=BOLD_DFC_WEIGHT,
    gamma_weight = GAMMA_WEIGHT,  # gamma loss is not used in this example
    learning_rate = LEARNING_RATE,
    learning_rate_bold = LEARNING_RATE_BOLD,
    bold_psd_weight = BOLD_PSD_WEIGHT,  # BOLD PSD loss is not used in this example

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
cfg_path = cfg.save()
print(f"Saved run config to {cfg_path}")

# %% [markdown]
# ## Load this subject's EEG chunks (only if EEG is actually a fit target)
# 
# Reads the subject's own splice-free EEG segments
# (`derivatives/EEG/sub-<ID>/..._task-<eeg_task>_eeg.npz` + its sidecar JSON
# for sampling rate/channel names) — a first-class Parrot output, so no more
# site-specific `all_data.pkl` schema to slice by hand.
# 
# When `OPTIMIZE == "bold"`, EEG isn't used as a fit target at all, so this is
# skipped by default -- no need to require (or wait on loading) the subject's
# EEG derivatives just to run a BOLD-only fit. You can still decide *later*,
# after the fit, to load it purely to visualize simulated-vs-real EEG -- see
# the optional EEG diagnostics section near the bottom.

# %%
LOAD_EEG = OPTIMIZE != "bold"  # flip to True to also load EEG in a bold-only run

dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length) if LOAD_EEG else None
if dataset is not None:
    print(f"Fitting {subject.subj}: {len(dataset)} chunks of {cfg.chunk_length} samples")
else:
    print(f"optimize={OPTIMIZE!r}: EEG not loaded (not a fit target). "
          "Load it later in the optional EEG diagnostics section if you want to look at it.")

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

if result.loss_history_eeg:
    print(f"Final EEG loss:  {result.loss_history_eeg[-1]:.5f}")
if result.loss_history_bold:
    print(f"Final BOLD loss: {result.loss_history_bold[-1]:.5f}")
for name, values in optimized.items():
    print(f"{name:6s} — mean {values.mean():.4f}  std {values.std():.4f}")
print(f"Saved to {out_dir}")

# %% [markdown]
# ## Diagnostics: simulation + BOLD (always available)
# 
# Doesn't need real EEG -- `sim_result_eeg`/`sim_result_bold` are just the
# fitted network's own simulator outputs, and BOLD's empirical target always
# comes from the subject's fMRI derivatives regardless of `OPTIMIZE`.

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
if cfg.bold_dfc_weight > 0:
    fig, dfc_w_dist = viz.plot_fcd_comparison(sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms,
                                               cfg.dfc_window_trs, cfg.dfc_step_trs,
                                               skip_t=cfg.bold_skip_trs, k_min=cfg.dfc_kmin,
                                               n_bins=cfg.dfc_n_bins, sigma=cfg.dfc_sigma)
    fig.savefig(out_dir / "fcd_comparison.png", dpi=150)
    print(f"dFC Wasserstein-1 distance (sim vs emp): {dfc_w_dist:.5f}")
fig, fc_corr = viz.plot_fc_comparison(sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms, skip_t=cfg.bold_skip_trs)
fig.savefig(out_dir / "fc_comparison.png", dpi=150)
print(f"FC Pearson correlation (sim vs emp): {fc_corr:.4f}")


# %% [markdown]
# ## Diagnostics: BOLD learning (first iteration vs last, always available)
# 
# `ctx.diff_params_init` (the network's parameters before any gradient step)
# was already sitting in `ctx` -- no retraining needed. Shows whether the fit
# actually moved sim BOLD toward the empirical target, not just that the loss
# went down.

# %%
combined_init = eqx.combine(ctx.diff_params_init, ctx.static_params)
sim_result_eeg_init = ctx.simulators.simulator_eeg(combined_init)
sim_result_bold_init = ctx.simulators.simulator_bold(combined_init)
sim_bold_2d_init = connectivity.extract_bold_2d(ctx.simulators.bold_monitor(sim_result_bold_init))
fig = viz.plot_bold_learning(sim_bold_2d_init, sim_bold_2d, ctx.sc.empirical_bold, ctx.mask_cortical,
                              cfg.tr_ms, skip_t=cfg.bold_skip_trs)
fig.savefig(out_dir / "bold_learning.png", dpi=150)

# %%
if cfg.bold_dfc_weight > 0:
    fig, dfc_w_dist_before, dfc_w_dist_after = viz.plot_fcd_learning(
        sim_bold_2d_init, sim_bold_2d, ctx.sc.empirical_bold, cfg.tr_ms,
        cfg.dfc_window_trs, cfg.dfc_step_trs, skip_t=cfg.bold_skip_trs,
        k_min=cfg.dfc_kmin, n_bins=cfg.dfc_n_bins, sigma=cfg.dfc_sigma,
    )
    fig.savefig(out_dir / "fcd_learning.png", dpi=150)
    print(f"dFC Wasserstein-1 distance: before={dfc_w_dist_before:.5f}  after={dfc_w_dist_after:.5f}")


# %% [markdown]
# ## Diagnostics: EEG (optional -- load real EEG now if you skipped it)
# 
# Everything below needs the subject's real EEG (to project simulated source
# activity onto the montage, and as the PSD comparison target) -- it's still
# not a fit target when `OPTIMIZE == "bold"`, purely a look at what the
# BOLD-fitted network's EEG output happens to look like. Skip this section
# entirely if you don't want it; leave `dataset` as `None` and stop here.

# %%
if dataset is None:
    dataset = data.load_subject_eeg(subject, cfg.eeg_task, cfg.chunk_length)
    print(f"Loaded {subject.subj} EEG for visualization only (still not used as a fit target)")

target_psd = ctx.target_psd if ctx.target_psd is not None else train.compute_target_psd(dataset)

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

fig = viz.plot_eeg_psd_comparison(sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max)
fig.savefig(out_dir / "eeg_psd_comparison.png", dpi=150)

# %%
fig = viz.plot_eeg_psd_comparison(sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max, log_scale=False)
fig.savefig(out_dir / "eeg_psd_comparison_linear.png", dpi=150)

# %%
fig, r_eeg = viz.plot_eeg_corr_comparison(simulated_eeg, dataset._chunks)
fig.savefig(out_dir / "eeg_corr_comparison.png", dpi=150)
print(f"EEG correlation matrix Pearson r: {r_eeg:.4f}")

# %%
source_activity_init = (
    sim_result_eeg_init.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 1].T
    - sim_result_eeg_init.ys[int(500.0 / cfg.dt)::int(4 / cfg.dt), 2].T
) * jnp.atleast_2d(ctx.mask_cortical).T
simulated_eeg_init = project_to_scalp(
    source_activity_init, dataset.channel_indices, ctx.leadfield, ctx.smoothing_blocks, ctx.dipole_labels
)
sim_psd_init = compute_psd(simulated_eeg_init)

fig = viz.plot_eeg_psd_learning(sim_psd_init, sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max)
fig.savefig(out_dir / "eeg_psd_learning.png", dpi=150)

# %%
fig = viz.plot_eeg_psd_learning(sim_psd_init, sim_psd, target_psd, ctx.freqs, ctx.idx_min, ctx.idx_max,
                                 log_scale=False)
fig.savefig(out_dir / "eeg_psd_learning_linear.png", dpi=150)


