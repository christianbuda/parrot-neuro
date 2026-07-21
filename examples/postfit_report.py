# postfit_analysis_heterogeneous.py
# Adapted from RWW postfit_analysis.py for the heterogeneous JR+WC model.
# Parameters: A, a, b (JR cortex) + P, c_ee (WC subcortex) + G (global)
from __future__ import annotations

import os
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import nibabel as nib
from nilearn import plotting
from scipy.stats import pearsonr, spearmanr, zscore
from scipy.spatial.distance import pdist, squareform

from parrot_neuro import Subject

# =========================
# Config
# =========================
subject = "010005"
BIDS_ROOT = "/srv/nfs-data/sisko/christian/parrot_LEMON"
ATLAS = 100      # connectivity parcellation -- matches optimization.config.ATLAS
FMRI_TASK = "rest"

# eeg_bold_fit.py writes results to <output_root>/<subject>; point RUNS_DIR at
# the same tree. OUT_DIR is where this report's summary figures/tables land.
RUNS_DIR = Path("eeg_bold_fit_res") / subject           # ← your fit-results root
OUT_DIR  = Path("eeg_bold_fit_res_summary") / subject   # ← where to save summary outputs
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Each subject folder must contain optimized_params.npz (saved at end of training)
# Single subject / flat layout: the npz is directly in RUNS_DIR
SUBJECTS = [""]
subject_dirs = {SUBJECTS[0]: RUNS_DIR}

# --- Missing labels + full atlas region count, both read straight off the
# subject's own fMRI derivatives (a region has no BOLD coverage <=> its row in
# the connectome-node-numbered "conn" timeseries is all-NaN) -- same logic as
# optimization.connectivity.load_structural_connectivity, so a fitted subject's
# results always expand back onto the atlas the same way they were fit.
parrot_subject = Subject(BIDS_ROOT, subject)
_ts = parrot_subject.load.fmri_timeseries(variant="conn", task=FMRI_TASK)[f"ts_{ATLAS}"]
SUBJECT_MISSING_LABELS = np.flatnonzero(np.isnan(_ts).all(axis=1))
TOTAL_ATLAS_NODES = _ts.shape[0]    # total parcels in the atlas (before dropping missing)

# Atlas config — adjust to your setup
ATLAS_PATH = Path("atlas100.nii.gz")  # ← edit: parcellation volume for brain maps

# Parameters in this model
CORTEX_PARAMS    = ["A", "a", "b"]       # JR — only meaningful on cortical nodes
SUBCORTEX_PARAMS = ["P", "c_ee"]         # WC — only meaningful on subcortical nodes
ALL_NODE_PARAMS  = CORTEX_PARAMS + SUBCORTEX_PARAMS

# Colormap for brain maps
BRAIN_CMAP = mpl.colormaps.get_cmap("RdBu_r")

# =========================
# Helpers
# =========================

def load_subject_result(subj_dir: Path) -> dict:
    """Load optimized_params.npz and return as a plain dict."""
    data = np.load(subj_dir / "optimized_params.npz", allow_pickle=True)
    return {k: data[k] for k in data.files}


def expand_to_atlas(
    param_vec: np.ndarray,
    missing_labels: np.ndarray,
    total_nodes: int,
    fill: float = np.nan,
) -> np.ndarray:
    """
    Expand a fitted param vector (length N_valid) to full atlas space
    (length total_nodes), placing NaN at missing parcel positions.

    missing_labels: 0-indexed node indices that were dropped before fitting.
    """
    kept = np.array(
        [i for i in range(total_nodes) if i not in set(missing_labels)],
        dtype=int,
    )
    assert len(param_vec) == len(kept), (
        f"param_vec length {len(param_vec)} != kept nodes {len(kept)}"
    )
    full = np.full(total_nodes, fill, dtype=float)
    full[kept] = param_vec
    return full


def fisher_mean_fc(fc_vectors: np.ndarray) -> np.ndarray:
    z = np.arctanh(np.clip(fc_vectors, -0.999999, 0.999999))
    return np.tanh(np.nanmean(z, axis=0))


def summarize_vector(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=float)
    valid = x[np.isfinite(x)]
    return {
        f"{prefix}_mean":   np.nanmean(valid),
        f"{prefix}_std":    np.nanstd(valid),
        f"{prefix}_median": np.nanmedian(valid),
        f"{prefix}_iqr":    np.nanpercentile(valid, 75) - np.nanpercentile(valid, 25),
        f"{prefix}_min":    np.nanmin(valid),
        f"{prefix}_max":    np.nanmax(valid),
        f"{prefix}_range":  np.nanmax(valid) - np.nanmin(valid),
    }


def corr_safe(x, y, method: str = "pearson") -> tuple:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok   = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    fn = pearsonr if method == "pearson" else spearmanr
    return fn(x[ok], y[ok])


def normalize(x: np.ndarray, method: str = "zscore") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if method == "minmax":
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + 1e-8)
    elif method == "zscore":
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)
    elif method == "robust":
        med = np.nanmedian(x)
        iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
        return (x - med) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown method '{method}'")


# =========================
# Load all subjects
# =========================

# Storage: one array per param, shape (n_subjects, TOTAL_ATLAS_NODES)
param_arrays = {p: [] for p in ALL_NODE_PARAMS}
#param_G      = []
loss_eeg_all  = []
loss_bold_all = []
records       = []
subject_results = {}

for subj in SUBJECTS:
    subj_dir = RUNS_DIR / subj
    res      = load_subject_result(subj_dir)
    subject_results[subj] = res

    # --- Missing labels for this subject (derived once, above, from its own
    # fMRI derivatives via parrot_neuro.Subject) ---
    missing_labels = SUBJECT_MISSING_LABELS

    # --- Load fitted node-wise params (length N_valid) ---
    A_fit    = np.asarray(res["A"],    dtype=float)
    a_fit    = np.asarray(res["a"],    dtype=float)
    b_fit    = np.asarray(res["b"],    dtype=float)
    P_fit    = np.asarray(res["P"],    dtype=float)
    c_ee_fit = np.asarray(res["c_ee"], dtype=float)
    #G        = float(res["G"])

    # --- Expand to full atlas space ---
    A_full    = expand_to_atlas(A_fit,    missing_labels, TOTAL_ATLAS_NODES)
    a_full    = expand_to_atlas(a_fit,    missing_labels, TOTAL_ATLAS_NODES)
    b_full    = expand_to_atlas(b_fit,    missing_labels, TOTAL_ATLAS_NODES)
    P_full    = expand_to_atlas(P_fit,    missing_labels, TOTAL_ATLAS_NODES)
    c_ee_full = expand_to_atlas(c_ee_fit, missing_labels, TOTAL_ATLAS_NODES)

    param_arrays["A"].append(A_full)
    param_arrays["a"].append(a_full)
    param_arrays["b"].append(b_full)
    param_arrays["P"].append(P_full)
    param_arrays["c_ee"].append(c_ee_full)
    #param_G.append(G)

    # --- Loss histories ---
    if "loss_eeg" in res:
        loss_eeg_all.append(np.asarray(res["loss_eeg"], dtype=float))
    if "loss_bold" in res:
        loss_bold_all.append(np.asarray(res["loss_bold"], dtype=float))

    # --- Per-subject summary row ---
    row = {"subject": subj, "n_valid_nodes": len(A_fit)}
    row.update(summarize_vector(A_fit,    "A"))
    row.update(summarize_vector(a_fit,    "a"))
    row.update(summarize_vector(b_fit,    "b"))
    row.update(summarize_vector(P_fit,    "P"))
    row.update(summarize_vector(c_ee_fit, "c_ee"))
    if "loss_eeg" in res:
        row["final_loss_eeg"]  = float(res["loss_eeg"][-1])
    if "loss_bold" in res:
        row["final_loss_bold"] = float(res["loss_bold"][-1])
    records.append(row)

# Stack into (n_subjects, TOTAL_ATLAS_NODES)
for p in ALL_NODE_PARAMS:
    param_arrays[p] = np.vstack(param_arrays[p])
#param_G = np.asarray(param_G)

df_subject = pd.DataFrame.from_records(records)
df_subject.to_csv(OUT_DIR / "subject_parameter_summary.csv", index=False)
print(df_subject)

# =========================
# Node summary (mean/std across subjects)
# =========================

node_summary_dict = {"node": np.arange(TOTAL_ATLAS_NODES)}
for p in ALL_NODE_PARAMS:
    node_summary_dict[f"{p}_mean"] = np.nanmean(param_arrays[p], axis=0)
    node_summary_dict[f"{p}_std"]  = np.nanstd(param_arrays[p],  axis=0)
    node_summary_dict[f"{p}_cv"]   = (
        np.nanstd(param_arrays[p], axis=0)
        / (np.nanmean(param_arrays[p], axis=0) + 1e-8)
    )

node_summary = pd.DataFrame(node_summary_dict)
node_summary.to_csv(OUT_DIR / "node_parameter_summary.csv", index=False)

# =========================
# 1) Loss curves
# =========================

if loss_eeg_all:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=150)

    ax = axes[0]
    for subj, curve in zip(SUBJECTS, loss_eeg_all):
        ax.plot(curve, alpha=0.6, linewidth=1, label=subj)
    ax.set_title("EEG PSD loss per subject")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)

    ax = axes[1]
    for subj, curve in zip(SUBJECTS, loss_bold_all):
        # BOLD is logged every BOLD_EVERY steps so x-axis is BOLD step number
        ax.plot(curve, alpha=0.6, linewidth=1, label=subj)
    ax.set_title("BOLD FC loss per subject")
    ax.set_xlabel("BOLD step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curves.png")
    plt.close()

# =========================
# 2) Subject-level parameter distributions
# =========================

fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150)
plot_items = [
    # (param_G,                    "Global G"),
    (df_subject["A_mean"],       "Mean A (JR EPSP amplitude)"),
    (df_subject["a_mean"],       "Mean a (JR excit. time const.)"),
    (df_subject["b_mean"],       "Mean b (JR inhib. time const.)"),
    (df_subject["P_mean"],       "Mean P (WC external input)"),
    (df_subject["c_ee_mean"],    "Mean c_ee (WC local E-E coupling)"),
]
for ax, (col, title) in zip(axes.ravel(), plot_items):
    ax.hist(np.asarray(col, dtype=float), bins=max(5, len(SUBJECTS) // 2), alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)

plt.suptitle("Parameter distributions across subjects", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "subject_parameter_distributions.png")
plt.close()

# =========================
# 3) Nodewise mean ± std profiles
# =========================

n_params = len(ALL_NODE_PARAMS)
fig, axes = plt.subplots(n_params, 1, figsize=(14, 3 * n_params), dpi=150, sharex=True)

param_labels = {
    "A":    "A — JR EPSP amplitude",
    "a":    "a — JR excit. time const.",
    "b":    "b — JR inhib. time const.",
    "P":    "P — WC external input",
    "c_ee": "c_ee — WC local E-E coupling",
}

for ax, p in zip(axes, ALL_NODE_PARAMS):
    mu = np.nanmean(param_arrays[p], axis=0)
    sd = np.nanstd(param_arrays[p],  axis=0)
    x  = np.arange(TOTAL_ATLAS_NODES)
    ax.plot(x, mu, lw=1.5)
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.25)
    ax.set_ylabel(param_labels[p], fontsize=8)
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Node (atlas index)")
plt.suptitle("Nodewise parameter mean ± std across subjects", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "nodewise_parameter_mean_sd.png")
plt.close()

# =========================
# 4) Subject × node heatmaps
# =========================

fig, axes = plt.subplots(n_params, 1, figsize=(14, 3 * n_params), dpi=150, sharex=True)

for ax, p in zip(axes, ALL_NODE_PARAMS):
    im = ax.imshow(param_arrays[p], aspect="auto", interpolation="nearest", cmap="RdBu_r")
    ax.set_ylabel("Subject")
    ax.set_title(f"{p} — subject × node", fontsize=9)
    ax.set_yticks(range(len(SUBJECTS)))
    ax.set_yticklabels(SUBJECTS, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

axes[-1].set_xlabel("Node (atlas index)")
plt.tight_layout()
plt.savefig(OUT_DIR / "subject_node_parameter_heatmaps.png")
plt.close()

# =========================
# 5) Inter-parameter correlations (nodewise, group mean)
# =========================

# Pairwise scatter of group-mean node params
from itertools import combinations

pairs = list(combinations(ALL_NODE_PARAMS, 2))
n_pairs = len(pairs)
fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4), dpi=150)
if n_pairs == 1:
    axes = [axes]

for ax, (p1, p2) in zip(axes, pairs):
    mu1 = np.nanmean(param_arrays[p1], axis=0)
    mu2 = np.nanmean(param_arrays[p2], axis=0)
    ok  = np.isfinite(mu1) & np.isfinite(mu2)
    ax.scatter(mu1[ok], mu2[ok], s=15, alpha=0.6)
    r, pv = corr_safe(mu1, mu2, "spearman")
    ax.set_xlabel(f"Mean {p1}", fontsize=8)
    ax.set_ylabel(f"Mean {p2}", fontsize=8)
    ax.set_title(f"rho={r:.2f}, p={pv:.3g}", fontsize=8)
    ax.grid(alpha=0.2)

plt.suptitle("Nodewise inter-parameter correlations (group mean)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "nodewise_interparam_correlations.png")
plt.close()

# =========================
# 6) Brain spatial maps
# =========================

def plot_param_brain_map(
    node_summary: pd.DataFrame,
    atlas_path,
    param: str = "A",
    stat: str  = "mean",
    out_dir: Path = OUT_DIR,
    normalize_method: str = "zscore",
    skip_labels: list = None,
):
    """Map a per-node parameter statistic onto the volumetric atlas."""
    col = f"{param}_{stat}"
    assert col in node_summary.columns, (
        f"Column '{col}' not found. Available: {list(node_summary.columns)}"
    )

    node_values = node_summary[col].values.copy()   # (TOTAL_ATLAS_NODES,)
    valid_mask  = np.isfinite(node_values)
    node_values[valid_mask] = normalize(node_values[valid_mask], method=normalize_method)

    out_dir     = Path(out_dir)
    skip_labels = set(skip_labels or [0])

    atlas_img  = nib.load(atlas_path)
    atlas_data = np.asarray(atlas_img.dataobj, dtype=float)
    parcel_ids = np.unique(atlas_data[atlas_data > 0]).astype(int)

    # Default mapping: parcel ID k (1-indexed) → node index k-1
    n_mapped = n_skipped = 0
    mapped   = np.zeros_like(atlas_data)

    for pid in parcel_ids:
        if pid in skip_labels:
            n_skipped += 1
            continue
        node_idx = int(pid) - 1    # 1-indexed parcel → 0-indexed node
        if node_idx < 0 or node_idx >= len(node_values):
            print(f"  [skip] parcel {pid}: out of range")
            n_skipped += 1
            continue
        val = node_values[node_idx]
        if not np.isfinite(val):   # missing node — leave transparent
            n_skipped += 1
            continue
        mapped[atlas_data == pid] = val
        n_mapped += 1

    print(f"  [{param} {stat}] Mapped {n_mapped} parcels, skipped/missing {n_skipped}.")

    mapped_img = nib.Nifti1Image(mapped, atlas_img.affine, atlas_img.header)
    nib.save(mapped_img, out_dir / f"{param}_{stat}_brain_map.nii.gz")

    display = plotting.plot_glass_brain(
        mapped_img, display_mode="lyrz", colorbar=True,
        title=f"Group {stat} of {param}",
        cmap=BRAIN_CMAP, symmetric_cbar=True,
    )
    display.savefig(out_dir / f"{param}_{stat}_glass_brain.png")
    display.close()

    display = plotting.plot_stat_map(
        mapped_img, display_mode="z", cut_coords=8, colorbar=True,
        title=f"Group {stat} of {param}",
        cmap=BRAIN_CMAP, symmetric_cbar=True,
    )
    display.savefig(out_dir / f"{param}_{stat}_axial_slices.png")
    display.close()


for param in ALL_NODE_PARAMS:
    for stat in ["mean", "std"]:
        print(f"\nPlotting brain map: {param} {stat}...")
        plot_param_brain_map(
            node_summary=node_summary,
            atlas_path=ATLAS_PATH,
            param=param,
            stat=stat,
            out_dir=OUT_DIR,
            normalize_method="zscore",
            skip_labels=[0],
        )

# =========================
# 7) G vs final losses scatter
# =========================

if "final_loss_eeg" in df_subject.columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    ax = axes[0]
    ax.scatter(df_subject["G"], df_subject["final_loss_eeg"], s=40)
    r, pv = corr_safe(df_subject["G"], df_subject["final_loss_eeg"], "spearman")
    ax.set_xlabel("G (global coupling)")
    ax.set_ylabel("Final EEG loss")
    ax.set_title(f"G vs EEG loss | rho={r:.2f}, p={pv:.3g}")
    ax.grid(alpha=0.2)

    if "final_loss_bold" in df_subject.columns:
        ax = axes[1]
        ax.scatter(df_subject["G"], df_subject["final_loss_bold"], s=40)
        r, pv = corr_safe(df_subject["G"], df_subject["final_loss_bold"], "spearman")
        ax.set_xlabel("G (global coupling)")
        ax.set_ylabel("Final BOLD FC loss")
        ax.set_title(f"G vs BOLD loss | rho={r:.2f}, p={pv:.3g}")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "G_vs_losses.png")
    plt.close()

print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")