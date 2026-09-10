#!/usr/bin/env python
"""Build a parameter-geography + learning report from a saved TVB-optim fit.

Given the output directory of one ``eeg_bold_fit_cli.py`` run (the folder
holding ``optimized_params.npz``, ``config.json``, ``diagnostics_metrics.json``
and the ``loss_history_*.npy`` files), this writes, into that same folder:

  - ``params_report.md`` -- a written report (learning summary + per-parameter
    spatial/statistical summary)
  - ``loss_curves.png``, ``param_distributions.png``, ``param_group_hemisphere.png``,
    ``param_brain_<PARAM>.png`` (one per per-node parameter) -- static figures
    for the markdown report
  - ``artifact.html`` -- a single self-contained interactive HTML page (an
    explorable, hoverable version of the same content, with whichever existing
    ``diagnostics.run_and_save`` PNGs it finds embedded) -- open it directly in
    a browser, or hand it to Claude Code to publish as an Artifact.

No re-simulation happens here (unlike ``postfit_diagnostics_cli.py``, which
rebuilds the network) -- this only reads what a prior fit run already saved,
so it needs no jax/GPU and runs in seconds. It DOES need the lightweight,
jax-free ``parrot_neuro`` core (for ``Subject.load.connectivity_labels`` and
``Subject.load.dipoles``/``dipole_node_labels``, to recover each optimized
node's region name/order and its real forward-model source points for the
glass-brain maps -- see ``write_glass_brain_plots``).

    pixi run python examples/params_report_cli.py \\
        --results-dir "eeg_bold_fit_res_balloon/atlas-100/010002_both"

The subject, bids_root and atlas are all read back out of that run's own
``config.json`` -- no need to pass them again.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk

from parrot_neuro import Subject

# VTK is noisy when it probes EGL/GLX before falling back to software rendering
# (offscreen, no GPU/X server needed here) -- silence the non-fatal warnings.
vtk.vtkObject.GlobalWarningDisplayOff()

# Named cameras: (offset direction of the camera from the scene centre, view-up),
# RAS world frame -- same convention as containers/parrot_qc/qc/render3d.py's
# `_CAMERAS` (pyvista's view_vector(v) puts the camera on the +v side looking
# back at the focal point, so e.g. anterior=(0,1,0) is a true face-on view).
DIPOLE_CAMERAS = {
    "left": ((-1, 0, 0), (0, 0, 1)),
    "anterior": ((0, 1, 0), (0, 0, 1)),
    "superior": ((0, 0, 1), (0, 1, 0)),
    "right": ((1, 0, 0), (0, 0, 1)),
}
DIPOLE_VIEWS = ("left", "anterior", "superior", "right")

# ---------------------------------------------------------------------------
# Classification: which functional group / hemisphere / node-type a connectome
# region belongs to, from its label name alone. This is a name-based PROXY for
# the pipeline's real orient-type-derived `mask_cortical` (pipeline.py:166,
# `np.isin(orient_atlas, ['N','G','P'])`) -- but an exact one for the parts
# that matter here: `is_surface` (see SURFACE_GROUPS below) predicts EXACTLY
# which nodes have a live gradient path for each parameter, confirmed by the
# fitted values themselves (a Wilson-Cowan-only param like `c_ee` sits at
# *exact* init on every node classified as surface, every time).
SURFACE_GROUPS = {
    "Vis", "VisCent", "VisPeri", "SomMot", "DorsAttn", "SalVentAttn", "Limbic",
    "Cont", "Default", "TempPar", "Cerebellum", "Hippocampus",
}

# which node type each per-node param's dynamics actually depend on (see
# optimization/model.py: HeterogeneousModel.dynamics -- A/B/a/b/mu only enter
# d_jr [surface/JR nodes], P/c_ee only enter d_wc [volumetric/WC nodes]).
LIVE_ON = {
    "A": "surface", "B": "surface", "a": "surface", "b": "surface", "mu": "surface",
    "P": "volumetric", "c_ee": "volumetric",
}
PARAM_UNITS = {
    "A": "mV", "B": "mV", "a": "1/ms", "b": "1/ms", "mu": "firing rate (1/ms)",
    "P": "external input (a.u.)", "c_ee": "coupling strength (a.u.)",
}
PER_NODE_PARAMS = ["P", "c_ee", "A", "B", "a", "b", "mu"]

GROUP_ORDER = ["VisCent", "VisPeri", "SomMot", "DorsAttn", "SalVentAttn", "Limbic",
               "Cont", "Default", "TempPar", "Cerebellum", "Hippocampus",
               "BasalGanglia", "Thalamus", "Brainstem"]

# candidate diagnostics.run_and_save() figures to embed in the artifact, in
# display order, with a caption template ({metrics} filled from
# diagnostics_metrics.json at generation time). Only the ones that actually
# exist in --results-dir are embedded -- a run with different config (e.g.
# bold_dfc_weight=0, which skips fcd_*.png) still gets a report.
CANDIDATE_FIGURES = [
    ("eeg_psd_learning.png", "wide" if False else "",
     "<b>EEG power spectrum, before vs after.</b> How close the dotted (init) "
     "and solid (fitted) lines are shows how much training has moved the "
     "model's spectral content so far."),
    ("eeg_corr_comparison.png", "",
     "<b>Simulated vs empirical EEG, per-channel.</b> Spectral correlation: "
     "<span class=\"mono\">{eeg_corr:.3f}</span>."),
    ("fc_comparison.png", "wide",
     "<b>Static functional connectivity, simulated vs empirical.</b> "
     "<span class=\"mono\">fc_corr = {fc_corr:.3f}</span>."),
    ("bold_learning.png", "wide",
     "<b>Simulated BOLD timeseries, before vs after, for representative nodes.</b> "
     "How closely the solid (after) trace tracks the dashed (before) one shows "
     "how much training has moved this network's actual output."),
    ("fcd_learning.png", "wide",
     "<b>Dynamic FC (FCD), before vs after training.</b> dFC Wasserstein "
     "distance: <span class=\"mono\">{dfc_w_dist_before:.4f} &rarr; {dfc_w_dist_after:.4f}</span>."),
    ("node_activity.png", "wide",
     "<b>Raw simulated node activity</b> for a handful of representative cortical/subcortical nodes."),
]


def classify(name: str) -> dict:
    """hemisphere + functional group + cortical/surface flags, from the label
    name alone."""
    is_schaefer = "Networks_" in name
    if is_schaefer:
        m = re.match(r"\d+Networks_(LH|RH)_([A-Za-z]+)", name)
        hemi = "L" if m.group(1) == "LH" else "R"
        group = re.sub(r"[ABC]$", "", m.group(2))
    else:
        hemi = "L" if name.startswith("Left-") or name.startswith("cerebellum_L") else (
            "R" if name.startswith("Right-") or name.startswith("cerebellum_R") else "M"
        )
        if any(s in name for s in ("Caudate", "Putamen", "Pallidum", "Accumbens")):
            group = "BasalGanglia"
        elif any(s in name for s in ("subiculum", "CA1", "CA2", "CA3", "CA4", "dentate_gyrus")):
            group = "Hippocampus"
        elif "cerebellum" in name:
            group = "Cerebellum"
        elif name in ("Midbrain", "Pons", "Medulla"):
            group = "Brainstem"
        else:
            group = "Thalamus"  # remaining named nuclei (LD, LGN, MDl, PuL, VA, VPL, ...)
    return {
        "hemi": hemi, "group": group,
        "is_cortical": is_schaefer,
        "is_surface": group in SURFACE_GROUPS,
    }


def load_run(results_dir: Path) -> tuple[dict, Subject]:
    """Read everything a prior eeg_bold_fit_cli.py run saved into
    results_dir, plus the region labels via the (jax-free) Subject API.
    Returns ``(data, subject)`` -- ``subject`` isn't JSON-serializable so it
    travels separately from ``data`` (which does get json.dumps'd, for the
    interactive artifact)."""
    config = json.loads((results_dir / "config.json").read_text())
    subject = Subject(config["subject"]["bids_root"], config["subject"]["subject"])
    atlas = config["atlas"]
    labels = subject.load.connectivity_labels(atlas, fmri_aligned=True)

    npz = np.load(results_dir / "optimized_params.npz")
    missing = [p for p in PER_NODE_PARAMS if p not in npz.files]
    if missing:
        raise KeyError(f"optimized_params.npz is missing {missing} -- not a per-node-param fit?")
    if npz[PER_NODE_PARAMS[0]].shape != (len(labels),):
        raise ValueError(
            f"{len(labels)} region labels for atlas {atlas} but "
            f"{npz[PER_NODE_PARAMS[0]].shape[0]} optimized nodes -- labels/params disagree "
            f"(wrong bids_root/subject/atlas in config.json, or a stale connectivity derivative?)"
        )

    init_by_param = {p["name"]: p["init"] for p in config["learnable_params"]}
    bounds_by_param = {p["name"]: (p["low"], p["high"]) for p in config["learnable_params"]}

    nodes = []
    for i, name in enumerate(labels):
        entry = {"idx": i, "name": name, **classify(name)}
        for p in PER_NODE_PARAMS:
            entry[p] = float(npz[p][i])
        nodes.append(entry)

    diagnostics_path = results_dir / "diagnostics_metrics.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}

    data = {
        "subject": config["subject"]["subject"],
        "atlas": atlas,
        "spacing": float(config["spacing"]),
        "fmri_task": config["fmri_task"],
        "config": config,
        "nodes": nodes,
        "per_node_params": PER_NODE_PARAMS,
        "init_by_param": init_by_param,
        "bounds_by_param": bounds_by_param,
        "live_on": LIVE_ON,
        "param_units": PARAM_UNITS,
        "G": float(npz["G"][0]),
        "G_init": init_by_param.get("G"),
        "G_bounds": bounds_by_param.get("G"),
        "loss_eeg": npz["loss_eeg"].tolist() if "loss_eeg" in npz.files else [],
        "loss_bold": npz["loss_bold"].tolist() if "loss_bold" in npz.files else [],
        "eeg_loss_ratio": float(npz["eeg_loss_ratio"]) if "eeg_loss_ratio" in npz.files else None,
        "bold_loss_ratio": float(npz["bold_loss_ratio"]) if "bold_loss_ratio" in npz.files else None,
        "diagnostics_metrics": diagnostics,
        "num_epochs": config["num_epochs"],
        "bold_every": config["bold_every"],
    }
    return data, subject


# ---------------------------------------------------------------------------
# static matplotlib figures

def write_static_plots(data: dict, out_dir: Path, subject: Subject) -> None:
    nodes = data["nodes"]
    per_node_params = data["per_node_params"]
    LIVE_COLOR, DORMANT_COLOR = "#2b6cb0", "#a0aec0"
    HEMI_COLORS = {"L": "#d69e2e", "R": "#38a169", "M": "#805ad5"}

    # -- 1. loss curves --------------------------------------------------
    if data["loss_eeg"] or data["loss_bold"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if data["loss_eeg"]:
            le = data["loss_eeg"]
            ep = np.arange(1, len(le) + 1)
            axes[0].plot(ep, le, "o-", color=LIVE_COLOR)
            axes[0].set_xlabel("epoch"); axes[0].set_ylabel("EEG loss"); axes[0].set_xticks(ep)
            axes[0].set_title(f"EEG loss  ({(1 - le[-1] / le[0]) * 100:.1f}% drop / {len(le)} epochs)")
            axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if data["loss_bold"]:
            lb = data["loss_bold"]
            ep = np.arange(data["bold_every"], data["bold_every"] * len(lb) + 1, data["bold_every"])
            axes[1].plot(ep, lb, "o-", color="#c05621")
            axes[1].set_xlabel("epoch"); axes[1].set_ylabel("BOLD loss"); axes[1].set_xticks(ep)
            axes[1].set_title(f"BOLD loss  ({(1 - lb[-1] / lb[0]) * 100:.2f}% drop / every {data['bold_every']} epochs)")
        fig.suptitle(f"sub-{data['subject']} atlas-{data['atlas']} -- training loss "
                     f"over {data['num_epochs']} epochs ({data['config']['schedule']})")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / "loss_curves.png", dpi=130)
        plt.close(fig)

    # -- 2. per-param distributions ---------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for i, p in enumerate(per_node_params):
        ax = axes[i]
        live_kind = data["live_on"][p]
        live_vals = np.array([n[p] for n in nodes if n["is_surface"] == (live_kind == "surface")])
        dorm_vals = np.array([n[p] for n in nodes if n["is_surface"] != (live_kind == "surface")])
        init = data["init_by_param"][p]
        bins = np.linspace(min(live_vals.min(), dorm_vals.min(), init) - 1e-6,
                            max(live_vals.max(), dorm_vals.max(), init) + 1e-6, 40)
        other_kind = "volumetric" if live_kind == "surface" else "surface"
        ax.hist(dorm_vals, bins=bins, color=DORMANT_COLOR, alpha=0.85, label=f"dormant ({other_kind}, n={len(dorm_vals)})")
        ax.hist(live_vals, bins=bins, color=LIVE_COLOR, alpha=0.85, label=f"live ({live_kind}, n={len(live_vals)})")
        ax.axvline(init, color="crimson", linestyle="--", linewidth=1.2, label=f"init={init:g}")
        ax.set_title(f"{p}  [{data['param_units'][p]}]", fontsize=11)
        ax.set_xlabel(p); ax.set_ylabel("# nodes"); ax.legend(fontsize=7, loc="upper right")
    for j in range(len(per_node_params), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"sub-{data['subject']} atlas-{data['atlas']} -- per-node parameter distributions "
                 "(live vs dormant node type)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "param_distributions.png", dpi=130)
    plt.close(fig)

    # -- 3. group x hemisphere summary ------------------------------------
    groups_present = [g for g in GROUP_ORDER if any(n["group"] == g for n in nodes)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.ravel()
    for i, p in enumerate(per_node_params):
        ax = axes[i]
        init = data["init_by_param"][p]
        xs = np.arange(len(groups_present))
        for side, dx in [("L", -0.2), ("R", 0.2)]:
            means, errs = [], []
            for g in groups_present:
                vv = [n[p] for n in nodes if n["group"] == g and n["hemi"] == side]
                means.append(np.mean(vv) - init if vv else np.nan)
                errs.append(np.std(vv) if vv else 0)
            ax.bar(xs + dx, means, width=0.38, yerr=errs, capsize=2, color=HEMI_COLORS[side], alpha=0.85, label=f"hemi {side}")
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1)
        ax.set_xticks(xs); ax.set_xticklabels(groups_present, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(f"{p} - init"); ax.set_title(p, fontsize=11)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(per_node_params), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"sub-{data['subject']} atlas-{data['atlas']} -- mean deviation from init, "
                 "by functional group and hemisphere", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "param_group_hemisphere.png", dpi=130)
    plt.close(fig)

    # -- 4. glass-brain maps, real subject dipole cloud, one PNG per param ----
    write_glass_brain_plots(data, out_dir, subject)


def write_glass_brain_plots(data: dict, out_dir: Path, subject: Subject) -> dict[str, Path]:
    """One PyVista glass-brain PNG per per-node parameter, colored by this
    subject's actual forward-model dipole cloud -- not centroids or an
    approximate mesh: the exact ~38k source points this fit's leadfield
    projects through, in this subject's own native space (so there is no
    coregistration to get wrong -- the dipoles ARE where this fit's sources
    live). Dense enough (2mm Poisson-disk spacing) that spheres at each point
    read as a continuous colored surface/volume, not a scatter of dots.

    Live nodes render as their value (viridis); dormant-node dipoles render
    as small pale-gray points underneath. 4-view montage (left/anterior/
    superior/right), same named-camera convention as
    containers/parrot_qc/qc/render3d.py's `_CAMERAS`.
    """
    nodes = data["nodes"]
    is_surface_by_node = np.array([n["is_surface"] for n in nodes])
    values_by_node = {p: np.array([n[p] for n in nodes]) for p in data["per_node_params"]}

    dipoles = subject.load.dipoles(data["spacing"])
    node_idx = subject.load.dipole_node_labels(data["atlas"], data["spacing"], task=data["fmri_task"])
    positions = dipoles.positions
    valid = node_idx >= 0
    node_idx_c = node_idx.clip(min=0)
    print(f"  dipole cloud: {len(positions)} points ({valid.sum()} mapped to a node) at "
          f"{data['spacing']}mm spacing")

    written = {}
    for p in data["per_node_params"]:
        live_kind = data["live_on"][p]
        is_live_by_node = is_surface_by_node == (live_kind == "surface")
        live_mask = valid & is_live_by_node[node_idx_c]
        dormant_mask = valid & ~is_live_by_node[node_idx_c]
        scalars = values_by_node[p][node_idx_c[live_mask]]
        vmin, vmax = float(scalars.min()), float(scalars.max())
        if vmax - vmin < 1e-9:
            vmax = vmin + 1e-9

        n = len(DIPOLE_VIEWS)
        pl = pv.Plotter(off_screen=True, shape=(1, n), window_size=(480 * n + 90, 480), border=False)
        for i, view in enumerate(DIPOLE_VIEWS):
            pl.subplot(0, i)
            if dormant_mask.any():
                pl.add_points(positions[dormant_mask], color="#b8c2bb", opacity=0.35,
                               render_points_as_spheres=True, point_size=3, lighting=False)
            pl.add_points(positions[live_mask], scalars=scalars, cmap="viridis", clim=(vmin, vmax),
                          render_points_as_spheres=True, point_size=7, lighting=True,
                          show_scalar_bar=(i == n - 1),
                          scalar_bar_args={"title": f"{p} [{data['param_units'][p]}]", "color": "black",
                                            "vertical": True, "width": 0.09, "height": 0.6,
                                            "position_x": 0.87, "position_y": 0.2, "title_font_size": 16,
                                            "label_font_size": 14, "fmt": "%.3g"})
            pl.set_background("white")
            offset, viewup = DIPOLE_CAMERAS[view]
            pl.view_vector(offset, viewup=viewup)
            pl.reset_camera()
            pl.add_text(view, font_size=14, color="black", position="lower_left")
        # Short in-image label only -- the full stats (live n, max |delta init|, etc.)
        # live in the surrounding report text/info panel, not baked into the pixels.
        pl.subplot(0, 0)
        pl.add_text(f"sub-{data['subject']} -- {p}", font_size=13, color="black", position="upper_left")
        out_path = out_dir / f"param_brain_{p}.png"
        pl.screenshot(str(out_path))
        pl.close()
        written[p] = out_path
    return written


# ---------------------------------------------------------------------------
# markdown report

def write_markdown(data: dict, out_dir: Path) -> None:
    nodes = data["nodes"]
    n_surface = sum(n["is_surface"] for n in nodes)
    n_vol = len(nodes) - n_surface
    dm = data["diagnostics_metrics"]

    lines = [
        f"# sub-{data['subject']} (atlas-{data['atlas']}) -- fit report",
        "",
        f"TVB-optim joint EEG+BOLD fit of a heterogeneous Jansen-Rit (cortex) / "
        f"Wilson-Cowan (subcortex) network, `{data['config']['leadfield_label']}` leadfield, "
        f"`{data['config']['bold_model']}` BOLD forward model. {len(nodes)} optimized nodes "
        f"({n_surface} surface-reconstructed, {n_vol} volumetric).",
        "",
        f"**{data['num_epochs']}-epoch run.** Read every finding below as \"which direction "
        "did these gradient steps push things\", scaled to how few/many epochs this was.",
        "",
        "## Learning",
        "",
    ]
    if data["loss_eeg"]:
        le = data["loss_eeg"]
        lines.append(f"- EEG loss: {le[0]:.4e} -> {le[-1]:.4e}  ({(1 - le[-1] / le[0]) * 100:+.2f}%)")
    if data["loss_bold"]:
        lb = data["loss_bold"]
        lines.append(f"- BOLD loss: {lb[0]:.4e} -> {lb[-1]:.4e}  ({(1 - lb[-1] / lb[0]) * 100:+.3f}%)")
    if dm:
        lines.append(f"- EEG spectral correlation: {dm.get('eeg_corr', float('nan')):.3f}")
        lines.append(f"- Static FC correlation: {dm.get('fc_corr', float('nan')):.4f}")
        if "dfc_w_dist_before" in dm and "dfc_w_dist_after" in dm:
            lines.append(f"- dFC Wasserstein distance: {dm['dfc_w_dist_before']:.4f} -> {dm['dfc_w_dist_after']:.4f}")
    lines.append(f"- Global coupling G: {data['G']:.4f} (init {data['G_init']}, bounds {data['G_bounds']})")
    lines += ["", "![loss curves](loss_curves.png)" if (out_dir / "loss_curves.png").exists() else "", ""]

    lines += [
        "## Parameter variation across the brain",
        "",
        "**Each per-node parameter is only dynamically active -- and therefore only "
        "trained -- on one of the two node types.** `A, B, a, b, mu` (Jansen-Rit) are "
        "live on surface-reconstructed nodes (neocortex + cerebellum + hippocampus); "
        "`P, c_ee` (Wilson-Cowan) are live on purely volumetric nodes (thalamus, basal "
        "ganglia, brainstem). Values for the *other* node type sit at (or very near) "
        "config `init` and should be read as noise, not signal.",
        "",
        "![param distributions](param_distributions.png)",
        "",
        "| param | live on | live n | max abs deviation from init | as % of allowed range |",
        "|---|---|---|---|---|",
    ]
    for p in data["per_node_params"]:
        init = data["init_by_param"][p]
        lo, hi = data["bounds_by_param"][p]
        live_kind = data["live_on"][p]
        live_vals = np.array([n[p] for n in nodes if n["is_surface"] == (live_kind == "surface")])
        max_dev = float(np.abs(live_vals - init).max()) if len(live_vals) else 0.0
        lines.append(f"| {p} | {live_kind} | {len(live_vals)} | {max_dev:.5f} | {max_dev / (hi - lo) * 100:.2f}% |")

    lines += [
        "",
        "![param by group and hemisphere](param_group_hemisphere.png)",
        "",
        "### Glass-brain maps (real subject anatomy)",
        "",
        "Each parameter's fitted value at this subject's own forward-model dipole "
        "cloud -- the exact source points this fit's leadfield projects through "
        "(`Subject.load.dipoles`, ~2mm Poisson-disk spacing, dense enough that the "
        "rendered spheres read as a continuous colored surface/volume, not a scatter "
        "of dots). Node coregistration is exact by construction: these are literally "
        "where this fit's sources live, not an approximation. Colored points are the "
        "live node type for that parameter; small pale points are the dormant one.",
        "",
    ]
    for p in data["per_node_params"]:
        img_name = f"param_brain_{p}.png"
        if (out_dir / img_name).exists():
            lines += [f"**{p}**", "", f"![{p} glass brain]({img_name})", ""]

    lines += [
        "## Caveats",
        "",
        f"- {data['num_epochs']} epochs -- treat findings as directional, not converged.",
        "- The live/dormant split is architectural fact (each parameter's own equations "
        "only touch one node type -- see `optimization/model.py`'s `HeterogeneousModel.dynamics`); "
        "confirm it holds for this run by checking that the *dormant* rows above have "
        "~0% deviation. If they don't, this run's node classification (or its model) may "
        "differ from what this script assumes.",
        "- Glass-brain dipole positions are this subject's own native scanner space "
        "(not MNI) -- not directly comparable pixel-for-pixel across subjects, but "
        "exactly where this fit's sources live for *this* subject.",
    ]
    (out_dir / "params_report.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# interactive artifact

_ARTIFACT_TEMPLATE = None  # filled in below, at module scope, from the .html.in-style string


def write_artifact_html(data: dict, out_dir: Path, out_path: Path) -> None:
    nodes = data["nodes"]
    n_surface = sum(n["is_surface"] for n in nodes)
    n_vol = len(nodes) - n_surface
    n_brainstem = sum(n["group"] == "Brainstem" for n in nodes)
    dm = data["diagnostics_metrics"]

    figures_html = []
    for fname, cls, caption_tpl in CANDIDATE_FIGURES:
        fpath = out_dir / fname
        if not fpath.exists():
            continue
        try:
            caption = caption_tpl.format(**dm)
        except (KeyError, IndexError):
            caption = re.sub(r"\{[^}]*\}", "n/a", caption_tpl)
        b64img = base64.b64encode(fpath.read_bytes()).decode("ascii")
        figures_html.append(
            f'<div class="card {cls}"><figure><img src="data:image/png;base64,{b64img}" alt="{fname}">'
            f'<figcaption>{caption}</figcaption></figure></div>'
        )
    figures_html = "\n".join(figures_html) or "<p style='color:var(--ink-muted)'>No diagnostics_*.png figures found next to optimized_params.npz.</p>"

    brain_images = {}
    for p in data["per_node_params"]:
        fpath = out_dir / f"param_brain_{p}.png"
        if fpath.exists():
            brain_images[p] = "data:image/png;base64," + base64.b64encode(fpath.read_bytes()).decode("ascii")

    callout = (
        f"The model is heterogeneous: cortex, cerebellum and hippocampus run Jansen-Rit "
        f"dynamics; thalamus, basal ganglia and brainstem run Wilson-Cowan. Each parameter's "
        f"equations only touch <i>one</i> of those two node types (see "
        f"<span class=\"mono\">HeterogeneousModel.dynamics</span> in "
        f"<span class=\"mono\">optimization/model.py</span>) &mdash; so on the other type it "
        f"has no gradient path at all and just sits at its config <span class=\"mono\">init</span>. "
        f"That's directly visible in the fitted values here: any Wilson-Cowan-only param sits at "
        f"exact init on all {n_surface} surface nodes"
        + (f", and the {n_brainstem} brainstem node{'s' if n_brainstem != 1 else ''} "
           f"stay{'s' if n_brainstem == 1 else ''} frozen at exact init on every Jansen-Rit param"
           if n_brainstem else "")
        + ". Colored points on the glass brain below are the <b>live</b> (trained) node type for "
          "the selected parameter; small pale points are the <b>dormant</b> one &mdash; read those "
          "as noise. Positions are this subject's own forward-model dipole cloud (the exact "
          "source points this fit's leadfield uses) &mdash; not an approximation, so there's "
          "nothing to coregister."
    )

    header_eyebrow = "TVB-optim &middot; joint EEG+BOLD fit report"
    header_h1 = f"sub&#8209;{data['subject']} <span class=\"dim\">atlas&#8209;{data['atlas']}</span>"
    header_dek = (
        f"Heterogeneous Jansen&#8209;Rit (cortex) / Wilson&#8209;Cowan (subcortex) network, "
        f"<span class=\"mono\">{data['config']['leadfield_label']}</span> leadfield, "
        f"<span class=\"mono\">{data['config']['bold_model']}</span> BOLD forward model. "
        f"{len(nodes)} optimized nodes, fit jointly against this subject's EEG and BOLD."
    )
    footer_txt = (
        f"Generated from <span class=\"mono\">optimized_params.npz</span>, "
        f"<span class=\"mono\">config.json</span>, <span class=\"mono\">diagnostics_metrics.json</span> "
        f"and the loss histories in <span class=\"mono\">{out_dir.name}/</span>. Region labels via "
        f"<span class=\"mono\">Subject.load.connectivity_labels({data['atlas']}, fmri_aligned=True)</span>; "
        f"glass&#8209;brain points are this run's own "
        f"<span class=\"mono\">Subject.load.dipoles({data['spacing']})</span> forward&#8209;model "
        f"source cloud, colored via <span class=\"mono\">dipole_node_labels</span> and rendered "
        f"with <span class=\"mono\">pyvista</span> (offscreen)."
    )

    data_for_js = {**data, "brain_images": brain_images}
    html = (HTML_TEMPLATE
            .replace("__TITLE__", f"sub-{data['subject']} Fit Geography")
            .replace("__EYEBROW__", header_eyebrow)
            .replace("__H1__", header_h1)
            .replace("__DEK__", header_dek)
            .replace("__CALLOUT__", callout)
            .replace("__FIGURES__", figures_html)
            .replace("__FOOTER__", footer_txt)
            .replace("__REPORT_DATA_JSON__", json.dumps(data_for_js)))
    out_path.write_text(html)


HTML_TEMPLATE = r"""<title>__TITLE__</title>
<style>
:root{
  --bg:#eef1ec; --surface:#ffffff; --surface-2:#f5f7f3; --border:#d9e0da; --border-2:#c7d0c9;
  --ink:#182420; --ink-2:#44534c; --ink-muted:#7c8b83;
  --accent:#184f95; --accent-mid:#2a78d6; --accent-soft:#e4edf9;
  --hemi-l:#2a78d6; --hemi-r:#eb6834;
  --live:#2a78d6; --dormant:#a9b3ac;
  --seq-100:#cde2fb; --seq-250:#86b6ef; --seq-400:#3987e5; --seq-500:#256abf; --seq-650:#104281;
  --shadow: 0 1px 2px rgba(20,30,25,.06), 0 6px 20px -8px rgba(20,30,25,.12);
  color-scheme: light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#12211b; --surface:#1a2b23; --surface-2:#1f342b; --border:#2c4438; --border-2:#3a5747;
    --ink:#eaf1ec; --ink-2:#b9c8bf; --ink-muted:#87998e;
    --accent:#7fb0ea; --accent-mid:#5b9be3; --accent-soft:#1f3a52;
    --hemi-l:#5b9be3; --hemi-r:#e8834f;
    --live:#5b9be3; --dormant:#5c6b62;
    --seq-100:#173049; --seq-250:#1c4f78; --seq-400:#2f74b6; --seq-500:#5b9be3; --seq-650:#a9cdf3;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 10px 28px -10px rgba(0,0,0,.5);
    color-scheme: dark;
  }
}
:root[data-theme="dark"]{
  --bg:#12211b; --surface:#1a2b23; --surface-2:#1f342b; --border:#2c4438; --border-2:#3a5747;
  --ink:#eaf1ec; --ink-2:#b9c8bf; --ink-muted:#87998e;
  --accent:#7fb0ea; --accent-mid:#5b9be3; --accent-soft:#1f3a52;
  --hemi-l:#5b9be3; --hemi-r:#e8834f;
  --live:#5b9be3; --dormant:#5c6b62;
  --seq-100:#173049; --seq-250:#1c4f78; --seq-400:#2f74b6; --seq-500:#5b9be3; --seq-650:#a9cdf3;
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 10px 28px -10px rgba(0,0,0,.5);
  color-scheme: dark;
}

@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

*{box-sizing:border-box;}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:"IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  font-size:15px; line-height:1.55;
}
.page{max-width:1180px; margin:0 auto; padding:0 28px 80px;}
h1,h2,h3{font-family:"Newsreader", Georgia, serif; font-weight:600; text-wrap:balance; margin:0;}
a{color:var(--accent);}
code, .mono{font-family:"IBM Plex Mono", ui-monospace, monospace;}

.hero{padding:56px 0 34px; border-bottom:1px solid var(--border);}
.eyebrow{font-family:"IBM Plex Mono", monospace; font-size:12px; letter-spacing:.09em; text-transform:uppercase; color:var(--accent); margin-bottom:14px;}
.hero h1{font-size:40px; letter-spacing:-.01em;}
.hero h1 .dim{color:var(--ink-muted); font-weight:500; font-size:.7em;}
.dek{max-width:62ch; color:var(--ink-2); font-size:16px; margin:16px 0 22px;}
.chips{display:flex; flex-wrap:wrap; gap:8px;}
.chip{font-family:"IBM Plex Mono", monospace; font-size:12px; padding:5px 10px; border-radius:6px; background:var(--surface-2); border:1px solid var(--border); color:var(--ink-2);}
.chip b{color:var(--ink); font-weight:600;}

.tocbar{position:sticky; top:0; z-index:20; display:flex; gap:22px; padding:14px 0; margin-bottom:8px; background:color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter:blur(6px); border-bottom:1px solid var(--border); font-size:13px;}
.tocbar a{color:var(--ink-2); text-decoration:none; font-weight:500;}
.tocbar a:hover{color:var(--accent);}

section{padding:46px 0; border-bottom:1px solid var(--border);}
section:last-of-type{border-bottom:none;}
.section-kicker{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--ink-muted); letter-spacing:.08em; text-transform:uppercase; margin-bottom:6px;}
section h2{font-size:27px; margin-bottom:6px;}
.section-lede{max-width:70ch; color:var(--ink-2); margin:10px 0 28px;}

.stat-grid{display:grid; grid-template-columns:repeat(auto-fit, minmax(160px,1fr)); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:10px; overflow:hidden; margin-bottom:30px;}
.stat{background:var(--surface); padding:16px 18px;}
.stat .lbl{font-size:12px; color:var(--ink-muted); margin-bottom:6px;}
.stat .val{font-family:"IBM Plex Mono",monospace; font-size:22px; font-weight:500; font-variant-numeric:tabular-nums;}
.stat .sub{font-size:12px; color:var(--ink-2); margin-top:4px;}
.stat .val.down{color:#1a7a4c;}
.stat .val.flat{color:var(--ink-muted);}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .stat .val.down{color:#5fd39a;}}
:root[data-theme="dark"] .stat .val.down{color:#5fd39a;}

.card{background:var(--surface); border:1px solid var(--border); border-radius:12px; box-shadow:var(--shadow);}
.chart-row{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:30px;}
.chart-card{padding:18px 20px 14px;}
.chart-card h4{margin:0 0 2px; font-family:"IBM Plex Sans"; font-size:14px; font-weight:600;}
.chart-card .chart-note{font-size:12px; color:var(--ink-muted); margin-bottom:8px;}
svg text{fill:var(--ink-2); font-family:"IBM Plex Sans", sans-serif;}
.axis-line{stroke:var(--border-2); stroke-width:1;}
.gridline{stroke:var(--border); stroke-width:1;}

.figure-grid{display:grid; grid-template-columns:1.1fr 1fr; gap:18px;}
.figure-grid .card.wide{grid-column:1/-1;}
figure{margin:0; padding:14px;}
figure img{width:100%; display:block; border-radius:6px; background:#fff; border:1px solid var(--border);}
figcaption{font-size:12.5px; color:var(--ink-2); margin-top:10px; line-height:1.5;}
figcaption b{color:var(--ink);}

.callout{background:var(--accent-soft); border:1px solid var(--border-2); border-left:3px solid var(--accent-mid); border-radius:8px; padding:18px 20px; margin-bottom:26px; font-size:14px; color:var(--ink-2);}
.callout b{color:var(--ink);}
.callout .mono{background:var(--surface); padding:1px 5px; border-radius:4px; font-size:12.5px;}

.pill-row{display:flex; flex-wrap:wrap; gap:8px; margin-bottom:22px;}
.pill{font-family:"IBM Plex Mono", monospace; font-size:13px; padding:8px 14px; border-radius:999px; border:1px solid var(--border-2); background:var(--surface); color:var(--ink-2); cursor:pointer; display:flex; align-items:center; gap:7px; transition:background .12s, color .12s, border-color .12s;}
.pill:hover{border-color:var(--accent-mid);}
.pill.active{background:var(--accent); border-color:var(--accent); color:var(--surface);}
.pill .dot{width:7px; height:7px; border-radius:50%; background:var(--live);}
.pill.active .dot{background:var(--surface);}

.brain-panel{display:grid; grid-template-columns:1fr 300px; gap:18px; margin-bottom:14px;}
.brain-info{padding:18px; display:flex; flex-direction:column; gap:14px;}
.brain-info .ig{display:flex; justify-content:space-between; font-size:13px; padding:8px 0; border-bottom:1px solid var(--border);}
.brain-info .ig:last-child{border-bottom:none;}
.brain-info .ig .k{color:var(--ink-muted);}
.brain-info .ig .v{font-family:"IBM Plex Mono",monospace; font-weight:500;}
.legend-row{display:flex; align-items:center; gap:10px; font-size:11.5px; color:var(--ink-muted); margin-top:6px;}
.seq-bar{height:8px; flex:1; border-radius:4px; background:linear-gradient(90deg, var(--seq-100), var(--seq-250), var(--seq-400), var(--seq-500), var(--seq-650));}
.mk-row{display:flex; align-items:center; gap:16px; font-size:12px; color:var(--ink-2);}
.mk{display:flex; align-items:center; gap:6px;}

.strip-grid{display:grid; grid-template-columns:repeat(2, 1fr); gap:10px; margin:26px 0;}
.strip-card{padding:12px 16px; cursor:pointer; border:1px solid var(--border);}
.strip-card.active{border-color:var(--accent-mid); box-shadow:0 0 0 1px var(--accent-mid);}
.strip-head{display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;}
.strip-head .name{font-family:"IBM Plex Mono",monospace; font-weight:600; font-size:13px;}
.strip-head .range{font-size:11px; color:var(--ink-muted);}

.bar-wrap{padding:20px 22px; margin-top:8px; overflow-x:auto;}

#tooltip{position:fixed; pointer-events:none; z-index:100; background:var(--ink); color:var(--bg); font-size:12px; padding:8px 10px; border-radius:7px; line-height:1.45; max-width:220px; opacity:0; transform:translateY(4px); transition:opacity .08s, transform .08s; box-shadow:var(--shadow);}
#tooltip b{display:block; font-size:12.5px; margin-bottom:2px;}
#tooltip.show{opacity:1; transform:translateY(0);}

.caveat-list{max-width:74ch; padding-left:0; list-style:none; margin:0;}
.caveat-list li{position:relative; padding-left:20px; margin-bottom:14px; color:var(--ink-2); font-size:14px;}
.caveat-list li::before{content:"\2014"; position:absolute; left:0; color:var(--ink-muted);}
footer{padding:34px 0 10px; color:var(--ink-muted); font-size:12.5px;}

@media (max-width: 880px){ .chart-row, .figure-grid, .brain-panel, .strip-grid{grid-template-columns:1fr;} }
</style>

<div class="page">
  <header class="hero">
    <div class="eyebrow">__EYEBROW__</div>
    <h1>__H1__</h1>
    <p class="dek">__DEK__</p>
    <div class="chips" id="chips"></div>
  </header>

  <nav class="tocbar">
    <a href="#learning">Learning</a>
    <a href="#geography">Parameter geography</a>
    <a href="#caveats">Caveats</a>
  </nav>

  <section id="learning">
    <div class="section-kicker">01 &middot; Training</div>
    <h2>How much did this fit actually learn?</h2>
    <p class="section-lede">Stat tiles and loss curves below are computed directly from this
      run's saved loss history and diagnostics; figures are whichever post-fit diagnostic
      PNGs this run produced.</p>

    <div class="stat-grid" id="stat-grid"></div>

    <div class="chart-row">
      <div class="card chart-card">
        <h4>EEG loss</h4>
        <div class="chart-note" id="eeg-loss-note"></div>
        <svg id="eeg-loss-svg" viewBox="0 0 420 220" width="100%"></svg>
      </div>
      <div class="card chart-card">
        <h4>BOLD loss</h4>
        <div class="chart-note" id="bold-loss-note"></div>
        <svg id="bold-loss-svg" viewBox="0 0 420 220" width="100%"></svg>
      </div>
    </div>

    <div class="figure-grid">
__FIGURES__
    </div>
  </section>

  <section id="geography">
    <div class="section-kicker">02 &middot; Parameter geography</div>
    <h2>Where the seven per&#8209;node parameters landed</h2>
    <p class="section-lede">Each parameter is a per&#8209;node vector. Select one below to see
      how its fitted value is distributed across the brain.</p>

    <div class="callout"><b>Read this before the maps.</b> __CALLOUT__</div>

    <div class="pill-row" id="param-pills"></div>

    <div class="brain-panel">
      <div class="card" style="padding:14px;">
        <img id="brain-img" alt="glass-brain map" style="width:100%; display:block; border-radius:6px;">
      </div>
      <div class="card brain-info" id="brain-info"></div>
    </div>

    <div class="strip-grid" id="strip-grid"></div>

    <div class="card bar-wrap">
      <h4 style="margin:0 0 2px; font-size:14px;">Mean deviation from init, by group &amp; hemisphere</h4>
      <div class="chart-note" style="font-size:12px;color:var(--ink-muted);margin-bottom:10px;">error bars = within&#8209;group standard deviation</div>
      <svg id="bar-svg" viewBox="0 0 1080 360" width="100%"></svg>
    </div>
  </section>

  <section id="caveats">
    <div class="section-kicker">03 &middot; Caveats</div>
    <h2>Before you read anything above as a finding</h2>
    <ul class="caveat-list">
      <li><b>Check the epoch count in the header chips.</b> A short run means every pattern
        above is "which direction did a few gradient steps push things," not a converged,
        subject&#8209;specific parameter set.</li>
      <li><b>The live/dormant split is architectural fact</b> (confirmed by exact&#8209;zero
        movement where predicted) &mdash; but <i>which specific regions</i> moved most, early
        in training, more likely reflects which EEG chunks/BOLD windows were seen early than a
        stable subject pattern.</li>
      <li><b>Node positions on the brain map are schematic</b> &mdash; grouped by functional
        network/structure and jittered for visibility, not this subject's real anatomical
        coordinates.</li>
    </ul>
  </section>

  <footer>__FOOTER__</footer>
</div>

<div id="tooltip"></div>

<script id="report-data" type="application/json">__REPORT_DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const NODES = DATA.nodes;
const PARAMS = DATA.per_node_params;
const tooltip = document.getElementById('tooltip');

function fmt(x, d=4){ return Number(x).toFixed(d); }
function pct(x){ return (x*100).toFixed(1) + '%'; }

function showTip(html, evt){ tooltip.innerHTML = html; tooltip.classList.add('show'); moveTip(evt); }
function moveTip(evt){
  const pad = 14; let x = evt.clientX + pad, y = evt.clientY + pad; const tw=230, th=90;
  if (x + tw > window.innerWidth) x = evt.clientX - tw - pad;
  if (y + th > window.innerHeight) y = evt.clientY - th - pad;
  tooltip.style.left = x + 'px'; tooltip.style.top = y + 'px';
}
function hideTip(){ tooltip.classList.remove('show'); }

(function(){
  const c = DATA.config;
  const chips = [
    ['subject', DATA.subject], ['atlas', DATA.atlas], ['epochs', DATA.num_epochs],
    ['schedule', c.schedule], ['bold model', c.bold_model], ['fs', c.fs + ' Hz'],
    ['leadfield', c.leadfield_label], ['lr', c.learning_rate],
  ];
  document.getElementById('chips').innerHTML = chips.map(([k,v]) => `<span class="chip">${k} <b>${v}</b></span>`).join('');
})();

(function(){
  const dm = DATA.diagnostics_metrics || {};
  const stats = [];
  if (DATA.loss_eeg.length){
    const e0=DATA.loss_eeg[0], e1=DATA.loss_eeg[DATA.loss_eeg.length-1];
    stats.push(['EEG loss &Delta;', pct((e1-e0)/e0*-1), `${e0.toExponential(2)} &rarr; ${e1.toExponential(2)}`, 'down']);
  }
  if (DATA.loss_bold.length){
    const b0=DATA.loss_bold[0], b1=DATA.loss_bold[DATA.loss_bold.length-1];
    stats.push(['BOLD loss &Delta;', pct((b1-b0)/b0*-1), `${b0.toFixed(4)} &rarr; ${b1.toFixed(4)}`, 'down']);
  }
  if ('eeg_corr' in dm) stats.push(['EEG spectral corr.', fmt(dm.eeg_corr,2), 'sim vs. empirical PSD', '']);
  if ('fc_corr' in dm) stats.push(['Static FC corr.', fmt(dm.fc_corr,3), 'sim vs. empirical FC', 'flat']);
  if ('dfc_w_dist_before' in dm && 'dfc_w_dist_after' in dm){
    stats.push(['dFC W&#8209;distance &Delta;', pct((dm.dfc_w_dist_after-dm.dfc_w_dist_before)/dm.dfc_w_dist_before*-1),
      `${fmt(dm.dfc_w_dist_before,4)} &rarr; ${fmt(dm.dfc_w_dist_after,4)}`, 'flat']);
  }
  stats.push(['Global coupling G', fmt(DATA.G,3), DATA.G_bounds ? `init ${DATA.G_init}, bound [0, ${DATA.G_bounds[1]}]` : '', 'flat']);
  document.getElementById('stat-grid').innerHTML = stats.map(([lbl,val,sub,cls]) => `
    <div class="stat"><div class="lbl">${lbl}</div><div class="val ${cls}">${val}</div><div class="sub">${sub}</div></div>`).join('');
})();

function drawLossChart(svgId, noteId, xs, ys, color, valFmt, note){
  document.getElementById(noteId).textContent = note;
  const svg = document.getElementById(svgId);
  if (!xs.length){ svg.closest('.chart-card').style.opacity = 0.4; svg.parentElement.insertAdjacentHTML('beforeend', '<div style="font-size:12px;color:var(--ink-muted);padding:8px 0;">not recorded in this run</div>'); return; }
  const W=420,H=220, ml=52,mr=16,mt=34,mb=32;
  const iw=W-ml-mr, ih=H-mt-mb;
  const xmin=Math.min(...xs), xmax=Math.max(...xs);
  const ymin=Math.min(...ys), ymax=Math.max(...ys);
  const yr=(ymax-ymin)||(ymax*0.001)||1, xr=(xmax-xmin)||1;
  const X = x => ml + (x-xmin)/xr*iw;
  const Y = y => mt + ih - (y-ymin)/yr*ih;
  let s = '';
  s += `<text x="0" y="14" font-size="10.5" text-anchor="start" font-family="IBM Plex Mono" fill="${color}">epoch ${xs[0]}: ${valFmt(ys[0])}</text>`;
  s += `<text x="${W}" y="14" font-size="10.5" text-anchor="end" font-family="IBM Plex Mono" fill="${color}">epoch ${xs[xs.length-1]}: ${valFmt(ys[ys.length-1])}</text>`;
  for(let i=0;i<=3;i++){ const gy = mt + ih*i/3; s += `<line class="gridline" x1="${ml}" x2="${W-mr}" y1="${gy}" y2="${gy}"/>`; }
  s += `<line class="axis-line" x1="${ml}" x2="${ml}" y1="${mt}" y2="${mt+ih}"/>`;
  s += `<line class="axis-line" x1="${ml}" x2="${W-mr}" y1="${mt+ih}" y2="${mt+ih}"/>`;
  s += `<polyline points="${xs.map((x,i)=>`${X(x)},${Y(ys[i])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;
  xs.forEach((x,i)=>{
    s += `<circle class="pt" data-x="${x}" data-y="${ys[i]}" cx="${X(x)}" cy="${Y(ys[i])}" r="5" fill="${color}" stroke="var(--surface)" stroke-width="1.5" style="cursor:pointer"/>`;
    s += `<text x="${X(x)}" y="${mt+ih+18}" font-size="10" text-anchor="middle">${x}</text>`;
  });
  svg.innerHTML = s;
  svg.querySelectorAll('.pt').forEach(p=>{
    p.addEventListener('mouseenter', e=>showTip(`<b>epoch ${p.dataset.x}</b>loss ${valFmt(+p.dataset.y)}`, e));
    p.addEventListener('mousemove', moveTip);
    p.addEventListener('mouseleave', hideTip);
  });
}
drawLossChart('eeg-loss-svg', 'eeg-loss-note', DATA.loss_eeg.map((_,i)=>i+1), DATA.loss_eeg, 'var(--accent-mid)', v=>v.toExponential(3), 'evaluated every epoch');
drawLossChart('bold-loss-svg', 'bold-loss-note', DATA.loss_bold.map((_,i)=>(i+1)*DATA.bold_every), DATA.loss_bold, 'var(--hemi-r)', v=>v.toFixed(5), `evaluated every ${DATA.bold_every} epochs`);

// Glass-brain maps are pre-rendered (pyvista, real subject dipole cloud -- see
// write_glass_brain_plots) rather than drawn in JS; pill clicks just swap the <img> src.
const BRAIN_IMAGES = DATA.brain_images || {};
let currentParam = PARAMS.includes('B') ? 'B' : PARAMS[0];

function renderBrain(param){
  const img = document.getElementById('brain-img');
  img.src = BRAIN_IMAGES[param] || '';
  img.alt = `${param} glass brain`;

  const liveKind = DATA.live_on[param];
  const init = DATA.init_by_param[param];
  const isLive = n => (liveKind === 'surface') === n.is_surface;
  const vals = NODES.map(n=>n[param]);
  const vmin = Math.min(...vals), vmax = Math.max(...vals);
  const liveVals = NODES.filter(isLive).map(n=>n[param]);
  const maxDev = liveVals.length ? Math.max(...liveVals.map(v=>Math.abs(v-init))) : 0;
  const [lo,hi] = DATA.bounds_by_param[param];
  document.getElementById('brain-info').innerHTML = `
    <div><div style="font-family:'IBM Plex Mono';font-weight:600;font-size:16px;">${param}</div>
      <div style="font-size:12px;color:var(--ink-muted);">${DATA.param_units[param]}</div></div>
    <div class="ig"><span class="k">live on</span><span class="v">${liveKind} (n=${liveVals.length})</span></div>
    <div class="ig"><span class="k">init</span><span class="v">${init}</span></div>
    <div class="ig"><span class="k">bounds</span><span class="v">[${lo}, ${hi}]</span></div>
    <div class="ig"><span class="k">observed range</span><span class="v">[${fmt(vmin,3)}, ${fmt(vmax,3)}]</span></div>
    <div class="ig"><span class="k">max |&Delta; init|</span><span class="v">${fmt(maxDev, param==='P'?6:4)}</span></div>
    <div class="ig"><span class="k">as % of range</span><span class="v">${pct(maxDev/(hi-lo))}</span></div>
    <div class="mk-row" style="margin-top:4px;">
      <div class="mk"><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--live)"/></svg> live node (colored by value)</div>
      <div class="mk"><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(--dormant)"/></svg> dormant node</div>
    </div>`;
}

function renderStrips(){
  const grid = document.getElementById('strip-grid');
  grid.innerHTML = PARAMS.map(p=>`<div class="card strip-card" data-param="${p}" id="strip-${p}">
    <div class="strip-head"><span class="name">${p}</span><span class="range">[${DATA.bounds_by_param[p][0]}, ${DATA.bounds_by_param[p][1]}]</span></div>
    <svg viewBox="0 0 640 64" width="100%" height="64" id="strip-svg-${p}"></svg>
  </div>`).join('');
  PARAMS.forEach(p=>{
    const svg = document.getElementById(`strip-svg-${p}`);
    const init = DATA.init_by_param[p];
    const liveKind = DATA.live_on[p];
    const isLive = n => (liveKind === 'surface') === n.is_surface;
    const vals = NODES.map(n=>n[p]);
    const vmin = Math.min(...vals, init), vmax = Math.max(...vals, init);
    const span = (vmax-vmin) || 1e-9;
    const ml=8, mr=8, W=640;
    const X = v => ml + (v-vmin)/span*(W-ml-mr);
    let s = `<line class="axis-line" x1="${ml}" x2="${W-mr}" y1="40" y2="40"/>`;
    s += `<line x1="${X(init)}" x2="${X(init)}" y1="10" y2="40" stroke="var(--hemi-r)" stroke-dasharray="2,2" stroke-width="1.3"/>`;
    NODES.forEach(n=>{
      const live_ = isLive(n);
      s += `<circle cx="${X(n[p])}" cy="${live_?20:30}" r="${live_?3.2:2.4}" fill="${live_?'var(--live)':'var(--dormant)'}" opacity="${live_?0.75:0.55}"/>`;
    });
    svg.innerHTML = s;
    document.getElementById(`strip-${p}`).addEventListener('click', ()=>selectParam(p));
  });
}

const GROUP_ORDER = ['VisCent','VisPeri','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default','TempPar','Cerebellum','Hippocampus','BasalGanglia','Thalamus','Brainstem'];
function mean(a){ return a.reduce((x,y)=>x+y,0)/a.length; }
function std(a){ const m=mean(a); return Math.sqrt(mean(a.map(v=>(v-m)**2))); }

function renderBar(param){
  const svg = document.getElementById('bar-svg');
  const init = DATA.init_by_param[param];
  const groups = GROUP_ORDER.filter(g => NODES.some(n=>n.group===g));
  const W=1080,H=360, ml=64,mr=20,mt=16,mb=70;
  const iw=W-ml-mr, ih=H-mt-mb;
  const bandW = iw/groups.length;
  const series = { L: [], R: [], M: [] };
  groups.forEach(g=>{
    ['L','R','M'].forEach(h=>{
      const vv = NODES.filter(n=>n.group===g && n.hemi===h).map(n=>n[param]-init);
      series[h].push(vv.length ? {m: mean(vv), s: std(vv), n: vv.length} : null);
    });
  });
  let allVals = [];
  groups.forEach((g,i)=>['L','R','M'].forEach(h=>{ const d=series[h][i]; if(d) allVals.push(d.m-d.s, d.m+d.s); }));
  if (!allVals.length) allVals=[0,1];
  let ymin = Math.min(0,...allVals), ymax = Math.max(0,...allVals);
  if (ymax-ymin < 1e-12){ ymax = 1e-9; ymin=-1e-9; }
  const pad = (ymax-ymin)*0.12; ymin -= pad; ymax += pad;
  const Y = v => mt + ih - (v-ymin)/(ymax-ymin)*ih;
  const y0 = Y(0);
  let s = '';
  for(let i=0;i<=4;i++){ const gy = mt+ih*i/4; s += `<line class="gridline" x1="${ml}" x2="${W-mr}" y1="${gy}" y2="${gy}"/>`; }
  s += `<line x1="${ml}" x2="${W-mr}" y1="${y0}" y2="${y0}" stroke="var(--border-2)" stroke-width="1.2"/>`;
  s += `<line class="axis-line" x1="${ml}" x2="${ml}" y1="${mt}" y2="${mt+ih}"/>`;
  s += `<text x="16" y="${mt+8}" font-size="10">${param}&minus;init</text>`;
  groups.forEach((g,i)=>{
    const bx = ml + i*bandW;
    const hemisHere = ['L','R','M'].filter(h=>series[h][i]);
    const bw = Math.min(30, (bandW-14)/Math.max(1,hemisHere.length));
    hemisHere.forEach((h,j)=>{
      const d = series[h][i];
      const cx = bx + bandW/2 - (hemisHere.length*bw)/2 + j*bw + bw/2;
      const color = h==='L' ? 'var(--hemi-l)' : (h==='R' ? 'var(--hemi-r)' : 'var(--ink-muted)');
      const yTop = Y(Math.max(d.m,0)), yBot = Y(Math.min(d.m,0));
      s += `<rect x="${cx-bw/2+1}" y="${yTop}" width="${bw-2}" height="${Math.max(1,yBot-yTop)}" fill="${color}" opacity="0.85"/>`;
      s += `<line x1="${cx}" x2="${cx}" y1="${Y(d.m-d.s)}" y2="${Y(d.m+d.s)}" stroke="${color}" stroke-width="1.3"/>`;
      s += `<g class="bar-mk" data-tip="${g} &middot; hemi ${h} (n=${d.n})<br>mean &Delta; ${fmt(d.m,5)} &plusmn; ${fmt(d.s,5)}">
        <rect x="${cx-bw/2-1}" y="${mt}" width="${bw+2}" height="${ih}" fill="transparent" style="cursor:pointer"/></g>`;
    });
    s += `<text x="${bx+bandW/2}" y="${mt+ih+16}" font-size="10.5" text-anchor="end" transform="rotate(-52 ${bx+bandW/2} ${mt+ih+16})">${g}</text>`;
  });
  svg.innerHTML = s;
  svg.querySelectorAll('.bar-mk').forEach(el=>{
    el.addEventListener('mouseenter', e=>showTip(el.dataset.tip, e));
    el.addEventListener('mousemove', moveTip);
    el.addEventListener('mouseleave', hideTip);
  });
}

function selectParam(p){
  currentParam = p;
  document.querySelectorAll('.pill').forEach(el=>el.classList.toggle('active', el.dataset.param===p));
  document.querySelectorAll('.strip-card').forEach(el=>el.classList.toggle('active', el.dataset.param===p));
  renderBrain(p);
  renderBar(p);
}

document.getElementById('param-pills').innerHTML = PARAMS.map(p=>
  `<button class="pill" data-param="${p}"><span class="dot"></span>${p}</button>`).join('') +
  (DATA.G_bounds ? `<span class="chip" style="margin-left:6px;">G (global, scalar) = <b>${fmt(DATA.G,3)}</b></span>` : '');
document.querySelectorAll('.pill').forEach(el=>el.addEventListener('click', ()=>selectParam(el.dataset.param)));

renderStrips();
selectParam(currentParam);
</script>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True,
                    help="a prior eeg_bold_fit_cli.py run's output dir (contains optimized_params.npz, config.json)")
    p.add_argument("--no-static", action="store_true", help="skip the matplotlib PNGs + params_report.md")
    p.add_argument("--no-artifact", action="store_true", help="skip artifact.html")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not (results_dir / "optimized_params.npz").exists():
        raise SystemExit(f"{results_dir} has no optimized_params.npz -- not a fit results dir?")

    print(f"Loading {results_dir} ...")
    data, subject = load_run(results_dir)
    print(f"  sub-{data['subject']} atlas-{data['atlas']}: {len(data['nodes'])} nodes, "
          f"{data['num_epochs']} epochs")

    if not args.no_static:
        write_static_plots(data, results_dir, subject)
        write_markdown(data, results_dir)
        print(f"  wrote params_report.md + PNGs (loss curves, distributions, group/hemisphere, "
              f"{len(data['per_node_params'])} glass brains) -> {results_dir}")

    if not args.no_artifact:
        out_path = results_dir / "artifact.html"
        write_artifact_html(data, results_dir, out_path)
        print(f"  wrote {out_path}  ({out_path.stat().st_size/1e6:.2f} MB) -- "
              "open in a browser, or hand this path to Claude Code to publish as an Artifact")


if __name__ == "__main__":
    main()
