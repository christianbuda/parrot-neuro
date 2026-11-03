#!/usr/bin/env python
"""Article figure 1 -- pipeline schematic (draft, vector PDF to polish).

subject MRI / DWI -> connectome + FEM leadfield -> JR (cortex) / WC
(subcortex) network -> EEG and BOLD observation models -> joint loss with
gradients back to the per-node parameters. Settings (atlas, BOLD model,
weights) are read from one run's config.json so the figure states what was
actually fitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _common import C_ALT, C_FIT, C_INIT, C_SC, DEFAULT_ROOT, INK, INK_2, Out, load_runs, plt

BOX = dict(boxstyle="round,pad=0.02,rounding_size=0.02", lw=1.2)


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=color + "22", ec=color, **BOX))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=8, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=6.2, color=INK_2, linespacing=1.35)


def arrow(ax, a, b, color=INK_2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = Out(args.out or args.root / "article")
    cfg = load_runs(args.root, need_sim=False)[0].cfg
    n_par = ", ".join(lp["name"] for lp in cfg["learnable_params"])

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.00, 0.55, 0.18, 0.40, "Subject MRI", "T1 (+T2)\nDWI\nresting fMRI\nEEG (59 ch)", INK_2)
    box(ax, 0.22, 0.70, 0.20, 0.25, "Connectome", f"Schaefer-{cfg['atlas']} + subcortex\nSC weights, delays\n"
        f"({cfg['conduction_speed']:g} m/s)", C_SC)
    box(ax, 0.22, 0.38, 0.20, 0.27, "Head model", f"CGAL tet mesh\nFEM leadfield\n({cfg['leadfield_label']}, "
        f"{cfg['spacing']} mm)", C_SC)
    box(ax, 0.46, 0.45, 0.20, 0.50, "Network model", "Jansen-Rit (cortex)\nWilson-Cowan (subcortex)\n"
        f"per-node: {n_par.replace(', G', '')}\nglobal coupling G\nnoise seed {cfg['noise_seed']}", C_FIT)
    box(ax, 0.70, 0.66, 0.29, 0.29, "EEG observation", f"leadfield projection\nPSD {cfg['fmin']:g}-{cfg['fmax']:g} Hz "
        f"({cfg['eeg_task']})", C_INIT)
    box(ax, 0.70, 0.30, 0.29, 0.31, "BOLD observation", f"{cfg['bold_model'].capitalize()} hemodynamics\n"
        f"FC + FCD ({cfg['tr_ms'] / 1000:g} s TR)", C_ALT)
    box(ax, 0.40, 0.02, 0.40, 0.20, "Joint loss", f"{cfg['joint_eeg_weight']:.0e} x EEG-PSD MSE + "
        f"{cfg['joint_bold_weight']:g} x ({cfg['bold_fc_weight']:.2f} FC MSE\n+ {cfg['bold_dfc_weight']:.2f} FCD W1"
        f" + {cfg['bold_psd_weight']:.3f} BOLD-PSD)", INK)
    arrow(ax, (0.18, 0.80), (0.22, 0.82))
    arrow(ax, (0.18, 0.62), (0.22, 0.52))
    arrow(ax, (0.42, 0.82), (0.46, 0.80))
    arrow(ax, (0.42, 0.50), (0.70, 0.74), rad=-0.15)
    arrow(ax, (0.66, 0.80), (0.70, 0.80))
    arrow(ax, (0.66, 0.55), (0.70, 0.50))
    arrow(ax, (0.85, 0.66), (0.80, 0.17), rad=-0.2)
    arrow(ax, (0.85, 0.30), (0.78, 0.22))
    arrow(ax, (0.40, 0.12), (0.56, 0.45), color=C_FIT, ls="--", rad=-0.3)
    ax.text(0.30, 0.25, "gradients (JAX autodiff)\nAdam, " f"{cfg['num_epochs']} epochs", fontsize=6.2, color=C_FIT,
            ha="center")
    out.fig(fig, "fig1_schematic")


if __name__ == "__main__":
    main()
