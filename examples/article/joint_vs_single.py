#!/usr/bin/env python
"""Article section 3 -- fitting both modalities together is worth it.

Compares, on the subjects present in every root:

  joint vs single modality (fig3_joint_vs_single, tables/joint_vs_single*.csv)
      the base joint fits (``--root``) against ``gpu_fit.py fit --variant
      eeg-only`` / ``bold-only`` fits, on BOTH modalities' agreement
      (EEG spectral r, alpha topography r, alpha-peak error, FC r, FCD KS),
      plus how much each fit moved every parameter away from its start
      (a parameter a modality leaves free stays near the start) and how
      well single-modality parameters agree with the joint ones.
  EEG-weight trade-off (fig3_weight_tradeoff, tables/weight_tradeoff.csv)
      ``--weight-roots 1e4=<root> 1e6=<root>`` (the base root is 1e5) ->
      EEG vs BOLD agreement per weight, the single-modality fits as the
      two end points.

    pixi run python examples/article/joint_vs_single.py \\
        --root new_leadfields/1/atlas-100 \\
        --eeg-only-root article_fits/single_modality/atlas-100 \\
        --bold-only-root article_fits/single_modality/atlas-100 \\
        --weight-roots 1e4=article_fits/eeg_weight/atlas-100 1e6=article_fits/eeg_weight/atlas-100
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common import (C_ALT, C_FIT, C_INIT, C_SC, DEFAULT_BIDS, DEFAULT_ROOT, INK_2, NEUTRAL_C, Out, Subject,
                     fit_metrics, load_runs, panel, plt, style, summary, wilcoxon)

METRICS = [("spec_r", "EEG spectral r", 1), ("alpha_topo_r", "alpha topography r", 1),
           ("apf_absdiff", "|alpha peak error| (Hz)", -1), ("fc_r", "FC r", 1), ("fcd_ks", "FCD KS distance", -1)]
COLORS = {"joint": C_FIT, "eeg-only": C_INIT, "bold-only": C_SC}


def runs_by_subject(root: Path, suffix: str | None) -> dict:
    """{subject: Run} of ``root``, optionally only folders ending in ``_<suffix>``."""
    return {r.sid: r for r in load_runs(root) if suffix is None or r.path.name.endswith(f"_{suffix}")}


def param_movement(run, live: dict) -> dict:
    """Median |fitted - start| / range over the nodes a param is live on."""
    fit, start = run.params("best"), run.params("init")
    bounds = {lp["name"]: (lp["low"], lp["high"]) for lp in run.cfg["learnable_params"]}
    out = {}
    for p, v in fit.items():
        lo, hi = bounds[p]
        m = live.get(p, np.ones(len(v), bool)) if len(v) > 1 else np.ones(1, bool)
        out[p] = float(np.median(np.abs(v - np.broadcast_to(start[p], v.shape))[m] / (hi - lo)))
    return out


def live_masks(run) -> dict:
    from params_report_cli import LIVE_ON
    surface = np.asarray(run.sim["mask_cortical"]) > 0.5
    return {p: (surface if LIVE_ON.get(p) == "surface" else ~surface) for p in LIVE_ON}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="base joint fits (EEG weight 1e5)")
    ap.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
    ap.add_argument("--eeg-only-root", type=Path, default=None)
    ap.add_argument("--bold-only-root", type=Path, default=None)
    ap.add_argument("--weight-roots", nargs="+", default=[], metavar="W=ROOT")
    ap.add_argument("--base-weight", default="1e5")
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/article")
    args = ap.parse_args()
    out = Out(args.out or args.root / "article")
    style()

    base = runs_by_subject(args.root, None)
    variants = {"joint": base}
    if args.eeg_only_root:
        variants["eeg-only"] = runs_by_subject(args.eeg_only_root, "eeg-only")
    if args.bold_only_root:
        variants["bold-only"] = runs_by_subject(args.bold_only_root, "bold-only")
    weights = {args.base_weight: base}
    for spec in args.weight_roots:
        w, root = spec.split("=", 1)
        weights[w] = runs_by_subject(Path(root), f"w{w}")
    num = {}
    subj = {}

    def metrics(run):
        s = subj.setdefault(run.sid, Subject(str(args.bids), run.sid))
        return fit_metrics(run, s)

    # -- joint vs single modality
    if len(variants) > 1:
        common = sorted(set.intersection(*(set(v) for v in variants.values())))
        print(f"[joint vs single] {len(common)} subjects with {', '.join(variants)}")
        rows, moves = [], []
        for name, rr in variants.items():
            for s in common:
                rows.append({"variant": name, **metrics(rr[s])})
                moves.append({"variant": name, "subject": s, **param_movement(rr[s], live_masks(rr[s]))})
        df, mv = pd.DataFrame(rows), pd.DataFrame(moves)
        out.table(df, "joint_vs_single")
        out.table(mv, "joint_vs_single_param_movement")

        agree = []
        for name in variants:
            if name == "joint":
                continue
            for s in common:
                pj, ps = variants["joint"][s].params("best"), variants[name][s].params("best")
                live = live_masks(variants["joint"][s])
                for p in pj:
                    if len(pj[p]) > 1:
                        m = live.get(p, np.ones(len(pj[p]), bool))
                        agree.append({"variant": name, "subject": s, "param": p,
                                      "rho_vs_joint": stats.spearmanr(pj[p][m], ps[p][m])[0]})
                    else:
                        agree.append({"variant": name, "subject": s, "param": p,
                                      "abs_diff_vs_joint": float(abs(pj[p][0] - ps[p][0]))})
        ag = pd.DataFrame(agree)
        out.table(ag, "joint_vs_single_param_agreement")

        num["joint_vs_single"] = {
            "n_subjects": len(common),
            "metrics": {name: {k: summary(df[df.variant == name][k]) for k, _, _ in METRICS if k in df}
                        for name in variants},
            "tests_joint_vs": {name: {k: wilcoxon(df[df.variant == "joint"].set_index("subject")[k].loc[common],
                                                  df[df.variant == name].set_index("subject")[k].loc[common])
                                      for k, _, _ in METRICS if k in df}
                               for name in variants if name != "joint"},
            "param_movement_median": mv.groupby("variant").median(numeric_only=True).to_dict(),
            "param_agreement_median_rho": ag.groupby(["variant", "param"]).rho_vs_joint.median().unstack().to_dict()
            if "rho_vs_joint" in ag else {},
        }

        ms = [m for m in METRICS if m[0] in df]
        fig, axs = plt.subplots(1, len(ms) + 1, figsize=(7.2, 2.6), constrained_layout=True,
                                gridspec_kw={"width_ratios": [1] * len(ms) + [1.6]})
        names = list(variants)
        for ax, (k, lab, _), letter in zip(axs, ms, "abcdef"):
            W = df.pivot(index="subject", columns="variant", values=k)[names]
            for _, r in W.iterrows():
                ax.plot(range(len(names)), r.values, color=NEUTRAL_C, lw=0.5, alpha=0.6)
            for j, n in enumerate(names):
                ax.scatter(np.full(len(W), j), W[n], s=10, color=COLORS[n], lw=0, zorder=3)
            ax.set_xticks(range(len(names)), [n.replace("-only", "\nonly") for n in names])
            ax.set_xlim(-0.5, len(names) - 0.5)
            ax.set_title(lab)
            panel(ax, letter)
        ax = axs[-1]
        M = mv.groupby("variant").median(numeric_only=True).loc[names]
        im = ax.imshow(M.to_numpy(), cmap="Blues", aspect="auto")
        ax.set_xticks(range(M.shape[1]), M.columns)
        ax.set_yticks(range(len(names)), names)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(M.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v > np.nanmax(M.to_numpy()) * 0.6 else INK_2)
        ax.set_title("median |fitted - start| / range")
        panel(ax, "f")
        out.fig(fig, "fig3_joint_vs_single")

    # -- EEG-weight trade-off
    if len(weights) > 1:
        common = sorted(set.intersection(*(set(v) for v in weights.values())))
        print(f"[weight trade-off] {len(common)} subjects at weights {', '.join(weights)}")
        rows = [{"weight": float(w), "label": w, **metrics(rr[s])} for w, rr in weights.items() for s in common]
        for name in ("eeg-only", "bold-only"):
            if name in variants:
                rows += [{"weight": np.inf if name == "eeg-only" else 0.0, "label": name, **metrics(variants[name][s])}
                         for s in common if s in variants[name]]
        tw = pd.DataFrame(rows).sort_values("weight")
        out.table(tw, "weight_tradeoff")
        num["weight_tradeoff"] = tw.groupby("label")[["spec_r", "alpha_topo_r", "fc_r", "fcd_ks"]].median().to_dict()

        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), constrained_layout=True)
        order = list(dict.fromkeys(tw.label))
        for ax, (y, ylab), letter in ((axs[0], ("fc_r", "FC r"), "a"), (axs[1], ("fcd_ks", "FCD KS distance"), "b")):
            for s, g in tw.groupby("subject"):
                g = g.set_index("label").loc[[o for o in order if o in g.label.values]]
                ax.plot(g.spec_r, g[y], color=NEUTRAL_C, lw=0.6, alpha=0.7)
            med = tw.groupby("label")[["spec_r", y]].median().loc[order]
            ax.plot(med.spec_r, med[y], color=C_FIT, lw=2, marker="o", ms=5, zorder=3)
            for lab, r in med.iterrows():
                ax.annotate(lab if lab in ("eeg-only", "bold-only") else f"w = {lab}", (r.spec_r, r[y]),
                            textcoords="offset points", xytext=(4, 4), fontsize=6, color=INK_2)
            ax.set_xlabel("EEG spectral r")
            ax.set_ylabel(ylab)
            panel(ax, letter)
        fig.suptitle("EEG weight trade-off (thin lines: subjects, thick: median)", fontsize=8)
        out.fig(fig, "fig3_weight_tradeoff")

    if not num:
        warnings.warn("nothing to compare: pass --eeg-only-root/--bold-only-root and/or --weight-roots")
        return
    out.numbers("section3_joint_vs_single", num)


if __name__ == "__main__":
    main()
