#!/usr/bin/env python
"""Article section 4 -- what is identifiable.

  Parameter recovery (fig4_recovery, tables/recovery_*.csv; ``--recovery-root``)
      ``gpu_fit.py recovery`` runs: true vs recovered per parameter (node
      level, within subject), and -- the part individual differences rest
      on -- the subject-specific deviation from the group map
      (true - group median vs recovered - group median); G pooled across runs.
  Fresh noise without refitting (fig4_fresh_noise; ``--fresh-noise-root``)
      ``gpu_fit.py fresh-noise`` runs: fitted params scored under new noise
      seeds vs the seed they were fitted with, and vs the starting params
      under the same new seeds.
  Existing diagnostics (fig4_diagnostics, tables/diag_*.csv; always)
      subject-level trade-offs of G with every parameter, what PC1 of the
      parameter space tracks (G, fit quality), and parameters sitting at a
      bound per region group (e.g. mu at its lower bound in cerebellum).

Refit stability (init x noise ICC, init-vs-noise split, memory of the start)
is the population report's ``--replicate-roots`` analysis -- see
``run_article.sh``.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from _common import (C_FIT, C_INIT, DEFAULT_BIDS, DEFAULT_ROOT, INK_2, NEUTRAL_C, Out, load_runs, panel, plt,
                     style, summary, wilcoxon)


# ---------------------------------------------------------------------------
# recovery

def recovery(root: Path, pop_root: Path, bids: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from gpu_fit import population_maps
    from params_report_cli import LIVE_ON
    med, _, _ = population_maps(pop_root, bids)
    rows, pts, gs = [], [], []
    for run in load_runs(root, need_sim=False):
        f = run.path / "true_params.npz"
        if not f.exists():
            continue
        truth = dict(np.load(f))
        syn = np.load(run.path / "synthetic_data.npz")
        tc, surface = syn["to_conn"], syn["mask_cortical"] > 0.5
        rec, start = run.params("best"), run.params("init")
        bounds = {lp["name"]: (lp["low"], lp["high"]) for lp in run.cfg["learnable_params"]}
        for p, t in truth.items():
            r = rec[p]
            lo, hi = bounds[p]
            if p == "G":
                gs.append({"run": run.path.name, "true": float(t[0]), "recovered": float(r[0]), "start": float(start[p][0])})
                continue
            m = surface if LIVE_ON.get(p) == "surface" else ~surface
            g = med[p][tc][m]
            ok = np.isfinite(g)
            t, r, g = t[m][ok], r[m][ok], g[ok]
            dev_t, dev_r = t - g, r - g
            rows.append({"run": run.path.name, "subject": run.sid, "param": p, "n_nodes": len(t),
                         "rho_nodes": stats.spearmanr(t, r)[0], "r_nodes": stats.pearsonr(t, r)[0],
                         "rho_deviation": stats.spearmanr(dev_t, dev_r)[0],
                         "slope": float(np.polyfit(t, r, 1)[0]),
                         "nrmse": float(np.sqrt(np.mean((r - t) ** 2)) / (hi - lo)),
                         "frac_moved": float(np.mean(np.abs(r - start[p][m][ok]) > 0.01 * (hi - lo))),
                         "final_loss_eeg": run.losses["loss_eeg"], "final_loss_bold": run.losses["loss_bold"]})
            pts += [{"param": p, "true": a, "recovered": b, "run": run.path.name} for a, b in zip(t, r)]
    return pd.DataFrame(rows), pd.DataFrame(pts), pd.DataFrame(gs)


def fig_recovery(tab: pd.DataFrame, pts: pd.DataFrame, gs: pd.DataFrame, out: Out) -> None:
    params = list(dict.fromkeys(tab.param)) + (["G"] if len(gs) else [])
    fig, axs = plt.subplots(2, len(params), figsize=(1.2 * len(params) + 1.5, 4.2), constrained_layout=True,
                            squeeze=False)
    for j, p in enumerate(params):
        ax = axs[0, j]
        if p == "G":
            ax.scatter(gs.true, gs.recovered, s=14, color=C_FIT, lw=0)
            x = gs.true
            title = f"G (r = {stats.pearsonr(gs.true, gs.recovered)[0]:.2f})" if len(gs) > 2 else "G"
        else:
            d = pts[pts.param == p]
            ax.scatter(d.true, d.recovered, s=2, color=C_FIT, lw=0, alpha=0.3)
            x = d.true
            title = f"{p} (rho {tab[tab.param == p].rho_nodes.median():.2f})"
        lim = [np.nanmin(x), np.nanmax(x)]
        ax.plot(lim, lim, color=INK_2, lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("true")
        if j == 0:
            ax.set_ylabel("recovered")
            panel(ax, "a")
    for j, p in enumerate(params):
        ax = axs[1, j]
        if p == "G":
            ax.axis("off")
            continue
        t = tab[tab.param == p]
        ax.boxplot([t.rho_nodes.dropna(), t.rho_deviation.dropna()], widths=0.5, showfliers=False)
        ax.set_xticks([1, 2], ["map", "subject\ndeviation"])
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.set_ylim(-0.5, 1.05)
        if j == 0:
            ax.set_ylabel("Spearman true vs recovered")
            panel(ax, "b")
    out.fig(fig, "fig4_recovery")


# ---------------------------------------------------------------------------
# fresh noise

def fresh_noise(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per = [], []
    for f in sorted(root.rglob("fresh_noise.json")):
        d = json.loads(f.read_text())
        r = pd.DataFrame(d["rows"])
        r["subject"] = d["subject"]
        rows.append(r)
        per.append({"subject": d["subject"], "reproduction_max_rel_diff": d["reproduction_max_rel_diff"]})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(rows, ignore_index=True), pd.DataFrame(per)


def fig_fresh_noise(fn: pd.DataFrame, out: Out) -> pd.DataFrame:
    keys = [k for k in ("spec_r", "alpha_topo_r", "fc_r", "fcd_ks") if k in fn]
    orig = fn[fn.original_seed].set_index("subject")
    new_fit = fn[~fn.original_seed & (fn.params == "fitted")].groupby("subject")[keys].mean()
    new_start = fn[~fn.original_seed & (fn.params == "start")].groupby("subject")[keys].mean()
    summ = pd.concat({"original_seed": orig[keys], "new_seeds_fitted": new_fit, "new_seeds_start": new_start}, axis=1)
    fig, axs = plt.subplots(1, len(keys), figsize=(1.8 * len(keys), 2.6), constrained_layout=True, squeeze=False)
    for ax, k, letter in zip(axs[0], keys, "abcd"):
        cols = [summ["original_seed"][k], summ["new_seeds_fitted"][k], summ["new_seeds_start"][k]]
        for a, b in zip(cols[0], cols[1]):
            ax.plot([0, 1], [a, b], color=NEUTRAL_C, lw=0.5)
        for j, (c, col) in enumerate(zip(cols, [C_FIT, C_FIT, C_INIT])):
            ax.scatter(np.full(len(c), j), c, s=10, color=col, lw=0, alpha=0.9 if j < 2 else 0.6, zorder=3)
        ax.set_xticks([0, 1, 2], ["fitted\nseed", "fitted\nnew seeds", "start\nnew seeds"])
        ax.set_title(k)
        panel(ax, letter)
    out.fig(fig, "fig4_fresh_noise")
    return summ


# ---------------------------------------------------------------------------
# existing diagnostics

def diagnostics(root: Path, bids: Path, out: Out) -> dict:
    from population_params_report_cli import find_frozen, load_population
    from params_report_cli import LIVE_ON
    pop = load_population(root, bids, bids / "participants.tsv")
    meta, regions = pop["meta"], pop["regions"]
    frozen, _ = find_frozen(pop)
    params = [p for p in pop["params"] if p not in frozen]

    # trade-offs: subject-level G vs node-mean of every param (live nodes)
    feats = {}
    for p in params:
        live = [r["idx"] for r in regions if r["is_surface"] == (LIVE_ON[p] == "surface")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            feats[f"mean {p}"] = np.nanmean(pop["V"][p][:, live], axis=1)
    F = pd.DataFrame(feats)
    trade = pd.DataFrame([{"param": c, "rho_with_G": stats.spearmanr(F[c], meta.G, nan_policy="omit")[0],
                           "p": stats.spearmanr(F[c], meta.G, nan_policy="omit")[1]} for c in F])
    out.table(trade, "diag_G_tradeoffs")

    # PC1 of the node-level parameter space
    blocks = []
    for p in params:
        live = [r["idx"] for r in regions if r["is_surface"] == (LIVE_ON[p] == "surface")]
        blocks.append(pop["V"][p][:, live])
    X = np.column_stack(blocks + [meta.G])
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X[:, np.isfinite(X).any(0)]))
    keep = ~np.all(X == 0, axis=0)
    pcs = PCA(n_components=5, random_state=0).fit(X[:, keep])
    S = pcs.transform(X[:, keep])
    pc_rows = []
    for c in range(5):
        row = {"pc": c + 1, "var_explained": pcs.explained_variance_ratio_[c]}
        for k in ("G", "log_eeg_psd_loss", "log_bold_loss", "fc_corr", "age"):
            row[f"rho_{k}"] = stats.spearmanr(S[:, c], meta[k], nan_policy="omit")[0]
        pc_rows.append(row)
    pcdf = pd.DataFrame(pc_rows)
    out.table(pcdf, "diag_pca")

    # at-bound fractions per param x region group
    rows = []
    for p in params:
        lo, hi = pop["bounds"][p]
        tol = 0.01 * (hi - lo)
        for g in dict.fromkeys(r["group"] for r in regions):
            idx = [r["idx"] for r in regions if r["group"] == g and r["is_surface"] == (LIVE_ON[p] == "surface")]
            if not idx:
                continue
            V = pop["V"][p][:, idx]
            V = V[np.isfinite(V)]
            if not len(V):
                continue
            rows.append({"param": p, "group": g, "n_values": len(V),
                         "frac_at_low": float(np.mean(V - lo < tol)), "frac_at_high": float(np.mean(hi - V < tol))})
    bnd = pd.DataFrame(rows)
    out.table(bnd, "diag_at_bounds")

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    ax = axs[0]
    ax.barh(trade.param, trade.rho_with_G, color=[C_FIT if v > 0 else C_INIT for v in trade.rho_with_G], height=0.6)
    ax.axvline(0, color=INK_2, lw=0.6)
    ax.set_xlabel("Spearman rho with G (subjects)")
    ax.set_title("trade-offs with G")
    panel(ax, "a")
    ax = axs[1]
    ax.scatter(meta.G, S[:, 0], s=8, c=meta.log_eeg_psd_loss, cmap="viridis", lw=0)
    ax.set_xlabel("G")
    ax.set_ylabel("PC1 score")
    ax.set_title(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%}) vs G, colour = log EEG loss")
    panel(ax, "b")
    ax = axs[2]
    piv = bnd.pivot(index="group", columns="param", values="frac_at_low").fillna(np.nan)
    im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction at lower bound")
    ax.set_title("parameters at their lower bound")
    panel(ax, "c")
    out.fig(fig, "fig4_diagnostics")
    return {"frozen_params": frozen, "G_tradeoffs": trade.set_index("param").rho_with_G.to_dict(),
            "pca": pcdf.set_index("pc").to_dict(orient="index"),
            "at_low_bound_top": bnd.sort_values("frac_at_low", ascending=False).head(8).to_dict(orient="records")}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/article")
    ap.add_argument("--recovery-root", type=Path, default=None)
    ap.add_argument("--fresh-noise-root", type=Path, default=None)
    args = ap.parse_args()
    out = Out(args.out or args.root / "article")
    style()
    num = {"diagnostics": diagnostics(args.root, args.bids, out)}

    if args.recovery_root:
        tab, pts, gs = recovery(args.recovery_root, args.root, args.bids)
        if len(tab):
            out.table(tab, "recovery_per_run")
            out.table(gs, "recovery_G")
            agg = tab.groupby("param")[["rho_nodes", "rho_deviation", "slope", "nrmse", "frac_moved"]].median()
            out.table(agg.reset_index(), "recovery_summary")
            num["recovery"] = {"n_runs": int(tab.run.nunique()), "per_param_median": agg.to_dict(orient="index")}
            if len(gs) > 2:
                num["recovery"]["G"] = {"pearson_r": stats.pearsonr(gs.true, gs.recovered)[0],
                                        "median_abs_err": float((gs.true - gs.recovered).abs().median())}
            fig_recovery(tab, pts, gs, out)
        else:
            warnings.warn(f"no true_params.npz under {args.recovery_root}")

    if args.fresh_noise_root:
        fn, per = fresh_noise(args.fresh_noise_root)
        if len(fn):
            out.table(fn, "fresh_noise_rows")
            summ = fig_fresh_noise(fn, out)
            out.table(summ, "fresh_noise_summary", index=True)
            keys = [k for k in ("spec_r", "alpha_topo_r", "fc_r", "fcd_ks") if k in fn]
            num["fresh_noise"] = {
                "n_subjects": int(fn.subject.nunique()), "n_seeds": int(fn[~fn.original_seed].seed.nunique()),
                "reproduction_max_rel_diff": float(per.reproduction_max_rel_diff.max()),
                "median": {c: {k: float(summ[c][k].median()) for k in keys} for c in summ.columns.levels[0]},
                "new_fitted_vs_original": {k: wilcoxon(summ["new_seeds_fitted"][k], summ["original_seed"][k]) for k in keys},
                "new_fitted_vs_start": {k: wilcoxon(summ["new_seeds_fitted"][k], summ["new_seeds_start"][k]) for k in keys},
                "seed_sd_fitted_median": {k: float(fn[~fn.original_seed & (fn.params == "fitted")]
                                                   .groupby("subject")[k].std().median()) for k in keys},
            }
    out.numbers("section4_identifiability", num)


if __name__ == "__main__":
    main()
