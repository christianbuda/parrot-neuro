#!/usr/bin/env python
"""Article section 2 -- the model reproduces each subject's EEG and BOLD.

From one population of fits (default ``new_leadfields/1/atlas-100``) and
the BIDS derivatives, writes into ``--out``:

  EEG, example subject (fig2_eeg_example)
      empirical vs simulated PSD at a few channels + relative-alpha
      topography side by side.
  EEG, across subjects (fig2_eeg_population, tables/fit_quality_subjects.csv)
      per-channel spectral correlation, simulated vs empirical alpha peak
      frequency, alpha-topography correlation -- instead of the raw loss.
  BOLD (fig2_bold)
      empirical vs simulated FC (example), FCD histograms overlaid,
      fc_corr distribution.
  Baselines (fig2_baselines)
      SC-FC correlation per subject; the model at its starting parameters;
      noise ceilings: FC between the two halves of the empirical run, EEG
      PSD between two halves of the chunks -- and, because the simulated
      PSD is ONE 2-s realization, between one empirical chunk and the rest.
  Subject specificity (fig2_specificity)
      subject i's fitted model vs subject j's data, for EEG spectra and FC;
      identification accuracy, differential identifiability, permutation p.
  Held-out data (fig2_heldout, only with ``--heldout-root``)
      fits made on half the EEG chunks / half the BOLD run
      (``gpu_fit.py variant --variant half-A``) scored on the other half.

    pixi run python examples/article/fit_quality.py \\
        --root new_leadfields/1/atlas-100 --bids /srv/nfs-data/sisko/christian/parrot_LEMON

Reads saved results only (no jax / GPU).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common import (C_ALT, C_FIT, C_INIT, C_SC, DEFAULT_BIDS, DEFAULT_ROOT, INK_2, NEUTRAL_C, OLD_C, YOUNG_C,
                     Out, Subject, bandpass_bold, channel_names, eeg_chunks, eeg_metrics, fc, fc_r, fcd_ks,
                     holdout_blocks, load_runs, mean_fd, panel, participants, plt, psd, smooth,
                     spectral_similarity, style, summary, to_conn, upper, wilcoxon)

EXAMPLE_CHANNELS = ["Fz", "Cz", "Pz", "Oz"]


# ---------------------------------------------------------------------------
# per-subject metrics

def subject_row(run, subj, args, rng) -> tuple[dict, dict]:
    """All per-subject numbers + the arrays the example / specificity panels need."""
    z, cfg = run.sim, run.cfg
    freqs = z["eeg_freqs"]
    names = channel_names(run, subj)
    fmin, fmax = cfg["fmin"], cfg["fmax"]
    row = {"subject": run.sid, "n_nodes": int(z["fc_emp"].shape[0])}

    # -- EEG: fitted + start params
    for tag, key in (("fit", "eeg_psd_sim"), ("init", "eeg_psd_sim_init")):
        for k, v in eeg_metrics(z[key], z["eeg_psd_target"], freqs, names, fmin, fmax).items():
            row[f"{k}_{tag}" if not k.endswith("_emp") else k] = v

    # -- EEG noise ceilings from the subject's own chunks
    chunks, _ = eeg_chunks(subj, cfg["eeg_task"], cfg["chunk_length"])
    P = np.stack([psd(c) for c in chunks])
    row["n_chunks"] = len(chunks)
    half = holdout_blocks(len(chunks), args.block_chunks)
    m = eeg_metrics(P[half == 0].mean(0), P[half == 1].mean(0), freqs, names, fmin, fmax)
    row.update({"ceil_half_spec_r": m["spec_r"], "ceil_half_alpha_topo_r": m["alpha_topo_r"],
                "ceil_half_apf_absdiff": abs(m["apf_sim"] - m["apf_emp"])})
    picks = rng.choice(len(chunks), size=min(args.n_single_chunks, len(chunks)), replace=False)
    single = []
    for i in picks:
        rest = np.delete(P, i, axis=0).mean(0)
        single.append(eeg_metrics(P[i], rest, freqs, names, fmin, fmax))
    single = pd.DataFrame(single)
    row.update({"ceil_chunk_spec_r": single.spec_r.median(),
                "ceil_chunk_alpha_topo_r": single.alpha_topo_r.median(),
                "ceil_chunk_apf_absdiff": (single.apf_sim - single.apf_emp).abs().median()})

    # -- BOLD
    skip = int(z["bold_skip_trs"])
    row["fc_r_fit"] = fc_r(z["fc_sim"], z["fc_emp"])
    row["fc_r_init"] = fc_r(z["fc_sim_init"], z["fc_emp"])
    if "fcd_sim" in z.files:
        row["fcd_ks_fit"] = fcd_ks(z["fcd_sim"], z["fcd_emp"])
        row["fcd_ks_init"] = fcd_ks(z["fcd_sim_init"], z["fcd_emp"])
    emp = np.asarray(z["bold_emp"])[skip:]
    h = len(emp) // 2
    r_half = fc_r(fc(emp[:h]), fc(emp[h:]))
    row["ceil_fc_half_r"] = r_half
    row["ceil_fc_sb"] = 2 * r_half / (1 + r_half)

    W = subj.load.weights(cfg["atlas"], normalized=True, fmri_aligned=True)
    w, f = upper(W), upper(z["fc_emp"])
    row["sc_fc_r"] = float(np.corrcoef(w, f)[0, 1])
    nz = w > 0
    row["sc_fc_r_log"] = float(np.corrcoef(np.log10(w[nz]), f[nz])[0, 1])
    row["sc_fc_r_best"] = max(row["sc_fc_r"], row["sc_fc_r_log"])
    row["mean_fd"] = mean_fd(subj, cfg["fmri_task"])

    arrays = {"names": names, "freqs": freqs, "psd_sim": z["eeg_psd_sim"], "psd_emp": z["eeg_psd_target"],
              "to_conn": to_conn(subj, cfg), "fc_sim": z["fc_sim"], "fc_emp": z["fc_emp"]}
    return row, arrays


# ---------------------------------------------------------------------------
# specificity: subject i's model vs subject j's data

def identifiability(S: np.ndarray, n_perm: int, rng) -> dict:
    """``S[i, j]`` = similarity of model i to data j (higher = closer)."""
    n = len(S)
    off = ~np.eye(n, dtype=bool)
    diag = np.diag(S)
    acc_row = float(np.mean(np.nanargmax(S, axis=1) == np.arange(n)))   # which data fits my model best?
    acc_col = float(np.mean(np.nanargmax(S, axis=0) == np.arange(n)))   # which model fits my data best?
    rank = np.array([np.mean(S[i, off[i]] < diag[i]) for i in range(n)])  # percentile of own data
    idiff = float(np.nanmean(diag) - np.nanmean(S[off]))
    null = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        null.append(np.nanmean(S[np.arange(n), p]) - np.nanmean(S[off]))
    return {"n": n, "acc_model_to_data": acc_row, "acc_data_to_model": acc_col, "chance": 1 / n,
            "own_rank_median": float(np.median(rank)), "idiff": idiff,
            "self_mean": float(np.nanmean(diag)), "other_mean": float(np.nanmean(S[off])),
            "p_perm": float((1 + np.sum(np.array(null) >= idiff)) / (1 + n_perm)),
            "frac_self_beats_mean_other": float(np.mean(diag > np.array([np.nanmean(S[i, off[i]]) for i in range(n)])))}


def eeg_specificity(arrs: list[dict], fmin: float, fmax: float) -> np.ndarray:
    chans = sorted(set.intersection(*(set(a["names"]) for a in arrs)))
    idx = [[a["names"].index(c) for c in chans] for a in arrs]
    n = len(arrs)
    S = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = spectral_similarity(arrs[i]["psd_sim"][idx[i]], arrs[j]["psd_emp"][idx[j]],
                                          arrs[i]["freqs"], fmin, fmax)
    return S


def fc_specificity(arrs: list[dict]) -> tuple[np.ndarray, int]:
    """FC similarity on the connectome nodes every subject kept."""
    common = sorted(set.intersection(*(set(a["to_conn"].tolist()) for a in arrs)))

    def vec(a, key):
        pos = [int(np.where(a["to_conn"] == c)[0][0]) for c in common]
        v = upper(a[key][np.ix_(pos, pos)])
        return (v - v.mean()) / v.std()
    Vs = np.stack([vec(a, "fc_sim") for a in arrs])
    Ve = np.stack([vec(a, "fc_emp") for a in arrs])
    return Vs @ Ve.T / Vs.shape[1], len(common)


# ---------------------------------------------------------------------------
# held-out

def heldout(root: Path, bids: Path, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in load_runs(root):
        f = run.path / "holdout.json"
        if not f.exists():
            continue
        ho = json.loads(f.read_text())
        subj = Subject(str(bids), run.sid)
        z, cfg = run.sim, run.cfg
        freqs, names = z["eeg_freqs"], channel_names(run, subj)
        chunks, _ = eeg_chunks(subj, cfg["eeg_task"], cfg["chunk_length"])
        P = np.stack([psd(c) for c in chunks])
        lab = holdout_blocks(len(chunks), ho["block_chunks"])
        train, test = P[lab == ho["eeg_half"]].mean(0), P[lab != ho["eeg_half"]].mean(0)
        assert np.allclose(train, z["eeg_psd_target"]), f"{run.sid}: train half != fit target"
        m_in = eeg_metrics(z["eeg_psd_sim"], train, freqs, names, cfg["fmin"], cfg["fmax"])
        m_out = eeg_metrics(z["eeg_psd_sim"], test, freqs, names, cfg["fmin"], cfg["fmax"])
        m_ceil = eeg_metrics(train, test, freqs, names, cfg["fmin"], cfg["fmax"])

        skip = int(z["bold_skip_trs"])
        full = np.asarray(subj.load.fmri_timeseries(variant="conn", task=cfg["fmri_task"])[f"ts_{cfg['atlas']}"])
        keep = np.asarray(subj.load.fmri_keep(cfg["atlas"], cfg["fmri_task"]))
        X = full[keep].T
        a, b = ho["bold_train_trs"], ho["bold_test_trs"]
        fc_train, fc_test = fc(X[a[0]:a[1]][skip:]), fc(X[b[0]:b[1]][skip:])
        rows.append({
            "subject": run.sid, "half": ho["eeg_half"],
            "spec_r_train": m_in["spec_r"], "spec_r_test": m_out["spec_r"], "spec_r_ceil": m_ceil["spec_r"],
            "alpha_topo_r_train": m_in["alpha_topo_r"], "alpha_topo_r_test": m_out["alpha_topo_r"],
            "alpha_topo_r_ceil": m_ceil["alpha_topo_r"],
            "apf_absdiff_train": abs(m_in["apf_sim"] - m_in["apf_emp"]),
            "apf_absdiff_test": abs(m_out["apf_sim"] - m_out["apf_emp"]),
            "fc_r_train": fc_r(z["fc_sim"], fc_train), "fc_r_test": fc_r(z["fc_sim"], fc_test),
            "fc_r_ceil": fc_r(fc_train, fc_test),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.merge(base[["subject", "fc_r_fit", "spec_r_fit"]].rename(
            columns={"fc_r_fit": "fc_r_fullfit", "spec_r_fit": "spec_r_fullfit"}), on="subject", how="left")
    return df


# ---------------------------------------------------------------------------
# figures

def _box(ax, data: list, labels: list, colors: list, ylabel: str, ref=None, ref_label=None):
    pos = np.arange(len(data))
    for p, d, c in zip(pos, data, colors):
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        ax.boxplot(d, positions=[p], widths=0.5, showfliers=False, patch_artist=True,
                   medianprops={"color": "white", "lw": 1.5},
                   boxprops={"facecolor": c, "edgecolor": c}, whiskerprops={"color": c}, capprops={"color": c})
        jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(d))
        ax.scatter(p + jit, d, s=4, color=c, alpha=0.35, lw=0, zorder=3)
    if ref is not None:
        ax.axhline(ref, color=INK_2, lw=0.8, ls="--")
        if ref_label:
            ax.text(len(data) - 0.5, ref, ref_label, ha="right", va="bottom", fontsize=6, color=INK_2)
    ax.set_xticks(pos, labels)
    ax.set_ylabel(ylabel)


def _topo(ax, pos2d: dict, names: list[str], values: np.ndarray, vmin, vmax, title):
    from matplotlib.tri import Triangulation
    xy = np.array([pos2d[n] for n in names])
    tri = Triangulation(xy[:, 1], xy[:, 0])
    cs = ax.tricontourf(tri, values, levels=np.linspace(vmin, vmax, 12), cmap="viridis", extend="both")
    ax.scatter(xy[:, 1], xy[:, 0], s=3, color="white", lw=0)
    r = np.abs(xy).max() * 1.08
    ax.add_patch(plt.Circle((0, 0), r, fill=False, color=INK_2, lw=0.8))
    ax.plot([-0.12 * r, 0, 0.12 * r], [r, 1.12 * r, r], color=INK_2, lw=0.8)
    ax.set_xlim(-1.2 * r, 1.2 * r)
    ax.set_ylim(-1.2 * r, 1.25 * r)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=10)
    return cs


def fig_eeg_example(arr: dict, row: dict, pos2d: dict, cfg: dict, out: Out) -> None:
    freqs, names = arr["freqs"], arr["names"]
    band = (freqs >= cfg["fmin"]) & (freqs <= 30)
    S, E = smooth(arr["psd_sim"]), smooth(arr["psd_emp"])
    fit = (freqs >= cfg["fmin"]) & (freqs <= cfg["fmax"])
    chans = [c for c in EXAMPLE_CHANNELS if c in names] or names[:4]
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.55, wspace=0.35)
    for k, c in enumerate(chans):
        ax = fig.add_subplot(gs[0, k])
        i = names.index(c)
        # both scaled to their own fitted-band power, as the loss compares them
        ax.semilogy(freqs[band], E[i, band] / E[i, fit].sum(), color=INK_2, label="empirical")
        ax.semilogy(freqs[band], S[i, band] / S[i, fit].sum(), color=C_FIT, label="model")
        ax.axvspan(cfg["fmax"], 30, color="#f0efec", lw=0)
        ax.set_title(c)
        ax.set_xlabel("Hz")
        if k == 0:
            ax.set_ylabel("relative power")
            ax.legend(frameon=False, loc="lower left")
            panel(ax, "a")
    a = (freqs >= 8) & (freqs <= 12)
    rel = lambda P: P[:, a].sum(1) / P[:, fit].sum(1)  # noqa: E731
    re, rs = rel(E), rel(S)
    lo, hi = np.percentile(np.r_[re, rs], [2, 98])
    ax1 = fig.add_subplot(gs[1, 0:2])
    _topo(ax1, pos2d, names, re, lo, hi, "empirical alpha (8-12 Hz / 1-15 Hz)")
    panel(ax1, "b")
    ax2 = fig.add_subplot(gs[1, 2:4])
    cs = _topo(ax2, pos2d, names, rs, lo, hi, f"model  (r = {row['alpha_topo_r_fit']:.2f})")
    fig.colorbar(cs, ax=[ax1, ax2], shrink=0.7, label="relative alpha power")
    fig.suptitle(f"sub-{row['subject']}: spectral r = {row['spec_r_fit']:.2f}, "
                 f"alpha peak {row['apf_emp']:.1f} Hz (data) vs {row['apf_sim_fit']:.1f} Hz (model)", fontsize=8)
    out.fig(fig, "fig2_eeg_example")


def fig_eeg_population(df: pd.DataFrame, out: Out) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    _box(axs[0], [df.spec_r_init, df.spec_r_fit, df.ceil_chunk_spec_r, df.ceil_half_spec_r],
         ["start", "fitted", "1 chunk\nvs rest", "half vs\nhalf"], [C_INIT, C_FIT, NEUTRAL_C, NEUTRAL_C],
         "spectral r (median over channels)")
    panel(axs[0], "a")
    ax = axs[1]
    old = df.old.to_numpy(bool)
    for m, c, lab in ((~old, YOUNG_C, "young"), (old, OLD_C, "old")):
        ax.scatter(df.apf_emp[m], df.apf_sim_fit[m], s=8, color=c, lw=0, alpha=0.8, label=lab)
    lim = [np.nanmin(np.r_[df.apf_emp, df.apf_sim_fit]) - 0.2, np.nanmax(np.r_[df.apf_emp, df.apf_sim_fit]) + 0.2]
    ax.plot(lim, lim, color=INK_2, lw=0.8, ls="--")
    ok = df[["apf_emp", "apf_sim_fit"]].dropna()
    r, p = stats.pearsonr(ok.apf_emp, ok.apf_sim_fit)
    ax.set_title(f"alpha peak frequency (r = {r:.2f})")
    ax.set_xlabel("empirical (Hz)")
    ax.set_ylabel("model (Hz)")
    ax.legend(frameon=False)
    panel(ax, "b")
    _box(axs[2], [df.alpha_topo_r_init, df.alpha_topo_r_fit, df.ceil_chunk_alpha_topo_r, df.ceil_half_alpha_topo_r],
         ["start", "fitted", "1 chunk\nvs rest", "half vs\nhalf"], [C_INIT, C_FIT, NEUTRAL_C, NEUTRAL_C],
         "alpha topography r")
    panel(axs[2], "c")
    out.fig(fig, "fig2_eeg_population")


def fig_bold(arr: dict, run, df: pd.DataFrame, regions_order, out: Out) -> None:
    z = run.sim
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 4, wspace=0.45, width_ratios=[1, 1, 1.1, 1])
    o = regions_order
    for k, (key, t) in enumerate((("fc_emp", "empirical FC"), ("fc_sim", f"model FC (r = {fc_r(z['fc_sim'], z['fc_emp']):.2f})"))):
        ax = fig.add_subplot(gs[0, k])
        M = z[key][np.ix_(o, o)].copy()
        np.fill_diagonal(M, np.nan)
        lim = np.nanpercentile(np.abs(M), 98)
        ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if k == 0:
            panel(ax, "a")
    ax = fig.add_subplot(gs[0, 2])
    kmin = int(run.cfg.get("dfc_kmin", 1))
    lo, hi = np.percentile(np.r_[upper(z["fcd_emp"], kmin), upper(z["fcd_sim"], kmin)], [0.5, 99.5])
    bins = np.linspace(lo, hi, 40)
    for key, c, lab in (("fcd_emp", INK_2, "empirical"), ("fcd_sim_init", C_INIT, "start"), ("fcd_sim", C_FIT, "model")):
        if key in z.files:
            ax.hist(upper(z[key], kmin), bins=bins, density=True, histtype="step", color=c, lw=1.5, label=lab)
    ax.set_xlabel("FCD value (corr. between windows)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("FCD distribution")
    panel(ax, "b")
    ax = fig.add_subplot(gs[0, 3])
    _box(ax, [df.fc_r_fit], ["fitted"], [C_FIT], "FC correlation (sim vs emp)")
    ax.set_title(f"median {df.fc_r_fit.median():.2f}, n = {df.fc_r_fit.notna().sum()}")
    panel(ax, "c")
    out.fig(fig, "fig2_bold")


def fig_baselines(df: pd.DataFrame, out: Out) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.5, 1, 1]})
    _box(axs[0], [df.fc_r_init, df.sc_fc_r_best, df.fc_r_fit, df.ceil_fc_half_r],
         ["model at\nstart", "SC-FC", "model\nfitted", "half vs half\n(ceiling)"],
         [C_INIT, C_SC, C_FIT, NEUTRAL_C], "FC correlation with empirical FC")
    panel(axs[0], "a")
    ax = axs[1]
    ax.scatter(df.sc_fc_r_best, df.fc_r_fit, s=8, color=C_FIT, lw=0, alpha=0.8)
    lim = [min(df.sc_fc_r_best.min(), df.fc_r_fit.min()) - 0.02, max(df.sc_fc_r_best.max(), df.fc_r_fit.max()) + 0.02]
    ax.plot(lim, lim, color=INK_2, lw=0.8, ls="--")
    ax.set_xlabel("SC-FC r")
    ax.set_ylabel("model FC r")
    ax.set_title(f"model > SC in {np.mean(df.fc_r_fit > df.sc_fc_r_best):.0%}")
    panel(ax, "b")
    ax = axs[2]
    for a, b, c in ((df.spec_r_init, df.spec_r_fit, INK_2),):
        for x0, x1 in zip(a, b):
            ax.plot([0, 1], [x0, x1], color=NEUTRAL_C, lw=0.4, alpha=0.5)
    ax.scatter(np.zeros(len(df)), df.spec_r_init, s=6, color=C_INIT, lw=0)
    ax.scatter(np.ones(len(df)), df.spec_r_fit, s=6, color=C_FIT, lw=0)
    ax.set_xticks([0, 1], ["start", "fitted"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("EEG spectral r")
    ax.set_title("EEG: start vs fitted")
    panel(ax, "c")
    out.fig(fig, "fig2_baselines")


def fig_specificity(S_eeg, S_fc, id_eeg, id_fc, order, out: Out) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, S, t, letter in ((axs[0], S_eeg, "EEG spectra", "a"), (axs[1], S_fc, "BOLD FC", "b")):
        M = S[np.ix_(order, order)]
        im = ax.imshow(M, cmap="viridis", interpolation="nearest",
                       vmin=np.percentile(M, 2), vmax=np.percentile(M, 99.5))
        ax.set_xlabel("data of subject j")
        ax.set_ylabel("model of subject i")
        ax.set_title(t)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
        panel(ax, letter)
    ax = axs[2]
    for k, (S, c, lab) in enumerate(((S_eeg, C_FIT, "EEG"), (S_fc, C_SC, "FC"))):
        n = len(S)
        off = ~np.eye(n, dtype=bool)
        own = np.diag(S)
        other = np.array([S[i, off[i]].mean() for i in range(n)])
        d = own - other
        ax.boxplot(d, positions=[k], widths=0.5, showfliers=False, patch_artist=True,
                   medianprops={"color": "white"}, boxprops={"facecolor": c, "edgecolor": c},
                   whiskerprops={"color": c}, capprops={"color": c})
    ax.axhline(0, color=INK_2, lw=0.8, ls="--")
    ax.set_xticks([0, 1], [f"EEG\nacc {id_eeg['acc_data_to_model']:.0%}", f"FC\nacc {id_fc['acc_data_to_model']:.0%}"])
    ax.set_ylabel("own data - mean other data")
    ax.set_title("self advantage")
    panel(ax, "c")
    out.fig(fig, "fig2_specificity")


def fig_heldout(ho: pd.DataFrame, out: Out) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.6), constrained_layout=True)
    for ax, key, lab, letter in ((axs[0], "spec_r", "EEG spectral r", "a"), (axs[1], "fc_r", "FC r", "b")):
        _box(ax, [ho[f"{key}_train"], ho[f"{key}_test"], ho[f"{key}_ceil"]],
             ["fitted half", "held-out\nhalf", "half vs half\n(ceiling)"], [C_FIT, C_ALT, NEUTRAL_C], lab)
        panel(ax, letter)
    out.fig(fig, "fig2_heldout")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="population of fits")
    ap.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/article")
    ap.add_argument("--example", default=None, help="example subject (default: median spectral r)")
    ap.add_argument("--block-chunks", type=int, default=15,
                    help="EEG halves = alternating blocks of this many 2-s chunks (as gpu_fit.py half-A/B)")
    ap.add_argument("--n-single-chunks", type=int, default=20, help="chunks drawn for the 1-chunk ceiling")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--heldout-root", type=Path, default=None, help="fits from gpu_fit.py --variant half-A/half-B")
    ap.add_argument("--age-split", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Out(args.out or args.root / "article")
    style()
    rng = np.random.default_rng(args.seed)

    runs = load_runs(args.root)
    print(f"[fit quality] {len(runs)} subjects under {args.root}")
    rows, arrs = [], []
    for k, run in enumerate(runs):
        row, arr = subject_row(run, Subject(str(args.bids), run.sid), args, rng)
        rows.append(row)
        arrs.append(arr)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(runs)}")
    df = pd.DataFrame(rows).merge(participants(args.bids, args.age_split)[["subject", "age", "sex", "old"]],
                                  on="subject", how="left")
    df["apf_absdiff_fit"] = (df.apf_sim_fit - df.apf_emp).abs()
    df["apf_absdiff_init"] = (df.apf_sim_init - df.apf_emp).abs()
    out.table(df, "fit_quality_subjects")

    # -- headline numbers
    ok = df[["apf_emp", "apf_sim_fit"]].dropna()
    num = {
        "n_subjects": len(df),
        "eeg": {k: summary(df[k]) for k in ("spec_r_fit", "spec_r_init", "ceil_chunk_spec_r", "ceil_half_spec_r",
                                             "alpha_topo_r_fit", "alpha_topo_r_init", "ceil_chunk_alpha_topo_r",
                                             "ceil_half_alpha_topo_r", "apf_absdiff_fit", "apf_absdiff_init")},
        "apf_sim_vs_emp": {"pearson_r": stats.pearsonr(ok.apf_emp, ok.apf_sim_fit)[0],
                           "pearson_p": stats.pearsonr(ok.apf_emp, ok.apf_sim_fit)[1],
                           "spearman_rho": stats.spearmanr(ok.apf_emp, ok.apf_sim_fit)[0], "n": len(ok)},
        "bold": {k: summary(df[k]) for k in ("fc_r_fit", "fc_r_init", "sc_fc_r", "sc_fc_r_log", "sc_fc_r_best",
                                              "ceil_fc_half_r", "ceil_fc_sb", "fcd_ks_fit", "fcd_ks_init")
                 if k in df},
        "tests": {
            "fc_fit_vs_sc_best": wilcoxon(df.fc_r_fit, df.sc_fc_r_best),
            "fc_fit_vs_init": wilcoxon(df.fc_r_fit, df.fc_r_init),
            "spec_r_fit_vs_init": wilcoxon(df.spec_r_fit, df.spec_r_init),
            "alpha_topo_fit_vs_init": wilcoxon(df.alpha_topo_r_fit, df.alpha_topo_r_init),
            "spec_r_fit_vs_single_chunk_ceiling": wilcoxon(df.spec_r_fit, df.ceil_chunk_spec_r),
            "fc_fit_vs_half_ceiling": wilcoxon(df.fc_r_fit, df.ceil_fc_half_r),
        },
        "fc_fit_over_ceiling_median": float(np.median(df.fc_r_fit / df.ceil_fc_half_r)),
    }

    # -- figures
    ex = args.example or df.iloc[(df.spec_r_fit - df.spec_r_fit.median()).abs().argsort().iloc[0]].subject
    i = int(np.where(df.subject == ex)[0][0])
    from population_params_report_cli import scalp_positions_2d
    from params_report_cli import GROUP_ORDER, classify
    subj = Subject(str(args.bids), ex)
    fig_eeg_example(arrs[i], df.iloc[i].to_dict(), scalp_positions_2d(subj), runs[i].cfg, out)
    fig_eeg_population(df, out)
    labels = subj.load.connectivity_labels(runs[i].cfg["atlas"])
    lab = [labels[c] for c in arrs[i]["to_conn"]]
    order = sorted(range(len(lab)), key=lambda k: (GROUP_ORDER.index(classify(lab[k])["group"])
                                                    if classify(lab[k])["group"] in GROUP_ORDER else 99, k))
    fig_bold(arrs[i], runs[i], df, order, out)
    fig_baselines(df, out)

    print("[fit quality] specificity matrices ...")
    S_eeg = eeg_specificity(arrs, runs[0].cfg["fmin"], runs[0].cfg["fmax"])
    S_fc, n_common = fc_specificity(arrs)
    id_eeg = identifiability(S_eeg, args.n_perm, rng)
    id_fc = identifiability(S_fc, args.n_perm, rng)
    id_fc["n_common_nodes"] = n_common
    num["specificity"] = {"eeg": id_eeg, "fc": id_fc}
    out.table(pd.DataFrame(S_eeg, index=df.subject, columns=df.subject), "specificity_eeg", index=True)
    out.table(pd.DataFrame(S_fc, index=df.subject, columns=df.subject), "specificity_fc", index=True)
    age_order = np.argsort(df.age.to_numpy())
    fig_specificity(S_eeg, S_fc, id_eeg, id_fc, age_order, out)

    if args.heldout_root:
        ho = heldout(args.heldout_root, args.bids, df)
        if len(ho):
            out.table(ho, "heldout")
            num["heldout"] = {k: summary(ho[k]) for k in ho.columns if k not in ("subject", "half")}
            num["heldout"]["tests"] = {"spec_r_test_vs_train": wilcoxon(ho.spec_r_test, ho.spec_r_train),
                                       "fc_r_test_vs_train": wilcoxon(ho.fc_r_test, ho.fc_r_train),
                                       "fc_r_test_vs_ceiling": wilcoxon(ho.fc_r_test, ho.fc_r_ceil)}
            fig_heldout(ho, out)
        else:
            warnings.warn(f"no holdout.json runs under {args.heldout_root}")

    out.numbers("section2_fit_quality", num)
    print(json.dumps({k: num[k] for k in ("n_subjects", "apf_sim_vs_emp", "fc_fit_over_ceiling_median")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
