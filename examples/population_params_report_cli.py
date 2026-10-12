#!/usr/bin/env python
"""Population analysis of TVB-optim per-node parameters across many subjects.

The population counterpart of ``params_report_cli.py``: given a folder of
``eeg_bold_fit_cli.py`` runs (one ``<subject>_*`` sub-folder per subject, each
holding ``optimized_params.npz``, ``config.json``, ``diagnostics_metrics.json``),
it answers three questions:

  1. **Regional variation** -- how does each fitted parameter vary per region
     across subjects, and is the regional pattern reproducible (split-half
     reliability of the group map, subject-to-subject map similarity, ICC)?
  2. **Clustering** -- do subjects cluster in parameter space (PCA + k-means,
     silhouette-selected k), and do regions cluster by functional network?
     Which parameters trade off against each other?
  3. **Age** -- do parameters differ between the young and old groups of
     ``participants.tsv`` (per region, per functional network, globally), and
     can the whole parameter set predict the age group better than the fit
     quality alone can?

  4. **Condition** (optional, ``--compare-root``) -- the same subjects fit
     under a second condition (e.g. eyes open vs eyes closed EEG): how stable
     is each subject's parameter set across conditions (ICC, subject
     identification from individual deviation maps), and which parameters
     shift systematically (paired Wilcoxon per region / network / globally,
     adjusted for the change in fit quality, plus a subject-grouped classifier)?

  5. **Refit stability** (optional, ``--replicate-roots``) -- the same
     subjects refit with other ``--init-seed`` / ``--noise-seed`` values:
     convergence, ICC (subject vs refit variance, split into init and noise),
     identification across refits, init memory, group-map stability,
     trade-offs, and whether the age / condition conclusions survive.

Everything is run on two cohorts: all ``gamma_weight == 0`` fits ("all") and
the subset whose BOLD fit worked (``fc_corr > --fc-threshold``, "bold_ok").
Runs with ``gamma_weight != 0`` are excluded and listed.

Node alignment: each subject's optimized nodes are a subject-specific subset of
the connectome (BOLD-coverage mask), so they are scattered back onto the shared
connectome-row axis with ``Subject.load.fmri_nodes(atlas).to_conn`` -- this is
why ``--bids-dir`` is required. Each parameter is analysed only on the node type
where it is live (see ``params_report_cli.LIVE_ON``); parameters that never moved
(e.g. ``P``) are reported and dropped.

Fit quality covariates: ``eeg_psd_loss`` (mean of the last 10 epochs of the
training EEG loss, i.e. the normalized-PSD MSE, log10) and ``fc_corr``.

Writes into ``--out`` (default ``<results-root>/population_report``):
  - ``population_report.html`` -- self-contained interactive report
  - ``summary.md`` + ``figures/*.png`` -- static written summary
  - ``tables/*.csv`` -- subjects, per-region stats, age tests, clusters

    pixi run python examples/population_params_report_cli.py \\
        --results-root fit_test_pop/atlas-100 \\
        --bids-dir /srv/nfs-data/sisko/christian/parrot_LEMON

    # eyes closed vs eyes open on the same subjects
    pixi run python examples/population_params_report_cli.py \\
        --results-root fit_test_pop_3/atlas-100 --compare-root fit_test_pop_3_eo/atlas-100 \\
        --bids-dir /srv/nfs-data/sisko/christian/parrot_LEMON

Reads saved results only (no jax/GPU). The permutation tests dominate the
runtime; lower ``--n-perm`` for a quick look, ``--no-brains`` skips pyvista.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score, permutation_test_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from params_report_cli import (DIPOLE_CAMERAS, DIPOLE_VIEWS, GROUP_ORDER,  # noqa: E402
                               HTML_TEMPLATE as _SUBJECT_HTML, LIVE_ON, PARAM_UNITS,
                               PER_NODE_PARAMS, classify)

from parrot_neuro import Subject  # noqa: E402

COHORTS = {
    "all": "all fits (gamma_weight = 0)",
    "bold_ok": "BOLD-fit subset (fc_corr > {thr:g})",
}
JR_PARAMS = [p for p in PER_NODE_PARAMS if LIVE_ON[p] == "surface"]
COND_SHORT = {"eyesclosed": "EC", "eyesopen": "EO"}
NUISANCE = ["sex_M", "log_eeg_psd_loss", "fc_corr"]
FIT_QUALITY_FEATURES = ["sex_M", "log_eeg_psd_loss", "eeg_loss_ratio", "fc_corr", "bold_loss_ratio"]

# reference palette (dataviz skill, references/palette.md) -- categorical slots
# 1/2 (validated all-pairs, so safe for scatters), the blue sequential ramp and
# the blue<->red diverging pair with a neutral gray midpoint
YOUNG_C, OLD_C = "#2a78d6", "#eb6834"
NEUTRAL_C = "#a9b3ac"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a7a6", "#e34948", "#9c2423"])


# ---------------------------------------------------------------------------
# loading

def short_name(name: str) -> str:
    return re.sub(r"^\d+Networks_", "", name)


def load_eeg_spectra(run_dir: Path, subj, cfg: dict) -> dict | None:
    """Empirical (fit target) and simulated scalp PSDs of one run, restricted
    to the fitted band [fmin, fmax] and normalized per channel to sum to 1.
    Both spectra lose absolute power (chunks are std-normalized, the loss
    compares band-normalized PSDs), so only *relative* band power is
    meaningful."""
    path = run_dir / "simulation_results.npz"
    if not path.exists():
        return None
    z = np.load(path)
    if "eeg_freqs" not in z.files:
        return None
    el = np.loadtxt(subj.path.electrodes_csv(), delimiter=",", usecols=0, dtype=str)
    freqs = z["eeg_freqs"]
    band = (freqs >= cfg.get("fmin", 1.0)) & (freqs <= cfg.get("fmax", 15.0))
    out = {"names": [str(el[i]).strip() for i in z["eeg_channel_indices"]], "freqs": freqs[band]}
    for k, key in (("emp", "eeg_psd_target"), ("sim", "eeg_psd_sim")):
        P = np.asarray(z[key], float)[:, band]
        out[k] = P / P.sum(axis=1, keepdims=True)
    return out


def scalp_positions_2d(subj) -> dict[str, tuple[float, float]]:
    """Azimuthal-equidistant projection of a subject's native-space montage
    (pole = Cz, up = nose), for drawing topographies."""
    names = np.loadtxt(subj.path.electrodes_csv(), delimiter=",", usecols=0, dtype=str)
    xyz = np.loadtxt(subj.path.electrodes_csv(), delimiter=",", usecols=(1, 2, 3))
    names = [str(n).strip() for n in names]
    c = xyz.mean(axis=0)
    pole = xyz[names.index("Cz")] - c
    pole /= np.linalg.norm(pole)
    fwd = xyz[names.index("Fpz") if "Fpz" in names else names.index("Fz")] - c
    fwd -= fwd @ pole * pole
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, pole)
    v = xyz - c
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    theta = np.arccos(np.clip(v @ pole, -1, 1))
    phi = np.arctan2(v @ fwd, v @ right)
    return {n: (float(t * np.cos(ph)), float(t * np.sin(ph))) for n, t, ph in zip(names, theta, phi)}


def stack_spectra(spectra: list[dict | None], ref_subj) -> dict | None:
    """Per-subject spectra onto a shared channel axis (NaN where a subject
    lacks a channel): emp/sim arrays of subjects x channels x freqs."""
    have = [s for s in spectra if s is not None]
    if not have:
        return None
    freqs = have[0]["freqs"]
    if any(len(s["freqs"]) != len(freqs) or not np.allclose(s["freqs"], freqs) for s in have):
        warnings.warn("EEG frequency grids differ between runs -- alpha analysis skipped")
        return None
    chans = list(dict.fromkeys(n for s in have for n in s["names"]))
    pos = {k: np.full((len(spectra), len(chans), len(freqs)), np.nan) for k in ("emp", "sim")}
    for i, s in enumerate(spectra):
        if s is None:
            continue
        ci = [chans.index(n) for n in s["names"]]
        for k in pos:
            pos[k][i, ci] = s[k]
    return {"channels": chans, "freqs": freqs, **pos, "pos2d": scalp_positions_2d(ref_subj)}


def load_population(results_root: Path, bids_dir: Path, participants_tsv: Path) -> dict:
    """Read every finished run under results_root, scatter each subject's
    per-node params onto the shared connectome axis, and attach age/sex."""
    run_dirs = sorted(d for d in results_root.iterdir() if d.is_dir() and (d / "config.json").exists())
    incomplete = [d.name for d in run_dirs if not (d / "optimized_params.npz").exists()]
    run_dirs = [d for d in run_dirs if (d / "optimized_params.npz").exists()]
    if not run_dirs:
        raise SystemExit(f"no finished runs (optimized_params.npz) under {results_root}")

    rows, per_subject, per_subject_init, spectra = [], [], [], []
    names, ref_cfg, ref_subj = None, None, None
    for d in run_dirs:
        cfg = json.loads((d / "config.json").read_text())
        sid = cfg["subject"]["subject"]
        npz = np.load(d / "optimized_params.npz")
        subj = Subject(str(bids_dir), sid)
        conn_names = subj.load.connectivity_labels(cfg["atlas"])
        if names is None:
            names, ref_cfg, ref_subj = conn_names, cfg, subj
        elif conn_names != names:
            raise ValueError(f"sub-{sid}: connectome labels differ from sub-{ref_cfg['subject']['subject']}")
        if cfg["learnable_params"] != ref_cfg["learnable_params"]:
            warnings.warn(f"{d.name}: learnable_params (init/bounds) differ from the first run")
        to_conn = np.asarray(subj.load.fmri_nodes(cfg["atlas"], cfg["fmri_task"]).to_conn)
        if len(to_conn) != npz[PER_NODE_PARAMS[0]].shape[0]:
            raise ValueError(f"{d.name}: {npz[PER_NODE_PARAMS[0]].shape[0]} optimized nodes but "
                             f"{len(to_conn)} fMRI-aligned nodes in {bids_dir} -- wrong --bids-dir?")
        vals = {}
        for p in PER_NODE_PARAMS:
            v = np.full(len(names), np.nan)
            v[to_conn] = npz[p]
            vals[p] = v
        per_subject.append(vals)
        init_vals = {p: np.full(len(names), np.nan) for p in PER_NODE_PARAMS}
        init_npz = d / "init_params.npz"
        if init_npz.exists():
            z0 = np.load(init_npz)
            for p in PER_NODE_PARAMS:
                if p in z0.files:
                    init_vals[p][to_conn] = np.broadcast_to(z0[p], (len(to_conn),))
        per_subject_init.append(init_vals)
        spectra.append(load_eeg_spectra(d, subj, cfg))

        diag_path = d / "diagnostics_metrics.json"
        diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
        loss_eeg = npz["loss_eeg"] if "loss_eeg" in npz.files else np.array([])
        rows.append({
            "participant_id": f"sub-{sid}", "subject": sid, "run": d.name,
            "gamma_weight": float(cfg.get("gamma_weight", 0.0)),
            "n_nodes": int(len(to_conn)),
            "G": float(npz["G"][0]),
            "eeg_psd_loss": float(np.mean(loss_eeg[-10:])) if len(loss_eeg) else np.nan,
            "eeg_loss_ratio": float(npz["eeg_loss_ratio"]) if "eeg_loss_ratio" in npz.files else np.nan,
            "bold_loss_ratio": float(npz["bold_loss_ratio"]) if "bold_loss_ratio" in npz.files else np.nan,
            "fc_corr": float(diag.get("fc_corr", np.nan)),
            "dfc_w_dist": float(diag.get("dfc_w_dist_after", np.nan)),
            "noise_seed": int(cfg.get("noise_seed", 69)),
            "init_seed": cfg.get("init_seed"),
            "init_spread": float(cfg.get("init_spread", 0.5)),
        })

    meta = pd.DataFrame(rows)
    dup = meta.subject[meta.subject.duplicated()].tolist()
    if dup:
        raise SystemExit(f"more than one run per subject under {results_root}: {dup}")
    part = pd.read_csv(participants_tsv, sep="\t")[["participant_id", "age", "sex"]]
    meta = meta.merge(part, on="participant_id", how="left")
    meta["log_eeg_psd_loss"] = np.log10(meta.eeg_psd_loss)
    meta["sex_M"] = (meta.sex == "M").astype(float)

    regions = [{"idx": i, "name": n, "short": short_name(n), **classify(n)}
               for i, n in enumerate(names)]
    hemi_rank = {"L": 0, "R": 1, "M": 2}
    order = sorted(range(len(regions)), key=lambda i: (
        GROUP_ORDER.index(regions[i]["group"]) if regions[i]["group"] in GROUP_ORDER else 99,
        hemi_rank[regions[i]["hemi"]], i))

    return {
        "meta": meta,
        "V": {p: np.stack([s[p] for s in per_subject]) for p in PER_NODE_PARAMS},
        "V_init": {p: np.stack([s[p] for s in per_subject_init]) for p in PER_NODE_PARAMS},
        "regions": regions,
        "display_order": order,
        "init": {q["name"]: q["init"] for q in ref_cfg["learnable_params"]},
        "bounds": {q["name"]: (q["low"], q["high"]) for q in ref_cfg["learnable_params"]},
        "atlas": ref_cfg["atlas"], "spacing": float(ref_cfg["spacing"]),
        "fmri_task": ref_cfg["fmri_task"], "config": ref_cfg,
        "incomplete": incomplete,
        "eeg": stack_spectra(spectra, ref_subj),
    }


def find_frozen(pop: dict) -> tuple[list[str], dict[str, list[int]]]:
    """Params that never left init on any live node (reported + dropped), and
    per-param live regions whose median movement is ~0 (no gradient reached
    them -- dropped from that param's analysis)."""
    frozen_params, frozen_cells = [], {}
    for p in PER_NODE_PARAMS:
        lo, hi = pop["bounds"][p]
        live = [r["idx"] for r in pop["regions"] if r["is_surface"] == (LIVE_ON[p] == "surface")]
        dev = np.abs(pop["V"][p][:, live] - pop["init"][p]) / (hi - lo)
        if np.nanmax(dev) < 1e-4:
            frozen_params.append(p)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(dev, axis=0)
        frozen_cells[p] = [live[j] for j in np.where(med < 1e-6)[0]]
    return frozen_params, frozen_cells


# ---------------------------------------------------------------------------
# statistics helpers

def _nan_spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok])[0])


def reliability(M: np.ndarray, rng: np.random.Generator, n_splits: int = 100) -> dict:
    """M: subjects x regions (NaN = missing). Split-half reliability of the
    group-median map, mean pairwise subject-map Spearman, and ICC(1) of
    'region' as the grouping factor."""
    n = M.shape[0]
    halves = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = np.nanmedian(M[perm[: n // 2]], axis=0)
            b = np.nanmedian(M[perm[n // 2:]], axis=0)
        halves.append(_nan_spearman(a, b))
    r_half = float(np.nanmean(halves))
    sb = 2 * r_half / (1 + r_half) if np.isfinite(r_half) and r_half > -1 else np.nan

    pair = pd.DataFrame(M.T).corr(method="spearman", min_periods=10).to_numpy()
    iu = np.triu_indices(n, 1)
    r_pair = float(np.nanmean(pair[iu]))

    groups = [M[:, j][np.isfinite(M[:, j])] for j in range(M.shape[1])]
    groups = [g for g in groups if len(g) >= 2]
    allv = np.concatenate(groups)
    k = len(groups)
    N = len(allv)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    msb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    icc = float((msb - msw) / (msb + (n0 - 1) * msw))
    return {"split_half_r": r_half, "split_half_sb": float(sb), "pairwise_subject_r": r_pair, "icc_region": icc}


def age_test(x: np.ndarray, old: np.ndarray, Z: np.ndarray) -> dict | None:
    """Young vs old on one variable: Mann-Whitney + Cliff's delta (>0 = old
    higher), and OLS z(x) ~ old + nuisance (Z, already z-scored)."""
    ok = np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    x, old, Z = x[ok], old[ok], Z[ok]
    n_o, n_y = int(old.sum()), int((~old).sum())
    if n_o < 5 or n_y < 5 or np.nanstd(x) == 0:
        return None
    U, p_mw = stats.mannwhitneyu(x[old], x[~old], alternative="two-sided")
    delta = 2 * U / (n_o * n_y) - 1
    y = (x - x.mean()) / x.std()
    X = sm.add_constant(np.column_stack([old.astype(float), Z]), has_constant="add")
    fit = sm.OLS(y, X).fit()
    return {
        "n_young": n_y, "n_old": n_o,
        "median_young": float(np.median(x[~old])), "median_old": float(np.median(x[old])),
        "cliffs_delta": float(delta), "p_mw": float(p_mw),
        "beta_old": float(fit.params[1]), "p_ols": float(fit.pvalues[1]),
    }


def _fdr(p: pd.Series) -> np.ndarray:
    q = np.full(len(p), np.nan)
    ok = p.notna().to_numpy()
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def _zcols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    Z = df[cols].to_numpy(float)
    sd = np.nanstd(Z, axis=0)
    sd[sd == 0] = 1
    return (Z - np.nanmean(Z, axis=0)) / sd


def _best_kmeans(X: np.ndarray, ks: range, seed: int) -> tuple[int, np.ndarray, dict]:
    sil, labels = {}, {}
    for k in ks:
        if k >= len(X):
            break
        lab = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
        sil[k] = float(silhouette_score(X, lab))
        labels[k] = lab
    best = max(sil, key=sil.get)
    return best, labels[best], sil


def age_classifier(X: np.ndarray, y: np.ndarray, n_perm: int, C: float, seed: int) -> dict:
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    aucs = cross_val_score(pipe, X, y, scoring="roc_auc", n_jobs=-1,
                           cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed))
    score, null, p = permutation_test_score(
        pipe, X, y, scoring="roc_auc", n_permutations=n_perm, n_jobs=-1, random_state=seed,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))
    return {"auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()),
            "auc_perm_cv": float(score), "p_perm": float(p), "null": null.tolist(),
            "pipeline": pipe}


# ---------------------------------------------------------------------------
# per-cohort analysis

def analyse_cohort(pop: dict, mask: np.ndarray, params: list[str], frozen_cells: dict,
                   args: argparse.Namespace) -> dict:
    meta = pop["meta"]
    regions = pop["regions"]
    rng = np.random.default_rng(args.seed)
    idx = np.where(mask)[0]
    sub = meta.iloc[idx].reset_index(drop=True)
    old = sub.old.to_numpy(bool)
    Z = _zcols(sub, NUISANCE)

    # regions analysed per param: live node type, enough coverage, not frozen
    cols = {}
    for p in params:
        Vp = pop["V"][p][idx]
        cov = np.isfinite(Vp).mean(axis=0)
        cols[p] = [r["idx"] for r in regions
                   if r["is_surface"] == (LIVE_ON[p] == "surface")
                   and cov[r["idx"]] >= args.min_coverage
                   and r["idx"] not in frozen_cells.get(p, [])]

    # -- 1. per-region stats + reliability --------------------------------
    stat_rows, rel = [], {}
    for p in params:
        lo, hi = pop["bounds"][p]
        M = pop["V"][p][idx][:, cols[p]]
        for j, r in enumerate(cols[p]):
            x = M[:, j][np.isfinite(M[:, j])]
            q1, med, q3 = np.percentile(x, [25, 50, 75])
            stat_rows.append({
                "param": p, "region": regions[r]["name"], "region_idx": r,
                "group": regions[r]["group"], "hemi": regions[r]["hemi"], "n": len(x),
                "median": med, "q1": q1, "q3": q3, "mean": x.mean(), "sd": x.std(),
                "cv": x.std() / abs(x.mean()) if x.mean() else np.nan,
                "median_shift_frac": (med - pop["init"][p]) / (hi - lo),
                "frac_at_bound": float(np.mean((x - lo < 0.01 * (hi - lo)) | (hi - x < 0.01 * (hi - lo)))),
            })
        rel[p] = reliability(M, rng)
    region_stats = pd.DataFrame(stat_rows)

    # -- 2. parameter trade-offs ------------------------------------------
    jr = [p for p in JR_PARAMS if p in params]
    jr_regions = sorted(set.intersection(*(set(cols[p]) for p in jr))) if jr else []
    cells = np.stack([pop["V"][p][idx][:, jr_regions] for p in jr], axis=-1)  # subj x reg x param
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        within = cells - np.nanmedian(cells, axis=0, keepdims=True)
    flat = within.reshape(-1, len(jr))
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    node_corr = stats.spearmanr(flat).statistic if len(jr) > 1 else np.ones((1, 1))

    subj_feats = {}
    for p in params:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            subj_feats[f"mean {p}"] = np.nanmean(pop["V"][p][idx][:, cols[p]], axis=1)
    subj_feats["G"] = sub.G.to_numpy()
    subj_feats["age"] = sub.age.to_numpy()
    subj_feats["log EEG-PSD loss"] = sub.log_eeg_psd_loss.to_numpy()
    subj_feats["fc_corr"] = sub.fc_corr.to_numpy()
    sf = pd.DataFrame(subj_feats)
    subj_corr = sf.corr(method="spearman")

    # -- 3. subject clustering ---------------------------------------------
    feat_names, blocks = [], []
    for p in params:
        blocks.append(pop["V"][p][idx][:, cols[p]])
        feat_names += [f"{p} | {regions[r]['short']}" for r in cols[p]]
    F_raw = np.concatenate(blocks, axis=1)
    F = SimpleImputer(strategy="median").fit_transform(F_raw)
    F = StandardScaler().fit_transform(F)
    pca = PCA(n_components=min(len(F) - 1, F.shape[1]), random_state=args.seed).fit(F)
    scores = pca.transform(F)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_pc = int(np.clip(np.searchsorted(cum, 0.8) + 1, 2, scores.shape[1]))
    n_pc_50 = 1 + int(np.searchsorted(cum, 0.5))
    best_k, clusters, sil = _best_kmeans(scores[:, :n_pc], range(2, 7), args.seed)
    ct = pd.crosstab(clusters, old)
    chi_p = float(stats.chi2_contingency(ct)[1]) if ct.shape == (best_k, 2) else np.nan

    pc_assoc = []
    for c in range(min(5, scores.shape[1])):
        s = scores[:, c]
        auc_age = stats.mannwhitneyu(s[old], s[~old])[0] / (old.sum() * (~old).sum())
        sexm = sub.sex_M.to_numpy(bool)
        auc_sex = stats.mannwhitneyu(s[sexm], s[~sexm])[0] / (sexm.sum() * (~sexm).sum())
        pc_assoc.append({
            "pc": c + 1, "var_explained": float(pca.explained_variance_ratio_[c]),
            "auc_old": float(auc_age), "p_old": float(stats.mannwhitneyu(s[old], s[~old])[1]),
            "auc_male": float(auc_sex),
            "rho_eeg_psd_loss": _nan_spearman(s, sub.log_eeg_psd_loss.to_numpy()),
            "rho_fc_corr": _nan_spearman(s, sub.fc_corr.to_numpy()),
            "rho_G": _nan_spearman(s, sub.G.to_numpy()),
        })
    loadings = {}
    for c in range(2):
        order = np.argsort(-np.abs(pca.components_[c]))[:10]
        loadings[f"PC{c + 1}"] = [(feat_names[i], float(pca.components_[c][i])) for i in order]

    # -- 4. region clustering (JR params, region medians) ------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        R = np.nanmedian(cells, axis=0)  # region x param
    R = StandardScaler().fit_transform(R)
    r_best_k, r_clusters, r_sil = _best_kmeans(R, range(2, 9), args.seed)
    r_groups = [regions[r]["group"] for r in jr_regions]
    r_ari = float(adjusted_rand_score(r_groups, r_clusters))
    r_xy = PCA(n_components=2, random_state=args.seed).fit_transform(R)
    region_clusters = pd.DataFrame({
        "region": [regions[r]["name"] for r in jr_regions], "region_idx": jr_regions,
        "group": r_groups, "hemi": [regions[r]["hemi"] for r in jr_regions],
        "cluster": r_clusters + 1, "pc1": r_xy[:, 0], "pc2": r_xy[:, 1],
    })

    # -- 5. age effects ----------------------------------------------------
    rows = []
    for p in params:
        for r in cols[p]:
            t = age_test(pop["V"][p][idx][:, r], old, Z)
            if t:
                rows.append({"level": "region", "param": p, "region": regions[r]["name"],
                             "region_idx": r, "group": regions[r]["group"], **t})
    age_region = pd.DataFrame(rows)
    age_region["q_mw"] = _fdr(age_region.p_mw)
    age_region["q_ols"] = _fdr(age_region.p_ols)

    rows = []
    for p in params:
        Vp = pop["V"][p][idx]
        for g in GROUP_ORDER:
            rr = [r for r in cols[p] if regions[r]["group"] == g]
            if not rr:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                x = np.nanmean(Vp[:, rr], axis=1)
            t = age_test(x, old, Z)
            if t:
                rows.append({"level": "network", "param": p, "group": g, "n_regions": len(rr), **t})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = age_test(np.nanmean(Vp[:, cols[p]], axis=1), old, Z)
        if t:
            rows.append({"level": "global", "param": p, "group": "whole brain", "n_regions": len(cols[p]), **t})
    t = age_test(sub.G.to_numpy(), old, Z)
    if t:
        rows.append({"level": "global", "param": "G", "group": "global coupling", "n_regions": 0, **t})
    age_group = pd.DataFrame(rows)
    age_group["q_mw"] = _fdr(age_group.p_mw)
    age_group["q_ols"] = _fdr(age_group.p_ols)

    # -- 6. multivariate age prediction ------------------------------------
    y = old.astype(int)
    X_params = np.column_stack([F_raw, sub.G.to_numpy()])
    X_nuis = sub[FIT_QUALITY_FEATURES].to_numpy(float)
    clf = {}
    for key, X in [("params", X_params), ("fit_quality", X_nuis),
                   ("params+fit_quality", np.column_stack([X_params, X_nuis]))]:
        print(f"    classifier [{key}] ({X.shape[1]} features, {args.n_perm} permutations) ...")
        clf[key] = age_classifier(X, y, args.n_perm, args.clf_c, args.seed)
    pipe = clf["params"].pop("pipeline").fit(X_params, y)
    coefs = pipe[-1].coef_[0]
    names_p = feat_names + ["G"]
    top = np.argsort(-np.abs(coefs))[:15]
    top_coefs = [(names_p[i], float(coefs[i])) for i in top]
    for v in clf.values():
        v.pop("pipeline", None)

    sub = sub.assign(**{f"PC{c + 1}": scores[:, c] for c in range(min(5, scores.shape[1]))},
                     cluster=clusters + 1)
    return {
        "condition": None,
        "idx": idx, "sub": sub, "cols": cols, "n": len(idx),
        "n_young": int((~old).sum()), "n_old": int(old.sum()),
        "region_stats": region_stats, "reliability": rel,
        "node_corr": {"labels": jr, "matrix": np.atleast_2d(node_corr).tolist(), "n_cells": len(flat)},
        "subj_corr": {"labels": list(subj_corr.columns), "matrix": subj_corr.to_numpy().tolist()},
        "pca": {"evr": pca.explained_variance_ratio_[:20].tolist(), "n_pc": n_pc, "n_pc_50": n_pc_50,
                "cum_var": float(cum[n_pc - 1]), "scores": scores[:, :5],
                "silhouette": sil, "best_k": best_k, "clusters": clusters + 1,
                "crosstab": ct, "chi2_p": chi_p, "ari_age": float(adjusted_rand_score(old, clusters)),
                "assoc": pc_assoc, "loadings": loadings, "n_features": F.shape[1]},
        "region_clusters": region_clusters,
        "region_cluster_info": {"best_k": r_best_k, "silhouette": r_sil, "ari_group": r_ari},
        "age_region": age_region, "age_group": age_group,
        "classifier": clf, "top_coefs": top_coefs,
    }


# ---------------------------------------------------------------------------
# condition comparison: the same subjects fit under a second condition
# (--compare-root, e.g. eyes-open vs eyes-closed EEG)
def icc_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-way ICC between two conditions over subjects: ICC(C,1) (consistency:
    is the subject ranking kept?) and ICC(A,1) (absolute agreement: also
    penalizes a systematic shift between conditions)."""
    ok = np.isfinite(x) & np.isfinite(y)
    X = np.column_stack([x[ok], y[ok]])
    n, k = X.shape
    if n < 5 or X.std() == 0:
        return np.nan, np.nan
    g = X.mean()
    ssr = k * ((X.mean(axis=1) - g) ** 2).sum()
    ssc = n * ((X.mean(axis=0) - g) ** 2).sum()
    sse = ((X - g) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        icc_c = (msr - mse) / (msr + (k - 1) * mse)
        icc_a = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    return float(icc_c), float(icc_a)


def paired_test(a: np.ndarray, b: np.ndarray, Zd: np.ndarray | None) -> dict | None:
    """Condition b vs a on one variable, same subjects: Wilcoxon signed-rank +
    matched-pairs rank-biserial r (> 0 = higher in b), and OLS z(b - a) ~ 1 +
    change in fit quality (Zd, z-scored); its intercept (dz_adj) is the shift
    left once the change in fit quality is accounted for."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = b - a
    if len(d) < 6 or np.all(d == 0):
        return None
    nz = d[d != 0]
    rk = stats.rankdata(np.abs(nz))
    out = {
        "n": len(d), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "median_diff": float(np.median(d)),
        "rank_biserial": float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()),
        "p_w": float(stats.wilcoxon(d).pvalue),
        "dz_adj": np.nan, "p_ols": np.nan,
    }
    if Zd is not None:
        # covariate gaps only drop a subject from the adjusted model, not from the Wilcoxon test
        okz = np.all(np.isfinite(Zd[ok]), axis=1)
        if okz.sum() >= Zd.shape[1] + 6 and d[okz].std() > 0:
            fit = sm.OLS(d[okz] / d[okz].std(), sm.add_constant(Zd[ok][okz], has_constant="add")).fit()
            out["dz_adj"], out["p_ols"] = float(fit.params[0]), float(fit.pvalues[0])
    return out


def identifiability(A: np.ndarray, B: np.ndarray) -> dict:
    """A, B: subjects x features, same subject order. Each feature is centred
    per condition (removing the group map, leaving individual deviations) and
    scaled by its pooled SD; then every A subject is Spearman-correlated with
    every B subject. A stable individual profile shows up as a bright diagonal:
    I_self (mean diagonal) > I_others (mean off-diagonal), and a top-1
    identification accuracy above chance (1/n)."""
    def prep(M):
        M = M.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(M, axis=0)
        M = np.where(np.isfinite(M), M, med)
        return M - M.mean(axis=0)
    A, B = prep(A), prep(B)
    sd = np.concatenate([A, B]).std(axis=0)
    sd[sd == 0] = 1
    RA = stats.rankdata(A / sd, axis=1)
    RB = stats.rankdata(B / sd, axis=1)
    RA = (RA - RA.mean(1, keepdims=True)) / RA.std(1, keepdims=True)
    RB = (RB - RB.mean(1, keepdims=True)) / RB.std(1, keepdims=True)
    S = RA @ RB.T / A.shape[1]
    n = len(S)
    off = S[~np.eye(n, dtype=bool)]
    acc = 0.5 * (np.mean(S.argmax(axis=1) == np.arange(n)) + np.mean(S.argmax(axis=0) == np.arange(n)))
    return {"S": S, "i_self": float(np.mean(np.diag(S))), "i_others": float(off.mean()),
            "i_diff": float(np.mean(np.diag(S)) - off.mean()), "accuracy": float(acc), "chance": 1 / n,
            "p_binom": float(stats.binomtest(int(round(acc * n)), n, 1 / n, alternative="greater").pvalue)}


def condition_classifier(XA: np.ndarray, XB: np.ndarray, n_perm: int, C: float, seed: int) -> dict:
    """Predict the condition from the within-subject-centred features (each
    subject's mean over the two conditions removed, so only the shift is
    left). CV and permutations are grouped by subject: labels are swapped
    within a subject, never across."""
    n = len(XA)
    XA = np.where(np.isfinite(XA), XA, np.nanmedian(XA, axis=0))
    XB = np.where(np.isfinite(XB), XB, np.nanmedian(XB, axis=0))
    m = 0.5 * (XA + XB)
    X = np.vstack([XA - m, XB - m])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    groups = np.r_[np.arange(n), np.arange(n)]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    aucs = np.concatenate([
        cross_val_score(pipe, X, y, groups=groups, scoring="roc_auc", n_jobs=-1,
                        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + r))
        for r in range(10)])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    score, null, p = permutation_test_score(pipe, X, y, groups=groups, scoring="roc_auc", cv=cv,
                                            n_permutations=n_perm, n_jobs=-1, random_state=seed)
    return {"auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()), "auc_perm_cv": float(score),
            "p_perm": float(p), "null": null.tolist()}


def analyse_condition(pop: dict, popb: dict, ia: np.ndarray, ib: np.ndarray, params: list[str],
                      frozen_cells: dict, labels: list[str], args: argparse.Namespace) -> dict:
    """Stability (ICC, identifiability) and shift (paired tests, classifier) of
    the fitted parameters between condition A (pop) and B (popb) on the
    subjects present in both, rows ia / ib."""
    regions = pop["regions"]
    ma = pop["meta"].iloc[ia].reset_index(drop=True)
    mb = popb["meta"].iloc[ib].reset_index(drop=True)
    old = ma.old.to_numpy(bool)
    VA = {p: pop["V"][p][ia] for p in params}
    VB = {p: popb["V"][p][ib] for p in params}

    cols = {}
    for p in params:
        cov = np.minimum(np.isfinite(VA[p]).mean(axis=0), np.isfinite(VB[p]).mean(axis=0))
        cols[p] = [r["idx"] for r in regions
                   if r["is_surface"] == (LIVE_ON[p] == "surface")
                   and cov[r["idx"]] >= args.min_coverage
                   and r["idx"] not in frozen_cells.get(p, [])]

    if all(np.array_equal(VA[p], VB[p], equal_nan=True) for p in params):
        raise SystemExit("--results-root and --compare-root hold identical fitted parameters for the paired "
                         "subjects -- are they the same runs?")
    dq = pd.DataFrame({k: mb[k].to_numpy() - ma[k].to_numpy() for k in ("log_eeg_psd_loss", "fc_corr")})
    usable = [k for k in dq.columns if np.isfinite(dq[k]).mean() >= 0.5 and np.nanstd(dq[k]) > 0]
    if len(usable) < len(dq.columns):
        print(f"    fit-quality adjustment drops {sorted(set(dq.columns) - set(usable))} (missing or constant)")
    Zd = _zcols(dq, usable) if usable else None

    def gmean(V, p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(V[p][:, cols[p]], axis=1)

    # subject-level variables, both conditions
    variables = {f"mean {p}": (gmean(VA, p), gmean(VB, p)) for p in params}
    variables["G"] = (ma.G.to_numpy(), mb.G.to_numpy())
    for k, lab in [("log_eeg_psd_loss", "log EEG-PSD loss"), ("eeg_loss_ratio", "EEG loss ratio"),
                   ("fc_corr", "fc_corr"), ("bold_loss_ratio", "BOLD loss ratio")]:
        variables[lab] = (ma[k].to_numpy(float), mb[k].to_numpy(float))
    fit_vars = ["log EEG-PSD loss", "EEG loss ratio", "fc_corr", "BOLD loss ratio"]
    var_stats = {}
    for v, (a, b) in variables.items():
        icc_c, icc_a = icc_pair(a, b)
        t = paired_test(a, b, None) or {}
        var_stats[v] = {"icc_c": icc_c, "icc_a": icc_a, "rho": _nan_spearman(a, b),
                        "rank_biserial": t.get("rank_biserial"), "p_w": t.get("p_w"),
                        "median_diff": t.get("median_diff")}

    # -- 1. stability -------------------------------------------------------
    stab_rows, region_rows, ident = [], [], {}
    for p in params:
        A, B = VA[p][:, cols[p]], VB[p][:, cols[p]]
        for j, r in enumerate(cols[p]):
            icc_c, icc_a = icc_pair(A[:, j], B[:, j])
            region_rows.append({"param": p, "region": regions[r]["name"], "region_idx": r,
                                "group": regions[r]["group"], "icc_c": icc_c, "icc_a": icc_a,
                                "rho": _nan_spearman(A[:, j], B[:, j])})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            map_r = _nan_spearman(np.nanmedian(A, axis=0), np.nanmedian(B, axis=0))
        ident[p] = identifiability(A, B)
        rr = [x for x in region_rows if x["param"] == p]
        stab_rows.append({
            "param": p, "n_regions": len(cols[p]),
            "region_icc_c": float(np.nanmedian([x["icc_c"] for x in rr])),
            "region_icc_a": float(np.nanmedian([x["icc_a"] for x in rr])),
            "frac_region_icc_c_gt_0.5": float(np.nanmean([x["icc_c"] > 0.5 for x in rr])),
            "global_icc_c": var_stats[f"mean {p}"]["icc_c"], "global_icc_a": var_stats[f"mean {p}"]["icc_a"],
            "group_map_r": map_r,
            **{k: ident[p][k] for k in ("i_self", "i_others", "i_diff", "accuracy", "p_binom")},
        })
    FA = np.concatenate([VA[p][:, cols[p]] for p in params], axis=1)
    FB = np.concatenate([VB[p][:, cols[p]] for p in params], axis=1)
    ident["all params"] = identifiability(FA, FB)
    stab_rows.append({"param": "all params", "n_regions": FA.shape[1],
                      **{k: ident["all params"][k] for k in ("i_self", "i_others", "i_diff", "accuracy", "p_binom")}})
    stab_rows.append({"param": "G", "global_icc_c": var_stats["G"]["icc_c"], "global_icc_a": var_stats["G"]["icc_a"]})
    stability = pd.DataFrame(stab_rows)
    region_icc = pd.DataFrame(region_rows)

    # -- 2. shift between conditions ---------------------------------------
    rows = []
    for p in params:
        for r in cols[p]:
            t = paired_test(VA[p][:, r], VB[p][:, r], Zd)
            if t:
                rows.append({"param": p, "region": regions[r]["name"], "region_idx": r,
                             "group": regions[r]["group"], **t})
    test_cols = ["n", "median_a", "median_b", "median_diff", "rank_biserial", "p_w", "dz_adj", "p_ols"]
    diff_region = pd.DataFrame(rows, columns=["param", "region", "region_idx", "group", *test_cols])
    diff_region["q_w"] = _fdr(diff_region.p_w)
    diff_region["q_ols"] = _fdr(diff_region.p_ols)

    def age_interaction(a, b):
        d = b - a
        ok = np.isfinite(d)
        if (old & ok).sum() < 5 or (~old & ok).sum() < 5:
            return {"age_delta": np.nan, "p_age": np.nan}
        U, p = stats.mannwhitneyu(d[old & ok], d[~old & ok])
        return {"age_delta": float(2 * U / ((old & ok).sum() * (~old & ok).sum()) - 1), "p_age": float(p)}

    rows = []
    for p in params:
        for g in GROUP_ORDER:
            rr = [r for r in cols[p] if regions[r]["group"] == g]
            if not rr:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                a, b = np.nanmean(VA[p][:, rr], axis=1), np.nanmean(VB[p][:, rr], axis=1)
            t = paired_test(a, b, Zd)
            if t:
                rows.append({"level": "network", "param": p, "group": g, "n_regions": len(rr), **t,
                             **age_interaction(a, b)})
        a, b = variables[f"mean {p}"]
        t = paired_test(a, b, Zd)
        if t:
            rows.append({"level": "global", "param": p, "group": "whole brain", "n_regions": len(cols[p]), **t,
                         **age_interaction(a, b)})
    a, b = variables["G"]
    t = paired_test(a, b, Zd)
    if t:
        rows.append({"level": "global", "param": "G", "group": "global coupling", "n_regions": 0, **t,
                     **age_interaction(a, b)})
    diff_group = pd.DataFrame(rows, columns=["level", "param", "group", "n_regions", *test_cols, "age_delta", "p_age"])
    diff_group["q_w"] = _fdr(diff_group.p_w)
    diff_group["q_ols"] = _fdr(diff_group.p_ols)

    fit_rows = []
    for v in fit_vars:
        t = paired_test(*variables[v], None)
        if t:
            fit_rows.append({"variable": v, **{k: t[k] for k in ("n", "median_a", "median_b", "median_diff",
                                                                    "rank_biserial", "p_w")}})
    fit_diff = pd.DataFrame(fit_rows, columns=["variable", "n", "median_a", "median_b", "median_diff",
                                               "rank_biserial", "p_w"])

    # -- 3. multivariate shift ---------------------------------------------
    clf = {}
    for key, XA, XB in [
            ("params", np.column_stack([FA, ma.G]), np.column_stack([FB, mb.G])),
            ("fit_quality", ma[["log_eeg_psd_loss", "eeg_loss_ratio", "fc_corr", "bold_loss_ratio"]].to_numpy(float),
             mb[["log_eeg_psd_loss", "eeg_loss_ratio", "fc_corr", "bold_loss_ratio"]].to_numpy(float))]:
        print(f"    condition classifier [{key}] ({XA.shape[1]} features, {args.n_perm} permutations) ...")
        clf[key] = condition_classifier(XA, XB, args.n_perm, args.clf_c, args.seed)

    alpha = analyse_alpha(pop, popb, ia, ib, VA, VB, cols, params, args)
    if alpha is not None:
        alpha["G_link"] = dict(zip(("rho", "p"), _spearman_p(mb.G.to_numpy() - ma.G.to_numpy(),
                                                             alpha["reactivity"]["emp"]["occ"])))
    return {
        "alpha": alpha,
        "labels": labels, "n": len(ia), "n_young": int((~old).sum()), "n_old": int(old.sum()),
        "subjects": ma[["participant_id", "age", "sex", "old"]].reset_index(drop=True),
        "ia": ia, "ib": ib, "cols": cols,
        "variables": variables, "var_stats": var_stats,
        "stability": stability, "region_icc": region_icc, "ident": ident,
        "diff_region": diff_region, "diff_group": diff_group, "fit_diff": fit_diff,
        "classifier": clf,
    }


def is_occipital(ch: str) -> bool:
    return ch.upper().startswith(("O", "PO", "I"))


def is_frontal(ch: str) -> bool:
    u = ch.upper()
    return u.startswith(("FP", "AF")) or (u.startswith("F") and not u.startswith(("FC", "FT")))


def _spearman_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return np.nan, np.nan
    r = stats.spearmanr(a[ok], b[ok])
    return float(r[0]), float(r[1])


def analyse_alpha(pop: dict, popb: dict, ia: np.ndarray, ib: np.ndarray, VA: dict, VB: dict,
                  cols: dict, params: list[str], args: argparse.Namespace) -> dict | None:
    """Hypothesis: the condition change acts on alpha in visual cortex.

    Electrodes (empirical target and simulated scalp EEG): log relative alpha
    (alpha-band share of the fitted band, per channel) per condition; paired
    tests per channel, in an occipital ROI, in a frontal reference ROI, and
    for occipital specificity (occipital - frontal). Alpha reactivity
    = log rel-alpha A - B per subject (> 0: alpha drops in B; with A = EC,
    B = EO that is the Berger effect). Does the model reproduce it
    (topography, per-subject reactivity)?

    Parameters: per subject, the B - A shift averaged over visual regions
    (Vis* networks) vs the other cortical regions (visual specificity), and
    whether the visual shift tracks the subject's empirical / simulated
    occipital alpha reactivity."""
    ea, eb = pop["eeg"], popb["eeg"]
    if ea is None or eb is None:
        return None
    chans = [c for c in ea["channels"] if c in eb["channels"]]
    if not np.allclose(ea["freqs"], eb["freqs"]):
        warnings.warn("EEG frequency grids differ between conditions -- alpha analysis skipped")
        return None
    freqs = ea["freqs"]
    lo, hi = args.alpha_band
    al = (freqs >= lo) & (freqs <= hi)
    roi = [c for c in chans if (c in args.alpha_roi if args.alpha_roi else is_occipital(c))]
    ref = [c for c in chans if is_frontal(c) and c not in roi]
    ca, cb = [ea["channels"].index(c) for c in chans], [eb["channels"].index(c) for c in chans]
    ri = [chans.index(c) for c in roi]
    fi = [chans.index(c) for c in ref]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spec = {("A", k): ea[k][ia][:, ca] for k in ("emp", "sim")}
        spec |= {("B", k): eb[k][ib][:, cb] for k in ("emp", "sim")}
        ra = {k: np.log(spec[("A", k)][:, :, al].sum(-1)) for k in ("emp", "sim")}
        rb = {k: np.log(spec[("B", k)][:, :, al].sum(-1)) for k in ("emp", "sim")}
        roi_mean = lambda M, j: np.nanmean(M[:, j], axis=1)
        n_ok = int(np.sum(np.isfinite(roi_mean(ra["emp"], ri)) & np.isfinite(roi_mean(rb["emp"], ri))))
    if n_ok < 10 or not ri or not fi:
        return None

    # -- electrodes -------------------------------------------------------
    rows = []
    for k in ("emp", "sim"):
        for j, c in enumerate(chans):
            t = paired_test(ra[k][:, j], rb[k][:, j], None)
            if t:
                rows.append({"source": k, "channel": c,
                             "roi": "occipital" if c in roi else ("frontal" if c in ref else "other"), **t})
    chan = pd.DataFrame(rows)
    chan["q_w"] = np.nan
    for k in ("emp", "sim"):
        m = chan.source == k
        chan.loc[m, "q_w"] = _fdr(chan.loc[m, "p_w"])

    react, roi_rows = {}, []
    for k in ("emp", "sim"):
        occ_a, occ_b = roi_mean(ra[k], ri), roi_mean(rb[k], ri)
        fr_a, fr_b = roi_mean(ra[k], fi), roi_mean(rb[k], fi)
        react[k] = {"occ": occ_a - occ_b, "front": fr_a - fr_b}
        for lab, a, b in [("occipital", occ_a, occ_b), ("frontal", fr_a, fr_b),
                          ("occipital - frontal", occ_a - fr_a, occ_b - fr_b)]:
            t = paired_test(a, b, None)
            if t:
                roi_rows.append({"source": k, "roi": lab,
                                 "frac_reactive": float(np.nanmean((a - b)[np.isfinite(a - b)] > 0)), **t})
    roi_tests = pd.DataFrame(roi_rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        topo = {k: np.nanmedian(ra[k] - rb[k], axis=0) for k in ("emp", "sim")}
    model_fit = {
        "topo_rho": _spearman_p(topo["emp"], topo["sim"])[0],
        "react_rho": _spearman_p(react["emp"]["occ"], react["sim"]["occ"]),
        "react_spec_rho": _spearman_p(react["emp"]["occ"] - react["emp"]["front"],
                                      react["sim"]["occ"] - react["sim"]["front"]),
        "level_rho_A": _spearman_p(roi_mean(ra["emp"], ri), roi_mean(ra["sim"], ri)),
        "level_rho_B": _spearman_p(roi_mean(rb["emp"], ri), roi_mean(rb["sim"], ri)),
    }

    # -- parameters: visual specificity + link to alpha reactivity ---------
    regions = pop["regions"]
    vis = {r["idx"] for r in regions if str(r["group"]).startswith("Vis")}
    prow, rrow, dvis_all = [], [], {}
    for p in params:
        vc = [r for r in cols[p] if r in vis]
        oc = [r for r in cols[p] if r not in vis and regions[r]["is_surface"]]
        if not vc or not oc:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            va_, vb_ = np.nanmean(VA[p][:, vc], axis=1), np.nanmean(VB[p][:, vc], axis=1)
            oa_, ob_ = np.nanmean(VA[p][:, oc], axis=1), np.nanmean(VB[p][:, oc], axis=1)
        tv = paired_test(va_, vb_, None) or {}
        to = paired_test(oa_, ob_, None) or {}
        ts = paired_test(va_ - oa_, vb_ - ob_, None) or {}
        dvis, drest = vb_ - va_, ob_ - oa_
        dvis_all[p] = dvis
        re_, pe_ = _spearman_p(dvis, react["emp"]["occ"])
        rs_, ps_ = _spearman_p(dvis, react["sim"]["occ"])
        rspec, pspec = _spearman_p(dvis - drest, react["emp"]["occ"] - react["emp"]["front"])
        prow.append({"param": p, "n_visual": len(vc), "n_other": len(oc),
                     "rb_visual": tv.get("rank_biserial"), "p_visual": tv.get("p_w"),
                     "rb_other": to.get("rank_biserial"), "p_other": to.get("p_w"),
                     "rb_specific": ts.get("rank_biserial"), "p_specific": ts.get("p_w"),
                     "median_shift_visual": tv.get("median_diff"), "median_shift_other": to.get("median_diff"),
                     "rho_react_emp": re_, "p_react_emp": pe_, "rho_react_sim": rs_, "p_react_sim": ps_,
                     "rho_specific_emp": rspec, "p_specific_emp": pspec})
        for r in cols[p]:
            if not regions[r]["is_surface"]:
                continue
            rho, pv = _spearman_p(VB[p][:, r] - VA[p][:, r], react["emp"]["occ"])
            rrow.append({"param": p, "region": regions[r]["name"], "region_idx": r, "group": regions[r]["group"],
                         "visual": r in vis, "rho_react_emp": rho, "p": pv})
    ptab = pd.DataFrame(prow)
    for k in ("p_visual", "p_specific", "p_react_emp", "p_react_sim", "p_specific_emp"):
        ptab[k.replace("p_", "q_", 1)] = _fdr(ptab[k])
    rtab = pd.DataFrame(rrow, columns=["param", "region", "region_idx", "group", "visual", "rho_react_emp", "p"])
    rtab["q"] = _fdr(rtab.p)
    enrich = []
    for p, g in rtab.groupby("param", sort=False):
        a, b = g.rho_react_emp[g.visual].dropna(), g.rho_react_emp[~g.visual].dropna()
        if len(a) >= 3 and len(b) >= 3:
            enrich.append({"param": p, "median_rho_visual": float(a.median()), "median_rho_other": float(b.median()),
                           "p_mw": float(stats.mannwhitneyu(a, b).pvalue)})

    def q(M, j):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X = np.nanmean(M[:, j], axis=1)
            return {"median": np.nanmedian(X, axis=0), "q1": np.nanpercentile(X, 25, axis=0),
                    "q3": np.nanpercentile(X, 75, axis=0)}

    return {
        "band": (lo, hi), "freqs": freqs, "channels": chans, "roi": roi, "ref": ref,
        "pos2d": ea["pos2d"], "n": n_ok,
        "spectra": {f"{c}|{k}|{r}": q(spec[(c, k)], j) for c in "AB" for k in ("emp", "sim")
                    for r, j in (("occ", ri), ("front", fi))},
        "chan_tests": chan, "roi_tests": roi_tests, "model_fit": model_fit,
        "reactivity": {k: {kk: vv for kk, vv in v.items()} for k, v in react.items()},
        "param_tests": ptab, "region_link": rtab, "enrichment": pd.DataFrame(enrich),
        "_dvis": dvis_all,
    }


# ---------------------------------------------------------------------------
# stability across initialization and noise (--replicate-roots): the same
# subjects refit with other --init-seed / --noise-seed values

def replicate_label(pop: dict) -> tuple[str, object, int]:
    """(label, init_seed, noise_seed) of one results root; all its runs must agree."""
    seeds = pop["meta"][["init_seed", "noise_seed"]].astype(object).drop_duplicates()
    if len(seeds) > 1:
        raise SystemExit(f"runs with different init/noise seeds mixed in one results root: {seeds.to_dict('records')}")
    init, noise = seeds.iloc[0]
    init = None if init is None or (isinstance(init, float) and np.isnan(init)) else int(init)
    return f"init-{'default' if init is None else init}_noise-{int(noise)}", init, int(noise)


def icc1(M: np.ndarray) -> float:
    """One-way random ICC(1): subjects x replicates. Share of variance that is
    between subjects rather than between refits of the same subject."""
    M = M[np.all(np.isfinite(M), axis=1)]
    n, k = M.shape
    if n < 3 or k < 2:
        return np.nan
    msb = k * M.mean(axis=1).var(ddof=1)
    msw = ((M - M.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    if msb + (k - 1) * msw == 0:
        return np.nan
    return float((msb - msw) / (msb + (k - 1) * msw))


def analyse_replicates(pop: dict, reps: list[dict], params: list[str], frozen_cells: dict,
                       results: dict, popb: dict | None, args: argparse.Namespace) -> dict | None:
    """Tests 1-7 of the initialization/noise stability design on the subjects
    fit in every replicate. ``reps`` are the replicate pops; ``pop`` (the
    --results-root) is replicate 0."""
    regions = pop["regions"]
    allp = [pop, *reps]
    labels, inits, noises = zip(*(replicate_label(x) for x in allp))
    if len(set(labels)) < len(labels):
        raise SystemExit(f"two results roots share the same init/noise seeds: {labels}")
    for x in reps:
        if [r["name"] for r in x["regions"]] != [r["name"] for r in regions]:
            raise SystemExit("a --replicate-roots folder uses a different connectome / atlas")
        if x["bounds"] != pop["bounds"]:
            warnings.warn("learnable_params bounds differ between replicates")
    ok_sets = [set(x["meta"].subject[(x["meta"].gamma_weight == 0) & x["meta"].age.notna()]) for x in allp]
    subjects = sorted(set.intersection(*ok_sets))
    if len(subjects) < 5:
        print(f"  replicates: only {len(subjects)} subjects fit in all {len(allp)} replicates -- skipped")
        return None
    rows = [x["meta"].reset_index(drop=True).set_index("subject").index.get_indexer(subjects) for x in allp]
    R, S = len(allp), len(subjects)
    X = {p: np.stack([x["V"][p][ri] for x, ri in zip(allp, rows)]) for p in params}        # R x S x M
    X0 = {p: np.stack([x["V_init"][p][ri] for x, ri in zip(allp, rows)]) for p in params}
    metas = [x["meta"].iloc[ri].reset_index(drop=True) for x, ri in zip(allp, rows)]
    G = np.stack([m.G.to_numpy() for m in metas])
    base_noise = noises[0]
    sub_init = [i for i in range(R) if noises[i] == base_noise]        # init varies, noise fixed
    sub_noise = [i for i in range(R) if inits[i] is None]              # noise varies, init default
    if len(sub_init) < 2:
        sub_init = []
    if len(sub_noise) < 2:
        sub_noise = []

    cols = {}
    for p in params:
        cov = np.isfinite(X[p]).all(axis=0).mean(axis=0)
        cols[p] = [r["idx"] for r in regions if r["is_surface"] == (LIVE_ON[p] == "surface")
                   and cov[r["idx"]] >= args.min_coverage and r["idx"] not in frozen_cells.get(p, [])]

    # -- 1. convergence ----------------------------------------------------
    L = np.stack([m.log_eeg_psd_loss.to_numpy() for m in metas])     # R x S
    FC = np.stack([m.fc_corr.to_numpy() for m in metas])
    worse = L > np.nanmin(L, axis=0, keepdims=True) + args.rep_loss_tol
    conv = pd.DataFrame({
        "replicate": labels, "init_seed": [("default" if i is None else i) for i in inits], "noise_seed": noises,
        "median_log_eeg_loss": np.nanmedian(L, axis=1), "median_fc_corr": np.nanmedian(FC, axis=1),
        "median_G": np.nanmedian(G, axis=1),
        "frac_worse_than_best": worse.mean(axis=1),
        "frac_best": np.bincount(np.nanargmin(L, axis=0), minlength=R) / S,
    })
    loss_spread = np.nanmax(L, axis=0) - np.nanmin(L, axis=0)

    # -- 2. variance decomposition -----------------------------------------
    def decomp(M3, lo, hi):          # M3: R x S
        out = {"icc_all": icc1(M3.T),
               "icc_init": icc1(M3[sub_init].T) if sub_init else np.nan,
               "icc_noise": icc1(M3[sub_noise].T) if sub_noise else np.nan}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["sd_within_frac"] = float(np.nanmean(np.nanstd(M3, axis=0, ddof=1))) / (hi - lo)
            out["sd_between_frac"] = float(np.nanstd(np.nanmean(M3, axis=0), ddof=1)) / (hi - lo)
        return out

    var_rows, reg_rows = [], []
    for p in params:
        lo, hi = pop["bounds"][p]
        for r in cols[p]:
            reg_rows.append({"param": p, "region": regions[r]["name"], "region_idx": r,
                             "group": regions[r]["group"], **decomp(X[p][:, :, r], lo, hi)})
        rr = pd.DataFrame([x for x in reg_rows if x["param"] == p])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gm = np.nanmean(X[p][:, :, cols[p]], axis=2)
        var_rows.append({"param": p, "n_regions": len(cols[p]),
                         **{f"region_{k}": float(rr[k].median()) for k in ("icc_all", "icc_init", "icc_noise",
                                                                           "sd_within_frac", "sd_between_frac")},
                         "frac_region_icc_gt_0.5": float((rr.icc_all > 0.5).mean()),
                         **{f"global_{k}": v for k, v in decomp(gm, lo, hi).items()}})
    gl, gh = pop["bounds"].get("G", (0.0, 1.0))
    var_rows.append({"param": "G", **{f"global_{k}": v for k, v in decomp(G, gl, gh).items()}})
    variance = pd.DataFrame(var_rows)
    region_var = pd.DataFrame(reg_rows)

    # -- 3. identifiability between refits ---------------------------------
    def kind(i, j):
        if noises[i] == noises[j]:
            return "init"
        if inits[i] == inits[j]:
            return "noise"
        return "init + noise"

    id_rows = []
    for i in range(R):
        for j in range(i + 1, R):
            for p in [*params, "all params"]:
                if p == "all params":
                    A = np.concatenate([X[q][i][:, cols[q]] for q in params], axis=1)
                    B = np.concatenate([X[q][j][:, cols[q]] for q in params], axis=1)
                else:
                    A, B = X[p][i][:, cols[p]], X[p][j][:, cols[p]]
                idn = identifiability(A, B)
                id_rows.append({"rep_a": labels[i], "rep_b": labels[j], "kind": kind(i, j), "param": p,
                                **{k: idn[k] for k in ("i_self", "i_others", "i_diff", "accuracy")}})
    ident = pd.DataFrame(id_rows)
    ident_summary = ident.groupby(["param", "kind"], sort=False)[["accuracy", "i_self", "i_others", "i_diff"]] \
        .mean().reset_index()
    ref = None
    cond = results.get("all", {}).get("condition")
    if cond is not None:
        csub = cond["subjects"].participant_id.str.replace("sub-", "", regex=False).tolist()
        keep = [k for k, sid in enumerate(csub) if sid in set(subjects)]
        if len(keep) >= 5:
            Sm = cond["ident"]["all params"]["S"][np.ix_(keep, keep)]
            n = len(keep)
            acc = 0.5 * (np.mean(Sm.argmax(1) == np.arange(n)) + np.mean(Sm.argmax(0) == np.arange(n)))
            ref = {"labels": cond["labels"], "n": n, "accuracy": float(acc),
                   "i_diff": float(np.mean(np.diag(Sm)) - Sm[~np.eye(n, dtype=bool)].mean())}

    # -- 4. does the fit remember its start? -------------------------------
    mem_rows = []
    rand = [i for i in range(R) if inits[i] is not None]
    for p in params:
        lo, hi = pop["bounds"][p]
        rhos, moved, still = [], [], []
        for i in rand:
            for s_ in range(S):
                f_, z_ = X[p][i, s_, cols[p]], X0[p][i, s_, cols[p]]
                ok = np.isfinite(f_) & np.isfinite(z_)
                if ok.sum() < 8 or np.std(z_[ok]) == 0:
                    continue
                rhos.append(stats.spearmanr(f_[ok], z_[ok])[0])
                d = np.abs(f_[ok] - z_[ok]) / (hi - lo)
                moved.append(np.median(d))
                still.append(np.mean(d < 0.02))
        if rhos:
            mem_rows.append({"param": p, "n_fits": len(rhos), "median_rho_final_init": float(np.median(rhos)),
                             "median_moved_frac": float(np.median(moved)),
                             "frac_nodes_unmoved": float(np.median(still))})
    memory = pd.DataFrame(mem_rows, columns=["param", "n_fits", "median_rho_final_init", "median_moved_frac",
                                             "frac_nodes_unmoved"])

    # -- 5. group-map stability --------------------------------------------
    gm_rows = []
    for p in params:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            maps = np.nanmedian(X[p][:, :, cols[p]], axis=1)          # R x regions
        rs = [_nan_spearman(maps[i], maps[j]) for i in range(R) for j in range(i + 1, R)]
        gm_rows.append({"param": p, "group_map_r_mean": float(np.nanmean(rs)), "group_map_r_min": float(np.nanmin(rs))})
    group_maps = pd.DataFrame(gm_rows)

    # -- 6. trade-offs within subject across refits ------------------------
    jr = [p for p in JR_PARAMS if p in params]
    jr_regions = sorted(set.intersection(*(set(cols[p]) for p in jr))) if jr else []
    cells = np.stack([X[p][:, :, jr_regions] for p in jr], axis=-1)       # R x S x reg x param
    within = (cells - np.nanmean(cells, axis=0, keepdims=True)).reshape(-1, len(jr))
    within = within[np.all(np.isfinite(within), axis=1) & np.any(within != 0, axis=1)]
    tradeoff = np.atleast_2d(stats.spearmanr(within).statistic) if len(jr) > 1 and len(within) > 10 else None

    # -- 7. do the conclusions survive? ------------------------------------
    age_rows = []
    for i, m in enumerate(metas):
        old = (m.age >= args.age_split).to_numpy()
        Z = _zcols(m, NUISANCE)
        for p in [*params, "G"]:
            if p == "G":
                x = m.G.to_numpy()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    x = np.nanmean(X[p][i][:, cols[p]], axis=1)
            t = age_test(x, old, Z)
            if t:
                age_rows.append({"replicate": labels[i], "param": p, "cliffs_delta": t["cliffs_delta"],
                                 "p_mw": t["p_mw"], "beta_old": t["beta_old"], "p_ols": t["p_ols"]})
    age_rob = pd.DataFrame(age_rows)
    if len(age_rob):
        agg = age_rob.groupby("param", sort=False).agg(
            median_delta=("cliffs_delta", "median"), min_delta=("cliffs_delta", "min"), max_delta=("cliffs_delta", "max"),
            n_sig=("p_mw", lambda v: int((v < 0.05).sum())))
        agg["same_sign"] = age_rob.groupby("param", sort=False).cliffs_delta.apply(
            lambda v: bool((np.sign(v) == np.sign(v.median())).all()))
        age_summary = agg.reset_index()
    else:
        age_summary = pd.DataFrame()

    snr = None
    if cond is not None and popb is not None:
        cs = cond["subjects"].participant_id.str.replace("sub-", "", regex=False).tolist()
        both = [(k, subjects.index(sid)) for k, sid in enumerate(cs) if sid in set(subjects)]
        if len(both) >= 5:
            kc, ks = map(np.array, zip(*both))
            srow = []
            dr = cond["diff_region"]
            for p in params:
                if p not in cond["cols"]:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sd = np.nanstd(X[p][:, ks, :], axis=0, ddof=1)        # S' x M, refit SD
                    d = popb["V"][p][cond["ib"][kc]] - pop["V"][p][cond["ia"][kc]]
                    z = np.nanmedian(d / np.where(sd > 0, sd, np.nan), axis=0)
                rs_ = [r for r in cond["cols"][p] if r in cols[p]]
                sig = set(dr[(dr.param == p) & (dr.q_w < 0.05)].region_idx)
                zz = np.abs(z[rs_])
                srow.append({"param": p, "n_subjects": len(both), "n_regions": len(rs_),
                             "median_abs_shift_over_refit_sd": float(np.nanmedian(zz)),
                             "frac_regions_gt_1": float(np.nanmean(zz > 1)),
                             "n_sig_regions": len(sig & set(rs_)),
                             "frac_sig_regions_gt_1": float(np.nanmean([abs(z[r]) > 1 for r in rs_ if r in sig]))
                             if sig & set(rs_) else np.nan})
            snr = {"labels": cond["labels"], "table": pd.DataFrame(srow)}

    return {
        "labels": list(labels), "inits": list(inits), "noises": list(noises), "n": S, "subjects": subjects,
        "sub_init": [labels[i] for i in sub_init], "sub_noise": [labels[i] for i in sub_noise],
        "cols": cols, "convergence": conv, "loss_spread": loss_spread, "L": L,
        "variance": variance, "region_var": region_var,
        "ident": ident, "ident_summary": ident_summary, "ident_ref": ref,
        "memory": memory, "group_maps": group_maps,
        "tradeoff": None if tradeoff is None else {"labels": jr, "matrix": tradeoff},
        "age": age_rob, "age_summary": age_summary, "snr": snr,
    }


# ---------------------------------------------------------------------------
# static figures

def _savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_fit_quality(meta: pd.DataFrame, base: np.ndarray, thr: float, out: Path) -> None:
    m = meta[base]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    for lab, c, sel in [("young", YOUNG_C, ~m.old), ("old", OLD_C, m.old)]:
        axes[0].scatter(m.log_eeg_psd_loss[sel], m.fc_corr[sel], s=22, color=c, alpha=0.8,
                        edgecolor="white", linewidth=0.6, label=f"{lab} (n={sel.sum()})")
    axes[0].axhline(thr, color="#44534c", ls="--", lw=1)
    axes[0].text(axes[0].get_xlim()[1], thr, f" fc_corr = {thr:g}", va="bottom", ha="right", fontsize=8, color="#44534c")
    axes[0].set_xlabel("log10 EEG-PSD loss (last 10 epochs)")
    axes[0].set_ylabel("fc_corr (sim vs emp static FC)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(m.fc_corr, bins=30, color=NEUTRAL_C, edgecolor="white")
    axes[1].axvline(thr, color="#44534c", ls="--", lw=1)
    axes[1].set_xlabel("fc_corr"); axes[1].set_ylabel("# subjects")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fit quality across subjects (gamma_weight = 0)")
    _savefig(fig, out)


def fig_region_spread(pop: dict, res: dict, params: list[str], out: Path) -> None:
    order = [r for r in pop["display_order"]]
    sub_old = res["sub"].old.to_numpy(bool)
    fig, axes = plt.subplots(len(params), 1, figsize=(20, 3.1 * len(params)))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, params):
        regs = [r for r in order if r in res["cols"][p]]
        M = pop["V"][p][res["idx"]][:, regs]
        xs = np.arange(len(regs))
        jit = (np.random.default_rng(0).random(M.shape[0]) - 0.5) * 0.6
        for lab, c, sel in [("young", YOUNG_C, ~sub_old), ("old", OLD_C, sub_old)]:
            xx = (xs[None, :] + jit[sel, None]).ravel()
            ax.scatter(xx, M[sel].ravel(), s=2.5, color=c, alpha=0.35, linewidths=0, label=lab)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(M[sel], axis=0)
            ax.plot(xs, med, color=c, lw=1.2)
        ax.axhline(pop["init"][p], color="#44534c", ls=":", lw=1)
        groups = [pop["regions"][r]["group"] for r in regs]
        starts = [i for i in range(len(groups)) if i == 0 or groups[i] != groups[i - 1]]
        for s in starts[1:]:
            ax.axvline(s - 0.5, color="#d9e0da", lw=1)
        ax.set_xticks([s + (([*starts[1:], len(groups)][k] - s) / 2) - 0.5 for k, s in enumerate(starts)])
        ax.set_xticklabels([groups[s] for s in starts], fontsize=8)
        ax.set_xlim(-1, len(regs))
        ax.set_ylabel(f"{p} [{PARAM_UNITS[p]}]", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", markerscale=5, frameon=False, fontsize=8)
    fig.suptitle(f"Per-region parameter values across subjects (n={res['n']}); line = group median, dotted = init")
    fig.tight_layout()
    _savefig(fig, out)


def fig_reliability(res: dict, params: list[str], out: Path) -> None:
    keys = [("split_half_r", "split-half r (group map)"), ("pairwise_subject_r", "mean subject-subject r"),
            ("icc_region", "ICC(region)")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.26
    for k, (key, lab) in enumerate(keys):
        vals = [res["reliability"][p][key] for p in params]
        ax.bar(np.arange(len(params)) + (k - 1) * w, vals, width=w - 0.02, label=lab,
               color=["#2a78d6", "#86b6ef", "#104281"][k])
    ax.set_xticks(range(len(params))); ax.set_xticklabels(params)
    ax.axhline(0, color="#44534c", lw=0.8)
    ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, out)


def fig_pca(res: dict, out: Path) -> None:
    pca = res["pca"]
    old = res["sub"].old.to_numpy(bool)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    evr = np.array(pca["evr"])
    axes[0].bar(np.arange(1, len(evr) + 1), evr * 100, color="#86b6ef")
    axes[0].plot(np.arange(1, len(evr) + 1), np.cumsum(evr) * 100, color="#104281", lw=1.5)
    if pca["n_pc"] < len(evr):
        axes[0].axvline(pca["n_pc"] + 0.5, color="#44534c", ls="--", lw=1)
    axes[0].set_xlabel("PC"); axes[0].set_ylabel("% variance (bar) / cumulative (line)")
    s = pca["scores"]
    for lab, c, sel in [("young", YOUNG_C, ~old), ("old", OLD_C, old)]:
        axes[1].scatter(s[sel, 0], s[sel, 1], s=20, color=c, alpha=0.8, edgecolor="white", linewidth=0.5, label=lab)
    axes[1].set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)"); axes[1].set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    axes[1].legend(frameon=False, fontsize=8)
    ks = sorted(pca["silhouette"])
    axes[2].bar(ks, [pca["silhouette"][k] for k in ks], color="#86b6ef")
    axes[2].bar([pca["best_k"]], [pca["silhouette"][pca["best_k"]]], color="#104281")
    axes[2].axhline(0.25, color="#44534c", ls=":", lw=1)
    axes[2].set_xlabel("k (k-means on retained PCs)"); axes[2].set_ylabel("silhouette")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Subject parameter space: {pca['n_features']} features, {pca['n_pc']} PCs retained "
                 f"({pca['cum_var']:.0%} var; first 20 shown)")
    fig.tight_layout()
    _savefig(fig, out)


def fig_param_corr(res: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.6]})
    for ax, key, title in [(axes[0], "node_corr", "Node level, within-region (trade-offs)"),
                           (axes[1], "subj_corr", "Subject level (region-averaged)")]:
        M = np.array(res[key]["matrix"]); lab = res[key]["labels"]
        im = ax.imshow(M, cmap=DIV_CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8)
        for i in range(len(lab)):
            for j in range(len(lab)):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(M[i, j]) > 0.55 else "#182420")
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Spearman rho")
    _savefig(fig, out)


def fig_volcano(res: dict, out: Path) -> None:
    a = res["age_region"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sig = a.q_mw < 0.05
    ax.scatter(a.cliffs_delta[~sig], -np.log10(a.p_mw[~sig]), s=10, color=NEUTRAL_C, alpha=0.6, linewidths=0,
               label="q >= 0.05")
    ax.scatter(a.cliffs_delta[sig], -np.log10(a.p_mw[sig]), s=16, color="#104281", alpha=0.9, linewidths=0,
               label="q < 0.05 (BH)")
    for _, r in a.nsmallest(8, "p_mw").iterrows():
        ax.annotate(f"{r.param} {short_name(r.region)}", (r.cliffs_delta, -np.log10(r.p_mw)),
                    fontsize=6.5, xytext=(3, 2), textcoords="offset points", color="#44534c")
    ax.axvline(0, color="#44534c", lw=0.8)
    ax.set_xlabel("Cliff's delta (old vs young; > 0 = higher in old)")
    ax.set_ylabel("-log10 p (Mann-Whitney)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Per-region age effects ({len(a)} region x param tests)", fontsize=10)
    _savefig(fig, out)


def fig_classifier(res: dict, out: Path) -> None:
    clf = res["classifier"]
    fig, axes = plt.subplots(1, len(clf), figsize=(5 * len(clf), 3.4), sharey=True)
    for ax, (k, v) in zip(axes, clf.items()):
        ax.hist(v["null"], bins=30, color=NEUTRAL_C, edgecolor="white")
        ax.axvline(v["auc_perm_cv"], color="#104281", lw=2)
        ax.set_title(f"{k}\nAUC {v['auc_mean']:.3f} +/- {v['auc_sd']:.3f}, perm p = {v['p_perm']:.3g}", fontsize=9)
        ax.set_xlabel("cross-validated AUC (null = label permutations)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("# permutations")
    fig.tight_layout()
    _savefig(fig, out)


def fig_condition_stability(cond: dict, params: list[str], out: Path) -> None:
    A, B = cond["labels"]
    st = cond["stability"].set_index("param")
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    keys = [("region_icc_c", "median region ICC(C,1)"), ("global_icc_c", "whole-brain mean ICC(C,1)"),
            ("global_icc_a", "whole-brain mean ICC(A,1)"), ("group_map_r", "group-map r")]
    labs = params + ["G"]
    w = 0.2
    for k, (key, lab) in enumerate(keys):
        vals = [st.loc[p, key] if p in st.index and key in st else np.nan for p in labs]
        axes[0].bar(np.arange(len(labs)) + (k - 1.5) * w, vals, width=w - 0.02, label=lab,
                    color=["#2a78d6", "#86b6ef", "#104281", "#a9b3ac"][k])
    axes[0].set_xticks(range(len(labs))); axes[0].set_xticklabels(labs)
    axes[0].axhline(0, color="#44534c", lw=0.8)
    axes[0].set_ylim(min(-0.05, axes[0].get_ylim()[0]), 1)
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    axes[0].set_title(f"{A} vs {B}: does each subject keep its value?", fontsize=10, pad=24)
    il = params + ["all params"]
    axes[1].bar(range(len(il)), [st.loc[p, "accuracy"] for p in il], color="#2a78d6")
    axes[1].axhline(cond["ident"]["all params"]["chance"], color="#44534c", ls="--", lw=1)
    axes[1].text(len(il) - 0.5, cond["ident"]["all params"]["chance"], " chance", va="bottom", ha="right",
                 fontsize=8, color="#44534c")
    axes[1].set_xticks(range(len(il))); axes[1].set_xticklabels(il, rotation=30, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Subject identification {A} -> {B} (top-1 accuracy)", fontsize=10, pad=24)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out)


def fig_condition_scatter(cond: dict, params: list[str], out: Path) -> None:
    A, B = cond["labels"]
    old = cond["subjects"].old.to_numpy(bool)
    vs = [f"mean {p}" for p in params] + ["G", "log EEG-PSD loss", "fc_corr"]
    nc = 4
    nr = int(np.ceil(len(vs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.4 * nr))
    axes = np.atleast_1d(axes).ravel()
    for ax, v in zip(axes, vs):
        a, b = cond["variables"][v]
        for lab, c, sel in [("young", YOUNG_C, ~old), ("old", OLD_C, old)]:
            ax.scatter(a[sel], b[sel], s=14, color=c, alpha=0.8, edgecolor="white", linewidth=0.4, label=lab)
        ok = np.isfinite(a) & np.isfinite(b)
        lo, hi = np.min([a[ok].min(), b[ok].min()]), np.max([a[ok].max(), b[ok].max()])
        ax.plot([lo, hi], [lo, hi], color="#44534c", ls=":", lw=1)
        s = cond["var_stats"][v]
        ax.set_title(f"{v}\nICC(C) {s['icc_c']:.2f}, ICC(A) {s['icc_a']:.2f}, r_rb {np.nan if s['rank_biserial'] is None else s['rank_biserial']:+.2f}",
                     fontsize=8.5)
        ax.set_xlabel(A, fontsize=8); ax.set_ylabel(B, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(vs):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Subject-level values, {A} (x) vs {B} (y); dotted = identity (n={cond['n']})")
    fig.tight_layout()
    _savefig(fig, out)


def fig_condition_ident(cond: dict, out: Path) -> None:
    A, B = cond["labels"]
    idn = cond["ident"]["all params"]
    S = idn["S"]
    order = np.argsort(cond["subjects"].age.to_numpy(), kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    lim = np.abs(S).max()
    im = axes[0].imshow(S[np.ix_(order, order)], cmap=DIV_CMAP, vmin=-lim, vmax=lim)
    axes[0].set_xlabel(f"subject in {B} (sorted by age)"); axes[0].set_ylabel(f"subject in {A}")
    axes[0].set_title("Individual deviation maps, all params (Spearman)", fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    off = S[~np.eye(len(S), dtype=bool)]
    bins = np.linspace(min(off.min(), np.diag(S).min()), max(off.max(), np.diag(S).max()), 40)
    axes[1].hist(off, bins=bins, color=NEUTRAL_C, density=True, label="different subjects")
    axes[1].hist(np.diag(S), bins=bins, color="#104281", alpha=0.75, density=True, label="same subject")
    axes[1].set_xlabel(f"similarity {A} vs {B}"); axes[1].set_ylabel("density")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title(f"I_self {idn['i_self']:.3f} vs I_others {idn['i_others']:.3f}; "
                      f"accuracy {idn['accuracy']:.0%} (chance {idn['chance']:.1%})", fontsize=10)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out)


def fig_condition_diff(pop: dict, cond: dict, params: list[str], out: Path) -> None:
    A, B = cond["labels"]
    d = cond["diff_region"]
    fig, axes = plt.subplots(len(params), 1, figsize=(20, 2.3 * len(params)), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, params):
        regs = [r for r in pop["display_order"] if r in cond["cols"][p]]
        dp = d[d.param == p].set_index("region_idx")
        vals = np.array([dp.rank_biserial.get(r, np.nan) for r in regs])
        sig = np.array([dp.q_w.get(r, 1.0) < 0.05 for r in regs])
        ax.bar(np.arange(len(regs)), vals, width=0.85,
               color=np.where(sig, np.where(vals > 0, "#b8322f", "#184f95"), NEUTRAL_C))
        groups = [pop["regions"][r]["group"] for r in regs]
        starts = [i for i in range(len(groups)) if i == 0 or groups[i] != groups[i - 1]]
        for s in starts[1:]:
            ax.axvline(s - 0.5, color="#d9e0da", lw=1)
        ax.set_xticks([s + (([*starts[1:], len(groups)][k] - s) / 2) - 0.5 for k, s in enumerate(starts)])
        ax.set_xticklabels([groups[s] for s in starts], fontsize=8)
        ax.set_xlim(-1, len(regs))
        ax.axhline(0, color="#44534c", lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(f"{p}\nr_rb", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Per-region shift {B} - {A} (matched-pairs rank-biserial; > 0 = higher in {B}); "
                 "colored = BH q < 0.05 (Wilcoxon)")
    fig.tight_layout()
    _savefig(fig, out)


def fig_alpha_spectra(cond: dict, out: Path) -> None:
    A, B = cond["labels"]
    al = cond["alpha"]
    fr = al["freqs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, k, title in [(axes[0], "emp", "Empirical EEG (fit target)"), (axes[1], "sim", "Simulated EEG (fitted model)")]:
        for c, lab, col in [("A", A, "#104281"), ("B", B, "#eb6834")]:
            for r, ls, lw, alpha in [("occ", "-", 2, 0.18), ("front", "--", 1.2, 0)]:
                sp = al["spectra"][f"{c}|{k}|{r}"]
                ax.plot(fr, sp["median"], color=col, ls=ls, lw=lw,
                        label=f"{lab}, {'occipital' if r == 'occ' else 'frontal'}")
                if alpha:
                    ax.fill_between(fr, sp["q1"], sp["q3"], color=col, alpha=alpha, linewidth=0)
        ax.axvspan(*al["band"], color="#d9e0da", alpha=0.5, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("Hz"); ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("relative power (per-channel normalized)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"ROI spectra, median across {al['n']} paired subjects (band = IQR, occipital); "
                 f"shaded = alpha {al['band'][0]:g}-{al['band'][1]:g} Hz")
    fig.tight_layout()
    _savefig(fig, out)


def fig_alpha_topo(cond: dict, out: Path) -> None:
    A, B = cond["labels"]
    al = cond["alpha"]
    ct = al["chan_tests"]
    pos = al["pos2d"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, k, title in [(axes[0], "emp", "Empirical"), (axes[1], "sim", "Simulated")]:
        d = ct[ct.source == k]
        ax.add_patch(plt.Circle((0, 0), np.pi / 2 * 1.05, fill=False, color="#44534c", lw=1))
        ax.plot([-0.12, 0, 0.12], [np.pi / 2 * 1.05, np.pi / 2 * 1.05 + 0.15, np.pi / 2 * 1.05], color="#44534c", lw=1)
        for _, r in d.iterrows():
            if r.channel not in pos:
                continue
            x, y = pos[r.channel]
            sig = r.q_w < 0.05
            ax.scatter(x, y, s=260 if sig else 150, c=[r.rank_biserial], cmap=DIV_CMAP, vmin=-1, vmax=1,
                       edgecolors="#182420" if sig else "white", linewidths=1.6 if sig else 0.8,
                       marker="s" if r.roi == "occipital" else "o")
            ax.text(x, y - 0.13, r.channel, ha="center", va="top", fontsize=5.5, color="#44534c")
        ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.95)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10)
    sm_ = plt.cm.ScalarMappable(cmap=DIV_CMAP, norm=plt.Normalize(-1, 1))
    fig.colorbar(sm_, ax=axes, shrink=0.7, label=f"rank-biserial, rel. alpha {B} - {A}")
    fig.suptitle(f"Per-channel alpha change ({B} - {A}; < 0 = less alpha in {B}). "
                 "Squares = occipital ROI, dark edge = BH q < 0.05")
    _savefig(fig, out)


def fig_alpha_params(cond: dict, out: Path) -> None:
    A, B = cond["labels"]
    al = cond["alpha"]
    pt = al["param_tests"]
    old = cond["subjects"].old.to_numpy(bool)
    re_, rs_ = al["reactivity"]["emp"]["occ"], al["reactivity"]["sim"]["occ"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    for lab, c, sel in [("young", YOUNG_C, ~old), ("old", OLD_C, old)]:
        axes[0].scatter(re_[sel], rs_[sel], s=16, color=c, alpha=0.8, edgecolor="white", linewidth=0.4, label=lab)
    for ax in axes[[0, 2]]:
        ax.axvline(0, color="#d9e0da", lw=1)
    axes[0].axhline(0, color="#d9e0da", lw=1)
    r, pv = al["model_fit"]["react_rho"]
    axes[0].set_xlabel(f"empirical occipital alpha reactivity ({A} - {B}, log)")
    axes[0].set_ylabel("simulated occipital alpha reactivity")
    axes[0].set_title(f"Does the model reproduce each subject's reactivity?\nrho {r:.2f}, p {pv:.2g}", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8)
    x = np.arange(len(pt))
    for k, (col, lab, off) in enumerate([("rb_visual", "visual regions", -0.27), ("rb_other", "other cortex", 0),
                                         ("rb_specific", "visual - other", 0.27)]):
        sig = pt[col.replace("rb_", "p_")] < 0.05
        axes[1].bar(x + off, pt[col], width=0.25, color=["#2a78d6", "#a9b3ac", "#104281"][k], label=lab,
                    edgecolor=np.where(sig, "#182420", "none"), linewidth=1.2)
    axes[1].axhline(0, color="#44534c", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(pt.param)
    axes[1].set_ylim(-1, 1)
    axes[1].set_ylabel(f"rank-biserial, shift {B} - {A}")
    axes[1].set_title("Parameter shift in visual vs other cortical regions (dark edge: p < 0.05)", fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    if len(pt):
        best = pt.loc[pt.p_react_emp.idxmin()]
        p = best.param
        dvis = cond["alpha"]["_dvis"][p]
        for lab, c, sel in [("young", YOUNG_C, ~old), ("old", OLD_C, old)]:
            axes[2].scatter(re_[sel], dvis[sel], s=16, color=c, alpha=0.8, edgecolor="white", linewidth=0.4)
        axes[2].axhline(0, color="#d9e0da", lw=1)
        axes[2].set_xlabel(f"empirical occipital alpha reactivity ({A} - {B}, log)")
        axes[2].set_ylabel(f"visual-region shift of {p} ({B} - {A})")
        axes[2].set_title(f"Strongest link: {p}, rho {best.rho_react_emp:.2f}, q {best.q_react_emp:.2g}", fontsize=9.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out)


def _short_rep(label: str) -> str:
    return label.replace("init-", "i:").replace("_noise-", " n:")


def fig_rep_convergence(rep: dict, out: Path) -> None:
    L, labs = rep["L"], [_short_rep(x) for x in rep["labels"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for s_ in range(L.shape[1]):
        axes[0].plot(range(len(labs)), L[:, s_], color=NEUTRAL_C, lw=0.7, alpha=0.6)
    axes[0].scatter(np.repeat(np.arange(len(labs)), L.shape[1]), L.ravel(), s=9, color="#2a78d6", alpha=0.7, zorder=3)
    axes[0].set_xticks(range(len(labs))); axes[0].set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("log10 EEG-PSD loss (last 10 epochs)")
    axes[0].set_title("Final loss per refit (lines join one subject)", fontsize=10)
    axes[1].hist(rep["loss_spread"], bins=20, color=NEUTRAL_C, edgecolor="white")
    axes[1].set_xlabel("max - min log10 loss across refits, per subject"); axes[1].set_ylabel("# subjects")
    axes[1].set_title("Loss spread within subject", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out)


def fig_rep_icc(rep: dict, out: Path) -> None:
    v = rep["variance"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    for ax, pre, title in [(axes[0], "region_", "Per region (median over regions)"),
                           (axes[1], "global_", "Whole-brain mean (+ G)")]:
        d = v[v[f"{pre}icc_all"].notna()]
        x = np.arange(len(d))
        for k, (key, lab, col) in enumerate([("icc_all", "all refits", "#104281"),
                                              ("icc_init", "init varies (noise fixed)", "#2a78d6"),
                                              ("icc_noise", "noise varies (init fixed)", "#86b6ef")]):
            ax.bar(x + (k - 1) * 0.26, d[f"{pre}{key}"], width=0.25, color=col, label=lab)
        ax.axhline(0.5, color="#44534c", ls=":", lw=1)
        ax.axhline(0, color="#44534c", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(d.param)
        ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ICC(1): between-subject share")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.2))
    _savefig(fig, out)


def fig_rep_ident(rep: dict, out: Path) -> None:
    d = rep["ident_summary"]
    kinds = [k for k in ("init", "noise", "init + noise") if k in set(d.kind)]
    ps = list(dict.fromkeys(d.param))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    w = 0.8 / max(1, len(kinds))
    for k, kd in enumerate(kinds):
        dd = d[d.kind == kd].set_index("param").reindex(ps)
        ax.bar(np.arange(len(ps)) + (k - (len(kinds) - 1) / 2) * w, dd.accuracy, width=w * 0.95,
               color=["#2a78d6", "#86b6ef", "#104281"][k], label=f"refits differing in {kd}")
    ax.axhline(1 / rep["n"], color="#44534c", ls="--", lw=1)
    ax.text(len(ps) - 0.5, 1 / rep["n"], " chance", va="bottom", ha="right", fontsize=8, color="#44534c")
    if rep["ident_ref"]:
        r = rep["ident_ref"]
        ax.axhline(r["accuracy"], color="#eb6834", lw=1.5)
        ax.text(-0.4, r["accuracy"], f" {r['labels'][0]} vs {r['labels'][1]}, all params (same subjects)",
                va="bottom", fontsize=8, color="#eb6834")
    ax.set_xticks(range(len(ps))); ax.set_xticklabels(ps)
    ax.set_ylim(0, 1.05); ax.set_ylabel("top-1 identification accuracy")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, out)


def fig_rep_memory_tradeoff(rep: dict, out: Path) -> None:
    m = rep["memory"]
    t = rep["tradeoff"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    if len(m):
        x = np.arange(len(m))
        axes[0].bar(x - 0.2, m.median_rho_final_init, width=0.38, color="#104281", label="rho(final, start) across regions")
        axes[0].bar(x + 0.2, m.frac_nodes_unmoved, width=0.38, color="#a9b3ac", label="nodes moved < 2% of range")
        axes[0].set_xticks(x); axes[0].set_xticklabels(m.param)
        axes[0].axhline(0, color="#44534c", lw=0.8)
        axes[0].set_ylim(min(-0.1, axes[0].get_ylim()[0]), 1)
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_title("Does a fit remember its random start? (high = yes)", fontsize=10)
    else:
        axes[0].text(0.5, 0.5, "no --init-seed refits", ha="center", va="center"); axes[0].axis("off")
    axes[0].spines[["top", "right"]].set_visible(False)
    if t is not None:
        M, lab = np.asarray(t["matrix"]), t["labels"]
        im = axes[1].imshow(M, cmap=DIV_CMAP, vmin=-1, vmax=1)
        axes[1].set_xticks(range(len(lab))); axes[1].set_xticklabels(lab)
        axes[1].set_yticks(range(len(lab))); axes[1].set_yticklabels(lab)
        for i in range(len(lab)):
            for j in range(len(lab)):
                if i != j:
                    axes[1].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                                 color="white" if abs(M[i, j]) > 0.55 else "#182420")
        fig.colorbar(im, ax=axes[1], shrink=0.8, label="Spearman rho")
        axes[1].set_title("Trade-offs: co-variation across refits of the same node", fontsize=10)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    _savefig(fig, out)


def fig_rep_age(rep: dict, out: Path) -> None:
    a = rep["age"]
    if not len(a):
        return
    P = a.pivot(index="param", columns="replicate", values="cliffs_delta")
    Pp = a.pivot(index="param", columns="replicate", values="p_mw").reindex_like(P)
    P = P.reindex(list(dict.fromkeys(a.param)))[rep["labels"]]
    Pp = Pp.reindex(P.index)[rep["labels"]]
    fig, ax = plt.subplots(figsize=(1.3 * P.shape[1] + 3, 0.45 * P.shape[0] + 1.6))
    lim = max(0.2, float(np.nanmax(np.abs(P.to_numpy()))))
    im = ax.imshow(P.to_numpy(), cmap=DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v, pv = P.iloc[i, j], Pp.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}{'*' if pv < 0.05 else ''}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 * lim else "#182420")
    ax.set_xticks(range(P.shape[1])); ax.set_xticklabels([_short_rep(x) for x in P.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(P.shape[0])); ax.set_yticklabels(P.index)
    fig.colorbar(im, ax=ax, label="Cliff's delta, old vs young (whole-brain mean)")
    ax.set_title("Age effect per refit (* p < 0.05)", fontsize=10)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# glass-brain maps on a reference subject

def render_brain(positions: np.ndarray, point_region: np.ndarray, region_values: np.ndarray,
                 cmap, clim: tuple[float, float], bar_title: str, label: str, out: Path) -> None:
    shown = (point_region >= 0) & np.isfinite(region_values[point_region.clip(min=0)])
    rest = (point_region >= 0) & ~shown
    scalars = region_values[point_region[shown]]
    n = len(DIPOLE_VIEWS)
    pl = pv.Plotter(off_screen=True, shape=(1, n), window_size=(380 * n + 90, 380), border=False)
    for i, view in enumerate(DIPOLE_VIEWS):
        pl.subplot(0, i)
        if rest.any():
            pl.add_points(positions[rest], color="#b8c2bb", opacity=0.3,
                          render_points_as_spheres=True, point_size=3, lighting=False)
        pl.add_points(positions[shown], scalars=scalars, cmap=cmap, clim=clim,
                      render_points_as_spheres=True, point_size=6, lighting=True,
                      show_scalar_bar=(i == n - 1),
                      scalar_bar_args={"title": bar_title, "color": "black", "vertical": True,
                                       "width": 0.09, "height": 0.6, "position_x": 0.87, "position_y": 0.2,
                                       "title_font_size": 14, "label_font_size": 12, "fmt": "%.3g"})
        pl.set_background("white")
        offset, viewup = DIPOLE_CAMERAS[view]
        pl.view_vector(offset, viewup=viewup)
        pl.reset_camera()
        pl.add_text(view, font_size=12, color="black", position="lower_left")
    pl.subplot(0, 0)
    pl.add_text(label, font_size=11, color="black", position="upper_left")
    pl.screenshot(str(out))
    pl.close()


def write_brains(pop: dict, results: dict, params: list[str], ref_sid: str, bids_dir: Path,
                 fig_dir: Path) -> dict:
    subj = Subject(str(bids_dir), ref_sid)
    positions = subj.load.dipoles(pop["spacing"]).positions
    node = subj.load.dipole_node_labels(pop["atlas"], pop["spacing"], task=pop["fmri_task"])
    to_conn = np.asarray(subj.load.fmri_nodes(pop["atlas"], pop["fmri_task"]).to_conn)
    point_region = np.where(node >= 0, to_conn[node.clip(min=0)], -1)
    M = len(pop["regions"])
    written = {}
    for p in params:
        vals = np.full(M, np.nan)
        rs = results["all"]["region_stats"].query("param == @p")
        vals[rs.region_idx.to_numpy()] = rs["median"].to_numpy()
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        path = fig_dir / f"brain_median_{p}.png"
        render_brain(positions, point_region, vals, SEQ_CMAP, (lo, hi if hi > lo else lo + 1e-9),
                     f"median {p}", f"{p} -- population median (on sub-{ref_sid})", path)
        written[f"median|{p}"] = path
        for c, res in results.items():
            vals = np.full(M, np.nan)
            a = res["age_region"].query("param == @p")
            vals[a.region_idx.to_numpy()] = a.cliffs_delta.to_numpy()
            lim = max(0.1, float(np.nanmax(np.abs(vals))))
            path = fig_dir / f"brain_delta_{p}_{c}.png"
            render_brain(positions, point_region, vals, DIV_CMAP, (-lim, lim),
                         "Cliff's delta", f"{p} -- old vs young, {c} (on sub-{ref_sid})", path)
            written[f"delta|{p}|{c}"] = path
            cond = res.get("condition")
            if cond is None:
                continue
            A, B = cond["labels"]
            vals = np.full(M, np.nan)
            d = cond["diff_region"].query("param == @p")
            vals[d.region_idx.to_numpy()] = d.rank_biserial.to_numpy()
            lim = max(0.1, float(np.nanmax(np.abs(vals))))
            path = fig_dir / f"brain_condition_{p}_{c}.png"
            render_brain(positions, point_region, vals, DIV_CMAP, (-lim, lim),
                         "rank-biserial", f"{p} -- {B} vs {A}, {c} (on sub-{ref_sid})", path)
            written[f"cond|{p}|{c}"] = path
    return written


# ---------------------------------------------------------------------------
# markdown summary

def _sil_verdict(s: float) -> str:
    if s >= 0.5:
        return "strong cluster structure"
    if s >= 0.25:
        return "weak but present cluster structure"
    return "no substantial cluster structure (a continuum, not groups)"


def _rel_verdict(r: float) -> str:
    if r >= 0.8:
        return "highly reproducible"
    if r >= 0.5:
        return "moderately reproducible"
    return "poorly reproducible"


def _condition_markdown(cond: dict, c: str, params: list[str]) -> list[str]:
    A, B = cond["labels"]
    st = cond["stability"].set_index("param")
    L = [f"### 4. Condition: {A} vs {B} (same subjects)", "",
         f"n = {cond['n']} subjects fit in both conditions ({cond['n_young']} young, {cond['n_old']} old). "
         "The BOLD target is the same resting fMRI in both runs, so a shift reflects the EEG condition.", "",
         "**Stability.** ICC(C,1) asks whether subjects keep their *ranking* across conditions; ICC(A,1) "
         "also penalizes a systematic shift. Identification: is a subject's individual deviation map in "
         f"{B} most similar to their own map in {A}?", "",
         "| param | regions | median region ICC(C) | median region ICC(A) | regions ICC(C)>0.5 | whole-brain ICC(C) | "
         "whole-brain ICC(A) | group-map r | I_self | I_others | ident. accuracy | p (binom) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    g = lambda r, k, fmt: format(r[k], fmt) if k in r and pd.notna(r[k]) else "-"
    for p in [*params, "all params", "G"]:
        r = st.loc[p]
        L.append(f"| {p} | {g(r, 'n_regions', '.0f')} | {g(r, 'region_icc_c', '.2f')} | {g(r, 'region_icc_a', '.2f')} | "
                 f"{g(r, 'frac_region_icc_c_gt_0.5', '.0%')} | {g(r, 'global_icc_c', '.2f')} | {g(r, 'global_icc_a', '.2f')} | "
                 f"{g(r, 'group_map_r', '.2f')} | {g(r, 'i_self', '.3f')} | {g(r, 'i_others', '.3f')} | "
                 f"{g(r, 'accuracy', '.0%')} | {g(r, 'p_binom', '.2g')} |")
    L += ["", f"Chance identification accuracy: {cond['ident']['all params']['chance']:.1%}. Both fits of a subject "
          "share the structural connectome, the BOLD/FC target, the leadfield, the init and the noise seed, so part of "
          "this identifiability comes from those shared inputs rather than from a condition-invariant trait: read "
          "it as an upper bound on parameter stability.", "",
          f"![condition stability](figures/condition_stability_{c}.png)", "",
          f"![condition scatter](figures/condition_scatter_{c}.png)", "",
          f"![condition identification](figures/condition_ident_{c}.png)", "",
          f"**Shift {B} - {A}.** Wilcoxon signed-rank with matched-pairs rank-biserial r_rb (> 0 = higher in "
          f"{B}); `dz_adj` is the shift after regressing out the change in fit quality (EEG-PSD loss, fc_corr).", "",
          "| variable | median " + A + " | median " + B + " | median diff | r_rb | p |", "|---|---|---|---|---|---|"]
    for _, r in cond["fit_diff"].iterrows():
        L.append(f"| {r.variable} | {r.median_a:.4g} | {r.median_b:.4g} | {r.median_diff:+.3g} | "
                 f"{r.rank_biserial:+.2f} | {r.p_w:.3g} |")
    clf = cond["classifier"]
    L += ["", "Condition classifier (within-subject-centred features, subject-grouped CV and permutations):", "",
          "| features | CV AUC (5x10) | permutation p |", "|---|---|---|"]
    for k, v in clf.items():
        L.append(f"| {k} | {v['auc_mean']:.3f} +/- {v['auc_sd']:.3f} | {v['p_perm']:.3g} |")
    dg = cond["diff_group"]
    L += ["", "Network / global level (`age delta`: Cliff's delta of the shift, old vs young):", "",
          "| level | param | network | median diff | r_rb | q W | dz_adj | q OLS | age delta | p age |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in dg.sort_values("p_w").iterrows():
        if r.q_w < 0.1 or r.level == "global":
            L.append(f"| {r.level} | {r.param} | {r.group} | {r.median_diff:+.3g} | {r.rank_biserial:+.2f} | {r.q_w:.3g} | "
                     f"{r.dz_adj:+.2f} | {r.q_ols:.3g} | {r.age_delta:+.2f} | {r.p_age:.3g} |")
    d = cond["diff_region"]
    L += ["", f"Region-level: {len(d)} tests, {(d.q_w < 0.05).sum()} with BH q < 0.05 (Wilcoxon), "
          f"{(d.q_ols < 0.05).sum()} with q < 0.05 after the fit-quality adjustment.", "",
          f"| param | region | median {A} | median {B} | r_rb | q W | dz_adj | q OLS |", "|---|---|---|---|---|---|---|---|"]
    for _, r in d.nsmallest(15, "p_w").iterrows():
        L.append(f"| {r.param} | {r.region} | {r.median_a:.4g} | {r.median_b:.4g} | {r.rank_biserial:+.2f} | "
                 f"{r.q_w:.3g} | {r.dz_adj:+.2f} | {r.q_ols:.3g} |")
    L += ["", f"![condition diff](figures/condition_diff_{c}.png)", ""]
    al = cond.get("alpha")
    if al is not None:
        mf = al["model_fit"]
        L += [f"#### Alpha in visual cortex ({al['band'][0]:g}-{al['band'][1]:g} Hz)", "",
              f"Hypothesis: {A} -> {B} mainly changes alpha over visual cortex. Relative alpha = alpha share of the "
              f"fitted band per channel (absolute power is not available: chunks are std-normalized). Reactivity "
              f"= log rel-alpha {A} - {B} (> 0 = less alpha in {B}). Occipital ROI: {', '.join(al['roi'])}; "
              f"frontal reference: {', '.join(al['ref'])}. n = {al['n']}.", "",
              "| source | ROI | median " + A + " | median " + B + " | r_rb (" + B + " - " + A + ") | p | subjects with alpha drop |",
              "|---|---|---|---|---|---|---|"]
        for _, r in al["roi_tests"].iterrows():
            L.append(f"| {r.source} | {r.roi} | {r.median_a:.3f} | {r.median_b:.3f} | {r.rank_biserial:+.2f} | "
                     f"{r.p_w:.3g} | {r.frac_reactive:.0%} |")
        ct = al["chan_tests"]
        L += ["", f"Channels with BH q < 0.05: empirical {int((ct[ct.source == 'emp'].q_w < 0.05).sum())}, simulated "
              f"{int((ct[ct.source == 'sim'].q_w < 0.05).sum())} of {len(al['channels'])}.", "",
              f"Model vs data: topography of the median change rho = {mf['topo_rho']:.2f} (across channels); "
              f"per-subject occipital reactivity rho = {mf['react_rho'][0]:.2f} (p {mf['react_rho'][1]:.2g}); "
              f"occipital-specific reactivity rho = {mf['react_spec_rho'][0]:.2f} (p {mf['react_spec_rho'][1]:.2g}); "
              f"occipital alpha level emp vs sim: {A} rho {mf['level_rho_A'][0]:.2f}, {B} rho {mf['level_rho_B'][0]:.2f}.", "",
              f"![alpha spectra](figures/alpha_spectra_{c}.png)", "", f"![alpha topo](figures/alpha_topo_{c}.png)", "",
              f"Parameters: shift {B} - {A} in visual (Vis*) regions vs the other cortical regions, and its link to the "
              "empirical / simulated occipital alpha reactivity (Spearman across subjects; q = BH over params).", "",
              "| param | r_rb visual | r_rb other | r_rb visual-other | q specific | rho vs emp reactivity | q | "
              "rho vs sim reactivity | q | rho specific (vis-other vs occ-front) | q |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in al["param_tests"].iterrows():
            L.append(f"| {r.param} | {r.rb_visual:+.2f} | {r.rb_other:+.2f} | {r.rb_specific:+.2f} | {r.q_specific:.3g} | "
                     f"{r.rho_react_emp:+.2f} | {r.q_react_emp:.3g} | {r.rho_react_sim:+.2f} | {r.q_react_sim:.3g} | "
                     f"{r.rho_specific_emp:+.2f} | {r.q_specific_emp:.3g} |")
        g = al.get("G_link", {})
        rl = al["region_link"]
        L += ["", f"G shift vs empirical reactivity: rho {g.get('rho', np.nan):.2f} (p {g.get('p', np.nan):.2g}). "
              f"Region level: {int((rl.q < 0.05).sum())} of {len(rl)} region x param shifts correlate with the empirical "
              "reactivity at q < 0.05.", ""]
        if len(al["enrichment"]):
            L += ["| param | median rho visual regions | median rho other | p (MW) |", "|---|---|---|---|"]
            for _, r in al["enrichment"].iterrows():
                L.append(f"| {r.param} | {r.median_rho_visual:+.2f} | {r.median_rho_other:+.2f} | {r.p_mw:.3g} |")
        L += ["", f"![alpha params](figures/alpha_params_{c}.png)", ""]
    return L


def _replicate_markdown(rep: dict) -> list[str]:
    fmt = lambda v, f=".2f": "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, f)
    L = ["## Stability across initialization and noise", "",
         f"{len(rep['labels'])} refits of the same {rep['n']} subjects: {', '.join(f'`{x}`' for x in rep['labels'])}. "
         f"Init varies at fixed noise in: {', '.join(rep['sub_init']) or 'none'}; noise varies at default init in: "
         f"{', '.join(rep['sub_noise']) or 'none'}.", "",
         "### 1. Convergence", "", "| refit | median log EEG loss | median fc_corr | median G | worse than subject's best | best fit |",
         "|---|---|---|---|---|---|"]
    for _, r in rep["convergence"].iterrows():
        L.append(f"| {r.replicate} | {r.median_log_eeg_loss:.3f} | {r.median_fc_corr:.3f} | {r.median_G:.3f} | "
                 f"{r.frac_worse_than_best:.0%} | {r.frac_best:.0%} |")
    L += ["", "![convergence](figures/rep_convergence.png)", "",
          "### 2. Is the subject signal larger than the refit noise?", "",
          "ICC(1) = between-subject share of variance (> 0.5: individual differences dominate refit noise). "
          "SD fractions are in units of the parameter's bound range.", "",
          "| param | region ICC all | region ICC init | region ICC noise | regions ICC>0.5 | whole-brain ICC all | "
          "whole-brain ICC init | whole-brain ICC noise | SD within (refits) | SD between (subjects) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in rep["variance"].iterrows():
        L.append(f"| {r.param} | {fmt(r.get('region_icc_all'))} | {fmt(r.get('region_icc_init'))} | "
                 f"{fmt(r.get('region_icc_noise'))} | {fmt(r.get('frac_region_icc_gt_0.5'), '.0%')} | "
                 f"{fmt(r.global_icc_all)} | {fmt(r.global_icc_init)} | {fmt(r.global_icc_noise)} | "
                 f"{fmt(r.global_sd_within_frac, '.3f')} | {fmt(r.global_sd_between_frac, '.3f')} |")
    L += ["", "![icc](figures/rep_icc.png)", "", "### 3. Do individual profiles survive a refit?", "",
          "| param | refits differ in | accuracy | I_self | I_others |", "|---|---|---|---|---|"]
    for _, r in rep["ident_summary"].iterrows():
        L.append(f"| {r.param} | {r.kind} | {r.accuracy:.0%} | {r.i_self:.3f} | {r.i_others:.3f} |")
    if rep["ident_ref"]:
        r = rep["ident_ref"]
        L += ["", f"Reference: {r['labels'][0]} vs {r['labels'][1]} on the same {r['n']} subjects, all params: accuracy "
              f"{r['accuracy']:.0%}, I_self - I_others {r['i_diff']:.3f}. If refits identify subjects much better than "
              "the condition change does, the condition really changes individual profiles."]
    L += ["", "![ident](figures/rep_ident.png)", "", "### 4-6. Init memory, group maps, trade-offs", ""]
    if len(rep["memory"]):
        L += ["| param | fits | median rho(final, start) | median moved (range frac.) | nodes moved < 2% |", "|---|---|---|---|---|"]
        for _, r in rep["memory"].iterrows():
            L.append(f"| {r.param} | {r.n_fits} | {r.median_rho_final_init:+.2f} | {r.median_moved_frac:.3f} | {r.frac_nodes_unmoved:.0%} |")
        L += ["", "A high rho(final, start) means the fitted map mostly reflects the random start: the data do not "
              "constrain that parameter.", ""]
    L += ["| param | group-map r (mean pair) | min pair |", "|---|---|---|"]
    for _, r in rep["group_maps"].iterrows():
        L.append(f"| {r.param} | {r.group_map_r_mean:.2f} | {r.group_map_r_min:.2f} |")
    L += ["", "![memory and trade-offs](figures/rep_memory_tradeoff.png)", "", "### 7. Do the conclusions survive?", ""]
    if len(rep["age_summary"]):
        L += ["Whole-brain age effect (Cliff's delta, old vs young) recomputed in every refit:", "",
              "| param | median delta | range | refits with p<0.05 | same sign in all |", "|---|---|---|---|---|"]
        for _, r in rep["age_summary"].iterrows():
            L.append(f"| {r.param} | {r.median_delta:+.2f} | [{r.min_delta:+.2f}, {r.max_delta:+.2f}] | "
                     f"{r.n_sig}/{len(rep['labels'])} | {'yes' if r.same_sign else 'no'} |")
        L += ["", "![age](figures/rep_age.png)", ""]
    if rep["snr"] is not None:
        A, B = rep["snr"]["labels"]
        L += [f"{B} - {A} shift in units of each subject's refit SD (median over subjects, per region):", "",
              "| param | subjects | regions | median abs(shift/SD) | regions > 1 | significant regions | of which > 1 |",
              "|---|---|---|---|---|---|---|"]
        for _, r in rep["snr"]["table"].iterrows():
            L.append(f"| {r.param} | {r.n_subjects} | {r.n_regions} | {r.median_abs_shift_over_refit_sd:.2f} | "
                     f"{r.frac_regions_gt_1:.0%} | {r.n_sig_regions} | {fmt(r.frac_sig_regions_gt_1, '.0%')} |")
        L += ["", "Only condition shifts larger than the refit SD (ratio > 1) are clearly beyond what refitting alone produces.", ""]
    return L


def write_markdown(pop: dict, results: dict, params: list[str], frozen: list[str], excluded: pd.DataFrame,
                   args: argparse.Namespace, out: Path, rep: dict | None = None) -> None:
    meta = pop["meta"]
    L = [
        f"# Population parameter report -- atlas-{pop['atlas']}",
        "",
        f"Results root `{args.results_root}`, BIDS `{args.bids_dir}`. "
        f"{len(meta)} finished fits ({len(pop['incomplete'])} run folders without `optimized_params.npz`), "
        f"{len(excluded)} excluded (`gamma_weight != 0` or no age). Age groups: young < {args.age_split:g} <= old.",
        "",
        *([f"Condition comparison: the same subjects fit again under `{args.compare_root}` (section 4 of each cohort).", ""]
          if args.compare_root else []),
        f"Parameters analysed: {', '.join(params)} (each on its live node type only)."
        + (f" Dropped as never moving from init: {', '.join(frozen)}." if frozen else ""),
        f" Regions need >= {args.min_coverage:.0%} subject coverage. Fit-quality covariates: sex, "
        "log10 EEG-PSD loss (mean of last 10 epochs), fc_corr.",
        "",
        "![fit quality](figures/fit_quality.png)",
        "",
    ]
    n_fail = int(meta.eeg_fit_failed.iloc[results["all"]["idx"]].sum())
    if n_fail:
        n_fail_bold = int(meta.eeg_fit_failed.iloc[results["bold_ok"]["idx"]].sum())
        L += [f"> **{n_fail} of the {results['all']['n']} fits in `all` look failed**: final EEG-PSD loss > "
              f"{args.eeg_fail_loss:g} (good fits sit around 1e-7). These fits typically also have G ~ 0 and no BOLD "
              f"improvement. {n_fail_bold} of them are in `bold_ok`. They are flagged `eeg_fit_failed` in "
              "`tables/subjects.csv`. In `all`, expect them to dominate PC1 and the fit-quality covariates.", ""]
    for c, res in results.items():
        title = COHORTS[c].format(thr=args.fc_threshold)
        pca, rci = res["pca"], res["region_cluster_info"]
        L += [f"## Cohort `{c}` -- {title}", "",
              f"n = {res['n']} ({res['n_young']} young, {res['n_old']} old).", "",
              "### 1. Regional variation across subjects", "",
              "| param | regions | split-half r (group map) | Spearman-Brown | mean subject-subject r | ICC(region) | verdict |",
              "|---|---|---|---|---|---|---|"]
        for p in params:
            r = res["reliability"][p]
            L.append(f"| {p} | {len(res['cols'][p])} | {r['split_half_r']:.2f} | {r['split_half_sb']:.2f} | "
                     f"{r['pairwise_subject_r']:.2f} | {r['icc_region']:.2f} | {_rel_verdict(r['split_half_sb'])} group map |")
        L += ["",
              "Split-half r: do two random halves of the cohort give the same region ranking? (reproducibility of the "
              "*group* map). Subject-subject r: how similar are two individuals' maps? ICC: share of variance "
              "explained by region identity.",
              "", f"![region spread](figures/region_spread_{c}.png)", "",
              f"![reliability](figures/reliability_{c}.png)", "",
              "### 2. Clustering", "",
              f"Subjects: {pca['n_features']} features -> {pca['n_pc']} PCs retained for clustering ({pca['cum_var']:.0%} variance; "
              f"PC1 {pca['evr'][0]:.1%}, PC2 {pca['evr'][1]:.1%}; {pca['n_pc_50']} PCs needed for 50%, so a flat "
              f"spectrum means no dominant axes). Best k-means k = {pca['best_k']} with silhouette "
              f"{pca['silhouette'][pca['best_k']]:.2f} -> **{_sil_verdict(pca['silhouette'][pca['best_k']])}**. "
              f"Cluster vs age group: chi2 p = {pca['chi2_p']:.3g}, ARI = {pca['ari_age']:.3f}.", "",
              "| PC | var | AUC old>young | p | AUC male | rho EEG-PSD loss | rho fc_corr | rho G |",
              "|---|---|---|---|---|---|---|---|"]
        for a in pca["assoc"]:
            L.append(f"| PC{a['pc']} | {a['var_explained']:.1%} | {a['auc_old']:.2f} | {a['p_old']:.2g} | "
                     f"{a['auc_male']:.2f} | {a['rho_eeg_psd_loss']:.2f} | {a['rho_fc_corr']:.2f} | {a['rho_G']:.2f} |")
        L += ["", "Top PC1 loadings: " + ", ".join(f"`{n}` ({v:+.2f})" for n, v in pca["loadings"]["PC1"][:6]), "",
              f"![pca](figures/pca_{c}.png)", "",
              f"Regions (median {', '.join(res['node_corr']['labels'])} per region): best k = {rci['best_k']}, "
              f"silhouette {rci['silhouette'][rci['best_k']]:.2f}; agreement with functional groups ARI = "
              f"{rci['ari_group']:.3f}.", "",
              f"![param corr](figures/param_corr_{c}.png)", "",
              "### 3. Age", ""]
        clf = res["classifier"]
        L += ["| features | CV AUC (5x10) | permutation p |", "|---|---|---|"]
        for k, v in clf.items():
            L.append(f"| {k} | {v['auc_mean']:.3f} +/- {v['auc_sd']:.3f} | {v['p_perm']:.3g} |")
        L += ["", "Read `params` against `fit_quality`: if fit quality alone predicts age as well, the parameter "
              "differences may just reflect how well old vs young subjects are fit.", "",
              f"![classifier](figures/age_classifier_{c}.png)", ""]
        a = res["age_region"]
        L += [f"Region-level: {len(a)} tests, {(a.q_mw < 0.05).sum()} with BH q < 0.05 (Mann-Whitney), "
              f"{(a.q_ols < 0.05).sum()} with q < 0.05 for the covariate-adjusted OLS.", "",
              "| param | region | delta | median young | median old | q MW | beta OLS | q OLS |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in a.nsmallest(15, "p_mw").iterrows():
            L.append(f"| {r.param} | {r.region} | {r.cliffs_delta:+.2f} | {r.median_young:.4g} | {r.median_old:.4g} | "
                     f"{r.q_mw:.3g} | {r.beta_old:+.2f} | {r.q_ols:.3g} |")
        g = res["age_group"]
        L += ["", "Network / global level:", "",
              "| level | param | network | delta | q MW | beta OLS | q OLS |", "|---|---|---|---|---|---|---|"]
        for _, r in g.sort_values("p_mw").iterrows():
            if r.q_mw < 0.1 or r.level == "global":
                L.append(f"| {r.level} | {r.param} | {r.group} | {r.cliffs_delta:+.2f} | {r.q_mw:.3g} | "
                         f"{r.beta_old:+.2f} | {r.q_ols:.3g} |")
        L += ["", f"![volcano](figures/age_volcano_{c}.png)", ""]
        if res.get("condition") is not None:
            L += _condition_markdown(res["condition"], c, params)
    if rep is not None:
        L += _replicate_markdown(rep)
    L += ["## Caveats", "",
          "- Age comes in 5-year bins and is bimodal, so it is analysed as a binary young/old contrast.",
          "- Young subjects are mostly male; sex is a covariate in the OLS model and the fit-quality baseline.",
          "- In the `all` cohort many subjects have a failed BOLD fit (fc_corr ~ 0): their parameters are "
          "constrained mostly by EEG. Trust effects that hold in `bold_ok` more.",
          "- Glass-brain maps are drawn on one reference subject's dipole cloud (native space): a display "
          "device, not a registration.", ""]
    out.write_text("\n".join(L))


# ---------------------------------------------------------------------------
# interactive HTML

def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python, floats to 5 significant digits."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(f"{o:.5g}") if np.isfinite(o) else None
    return o


def build_html_data(pop: dict, results: dict, params: list[str], frozen: list[str], excluded: pd.DataFrame,
                    brains: dict, cond_figs: dict, ref_sid: str | None, args: argparse.Namespace,
                    rep: dict | None = None) -> dict:
    meta = pop["meta"]
    regions = pop["regions"]
    base = results["all"]["idx"]
    subjects = meta.iloc[base].reset_index(drop=True)
    in_bold = set(results["bold_ok"]["idx"].tolist())
    subj_rows = [{
        "id": r.participant_id, "age": r.age, "old": bool(r.old), "sex": r.sex,
        "fc_corr": r.fc_corr, "eeg": r.log_eeg_psd_loss, "failed": bool(r.eeg_fit_failed), "eeg_ratio": r.eeg_loss_ratio, "G": r.G,
        "bold_ok": int(base[i]) in in_bold,
    } for i, r in enumerate(subjects.itertuples())]
    values = {p: pop["V"][p][base] for p in params}

    cohorts = {}
    for c, res in results.items():
        pos = {int(g): i for i, g in enumerate(base)}
        member = [pos[int(g)] for g in res["idx"]]
        pca = res["pca"]
        cohorts[c] = {
            "label": COHORTS[c].format(thr=args.fc_threshold),
            "n": res["n"], "n_young": res["n_young"], "n_old": res["n_old"], "members": member,
            "cols": res["cols"],
            "region_stats": res["region_stats"].to_dict("records"),
            "reliability": res["reliability"],
            "node_corr": res["node_corr"], "subj_corr": res["subj_corr"],
            "pca": {"evr": pca["evr"], "n_pc": pca["n_pc"], "n_pc_50": pca["n_pc_50"], "cum_var": pca["cum_var"], "scores": pca["scores"], "silhouette": pca["silhouette"],
                    "best_k": pca["best_k"], "clusters": pca["clusters"], "chi2_p": pca["chi2_p"],
                    "ari_age": pca["ari_age"], "assoc": pca["assoc"], "loadings": pca["loadings"],
                    "n_features": pca["n_features"]},
            "region_clusters": res["region_clusters"].to_dict("records"),
            "region_cluster_info": res["region_cluster_info"],
            "age_region": res["age_region"].to_dict("records"),
            "age_group": res["age_group"].to_dict("records"),
            "classifier": res["classifier"], "top_coefs": res["top_coefs"],
            "condition": None,
        }
        cond = res.get("condition")
        if cond is not None:
            cohorts[c]["condition"] = {
                "n": cond["n"], "n_young": cond["n_young"], "n_old": cond["n_old"],
                "pairs": [{"id": r.participant_id, "age": r.age, "old": bool(r.old), "sex": r.sex}
                          for r in cond["subjects"].itertuples()],
                "vars": {v: np.column_stack(ab) for v, ab in cond["variables"].items()},
                "var_stats": cond["var_stats"],
                "stability": cond["stability"].to_dict("records"),
                "ident_all": {k: v for k, v in cond["ident"]["all params"].items() if k != "S"},
                "fit_diff": cond["fit_diff"].to_dict("records"),
                "diff_group": cond["diff_group"].to_dict("records"),
                "diff_region": cond["diff_region"].to_dict("records"),
                "classifier": cond["classifier"],
                "alpha": None if cond["alpha"] is None else {
                    k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                    for k, v in cond["alpha"].items()
                    if k in ("band", "roi", "ref", "n", "model_fit", "roi_tests", "param_tests", "enrichment", "G_link")},
            }

    def b64(path: Path) -> str:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    return _clean({
        "atlas": pop["atlas"], "params": params, "frozen": frozen,
        "init": pop["init"], "bounds": pop["bounds"], "units": PARAM_UNITS, "live_on": LIVE_ON,
        "fc_threshold": args.fc_threshold, "eeg_fail_loss": args.eeg_fail_loss, "age_split": args.age_split, "min_coverage": args.min_coverage,
        "n_finished": len(meta), "n_incomplete": len(pop["incomplete"]),
        "excluded": excluded[["participant_id", "gamma_weight", "age"]].to_dict("records"),
        "regions": regions, "display_order": pop["display_order"], "group_order": GROUP_ORDER,
        "subjects": subj_rows, "values": values, "cohorts": cohorts,
        "brains": {k: b64(v) for k, v in brains.items()}, "ref_subject": ref_sid,
        "cond_labels": next((r["condition"]["labels"] for r in results.values() if r.get("condition")), None),
        "compare_root": str(args.compare_root) if args.compare_root else None,
        "cond_figs": {k: b64(v) for k, v in cond_figs.items()},
        "rep": None if rep is None else {
            "labels": rep["labels"], "n": rep["n"], "sub_init": rep["sub_init"], "sub_noise": rep["sub_noise"],
            **{k: rep[k].to_dict("records") for k in ("convergence", "variance", "ident_summary", "memory",
                                                       "group_maps", "age_summary")},
            "ident_ref": rep["ident_ref"],
            "snr": None if rep["snr"] is None else {"labels": rep["snr"]["labels"],
                                                     "table": rep["snr"]["table"].to_dict("records")}},
        "n_perm": args.n_perm, "clf_c": args.clf_c,
        "config": {k: pop["config"].get(k) for k in ("leadfield_label", "bold_model", "schedule", "num_epochs", "fs")},
    })


def write_html(data: dict, out: Path) -> None:
    style = _SUBJECT_HTML[_SUBJECT_HTML.index("<style>"):_SUBJECT_HTML.index("</style>")]
    html = (POP_TEMPLATE
            .replace("__BASE_STYLE__", style.removeprefix("<style>"))
            .replace("__DATA__", json.dumps(data, allow_nan=False).replace("</", "<\\/")))
    out.write_text(html)


POP_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Population Parameter Report</title>
<style>
__BASE_STYLE__
:root{ --young:#2a78d6; --old:#eb6834; --div-neg:#184f95; --div-mid:#f0efec; --div-pos:#b8322f; --sig:#104281; --muted-pt:#a9b3ac; }
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){ --young:#3987e5; --old:#d95926; --div-neg:#6da7ec; --div-mid:#383835; --div-pos:#e66767; --sig:#86b6ef; --muted-pt:#5c6b62; } }
:root[data-theme="dark"]{ --young:#3987e5; --old:#d95926; --div-neg:#6da7ec; --div-mid:#383835; --div-pos:#e66767; --sig:#86b6ef; --muted-pt:#5c6b62; }
.page{padding:0 16px 80px;}
.cohort-bar{display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-left:auto;}
.tocbar{flex-wrap:wrap; align-items:center;}
.seg{display:inline-flex; border:1px solid var(--border-2); border-radius:8px; overflow:hidden;}
.seg button{font:500 12.5px "IBM Plex Sans",sans-serif; padding:6px 12px; border:0; background:var(--surface); color:var(--ink-2); cursor:pointer;}
.seg button + button{border-left:1px solid var(--border-2);}
.seg button.active{background:var(--accent); color:var(--surface);}
.controls{display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:14px; font-size:13px; color:var(--ink-2);}
select{font:13px "IBM Plex Sans",sans-serif; padding:5px 8px; border-radius:6px; border:1px solid var(--border-2); background:var(--surface); color:var(--ink);}
.legend{display:flex; gap:16px; font-size:12px; color:var(--ink-2); align-items:center; flex-wrap:wrap;}
.legend .sw{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px;}
.grid2{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:18px;}
.grid-side{display:grid; grid-template-columns:1fr 320px; gap:18px; margin-bottom:18px;}
.pad{padding:16px 18px;}
.card h4{margin:0 0 4px; font:600 14px "IBM Plex Sans",sans-serif;}
.note{font-size:12px; color:var(--ink-muted); margin-bottom:8px;}
canvas{display:block; width:100%;}
table.t{width:100%; border-collapse:collapse; font-size:12.5px;}
table.t th{text-align:left; font-weight:600; color:var(--ink-muted); border-bottom:1px solid var(--border-2); padding:6px 8px; white-space:nowrap;}
table.t td{border-bottom:1px solid var(--border); padding:5px 8px; font-variant-numeric:tabular-nums;}
table.t td.num{font-family:"IBM Plex Mono",monospace; text-align:right;}
table.t tr.hl td{background:var(--accent-soft);}
.tablewrap{max-height:420px; overflow:auto;}
.verdict{display:inline-block; font-size:11.5px; padding:2px 8px; border-radius:999px; background:var(--surface-2); border:1px solid var(--border); color:var(--ink-2);}
.brain-img{width:100%; display:block; border-radius:6px; background:#fff;}
details summary{cursor:pointer; color:var(--accent); font-size:13px; margin:8px 0;}
@media (max-width: 880px){ .grid2, .grid-side{grid-template-columns:1fr;} .hero h1{font-size:30px;} }
</style></head><body>
<div class="page">
  <header class="hero">
    <div class="eyebrow">TVB-optim &middot; population parameter report</div>
    <h1 id="h1"></h1>
    <p class="dek" id="dek"></p>
    <div class="chips" id="chips"></div>
  </header>

  <nav class="tocbar">
    <a href="#cohort">Cohort</a><a href="#regions">Regional variation</a><a href="#clusters">Clustering</a><a href="#age">Age</a><a href="#condition" id="nav-cond" style="display:none">Condition</a><a href="#replicates" id="nav-rep" style="display:none">Refit stability</a><a href="#methods">Methods</a>
    <div class="cohort-bar"><span style="font-size:12px;color:var(--ink-muted)">cohort</span><div class="seg" id="cohort-seg"></div></div>
  </nav>

  <section id="cohort">
    <div class="section-kicker">01 &middot; Cohort &amp; fit quality</div>
    <h2>Who is in the analysis, and how well were they fit?</h2>
    <p class="section-lede">Every analysis below runs on the cohort selected in the bar above: either all fits, or only subjects whose BOLD fit worked. Subjects outside the selected cohort show as hollow markers here.</p>
    <div class="stat-grid" id="cohort-stats"></div>
    <div class="callout" id="fail-callout" style="display:none"></div>
    <div class="card pad">
      <h4>EEG-PSD loss vs FC correlation</h4>
      <div class="note">Each dot is a subject. The dashed line is the fc_corr threshold for the BOLD-fit subset. Lower EEG loss is better.</div>
      <div class="legend" id="fq-legend"></div>
      <svg id="fq-svg" viewBox="0 0 900 340" width="100%"></svg>
    </div>
  </section>

  <section id="regions">
    <div class="section-kicker">02 &middot; Regional variation</div>
    <h2>How each parameter varies per region across subjects</h2>
    <p class="section-lede">Each column is a region and each dot is one subject's fitted value; the lines are the young and old medians. Pick a parameter, zoom to one network, then hover a dot or click a region for its details.</p>
    <div class="pill-row" id="param-pills"></div>
    <div class="stat-grid" id="rel-stats"></div>
    <div class="controls">
      <label>network <select id="group-sel"></select></label>
      <div class="legend"><span><span class="sw" style="background:var(--young)"></span>young</span><span><span class="sw" style="background:var(--old)"></span>old</span><span><span class="sw" style="background:transparent;border:1px dashed var(--ink-muted);border-radius:0;height:0;width:14px"></span>init</span></div>
    </div>
    <div class="grid-side">
      <div class="card pad"><canvas id="strip" height="380"></canvas></div>
      <div class="card pad" id="region-detail"><div class="note">Click a region column to see young vs old for it.</div></div>
    </div>
    <div class="card pad">
      <div class="controls" style="margin-bottom:8px">
        <h4 style="margin:0">Glass brain</h4>
        <div class="seg" id="brain-seg"><button data-k="median" class="active">population median</button><button data-k="delta">age effect (Cliff's &delta;)</button></div>
        <span class="note" id="brain-note" style="margin:0"></span>
      </div>
      <img id="brain-img" class="brain-img" alt="glass brain map">
    </div>
  </section>

  <section id="clusters">
    <div class="section-kicker">03 &middot; Clustering</div>
    <h2>Do subjects or regions form groups?</h2>
    <p class="section-lede">For subjects, every live parameter at every covered region becomes one feature. The features are z-scored and reduced with PCA, and k-means runs on the PCs that explain 80% of the variance. For regions, clustering uses each region's median Jansen-Rit parameters, which are then compared with the functional networks.</p>
    <div class="stat-grid" id="clu-stats"></div>
    <div class="grid-side">
      <div class="card pad">
        <div class="controls">
          <label>x <select id="pcx"></select></label><label>y <select id="pcy"></select></label>
          <label>color by <select id="pc-color"></select></label>
        </div>
        <div class="legend" id="pc-legend"></div>
        <svg id="pca-svg" viewBox="0 0 640 420" width="100%"></svg>
      </div>
      <div style="display:flex;flex-direction:column;gap:18px">
        <div class="card pad"><h4>Variance explained</h4><div class="note">The first 20 PCs. Darker bars are kept for clustering, and a flat spectrum means there are no dominant axes.</div><svg id="scree-svg" viewBox="0 0 300 150" width="100%"></svg></div>
        <div class="card pad"><h4>Silhouette by k</h4><div class="note">Above 0.5 means real clusters; below 0.25 means no structure.</div><svg id="sil-svg" viewBox="0 0 300 150" width="100%"></svg></div>
      </div>
    </div>
    <div class="card pad" style="margin-bottom:18px"><h4>What the first PCs track</h4><div class="note">AUC is the chance that a random old (or male) subject scores higher than a random young (or female) one; 0.5 means no association. &rho; is the Spearman correlation.</div><div class="tablewrap"><table class="t" id="pc-assoc"></table></div>
      <details><summary>Top loadings on PC1 / PC2</summary><div class="grid2" id="loadings"></div></details></div>
    <div class="grid2">
      <div class="card pad"><div class="controls" style="margin-bottom:6px"><h4 style="margin:0">Parameter correlations</h4><div class="seg" id="corr-seg"><button data-k="node_corr" class="active">node, within-region</button><button data-k="subj_corr">subject level</button></div></div>
        <div class="note" id="corr-note"></div><svg id="corr-svg" viewBox="0 0 460 420" width="100%"></svg></div>
      <div class="card pad"><h4>Region clusters vs functional networks</h4><div class="note" id="rclu-note"></div><div class="tablewrap"><table class="t" id="rclu-table"></table></div></div>
    </div>
  </section>

  <section id="age">
    <div class="section-kicker">04 &middot; Age</div>
    <h2>Do parameters separate young from old?</h2>
    <p class="section-lede" id="age-lede"></p>
    <div class="stat-grid" id="clf-stats"></div>
    <div class="card pad" style="margin-bottom:18px"><h4>Permutation nulls</h4><div class="note">Gray bars show the cross-validated AUC under shuffled age labels; the line is the real AUC. Compare <b>params</b> with <b>fit_quality</b>: if fit quality alone predicts age just as well, the parameter differences may only reflect how well each group was fit.</div><svg id="null-svg" viewBox="0 0 960 200" width="100%"></svg>
      <details><summary>Most predictive features (params model, standardized coefficients)</summary><table class="t" id="coef-table"></table></details></div>
    <div class="grid2">
      <div class="card pad">
        <div class="controls"><h4 style="margin:0">Per-region effects</h4>
          <div class="seg" id="test-seg"><button data-k="mw" class="active">Mann&ndash;Whitney</button><button data-k="ols">OLS + covariates</button></div>
          <label>param <select id="volc-param"></select></label></div>
        <div class="legend"><span><span class="sw" style="background:var(--sig)"></span>BH q &lt; 0.05</span><span><span class="sw" style="background:var(--muted-pt)"></span>not significant</span><span class="note" style="margin:0">x &gt; 0 means higher in old</span></div>
        <svg id="volc-svg" viewBox="0 0 640 400" width="100%"></svg>
      </div>
      <div class="card pad"><h4>Network &amp; global level</h4><div class="note">The parameter is averaged over each network's regions per subject, and FDR is applied across these tests.</div><div class="tablewrap" style="max-height:440px"><table class="t" id="net-table"></table></div></div>
    </div>
    <div class="card pad"><h4>Top region-level hits</h4><div class="tablewrap"><table class="t" id="hits-table"></table></div></div>
  </section>

  <section id="condition" style="display:none">
    <div class="section-kicker">05 &middot; Condition</div>
    <h2 id="cond-h2"></h2>
    <p class="section-lede" id="cond-lede"></p>
    <div class="callout" id="cond-missing" style="display:none"></div>
    <div id="cond-body">
    <div class="stat-grid" id="cond-stats"></div>
    <div class="card pad" style="margin-bottom:18px"><h4>Stability per parameter</h4>
      <div class="note"><b>ICC(C,1)</b>: do subjects keep their ranking across conditions? <b>ICC(A,1)</b>: absolute agreement, which a systematic shift also lowers. <b>Group-map r</b>: Spearman between the two conditions' median regional maps. <b>Identification</b>: each feature is centred per condition, so the group map is removed and only individual deviations remain; is each subject's deviation map in one condition most similar to their own map in the other? I_self / I_others are the mean same-subject / different-subject similarities.</div>
      <div class="tablewrap"><table class="t" id="cond-stab"></table></div></div>
    <div class="grid2">
      <div class="card pad"><div class="controls" style="margin-bottom:6px"><h4 style="margin:0">Same subject, two conditions</h4><label>variable <select id="cond-var"></select></label></div>
        <div class="note" id="cond-scatter-note"></div>
        <div class="legend"><span><span class="sw" style="background:var(--young)"></span>young</span><span><span class="sw" style="background:var(--old)"></span>old</span><span><span class="sw" style="background:transparent;border:1px dashed var(--ink-muted);border-radius:0;height:0;width:14px"></span>identity</span></div>
        <svg id="cond-svg" viewBox="0 0 520 440" width="100%"></svg></div>
      <div class="card pad"><h4>Subject identification (all parameters)</h4><div class="note" id="cond-ident-note"></div><img id="cond-ident-img" class="brain-img" alt="identification matrix"></div>
    </div>
    <div class="card pad" style="margin-bottom:18px"><h4>Per-region shift</h4><div class="note" id="cond-diff-note"></div><img id="cond-diff-img" class="brain-img" alt="per-region shift"></div>
    <div class="card pad" style="margin-bottom:18px">
      <div class="controls" style="margin-bottom:8px"><h4 style="margin:0">Glass brain: shift</h4><label>param <select id="cond-brain-param"></select></label><span class="note" id="cond-brain-note" style="margin:0"></span></div>
      <img id="cond-brain-img" class="brain-img" alt="glass brain shift map"></div>
    <div class="grid2">
      <div class="card pad"><h4>Fit quality and condition classifier</h4><div class="note" id="cond-clf-note"></div><table class="t" id="cond-fit"></table><table class="t" id="cond-clf" style="margin-top:12px"></table></div>
      <div class="card pad"><h4>Network &amp; global shift</h4><div class="note" id="cond-net-note"></div><div class="tablewrap" style="max-height:440px"><table class="t" id="cond-net"></table></div></div>
    </div>
    <div class="card pad" style="margin-bottom:18px"><h4>Top region-level shifts</h4><div class="tablewrap"><table class="t" id="cond-hits"></table></div></div>
    <div id="alpha-block">
      <h3 style="margin:28px 0 6px">Alpha in visual cortex</h3>
      <p class="section-lede" id="alpha-lede"></p>
      <div class="stat-grid" id="alpha-stats"></div>
      <div class="card pad" style="margin-bottom:18px"><h4>Scalp: is the alpha change occipital?</h4><div class="note" id="alpha-roi-note"></div><table class="t" id="alpha-roi"></table>
        <img id="alpha-spec-img" class="brain-img" style="margin-top:12px" alt="alpha ROI spectra"><img id="alpha-topo-img" class="brain-img" style="margin-top:12px" alt="alpha topography"></div>
      <div class="card pad"><h4>Parameters: is the shift visual, and does it track alpha?</h4><div class="note" id="alpha-par-note"></div><div class="tablewrap"><table class="t" id="alpha-par"></table></div>
        <img id="alpha-par-img" class="brain-img" style="margin-top:12px" alt="parameter shift vs alpha"></div>
    </div>
    </div>
  </section>

  <section id="replicates" style="display:none">
    <div class="section-kicker" id="rep-kicker">06 &middot; Refit stability</div>
    <h2>Are the parameters stable across initialization and noise?</h2>
    <p class="section-lede" id="rep-lede"></p>
    <div class="stat-grid" id="rep-stats"></div>
    <div class="card pad" style="margin-bottom:18px"><h4>1 &middot; Convergence</h4><div class="note">Parameter differences between refits only mean something if the refits fit equally well. &ldquo;Worse than best&rdquo; = final log10 EEG loss more than the tolerance above that subject's best refit.</div><table class="t" id="rep-conv"></table><img class="brain-img" style="margin-top:12px" id="rep-conv-img" alt="convergence"></div>
    <div class="card pad" style="margin-bottom:18px"><h4>2 &middot; Subject signal vs refit noise</h4><div class="note">ICC(1) is the share of variance between subjects rather than between refits of the same subject. Above 0.5, individual differences dominate the refit noise. &ldquo;init&rdquo; uses only refits that differ in init (noise fixed); &ldquo;noise&rdquo; only refits that differ in noise (default init).</div><div class="tablewrap"><table class="t" id="rep-var"></table></div><img class="brain-img" style="margin-top:12px" id="rep-icc-img" alt="ICC"></div>
    <div class="card pad" style="margin-bottom:18px"><h4>3 &middot; Do individual profiles survive a refit?</h4><div class="note" id="rep-ident-note"></div><div class="tablewrap"><table class="t" id="rep-ident"></table></div><img class="brain-img" style="margin-top:12px" id="rep-ident-img" alt="identification"></div>
    <div class="grid2">
      <div class="card pad"><h4>4 &middot; Does a fit remember its start?</h4><div class="note">For refits with a random start: Spearman between each node's fitted value and its start, across regions. High &rho; or many unmoved nodes means the data do not constrain that parameter.</div><table class="t" id="rep-mem"></table></div>
      <div class="card pad"><h4>5 &middot; Group-map stability</h4><div class="note">Spearman between the group-median regional maps of every pair of refits.</div><table class="t" id="rep-maps"></table></div>
    </div>
    <div class="card pad" style="margin:18px 0"><h4>6 &middot; Trade-offs</h4><img class="brain-img" id="rep-mem-img" alt="init memory and trade-offs"></div>
    <div class="card pad"><h4>7 &middot; Do the conclusions survive?</h4><div class="note" id="rep-age-note"></div><table class="t" id="rep-age"></table><img class="brain-img" style="margin-top:12px" id="rep-age-img" alt="age per refit"><table class="t" id="rep-snr" style="margin-top:12px"></table></div>
  </section>

  <section id="methods">
    <div class="section-kicker" id="methods-kicker">05 &middot; Methods &amp; caveats</div>
    <h2>How to read this</h2>
    <ul class="caveat-list" id="caveats"></ul>
  </section>
  <footer id="footer"></footer>
</div>
<div id="tooltip"></div>

<script id="pop-data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('pop-data').textContent);
const PARAMS = D.params, REG = D.regions, SUBJ = D.subjects;
let cohort = 'all', param = PARAMS.includes('B') ? 'B' : PARAMS[0], group = 'all', selRegion = null;
let brainKind = 'median', corrKind = 'node_corr', testKind = 'mw', volcParam = 'all';
const tip = document.getElementById('tooltip');
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const f = (x, d=3) => x == null ? '&ndash;' : Number(x).toFixed(d);
const g3 = x => x == null ? '&ndash;' : (Math.abs(x) >= 1e3 || (Math.abs(x) < 1e-2 && x !== 0) ? Number(x).toExponential(2) : Number(x).toPrecision(3));
const pf = p => p == null ? '&ndash;' : (p < 1e-3 ? Number(p).toExponential(1) : Number(p).toFixed(3));
function showTip(h, e){ tip.innerHTML = h; tip.classList.add('show'); moveTip(e); }
function moveTip(e){ let x=e.clientX+14, y=e.clientY+14; if(x+240>innerWidth) x=e.clientX-254; if(y+110>innerHeight) y=e.clientY-120; tip.style.left=x+'px'; tip.style.top=y+'px'; }
function hideTip(){ tip.classList.remove('show'); }
function bindTips(root){ root.querySelectorAll('[data-tip]').forEach(el=>{ el.addEventListener('mouseenter', e=>showTip(el.dataset.tip, e)); el.addEventListener('mousemove', moveTip); el.addEventListener('mouseleave', hideTip); }); }
function hex2rgb(h){ h=h.replace('#',''); return [0,2,4].map(i=>parseInt(h.slice(i,i+2),16)); }
function mix(a,b,t){ const A=hex2rgb(a),B=hex2rgb(b); return `rgb(${A.map((v,i)=>Math.round(v+(B[i]-v)*t)).join(',')})`; }
function seqColor(t){ const s=['--seq-100','--seq-250','--seq-400','--seq-500','--seq-650'].map(css); t=Math.max(0,Math.min(1,t))*(s.length-1); const i=Math.min(s.length-2,Math.floor(t)); return mix(s[i],s[i+1],t-i); }
function divColor(v){ v=Math.max(-1,Math.min(1,v)); return v<0 ? mix(css('--div-mid'),css('--div-neg'),-v) : mix(css('--div-mid'),css('--div-pos'),v); }
const C = () => D.cohorts[cohort];
const short = n => n.replace(/^\d+Networks_/,'');
function seg(id, cb){ document.querySelectorAll(`#${id} button`).forEach(b=>b.addEventListener('click',()=>{ document.querySelectorAll(`#${id} button`).forEach(x=>x.classList.toggle('active',x===b)); cb(b.dataset.k); })); }
function statTiles(id, tiles){ document.getElementById(id).innerHTML = tiles.map(([l,v,s])=>`<div class="stat"><div class="lbl">${l}</div><div class="val">${v}</div><div class="sub">${s||''}</div></div>`).join(''); }
function axes(W,H,m,xr,yr,xl,yl,nt=4){
  const X=v=>m.l+(v-xr[0])/(xr[1]-xr[0])*(W-m.l-m.r), Y=v=>H-m.b-(v-yr[0])/(yr[1]-yr[0])*(H-m.t-m.b);
  let s='';
  for(let i=0;i<=nt;i++){ const yv=yr[0]+(yr[1]-yr[0])*i/nt, xv=xr[0]+(xr[1]-xr[0])*i/nt;
    s+=`<line class="gridline" x1="${m.l}" x2="${W-m.r}" y1="${Y(yv)}" y2="${Y(yv)}"/><text x="${m.l-6}" y="${Y(yv)+3.5}" font-size="10" text-anchor="end">${g3(yv)}</text>`;
    s+=`<text x="${X(xv)}" y="${H-m.b+15}" font-size="10" text-anchor="middle">${g3(xv)}</text>`; }
  s+=`<line class="axis-line" x1="${m.l}" x2="${W-m.r}" y1="${H-m.b}" y2="${H-m.b}"/>`;
  s+=`<text x="${(m.l+W-m.r)/2}" y="${H-4}" font-size="11" text-anchor="middle">${xl}</text><text x="12" y="${(m.t+H-m.b)/2}" font-size="11" text-anchor="middle" transform="rotate(-90 12 ${(m.t+H-m.b)/2})">${yl}</text>`;
  return {s,X,Y};
}
function ext(a, pad=0.05){ a=a.filter(v=>v!=null&&isFinite(v)); let lo=Math.min(...a), hi=Math.max(...a); if(hi-lo<1e-12){lo-=1;hi+=1;} const p=(hi-lo)*pad; return [lo-p,hi+p]; }

// ---------- header ----------
(function(){
  document.getElementById('h1').innerHTML = `${D.cohorts.all.n} subjects <span class="dim">atlas&#8209;${D.atlas}</span>`;
  document.getElementById('dek').innerHTML = `Fitted per-node parameters from TVB-optim joint EEG+BOLD fits of a heterogeneous Jansen&#8209;Rit / Wilson&#8209;Cowan network. Parameters analysed: <span class="mono">${PARAMS.join(', ')}</span>${D.frozen.length?` (dropped as never moving from init: <span class="mono">${D.frozen.join(', ')}</span>)`:''}, plus global coupling <span class="mono">G</span>. Each parameter is analysed only on the node type where it is live.`;
  const c = D.config;
  document.getElementById('chips').innerHTML = [['finished fits',D.n_finished],['excluded',D.excluded.length],['unfinished',D.n_incomplete],['epochs',c.num_epochs],['schedule',c.schedule],['leadfield',c.leadfield_label],['bold',c.bold_model],['age split',`< ${D.age_split} &le;`]]
    .map(([k,v])=>`<span class="chip">${k} <b>${v}</b></span>`).join('');
  document.getElementById('cohort-seg').innerHTML = Object.entries(D.cohorts).map(([k,v])=>`<button data-k="${k}" class="${k===cohort?'active':''}" title="${v.label}">${k==='all'?'all fits':'BOLD-fit subset'} (${v.n})</button>`).join('');
  seg('cohort-seg', k=>{ cohort=k; selRegion=null; renderAll(); });
  document.getElementById('param-pills').innerHTML = PARAMS.map(p=>`<button class="pill" data-param="${p}"><span class="dot"></span>${p}</button>`).join('');
  document.querySelectorAll('#param-pills .pill').forEach(el=>el.addEventListener('click',()=>{ param=el.dataset.param; selRegion=null; renderRegions(); }));
  const groups = D.group_order.filter(g=>REG.some(r=>r.group===g));
  document.getElementById('group-sel').innerHTML = `<option value="all">all networks</option>` + groups.map(g=>`<option>${g}</option>`).join('');
  document.getElementById('group-sel').addEventListener('change', e=>{ group=e.target.value; drawStrip(); });
  seg('brain-seg', k=>{ brainKind=k; renderBrain(); });
  seg('corr-seg', k=>{ corrKind=k; drawCorr(); });
  seg('test-seg', k=>{ testKind=k; drawVolcano(); });
  document.getElementById('volc-param').innerHTML = `<option value="all">all</option>` + PARAMS.map(p=>`<option>${p}</option>`).join('');
  document.getElementById('volc-param').addEventListener('change', e=>{ volcParam=e.target.value; drawVolcano(); });
  const pcs = [1,2,3,4,5].map(i=>`<option value="${i-1}">PC${i}</option>`).join('');
  document.getElementById('pcx').innerHTML = pcs; document.getElementById('pcy').innerHTML = pcs; document.getElementById('pcy').value = 1;
  document.getElementById('pc-color').innerHTML = [['age','age group'],['cluster','k-means cluster'],['failed','failed-fit flag'],['sex','sex'],['fc_corr','fc_corr'],['eeg','EEG-PSD loss (log10)'],['G','G']].map(([v,l])=>`<option value="${v}">${l}</option>`).join('');
  ['pcx','pcy','pc-color'].forEach(id=>document.getElementById(id).addEventListener('change', drawPCA));
})();

// ---------- 01 cohort ----------
function renderCohort(){
  const c=C(), mem=new Set(c.members);
  const fcs=c.members.map(i=>SUBJ[i].fc_corr).sort((a,b)=>a-b), eegs=c.members.map(i=>SUBJ[i].eeg).sort((a,b)=>a-b);
  const med=a=>a[Math.floor(a.length/2)];
  statTiles('cohort-stats', [['subjects', c.n, c.label], ['young / old', `${c.n_young} / ${c.n_old}`, `split at age ${D.age_split}`],
    ['median fc_corr', f(med(fcs),3), `threshold ${D.fc_threshold}`], ['median EEG-PSD loss', `10<sup>${f(med(eegs),2)}</sup>`, 'mean of last 10 epochs'],
    ['excluded', D.excluded.length, 'gamma_weight &ne; 0 or no age']]);
  const nf=c.members.filter(i=>SUBJ[i].failed).length, cb=document.getElementById('fail-callout');
  cb.style.display = nf ? 'block' : 'none';
  cb.innerHTML = `<b>${nf} of ${c.n} fits in this cohort look failed.</b> Their final EEG-PSD loss is above ${D.eeg_fail_loss} (good fits sit near 10<sup>-7</sup>; they are the far-right group in the plot below), and they usually also have G &asymp; 0 and no BOLD improvement. In this cohort they will dominate PC1 and the fit-quality covariates. Switch to the BOLD-fit subset to drop them.`;
  const W=900,H=340,m={l:56,r:16,t:12,b:38};
  const A=axes(W,H,m,ext(SUBJ.map(s=>s.eeg)),ext(SUBJ.map(s=>s.fc_corr)),'log10 EEG-PSD loss','fc_corr');
  let s=A.s+`<line x1="${m.l}" x2="${W-m.r}" y1="${A.Y(D.fc_threshold)}" y2="${A.Y(D.fc_threshold)}" stroke="var(--ink-muted)" stroke-dasharray="4,3"/>`;
  SUBJ.forEach((u,i)=>{ const inC=mem.has(i), col=u.old?'var(--old)':'var(--young)';
    s+=`<circle cx="${A.X(u.eeg)}" cy="${A.Y(u.fc_corr)}" r="4.5" fill="${inC?col:'none'}" stroke="${inC?'var(--surface)':col}" stroke-width="${inC?1.5:1.2}" opacity="${inC?0.9:0.6}"/>`;
    s+=`<circle cx="${A.X(u.eeg)}" cy="${A.Y(u.fc_corr)}" r="9" fill="transparent" data-tip="<b>${u.id}</b>age ${u.age} &middot; ${u.sex}<br>fc_corr ${f(u.fc_corr)}<br>EEG-PSD loss 10^${f(u.eeg,2)}<br>G ${f(u.G)}${u.failed?'<br><i>flagged: failed fit</i>':''}${inC?'':'<br><i>not in this cohort</i>'}"/>`; });
  const svg=document.getElementById('fq-svg'); svg.innerHTML=s; bindTips(svg);
  document.getElementById('fq-legend').innerHTML = `<span><span class="sw" style="background:var(--young)"></span>young</span><span><span class="sw" style="background:var(--old)"></span>old</span><span><span class="sw" style="border:1.5px solid var(--ink-muted)"></span>outside cohort</span>`;
}

// ---------- 02 regions ----------
function regionsFor(p){ const cols=new Set(C().cols[p]); return D.display_order.filter(r=>cols.has(r) && (group==='all'||REG[r].group===group)); }
function rstat(p, r){ return C().region_stats.find(x=>x.param===p && x.region_idx===r); }
function atest(p, r){ return C().age_region.find(x=>x.param===p && x.region_idx===r); }
function quant(a,q){ const s=a.slice().sort((x,y)=>x-y); const i=(s.length-1)*q, lo=Math.floor(i); return s[lo]+(s[Math.min(lo+1,s.length-1)]-s[lo])*(i-lo); }
function renderRegions(){
  document.querySelectorAll('#param-pills .pill').forEach(el=>el.classList.toggle('active', el.dataset.param===param));
  const r=C().reliability[param];
  const verdict = v => v>=0.8?'highly reproducible':(v>=0.5?'moderately reproducible':'poorly reproducible');
  statTiles('rel-stats', [['regions analysed', C().cols[param].length, `${D.live_on[param]} nodes, &ge;${Math.round(D.min_coverage*100)}% coverage`],
    ['split-half r', f(r.split_half_r,2), `group map: ${verdict(r.split_half_sb)} (SB ${f(r.split_half_sb,2)})`],
    ['subject&ndash;subject r', f(r.pairwise_subject_r,2), 'similarity of two individuals\' maps'],
    ['ICC(region)', f(r.icc_region,2), 'variance share of region identity'],
    ['init / bounds', `${D.init[param]}`, `[${D.bounds[param].join(', ')}] ${D.units[param]}`]]);
  drawStrip(); renderDetail(); renderBrain();
}
let stripPts=[], stripBands=null;
function drawStrip(){
  const cv=document.getElementById('strip'), dpr=devicePixelRatio||1, W=cv.clientWidth, H=380;
  cv.width=W*dpr; cv.height=H*dpr; cv.style.height=H+'px';
  const ctx=cv.getContext('2d'); ctx.scale(dpr,dpr); ctx.clearRect(0,0,W,H);
  const regs=regionsFor(param), vals=D.values[param], mem=C().members, m={l:52,r:8,t:10,b:group==='all'?40:110};
  const all=[]; mem.forEach(i=>regs.forEach(r=>{ const v=vals[i][r]; if(v!=null) all.push(v); }));
  const yr=ext(all.concat([D.init[param]]),0.04), iw=W-m.l-m.r, ih=H-m.t-m.b, bw=iw/Math.max(1,regs.length);
  const Y=v=>m.t+ih-(v-yr[0])/(yr[1]-yr[0])*ih;
  ctx.font='10px IBM Plex Sans, sans-serif'; ctx.fillStyle=css('--ink-2'); ctx.strokeStyle=css('--border'); ctx.lineWidth=1;
  for(let i=0;i<=4;i++){ const v=yr[0]+(yr[1]-yr[0])*i/4; ctx.beginPath(); ctx.moveTo(m.l,Y(v)); ctx.lineTo(W-m.r,Y(v)); ctx.stroke(); ctx.textAlign='right'; ctx.fillText(g3(v).replace('&ndash;','-'), m.l-5, Y(v)+3); }
  ctx.setLineDash([4,3]); ctx.strokeStyle=css('--ink-muted'); ctx.beginPath(); ctx.moveTo(m.l,Y(D.init[param])); ctx.lineTo(W-m.r,Y(D.init[param])); ctx.stroke(); ctx.setLineDash([]);
  let prev=null;
  regs.forEach((r,j)=>{ const x0=m.l+j*bw;
    if(selRegion===r){ ctx.fillStyle=css('--accent-soft'); ctx.fillRect(x0,m.t,bw,ih); }
    if(group==='all' && REG[r].group!==prev){ ctx.strokeStyle=css('--border-2'); ctx.beginPath(); ctx.moveTo(x0,m.t); ctx.lineTo(x0,m.t+ih+6); ctx.stroke();
      const n=regs.filter(q=>REG[q].group===REG[r].group).length; ctx.save(); ctx.fillStyle=css('--ink-2'); ctx.textAlign='left'; ctx.translate(x0+2,m.t+ih+12); ctx.rotate(Math.PI/6); if(n*bw>14) ctx.fillText(REG[r].group,0,0); ctx.restore(); prev=REG[r].group; }
    if(group!=='all'){ ctx.save(); ctx.fillStyle=css('--ink-2'); ctx.textAlign='right'; ctx.translate(x0+bw/2+3,m.t+ih+6); ctx.rotate(-Math.PI/3); ctx.fillText(short(REG[r].name).slice(0,26),0,0); ctx.restore(); }
  });
  stripPts=[]; const old=css('--old'), young=css('--young'), rad=group==='all'?1.6:2.6;
  mem.forEach(i=>{ const u=SUBJ[i], h=((i*2654435761)%1000)/1000-0.5; ctx.fillStyle=u.old?old:young; ctx.globalAlpha=0.45;
    regs.forEach((r,j)=>{ const v=vals[i][r]; if(v==null) return; const x=m.l+(j+0.5+h*0.6)*bw, y=Y(v); ctx.beginPath(); ctx.arc(x,y,rad,0,7); ctx.fill(); stripPts.push([x,y,i,r,v]); }); });
  ctx.globalAlpha=1; ctx.lineWidth=2;
  [[false,young],[true,old]].forEach(([isOld,col])=>{ ctx.strokeStyle=col; ctx.beginPath(); let started=false;
    regs.forEach((r,j)=>{ const xs=mem.filter(i=>SUBJ[i].old===isOld).map(i=>vals[i][r]).filter(v=>v!=null); if(!xs.length) return; const y=Y(quant(xs,0.5)), x=m.l+(j+0.5)*bw;
      if(!started){ctx.moveTo(x,y);started=true;} else ctx.lineTo(x,y); }); ctx.stroke(); });
  stripBands={m,bw,regs};
}
(function(){ const cv=document.getElementById('strip'); let raf=null;
  cv.addEventListener('mousemove', e=>{ if(raf) return; raf=requestAnimationFrame(()=>{ raf=null; const b=cv.getBoundingClientRect(), x=e.clientX-b.left, y=e.clientY-b.top;
    let best=null, bd=64; for(const p of stripPts){ const d=(p[0]-x)**2+(p[1]-y)**2; if(d<bd){bd=d;best=p;} }
    if(best){ const u=SUBJ[best[2]]; showTip(`<b>${short(REG[best[3]].name)}</b>${u.id} &middot; age ${u.age} ${u.sex}<br>${param} = ${g3(best[4])}`, e); }
    else if(stripBands){ const j=Math.floor((x-stripBands.m.l)/stripBands.bw); const r=stripBands.regs[j]; if(r!=null){ const t=atest(param,r); showTip(`<b>${short(REG[r].name)}</b>${REG[r].group} &middot; hemi ${REG[r].hemi}${t?`<br>Cliff's &delta; ${f(t.cliffs_delta,2)} (q ${pf(t.q_mw)})`:''}<br><i>click for details</i>`, e); } else hideTip(); }
  }); });
  cv.addEventListener('mouseleave', hideTip);
  cv.addEventListener('click', e=>{ if(!stripBands) return; const b=cv.getBoundingClientRect(); const j=Math.floor((e.clientX-b.left-stripBands.m.l)/stripBands.bw); const r=stripBands.regs[j]; if(r!=null){ selRegion=r; drawStrip(); renderDetail(); } });
  addEventListener('resize', ()=>{ drawStrip(); });
})();
function renderDetail(){
  const el=document.getElementById('region-detail');
  if(selRegion==null){ el.innerHTML='<h4>Region detail</h4><div class="note">Click a region column in the plot to compare young and old for it.</div>'; return; }
  const r=selRegion, s=rstat(param,r), t=atest(param,r), vals=D.values[param], mem=C().members;
  const ys=mem.filter(i=>!SUBJ[i].old).map(i=>vals[i][r]).filter(v=>v!=null), os=mem.filter(i=>SUBJ[i].old).map(i=>vals[i][r]).filter(v=>v!=null);
  const W=280,H=150,m={l:40,r:8,t:8,b:20}, yr=ext(ys.concat(os)), Y=v=>m.t+(H-m.t-m.b)-(v-yr[0])/(yr[1]-yr[0])*(H-m.t-m.b);
  let svg=''; [[ys,'var(--young)',90,'young'],[os,'var(--old)',190,'old']].forEach(([a,col,cx,lab])=>{ a.forEach((v,k)=>{ svg+=`<circle cx="${cx+(((k*7919)%100)/100-0.5)*50}" cy="${Y(v)}" r="2.6" fill="${col}" opacity="0.55"/>`; });
    if(a.length){ const q1=quant(a,.25),q2=quant(a,.5),q3=quant(a,.75); svg+=`<rect x="${cx-30}" y="${Y(q3)}" width="60" height="${Math.max(1,Y(q1)-Y(q3))}" fill="none" stroke="var(--ink-2)" stroke-width="1"/><line x1="${cx-30}" x2="${cx+30}" y1="${Y(q2)}" y2="${Y(q2)}" stroke="var(--ink)" stroke-width="2"/>`; }
    svg+=`<text x="${cx}" y="${H-4}" font-size="11" text-anchor="middle">${lab} (n=${a.length})</text>`; });
  for(let i=0;i<=3;i++){ const v=yr[0]+(yr[1]-yr[0])*i/3; svg+=`<text x="${m.l-5}" y="${Y(v)+3}" font-size="9.5" text-anchor="end">${g3(v)}</text>`; }
  el.innerHTML = `<h4>${short(REG[r].name)}</h4><div class="note">${REG[r].group} &middot; hemi ${REG[r].hemi} &middot; ${param}</div>
    <svg viewBox="0 0 ${W} ${H}" width="100%">${svg}</svg>
    <table class="t">${s?`<tr><td>median [IQR]</td><td class="num">${g3(s.median)} [${g3(s.q1)}, ${g3(s.q3)}]</td></tr><tr><td>CV across subjects</td><td class="num">${f(s.cv,2)}</td></tr><tr><td>at a bound</td><td class="num">${f(s.frac_at_bound*100,0)}%</td></tr>`:''}
    ${t?`<tr><td>Cliff's &delta; (old&minus;young)</td><td class="num">${f(t.cliffs_delta,2)}</td></tr><tr><td>p / q (MW)</td><td class="num">${pf(t.p_mw)} / ${pf(t.q_mw)}</td></tr><tr><td>&beta; old (OLS, z)</td><td class="num">${f(t.beta_old,2)}</td></tr><tr><td>p / q (OLS)</td><td class="num">${pf(t.p_ols)} / ${pf(t.q_ols)}</td></tr>`:''}</table>`;
}
function renderBrain(){
  const img=document.getElementById('brain-img'), key = brainKind==='median' ? `median|${param}` : `delta|${param}|${cohort}`;
  const src=D.brains[key]; img.style.display=src?'block':'none'; if(src) img.src=src;
  document.getElementById('brain-note').innerHTML = !Object.keys(D.brains).length ? 'Brain maps were not rendered (--no-brains).' :
    (brainKind==='median' ? `The median across all fits, drawn on sub-${D.ref_subject}'s source cloud. Gray points are regions not analysed for ${param}.` : `Cliff's &delta; for old vs young: red means higher in old, blue higher in young. This is unthresholded, so check q in the age section.`);
}

// ---------- 03 clustering ----------
function renderClusters(){
  const p=C().pca, ri=C().region_cluster_info, sb=p.silhouette[p.best_k];
  const v = sb>=0.5?'strong clusters':(sb>=0.25?'weak clusters':'no real clusters (a continuum)');
  statTiles('clu-stats', [['features', p.n_features, 'parameter &times; region'], ['PCs kept', p.n_pc, `${f(p.cum_var*100,0)}% var; ${p.n_pc_50} PCs for 50%`],
    ['best k (subjects)', p.best_k, `silhouette ${f(sb,2)}: ${v}`], ['cluster vs age', `ARI ${f(p.ari_age,2)}`, `&chi;&sup2; p = ${pf(p.chi2_p)}`],
    ['best k (regions)', ri.best_k, `silhouette ${f(ri.silhouette[ri.best_k],2)}, ARI vs networks ${f(ri.ari_group,2)}`]]);
  drawPCA(); drawBars('scree-svg', p.evr.map((v,i)=>[`${i+1}`, v*100, i<p.n_pc]), '% var');
  drawBars('sil-svg', Object.entries(p.silhouette).map(([k,v])=>[k, v, +k===p.best_k]), 'silhouette', 0.25);
  document.getElementById('pc-assoc').innerHTML = `<tr><th>PC</th><th>var</th><th>AUC old</th><th>p</th><th>AUC male</th><th>&rho; EEG loss</th><th>&rho; fc_corr</th><th>&rho; G</th></tr>` +
    p.assoc.map(a=>`<tr><td>PC${a.pc}</td><td class="num">${f(a.var_explained*100,1)}%</td><td class="num">${f(a.auc_old,2)}</td><td class="num">${pf(a.p_old)}</td><td class="num">${f(a.auc_male,2)}</td><td class="num">${f(a.rho_eeg_psd_loss,2)}</td><td class="num">${f(a.rho_fc_corr,2)}</td><td class="num">${f(a.rho_G,2)}</td></tr>`).join('');
  document.getElementById('loadings').innerHTML = Object.entries(p.loadings).map(([k,l])=>`<table class="t"><tr><th>${k}</th><th>loading</th></tr>${l.map(([n,v])=>`<tr><td>${n}</td><td class="num">${f(v,3)}</td></tr>`).join('')}</table>`).join('');
  drawCorr();
  const rc=C().region_clusters, ks=[...new Set(rc.map(r=>r.cluster))].sort(), gs=D.group_order.filter(g=>rc.some(r=>r.group===g));
  document.getElementById('rclu-note').innerHTML = `${rc.length} regions clustered on their median ${C().node_corr.labels.join(', ')}. Each cell counts the regions from that network in that cluster. Adjusted Rand index vs networks: <b>${f(ri.ari_group,2)}</b> (0 = unrelated, 1 = identical).`;
  document.getElementById('rclu-table').innerHTML = `<tr><th>network</th>${ks.map(k=>`<th>cluster ${k}</th>`).join('')}</tr>` +
    gs.map(g=>{ const row=ks.map(k=>rc.filter(r=>r.group===g&&r.cluster===k).length), tot=row.reduce((a,b)=>a+b,0);
      return `<tr><td>${g}</td>${row.map(n=>`<td class="num" style="background:${n?seqColor(n/tot*0.8):'transparent'};color:${n/tot>0.6?'var(--surface)':'inherit'}">${n||''}</td>`).join('')}</tr>`; }).join('');
}
function drawBars(id, rows, yl, ref){
  const W=300,H=150,m={l:34,r:6,t:8,b:22}, ymax=Math.max(...rows.map(r=>r[1]),ref||0)*1.1, ymin=Math.min(0,...rows.map(r=>r[1]));
  const bw=(W-m.l-m.r)/rows.length, Y=v=>m.t+(H-m.t-m.b)*(1-(v-ymin)/(ymax-ymin));
  let s=`<line class="axis-line" x1="${m.l}" x2="${W-m.r}" y1="${Y(0)}" y2="${Y(0)}"/><text x="${m.l-4}" y="${Y(ymax/1.1)+3}" font-size="9" text-anchor="end">${g3(ymax/1.1)}</text>`;
  if(ref!=null) s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(ref)}" y2="${Y(ref)}" stroke="var(--ink-muted)" stroke-dasharray="3,3"/>`;
  rows.forEach(([l,v,on],i)=>{ const x=m.l+i*bw+1, y0=Y(Math.max(v,0)), h=Math.max(1,Math.abs(Y(v)-Y(0)));
    s+=`<rect x="${x}" y="${y0}" width="${bw-2}" height="${h}" rx="2" fill="${on?'var(--seq-500)':'var(--seq-250)'}" data-tip="<b>${l}</b>${yl} ${g3(v)}"/>`;
    if(rows.length<=12 || i%2===0) s+=`<text x="${x+bw/2-1}" y="${H-8}" font-size="9" text-anchor="middle">${l}</text>`; });
  const el=document.getElementById(id); el.innerHTML=s; bindTips(el);
}
function drawPCA(){
  const p=C().pca, mem=C().members, xi=+document.getElementById('pcx').value, yi=+document.getElementById('pcy').value, by=document.getElementById('pc-color').value;
  const W=640,H=420,m={l:48,r:12,t:10,b:36}, xs=p.scores.map(s=>s[xi]), ys=p.scores.map(s=>s[yi]);
  const A=axes(W,H,m,ext(xs),ext(ys),`PC${xi+1} (${f(p.evr[xi]*100,1)}%)`,`PC${yi+1} (${f(p.evr[yi]*100,1)}%)`);
  const cont = ['fc_corr','eeg','G'].includes(by); let lo=0, hi=1;
  if(cont){ const v=mem.map(i=>SUBJ[i][by]); lo=Math.min(...v); hi=Math.max(...v); }
  const catCol = ['var(--young)','var(--old)','#1baf7a'], shapes=['circle','square','triangle','diamond','cross','star'];
  function mark(x,y,shape,fill){ const r=5; if(shape==='circle') return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="var(--surface)" stroke-width="1.5"/>`;
    if(shape==='square') return `<rect x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}" fill="${fill}" stroke="var(--surface)" stroke-width="1.5"/>`;
    if(shape==='triangle') return `<path d="M${x},${y-r-1} L${x+r+1},${y+r} L${x-r-1},${y+r} Z" fill="${fill}" stroke="var(--surface)" stroke-width="1.5"/>`;
    if(shape==='diamond') return `<path d="M${x},${y-r-1} L${x+r+1},${y} L${x},${y+r+1} L${x-r-1},${y} Z" fill="${fill}" stroke="var(--surface)" stroke-width="1.5"/>`;
    return `<path d="M${x-r},${y-r} L${x+r},${y+r} M${x+r},${y-r} L${x-r},${y+r}" stroke="${fill}" stroke-width="2.5"/>`; }
  let s=A.s;
  mem.forEach((i,k)=>{ const u=SUBJ[i], cl=p.clusters[k]; let fill, shape='circle';
    if(by==='age') fill=u.old?'var(--old)':'var(--young)';
    else if(by==='sex') fill=u.sex==='M'?'var(--young)':'var(--old)';
    else if(by==='failed') fill=u.failed?'var(--div-pos)':'var(--muted-pt)';
    else if(by==='cluster'){ fill=cl<=3?catCol[cl-1]:'var(--ink-2)'; shape=shapes[(cl-1)%shapes.length]; }
    else fill=seqColor((u[by]-lo)/((hi-lo)||1));
    s+=mark(A.X(xs[k]),A.Y(ys[k]),shape,fill);
    s+=`<circle cx="${A.X(xs[k])}" cy="${A.Y(ys[k])}" r="9" fill="transparent" data-tip="<b>${u.id}</b>age ${u.age} &middot; ${u.sex} &middot; cluster ${cl}<br>fc_corr ${f(u.fc_corr)} &middot; EEG loss 10^${f(u.eeg,2)}<br>G ${f(u.G)}"/>`; });
  const el=document.getElementById('pca-svg'); el.innerHTML=s; bindTips(el);
  const L=document.getElementById('pc-legend');
  if(by==='age') L.innerHTML=`<span><span class="sw" style="background:var(--young)"></span>young</span><span><span class="sw" style="background:var(--old)"></span>old</span>`;
  else if(by==='sex') L.innerHTML=`<span><span class="sw" style="background:var(--young)"></span>M</span><span><span class="sw" style="background:var(--old)"></span>F</span>`;
  else if(by==='failed') L.innerHTML=`<span><span class="sw" style="background:var(--div-pos)"></span>failed fit</span><span><span class="sw" style="background:var(--muted-pt)"></span>ok</span>`;
  else if(by==='cluster') L.innerHTML=[...new Set(p.clusters)].sort().map(c=>`<span><svg width="12" height="12" style="vertical-align:-1px;margin-right:5px">${mark(6,6,shapes[(c-1)%shapes.length],c<=3?catCol[c-1]:'var(--ink-2)').replace(/r="5"/,'r="4"')}</svg>cluster ${c} (n=${p.clusters.filter(x=>x===c).length})</span>`).join('');
  else L.innerHTML=`<span>${g3(lo)}</span><span style="width:120px;height:8px;border-radius:4px;background:linear-gradient(90deg,var(--seq-100),var(--seq-250),var(--seq-400),var(--seq-500),var(--seq-650))"></span><span>${g3(hi)}</span>`;
}
function drawCorr(){
  const c=C()[corrKind], n=c.labels.length, W=460,H=420, m={l:110,t:10,r:10,b:100}, cw=(W-m.l-m.r)/n, ch=(H-m.t-m.b)/n;
  let s='';
  c.matrix.forEach((row,i)=>row.forEach((v,j)=>{ const x=m.l+j*cw, y=m.t+i*ch;
    s+=`<rect x="${x+1}" y="${y+1}" width="${cw-2}" height="${ch-2}" rx="2" fill="${i===j?'var(--surface-2)':divColor(v)}" data-tip="<b>${c.labels[i]} &times; ${c.labels[j]}</b>&rho; = ${f(v,2)}"/>`;
    if(i!==j && cw>28) s+=`<text x="${x+cw/2}" y="${y+ch/2+3.5}" font-size="9.5" text-anchor="middle" style="fill:${Math.abs(v)>0.6?'var(--surface)':'var(--ink-2)'}">${f(v,2)}</text>`; }));
  c.labels.forEach((l,i)=>{ s+=`<text x="${m.l-6}" y="${m.t+i*ch+ch/2+3.5}" font-size="10.5" text-anchor="end">${l}</text>`;
    s+=`<text x="${m.l+i*cw+cw/2}" y="${H-m.b+10}" font-size="10.5" text-anchor="end" transform="rotate(-50 ${m.l+i*cw+cw/2} ${H-m.b+10})">${l}</text>`; });
  const el=document.getElementById('corr-svg'); el.innerHTML=s; bindTips(el);
  document.getElementById('corr-note').innerHTML = corrKind==='node_corr' ? `Spearman &rho; across ${C().node_corr.n_cells.toLocaleString()} (subject, region) cells after removing each region's median. A strong &rho; means two parameters trade off, so the fit cannot tell them apart. Blue is negative, red positive.` :
    'Spearman &rho; across subjects between region-averaged parameters, G, age and fit quality. Blue is negative, red positive.';
}

// ---------- 04 age ----------
function renderAge(){
  const c=C(), clf=c.classifier;
  document.getElementById('age-lede').innerHTML = `Young (age &lt; ${D.age_split}, n=${c.n_young}) vs old (n=${c.n_old}). Per region and per network, a Mann&ndash;Whitney test gives Cliff's &delta;, and an OLS model of the z-scored parameter adds sex, log EEG-PSD loss and fc_corr as covariates. Both use Benjamini&ndash;Hochberg FDR. The multivariate test is an L2 logistic regression (C=${D.clf_c}) scored by 5&times;10 cross-validated AUC, with ${D.n_perm} label permutations for p.`;
  statTiles('clf-stats', Object.entries(clf).map(([k,v])=>[`AUC &middot; ${k}`, f(v.auc_mean,3), `&plusmn;${f(v.auc_sd,3)} &middot; perm p ${pf(v.p_perm)}`])
    .concat([[`region tests q&lt;0.05`, `${c.age_region.filter(r=>r.q_mw<0.05).length} / ${c.age_region.length}`, `MW; OLS: ${c.age_region.filter(r=>r.q_ols<0.05).length}`]]));
  const keys=Object.keys(clf), W=960,H=200, pw=W/keys.length; let s='';
  keys.forEach((k,ki)=>{ const v=clf[k], nb=24, lo=Math.min(0.2,...v.null), hi=Math.max(1,...v.null), bins=new Array(nb).fill(0);
    v.null.forEach(x=>bins[Math.min(nb-1,Math.floor((x-lo)/(hi-lo)*nb))]++); const mx=Math.max(...bins), x0=ki*pw+30, iw=pw-50, ih=140, X=x=>x0+(x-lo)/(hi-lo)*iw;
    bins.forEach((b,i)=>{ s+=`<rect x="${x0+i*iw/nb+1}" y="${20+ih-b/mx*ih}" width="${iw/nb-2}" height="${b/mx*ih}" rx="1.5" fill="var(--muted-pt)"/>`; });
    s+=`<line x1="${X(v.auc_perm_cv)}" x2="${X(v.auc_perm_cv)}" y1="14" y2="${20+ih}" stroke="var(--sig)" stroke-width="2.5"/><text x="${X(v.auc_perm_cv)+4}" y="24" font-size="10.5">AUC ${f(v.auc_perm_cv,2)}</text>`;
    s+=`<line class="axis-line" x1="${x0}" x2="${x0+iw}" y1="${20+ih}" y2="${20+ih}"/>`; [lo,0.5,hi].forEach(t=>{ s+=`<text x="${X(t)}" y="${20+ih+14}" font-size="9.5" text-anchor="middle">${f(t,2)}</text>`; });
    s+=`<text x="${x0+iw/2}" y="${H-4}" font-size="11.5" text-anchor="middle" style="font-weight:600">${k} &middot; p = ${pf(v.p_perm)}</text>`; });
  document.getElementById('null-svg').innerHTML=s;
  document.getElementById('coef-table').innerHTML = `<tr><th>feature</th><th>coef (&gt;0 = older)</th></tr>` + c.top_coefs.map(([n,v])=>`<tr><td>${n}</td><td class="num">${f(v,3)}</td></tr>`).join('');
  document.getElementById('net-table').innerHTML = `<tr><th>param</th><th>network</th><th>&delta;</th><th>q MW</th><th>&beta;</th><th>q OLS</th></tr>` +
    c.age_group.slice().sort((a,b)=>a.p_mw-b.p_mw).map(r=>`<tr class="${r.q_mw<0.05||r.q_ols<0.05?'hl':''}"><td>${r.param}</td><td>${r.group}</td><td class="num" style="color:${r.cliffs_delta>0?'var(--div-pos)':'var(--div-neg)'}">${f(r.cliffs_delta,2)}</td><td class="num">${pf(r.q_mw)}</td><td class="num">${f(r.beta_old,2)}</td><td class="num">${pf(r.q_ols)}</td></tr>`).join('');
  document.getElementById('hits-table').innerHTML = `<tr><th>param</th><th>region</th><th>network</th><th>median young</th><th>median old</th><th>&delta;</th><th>p MW</th><th>q MW</th><th>&beta; OLS</th><th>q OLS</th></tr>` +
    c.age_region.slice().sort((a,b)=>a.p_mw-b.p_mw).slice(0,30).map(r=>`<tr class="${r.q_mw<0.05?'hl':''}"><td>${r.param}</td><td>${short(r.region)}</td><td>${r.group}</td><td class="num">${g3(r.median_young)}</td><td class="num">${g3(r.median_old)}</td><td class="num">${f(r.cliffs_delta,2)}</td><td class="num">${pf(r.p_mw)}</td><td class="num">${pf(r.q_mw)}</td><td class="num">${f(r.beta_old,2)}</td><td class="num">${pf(r.q_ols)}</td></tr>`).join('');
  drawVolcano();
}
function drawVolcano(){
  const rows=C().age_region.filter(r=>volcParam==='all'||r.param===volcParam);
  const xk=testKind==='mw'?'cliffs_delta':'beta_old', pk=testKind==='mw'?'p_mw':'p_ols', qk=testKind==='mw'?'q_mw':'q_ols';
  const W=640,H=400,m={l:48,r:12,t:10,b:38}, xs=rows.map(r=>r[xk]), ys=rows.map(r=>-Math.log10(Math.max(r[pk],1e-300)));
  const xm=Math.max(...xs.map(Math.abs),0.1)*1.08;
  const A=axes(W,H,m,[-xm,xm],[0,Math.max(...ys,2)*1.08], testKind==='mw'?"Cliff's δ (old − young)":'β old (z units, adjusted)','−log10 p');
  let s=A.s+`<line x1="${A.X(0)}" x2="${A.X(0)}" y1="${m.t}" y2="${H-m.b}" stroke="var(--border-2)"/>`;
  rows.forEach((r,i)=>{ const sig=r[qk]<0.05, x=A.X(xs[i]), y=A.Y(ys[i]);
    s+=`<circle cx="${x}" cy="${y}" r="${sig?4.5:3.2}" fill="${sig?'var(--sig)':'var(--muted-pt)'}" stroke="var(--surface)" stroke-width="1" opacity="${sig?0.95:0.7}"/>`;
    s+=`<circle class="vh" data-p="${r.param}" data-r="${r.region_idx}" cx="${x}" cy="${y}" r="8" fill="transparent" style="cursor:pointer" data-tip="<b>${r.param} &middot; ${short(r.region)}</b>${r.group}<br>${testKind==='mw'?'&delta;':'&beta;'} ${f(xs[i],2)} &middot; p ${pf(r[pk])} &middot; q ${pf(r[qk])}<br>median young ${g3(r.median_young)} / old ${g3(r.median_old)}<br><i>click to open in the regional view</i>"/>`; });
  const el=document.getElementById('volc-svg'); el.innerHTML=s; bindTips(el);
  el.querySelectorAll('.vh').forEach(c=>c.addEventListener('click',()=>{ param=c.dataset.p; selRegion=+c.dataset.r; group=REG[selRegion].group; document.getElementById('group-sel').value=group; hideTip(); renderRegions(); document.getElementById('regions').scrollIntoView({behavior:'smooth'}); }));
}

// ---------- 05 condition ----------
let condVar=null, condBrainParam=PARAMS.includes('B')?'B':PARAMS[0];
(function(){
  if(!D.cond_labels) return;
  const [A,B]=D.cond_labels, first=Object.values(D.cohorts).find(c=>c.condition).condition;
  ['condition','nav-cond'].forEach(id=>document.getElementById(id).style.display='');
  document.getElementById('methods-kicker').innerHTML='06 &middot; Methods &amp; caveats';
  document.getElementById('cond-h2').textContent=`Are the parameters stable between ${A} and ${B}, and what shifts?`;
  document.getElementById('cond-lede').innerHTML=`The same subjects were also fit under a second condition (<span class="mono">${D.compare_root}</span>). Only subjects with a finished fit in both are paired here, and each cohort's filters must hold in both conditions. The BOLD target is the same resting fMRI in both runs, so any shift comes from the EEG condition, possibly through how well each condition could be fit.`;
  const vs=Object.keys(first.vars); condVar=vs.includes('mean B')?'mean B':vs[0];
  document.getElementById('cond-var').innerHTML=vs.map(v=>`<option${v===condVar?' selected':''}>${v}</option>`).join('');
  document.getElementById('cond-var').addEventListener('change', e=>{ condVar=e.target.value; drawCondScatter(); });
  document.getElementById('cond-brain-param').innerHTML=PARAMS.map(p=>`<option${p===condBrainParam?' selected':''}>${p}</option>`).join('');
  document.getElementById('cond-brain-param').addEventListener('change', e=>{ condBrainParam=e.target.value; renderCondBrain(); });
})();
function renderCondition(){
  if(!D.cond_labels) return;
  const cd=C().condition, [A,B]=D.cond_labels;
  document.getElementById('cond-body').style.display = cd ? '' : 'none';
  const miss=document.getElementById('cond-missing'); miss.style.display = cd ? 'none' : 'block';
  if(!cd){ miss.innerHTML='Too few subjects in this cohort have a fit in both conditions.'; return; }
  const st=Object.fromEntries(cd.stability.map(r=>[r.param,r])), ia=cd.ident_all, clf=cd.classifier;
  statTiles('cond-stats', [['paired subjects', cd.n, `${cd.n_young} young / ${cd.n_old} old`],
    ['identification', `${f(ia.accuracy*100,0)}%`, `all params; chance ${f(ia.chance*100,1)}%, p ${pf(ia.p_binom)}`],
    ['I_self &minus; I_others', f(ia.i_diff,3), `${f(ia.i_self,3)} vs ${f(ia.i_others,3)}`],
    ['region shifts q&lt;0.05', `${cd.diff_region.filter(r=>r.q_w<0.05).length} / ${cd.diff_region.length}`, `Wilcoxon; adjusted for fit quality: ${cd.diff_region.filter(r=>r.q_ols<0.05).length}`],
    [`AUC &middot; ${A} vs ${B}`, f(clf.params.auc_mean,3), `params; fit quality alone ${f(clf.fit_quality.auc_mean,3)}`]]);
  const cell=(v,d=2)=>v==null?'<td class="num">&ndash;</td>':`<td class="num" style="background:${v>0?seqColor(v*0.8):'transparent'};color:${v>0.6?'var(--surface)':'inherit'}">${f(v,d)}</td>`;
  const pc=v=>v==null?'&ndash;':`${f(v*100,0)}%`;
  document.getElementById('cond-stab').innerHTML=`<tr><th>param</th><th>regions</th><th>region ICC(C)</th><th>region ICC(A)</th><th>regions ICC(C)&gt;0.5</th><th>whole-brain ICC(C)</th><th>whole-brain ICC(A)</th><th>group-map r</th><th>I_self</th><th>I_others</th><th>ident.</th><th>p</th></tr>`+
    [...PARAMS,'all params','G'].filter(p=>st[p]).map(p=>{ const r=st[p]; return `<tr><td>${p}</td><td class="num">${r.n_regions??'&ndash;'}</td>${cell(r.region_icc_c)}${cell(r.region_icc_a)}<td class="num">${pc(r['frac_region_icc_c_gt_0.5'])}</td>${cell(r.global_icc_c)}${cell(r.global_icc_a)}${cell(r.group_map_r)}<td class="num">${f(r.i_self,3)}</td><td class="num">${f(r.i_others,3)}</td><td class="num">${pc(r.accuracy)}</td><td class="num">${pf(r.p_binom)}</td></tr>`; }).join('');
  document.getElementById('cond-ident-note').innerHTML=`Left: similarity of every subject's ${A} deviation map (rows) to every ${B} map (columns), sorted by age. A bright diagonal means individual profiles survive the change of condition. Right: same-subject vs different-subject similarities. Top-1 accuracy ${f(ia.accuracy*100,0)}% vs chance ${f(ia.chance*100,1)}%.`;
  const fig=k=>D.cond_figs[`${k}|${cohort}`];
  document.getElementById('cond-ident-img').src=fig('ident')||''; document.getElementById('cond-diff-img').src=fig('diff')||'';
  document.getElementById('cond-diff-note').innerHTML=`Matched-pairs rank-biserial r per region (${B} &minus; ${A}; &gt; 0 = higher in ${B}, &plusmn;1 = every subject shifts the same way). Colored bars pass BH q &lt; 0.05 (Wilcoxon signed-rank).`;
  document.getElementById('cond-clf-note').innerHTML=`Paired Wilcoxon on the fit-quality metrics, then an L2 logistic regression that predicts the condition from within-subject-centred features (each subject's mean over both conditions removed). CV folds and label permutations are grouped by subject. If <b>fit_quality</b> separates the conditions as well as <b>params</b>, the parameter shift may only reflect a difference in fit.`;
  document.getElementById('cond-fit').innerHTML=`<tr><th>fit metric</th><th>${A}</th><th>${B}</th><th>r<sub>rb</sub></th><th>p</th></tr>`+cd.fit_diff.map(r=>`<tr><td>${r.variable}</td><td class="num">${g3(r.median_a)}</td><td class="num">${g3(r.median_b)}</td><td class="num">${f(r.rank_biserial,2)}</td><td class="num">${pf(r.p_w)}</td></tr>`).join('');
  document.getElementById('cond-clf').innerHTML=`<tr><th>classifier</th><th>CV AUC (5&times;10)</th><th>perm p</th></tr>`+Object.entries(clf).map(([k,v])=>`<tr><td>${k}</td><td class="num">${f(v.auc_mean,3)} &plusmn; ${f(v.auc_sd,3)}</td><td class="num">${pf(v.p_perm)}</td></tr>`).join('');
  document.getElementById('cond-net-note').innerHTML=`Parameter averaged over each network's regions per subject. &beta;<sub>adj</sub> is the standardized shift after regressing out the change in EEG-PSD loss and fc_corr. Age &delta; is Cliff's &delta; of the shift, old vs young: does the condition effect depend on age?`;
  const col=v=>`style="color:${v>0?'var(--div-pos)':'var(--div-neg)'}"`;
  document.getElementById('cond-net').innerHTML=`<tr><th>param</th><th>network</th><th>r<sub>rb</sub></th><th>q W</th><th>&beta;<sub>adj</sub></th><th>q OLS</th><th>age &delta;</th><th>p age</th></tr>`+
    cd.diff_group.slice().sort((a,b)=>a.p_w-b.p_w).map(r=>`<tr class="${r.q_w<0.05||r.q_ols<0.05?'hl':''}"><td>${r.param}</td><td>${r.group}</td><td class="num" ${col(r.rank_biserial)}>${f(r.rank_biserial,2)}</td><td class="num">${pf(r.q_w)}</td><td class="num">${f(r.dz_adj,2)}</td><td class="num">${pf(r.q_ols)}</td><td class="num">${f(r.age_delta,2)}</td><td class="num">${pf(r.p_age)}</td></tr>`).join('');
  document.getElementById('cond-hits').innerHTML=`<tr><th>param</th><th>region</th><th>network</th><th>median ${A}</th><th>median ${B}</th><th>median diff</th><th>r<sub>rb</sub></th><th>p W</th><th>q W</th><th>&beta;<sub>adj</sub></th><th>q OLS</th></tr>`+
    cd.diff_region.slice().sort((a,b)=>a.p_w-b.p_w).slice(0,30).map(r=>`<tr class="${r.q_w<0.05?'hl':''}"><td>${r.param}</td><td>${short(r.region)}</td><td>${r.group}</td><td class="num">${g3(r.median_a)}</td><td class="num">${g3(r.median_b)}</td><td class="num">${g3(r.median_diff)}</td><td class="num" ${col(r.rank_biserial)}>${f(r.rank_biserial,2)}</td><td class="num">${pf(r.p_w)}</td><td class="num">${pf(r.q_w)}</td><td class="num">${f(r.dz_adj,2)}</td><td class="num">${pf(r.q_ols)}</td></tr>`).join('');
  drawCondScatter(); renderCondBrain(); renderAlpha();
}
function renderAlpha(){
  const cd=C().condition, al=cd&&cd.alpha, blk=document.getElementById('alpha-block');
  blk.style.display = al ? '' : 'none'; if(!al) return;
  const [A,B]=D.cond_labels, mf=al.model_fit, fig=k=>D.cond_figs[`${k}|${cohort}`]||'';
  const roi=(src,r)=>al.roi_tests.find(x=>x.source===src&&x.roi===r)||{};
  document.getElementById('alpha-lede').innerHTML=`Hypothesis: going from ${A} to ${B} mainly changes alpha (${al.band[0]}&ndash;${al.band[1]} Hz) over visual cortex. Relative alpha is the alpha share of each channel's fitted band; absolute power is lost because the EEG chunks are std-normalized. <b>Reactivity</b> = log rel-alpha ${A} &minus; ${B} (&gt; 0 = less alpha in ${B}). Occipital ROI: <span class="mono">${al.roi.join(' ')}</span>; frontal reference: <span class="mono">${al.ref.join(' ')}</span>.`;
  const t=(src,r)=>{ const x=roi(src,r); return [`r<sub>rb</sub> ${f(x.rank_biserial,2)}`, `p ${pf(x.p_w)} &middot; ${f((x.frac_reactive||0)*100,0)}% drop`]; };
  statTiles('alpha-stats', [['empirical occipital', ...t('emp','occipital')], ['empirical occ &minus; front', ...t('emp','occipital - frontal')],
    ['simulated occipital', ...t('sim','occipital')], ['simulated occ &minus; front', ...t('sim','occipital - frontal')],
    ['model vs data', `&rho; ${f(mf.react_rho[0],2)}`, `per-subject reactivity (p ${pf(mf.react_rho[1])}); topography &rho; ${f(mf.topo_rho,2)}`]]);
  document.getElementById('alpha-roi-note').innerHTML=`Paired Wilcoxon on log relative alpha, ${B} vs ${A} (r<sub>rb</sub> &lt; 0 = less alpha in ${B}). <b>occipital &minus; frontal</b> tests specificity: does alpha drop more over the occipital ROI than frontally?`;
  document.getElementById('alpha-roi').innerHTML=`<tr><th>source</th><th>ROI</th><th>median ${A}</th><th>median ${B}</th><th>r<sub>rb</sub></th><th>p</th><th>subjects with drop</th></tr>`+
    al.roi_tests.map(r=>`<tr class="${r.p_w<0.05?'hl':''}"><td>${r.source==='emp'?'empirical':'simulated'}</td><td>${r.roi}</td><td class="num">${f(r.median_a,3)}</td><td class="num">${f(r.median_b,3)}</td><td class="num" style="color:${r.rank_biserial>0?'var(--div-pos)':'var(--div-neg)'}">${f(r.rank_biserial,2)}</td><td class="num">${pf(r.p_w)}</td><td class="num">${f(r.frac_reactive*100,0)}%</td></tr>`).join('');
  document.getElementById('alpha-spec-img').src=fig('alpha_spectra'); document.getElementById('alpha-topo-img').src=fig('alpha_topo'); document.getElementById('alpha-par-img').src=fig('alpha_params');
  document.getElementById('alpha-par-note').innerHTML=`For each parameter, each subject's ${B} &minus; ${A} shift is averaged over visual (Vis*) regions and over the other cortical regions. <b>visual &minus; other</b> tests whether the shift is specific to visual cortex. The &rho; columns ask whether subjects whose visual parameters shift more also lose more occipital alpha, empirically or in the simulation. q is BH-corrected over parameters. G shift vs empirical reactivity: &rho; ${f(al.G_link&&al.G_link.rho,2)} (p ${pf(al.G_link&&al.G_link.p)}).`;
  document.getElementById('alpha-par').innerHTML=`<tr><th>param</th><th>r<sub>rb</sub> visual</th><th>r<sub>rb</sub> other</th><th>r<sub>rb</sub> visual&minus;other</th><th>q</th><th>&rho; emp reactivity</th><th>q</th><th>&rho; sim reactivity</th><th>q</th></tr>`+
    al.param_tests.map(r=>`<tr class="${r.q_specific<0.05||r.q_react_emp<0.05?'hl':''}"><td>${r.param}</td><td class="num">${f(r.rb_visual,2)}</td><td class="num">${f(r.rb_other,2)}</td><td class="num">${f(r.rb_specific,2)}</td><td class="num">${pf(r.q_specific)}</td><td class="num">${f(r.rho_react_emp,2)}</td><td class="num">${pf(r.q_react_emp)}</td><td class="num">${f(r.rho_react_sim,2)}</td><td class="num">${pf(r.q_react_sim)}</td></tr>`).join('');
}
function drawCondScatter(){
  const cd=C().condition; if(!cd) return;
  const [A,B]=D.cond_labels, xy=cd.vars[condVar], s0=cd.var_stats[condVar];
  const W=520,H=440,m={l:56,r:12,t:10,b:38}, r=ext(xy.flat());
  const Ax=axes(W,H,m,r,r,`${condVar} &middot; ${A}`,`${condVar} &middot; ${B}`);
  let s=Ax.s+`<line x1="${Ax.X(r[0])}" y1="${Ax.Y(r[0])}" x2="${Ax.X(r[1])}" y2="${Ax.Y(r[1])}" stroke="var(--ink-muted)" stroke-dasharray="4,3"/>`;
  cd.pairs.forEach((u,k)=>{ const [a,b]=xy[k]; if(a==null||b==null) return; const x=Ax.X(a), y=Ax.Y(b);
    s+=`<circle cx="${x}" cy="${y}" r="4.5" fill="${u.old?'var(--old)':'var(--young)'}" stroke="var(--surface)" stroke-width="1.5" opacity="0.9"/>`;
    s+=`<circle cx="${x}" cy="${y}" r="9" fill="transparent" data-tip="<b>${u.id}</b>age ${u.age} &middot; ${u.sex}<br>${A} ${g3(a)} &middot; ${B} ${g3(b)}<br>shift ${g3(b-a)}"/>`; });
  const el=document.getElementById('cond-svg'); el.innerHTML=s; bindTips(el);
  document.getElementById('cond-scatter-note').innerHTML=`Each dot is a subject. Points on the dashed line kept their value; a cloud parallel to it means a shared shift. ICC(C) ${f(s0.icc_c,2)} &middot; ICC(A) ${f(s0.icc_a,2)} &middot; &rho; ${f(s0.rho,2)} &middot; shift r<sub>rb</sub> ${f(s0.rank_biserial,2)} (p ${pf(s0.p_w)}).`;
}
function renderCondBrain(){
  const src=D.brains[`cond|${condBrainParam}|${cohort}`], img=document.getElementById('cond-brain-img');
  img.style.display=src?'block':'none'; if(src) img.src=src;
  const [A,B]=D.cond_labels;
  document.getElementById('cond-brain-note').innerHTML = src ? `Rank-biserial r of ${B} &minus; ${A}, drawn on sub-${D.ref_subject}'s source cloud. Red means higher in ${B}. This is unthresholded.` : 'Brain maps were not rendered (--no-brains).';
}

// ---------- refit stability ----------
(function(){
  const R=D.rep; if(!R) return;
  ['replicates','nav-rep'].forEach(id=>document.getElementById(id).style.display='');
  const n=5+(D.cond_labels?1:0);
  document.getElementById('rep-kicker').innerHTML=`0${n} &middot; Refit stability`;
  document.getElementById('methods-kicker').innerHTML=`0${n+1} &middot; Methods &amp; caveats`;
  document.getElementById('rep-lede').innerHTML=`${R.labels.length} refits of the same ${R.n} subjects: <span class="mono">${R.labels.join(', ')}</span>. Init varies at fixed noise in ${R.sub_init.length} of them and noise varies at default init in ${R.sub_noise.length}. This section is independent of the cohort toggle.`;
  const fig=k=>D.cond_figs[k]||'', fm=(v,d=2)=>f(v,d), pc=v=>v==null?'&ndash;':`${f(v*100,0)}%`;
  const cell=(v,d=2)=>v==null?'<td class="num">&ndash;</td>':`<td class="num" style="background:${v>0?seqColor(v*0.8):'transparent'};color:${v>0.6?'var(--surface)':'inherit'}">${f(v,d)}</td>`;
  const all=R.ident_summary.filter(r=>r.param==='all params'), vv=Object.fromEntries(R.variance.map(r=>[r.param,r]));
  const icc=R.variance.filter(r=>r.region_icc_all!=null).map(r=>r.region_icc_all).sort((a,b)=>a-b);
  statTiles('rep-stats', [['subjects &times; refits', `${R.n} &times; ${R.labels.length}`, 'fit in every refit'],
    ['median region ICC', f(icc[Math.floor(icc.length/2)],2), 'over params (subject vs refit variance)'],
    ...all.map(r=>[`identification &middot; ${r.kind}`, pc(r.accuracy), `all params; chance ${f(100/R.n,1)}%`]),
    ...(R.ident_ref?[[`${R.ident_ref.labels.join(' vs ')}`, pc(R.ident_ref.accuracy), 'same subjects, all params']]:[])]);
  document.getElementById('rep-conv').innerHTML=`<tr><th>refit</th><th>median log EEG loss</th><th>median fc_corr</th><th>median G</th><th>worse than best</th><th>best fit</th></tr>`+
    R.convergence.map(r=>`<tr><td class="mono">${r.replicate}</td><td class="num">${fm(r.median_log_eeg_loss,3)}</td><td class="num">${fm(r.median_fc_corr,3)}</td><td class="num">${fm(r.median_G,3)}</td><td class="num">${pc(r.frac_worse_than_best)}</td><td class="num">${pc(r.frac_best)}</td></tr>`).join('');
  document.getElementById('rep-var').innerHTML=`<tr><th>param</th><th>region ICC all</th><th>init</th><th>noise</th><th>regions ICC&gt;0.5</th><th>whole-brain ICC all</th><th>init</th><th>noise</th><th>SD within</th><th>SD between</th></tr>`+
    R.variance.map(r=>`<tr><td>${r.param}</td>${cell(r.region_icc_all)}${cell(r.region_icc_init)}${cell(r.region_icc_noise)}<td class="num">${pc(r['frac_region_icc_gt_0.5'])}</td>${cell(r.global_icc_all)}${cell(r.global_icc_init)}${cell(r.global_icc_noise)}<td class="num">${fm(r.global_sd_within_frac,3)}</td><td class="num">${fm(r.global_sd_between_frac,3)}</td></tr>`).join('');
  document.getElementById('rep-ident-note').innerHTML=`Same test as the condition section, between pairs of refits, averaged by what the pair differs in. Chance is ${f(100/R.n,1)}%.`+(R.ident_ref?` <b>Reference:</b> ${R.ident_ref.labels.join(' vs ')} on the same ${R.ident_ref.n} subjects identifies ${pc(R.ident_ref.accuracy)}. If refits identify subjects much better than that, the condition really changes individual profiles; if about equal, the condition comparison is at the optimizer's own noise floor.`:'');
  document.getElementById('rep-ident').innerHTML=`<tr><th>param</th><th>refits differ in</th><th>accuracy</th><th>I_self</th><th>I_others</th></tr>`+
    R.ident_summary.map(r=>`<tr><td>${r.param}</td><td>${r.kind}</td><td class="num">${pc(r.accuracy)}</td><td class="num">${fm(r.i_self,3)}</td><td class="num">${fm(r.i_others,3)}</td></tr>`).join('');
  document.getElementById('rep-mem').innerHTML=R.memory.length?`<tr><th>param</th><th>fits</th><th>&rho;(final, start)</th><th>moved</th><th>unmoved nodes</th></tr>`+
    R.memory.map(r=>`<tr class="${r.median_rho_final_init>0.5?'hl':''}"><td>${r.param}</td><td class="num">${r.n_fits}</td><td class="num">${fm(r.median_rho_final_init)}</td><td class="num">${fm(r.median_moved_frac,3)}</td><td class="num">${pc(r.frac_nodes_unmoved)}</td></tr>`).join(''):'<tr><td>No refits with --init-seed.</td></tr>';
  document.getElementById('rep-maps').innerHTML=`<tr><th>param</th><th>mean pair r</th><th>min pair r</th></tr>`+R.group_maps.map(r=>`<tr><td>${r.param}</td><td class="num">${fm(r.group_map_r_mean)}</td><td class="num">${fm(r.group_map_r_min)}</td></tr>`).join('');
  document.getElementById('rep-age-note').innerHTML=`The whole-brain age test (Cliff's &delta;, old vs young) recomputed in every refit. A conclusion is robust if the sign holds in every refit and it is significant in most.`+(R.snr?` Below: the ${R.snr.labels[1]} &minus; ${R.snr.labels[0]} shift per region in units of each subject's refit SD; only ratios above 1 are clearly beyond what refitting alone produces.`:'');
  document.getElementById('rep-age').innerHTML=R.age_summary.length?`<tr><th>param</th><th>median &delta;</th><th>range</th><th>refits p&lt;0.05</th><th>same sign</th></tr>`+
    R.age_summary.map(r=>`<tr class="${r.same_sign&&r.n_sig===R.labels.length?'hl':''}"><td>${r.param}</td><td class="num">${fm(r.median_delta)}</td><td class="num">[${fm(r.min_delta)}, ${fm(r.max_delta)}]</td><td class="num">${r.n_sig}/${R.labels.length}</td><td>${r.same_sign?'yes':'no'}</td></tr>`).join(''):'';
  document.getElementById('rep-snr').innerHTML=R.snr?`<tr><th>param</th><th>regions</th><th>median |shift / refit SD|</th><th>regions &gt; 1</th><th>significant regions</th><th>of which &gt; 1</th></tr>`+
    R.snr.table.map(r=>`<tr><td>${r.param}</td><td class="num">${r.n_regions}</td><td class="num">${fm(r.median_abs_shift_over_refit_sd)}</td><td class="num">${pc(r.frac_regions_gt_1)}</td><td class="num">${r.n_sig_regions}</td><td class="num">${pc(r.frac_sig_regions_gt_1)}</td></tr>`).join(''):'';
  [['rep-conv-img','rep_convergence'],['rep-icc-img','rep_icc'],['rep-ident-img','rep_ident'],['rep-mem-img','rep_memory_tradeoff'],['rep-age-img','rep_age']].forEach(([id,k])=>{ const el=document.getElementById(id), src=fig(k); el.style.display=src?'block':'none'; if(src) el.src=src; });
})();

// ---------- methods ----------
document.getElementById('caveats').innerHTML = [
  `<b>Node alignment.</b> Each subject's optimized nodes are mapped back to the shared connectome rows with <span class="mono">fmri_nodes().to_conn</span>. A region enters a parameter's analysis only if the parameter is live on its node type, it is covered in &ge;${Math.round(D.min_coverage*100)}% of the cohort, and it moved off init.`,
  `<b>Fit quality is a confound.</b> The EEG-PSD loss is the training loss (normalized-PSD MSE, mean of the last 10 epochs). In the <i>all</i> cohort, many BOLD fits failed (fc_corr ~ 0), so those subjects' parameters are shaped mostly by EEG. Trust an effect more when it also holds in the BOLD-fit subset and survives the OLS covariates.`,
  `<b>Age is binned and bimodal</b> (5-year bins, a gap in middle age), so it is a binary contrast rather than a trend. Young subjects are mostly male, so sex is a covariate.`,
  `<b>Reproducibility &ne; individuality.</b> A high split-half r with a low subject&ndash;subject r means the group map is stable while individual maps are noisy. That is expected when single-subject fits are weakly identified.`,
  `<b>Glass brains</b> use sub-${D.ref_subject}'s native-space source cloud as a canvas. They are a display device, not a registration.`,
  `<b>Silhouette guide:</b> &ge;0.5 means clear clusters, 0.25&ndash;0.5 weak, &lt;0.25 a continuum. With high-dimensional, weakly identified parameters, a continuum is the expected null.`,
  ...(D.cond_labels ? [`<b>Condition stability is not test-retest.</b> ${D.cond_labels[0]} and ${D.cond_labels[1]} differ in the EEG target, so a low ICC mixes fit noise with a genuine condition effect. No same-condition refit exists here to separate the two. A high ICC(C) with a low ICC(A) means subjects keep their ranking while the whole group shifts.`,
    `<b>Identifiability is an upper bound.</b> Both fits of a subject share the structural connectome, the BOLD/FC target, the leadfield, the init and the noise seed. Part of the same-subject similarity therefore comes from those shared inputs, not from a condition-invariant trait.`] : []),
].map(x=>`<li>${x}</li>`).join('');
document.getElementById('footer').innerHTML = `Generated by <span class="mono">examples/population_params_report_cli.py</span> from ${D.n_finished} finished fits. Excluded: ${D.excluded.map(e=>e.participant_id).join(', ')||'none'}.`;

function renderAll(){ document.querySelectorAll('#cohort-seg button').forEach(b=>b.classList.toggle('active', b.dataset.k===cohort)); renderCohort(); renderRegions(); renderClusters(); renderAge(); renderCondition(); }
renderAll();
</script>
</body></html>
"""


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True, type=Path,
                    help="folder of per-subject eeg_bold_fit_cli.py run dirs (e.g. fit_test_pop/atlas-100)")
    ap.add_argument("--bids-dir", required=True, type=Path, help="BIDS root holding the connectivity/fMRI derivatives")
    ap.add_argument("--participants", type=Path, default=None,
                    help="participants.tsv with age/sex (default: <bids-dir>/participants.tsv)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: <results-root>/population_report)")
    ap.add_argument("--age-split", type=float, default=45.0, help="young < split <= old (default 45)")
    ap.add_argument("--fc-threshold", type=float, default=0.05, help="fc_corr above which a BOLD fit counts as working")
    ap.add_argument("--eeg-fail-loss", type=float, default=1e-4,
                    help="final EEG-PSD loss above which a fit is flagged as failed (good fits sit ~1e-7)")
    ap.add_argument("--min-coverage", type=float, default=0.8, help="min fraction of subjects covering a region")
    ap.add_argument("--n-perm", type=int, default=500, help="label permutations for the age classifier")
    ap.add_argument("--clf-c", type=float, default=0.05, help="inverse L2 strength of the age classifier")
    ap.add_argument("--ref-subject", default=None, help="subject whose dipole cloud carries the glass brains")
    ap.add_argument("--no-brains", action="store_true", help="skip the pyvista glass-brain renders")
    ap.add_argument("--compare-root", type=Path, default=None,
                    help="second results root with the SAME subjects fit under another condition "
                         "(e.g. fit_test_pop_3_eo/atlas-100): adds the condition stability/shift section")
    ap.add_argument("--condition-labels", nargs=2, default=None, metavar=("A", "B"),
                    help="names of the --results-root / --compare-root conditions "
                         "(default: from each run's eeg_task, e.g. EC EO)")
    ap.add_argument("--alpha-band", nargs=2, type=float, default=(8.0, 12.0), metavar=("LO", "HI"),
                    help="alpha band in Hz for the condition alpha analysis (default 8 12)")
    ap.add_argument("--alpha-roi", nargs="+", default=None,
                    help="channels of the alpha ROI (default: all occipital O*/PO*/I* channels)")
    ap.add_argument("--replicate-roots", nargs="+", type=Path, default=None,
                    help="results roots with the SAME subjects refit under other --init-seed / --noise-seed "
                         "values (e.g. from hpc/*/submit_optim.sh replicates); --results-root is refit 0. "
                         "Adds the refit-stability section")
    ap.add_argument("--rep-loss-tol", type=float, default=0.3,
                    help="a refit is 'worse than best' if its log10 EEG loss exceeds the subject's best by more "
                         "than this (default 0.3, i.e. 2x)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.compare_root and args.compare_root.resolve() == args.results_root.resolve():
        raise SystemExit("--compare-root is the same folder as --results-root")

    out = args.out or args.results_root / "population_report"
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs under {args.results_root} ...")
    pop = load_population(args.results_root, args.bids_dir, args.participants or args.bids_dir / "participants.tsv")
    meta = pop["meta"]
    meta["old"] = meta.age >= args.age_split
    meta["eeg_fit_failed"] = meta.eeg_psd_loss > args.eeg_fail_loss
    frozen, frozen_cells = find_frozen(pop)
    params = [p for p in PER_NODE_PARAMS if p not in frozen]

    base = ((meta.gamma_weight == 0) & meta.age.notna()).to_numpy()
    excluded = meta[~base]
    masks = {"all": base, "bold_ok": base & (meta.fc_corr > args.fc_threshold).to_numpy()}
    print(f"  {len(meta)} finished fits, {len(pop['incomplete'])} unfinished, {len(excluded)} excluded "
          f"(gamma_weight != 0 / no age); params {params}, frozen {frozen}")
    n_fail = int(meta.eeg_fit_failed[base].sum())
    if n_fail:
        print(f"  WARNING: {n_fail} included fits have EEG-PSD loss > {args.eeg_fail_loss:g} (failed fits); "
              f"{int((meta.eeg_fit_failed & masks['bold_ok']).sum())} of them are in bold_ok")

    popb = None
    if args.compare_root:
        print(f"Loading comparison runs under {args.compare_root} ...")
        popb = load_population(args.compare_root, args.bids_dir, args.participants or args.bids_dir / "participants.tsv")
        if [r["name"] for r in popb["regions"]] != [r["name"] for r in pop["regions"]]:
            raise SystemExit("--compare-root uses a different connectome / atlas than --results-root")
        if popb["init"] != pop["init"] or popb["bounds"] != pop["bounds"]:
            warnings.warn("learnable_params (init/bounds) differ between the two conditions")
        mb = popb["meta"]
        mb["old"] = mb.age >= args.age_split
        mb["eeg_fit_failed"] = mb.eeg_psd_loss > args.eeg_fail_loss
        frozen_b, frozen_cells_b = find_frozen(popb)
        cond_params = [q for q in params if q not in frozen_b]
        cond_frozen = {q: sorted(set(frozen_cells.get(q, [])) | set(frozen_cells_b.get(q, []))) for q in cond_params}
        labels = args.condition_labels or [COND_SHORT.get(x["config"]["eeg_task"], x["config"]["eeg_task"])
                                           for x in (pop, popb)]
        if labels[0] == labels[1]:
            labels = [f"{labels[0]} (A)", f"{labels[1]} (B)"]
        base_b = ((mb.gamma_weight == 0) & mb.age.notna()).to_numpy()
        masks_b = {"all": base_b, "bold_ok": base_b & (mb.fc_corr > args.fc_threshold).to_numpy()}
        pos_b = {sid: i for i, sid in enumerate(mb.subject)}
        print(f"  {len(mb)} finished fits in {labels[1]}, {len(set(pos_b) & set(meta.subject[base]))} paired "
              f"with {labels[0]}; condition params {cond_params}")

    results = {}
    for c, mask in masks.items():
        print(f"  cohort {c}: n={mask.sum()}")
        results[c] = analyse_cohort(pop, mask, params, frozen_cells, args)
        if popb is None:
            continue
        ia = np.array([i for i in np.where(mask)[0]
                       if meta.subject[i] in pos_b and masks_b[c][pos_b[meta.subject[i]]]], dtype=int)
        ib = np.array([pos_b[meta.subject[i]] for i in ia], dtype=int)
        if len(ia) < 10:
            print(f"  cohort {c}: only {len(ia)} subjects in both conditions -- condition analysis skipped")
            continue
        print(f"  cohort {c}: condition {labels[0]} vs {labels[1]} on {len(ia)} paired subjects")
        results[c]["condition"] = analyse_condition(pop, popb, ia, ib, cond_params, cond_frozen, labels, args)

    rep = None
    if args.replicate_roots:
        reps = []
        for root in args.replicate_roots:
            print(f"Loading refits under {root} ...")
            reps.append(load_population(root, args.bids_dir, args.participants or args.bids_dir / "participants.tsv"))
        print(f"  refit stability over {1 + len(reps)} refits ...")
        rep = analyse_replicates(pop, reps, params, frozen_cells, results, popb, args)

    # tables
    cohort_cols = pd.DataFrame({f"in_{c}": m for c, m in masks.items()})
    meta.join(cohort_cols).to_csv(tab_dir / "subjects.csv", index=False)
    for c, res in results.items():
        res["region_stats"].to_csv(tab_dir / f"region_stats_{c}.csv", index=False)
        res["age_region"].sort_values("p_mw").to_csv(tab_dir / f"age_tests_region_{c}.csv", index=False)
        res["age_group"].sort_values("p_mw").to_csv(tab_dir / f"age_tests_network_{c}.csv", index=False)
        res["region_clusters"].to_csv(tab_dir / f"region_clusters_{c}.csv", index=False)
        res["sub"][["participant_id", "age", "sex", "old", "cluster", *[f"PC{i}" for i in range(1, 6)]]] \
            .to_csv(tab_dir / f"subject_pca_clusters_{c}.csv", index=False)
        cond = res["condition"]
        if cond is not None:
            A, B = cond["labels"]
            cond["stability"].to_csv(tab_dir / f"condition_stability_{c}.csv", index=False)
            cond["region_icc"].to_csv(tab_dir / f"condition_region_icc_{c}.csv", index=False)
            cond["diff_region"].sort_values("p_w").to_csv(tab_dir / f"condition_tests_region_{c}.csv", index=False)
            cond["diff_group"].sort_values("p_w").to_csv(tab_dir / f"condition_tests_network_{c}.csv", index=False)
            cond["fit_diff"].to_csv(tab_dir / f"condition_fit_quality_{c}.csv", index=False)
            paired = cond["subjects"].copy()
            for v, (a, b) in cond["variables"].items():
                paired[f"{v} | {A}"], paired[f"{v} | {B}"] = a, b
            al = cond["alpha"]
            if al is not None:
                al["chan_tests"].to_csv(tab_dir / f"alpha_channel_tests_{c}.csv", index=False)
                al["roi_tests"].to_csv(tab_dir / f"alpha_roi_tests_{c}.csv", index=False)
                al["param_tests"].to_csv(tab_dir / f"alpha_param_tests_{c}.csv", index=False)
                al["region_link"].sort_values("p").to_csv(tab_dir / f"alpha_region_link_{c}.csv", index=False)
                for k in ("emp", "sim"):
                    for r in ("occ", "front"):
                        paired[f"alpha reactivity {k} {r} ({A} - {B})"] = al["reactivity"][k][r]
            paired.to_csv(tab_dir / f"condition_subjects_{c}.csv", index=False)

    # figures
    print("  static figures ...")
    cond_figs = {}
    fig_fit_quality(meta, base, args.fc_threshold, fig_dir / "fit_quality.png")
    for c, res in results.items():
        fig_region_spread(pop, res, params, fig_dir / f"region_spread_{c}.png")
        fig_reliability(res, params, fig_dir / f"reliability_{c}.png")
        fig_pca(res, fig_dir / f"pca_{c}.png")
        fig_param_corr(res, fig_dir / f"param_corr_{c}.png")
        fig_volcano(res, fig_dir / f"age_volcano_{c}.png")
        fig_classifier(res, fig_dir / f"age_classifier_{c}.png")
        cond = res["condition"]
        if cond is not None:
            cp = list(cond["cols"])
            fig_condition_stability(cond, cp, fig_dir / f"condition_stability_{c}.png")
            fig_condition_scatter(cond, cp, fig_dir / f"condition_scatter_{c}.png")
            fig_condition_ident(cond, fig_dir / f"condition_ident_{c}.png")
            fig_condition_diff(pop, cond, cp, fig_dir / f"condition_diff_{c}.png")
            if cond["alpha"] is not None:
                for kind, fn in [("alpha_spectra", fig_alpha_spectra), ("alpha_topo", fig_alpha_topo),
                                 ("alpha_params", fig_alpha_params)]:
                    fn(cond, fig_dir / f"{kind}_{c}.png")
                    cond_figs[f"{kind}|{c}"] = fig_dir / f"{kind}_{c}.png"
            cond_figs[f"ident|{c}"] = fig_dir / f"condition_ident_{c}.png"
            cond_figs[f"diff|{c}"] = fig_dir / f"condition_diff_{c}.png"

    if rep is not None:
        for k in ("convergence", "variance", "region_var", "ident", "ident_summary", "memory", "group_maps", "age",
                  "age_summary"):
            rep[k].to_csv(tab_dir / f"refit_{k}.csv", index=False)
        if rep["snr"] is not None:
            rep["snr"]["table"].to_csv(tab_dir / "refit_condition_shift_vs_refit_sd.csv", index=False)
        for k, fn in [("rep_convergence", fig_rep_convergence), ("rep_icc", fig_rep_icc), ("rep_ident", fig_rep_ident),
                      ("rep_memory_tradeoff", fig_rep_memory_tradeoff), ("rep_age", fig_rep_age)]:
            fn(rep, fig_dir / f"{k}.png")
            if (fig_dir / f"{k}.png").exists():
                cond_figs[k] = fig_dir / f"{k}.png"

    brains, ref_sid = {}, None
    if not args.no_brains:
        ref_sid = args.ref_subject or meta.subject[base].iloc[0]
        print(f"  glass brains on sub-{ref_sid} ...")
        brains = write_brains(pop, results, params, ref_sid, args.bids_dir, fig_dir)

    write_markdown(pop, results, params, frozen, excluded, args, out / "summary.md", rep)
    html_path = out / "population_report.html"
    write_html(build_html_data(pop, results, params, frozen, excluded, brains, cond_figs, ref_sid, args, rep), html_path)
    print(f"  wrote {out / 'summary.md'}, {tab_dir}/*.csv, {fig_dir}/*.png")
    print(f"  wrote {html_path}  ({html_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
