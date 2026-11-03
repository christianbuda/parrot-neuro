#!/usr/bin/env python
"""Article section 7 -- aging.

  AUC table (tables/age_auc.csv, fig7_auc)
      the population report's classifiers (params; fit quality; params +
      fit quality; params with fit quality regressed out inside each fold
      -- the row the "almost nothing is lost" sentence needs), plus
      params with fit quality AND the extra confounds regressed out, and
      the context baselines: empirical EEG features (per-channel relative
      band powers, alpha peak, 1/f exponent), empirical FC edges, SC edges,
      and their combination. Same CV / permutation scheme for every row.
  Haufe pattern + null (tables/age_pattern.csv, age_pattern_null.csv)
      the stable-feature count (|bootstrap ratio| >= --bsr-thr) of the real
      labels against the same count under permuted labels.
  Mechanistic link (fig7_mechanism, tables/age_markers.csv)
      do the fits reproduce the empirical young/old difference in alpha peak
      frequency and 1/f exponent, and do a and b explain it (OLS of each
      marker on age group, with and without mean a / b)?
  Confounds (tables/age_confounds.csv, age_univariate_confounds.csv)
      fMRI mean FD, EEG chunks kept, nodes kept, head size: do they differ
      by age, and how many univariate FDR hits survive adding them.

Age effects across refits: the population report with ``--replicate-roots``.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

from _common import (C_FIT, C_INIT, C_SC, DEFAULT_BIDS, DEFAULT_ROOT, INK_2, NEUTRAL_C, OLD_C, YOUNG_C, Out,
                     Subject, aperiodic_fit, alpha_peak, channel_names, eeg_chunks, head_radius_mm, load_runs,
                     mean_fd, panel, plt, posterior, style, to_conn, upper)

CONFOUNDS = ["mean_fd", "n_chunks", "n_nodes", "head_radius_mm"]
EEG_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


# ---------------------------------------------------------------------------
# per-subject empirical features

def subject_features(run, bids: Path, common_nodes: list[int] | None) -> dict:
    subj = Subject(str(bids), run.sid)
    z, cfg = run.sim, run.cfg
    freqs, names = z["eeg_freqs"], channel_names(run, subj)
    E, S = np.asarray(z["eeg_psd_target"]), np.asarray(z["eeg_psd_sim"])
    tot = (freqs >= 1) & (freqs <= 30)
    feat = {"subject": run.sid}
    for b, (lo, hi) in EEG_BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        rel = E[:, m].sum(1) / E[:, tot].sum(1)
        feat.update({f"eeg_{b}_{n}": v for n, v in zip(names, rel)})
    roi = np.array([posterior(n) for n in names])
    from _common import smooth
    Es, Ss = smooth(E)[roi].mean(0), smooth(S)[roi].mean(0)
    fb = (cfg["fmin"], cfg["fmax"])
    feat["apf_emp"] = float(alpha_peak(freqs, Es, fb)[0])
    feat["apf_sim"] = float(alpha_peak(freqs, Ss, fb)[0])
    feat["exp_emp"] = float(aperiodic_fit(freqs, Es)[0][0])
    feat["exp_sim"] = float(aperiodic_fit(freqs, Ss)[0][0])
    # inside the fitted band only (the model never saw > fmax)
    feat["exp_fitband_emp"] = float(aperiodic_fit(freqs, Es, band=fb)[0][0])
    feat["exp_fitband_sim"] = float(aperiodic_fit(freqs, Ss, band=fb)[0][0])
    feat["mean_fd"] = mean_fd(subj, cfg["fmri_task"])
    feat["n_chunks"] = len(eeg_chunks(subj, cfg["eeg_task"], cfg["chunk_length"])[0])
    feat["head_radius_mm"] = head_radius_mm(subj)
    if common_nodes is not None:
        tc = to_conn(subj, cfg)
        pos = [int(np.where(tc == c)[0][0]) for c in common_nodes]
        feat["_fc"] = upper(np.asarray(z["fc_emp"])[np.ix_(pos, pos)])
        W = subj.load.weights(cfg["atlas"], normalized=True, fmri_aligned=True)
        feat["_sc"] = np.log10(upper(W[np.ix_(pos, pos)]) + 1e-6)
    return feat


# ---------------------------------------------------------------------------
# Haufe-pattern null

def n_stable(XX: np.ndarray, y: np.ndarray, k: int, C: float, n_boot: int, thr: float, seed: int) -> int:
    from population_params_report_cli import _boot_pattern, _haufe_pattern, _resid_pipe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pattern = _haufe_pattern(_resid_pipe(k, C).fit(XX, y), XX)
    seeds = np.random.SeedSequence(seed).generate_state(n_boot)
    boots = np.array([_boot_pattern(XX, y, k, C, int(s)) for s in seeds])
    with np.errstate(invalid="ignore", divide="ignore"):
        bsr = pattern / np.nanstd(boots, axis=0)
    return int(np.sum(np.abs(bsr) >= thr))


# ---------------------------------------------------------------------------
# mechanism

def marker_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Per marker (empirical / simulated): young-old difference, and how
    much of it mean a / b account for."""
    rows = []
    for mk in ("apf", "exp", "exp_fitband"):
        for src in ("emp", "sim"):
            col = f"{mk}_{src}"
            d = df[[col, "old", "sex_M", "mean_a", "mean_b", "mean_a_vis", "mean_b_vis", "age"]].dropna()
            y = (d[col] - d[col].mean()) / d[col].std()
            old = d.old.astype(float)
            U, p = stats.mannwhitneyu(d[col][d.old], d[col][~d.old])
            base = sm.OLS(y, sm.add_constant(np.column_stack([old, d.sex_M]))).fit()
            full = sm.OLS(y, sm.add_constant(np.column_stack([old, d.sex_M, d.mean_a, d.mean_b]))).fit()
            ab = sm.OLS(y, sm.add_constant(np.column_stack([d.mean_a, d.mean_b]))).fit()
            vis = sm.OLS(y, sm.add_constant(np.column_stack([d.mean_a_vis, d.mean_b_vis]))).fit()
            rows.append({
                "marker": mk, "source": src, "n": len(d),
                "median_young": d[col][~d.old].median(), "median_old": d[col][d.old].median(),
                "cliffs_delta": 2 * U / (d.old.sum() * (~d.old).sum()) - 1, "p_mw": p,
                "rho_age": stats.spearmanr(d[col], d.age)[0],
                "beta_old": base.params.iloc[1], "p_old": base.pvalues.iloc[1],
                "beta_old_given_ab": full.params.iloc[1], "p_old_given_ab": full.pvalues.iloc[1],
                "frac_age_effect_explained_by_ab": 1 - full.params.iloc[1] / base.params.iloc[1],
                "r2_ab": ab.rsquared, "r2_ab_visual": vis.rsquared,
                "beta_a": ab.params.iloc[1], "p_a": ab.pvalues.iloc[1],
                "beta_b": ab.params.iloc[2], "p_b": ab.pvalues.iloc[2],
            })
    return pd.DataFrame(rows)


def fig_mechanism(df: pd.DataFrame, mk: pd.DataFrame, out: Out) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.4), constrained_layout=True)
    for ax, m, lab, letter in ((axs[0], "apf", "alpha peak (Hz)", "a"), (axs[1], "exp", "1/f exponent (2-40 Hz)", "b")):
        for j, src in enumerate(("emp", "sim")):
            for k, (grp, c) in enumerate(((False, YOUNG_C), (True, OLD_C))):
                v = df[df.old == grp][f"{m}_{src}"].dropna()
                x = j * 2.6 + k
                ax.boxplot(v, positions=[x], widths=0.7, showfliers=False, patch_artist=True,
                           medianprops={"color": "white"}, boxprops={"facecolor": c, "edgecolor": c},
                           whiskerprops={"color": c}, capprops={"color": c})
            r = mk[(mk.marker == m) & (mk.source == src)].iloc[0]
            ax.text(j * 2.6 + 0.5, 1.0, f"d = {r.cliffs_delta:+.2f}\np = {r.p_mw:.1g}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6, color=INK_2)
        ax.set_xticks([0.5, 3.1], ["data", "model"])
        ax.set_ylabel(lab)
        panel(ax, letter)
    for ax, (x, y, lab), letter in ((axs[2], ("mean_b", "exp_fitband_sim", "model 1/f exponent (1-15 Hz) vs b"), "c"),
                                    (axs[3], ("mean_b", "exp_fitband_emp", "data 1/f exponent (1-15 Hz) vs b"), "d")):
        for grp, c in ((False, YOUNG_C), (True, OLD_C)):
            d = df[df.old == grp]
            ax.scatter(d[x], d[y], s=6, color=c, lw=0, alpha=0.8)
        ok = df[[x, y]].dropna()
        ax.set_title(f"{lab}\nrho = {stats.spearmanr(ok[x], ok[y])[0]:.2f}", fontsize=7)
        ax.set_xlabel(x.replace("_", " ") + " (cortex, 1/ms)")
        ax.set_ylabel("exponent")
        panel(ax, letter)
    out.fig(fig, "fig7_mechanism")


def fig_auc(auc: pd.DataFrame, out: Out) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 0.28 * len(auc) + 0.8), constrained_layout=True)
    y = np.arange(len(auc))[::-1]
    col = [C_FIT if r.startswith("params") else (C_SC if "SC" in r else (NEUTRAL_C if r == "fit quality" else C_INIT))
           for r in auc.model]
    ax.barh(y, auc.auc_mean - 0.5, left=0.5, xerr=auc.auc_sd, color=col, height=0.6, error_kw={"lw": 0.8})
    ax.axvline(0.5, color=INK_2, lw=0.8)
    for yy, (_, r) in zip(y, auc.iterrows()):
        ax.text(max(r.auc_mean + r.auc_sd, 0.5) + 0.01, yy, f"{r.auc_mean:.2f}  (p={r.p_perm:.3g})", va="center", fontsize=6)
    ax.set_yticks(y, [f"{m} [{n}]" for m, n in zip(auc.model, auc.n_features)], fontsize=6)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("young vs old ROC AUC (5-fold x 10)")
    out.fig(fig, "fig7_auc")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/article")
    ap.add_argument("--age-split", type=float, default=45.0)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--clf-c", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--bsr-thr", type=float, default=3.0)
    ap.add_argument("--n-null", type=int, default=100, help="label permutations for the Haufe stable-count null")
    ap.add_argument("--n-boot-null", type=int, default=100, help="bootstraps per permutation (and for the matched observed count)")
    ap.add_argument("--no-connectivity-baselines", action="store_true", help="skip the FC / SC edge classifiers (slow)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Out(args.out or args.root / "article")
    style()

    from population_params_report_cli import (NUISANCE, _resid_pipe, age_classifier, age_test, analyse_cohort, _fdr,
                                              _zcols, find_frozen, load_population)
    from params_report_cli import LIVE_ON

    pop = load_population(args.root, args.bids, args.bids / "participants.tsv")
    meta = pop["meta"]
    meta["old"] = meta.age >= args.age_split
    frozen, frozen_cells = find_frozen(pop)
    params = [p for p in pop["params"] if p not in frozen]
    mask = ((meta.gamma_weight == 0) & meta.age.notna()).to_numpy()
    print(f"[aging] cohort all: n = {mask.sum()} ({int(meta.old[mask].sum())} old)")
    res = analyse_cohort(pop, mask, params, frozen_cells, args)
    sub = res["sub"]
    y = sub.old.to_numpy(int)
    num = {"n": int(len(sub)), "n_old": int(y.sum())}

    # -- empirical features per subject (same order as `sub`)
    runs = {r.sid: r for r in load_runs(args.root)}
    common = None
    if not args.no_connectivity_baselines:
        sets = [set(to_conn(Subject(str(args.bids), s), runs[s].cfg).tolist()) for s in sub.subject]
        common = sorted(set.intersection(*sets))
    feats = [subject_features(runs[s], args.bids, common) for s in sub.subject]
    F = pd.DataFrame([{k: v for k, v in f.items() if not k.startswith("_")} for f in feats])
    eeg_cols = [c for c in F if c.startswith("eeg_")] + ["apf_emp", "exp_emp"]
    df = sub.merge(F, on="subject")
    assert (df.subject.to_numpy() == sub.subject.to_numpy()).all()

    # -- classifiers
    X_params = np.column_stack([np.concatenate([pop["V"][p][res["idx"]][:, res["cols"][p]] for p in params], axis=1),
                                sub.G.to_numpy()])
    nuis_ext = NUISANCE + CONFOUNDS
    print(f"  classifier [params, fit quality + confounds regressed out] ({args.n_perm} permutations) ...")
    extra = {"params, fit quality + confounds out": age_classifier(
        np.column_stack([X_params, df[nuis_ext].to_numpy(float)]), y, args.n_perm, args.clf_c, args.seed,
        pipe=_resid_pipe(len(nuis_ext), args.clf_c))}
    X_eeg = df[eeg_cols].to_numpy(float)
    blocks = {"empirical EEG": X_eeg, "params + empirical EEG": np.column_stack([X_params, X_eeg])}
    if common is not None:
        X_fc = np.stack([f["_fc"] for f in feats])
        X_sc = np.stack([f["_sc"] for f in feats])
        blocks |= {"empirical FC edges": X_fc, "SC edges": X_sc,
                   "empirical EEG + FC": np.column_stack([X_eeg, X_fc]),
                   "SC + empirical EEG + FC": np.column_stack([X_sc, X_eeg, X_fc])}
    for name, X in blocks.items():
        print(f"  classifier [{name}] ({X.shape[1]} features, {args.n_perm} permutations) ...")
        extra[name] = age_classifier(X, y, args.n_perm, args.clf_c, args.seed)
    n_feat = {"params": X_params.shape[1], "fit quality": 5, "params+fit_quality": X_params.shape[1] + 5,
              "params, fit quality out": X_params.shape[1], "params, fit quality + confounds out": X_params.shape[1]}
    n_feat |= {k: v.shape[1] for k, v in blocks.items()}
    rows = []
    for name, c in list(res["classifier"].items()) + list(extra.items()):
        c.pop("pipeline", None)
        label = {"fit_quality": "fit quality"}.get(name, name)
        rows.append({"model": label, "n_features": n_feat.get(name, n_feat.get(label)), "auc_mean": c["auc_mean"],
                     "auc_sd": c["auc_sd"], "auc_perm_cv": c["auc_perm_cv"], "p_perm": c["p_perm"]})
    auc = pd.DataFrame(rows)
    out.table(auc, "age_auc")
    fig_auc(auc, out)
    num["auc"] = auc.set_index("model")[["auc_mean", "auc_sd", "p_perm", "n_features"]].to_dict(orient="index")

    # -- Haufe pattern + permuted-label null of the stable-feature count
    pat = res["age_pattern"]["table"]
    out.table(pat, "age_pattern")
    k = len(NUISANCE)
    XX = np.column_stack([X_params, sub[NUISANCE].to_numpy(float)])
    print(f"  Haufe null: {args.n_null} permutations x {args.n_boot_null} bootstraps ...")
    obs = n_stable(XX, y, k, args.clf_c, args.n_boot_null, args.bsr_thr, args.seed)
    rng = np.random.default_rng(args.seed)
    perms = [rng.permutation(y) for _ in range(args.n_null)]
    null = Parallel(n_jobs=-1)(delayed(n_stable)(XX, yp, k, args.clf_c, args.n_boot_null, args.bsr_thr, args.seed + i + 1)
                               for i, yp in enumerate(perms))
    null = np.asarray(null)
    out.table(pd.DataFrame({"n_stable_null": null}), "age_pattern_null")
    num["haufe"] = {"n_features": XX.shape[1] - k, "n_stable_report": int(pat.stable.sum()), "n_boot_report": args.n_boot,
                    "n_stable_observed_matched": obs, "n_boot_matched": args.n_boot_null,
                    "null_median": float(np.median(null)), "null_95pct": float(np.percentile(null, 95)),
                    "p_perm": float((1 + np.sum(null >= obs)) / (1 + len(null)))}

    # -- univariate: report counts, then with the extra confounds
    ar = res["age_region"]
    num["univariate"] = {"n_region_tests": len(ar), "n_fdr_mw": int((ar.q_mw < 0.05).sum()),
                         "n_fdr_ols": int((ar.q_ols < 0.05).sum()),
                         "hits": ar[(ar.q_mw < 0.05) | (ar.q_ols < 0.05)][["param", "region", "cliffs_delta", "q_mw", "q_ols"]]
                         .to_dict(orient="records")}
    Z_ext = _zcols(df, nuis_ext)
    old = df.old.to_numpy(bool)
    rows = []
    for p in params:
        for r in res["cols"][p]:
            t = age_test(pop["V"][p][res["idx"]][:, r], old, Z_ext)
            if t:
                rows.append({"param": p, "region": pop["regions"][r]["name"], **t})
    ar2 = pd.DataFrame(rows)
    ar2["q_ols"] = _fdr(ar2.p_ols)
    out.table(ar2, "age_univariate_confounds")
    num["univariate"]["n_fdr_ols_with_confounds"] = int((ar2.q_ols < 0.05).sum())
    conf = pd.DataFrame([{"confound": c, "median_young": df[c][~old].median(), "median_old": df[c][old].median(),
                          "p_mw": stats.mannwhitneyu(df[c][old].dropna(), df[c][~old].dropna())[1],
                          "rho_age": stats.spearmanr(df[c], df.age, nan_policy="omit")[0]} for c in CONFOUNDS])
    out.table(conf, "age_confounds")
    num["confounds"] = conf.set_index("confound").to_dict(orient="index")

    # -- mechanism: alpha peak / 1/f in data and model, and a / b
    cort = [r["idx"] for r in pop["regions"] if r["is_cortical"]]
    vis = [r["idx"] for r in pop["regions"] if r["group"] in ("VisCent", "VisPeri")]
    for p in ("a", "b"):
        assert LIVE_ON[p] == "surface"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df[f"mean_{p}"] = np.nanmean(pop["V"][p][res["idx"]][:, cort], axis=1)
            df[f"mean_{p}_vis"] = np.nanmean(pop["V"][p][res["idx"]][:, vis], axis=1)
    mk = marker_tests(df)
    out.table(mk, "age_markers")
    out.table(df.drop(columns=[c for c in df if c.startswith("eeg_")]), "age_subject_features")
    num["markers"] = mk.set_index(["marker", "source"]).to_dict(orient="index")
    num["markers_sim_vs_emp_rho"] = {m: stats.spearmanr(df[f"{m}_emp"], df[f"{m}_sim"], nan_policy="omit")[0]
                                     for m in ("apf", "exp", "exp_fitband")}
    fig_mechanism(df, mk, out)
    out.numbers("section7_aging", num)


if __name__ == "__main__":
    main()
