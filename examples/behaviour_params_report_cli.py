#!/usr/bin/env python
"""Fitted TVB parameters vs the LEMON behavioural, personality and medical data.

Reads a finished population report (``population_params_report_cli.py --out``:
``tables/param_values.csv.gz``, ``tables/subjects.csv``) plus the LEMON
``Behavioural_Data_MPILMBB_LEMON`` folder, whose tables are keyed by INDI IDs
(``sub-0323xx``) and mapped to the fit IDs (``sub-0100xx``) through its
``name_match.csv``. Nothing is refit.

Three independent analyses, each its own FDR family:

  cognitive    curated scores of the 8 tests (sign-flipped so higher = better)
               plus 5 domain composites (mean z-score)
  personality  the published scale scores of the questionnaires
  medical      BMI, waist-hip ratio, blood pressure / pulse, blood markers

Brain side: per subject, the whole-brain mean of every parameter on its live
regions + ``G`` and the per-network means ("subject" level), and every
parameter x region value ("region" level).

Four models per test, all rank-based (Spearman / partial Spearman: ranks of
both variables, covariates regressed out of the ranks, then the partial
correlation and its t-test) so skewed scores (CRP, RT) can't drive a hit:

  raw           no covariates
  adjusted      age group + sex + log EEG loss + log BOLD loss
                (+ education with --cog-education, cognitive only)
  within_young  sex + log EEG loss + log BOLD loss, young subjects only
  within_old    same, old subjects only

BH-FDR is applied separately per family x model x cohort x level.
Multivariate: cross-validated ridge from all region-level parameters, on top
of the adjusted covariates (Delta CV R^2 over covariates alone), with a
Freedman-Lane permutation p.

Writes into ``--out`` (default ``<report-dir>/behaviour``): ``summary.md``,
``behaviour_summary.ipynb``, ``tables/*.csv``, ``figures/*.png`` (heatmaps,
top-hit scatters, significance glass brains).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from params_report_cli import GROUP_ORDER  # noqa: E402
from population_params_report_cli import (DIV_CMAP, OLD_C, SEQ_CMAP, YOUNG_C,  # noqa: E402
                                          _fdr, _nb_cell, render_brain)

FAMILIES = ("cognitive", "personality", "medical")
MODELS = ("raw", "adjusted", "within_young", "within_old")
COVARIATES = ["old", "sex_M", "log_eeg_psd_loss", "log_bold_loss"]
FIT_COVARIATES = ["sex_M", "log_eeg_psd_loss", "log_bold_loss"]

# (score, file, column or callable, sign, domain): sign -1 for times / errors so
# higher is always better; domain None = not part of a composite
COGNITIVE = [
    ("CVLT_learning", "CVLT.csv", "CVLT_6", +1, "memory"),
    ("CVLT_delayed_recall", "CVLT.csv", "CVLT_11", +1, "memory"),
    ("LPS_reasoning", "LPS.csv", "LPS_1", +1, "executive"),
    ("RWT_s_words", "RWT.csv", "RWT_8", +1, "fluency"),
    ("RWT_animals", "RWT.csv", "RWT_20", +1, "fluency"),
    ("WST_vocabulary", "WST.csv", "WST_1", +1, "crystallised"),
    ("TMT_A_time", "TMT.csv", "TMT_1", -1, "speed"),
    ("TMT_B_time", "TMT.csv", "TMT_5", -1, "executive"),
    ("TAP_alert_RT_nosignal", "TAP-Alertness.csv", "TAP_A_6", -1, "speed"),
    ("TAP_alert_RT_signal", "TAP-Alertness.csv", "TAP_A_11", -1, "speed"),
    ("TAP_phasic_alertness", "TAP-Alertness.csv", "TAP_A_15", +1, None),
    ("TAP_simon_effect", "TAP-Incompatibility.csv", lambda d: d.TAP_I_9 - d.TAP_I_2, -1, "executive"),
    ("TAP_WM_correct", "TAP-Working Memory.csv", "TAP_WM_6", +1, "executive"),
    ("TAP_WM_omissions", "TAP-Working Memory.csv", "TAP_WM_9", -1, "executive"),
]
PERSONALITY_SKIP = {"NYC_Q_lemon.csv", "FTP.csv"}      # item-level only / mostly empty
EDUCATION = {"Gymnasium": 3, "Gymansium": 3, "Realschule": 2, "Hauptschule": 1,
             "none (Hauptschule not finished)": 0}


# ---------------------------------------------------------------------------
# loading

def read_table(path: Path, to_initial: dict) -> pd.DataFrame:
    """One LEMON csv as numbers, indexed by the fit ID (sub-0100xx)."""
    d = pd.read_csv(path, index_col=0, dtype=str)
    if "ID" in d.columns and d.index.isna().all():            # YFAS: keyed by Initial ID already
        d = d.set_index("ID")
    else:
        d.index = d.index.map(to_initial)
    d = d[d.index.notna() & ~d.index.duplicated()]
    d = d.apply(lambda c: c.str.strip().str.replace(r"^[<>]\s*", "", regex=True))   # "<0.6", ">76"
    return d.apply(pd.to_numeric, errors="coerce")


def find(bdir: Path, name: str) -> Path:
    hits = sorted(bdir.rglob(name))
    if not hits:
        raise SystemExit(f"{name} not found under {bdir}")
    return hits[0]


def load_cognitive(bdir: Path, to_initial: dict) -> tuple[pd.DataFrame, dict]:
    cache, cols, domain = {}, {}, {}
    for name, fname, col, sign, dom in COGNITIVE:
        d = cache.setdefault(fname, read_table(find(bdir / "Cognitive_Test_Battery_LEMON", fname), to_initial))
        cols[name] = sign * (col(d) if callable(col) else d[col])
        domain[name] = dom
    S = pd.DataFrame(cols)
    z = (S - S.mean()) / S.std()
    for dom in sorted({v for v in domain.values() if v}):
        members = [k for k, v in domain.items() if v == dom]
        S[f"{dom}_composite"] = z[members].mean(axis=1, skipna=True)
        domain[f"{dom}_composite"] = dom
    return S, domain


def load_personality(bdir: Path, to_initial: dict) -> tuple[pd.DataFrame, dict]:
    parts, src, mdbf = [], {}, []
    for f in sorted((bdir / "Emotion_and_Personality_Test_Battery_LEMON").glob("*.csv")):
        if f.name in PERSONALITY_SKIP:
            continue
        d = read_table(f, to_initial)
        if f.name.startswith("MDBF_Day"):
            mdbf.append(d.rename(columns=lambda c: c.replace(f.stem.split("_")[1] + "_", "")))
            continue
        if f.name == "YFAS.csv":
            d = d[["YFAS_symptom_count"]]
        parts.append(d)
        src.update({c: f.stem for c in d.columns})
    if mdbf:      # mood state on up to 3 days -> mean per scale
        m = pd.concat(mdbf).groupby(level=0).mean()
        parts.append(m)
        src.update({c: "MDBF" for c in m.columns})
    return pd.concat(parts, axis=1), src


def load_medical(bdir: Path, to_initial: dict) -> tuple[pd.DataFrame, dict]:
    med = bdir / "Medical_LEMON"
    a = read_table(find(med, "Anthropometry_LEMON.csv"), to_initial)
    bp = read_table(find(med, "Blood_Pressure_LEMON.csv"), to_initial)
    bl = read_table(find(med, "Blood_Results_LEMON.csv"), to_initial)
    S = pd.DataFrame({
        "BMI": a.Weight_kg / (a.Height_cm / 100) ** 2,
        "waist_hip_ratio": a.Waist_cm / a.Hip_cm,
        "BP_systolic": bp.filter(like="systole").mean(axis=1),
        "BP_diastolic": bp.filter(like="diastole").mean(axis=1),
        "pulse": bp.filter(like="pulse").mean(axis=1),
    })
    src = {c: ("anthropometry" if c in ("BMI", "waist_hip_ratio") else "blood pressure") for c in S.columns}
    blood = bl[[c for c in bl.columns if "Reference" not in c and "Date" not in c]]
    blood = blood.rename(columns=lambda c: c.replace(" ", "").replace("_in_", " [").replace("µ", "u") + "]"
                         if "_in_" in c else c.replace(" ", ""))
    src.update({c: "blood" for c in blood.columns})
    return pd.concat([S, blood], axis=1), src


def load_education(bdir: Path, to_initial: dict) -> pd.Series:
    meta = pd.read_csv(find(bdir, "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"), index_col=0)
    meta.index = meta.index.map(to_initial)
    return meta[meta.index.notna()].Education.map(EDUCATION)


def brain_features(values: pd.DataFrame, subj: pd.DataFrame, params: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject level (whole-brain + network means, G) and region level, both
    subjects x features; columns are a (param, region/group) MultiIndex."""
    wb = values.groupby(["participant_id", "param"]).value.mean().unstack()[params]
    wb.columns = pd.MultiIndex.from_tuples([(p, "whole brain") for p in wb.columns])
    net = values.groupby(["participant_id", "param", "group"]).value.mean().unstack(["param", "group"])
    net = net[[c for c in net.columns if c[0] in params]]
    net = net[sorted(net.columns, key=lambda c: (params.index(c[0]), GROUP_ORDER.index(c[1])
                                                  if c[1] in GROUP_ORDER else 99))]
    G = subj.set_index("participant_id")[["G"]]
    G.columns = pd.MultiIndex.from_tuples([("G", "global")])
    subject_level = pd.concat([wb, G.reindex(wb.index), net], axis=1)
    region = values.pivot_table(index="participant_id", columns=["param", "region"], values="value")
    region = region[[c for c in region.columns if c[0] in params]]
    return subject_level, region


# ---------------------------------------------------------------------------
# statistics

def _rank(A: np.ndarray) -> np.ndarray:
    return stats.rankdata(A, axis=0)


def partial_spearman(y: np.ndarray, X: np.ndarray, C: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r, p, n) of y against every column of X, ranks of both with the
    covariates C regressed out (C=None: plain Spearman). Columns of X with
    missing values are done one by one on their own complete cases."""
    ok = np.isfinite(y) & (np.all(np.isfinite(C), axis=1) if C is not None else True)
    y, X = y[ok], X[ok]
    C = C[ok] if C is not None else None
    f = X.shape[1]
    r, p, n = np.full(f, np.nan), np.full(f, np.nan), np.zeros(f, int)

    def run(rows, cols):
        yy, XX = _rank(y[rows]), _rank(X[rows][:, cols])
        D = np.ones((len(yy), 1)) if C is None else np.column_stack([np.ones(len(yy)), C[rows]])
        D = D[:, np.concatenate([[True], D[:, 1:].std(0) > 0])]          # drop covariates constant here
        H = np.linalg.pinv(D)
        ry = yy - D @ (H @ yy)
        rX = XX - D @ (H @ XX)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = (rX * ry[:, None]).sum(0) / np.sqrt((rX ** 2).sum(0) * (ry ** 2).sum())
            df = len(yy) - D.shape[1] - 1
            t = rr * np.sqrt(df / np.clip(1 - rr ** 2, 1e-12, None))
        r[cols], p[cols], n[cols] = rr, 2 * stats.t.sf(np.abs(t), df), len(yy)

    complete = np.all(np.isfinite(X), axis=0)
    if len(y) >= 10 and complete.any():
        run(np.ones(len(y), bool), np.where(complete)[0])
    for j in np.where(~complete)[0]:
        rows = np.isfinite(X[:, j])
        if rows.sum() >= 10:
            run(rows, np.array([j]))
    return r, p, n


def covariates(subj: pd.DataFrame, model: str, extra: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray | None]:
    """(row mask, covariate matrix) of one model."""
    rows = {"raw": np.ones(len(subj), bool), "adjusted": np.ones(len(subj), bool),
            "within_young": ~subj.old.to_numpy(bool), "within_old": subj.old.to_numpy(bool)}[model]
    if model == "raw":
        return rows, None
    cols = COVARIATES if model == "adjusted" else FIT_COVARIATES
    C = subj[cols].astype(float)
    if extra is not None:
        C = pd.concat([C, extra.astype(float)], axis=1)
    return rows, C.to_numpy()


def run_tests(scores: pd.DataFrame, feats: pd.DataFrame, subj: pd.DataFrame, extra: pd.DataFrame | None,
              level: str) -> pd.DataFrame:
    out = []
    for model in MODELS:
        rows, C = covariates(subj, model, extra)
        X = feats.to_numpy(float)[rows]
        for b in scores.columns:
            r, p, n = partial_spearman(scores[b].to_numpy(float)[rows], X, None if C is None else C[rows])
            out.append(pd.DataFrame({"model": model, "level": level, "behaviour": b,
                                     "param": feats.columns.get_level_values(0),
                                     "where": feats.columns.get_level_values(1), "n": n, "r": r, "p": p}))
    res = pd.concat(out, ignore_index=True)
    res["q"] = res.groupby("model").p.transform(lambda s: pd.Series(_fdr(s), index=s.index))
    return res


def _ridge_prep(Xb: np.ndarray, C: np.ndarray, tr: np.ndarray, te: np.ndarray) -> tuple:
    """Everything of one train/test split that doesn't depend on y: the
    covariate projection, and the SVD of the train params with the covariates
    regressed out (train fit) and z-scored."""
    Hc = np.linalg.pinv(C[tr])
    Bx = Hc @ Xb[tr]
    Xr_tr, Xr_te = Xb[tr] - C[tr] @ Bx, Xb[te] - C[te] @ Bx
    mu, sd = Xr_tr.mean(0), Xr_tr.std(0) + 1e-12
    Ztr, Zte = (Xr_tr - mu) / sd, (Xr_te - mu) / sd
    # thin SVD via the n x n Gram matrix (p >> n; LAPACK's SVD fails to converge on
    # some of these near-degenerate folds, eigh doesn't)
    ev, U = np.linalg.eigh(Ztr @ Ztr.T)
    keep = ev > ev.max() * 1e-10
    U, s = U[:, keep][:, ::-1], np.sqrt(ev[keep])[::-1]
    return tr, te, Hc, U, s, (Zte @ Ztr.T) @ U / s


def _ridge_predict(prep: tuple, C: np.ndarray, y: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Test-set predictions of the covariate-only model and of covariates +
    ridge(params), the latter for every alpha in ``a`` (n_test x n_alpha)."""
    tr, te, Hc, U, s, XteV = prep
    bc = Hc @ y[tr]
    res = y[tr] - C[tr] @ bc
    Uy = U.T @ (res - res.mean())
    coef = (s / (s ** 2 + a[:, None])).T * Uy[:, None]           # n_comp x n_alpha
    base = C[te] @ bc
    return base, base[:, None] + res.mean() + XteV @ coef


def _ridge_one(y: np.ndarray, C: np.ndarray, Xb: np.ndarray, n_perm: int, seed: int,
               n_folds: int = 5) -> dict:
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(seed)
    alphas = np.logspace(-3, 3, 25)
    with threadpool_limits(1):                    # small matrices: BLAS threads only spin
        outer = []
        for te in np.array_split(rng.permutation(len(y)), n_folds):
            tr = np.setdiff1d(np.arange(len(y)), te)
            p = _ridge_prep(Xb, C, tr, te)
            a = alphas * p[4][0] ** 2                                  # scale by the top singular value
            inner = [_ridge_prep(Xb, C, tr[np.setdiff1d(np.arange(len(tr)), ite)], tr[ite])
                     for ite in np.array_split(rng.permutation(len(tr)), n_folds)]
            outer.append((p, a, inner))

        def cv_r2(yy):
            pred_c, pred_f = np.empty_like(yy), np.empty_like(yy)
            for p, a, inner in outer:
                err = sum(((yy[q[1]][:, None] - _ridge_predict(q, C, yy, a)[1]) ** 2).sum(0) for q in inner)
                base, full = _ridge_predict(p, C, yy, a)
                pred_c[p[1]], pred_f[p[1]] = base, full[:, int(np.argmin(err))]
            sst = ((yy - yy.mean()) ** 2).sum()
            return 1 - ((yy - pred_c) ** 2).sum() / sst, 1 - ((yy - pred_f) ** 2).sum() / sst

        r2c, r2f = cv_r2(y)
        bfull = np.linalg.pinv(C) @ y
        yhat, e = C @ bfull, y - C @ bfull
        null = np.array([np.subtract(*cv_r2(yhat + rng.permutation(e))[::-1]) for _ in range(n_perm)])
    delta = r2f - r2c
    return {"n": len(y), "cv_r2_covariates": r2c, "cv_r2_params": r2f, "delta_r2": delta,
            "p_perm": (1 + np.sum(null >= delta)) / (1 + n_perm)}


def ridge_prediction(scores: pd.DataFrame, X: pd.DataFrame, subj: pd.DataFrame, extra: pd.DataFrame | None,
                     n_perm: int, seed: int, n_jobs: int = 1) -> pd.DataFrame:
    """Delta CV R^2 of ridge(region params) on top of the adjusted covariates.
    Alpha is chosen by an inner 5-fold CV inside each outer training fold
    (GCV / leave-one-out shortcuts degenerate here: with the covariates
    regressed out and p >> n the training fit is exact at small alpha).
    Freedman-Lane permutations: residuals of the covariate-only model are
    shuffled, X and C stay fixed, so every SVD is computed once. One worker
    process per score."""
    from joblib import Parallel, delayed
    _, C_all = covariates(subj, "adjusted", extra)
    Xv = X.to_numpy(float)
    Xv = np.where(np.isfinite(Xv), Xv, np.nanmedian(Xv, axis=0))         # few missing regions -> median
    jobs, names = [], []
    seeds = np.random.SeedSequence(seed).spawn(scores.shape[1])
    for b, ss in zip(scores.columns, seeds):
        y = scores[b].to_numpy(float)
        ok = np.isfinite(y) & np.all(np.isfinite(C_all), axis=1)
        if ok.sum() < 30:
            continue
        C = np.column_stack([np.ones(ok.sum()), C_all[ok]])
        jobs.append(delayed(_ridge_one)(y[ok], C, Xv[ok], n_perm, int(ss.generate_state(1)[0])))
        names.append(b)
    df = pd.DataFrame(Parallel(n_jobs=n_jobs)(jobs), index=pd.Index(names, name="behaviour")).reset_index()
    if len(df):
        df["q_perm"] = _fdr(df.p_perm)
    return df


# ---------------------------------------------------------------------------
# figures

def fig_heatmap(res: pd.DataFrame, model: str, family: str, cohort: str, q: float, out: Path) -> None:
    d = res[(res.model == model) & (res["where"].isin(["whole brain", "global"]))]
    R = d.pivot(index="behaviour", columns="param", values="r")
    Q = d.pivot(index="behaviour", columns="param", values="q").reindex_like(R)
    order = [c for c in [*dict.fromkeys(d.param)] if c != "G"] + (["G"] if "G" in R.columns else [])
    R, Q = R[order], Q[order]
    lim = max(0.2, float(np.nanmax(np.abs(R.to_numpy()))))
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(order), 1.2 + 0.24 * len(R)))
    im = ax.imshow(R.to_numpy(), cmap=DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(Q.to_numpy()):
        if v < q:
            ax.text(j, i, "*", ha="center", va="center", fontsize=11, color="black")
    ax.set_xticks(range(len(order)), order)
    ax.set_yticks(range(len(R)), R.index, fontsize=7)
    label = "Spearman rho" if model == "raw" else "partial Spearman r"
    fig.colorbar(im, ax=ax, shrink=0.6, label=label)
    ax.set_title(f"{family}, {model}, {cohort}: whole-brain means (* subject-level q < {q:g})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_top_hits(res: pd.DataFrame, scores: pd.DataFrame, feats: pd.DataFrame, subj: pd.DataFrame,
                 model: str, family: str, out: Path, k: int = 6) -> bool:
    top = res[res.model == model].nsmallest(k, "p")
    if top.empty:
        return False
    nc = 3
    nr = int(np.ceil(len(top) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.4 * nr), squeeze=False)
    old = subj.old.to_numpy(bool)
    for ax, r in zip(axes.flat, top.itertuples()):
        x, y = feats[(r.param, r.where)].to_numpy(float), scores[r.behaviour].to_numpy(float)
        for m, c, lab in [(~old, YOUNG_C, "young"), (old, OLD_C, "old")]:
            ax.scatter(x[m], y[m], s=12, color=c, alpha=0.7, label=lab)
        ax.set_xlabel(f"{r.param} | {r.where}", fontsize=8)
        ax.set_ylabel(r.behaviour, fontsize=8)
        ax.set_title(f"r {r.r:+.2f}, q {r.q:.2g}", fontsize=9)
    for ax in axes.flat[len(top):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{family}: strongest subject-level associations ({model})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def write_brains(region_res: pd.DataFrame, family: str, q: float, ref_sid: str, cfg: dict, bids_dir: Path,
                 fig_dir: Path, max_behaviours: int) -> list[str]:
    from parrot_neuro import Subject
    subj = Subject(str(bids_dir), ref_sid)
    names = subj.load.connectivity_labels(cfg["atlas"])
    idx = {n: i for i, n in enumerate(names)}
    positions = subj.load.dipoles(float(cfg["spacing"])).positions
    node = subj.load.dipole_node_labels(cfg["atlas"], float(cfg["spacing"]), task=cfg["fmri_task"])
    to_conn = np.asarray(subj.load.fmri_nodes(cfg["atlas"], cfg["fmri_task"]).to_conn)
    point_region = np.where(node >= 0, to_conn[node.clip(min=0)], -1)
    M = len(names)
    written = []
    for model in ("raw", "adjusted"):
        sig = region_res[(region_res.model == model) & (region_res.q < q)]
        ranked = sig.groupby("behaviour").size().sort_values(ascending=False)
        for b in ranked.index[:max_behaviours]:
            s = sig[sig.behaviour == b]
            counts = s.groupby("where").size()
            vals = np.full(M, np.nan)
            vals[[idx[r] for r in counts.index]] = counts.to_numpy()
            top = max(1, int(counts.max()))
            path = fig_dir / f"brain_{family}_{model}_{b}_any.png"
            render_brain(positions, point_region, vals, SEQ_CMAP, (0.5, top + 0.5 if top > 1 else 1.5), "# params",
                         f"{b} ({model}): {len(counts)} regions with any param q < {q:g}", path)
            written.append(path.name)
            for p in s.param.unique():
                sp = s[s.param == p]
                vals = np.full(M, np.nan)
                vals[[idx[r] for r in sp["where"]]] = sp.r.to_numpy()
                lim = max(0.1, float(np.abs(sp.r).max()))
                path = fig_dir / f"brain_{family}_{model}_{b}_{p}.png"
                render_brain(positions, point_region, vals, DIV_CMAP, (-lim, lim), "r",
                             f"{b} vs {p} ({model}): {len(sp)} regions q < {q:g}", path)
                written.append(path.name)
    return written


# ---------------------------------------------------------------------------
# markdown + notebook

def md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if df.empty:
        return ["(none)"]
    fmt = lambda v: f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)   # noqa: E731
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
            *["| " + " | ".join(fmt(getattr(r, c)) for c in cols) + " |" for r in df.itertuples()]]


def write_markdown(fam_res: dict, args: argparse.Namespace, out: Path) -> None:
    L = ["# Fitted parameters vs LEMON behaviour", "",
         f"Report `{args.report_dir}`, behaviour `{args.behaviour_dir}`. Rank-based tests (Spearman / partial "
         f"Spearman); BH-FDR q < {args.sig_q:g} within each family x model x cohort x level.", "",
         "Models: **raw** no covariates; **adjusted** age group + sex + log EEG loss + log BOLD loss"
         + (" (+ education, cognitive)" if args.cog_education else "")
         + "; **within_young / within_old** the same without age, inside each group.", ""]
    for fam, F in fam_res.items():
        L += [f"## {fam.capitalize()}", "",
              f"{F['n_scores']} scores ({', '.join(F['dropped']) + ' dropped for < 50% coverage' if F['dropped'] else 'none dropped'}).", ""]
        if fam == "cognitive":
            L += ["Sanity check, Spearman with age (scores are signed so higher = better; LEMON's older group is "
                  "slower and recalls less):", "", *md_table(F["age_check"], ["score", "rho_age", "n"]), ""]
        for c, R in F["cohorts"].items():
            L += [f"### Cohort `{c}`", "", "| model | subject-level hits | region-level hits |", "|---|---|---|"]
            for m in MODELS:
                s, r = R["subject"], R["region"]
                L.append(f"| {m} | {int((s[s.model == m].q < args.sig_q).sum())} / {int((s.model == m).sum())} "
                         f"| {int((r[r.model == m].q < args.sig_q).sum())} / {int((r.model == m).sum())} |")
            s = R["subject"]
            raw_hits = s[(s.model == "raw") & (s.q < args.sig_q)]
            adj = s[s.model == "adjusted"].set_index(["behaviour", "param", "where"]).q
            surv = adj.reindex(pd.MultiIndex.from_frame(raw_hits[["behaviour", "param", "where"]])) < args.sig_q
            L += ["", f"Of the {len(raw_hits)} raw subject-level hits, {int(surv.sum())} survive the age/fit "
                      "adjustment.", ""]
            for m in ("raw", "adjusted"):
                top = s[(s.model == m) & (s.q < args.sig_q)].nsmallest(12, "p")
                L += [f"Top subject-level hits, {m}:", "", *md_table(top, ["behaviour", "param", "where", "n", "r", "q"]), ""]
            for m in ("raw", "adjusted"):
                r = R["region"]
                top = r[(r.model == m) & (r.q < args.sig_q)].nsmallest(10, "p")
                L += [f"Top region-level hits, {m}:", "", *md_table(top, ["behaviour", "param", "where", "n", "r", "q"]), ""]
            P = R["prediction"]
            if P is not None and len(P):
                L += ["Ridge from all region params on top of the adjusted covariates (Delta CV R^2, permutation p, "
                      "FDR across scores):", "",
                      *md_table(P.sort_values("p_perm").head(10),
                                ["behaviour", "n", "cv_r2_covariates", "cv_r2_params", "delta_r2", "p_perm", "q_perm"]), ""]
    out.write_text("\n".join(L) + "\n")


def write_notebook(out: Path, args: argparse.Namespace) -> Path:
    md, code = (lambda t: _nb_cell("markdown", t)), (lambda t: _nb_cell("code", t))
    cells = [
        md(f"""
# Fitted parameters vs LEMON behaviour

Companion to `summary.md`. Report `{args.report_dir}`, behaviour data `{args.behaviour_dir}`.
Rank-based tests; BH-FDR q < {args.sig_q:g} within each family x model x cohort x level.
Models: `raw` (no covariates), `adjusted` (age group, sex, log EEG / BOLD loss), `within_young`, `within_old`.
"""),
        code(f"""
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

OUT = Path({str(out.resolve())!r})
if not OUT.exists():
    OUT = Path.cwd()
TAB, FIG = OUT / "tables", OUT / "figures"
Q = {args.sig_q!r}
FAMILIES = {list(FAMILIES)!r}
MODELS = {list(MODELS)!r}
pd.set_option("display.max_rows", 80, "display.float_format", "{{:.4g}}".format)
scores = pd.read_csv(TAB / "behaviour_scores.csv", index_col=0)

def show(name, width=900):
    p = FIG / name
    display(Image(filename=str(p), width=width)) if p.exists() else print(f"(no {{name}})")

def tests(family, level, cohort="all"):
    return pd.read_csv(TAB / f"{{family}}_{{level}}_{{cohort}}.csv")
"""),
        md("## Hits per family, model and level"),
        code("""
rows = []
for f in FAMILIES:
    for c in ("all", "bold_ok"):
        for lev in ("subject", "region"):
            t = tests(f, lev, c)
            for m in MODELS:
                d = t[t.model == m]
                rows.append({"family": f, "cohort": c, "level": lev, "model": m,
                             "hits": int((d.q < Q).sum()), "tests": len(d)})
pd.DataFrame(rows).pivot_table(index=["family", "cohort", "level"], columns="model", values="hits")[MODELS]
"""),
    ]
    for fam in FAMILIES:
        cells += [
            md(f"## {fam.capitalize()}"),
            code(f"""
t = tests({fam!r}, "subject")
for m in ("raw", "adjusted"):
    display(Markdown(f"**{{m}}: significant subject-level associations**"))
    display(t[(t.model == m) & (t.q < Q)].sort_values("p").head(25))
    show(f"heatmap_{fam}_{{m}}_all.png")
show("top_{fam}_raw.png"); show("top_{fam}_adjusted.png")
"""),
            code(f"""
r = tests({fam!r}, "region")
for m in ("raw", "adjusted"):
    s = r[(r.model == m) & (r.q < Q)]
    display(Markdown(f"**{{m}}: {{len(s)}} significant region-level tests** (by score and parameter)"))
    if len(s):
        display(s.groupby(["behaviour", "param"]).size().unstack(fill_value=0))
p = TAB / "{fam}_prediction_all.csv"
if p.exists():
    display(Markdown("**Ridge: params on top of the adjusted covariates**"))
    display(pd.read_csv(p).sort_values("p_perm").head(15))
"""),
            code(f"""
for name in sorted(n.name for n in FIG.glob("brain_{fam}_*_any.png")):
    show(name)
"""),
        ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = out / "behaviour_summary.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    return path


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", required=True, type=Path,
                    help="population_params_report_cli.py output dir (needs tables/param_values.csv.gz)")
    ap.add_argument("--behaviour-dir", required=True, type=Path, help="Behavioural_Data_MPILMBB_LEMON folder")
    ap.add_argument("--bids-dir", type=Path, default=None, help="BIDS root, for the glass brains")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="fit runs (atlas/spacing/task for the glass brains; default: parent of --report-dir)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: <report-dir>/behaviour)")
    ap.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    ap.add_argument("--sig-q", type=float, default=0.05)
    ap.add_argument("--min-coverage", type=float, default=0.5, help="drop scores missing in more fitted subjects")
    ap.add_argument("--cog-education", action="store_true", help="add education to the cognitive covariates")
    ap.add_argument("--n-perm", type=int, default=200, help="permutations of the ridge prediction (0 = skip)")
    ap.add_argument("--ref-subject", default=None)
    ap.add_argument("--no-brains", action="store_true")
    ap.add_argument("--max-brain-scores", type=int, default=8,
                    help="glass brains for at most this many scores per family/model (most region hits first)")
    ap.add_argument("--n-jobs", type=int, default=16, help="worker processes for the ridge permutations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tab_in = args.report_dir / "tables"
    if not (tab_in / "param_values.csv.gz").exists():
        raise SystemExit(f"{tab_in}/param_values.csv.gz missing -- rerun population_params_report_cli.py on this "
                         "results root (the table was added together with this script)")
    out = args.out or args.report_dir / "behaviour"
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    subj = pd.read_csv(tab_in / "subjects.csv", dtype={"subject": str})
    subj = subj[subj.in_all].reset_index(drop=True)
    missing = [c for c in COVARIATES if c not in subj.columns]
    if missing:
        raise SystemExit(f"subjects.csv lacks {missing} -- rerun population_params_report_cli.py")
    subj["old"] = subj.old.astype(bool)
    values = pd.read_csv(tab_in / "param_values.csv.gz")
    params = [p for p in pd.read_csv(tab_in / "param_bounds.csv").param if p in set(values.param)]
    feats_s, feats_r = brain_features(values, subj, params)
    feats_s, feats_r = feats_s.reindex(subj.participant_id), feats_r.reindex(subj.participant_id)

    nm = pd.read_csv(find(args.behaviour_dir, "name_match.csv"))
    to_initial = dict(zip(nm.INDI_ID, nm.Initial_ID))
    unmapped = sorted(set(subj.participant_id) - set(nm.Initial_ID))
    print(f"{len(subj)} fitted subjects ({int(subj.old.sum())} old), {len(subj) - len(unmapped)} in name_match.csv"
          + (f"; unmapped: {unmapped}" if unmapped else ""))
    edu = load_education(args.behaviour_dir, to_initial).reindex(subj.participant_id) if args.cog_education else None

    loaders = {"cognitive": load_cognitive, "personality": load_personality, "medical": load_medical}
    all_scores, fam_res = [], {}
    cohorts = {"all": np.ones(len(subj), bool), "bold_ok": subj.in_bold_ok.to_numpy(bool)}
    for fam in args.families:
        S, _ = loaders[fam](args.behaviour_dir, to_initial)
        S = S.reindex(subj.participant_id)
        cov = S.notna().mean()
        dropped = sorted(cov[cov < args.min_coverage].index)
        S = S.drop(columns=dropped)
        S = S.loc[:, S.nunique() > 2]                                  # constant / binary columns out
        print(f"[{fam}] {S.shape[1]} scores, median coverage {S.notna().mean().median():.0%}"
              + (f", dropped {dropped}" if dropped else ""))
        all_scores.append(S.add_prefix(f"{fam}:"))
        F = {"n_scores": S.shape[1], "dropped": dropped, "cohorts": {}}
        if fam == "cognitive":
            F["age_check"] = pd.DataFrame([{"score": c, "rho_age": stats.spearmanr(S[c], subj.age, nan_policy="omit")[0],
                                            "n": int(S[c].notna().sum())} for c in S.columns])
            print(F["age_check"].round(2).to_string(index=False))
        extra = edu.to_frame("education") if (edu is not None and fam == "cognitive") else None
        for c, mask in cohorts.items():
            sc, sb = S[mask].reset_index(drop=True), subj[mask].reset_index(drop=True)
            fs, fr = feats_s[mask].reset_index(drop=True), feats_r[mask].reset_index(drop=True)
            ex = None if extra is None else extra[mask].reset_index(drop=True)
            print(f"  [{fam}/{c}] n={mask.sum()}: subject-level ...")
            R = {"subject": run_tests(sc, fs, sb, ex, "subject")}
            print(f"  [{fam}/{c}] region-level ({fr.shape[1]} features x {sc.shape[1]} scores x {len(MODELS)} models) ...")
            R["region"] = run_tests(sc, fr, sb, ex, "region")
            R["prediction"] = None
            if args.n_perm > 0:
                print(f"  [{fam}/{c}] ridge prediction ({args.n_perm} permutations) ...")
                R["prediction"] = ridge_prediction(sc, fr, sb, ex, args.n_perm, args.seed, args.n_jobs)
                R["prediction"].to_csv(tab_dir / f"{fam}_prediction_{c}.csv", index=False)
            for lev in ("subject", "region"):
                R[lev].sort_values(["model", "p"]).to_csv(tab_dir / f"{fam}_{lev}_{c}.csv", index=False)
            F["cohorts"][c] = R
            if c == "all":
                for m in ("raw", "adjusted"):
                    fig_heatmap(R["subject"], m, fam, c, args.sig_q, fig_dir / f"heatmap_{fam}_{m}_{c}.png")
                    fig_top_hits(R["subject"], sc, fs, sb, m, fam, fig_dir / f"top_{fam}_{m}.png")
        fam_res[fam] = F
    pd.concat(all_scores, axis=1).set_index(subj.participant_id).to_csv(tab_dir / "behaviour_scores.csv")

    if not args.no_brains:
        if args.bids_dir is None:
            print("  no --bids-dir: glass brains skipped")
        else:
            root = args.results_root or args.report_dir.parent
            cfg_path = next(iter(sorted(root.glob("*/config.json"))), None)
            if cfg_path is None:
                print(f"  no */config.json under {root}: glass brains skipped (pass --results-root)")
            else:
                cfg = json.loads(cfg_path.read_text())
                ref = (args.ref_subject or subj.subject.iloc[0]).replace("sub-", "")
                for fam, F in fam_res.items():
                    print(f"  [{fam}] glass brains on sub-{ref} ...")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        write_brains(F["cohorts"]["all"]["region"], fam, args.sig_q, ref, cfg, args.bids_dir,
                                     fig_dir, args.max_brain_scores)

    write_markdown(fam_res, args, out / "summary.md")
    nb = write_notebook(out, args)
    print(f"  wrote {out / 'summary.md'}, {nb.name}, {tab_dir}/*.csv, {fig_dir}/*.png")


if __name__ == "__main__":
    main()
