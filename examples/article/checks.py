#!/usr/bin/env python
"""Article -- things to settle before making figures.

  Main configuration     atlas, BOLD model (Balloon vs HRF), schedule, EEG
                         condition, learnable params (is c_ei in?), loss
                         weights -- and whether every run agrees.
  Fit dates              when each fit finished vs the fixes the results
                         depend on (``--fix DATE=LABEL``, defaults from git:
                         31 Aug smoothing-block order, 16 Oct c_ei).
  Hyperparameter search  with ``--optuna-root`` (trial run folders): the
                         schedule(s) the search ran with, and the schedule
                         of the trial(s) whose learning rate the population
                         fits use.
  Sample sizes           folders vs finished fits vs participants.tsv, and
                         whether the unfinished fits differ in age / sex.

Writes ``tables/checks_runs.csv`` and the ``checks`` section of numbers.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common import DEFAULT_BIDS, DEFAULT_ROOT, Out, participants

KEYS = ["atlas", "spacing", "leadfield_label", "bold_model", "schedule", "optimize", "eeg_task", "fmri_task",
        "joint_eeg_weight", "joint_bold_weight", "bold_fc_weight", "bold_dfc_weight", "bold_psd_weight",
        "gamma_weight", "learning_rate", "num_epochs", "noise_seed", "init_seed", "t1_bold", "dfc_window_trs",
        "dfc_step_trs"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/article")
    ap.add_argument("--fix", nargs="+", default=["2026-08-31=smoothing-block order fix (0d8ef89)",
                                                  "2026-10-16=c_ei learnable (4863d3f)"],
                    metavar="DATE=LABEL", help="fixes every fit must postdate (add the WC fix date here)")
    ap.add_argument("--optuna-root", type=Path, default=None, help="folder of Optuna trial run folders")
    ap.add_argument("--age-split", type=float, default=45.0)
    args = ap.parse_args()
    out = Out(args.out or args.root / "article")

    rows = []
    for d in sorted(p for p in args.root.iterdir() if p.is_dir()):
        cfgf = d / "config.json"
        if not cfgf.exists():
            continue
        cfg = json.loads(cfgf.read_text())
        done = d / "optimized_params.npz"
        row = {"run": d.name, "subject": cfg["subject"]["subject"], "finished": done.exists(),
               "has_diagnostics": (d / "simulation_results.npz").exists(),
               "started": dt.datetime.fromtimestamp(cfgf.stat().st_mtime),
               "finished_at": dt.datetime.fromtimestamp(done.stat().st_mtime) if done.exists() else pd.NaT,
               "learnable": ",".join(lp["name"] for lp in cfg["learnable_params"])}
        row |= {k: cfg.get(k) for k in KEYS}
        if done.exists():
            z = np.load(done)
            row["epochs_run"] = len(z["loss_eeg"]) if "loss_eeg" in z.files else np.nan
        rows.append(row)
    runs = pd.DataFrame(rows)
    p = participants(args.bids, args.age_split)
    runs = runs.merge(p[["subject", "age", "sex", "old"]], on="subject", how="left")
    out.table(runs, "checks_runs")
    fin = runs[runs.finished]

    num = {"configuration": {k: dict(Counter(map(str, runs[k]))) for k in KEYS + ["learnable"]}}
    num["configuration"]["uniform"] = [k for k in KEYS + ["learnable"] if runs[k].astype(str).nunique() == 1]
    num["configuration"]["differs"] = [k for k in KEYS + ["learnable"] if runs[k].astype(str).nunique() > 1]
    num["configuration"]["c_ei_learnable"] = bool(runs.learnable.str.contains("c_ei").any())

    dates = {"started_min": runs.started.min(), "started_max": runs.started.max(),
             "finished_min": fin.finished_at.min(), "finished_max": fin.finished_at.max(), "fixes": {}}
    for spec in args.fix:
        day, label = spec.split("=", 1)
        cut = dt.datetime.fromisoformat(day)
        dates["fixes"][label] = {"date": day, "n_started_before": int((runs.started < cut).sum()),
                                 "all_finished_fits_after": bool((fin.started >= cut).all())}
    num["dates"] = {k: (str(v) if not isinstance(v, dict) else v) for k, v in dates.items()}

    n_part = len(p)
    unfinished = runs[~runs.finished]
    num["sample"] = {"participants_tsv": n_part, "run_folders": len(runs), "finished": int(runs.finished.sum()),
                     "finished_with_diagnostics": int(fin.has_diagnostics.sum()),
                     "unfinished": unfinished.subject.tolist(),
                     "participants_without_run_folder": sorted(set(p.subject) - set(runs.subject)),
                     "finished_with_age": int(fin.age.notna().sum())}
    if len(unfinished) >= 3:
        a, b = fin.age.dropna(), unfinished.age.dropna()
        ct = pd.crosstab(runs.finished, runs.old)
        num["sample"]["unfinished_vs_finished"] = {
            "median_age_finished": float(a.median()), "median_age_unfinished": float(b.median()),
            "p_mw_age": float(stats.mannwhitneyu(a, b)[1]),
            "frac_old_finished": float(fin.old.mean()), "frac_old_unfinished": float(unfinished.old.mean()),
            "p_fisher_old": float(stats.fisher_exact(ct.to_numpy())[1]) if ct.shape == (2, 2) else None,
            "frac_male_finished": float((fin.sex == "M").mean()),
            "frac_male_unfinished": float((unfinished.sex == "M").mean()),
        }

    if args.optuna_root:
        trials = []
        for f in args.optuna_root.rglob("config.json"):
            c = json.loads(f.read_text())
            trials.append({"trial": str(f.parent), "schedule": c.get("schedule"), "learning_rate": c.get("learning_rate"),
                           "joint_eeg_weight": c.get("joint_eeg_weight")})
        tr = pd.DataFrame(trials)
        lr = runs.learning_rate.iloc[0]
        match = tr[np.isclose(tr.learning_rate.astype(float), lr)] if len(tr) else tr
        num["optuna"] = {"n_trials": len(tr), "schedules": dict(Counter(tr.schedule)) if len(tr) else {},
                         "trials_matching_population_lr": match.to_dict(orient="records")}
    out.numbers("checks", num)
    print(json.dumps(num, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
