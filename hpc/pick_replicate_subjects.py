#!/usr/bin/env python
"""Pick the subjects for the refit-stability replicates (submit_optim.sh replicates).

Candidates must have a finished fit (``optimized_params.npz``) under EVERY
``--require-root`` (e.g. the eyes-closed base run AND the eyes-open run, so the
refits also give the EC/EO comparison its noise baseline), a known age, and,
optionally, a usable fit (EEG loss below ``--max-eeg-loss``, fc_corr above
``--min-fc-corr`` in every root). From those, ``--n`` subjects are drawn at
random (seeded), split as evenly as possible between the young and old groups
of participants.tsv, so the age analysis can be repeated on the refits.

    pixi run python hpc/pick_replicate_subjects.py \\
        --participants $BIDS/participants.tsv \\
        --require-root $WORKDIR/.../fit_test_pop_2/atlas-100 \\
        --require-root $WORKDIR/.../fit_test_pop_3_eo/atlas-100 \\
        --n 32 --out replicate_subjects.txt

then ``REPLICATE_SUBJECTS_FILE=replicate_subjects.txt ./submit_optim.sh replicates``.
Writes one subject id per line (no ``sub-`` prefix). Needs only numpy + pandas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finished(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(root.iterdir()):
        if not (d / "optimized_params.npz").exists() or not (d / "config.json").exists():
            continue
        sid = json.loads((d / "config.json").read_text())["subject"]["subject"]
        z = np.load(d / "optimized_params.npz")
        loss = float(np.mean(z["loss_eeg"][-10:])) if "loss_eeg" in z.files and len(z["loss_eeg"]) else np.nan
        diag = d / "diagnostics_metrics.json"
        fc = json.loads(diag.read_text()).get("fc_corr", np.nan) if diag.exists() else np.nan
        rows.append({"subject": sid, "eeg_loss": loss, "fc_corr": float(fc)})
    return pd.DataFrame(rows, columns=["subject", "eeg_loss", "fc_corr"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participants", required=True, type=Path, help="participants.tsv with participant_id, age, sex")
    ap.add_argument("--require-root", action="append", type=Path, default=[],
                    help="results root (…/atlas-100) every picked subject must have a finished fit in; repeatable")
    ap.add_argument("--n", type=int, default=32, help="number of subjects (default 32 = 8 nodes x 4 GPUs per refit cell)")
    ap.add_argument("--age-split", type=float, default=45.0, help="young < split <= old (as in the population report)")
    ap.add_argument("--max-eeg-loss", type=float, default=1e-4, help="drop failed fits (final EEG-PSD loss above this)")
    ap.add_argument("--min-fc-corr", type=float, default=None, help="optionally require fc_corr above this in every root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("replicate_subjects.txt"))
    args = ap.parse_args()

    part = pd.read_csv(args.participants, sep="\t")
    part["subject"] = part.participant_id.astype(str).str.replace("sub-", "", regex=False)
    cand = part[part.age.notna()][["subject", "age", "sex"]]
    for root in args.require_root:
        f = finished(root)
        ok = f.eeg_loss.isna() | (f.eeg_loss <= args.max_eeg_loss)
        if args.min_fc_corr is not None:
            ok &= f.fc_corr > args.min_fc_corr
        print(f"{root}: {len(f)} finished, {int(ok.sum())} usable")
        cand = cand[cand.subject.isin(f.subject[ok])]
    cand = cand.assign(old=cand.age >= args.age_split)
    print(f"candidates in all roots: {len(cand)} ({int((~cand.old).sum())} young, {int(cand.old.sum())} old)")
    if len(cand) < args.n:
        raise SystemExit(f"only {len(cand)} candidates for --n {args.n}")

    rng = np.random.default_rng(args.seed)
    young, old = cand[~cand.old], cand[cand.old]
    n_old = min(len(old), args.n // 2)
    n_young = min(len(young), args.n - n_old)
    n_old = args.n - n_young
    pick = pd.concat([young.iloc[rng.permutation(len(young))[:n_young]],
                      old.iloc[rng.permutation(len(old))[:n_old]]]).sort_values("subject")
    args.out.write_text("".join(f"{s}\n" for s in pick.subject))
    print(f"wrote {len(pick)} subjects to {args.out}: {n_young} young / {n_old} old, "
          f"sex {pick.sex.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
