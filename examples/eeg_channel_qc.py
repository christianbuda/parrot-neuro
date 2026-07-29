"""Per-channel EEG QC: flags missing/unpositioned/noisy electrodes in a
subject's ``derivatives/EEG`` recordings and writes an HTML report.

Thin driver over ``parrot_neuro.eeg_qc`` -- all the metric/flagging/plotting
logic lives there and is reusable outside this script (e.g. from a notebook).
Needs the ``eeg_qc`` extra (``pip install parrot-neuro[eeg_qc]``; the pixi
dev environment already has scipy/matplotlib/jinja2).

    pixi run python examples/eeg_channel_qc.py /srv/.../parrot_LEMON 010005

Writes ``<output_dir>/sub-<id>/index.html`` (open it in a browser) plus
``channel_qc.json`` and the per-task figures next to it.
"""
from __future__ import annotations

import sys

from parrot_neuro import Subject
from parrot_neuro.eeg_qc import run_channel_qc


def main(bids_root: str, subject_id: str, output_dir: str = "eeg_channel_qc_res") -> None:
    subject = Subject(bids_root, subject_id)
    if not subject.has_eeg:
        sys.exit(f"{subject.subj}: no EEG derivatives found under derivatives/EEG/")

    results = run_channel_qc(subject, output_dir=output_dir)

    for task, result in results.items():
        counts = result.counts()
        print(f"{subject.subj} [{task}]: {result.status.upper()} "
              f"({counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail)")
        if result.missing:
            print(f"  missing: {', '.join(result.missing)}")
        if result.unpositioned:
            print(f"  no montage position: {', '.join(result.unpositioned)}")
        bad = [f.name for f in result.flags if f.status != "pass"]
        if bad:
            print(f"  flagged channels: {', '.join(bad)}")

    print(f"\nReport: {output_dir}/{subject.subj}/index.html")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(f"usage: python {sys.argv[0]} <bids_root> <subject_id> [output_dir]")
    main(*sys.argv[1:])
