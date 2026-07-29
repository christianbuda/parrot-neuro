"""Per-channel EEG quality control: flags missing/unpositioned/noisy
electrodes in a subject's ``derivatives/EEG`` recordings, with figures and an
HTML report.

Deliberately independent of :mod:`parrot_neuro.optimization` (no jax/tvboptim
dependency) -- install with ``pip install parrot-neuro[eeg_qc]``.

    >>> from parrot_neuro import Subject
    >>> from parrot_neuro.eeg_qc import run_channel_qc
    >>> subject = Subject(bids_root, "010005")
    >>> results = run_channel_qc(subject, output_dir="qc_out")
    >>> results["eyesclosed"].status
    'warn'

See ``examples/eeg_channel_qc.py`` for a full driver script.
"""
from __future__ import annotations

from pathlib import Path

from .data import TaskEEG, discover_tasks, load_task_eeg
from .flags import ChannelFlag, Thresholds, flag_channels, missing_channels
from .metrics import ChannelMetrics, compute_channel_metrics
from .report import TaskQCResult, run_task_qc, write_subject_report

__all__ = [
    "run_channel_qc",
    "TaskEEG", "discover_tasks", "load_task_eeg",
    "ChannelFlag", "Thresholds", "flag_channels", "missing_channels",
    "ChannelMetrics", "compute_channel_metrics",
    "TaskQCResult", "run_task_qc", "write_subject_report",
]


def run_channel_qc(
    subject,
    tasks: list[str] | None = None,
    output_dir: str | Path | None = None,
    expected_channels: list[str] | None = None,
    thresholds: Thresholds | None = None,
) -> dict[str, TaskQCResult]:
    """Run channel QC for every requested task and, if ``output_dir`` is
    given, write ``<output_dir>/sub-<id>/index.html`` + ``channel_qc.json``.

    ``tasks`` defaults to every task this subject has EEG derivatives for
    (``discover_tasks``). ``expected_channels`` is optional: pass a reference
    montage (e.g. another task's channel list, or a dataset-wide cap layout)
    to also flag electrodes that should have been recorded but weren't --
    without it, "missing" only covers channels present in the recording but
    absent from the subject's own electrode-position montage.
    """
    tasks = tasks if tasks is not None else discover_tasks(subject)
    if not tasks:
        raise ValueError(f"{subject.subj}: no EEG task derivatives found under derivatives/EEG/")

    task_eegs = {task: load_task_eeg(subject, task) for task in tasks}
    results = {
        task: run_task_qc(eeg, expected_channels=expected_channels, thresholds=thresholds)
        for task, eeg in task_eegs.items()
    }

    if output_dir is not None:
        out_dir = Path(output_dir) / subject.subj
        write_subject_report(subject.subj, results, task_eegs, out_dir)

    return results
