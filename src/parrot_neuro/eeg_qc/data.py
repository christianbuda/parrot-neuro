"""Lightweight EEG loading for channel QC -- deliberately independent of
``parrot_neuro.optimization`` (which pulls in jax/tvboptim just to build the
TVB fit). Channel QC never needs gradients, so it reads the same splice-free
``derivatives/EEG`` segments + sidecar JSON directly with numpy/json only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TaskEEG:
    """One subject/task's loaded EEG: raw segments + montage positions."""

    subject: str
    task: str
    sfreq: float
    channel_names: list[str]
    segments: list[np.ndarray]  # each (n_channels, n_samples)
    positions: dict[str, np.ndarray]  # full 10-5 montage, {name: [x, y, z] mm}


def discover_tasks(subject) -> list[str]:
    """EEG task names this subject has derivatives for (e.g. ``['eyesclosed', 'eyesopen']``)."""
    eeg_dir = subject.deriv / "EEG" / subject.subj
    if not eeg_dir.is_dir():
        return []
    tasks = set()
    for f in eeg_dir.glob(f"{subject.subj}_task-*_eeg.json"):
        m = re.search(r"_task-([^_]+)_eeg\.json$", f.name)
        if m:
            tasks.add(m.group(1))
    return sorted(tasks)


def load_task_eeg(subject, task: str) -> TaskEEG:
    """Read one task's splice-free segments + sidecar metadata + electrode montage."""
    npz = subject.load.eeg(task)
    segments = [np.asarray(npz[k], dtype=np.float64) for k in sorted(npz.files)]

    sidecar = subject.path.eeg(task).with_suffix(".json")
    meta = json.loads(Path(sidecar).read_text())

    return TaskEEG(
        subject=subject.subj,
        task=task,
        sfreq=float(meta["sampling_frequency"]),
        channel_names=list(meta["channel_names"]),
        segments=segments,
        positions=subject.load.electrodes(),
    )
