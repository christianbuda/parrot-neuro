"""EEG chunk dataset + a dependency-free (numpy) DataLoader.

The loader is a pure-numpy replacement for ``torch.utils.data`` (torch is not
in this environment); it collates batches into stacked numpy arrays.
"""
from __future__ import annotations

import json

import numpy as np

from .signal import resample_signals


def find_indices(values, reference):
    """Index of each ``values`` entry within ``reference`` (order preserved)."""
    sorter = np.argsort(reference)
    return sorter[np.searchsorted(reference, values, sorter=sorter)]


def load_electrode_names(subject):
    """Electrode names (montage) for ``subject``, from the landmarks CSV."""
    return np.loadtxt(subject.path.electrodes_csv(), delimiter=",", usecols=0, dtype=str)


def load_subject_eeg(subject, task="eyesclosed", chunk_length=None):
    """Build a :class:`SingleSubjectDataset` from the subject's own EEG derivatives.

    Reads the splice-free segments ``subject.load.eeg(task)`` writes (npz of
    ``seg_000``, ``seg_001``, ... (n_channels, n_samples) arrays) plus the
    matching sidecar JSON for ``sampling_frequency``/``channel_names`` —
    replacing the old ad hoc ``all_data.pkl`` + manual rest/task slicing
    (that schema was site-specific; this one is a first-class Parrot output).
    """
    npz = subject.load.eeg(task)
    recordings = [npz[k] for k in sorted(npz.files)]

    sidecar = subject.path.eeg(task).with_suffix(".json")
    meta = json.loads(sidecar.read_text())
    sfreq = meta["sampling_frequency"]
    channel_names = meta["channel_names"]

    return SingleSubjectDataset(
        subject.subject, sfreq, channel_names, recordings, chunk_length=chunk_length,
        electrodes=load_electrode_names(subject),
    )


def numpy_collate(batch):
    """Stack a list of samples into batched numpy arrays (JAX-friendly).

    Recurses into dicts / lists / tuples and stacks the leaves along a new
    axis 0 (a pure-numpy stand-in for torch's default_collate).
    """
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: numpy_collate([sample[key] for sample in batch]) for key in elem}
    if isinstance(elem, (tuple, list)):
        return type(elem)(numpy_collate(list(field)) for field in zip(*batch))
    return np.stack([np.asarray(sample) for sample in batch])


class NumpyLoader:
    """Minimal batching DataLoader over a ``__len__``/``__getitem__`` dataset."""

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.dataset)
        order = np.random.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            batch_idx = order[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            yield numpy_collate([self.dataset[i] for i in batch_idx])

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return -(-n // self.batch_size)  # ceil division


class SingleSubjectDataset:
    """Slices one subject's continuous recordings into fixed-length EEG chunks.

    Each item is a per-chunk-normalized (Channels, chunk_length) array plus the
    channel indices mapping the subject's channels onto the montage order.
    """

    def __init__(self, subj, sfreq, channel_names, eeg_recordings,
                 chunk_length=None, electrodes=None):
        if electrodes is None:
            raise ValueError(
                "electrodes must be given explicitly (e.g. data.load_electrode_names(subject)) "
                "-- there is no longer an implicit single-subject default montage."
            )
        self.electrodes = electrodes
        self.channel_indices = find_indices(channel_names, self.electrodes)
        self.all_recordings = eeg_recordings
        self.sfreq = sfreq
        self.name = subj
        if chunk_length is None:
            chunk_length = 2 * self.sfreq
        self.make_chunks(chunk_length)

    def __len__(self):
        return len(self._chunks)

    def __getitem__(self, idx):
        return {
            "chunk": self._chunks[idx] / np.std(self._chunks[idx]),
            "channel_indices": self.channel_indices,
        }

    def resample(self, target_sfreq):
        self.all_recordings = [
            resample_signals(x, self.sfreq, target_sfreq) for x in self.all_recordings
        ]
        self.sfreq = target_sfreq

    def make_chunks(self, chunk_length):
        self.chunk_length = chunk_length
        self._chunks = []
        for rec in self.all_recordings:
            num_chunks = rec.shape[1] // chunk_length
            if num_chunks > 0:
                self._chunks += np.split(
                    rec[:, : num_chunks * chunk_length], num_chunks, axis=1
                )

    @property
    def chunks(self):
        import jax.numpy as jnp
        return jnp.array(self._chunks)
