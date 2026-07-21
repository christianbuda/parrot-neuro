"""Extract splice-free rest / checkerboard EEG segments from EEGLAB .set files.

Kept separate from ``data`` so the optimization notebook need not import mne.
Used by the extraction notebook to build ``EEG/all_data.pkl``.
"""
from __future__ import annotations

from . import config


def make_arrays(subj, eeg_dir=config.EEG_DIR, task="checker"):
    """Split one subject's continuous recording into rest / trial segments.

    Walks the event timeline as a state machine: markers ``S 10`` (rest) and
    ``S 25`` (trial/checkerboard) toggle state; ``boundary`` events (splices)
    close the current segment so no chunk spans a discontinuity.

    Returns ``(sfreq, channel_names, checkerboard_arrays, rest_arrays)`` where
    each ``*_arrays`` is a list of (n_channels, n_samples) numpy arrays.
    """
    import mne

    print(f"\n\nLoading subject {subj}")
    file_path = f"{eeg_dir}/{subj}_task-{task}_eeg.set"
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    sfreq = raw.info["sfreq"]

    events, event_dict = mne.events_from_annotations(raw)
    rest_id = event_dict["S 10"]
    trial_id = event_dict["S 25"]
    boundary_id = event_dict.get("boundary")

    rest_arrays, checkerboard_arrays = [], []
    current_state = None
    current_start_sample = None

    print("Extracting splice-free continuous matrices...")

    def _save(state, start, stop):
        if state is not None and start is not None and stop > start:
            chunk = raw.get_data(start=start, stop=stop)
            (rest_arrays if state == "rest" else checkerboard_arrays).append(chunk)

    for i in range(len(events)):
        event_sample = events[i, 0]
        event_id = events[i, 2]

        if event_id in (rest_id, trial_id):
            # Condition marker: close the previous block, then switch state.
            _save(current_state, current_start_sample, event_sample)
            current_state = "rest" if event_id == rest_id else "trial"
            current_start_sample = event_sample
        elif event_id == boundary_id:
            # Splice: close the block; same state resumes from here.
            if current_state is not None and current_start_sample is not None:
                _save(current_state, current_start_sample, event_sample)
                current_start_sample = event_sample

    # Grab the final block at the end of the recording.
    _save(current_state, current_start_sample, raw.n_times)

    channel_names = raw.ch_names
    print("\n✅ Success!")
    print(f"Extracted {len(rest_arrays)} Rest and {len(checkerboard_arrays)} Checkerboard arrays.")
    print(f"Checkerboard shapes: {[x.shape for x in checkerboard_arrays]}")
    print(f"Rest shapes: {[x.shape for x in rest_arrays]}\n")

    return sfreq, channel_names, checkerboard_arrays, rest_arrays
