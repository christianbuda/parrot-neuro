#!/usr/bin/env python3
"""Import LEMON preprocessed EEG into ``derivatives/EEG`` as splice-free segments.

LEMON ships each subject's preprocessed resting-state EEG as an EEGLAB pair
(``sub-<ID>_EC.set``/``.fdt`` = eyes-closed, ``_EO`` = eyes-open). The samples
live in the sidecar ``.fdt``; the ``.set`` is a MATLAB struct that
``mne.io.read_raw_eeglab`` resolves transparently. The recordings are already
condition-separated (no rest/task state machine needed, unlike the tvb-optim
checker prototype) and already at 250 Hz -- the rate the NMM simulator matches,
so no resampling here.

They do, however, carry ``boundary`` events: splices where LEMON's artifact
rejection removed a bad span. A chunk that straddles a splice is not continuous,
so this stage cuts each recording at every boundary into **splice-free
continuous segments** and stores those. It deliberately does NOT chunk -- chunk
length is a free knob chosen later by the optimizer (its ``SingleSubjectDataset``
slices these segments at load time).

Per subject x condition it writes, under ``<dataset>/derivatives/EEG/sub-<ID>/``:

  * ``sub-<ID>_task-<cond>_eeg.npz``  -- one float32 (n_channels, n_samples)
    array per splice-free segment (keys ``seg_000``, ``seg_001``, ...), in Volts.
    Load as ``segs = [z[k] for k in sorted(z)]`` where ``z = np.load(path)``.
  * ``sub-<ID>_task-<cond>_eeg.json`` -- sidecar metadata for the optimizer
    (sampling rate, ordered channel names) + provenance/QC (segment count &
    lengths, total duration, source file).

Runs INSIDE the parrot_mri_reconstruction image (neuro env; the host has no
mne). Launch via bin/stage.sh, pointing <src_dir> at the EEGLAB folder and
<bids_out_dir> at the LEMON dataset root:

    ./bin/stage.sh lemon_eeg \\
        /srv/nfs-data/sisko/christian/LEMON_EEG/EEG_Preprocessed \\
        /srv/nfs-data/sisko/christian/BIDS_LEMON \\
        [sub-010002 ...] [--force] [--min-seg-sec 0.5]

Omit the subject list to import every subject found in <src_dir>.

FIRST-RUN CHECK: this relies on ``mne`` (+ ``pymatreader``, the EEGLAB reader
backend) being present in the neuro env -- verify once before a full batch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SRC_ROOT = Path("/src")                          # LEMON_EEG/EEG_Preprocessed mounted here
DST_ROOT = Path("/dst")                          # the BIDS_LEMON dataset root
EEG_DERIV = DST_ROOT / "derivatives" / "EEG"     # this stage's output tree

# BIDS-ish task label -> the source filename condition code(s) it may appear under.
# LEMON is uniformly EC/EO, except one subject whose eyes-open file is misnamed with a
# zero ("..._E0.set", O->0), so eyes-open accepts that typo alias too. First match wins.
CONDITIONS = {"eyesclosed": ("EC",), "eyesopen": ("EO", "E0")}

BOUNDARY_DESC = "boundary"  # EEGLAB splice marker (an artifact-rejected span was removed here)


def discover_subjects() -> list[str]:
    """All sub-<ID> ids that have at least one EEGLAB file in the source folder."""
    return sorted({p.name.split("_")[0] for p in SRC_ROOT.glob("sub-*_E*.set")})


def read_splicefree_segments(set_path: Path, min_seg_sec: float):
    """Read one EEGLAB recording and split it at boundary events.

    Returns ``(sfreq, channel_names, segments)`` where ``segments`` is a list of
    (n_channels, n_samples) float32 arrays in Volts, each free of splices and at
    least ``min_seg_sec`` seconds long (shorter fragments are dropped).
    """
    import mne

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    data = raw.get_data().astype(np.float32)  # (n_ch, n_times), MNE returns Volts

    # Boundary onsets are seconds from recording start (first_samp == 0 for EEGLAB);
    # the removed span is already gone, so the onset is exactly the splice point.
    boundaries = sorted(
        int(round(onset * sfreq))
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if desc == BOUNDARY_DESC
    )
    cuts = [0, *boundaries, data.shape[1]]
    min_len = int(round(min_seg_sec * sfreq))
    segments = [
        data[:, a:b] for a, b in zip(cuts, cuts[1:]) if (b - a) >= max(min_len, 1)
    ]
    return sfreq, ch_names, segments


def stage_recording(sub: str, task: str, codes: tuple[str, ...], min_seg_sec: float,
                    force: bool) -> None:
    """Import one subject x condition recording (skips if absent or already done)."""
    set_path = next((SRC_ROOT / f"{sub}_{c}.set" for c in codes
                     if (SRC_ROOT / f"{sub}_{c}.set").exists()), None)
    if set_path is None:
        print(f"  {task}: ABSENT -> skipped")
        return
    code = set_path.stem.rsplit("_", 1)[1]  # the code that actually matched (e.g. EO / E0)

    out_dir = EEG_DERIV / sub
    npz_path = out_dir / f"{sub}_task-{task}_eeg.npz"
    json_path = out_dir / f"{sub}_task-{task}_eeg.json"
    if npz_path.exists() and not force:
        print(f"  {task}: exists -> skip (use --force to overwrite)")
        return

    sfreq, ch_names, segments = read_splicefree_segments(set_path, min_seg_sec)
    if not segments:
        print(f"  {task}: WARNING no segments >= {min_seg_sec}s -> nothing written")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    # One named array per splice-free segment -> maps 1:1 onto the optimizer's
    # list-of-continuous-recordings input; avoids pickle (no allow_pickle needed).
    np.savez_compressed(
        npz_path, **{f"seg_{i:03d}": s for i, s in enumerate(segments)}
    )

    lengths = [int(s.shape[1]) for s in segments]
    meta = {
        "condition": task,
        "condition_code": code,
        "sampling_frequency": sfreq,          # Hz; already the sim's target (250)
        "n_channels": len(ch_names),
        "channel_names": ch_names,            # data-row order; optimizer maps -> montage
        "units": "V",                         # MNE scales EEGLAB uV -> V on read
        "n_segments": len(segments),
        "segment_lengths_samples": lengths,
        "total_samples": int(sum(lengths)),
        "total_duration_sec": round(sum(lengths) / sfreq, 3),
        "n_boundaries_removed": len(segments) - 1,
        "source_file": set_path.name,
        "notes": (
            "Splice-free continuous segments (EEGLAB boundary events removed). "
            "Load: z = np.load(npz); segments = [z[k] for k in sorted(z)]. "
            "Chunk downstream at any chunk_length; channel positions come from the "
            "subject's Parrot electrodes, matched by channel_names."
        ),
    }
    json_path.write_text(json.dumps(meta, indent=2))
    print(
        f"  {task}: {len(segments)} segments, {meta['total_duration_sec']}s "
        f"-> {npz_path.name}"
    )


def write_eeg_dataset_description() -> None:
    """Minimal BIDS-derivative dataset_description.json for derivatives/EEG."""
    EEG_DERIV.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": "LEMON preprocessed EEG (splice-free segments for NMM fitting)",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "parrot-neuro staging: lemon_eeg.py"}],
        "SourceDatasets": [{"URL": "LEMON EEG_Preprocessed (EEGLAB .set/.fdt)"}],
    }
    (EEG_DERIV / "dataset_description.json").write_text(json.dumps(desc, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Import LEMON EEG as splice-free segments.")
    ap.add_argument("subjects", nargs="*", help="sub-<ID> ids (default: all in <src_dir>)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument(
        "--min-seg-sec",
        type=float,
        default=0.5,
        help="drop splice-free segments shorter than this many seconds (default: 0.5)",
    )
    args = ap.parse_args()

    subjects = args.subjects or discover_subjects()
    print(f"Importing EEG for {len(subjects)} subject(s) -> {EEG_DERIV}")
    for sub in subjects:
        print(f"\n=== {sub} ===")
        for task, codes in CONDITIONS.items():
            stage_recording(sub, task, codes, args.min_seg_sec, args.force)

    write_eeg_dataset_description()
    print("\nDone.")


if __name__ == "__main__":
    main()
