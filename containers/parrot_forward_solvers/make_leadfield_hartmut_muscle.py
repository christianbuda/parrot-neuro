#!/usr/bin/env python3
"""Fallback muscle artifact leadfield: HArtMuT's canned leadfield interpolated to the subject.

Used **only** when the subject cannot host warped muscle sources on its own mesh (no neck FOV /
the ray-cast drops most sources). Instead of solving on the subject mesh, we take HArtMuT's
precomputed NYhead muscle leadfield and interpolate its 231-channel electrode dimension onto the
subject's montage.

Interpolation is done in the **template (MNI) frame**: the subject electrodes are mapped into MNI
space via the subject->MNI affine (artifacts/registration/sub-<S>/subject_to_mni_affine.npy), and each
subject electrode's leadfield row is an inverse-distance-weighted blend of its k nearest HArtMuT
electrodes. This is montage-agnostic (no reliance on electrode naming) and position-aware.

The HArtMuT muscle leadfield is `(n_hartmut_elec, n_src, 3)` — free-orientation (3 components per
source), the SAME convention as Parrot's solved artifact leadfields — so the output
`(n_subject_elec, 3 * n_src)` is stackable/consistent with them. Average-referenced to match.

NOTE this path uses HArtMuT's *template* head conductivities/geometry for the source->electrode
physics (only the electrode sampling is subject-specific), so it is a coarser approximation than
the warp+solve path; `artifactsources.json` records when it was used.
"""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree

# A handful of HArtMuT source columns are single-electrode spikes: essentially all of the source's
# energy sits on ONE channel, with a dominant/runner-up amplitude ratio far above what 1/r^2 allows
# for those two electrodes. Measured on the shipped asset: 19/3180 columns, holding ~70% of the
# unit-moment energy, all winning at peripheral electrodes at the edge of the NYhead domain (Nz and
# the neck electrodes Nk1/Nk3/Nk4) -- a boundary artifact, not proximity (they sit 7-34 mm away and
# pass our own clearance rule). Harmless for HArtMuT's own use (one source at a time), but this
# script stacks all 3180 into one leadfield, so they are zeroed. Excluding them brings max/p99 from
# 18.7x to ~2.3x, inside the range the solved artifact leadfields are checked against.
# Both criteria are required: the raw ratio alone flags 46 sources whose two nearest electrodes are
# legitimately at very different distances. Across all sources the 1/r^2 prediction is accurate
# (median observed/expected = 1.01), which is what makes the excess meaningful.
SPIKE_RATIO_MIN = 15.0    # dominant / runner-up channel amplitude
SPIKE_EXCESS_MIN = 3.0    # that ratio, over the (d2/d1)^2 a point dipole would give


def read_subject_electrodes(path):
    """Read Parrot's electrode CSV (name,x,y,z in mm) -> (labels, positions)."""
    labels, pos = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            labels.append(parts[0])
            pos.append([float(v) for v in parts[1:4]])
    return np.array(labels), np.array(pos, dtype=np.float64)


def read_hartmut_electrodes(path):
    """Read the fetched muscle_leadfield_electrodes.csv (header label,x,y,z)."""
    labels, pos = [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            labels.append(parts[0])
            pos.append([float(v) for v in parts[1:4]])
    return np.array(labels), np.array(pos, dtype=np.float64)


def apply_affine(A, pts):
    return (np.hstack([pts, np.ones((len(pts), 1))]) @ A.T)[:, :3]


def find_spike_sources(leadfield, src_pos, elec_pos):
    """Boolean mask of source columns that are single-electrode spikes (see SPIKE_* above).

    Operates on HArtMuT's NATIVE montage, before interpolation -- the IDW blend smears a spike
    across neighbouring subject channels, where it is both harder to detect and more contaminating.
    """
    amp = np.linalg.norm(leadfield, axis=2)                  # (n_elec, n_src)
    order = np.argsort(-amp, axis=0)[:2]
    cols = np.arange(amp.shape[1])
    first, second = amp[order[0], cols], amp[order[1], cols]
    ratio = first / np.maximum(second, np.finfo(float).tiny)
    d1 = np.linalg.norm(elec_pos[order[0]] - src_pos, axis=1)
    d2 = np.linalg.norm(elec_pos[order[1]] - src_pos, axis=1)
    expected = (d2 / np.maximum(d1, 1e-9)) ** 2              # 1/r^2 for those same two electrodes
    return (ratio > SPIKE_RATIO_MIN) & (ratio > SPIKE_EXCESS_MIN * expected)


def avg_ref(mat):
    """Average-reference across the electrode (row) dimension, matching the DUNEuro path."""
    n = mat.shape[0]
    op = -np.ones((n, n)) / n
    op[np.diag_indices_from(op)] = 1 - 1 / n
    return op @ mat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--hartmut-dir", required=True, help="fetched HArtMuT asset cache")
    ap.add_argument("--k", type=int, default=4, help="k nearest HArtMuT electrodes for IDW")
    args = ap.parse_args()

    hm_lf = np.load(os.path.join(args.hartmut_dir, "muscle_leadfield.npy"))  # (n_hm_elec, n_src, 3)
    hm_lab, hm_pos = read_hartmut_electrodes(
        os.path.join(args.hartmut_dir, "muscle_leadfield_electrodes.csv"))
    hm_src = np.load(os.path.join(args.hartmut_dir, "muscle_sources.npy"))

    spikes = find_spike_sources(hm_lf, hm_src, hm_pos)
    if spikes.any():
        won = hm_lab[np.linalg.norm(hm_lf[:, spikes, :], axis=2).argmax(0)]
        hm_lf = hm_lf.copy()
        hm_lf[:, spikes, :] = 0.0
        print(f"Zeroed {int(spikes.sum())}/{len(spikes)} single-electrode spike source column(s); "
              f"dominant channels: {', '.join(sorted(set(won)))}.")

    subj_lab, subj_pos = read_subject_electrodes(
        os.path.join(args.output_dir, f"electrodes/sub-{args.subject}/landmarks_10-5-full.csv"))
    A_s2m = np.load(os.path.join(args.output_dir,
                                 f"artifacts/registration/sub-{args.subject}/subject_to_mni_affine.npy"))
    subj_in_mni = apply_affine(A_s2m, subj_pos)  # map subject electrodes into HArtMuT's frame

    # Inverse-distance-weighted blend of the k nearest HArtMuT electrodes for each subject channel.
    tree = cKDTree(hm_pos)
    k = min(args.k, len(hm_pos))
    dist, idx = tree.query(subj_in_mni, k=k)
    dist = np.atleast_2d(dist.T).T
    idx = np.atleast_2d(idx.T).T
    w = 1.0 / (dist + 1e-6)
    w /= w.sum(axis=1, keepdims=True)                       # (n_subj_elec, k)

    # hm_lf[idx] -> (n_subj_elec, k, n_src, 3); weight over k.
    interp = np.einsum("ek,eksc->esc", w, hm_lf[idx])       # (n_subj_elec, n_src, 3)
    n_elec, n_src, _ = interp.shape
    leadfield = interp.reshape(n_elec, n_src * 3)           # rows=subject montage, free-orientation
    leadfield = avg_ref(leadfield)

    out = os.path.join(args.output_dir, f"leadfields/sub-{args.subject}")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "processed_hartmut_muscle-leadfield.npy"), leadfield)
    print(f"Fallback muscle leadfield: {leadfield.shape} "
          f"({n_elec} subject electrodes x 3*{n_src} sources), avg-referenced, "
          f"IDW-interp (k={k}) from {len(hm_pos)} HArtMuT electrodes.")


if __name__ == "__main__":
    main()
