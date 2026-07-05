#!/usr/bin/env python3
"""Fallback muscle artifact leadfield: HArtMuT's canned leadfield interpolated to the subject.

Used **only** when the subject cannot host warped muscle sources on its own mesh (no neck FOV /
the ray-cast drops most sources). Instead of solving on the subject mesh, we take HArtMuT's
precomputed NYhead muscle leadfield and interpolate its 231-channel electrode dimension onto the
subject's montage.

Interpolation is done in the **template (MNI) frame**: the subject electrodes are mapped into MNI
space via the subject->MNI affine (registration/sub-<S>/subject_to_mni_affine.npy), and each
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

    subj_lab, subj_pos = read_subject_electrodes(
        os.path.join(args.output_dir, f"electrodes/sub-{args.subject}/landmarks_10-5-full.csv"))
    A_s2m = np.load(os.path.join(args.output_dir,
                                 f"registration/sub-{args.subject}/subject_to_mni_affine.npy"))
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
