#!/usr/bin/env python3
"""Muscle artifact leadfield columns taken from HArtMuT's canned leadfield. Two modes.

**`--mode template-block`** (the normal path). A subject's head model is usually cut off below the
mouth, so the muscle sources belonging to chin, jaw and neck -- and on a short FOV the whole
perioral group -- have no anatomy to sit on. `place_artifact_dipoles.py` therefore splits them off
into `muscle_template/` rather than warping them onto the scalp's bottom rim. This mode builds
their leadfield columns from HArtMuT's canned leadfield, which *does* model a full head with jaw
and neck, and **stacks them onto the solved block** so the muscle leadfield stays one array:

    processed_duneuro_artifact-muscle-CGAL-leadfield.npy
        columns [0, 3*n_solved)                     solved on the subject mesh
        columns [3*n_solved, 3*(n_solved+n_template))   from the template, rescaled

**`--mode fallback`** (unchanged). Used when the subject cannot host warped muscle sources at all
(`solve_muscle: false`): there is no solve, so all 3180 sources come from the canned leadfield, at a
nominal scale, written to its own file.

Interpolation (both modes) is done in the **template (MNI) frame**: the subject electrodes are
mapped into MNI space via `subject_to_mni_affine.npy`, and each subject electrode's leadfield row is
an inverse-distance-weighted blend of its k nearest HArtMuT electrodes. Montage-agnostic (no
reliance on electrode naming) and position-aware.

The HArtMuT muscle leadfield is `(n_hartmut_elec, n_src, 3)` -- free-orientation, the SAME convention
as Parrot's solved artifact leadfields -- so the result is stackable with them. Average-referenced to
match.

NOTE the template columns use HArtMuT's *template* head conductivities and geometry for the
source->electrode physics (only the electrode sampling is subject-specific), so they are a coarser
approximation than the solved ones; the sidecar records the split and the calibration.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

# A handful of HArtMuT source columns are single-electrode spikes: essentially all of the source's
# energy sits on ONE channel, with a dominant/runner-up amplitude ratio far above what 1/r^2 allows
# for those two electrodes. Measured on the shipped asset: 19/3180 columns, holding ~70% of the
# unit-moment energy, all winning at peripheral electrodes at the edge of the NYhead domain (Nz and
# the neck electrodes Nk1/Nk3/Nk4) -- a boundary artifact, not proximity (they sit 7-34 mm away and
# pass our own clearance rule). Harmless for HArtMuT's own use (one source at a time), but this
# script stacks many sources into one leadfield, so they are zeroed. Excluding them brings max/p99
# from 18.7x to ~2.3x, inside the range the solved artifact leadfields are checked against.
# Both criteria are required: the raw ratio alone flags 46 sources whose two nearest electrodes are
# legitimately at very different distances. Across all sources the 1/r^2 prediction is accurate
# (median observed/expected = 1.01), which is what makes the excess meaningful.
SPIKE_RATIO_MIN = 15.0    # dominant / runner-up channel amplitude
SPIKE_EXCESS_MIN = 3.0    # that ratio, over the (d2/d1)^2 a point dipole would give

# Fewer paired sources than this and the per-subject calibration is not worth trusting; fall back to
# the nominal scale. Real subjects yield ~2900 pairs, so this only fires on a broken solve.
MIN_CALIBRATION_PAIRS = 100

# Nominal solved/canned amplitude ratio, used when there is no solve to calibrate against. Median of
# the per-subject factors measured on AEGEUS P001/P004/P010 (2.85e6, 3.61e6, 3.62e6); sub-MNI09b
# gives 5.59e6 but it is a template, not a head. The factor varies ~2x between subjects, which is
# why the calibrated path is preferred whenever a solve exists.
NOMINAL_SCALE = 3.6e6


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


def interpolate_to_subject(hm_lf, hm_pos, subj_pos, A_s2m, k):
    """Canned leadfield on the subject montage, average-referenced -> (n_subj_elec, n_src, 3)."""
    subj_in_mni = apply_affine(A_s2m, subj_pos)              # into HArtMuT's frame
    k = min(k, len(hm_pos))
    dist, idx = cKDTree(hm_pos).query(subj_in_mni, k=k)
    dist, idx = np.atleast_2d(dist.T).T, np.atleast_2d(idx.T).T
    w = 1.0 / (dist + 1e-6)
    w /= w.sum(axis=1, keepdims=True)                        # (n_subj_elec, k)
    # hm_lf[idx] -> (n_subj_elec, k, n_src, 3); weight over k.
    interp = np.einsum("ek,eksc->esc", w, hm_lf[idx])
    n_elec, n_src, _ = interp.shape
    return avg_ref(interp.reshape(n_elec, n_src * 3)).reshape(n_elec, n_src, 3)


def source_footprints(leadfield_3d):
    """Per-source column norm: one number for how loud each source is across the whole montage."""
    return np.linalg.norm(leadfield_3d, axis=(0, 2))


def calibrate(solved_lf, canned_sub, idx_solved, spikes):
    """Amplitude factor converting canned columns into the solved block's units.

    The canned leadfield covers ALL the template's muscle sources, including the ones we solved
    ourselves, so every solved source is the same source computed twice -- once by DUNEuro on the
    subject mesh, once by HArtMuT on the NYhead template. The ratio of the two column norms is the
    conversion factor; we take the median over the pairs.

    Both sides are compared on the subject montage, average-referenced, because a column norm
    depends on which electrodes you measure at and on the reference: against HArtMuT's raw 231-channel
    leadfield the same subjects give a factor 1.3-1.9x different, which would be baked in silently.

    No filtering by how far the warp moved a source: measured on four subjects, restricting to
    sources that barely moved shifts the factor by 1-13%, against a within-subject IQR of ~2x.
    """
    n_solved = solved_lf.shape[1] // 3
    fp_solved = source_footprints(solved_lf.reshape(solved_lf.shape[0], n_solved, 3))
    fp_canned = source_footprints(canned_sub[:, idx_solved, :])
    ok = ~spikes[idx_solved] & (fp_canned > 0) & (fp_solved > 0)
    if int(ok.sum()) < MIN_CALIBRATION_PAIRS:
        return NOMINAL_SCALE, {"source": "nominal", "n_pairs": int(ok.sum()),
                               "reason": f"fewer than {MIN_CALIBRATION_PAIRS} usable pairs"}
    lg = np.log10(fp_solved[ok] / fp_canned[ok])
    q1, q2, q3 = np.percentile(lg, [25, 50, 75])
    return float(10 ** q2), {"source": "paired", "n_pairs": int(ok.sum()),
                             "iqr_factor": round(float(10 ** (q3 - q1)), 3),
                             "p5_p95_factor": round(float(10 ** np.ptp(np.percentile(lg, [5, 95]))), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--hartmut-dir", required=True, help="fetched HArtMuT asset cache")
    ap.add_argument("--mode", choices=["template-block", "fallback"], default="fallback",
                    help="template-block: build the below-FOV columns and stack them onto the "
                         "solved leadfield. fallback: all sources from the canned leadfield.")
    ap.add_argument("--k", type=int, default=4, help="k nearest HArtMuT electrodes for IDW")
    args = ap.parse_args()

    subject, output_dir = args.subject, args.output_dir
    adip = os.path.join(output_dir, f"artifacts/dipoles/sub-{subject}")
    lf_dir = os.path.join(output_dir, f"leadfields/sub-{subject}")

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
        os.path.join(output_dir, f"electrodes/sub-{subject}/landmarks_10-5-full.csv"))
    A_s2m = np.load(os.path.join(output_dir,
                                 f"artifacts/registration/sub-{subject}/subject_to_mni_affine.npy"))
    canned_sub = interpolate_to_subject(hm_lf, hm_pos, subj_pos, A_s2m, args.k)
    n_elec = len(subj_pos)
    os.makedirs(lf_dir, exist_ok=True)

    if args.mode == "fallback":
        leadfield = canned_sub.reshape(n_elec, -1)   # already average-referenced by the interp
        out = os.path.join(lf_dir, "processed_hartmut_muscle-leadfield.npy")
        np.save(out, leadfield)
        print(f"Fallback muscle leadfield: {leadfield.shape} "
              f"({n_elec} subject electrodes x 3*{canned_sub.shape[1]} sources), avg-referenced, "
              f"IDW-interp (k={args.k}) from {len(hm_pos)} HArtMuT electrodes.")
        return

    # ---------------------------------------------------------------- template-block mode -----
    lf_path = os.path.join(lf_dir, "processed_duneuro_artifact-muscle-CGAL-leadfield.npy")
    idx_solved = np.load(os.path.join(adip, "muscle", "template_index.npy"))
    idx_template = np.load(os.path.join(adip, "muscle_template", "template_index.npy"))
    solved_lf = np.load(lf_path)

    # Guards the in-place stack below: if this array already has both blocks, a rerun would stack
    # the template columns a second time. The solved leadfield is the only valid input here.
    if solved_lf.shape[1] != 3 * len(idx_solved):
        raise SystemExit(
            f"{lf_path} has {solved_lf.shape[1]} columns, expected {3 * len(idx_solved)} for the "
            f"{len(idx_solved)} solved sources. It has probably already been stacked -- rerun the "
            "artifacts leadfield solve (delete its log) before building the template block.")

    scale, calib = calibrate(solved_lf, canned_sub, idx_solved, spikes)
    print(f"Calibration ({calib['source']}, n={calib['n_pairs']} paired sources): "
          f"solved/canned = {scale:.3e}" +
          (f", IQR x{calib['iqr_factor']}" if "iqr_factor" in calib else ""))

    block = canned_sub[:, idx_template, :].reshape(n_elec, -1) * scale
    n_dead = int((source_footprints(canned_sub[:, idx_template, :]) == 0).sum())
    stacked = np.hstack([solved_lf, block])
    np.save(lf_path, stacked)

    sidecar = {
        "n_solved": int(len(idx_solved)), "n_template": int(len(idx_template)),
        "columns_solved": [0, 3 * int(len(idx_solved))],
        "columns_template": [3 * int(len(idx_solved)), int(stacked.shape[1])],
        "calibration_scale": scale,
        "calibration": calib,
        "n_template_spike_zeroed": n_dead,
        "units": "template columns rescaled into the solved block's units",
        # The factor is measured on sources near the electrodes; the template sources are far below
        # every one of them, so it is applied outside the range where it was fitted. Extrapolating
        # the observed distance trend instead was tested and rejected -- it barely improves the fit
        # where measured but changes the answer ~2x where it would be applied.
        "template_block_out_of_calibration_range": True,
    }
    with open(lf_path.replace(".npy", ".json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Stacked muscle leadfield: {stacked.shape} = {len(idx_solved)} solved + "
          f"{len(idx_template)} template sources"
          + (f" ({n_dead} template columns zeroed as spikes)" if n_dead else "") + ".")
    print(f"  -> {lf_path}")


if __name__ == "__main__":
    main()
