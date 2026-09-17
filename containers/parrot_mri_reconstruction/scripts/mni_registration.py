#!/usr/bin/env python3
"""Subject T1 <-> MNI affine registration (ANTs/antspyx) for EEG-artifact source warping.

Produces a plain 4x4 RAS-world affine mapping MNI152 template coordinates into the subject's
T1/mesh world frame (and its inverse). This is the single shared geometric bridge used by:
  * the muscle-source warp (bring HArtMuT NYhead template geometry into subject space, where
    the ray-cast in hartmut_warp.py then adapts it to the subject's own skull/scalp), and
  * the fallback electrode interpolation (map the subject montage into template space).

Design notes
------------
* **Affine only.** Per the locked design (HArtMuT-faithful "affine bring-into-frame + ray-cast
  shape warp"), we register with an affine and let the ray-cast do the anatomical adaptation.
  The NYhead template lives in the MNI152NLin2009 world frame (verified: its scalp matches the
  MNI152NLin2009cAsym head at the vertex and in x/y; it only extends further down the neck).
* **Convention-proof 4x4 derivation.** ANTs point transforms carry the usual LPS/inverse-order
  gotchas. Rather than trust a hand-derived matrix, we push a spanning set of reference points
  through antspyx's own `apply_transforms_to_points` and least-squares-fit the 4x4 to the result.
  `apply_transforms_to_points` works in **LPS**, so RAS mesh coordinates are flipped into LPS on
  the way in and back to RAS on the way out. Skipping that flip silently yields the LPS matrix,
  which *looks* like a plausible affine but mirrors the head in x/y.
* **Validated, not auto-selected.** We registered fixed=subject, moving=MNI, so the stored affine
  maps subject->MNI and bringing MNI points into subject space needs `whichtoinvert=[True]`. That
  is the only correct direction, so we assert the result instead of scoring candidates: a
  score-the-candidates fallback just picks the least-wrong of two broken matrices.

Outputs (artifacts/registration/sub-<S>/):
  mni_to_subject_affine.npy   (4,4) MNI-world -> subject-T1-world (mm, RAS homogeneous)
  subject_to_mni_affine.npy   (4,4) inverse
  ants_affine.mat             raw ANTs affine transform (for the record / fallback interp)
  registration_qc.json        both scalp residuals (audit trail)
"""
import argparse
import contextlib
import json
import os

import ants
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


# A correct affine puts the NYhead scalp on the subject scalp to ~2.1-2.6 mm; these bound that.
REG_RESIDUAL_WARN_MM = 4.0
REG_RESIDUAL_MAX_MM = 6.0

# RAS <-> LPS is a sign flip on x and y, and is its own inverse.
_LPS_FLIP = np.array([-1.0, -1.0, 1.0])


def ras_lps(pts):
    return np.asarray(pts, dtype=np.float64) * _LPS_FLIP


def lps_ras(pts):
    return np.asarray(pts, dtype=np.float64) * _LPS_FLIP


def add_output_dir(output_dir, *paths):
    return os.path.join(output_dir, *paths)


def load_scalp_vertices(path):
    m = trimesh.load_mesh(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return np.asarray(m.vertices, dtype=np.float64), m


def fit_affine(src, dst):
    """Least-squares 4x4 A with dst ~= A @ [src; 1]. src, dst: (N,3)."""
    n = len(src)
    src_h = np.hstack([src, np.ones((n, 1))])          # (N,4)
    # Solve A[:3] (3x4) from src_h @ A[:3].T = dst
    sol, *_ = np.linalg.lstsq(src_h, dst, rcond=None)  # (4,3)
    A = np.eye(4)
    A[:3, :] = sol.T
    return A


def apply_affine(A, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (pts_h @ A.T)[:, :3]


def points_to_df(pts):
    return pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--output_dir", required=True, help="derivatives root (e.g. /derivatives)")
    ap.add_argument("--t1", required=True, help="subject T1 NIfTI (subject/mesh world frame)")
    ap.add_argument("--template", required=True, help="MNI152NLin2009 T1w template NIfTI")
    ap.add_argument("--template-scalp", required=True,
                    help="NYhead template scalp mesh (MNI frame), for convention auto-select + QC")
    ap.add_argument("--subject-scalp", required=True,
                    help="subject charm_scalp.ply (subject frame), the overlap target")
    args = ap.parse_args()

    out_dir = add_output_dir(args.output_dir, f"artifacts/registration/sub-{args.subject}")
    os.makedirs(out_dir, exist_ok=True)
    # Drop any previous run's affine up front: if the residual gate below rejects this run, a stale
    # (possibly mirrored) matrix must not survive for the dipole/leadfield stages to consume.
    for stale in ("mni_to_subject_affine.npy", "subject_to_mni_affine.npy"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(out_dir, stale))

    fixed = ants.image_read(args.t1)          # subject frame
    moving = ants.image_read(args.template)   # MNI frame
    print("Running affine registration (MNI template -> subject T1) ...")
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine")

    # Reference points spanning the head, in the MNI frame, to derive the 4x4.
    tmpl_verts, _ = load_scalp_vertices(args.template_scalp)
    rng = np.random.default_rng(0)
    ref = tmpl_verts[rng.choice(len(tmpl_verts), size=min(500, len(tmpl_verts)), replace=False)]

    subj_verts, _ = load_scalp_vertices(args.subject_scalp)

    # `apply_transforms_to_points` works in LPS, while every mesh here is in RAS world mm.
    # Feed it LPS and flip the result back, or the fitted 4x4 comes out mirrored in x/y.
    ref_lps = ras_lps(ref)
    mapped = lps_ras(ants.apply_transforms_to_points(
        3, points_to_df(ref_lps), reg["fwdtransforms"], whichtoinvert=[True]).to_numpy())
    A = fit_affine(ref, mapped)                         # MNI -> subject, RAS world mm

    # apply_transforms_to_points is exactly affine here, so the fit must be exact. A nonzero
    # residual means the transform list was not a plain affine and the 4x4 is a lossy summary.
    fit_err = float(np.abs(apply_affine(A, ref) - mapped).max())
    if fit_err > 1e-3:
        raise SystemExit(f"ERROR: 4x4 fit residual {fit_err:.3g} mm — transform is not affine.")

    nyhead_in_subj = apply_affine(A, tmpl_verts)
    # Gate on template->subject: every mapped NYhead vertex should land on the subject scalp.
    # The reverse direction (subject->template) is NOT usable as a check -- it stays at 14-15 mm
    # even for a correct fit, because the subject scalp has detail (ears, nose, neck cut) the
    # smooth NYhead surface lacks, so it barely separates a good registration from a mirrored one.
    d_ts, _ = cKDTree(subj_verts).query(nyhead_in_subj, k=1)
    d_st, _ = cKDTree(nyhead_in_subj).query(subj_verts, k=1)
    err = float(np.mean(d_ts))
    print(f"  scalp residual template->subject = {err:.2f} mm "
          f"(subject->template {np.mean(d_st):.2f} mm, not gated)")

    qc = {
        "scalp_residual_template_to_subject_mm": err,
        "scalp_residual_subject_to_template_mm": float(np.mean(d_st)),
        "affine_fit_residual_mm": fit_err,
        "template": os.path.basename(args.template),
        "note": "MNI152NLin2009 world frame; affine-only bring-into-frame for the ray-cast warp",
    }
    with open(os.path.join(out_dir, "registration_qc.json"), "w") as f:
        json.dump(qc, f, indent=2)

    # Gate BEFORE writing the affine itself. A correct affine lands the NYhead scalp on the subject
    # scalp to ~2-3 mm (measured 2.1-2.6 mm across the 10 AEGEUS subjects); anything near 10 mm is
    # a failed registration or a reintroduced convention bug, either of which puts the warped
    # muscle sources centimetres off. Failing here leaves no affine on disk for a later stage to
    # pick up -- registration_qc.json is already written, so the residual is still auditable.
    if err > REG_RESIDUAL_MAX_MM:
        raise SystemExit(f"ERROR: template->subject scalp residual {err:.2f} mm exceeds "
                         f"{REG_RESIDUAL_MAX_MM} mm — registration failed; artifact sources "
                         f"would be misplaced. No affine written; inspect before use.")
    if err > REG_RESIDUAL_WARN_MM:
        print(f"WARNING: template->subject scalp residual {err:.2f} mm is above the "
              f"{REG_RESIDUAL_WARN_MM} mm typical range; inspect the registration.")

    np.save(os.path.join(out_dir, "mni_to_subject_affine.npy"), A)
    np.save(os.path.join(out_dir, "subject_to_mni_affine.npy"), np.linalg.inv(A))
    # keep the raw ANTs affine for the record / fallback electrode interp
    if reg["fwdtransforms"]:
        import shutil
        shutil.copy(reg["fwdtransforms"][0], os.path.join(out_dir, "ants_affine.mat"))

    print(f"Saved MNI<->subject affine to {out_dir} (template->subject scalp {err:.2f} mm).")


if __name__ == "__main__":
    main()
