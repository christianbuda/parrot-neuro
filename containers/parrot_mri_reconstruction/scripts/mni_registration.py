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
  through antspyx's own `apply_transforms_to_points` and least-squares-fit the 4x4 to the
  result. We try both the fwd and inv transform lists and **auto-select** the one whose 4x4,
  applied to the NYhead scalp, best overlaps the subject's real `charm_scalp.ply` — which also
  doubles as an on-the-spot sanity check (a bad registration shows up as large overlap error).

Outputs (registration/sub-<S>/):
  mni_to_subject_affine.npy   (4,4) MNI-world -> subject-T1-world (mm, RAS homogeneous)
  subject_to_mni_affine.npy   (4,4) inverse
  ants_affine.mat             raw ANTs affine transform (for the record / fallback interp)
  registration_qc.json        overlap error + which transform list won (audit trail)
"""
import argparse
import json
import os

import ants
import numpy as np
import pandas as pd
import trimesh


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

    out_dir = add_output_dir(args.output_dir, f"registration/sub-{args.subject}")
    os.makedirs(out_dir, exist_ok=True)

    fixed = ants.image_read(args.t1)          # subject frame
    moving = ants.image_read(args.template)   # MNI frame
    print("Running affine registration (MNI template -> subject T1) ...")
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine")

    # Reference points spanning the head, in the MNI frame, to derive the 4x4.
    tmpl_verts, _ = load_scalp_vertices(args.template_scalp)
    rng = np.random.default_rng(0)
    ref = tmpl_verts[rng.choice(len(tmpl_verts), size=min(500, len(tmpl_verts)), replace=False)]

    subj_verts, _ = load_scalp_vertices(args.subject_scalp)

    # antspyx point convention is fiddly (LPS + inverse order). We registered fixed=subject,
    # moving=MNI, so the stored affine maps subject->MNI; moving MNI points into subject space
    # needs the INVERSE (whichtoinvert=[True]). For an affine reg fwd/invtransforms are the same
    # .mat, so the direction is chosen by whichtoinvert, not by which list. Try both and keep
    # whichever 4x4 lands the NYhead scalp on the subject scalp.
    from scipy.spatial import cKDTree
    candidates = {"invert=True": [True], "invert=False": [False]}
    best = None
    errs = {}
    for name, wti in candidates.items():
        mapped = ants.apply_transforms_to_points(
            3, points_to_df(ref), reg["fwdtransforms"], whichtoinvert=wti).to_numpy()
        A = fit_affine(ref, mapped)                     # MNI -> subject (candidate)
        nyhead_in_subj = apply_affine(A, tmpl_verts)
        # Score subject->NYhead: "is the subject scalp covered by the mapped NYhead scalp?".
        # This direction is robust to NYhead's extra neck (which the subject scan usually lacks
        # and which would inflate the NYhead->subject mean even for a correct fit).
        d, _ = cKDTree(nyhead_in_subj).query(subj_verts, k=1)
        err = float(np.mean(d))
        print(f"  [{name}] mean subject->NYhead scalp NN distance = {err:.2f} mm")
        errs[name] = err
        if best is None or err < best["err"]:
            best = {"err": err, "A": A, "which": name}

    A = best["A"]
    A_inv = np.linalg.inv(A)
    np.save(os.path.join(out_dir, "mni_to_subject_affine.npy"), A)
    np.save(os.path.join(out_dir, "subject_to_mni_affine.npy"), A_inv)
    # keep the raw ANTs affine for the record / fallback electrode interp
    if reg["fwdtransforms"]:
        import shutil
        shutil.copy(reg["fwdtransforms"][0], os.path.join(out_dir, "ants_affine.mat"))

    qc = {
        "selected_transformlist": best["which"],
        "mean_scalp_overlap_mm": best["err"],
        "template": os.path.basename(args.template),
        "note": "MNI152NLin2009 world frame; affine-only bring-into-frame for the ray-cast warp",
    }
    with open(os.path.join(out_dir, "registration_qc.json"), "w") as f:
        json.dump(qc, f, indent=2)

    print(f"Saved MNI<->subject affine to {out_dir} "
          f"(subject->NYhead scalp {best['err']:.2f} mm via {best['which']}).")
    # This residual is dominated by the *shape* difference between the NYhead template scalp and
    # the subject's scalp, which an affine cannot remove (and the ray-cast then corrects radially)
    # -- so a value of ~15-25 mm is normal, not an error. Warn only if it is grossly large (likely
    # a genuinely failed registration) or if the two directions are near-tied (ambiguous choice).
    if best["err"] > 35.0:
        print("WARNING: scalp residual > 35 mm — registration likely failed; inspect before use.")
    ordered = sorted(errs.values())
    if len(ordered) > 1 and ordered[1] - ordered[0] < 3.0:
        print("WARNING: the two transform directions scored within 3 mm — direction choice is "
              "ambiguous; inspect the registration.")


if __name__ == "__main__":
    main()
