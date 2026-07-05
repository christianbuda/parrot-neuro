#!/usr/bin/env python3
"""Place extra-brain EEG *artifact* source dipoles (eyes + face/neck muscle).

Produces subject-space dipole sets for physiological EEG noise sources, in the same
dipole-contract layout the DUNEuro solver consumes, but under `artifactdipoles/sub-<S>/<group>/`
rather than the brain `dipoles/…/spacing…mm/` tree. Two groups:

* **eyes** — sampled *natively* inside the subject's own eyeball compartment
  (`surfaces/sub-<S>/charm_eyes_balls.ply`). Strictly more subject-specific than warping a
  template eye; the per-eye corneo-retinal axis is stored in `dipole_directions.npy` as metadata
  for the future EOG generator (the free-orientation solver ignores it).

* **muscle** — HArtMuT's template muscle source positions warped into the subject via
  affine-bring-into-frame (`registration/sub-<S>/mni_to_subject_affine.npy`) followed by the
  ray-cast layer-normalized projection (`hartmut_warp.ray_cast_warp`). Sources whose ray misses
  the subject skull/scalp shell (e.g. no neck FOV) are dropped and counted.

Artifacts are a small, fixed source set (spacing-independent), so this runs **once** per subject,
not per dipole spacing. Orientation/amplitude are deferred to the simulation/noise stage; here we
emit only correct **positions** (Parrot solves free-orientation leadfields).

Reuses the pure helpers in `place_dipoles.py` (poisson-disk subsampling) and the warp in
`hartmut_warp.py` — no sampler/writer logic is duplicated.
"""
import argparse
import json
import os

import numpy as np
import trimesh

# Below this fraction of muscle sources surviving the warp (e.g. a scan with no neck FOV, so most
# rays miss the subject scalp), the subject can't host the muscle sources on its own mesh and the
# orchestrator uses HArtMuT's canned muscle leadfield instead of solving.
MUSCLE_NECK_COVERAGE_MIN = 0.5

from hartmut_warp import ray_cast_warp, load_mesh
# Pure helper reuse — importing place_dipoles is side-effect free (its globals/paths are only
# set inside its __main__), so we can borrow its Poisson-disk subsampler.
from place_dipoles import poisson_disk_subsampling


def apply_affine(A, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (pts_h @ A.T)[:, :3]


def decimate_for_raycast(mesh, target_faces=20000):
    """Coarsen a dense mesh for ray-casting (intersection *locations* are insensitive to modest
    decimation, and pure-Python casting cost scales with face count). No-op if already small or if
    the trimesh decimation backend is unavailable."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except TypeError:
        try:
            return mesh.simplify_quadric_decimation(target_faces)  # older trimesh signature
        except Exception:
            return mesh
    except Exception:
        return mesh


def save_group(out_dir, positions, directions, labels, volume=None):
    """Write one artifact source group in the dipole contract the solver reads."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(positions)
    if volume is None:
        volume = np.ones(n)  # geometry-only solve ignores volume; keep the contract populated
    np.save(os.path.join(out_dir, "dipole_positions.npy"), positions.astype(np.float64))
    np.save(os.path.join(out_dir, "dipole_volume.npy"), volume.astype(np.float64))
    np.save(os.path.join(out_dir, "dipole_preferential_direction.npy"), directions.astype(np.float64))
    np.save(os.path.join(out_dir, "dipole_labels.npy"), np.asarray(labels))
    # Benign non-'U' orient type: artifacts never go through neural-strength weighting, but the
    # contract expects a non-Unassigned marker.
    np.save(os.path.join(out_dir, "orient_type.npy"), np.repeat("A", n))
    print(f"  wrote {n} dipoles -> {out_dir}")


# ------------------------------------------------------------------ eyes (native) --------------
def sample_eye_interior(mesh, spacing, generator):
    """Sample points strictly inside a (star-convex) eyeball, then Poisson-disk thin to `spacing`.

    The eyeball meshes are small and not always watertight, so we avoid ray-based
    `mesh.contains` (slow + unreliable on open meshes). Instead we contract the surface vertices
    halfway toward the centroid — for a convex eyeball these are guaranteed interior — and add the
    centroid itself, giving a compact interior point cloud that Poisson-disk thinning turns into
    well-spaced dipoles.
    """
    c = mesh.centroid
    interior = c + 0.5 * (mesh.vertices - c)
    candidates = np.vstack([c[None, :], interior])
    keep = poisson_disk_subsampling(candidates, radius=spacing, generator=generator)
    return candidates[keep]


def place_eyes(output_dir, subject, spacing, generator, head_center):
    eyes_path = os.path.join(output_dir, f"surfaces/sub-{subject}/charm_eyes_balls.ply")
    mesh = load_mesh(eyes_path)
    # Split the two eyeballs (separate connected components).
    components = mesh.split(only_watertight=False)
    components = [c for c in components if len(c.vertices) >= 4] or [mesh]
    # Left/right by centroid x (RAS: +x = right/left depending on convention; label by sign only).
    components = sorted(components, key=lambda c: c.centroid[0])

    all_pos, all_dir, all_lab = [], [], []
    for i, comp in enumerate(components):
        side = "left" if comp.centroid[0] < 0 else "right"
        pos = sample_eye_interior(comp, spacing, generator)
        if len(pos) == 0:
            print(f"  WARNING: no interior points sampled for eye component {i} ({side}).")
            continue
        # Corneo-retinal axis proxy: eyeball centroid pointing away from head center (eyes look
        # outward/forward). Metadata only; the solver ignores dipole orientation.
        axis = comp.centroid - head_center
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        all_pos.append(pos)
        all_dir.append(np.repeat(axis[None, :], len(pos), axis=0))
        all_lab.append(np.repeat(f"Eye_{side}", len(pos)))
        print(f"  eye {side}: {len(pos)} dipoles")

    if not all_pos:
        raise RuntimeError("No eye dipoles could be sampled.")
    positions = np.vstack(all_pos)
    save_group(os.path.join(output_dir, f"artifactdipoles/sub-{subject}/eyes"),
               positions, np.vstack(all_dir), np.concatenate(all_lab))
    return len(positions)


# ------------------------------------------------------------------ muscle (warp) --------------
def place_muscle(output_dir, subject, hartmut_dir, generator, head_center):
    A = np.load(os.path.join(output_dir, f"registration/sub-{subject}/mni_to_subject_affine.npy"))

    src_pos = np.load(os.path.join(hartmut_dir, "muscle_sources.npy"))
    src_lab = np.load(os.path.join(hartmut_dir, "muscle_labels.npy"), allow_pickle=True)

    # Bring template geometry into the subject frame with the affine (consistent for sources AND
    # meshes, so the ray-cast depth fraction is computed in a self-consistent template geometry).
    tmpl_skull = load_mesh(os.path.join(hartmut_dir, "nyhead_skull.stl"))
    tmpl_scalp = load_mesh(os.path.join(hartmut_dir, "nyhead_scalp.stl"))
    tmpl_skull.vertices = apply_affine(A, tmpl_skull.vertices)
    tmpl_scalp.vertices = apply_affine(A, tmpl_scalp.vertices)
    src_pos = apply_affine(A, src_pos)

    subj_skull = load_mesh(os.path.join(output_dir, f"surfaces/sub-{subject}/freesurfer_BEM_outer_skull.ply"))
    subj_scalp = load_mesh(os.path.join(output_dir, f"surfaces/sub-{subject}/charm_scalp.ply"))
    subj_scalp = decimate_for_raycast(subj_scalp)  # 105k-face charm scalp -> ~20k for fast casting

    warped, keep = ray_cast_warp(src_pos, tmpl_skull, tmpl_scalp, subj_skull, subj_scalp,
                                 center=head_center)
    kept_lab = np.asarray(src_lab)[keep]
    # Placeholder orientation = outward radial (metadata only).
    radial = warped - head_center
    radial /= (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12)

    save_group(os.path.join(output_dir, f"artifactdipoles/sub-{subject}/muscle"),
               warped, radial, kept_lab)
    return int(keep.sum()), int((~keep).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--output_dir", required=True, help="derivatives root (e.g. /derivatives)")
    ap.add_argument("--hartmut-dir", required=True, help="HArtMuT asset cache (fetch_hartmut.py dest)")
    ap.add_argument("--groups", nargs="+", default=["eyes", "muscle"], choices=["eyes", "muscle"])
    ap.add_argument("--eye-spacing", type=float, default=3.0, help="eye dipole spacing (mm)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    generator = np.random.default_rng(args.seed)

    # Head-center reference shared by the eye axis and the muscle ray-cast: the subject skull
    # centroid (inside the head, frame-consistent with the warp's template meshes after affine).
    subj_skull = load_mesh(os.path.join(args.output_dir,
                                        f"surfaces/sub-{args.subject}/freesurfer_BEM_outer_skull.ply"))
    head_center = np.asarray(subj_skull.centroid, dtype=np.float64)

    summary = {}
    if "eyes" in args.groups:
        print("Placing eye dipoles (native)...")
        n_eyes = place_eyes(args.output_dir, args.subject, args.eye_spacing, generator, head_center)
        summary["eyes"] = {"n_dipoles": int(n_eyes)}
    if "muscle" in args.groups:
        print("Placing muscle dipoles (warp)...")
        kept, dropped = place_muscle(args.output_dir, args.subject, args.hartmut_dir,
                                     generator, head_center)
        total = kept + dropped
        frac = kept / total if total else 0.0
        # neck_coverage=False signals the orchestrator to use the canned-leadfield fallback:
        # too few muscle sources survived the warp for a subject-specific solve to be meaningful.
        summary["muscle"] = {"n_kept": int(kept), "n_dropped": int(dropped),
                             "n_total": int(total), "kept_fraction": round(frac, 4),
                             "neck_coverage": bool(frac >= MUSCLE_NECK_COVERAGE_MIN)}
        print(f"Muscle: kept {kept}, dropped {dropped} (no-neck / ray-miss); "
              f"neck_coverage={summary['muscle']['neck_coverage']}.")

    # Machine-readable record the orchestrator reads to pick the muscle path (solve vs fallback).
    out_json = os.path.join(args.output_dir, f"artifactdipoles/sub-{args.subject}/artifactsources.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
