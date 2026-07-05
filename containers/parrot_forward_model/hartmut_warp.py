"""Ray-cast, layer-normalized source warp — a clean-room Python port of HArtMuT.

Reimplements the projection in HArtMuT's `individualwarp/project_points.jl`
(Harmening et al. 2022, GPL-3.0). Only the *method* is reproduced here — no GPL code is
copied (algorithms are not copyrightable); the template data it consumes is fetched
separately (see `template_data/hartmut/`).

Method (per source point `p`):
  1. Cast a ray from a head-center reference through `p`.
  2. Find where that ray crosses the **template** skull and scalp; `p`'s fractional depth
     between them is `f = dist(p, skull) / (dist(p, skull) + dist(p, scalp))`.
  3. Cast the *same* ray against the **subject** skull and scalp, and place the warped point
     at the same skull->scalp depth fraction `f` on the subject's shell.

Because template and subject meshes must be intersected by one shared ray, this function
assumes they are already in a **common frame** (Parrot co-registers them via the ANTs
subject<->MNI transform before calling this). Sources whose ray misses the subject skull or
scalp — e.g. a subject scan with no neck FOV, whose scalp mesh is open at the bottom — are
**dropped** and reported, which is exactly the intended "no-neck" behavior.

This module is pure geometry (numpy + trimesh); it has no notion of dipoles, tissue labels,
or Parrot's I/O — the caller (`place_artifact_dipoles.py`) owns those.
"""
import numpy as np
import trimesh


def _closest_intersections(mesh, origins, directions, reference_points):
    """For each ray, return the mesh intersection closest to that ray's reference point.

    Batched ray-cast against `mesh`; among all hits of a given ray, pick the one nearest to
    `reference_points[i]` (mirrors HArtMuT's "take intersection close to original point").

    Returns
    -------
    hits : (n_rays, 3) float — closest intersection per ray (NaN row if the ray missed)
    found : (n_rays,) bool — whether the ray hit the mesh at all
    """
    n = len(origins)
    hits = np.full((n, 3), np.nan)
    found = np.zeros(n, dtype=bool)

    # trimesh returns a flat list of intersection locations plus the ray index each belongs to.
    locs, ray_idx, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True)
    if len(ray_idx) == 0:
        return hits, found

    best_dist = np.full(n, np.inf)
    d = np.linalg.norm(locs - reference_points[ray_idx], axis=1)
    # Scan hits, keeping the nearest per ray. (Vectorizing this with np.minimum.at on distance
    # then re-selecting is fiddly because we also need the winning location; a single pass over
    # the hit list — at most a few thousand — is clearer and plenty fast.)
    for loc, ri, dist in zip(locs, ray_idx, d):
        if dist < best_dist[ri]:
            best_dist[ri] = dist
            hits[ri] = loc
            found[ri] = True
    return hits, found


def ray_cast_warp(source_pos, src_skull, src_scalp, tgt_skull, tgt_scalp,
                  center=None, verbose=True):
    """Warp template source positions onto the subject via layer-normalized ray projection.

    Parameters
    ----------
    source_pos : (N, 3) array
        Template source positions, in the shared frame (mm).
    src_skull, src_scalp, tgt_skull, tgt_scalp : trimesh.Trimesh
        Template ("src") and subject ("tgt") skull/scalp meshes, in the shared frame (mm).
        Watertight meshes give the most robust casting; the subject scalp being *open* at a
        clipped neck is precisely what drops out-of-FOV sources.
    center : (3,) array, optional
        Head-center reference the rays emanate from. Defaults to the template skull centroid,
        which lies inside both heads once co-registered.
    verbose : bool
        Print a drop summary.

    Returns
    -------
    warped_pos : (M, 3) array — warped positions for the M kept sources (subject frame, mm).
    keep_mask  : (N,) bool   — which input sources were kept (ray hit all four shells).
    """
    source_pos = np.asarray(source_pos, dtype=np.float64)
    n = len(source_pos)
    if center is None:
        center = np.asarray(src_skull.centroid, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    # One ray per source: from center, through the source, extended well past the scalp so the
    # far shell is always reached (mirrors project_points.jl's `p + 1000*(p - mean_pnt)`).
    directions = source_pos - center
    origins = np.repeat(center[None, :], n, axis=0)

    src_sk, ok_src_sk = _closest_intersections(src_skull, origins, directions, source_pos)
    src_sc, ok_src_sc = _closest_intersections(src_scalp, origins, directions, source_pos)
    tgt_sk, ok_tgt_sk = _closest_intersections(tgt_skull, origins, directions, source_pos)
    tgt_sc, ok_tgt_sc = _closest_intersections(tgt_scalp, origins, directions, source_pos)

    keep = ok_src_sk & ok_src_sc & ok_tgt_sk & ok_tgt_sc

    # Template depth fraction f = dist(p, skull) / (dist(p, skull) + dist(p, scalp)), per kept src.
    d_skull = np.linalg.norm(source_pos - src_sk, axis=1)
    d_scalp = np.linalg.norm(source_pos - src_sc, axis=1)
    denom = d_skull + d_scalp
    # Guard the degenerate case of a source landing exactly on a shell (denom==0 -> mid-shell).
    frac_skull = np.where(denom > 0, d_skull / np.where(denom > 0, denom, 1.0), 0.5)

    # Place at the same fraction along the subject's skull->scalp segment.
    tgt_dir = tgt_sc - tgt_sk
    warped_all = tgt_sk + tgt_dir * frac_skull[:, None]

    warped_pos = warped_all[keep]

    if verbose:
        dropped = int((~keep).sum())
        print(f"[hartmut_warp] kept {int(keep.sum())}/{n} sources; dropped {dropped} "
              f"(ray missed subject shell — e.g. no-neck FOV).")
        if dropped:
            miss_src = int((~(ok_src_sk & ok_src_sc)).sum())
            miss_tgt = int((keep.sum() == 0) and 0 or (~(ok_tgt_sk & ok_tgt_sc) &
                            (ok_src_sk & ok_src_sc)).sum())
            print(f"[hartmut_warp]   template-side misses: {miss_src}; "
                  f"subject-side misses: {miss_tgt}.")

    return warped_pos, keep


def load_mesh(path):
    """Load a surface mesh (STL/PLY/…) as a trimesh, forcing a single Trimesh (not a Scene)."""
    m = trimesh.load_mesh(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m
