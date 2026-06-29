"""Cortical segmentation/surfaces QC (FastSurfer-full or FreeSurfer backend).

Checks the volumetric segmentation (aseg / aparc+aseg) and the cortical surface
meshes, and overlays the parcellation on the T1 (segmentation-follows-anatomy).
True surface geometry is shown in the `surfaces` stage; here we confirm the seg
exists, has a plausible label count, and the surfaces are readable & non-empty.
"""
import numpy as np
import nibabel as nib

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render2d, render3d
from ._common import load_nifti, n_labels, first_existing

NAME = "fastsurfer"
TITLE = "Cortical segmentation & surfaces"

_SURFS = ["lh.white", "rh.white", "lh.pial", "rh.pial"]


def _mri_dir(ctx):
    """Backend-agnostic: FastSurfer-full or FreeSurfer both write mri/ + surf/."""
    for backend in ("fastsurfer", "freesurfer"):
        d = ctx.stage_dir(backend)
        if (d / "mri").exists():
            return d
    return None


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    base = _mri_dir(ctx)
    if base is None:
        return r.skip("no fastsurfer/ or freesurfer/ mri tree")
    r.notes.append(f"backend: {base.parent.name}")

    aseg = first_existing(base / "mri" / "aseg.mgz", base / "mri" / "aparc+aseg.mgz")
    aparc = first_existing(base / "mri" / "aparc+aseg.mgz", base / "mri" / "aparc.DKTatlas+aseg.mgz")

    if aseg is not None:
        img = load_nifti(r, aseg, "aseg readable", ndim=3)
        if img is not None:
            nl = n_labels(img)
            r.add(PASS if nl >= 30 else WARN, "aseg label count", f"{nl} labels")

    if aparc is not None and ctx.t1_path().exists():
        ctx.add_figure(r, "fs_aparc_on_t1", "aparc+aseg parcellation on T1",
                       lambda p: render2d.roi_overlay(ctx.t1_path(), aparc, p,
                                                      "aparc+aseg", cmap="gist_ncar"))

    # cortical surfaces (FreeSurfer binary geometry)
    present = 0
    for s in _SURFS:
        sp = base / "surf" / s
        if not sp.exists():
            r.warn(f"surface {s}", "missing")
            continue
        try:
            verts, faces = nib.freesurfer.read_geometry(str(sp))
            ok = verts.size > 0 and np.isfinite(verts).all()
            # Euler characteristic of a closed triangulated surface: chi = V - E + F
            # with E = 3F/2, so chi = V - F/2. FreeSurfer/FastSurfer cortical
            # surfaces are topology-corrected to a sphere (genus 0) -> chi = 2.
            # chi != 2 means a topological defect (handle/hole) survived.
            euler = len(verts) - len(faces) // 2
            status = (FAIL if not ok else (PASS if euler == 2 else WARN))
            r.add(status, f"surface {s}",
                  f"{len(verts)} verts, {len(faces)} faces, Euler chi={euler} "
                  f"({'sphere topology' if euler == 2 else 'expected 2 -- defect?'})")
            present += 1
        except Exception as e:  # noqa: BLE001
            r.fail(f"surface {s}", f"unreadable: {e}")
    if present == 0:
        r.warn("cortical surfaces", "none readable (seg-only backend?)")

    # 3D render of the pial surface (gyral/sulcal pattern, gross-defect check)
    lh_pial, rh_pial = base / "surf" / "lh.pial", base / "surf" / "rh.pial"
    if lh_pial.exists() and rh_pial.exists():
        def _pial(p):
            items = []
            for sp_ in (lh_pial, rh_pial):
                v, f = nib.freesurfer.read_geometry(str(sp_))
                items.append({"mesh": render3d.polydata(v, f), "color": "lightpink",
                              "opacity": 1.0})
            render3d.snapshot_meshes(items, p, "pial surface")
        ctx.add_figure(r, "fs_pial_3d", "Pial surface (3D)", _pial)
    return r
