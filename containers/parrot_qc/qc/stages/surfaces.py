"""Surfaces QC: world-space .ply surfaces + BEM layer nesting.

The aggregated world-space surfaces (.ply) feed dipole placement and meshing, and
the BEM layers (scalp ⊃ outer skull ⊃ inner skull ⊃ brain) feed the OpenMEEG
forward model. The key visual is the 3D nesting check: the layers must be properly
contained, not intersecting.
"""
import numpy as np
import nibabel as nib

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render3d

NAME = "surfaces"
TITLE = "Surfaces — world space & BEM nesting"


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("surfaces")
    plys = sorted(d.glob("*.ply")) if d.exists() else []
    if not plys:
        return r.skip("no surfaces/*.ply")

    bad = 0
    total_v = 0
    for p in plys:
        try:
            m = render3d.load_surface(p)
            if m.n_points == 0:
                bad += 1
            total_v += m.n_points
        except Exception:  # noqa: BLE001
            bad += 1
    r.add(FAIL if bad else PASS, "world-space .ply surfaces",
          f"{len(plys)} files, {total_v} verts total, {bad} unreadable/empty")

    # head surfaces montage: cortex + cerebellum + subcortical
    def _head(p):
        items = []
        for name, col in [("freesurfer_lh_pial.ply", "lightpink"),
                          ("freesurfer_rh_pial.ply", "lightpink"),
                          ("cereb_white.ply", "tan"),
                          ("cereb_inner.ply", "tan")]:
            f = d / name
            if f.exists():
                items.append({"mesh": render3d.load_surface(f), "color": col, "opacity": 1.0})
        for f in sorted(d.glob("first_*.ply")):
            items.append({"mesh": render3d.load_surface(f), "color": "salmon", "opacity": 1.0})
        if items:
            render3d.snapshot_meshes(items, p, "head surfaces")
    ctx.add_figure(r, "surfaces_head", "Head surfaces (cortex + cerebellum + subcortical)", _head)

    # BEM nesting (FreeSurfer-format layers under fastsurfer/bem + scalp from npy)
    bem = ctx.stage_dir("fastsurfer") / "bem"
    if not bem.exists():
        bem = ctx.stage_dir("freesurfer") / "bem"
    if bem.exists():
        ctx.add_figure(r, "bem_nesting", "BEM layer nesting (scalp ⊃ skull ⊃ brain)",
                       lambda p: _render_bem(r, bem, p))
    else:
        r.warn("BEM layers", "no bem/ directory found")
    return r


def _render_bem(result, bem, out):
    items = []
    layers = [("outer_skin", "wheat", 0.18), ("outer_skull", "lightgray", 0.30),
              ("inner_skull", "lightblue", 0.45), ("brain", "pink", 0.9)]
    # scalp may live as vertices/faces .npy; skull/brain as FreeSurfer .surf
    vs, fs = bem / "vertices-scalp.npy", bem / "faces-scalp.npy"
    if vs.exists() and fs.exists():
        items.append({"mesh": render3d.polydata(np.load(vs), np.load(fs)),
                      "color": "wheat", "opacity": 0.18})
    for name, col, op in layers:
        sp = bem / f"{name}.surf"
        if sp.exists():
            v, f = nib.freesurfer.read_geometry(str(sp))
            items.append({"mesh": render3d.polydata(v, f), "color": col, "opacity": op})
    if not items:
        result.warn("BEM layers", "no readable scalp/skull/brain surfaces")
        return
    render3d.snapshot_meshes(items, out, "BEM nesting")
