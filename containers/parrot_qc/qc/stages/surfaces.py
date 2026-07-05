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
DESCRIPTION = ("The co-registered world-space surfaces that feed dipole placement + meshing, plus the nested BEM layers. Deep structures should sit inside the translucent cortex; the BEM layers must be strictly nested (scalp > outer skull > inner skull > brain) with no intersections.")


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

    # head surfaces montage: translucent cortex shell so the deep structures show;
    # cerebellum + subcortical opaque in distinct colours (one legend entry / group).
    def _head(p):
        items = []
        first = {"cortex": True, "cerebellum": True, "subcortical": True}

        def add(fname, col, op, group):
            f = d / fname
            if f.exists():
                items.append({"mesh": render3d.load_surface(f), "color": col, "opacity": op,
                              "label": group if first[group] else None})
                first[group] = False

        add("freesurfer_lh_pial.ply", "lightpink", 0.12, "cortex")
        add("freesurfer_rh_pial.ply", "lightpink", 0.12, "cortex")
        add("cereb_gray.ply", "tan", 1.0, "cerebellum")
        add("cereb_white.ply", "tan", 1.0, "cerebellum")
        for f in sorted(d.glob("first_*.ply")):
            add(f.name, "mediumpurple", 1.0, "subcortical")
        if items:
            render3d.snapshot_meshes(items, p, title="head surfaces", legend=True,
                                     views=("left", "anterior", "superior"))
    ctx.add_figure(r, "surfaces_head", "Head surfaces (translucent cortex + deep structures)", _head)

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
    layers = [("outer_skin", "wheat", 0.18, "scalp"), ("outer_skull", "lightgray", 0.30, "outer skull"),
              ("inner_skull", "lightblue", 0.45, "inner skull"), ("brain", "pink", 0.9, "brain")]
    for name, col, op, lab in layers:
        sp = bem / f"{name}.surf"
        if sp.exists():
            v, f = nib.freesurfer.read_geometry(str(sp))
            items.append({"mesh": render3d.polydata(v, f), "color": col, "opacity": op, "label": lab})
    if not items:
        result.warn("BEM layers", "no readable scalp/skull/brain surfaces")
        return
    render3d.snapshot_meshes(items, out, title="BEM nesting", legend=True,
                             views=("left", "anterior", "superior"))
