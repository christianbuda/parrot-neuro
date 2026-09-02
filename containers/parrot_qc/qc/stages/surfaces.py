"""Surfaces QC: world-space .ply surfaces + BEM layer nesting.

The aggregated world-space surfaces (.ply) feed dipole placement and meshing, and
the BEM layers (scalp ⊃ outer skull ⊃ inner skull ⊃ brain) feed the OpenMEEG
forward model. The key visual is the 3D nesting check: the layers must be properly
contained, not intersecting.
"""
import numpy as np
import nibabel as nib
import trimesh

from ..checks import StageResult, PASS, WARN, FAIL, SKIP
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

    _check_bem_nesting(r, d, ctx)

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


# Shells the OpenMEEG BEM actually solves on, innermost first.
_BEM_SHELLS = [("freesurfer_BEM_brain.ply", "brain"),
               ("freesurfer_BEM_inner_skull.ply", "inner_skull"),
               ("freesurfer_BEM_outer_skull.ply", "outer_skull"),
               ("freesurfer_BEM_outer_skin.ply", "outer_skin")]
# Only the inner shells are FOV-checked: on a head truncated at the skull base the scalp
# and skull legitimately continue past the image edge, whereas the brain is enclosed by
# the skull, so a brain shell outside the image was fabricated in empty padding.
_FOV_CHECKED = ("brain", "inner_skull")
# A shell may sit marginally below the floor without harm (watershed flattens it against
# the edge). Threshold is a heuristic: observed 1-4 mm on a head whose BEM solves fine vs
# 18-29 mm on one whose brain shell ran away.
_FOV_TOL_MM = 5.0
# The solver re-repairs the decimated shells before solving, which clears small
# violations; only larger ones actually cost the BEM leadfield. Also a heuristic,
# calibrated the same way (2.3 mm repaired successfully, 15 mm did not).
_NEST_FAIL_MM = 5.0


def _check_bem_nesting(result, surf_dir, ctx):
    """Geometrically verify the BEM shells are strictly nested and inside the FOV.

    The montage below is only a picture; without this the pipeline's one signal for a
    non-nested head is a cryptic om_assemble crash in the solver log. Containment is
    tested in BOTH directions and on every pair -- "all of the outer shell's vertices
    are outside the inner one" is not containment, and the adjacent-pair chain assumes
    a transitivity that does not hold once any pair is broken.
    """
    shells = []
    for fname, name in _BEM_SHELLS:
        p = surf_dir / fname
        if p.exists():
            try:
                shells.append((name, trimesh.load(p)))
            except Exception:  # noqa: BLE001
                result.warn("BEM nesting", f"could not read {fname}")
                return
    if len(shells) < 2:
        result.add(SKIP, "BEM nesting", "no BEM shells to check")
        return

    t1 = ctx.t1_path()
    if t1.exists():
        img = nib.load(str(t1))
        sh = np.array(img.shape[:3])
        corners = np.array([[i, j, k, 1] for i in (0, sh[0]-1) for j in (0, sh[1]-1) for k in (0, sh[2]-1)])
        w = (img.affine @ corners.T).T[:, :3]
        lo, hi = w.min(0), w.max(0)
        escapes = []
        for name, m in shells:
            if name not in _FOV_CHECKED:
                continue
            v = np.asarray(m.vertices)
            depth = float(np.maximum(lo - v, v - hi).max(1).max())
            if depth > _FOV_TOL_MM:
                escapes.append(f"{name} by {depth:.1f} mm")
        result.add(WARN if escapes else PASS, "BEM shells within the image FOV",
                   ("outside the T1 FOV: " + ", ".join(escapes) +
                    " -- watershed fitted these where there is no data")
                   if escapes else "all shells inside the acquired field of view")

    bad, worst = [], 0.0
    for i, (ni, mi) in enumerate(shells):
        for no, mo in shells[i+1:]:
            ins = trimesh.proximity.signed_distance(mo, mi.vertices)
            out = -trimesh.proximity.signed_distance(mi, mo.vertices)
            depth = max(-float(ins.min()), -float(out.min()))
            if (ins < 0).any() or (out < 0).any():
                worst = max(worst, depth)
                bad.append(f"{ni} vs {no}: {int((ins < 0).sum())} {ni} verts up to "
                           f"{-ins.min():.1f} mm outside")
    if not bad:
        result.add(PASS, "BEM layer nesting",
                   f"{len(shells)} shells strictly nested (all pairs, both directions)")
    elif worst < _NEST_FAIL_MM:
        result.add(WARN, "BEM layer nesting", "; ".join(bad) +
                   " -- small enough that the solver's post-decimation repair should clear it")
    else:
        result.add(FAIL, "BEM layer nesting", "; ".join(bad) +
                   " -- too large to repair, so OpenMEEG refuses the geometry and the "
                   "BEM leadfield is skipped (FEM leadfields unaffected)")
