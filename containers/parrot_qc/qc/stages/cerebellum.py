"""Cerebellum QC: template->subject warped labels/surfaces (registration sanity)."""
from ..checks import StageResult, PASS, WARN
from .. import render2d, render3d
from ._common import load_nifti, n_labels, first_existing

NAME = "cerebellum"
TITLE = "Cerebellum — warped atlas"
DESCRIPTION = ("Template->subject warped cerebellar atlas + surface. The cerebellum should nest snugly below/behind the cerebrum (shown translucent) and its labels should follow the folia -- a registration sanity check.")


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("cerebellum")
    gray = d / "nonlinear_gray_labels.nii.gz"
    if not gray.exists():
        return r.skip("no nonlinear_gray_labels.nii.gz")

    img = load_nifti(r, gray, "nonlinear gray labels", ndim=3)
    if img is not None:
        r.add(PASS, "cerebellar regions", f"{n_labels(img)} labels")
        if ctx.t1_path().exists():
            ctx.add_figure(r, "cereb_labels_on_t1", "Cerebellar labels on T1 (registration)",
                           lambda p: render2d.roi_overlay(ctx.t1_path(), gray, p,
                                                          "cerebellum", cmap="gist_ncar"))
    white = d / "nonlinear_white_labels.nii.gz"
    if white.exists():
        load_nifti(r, white, "nonlinear white labels", ndim=3)

    # 3D from the co-registered world-space cereb_*.ply, zoomed onto the cerebellum,
    # with a translucent cerebrum for registration context (the cerebellum should
    # sit snugly below/behind the occipital lobes).
    surf_dir = ctx.stage_dir("surfaces")
    cereb = first_existing(surf_dir / "cereb_gray.ply",
                           surf_dir / "cereb_inner_processed.ply",
                           surf_dir / "cereb_inner.ply")
    if cereb is not None:
        def _render(p):
            cm = render3d.load_surface(cereb)
            items = [{"mesh": cm, "color": "tan", "opacity": 1.0, "label": "cerebellum"}]
            brain = surf_dir / "freesurfer_BEM_brain.ply"
            if brain.exists():
                items.append({"mesh": render3d.load_surface(brain), "color": "lightgray",
                              "opacity": 0.10, "label": "cerebrum"})
            render3d.snapshot_meshes(items, p, title="cerebellum", legend=True, focus=cm,
                                     views=("left", "anterior", "superior"))
        ctx.add_figure(r, "cereb_surface", "Warped cerebellar surface (3D, zoomed)", _render)
    else:
        r.warn("cerebellar surface", "no surfaces/cereb_*.ply")
    return r
