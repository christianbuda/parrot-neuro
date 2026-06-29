"""Cerebellum QC: template->subject warped labels/surfaces (registration sanity)."""
from ..checks import StageResult, PASS, WARN
from .. import render2d, render3d
from ._common import load_nifti, n_labels

NAME = "cerebellum"
TITLE = "Cerebellum — warped atlas"


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

    surf = d / "nonlinear_Cerebellum_Inner_Surf_With_Features.vtk"
    if surf.exists():
        def _render(p):
            m = render3d.load_surface(surf)
            render3d.snapshot_meshes([{"mesh": m, "color": "tan", "opacity": 1.0}], p,
                                     "cerebellum inner surface")
        ctx.add_figure(r, "cereb_surface", "Warped cerebellar surface (3D)", _render)
    else:
        r.warn("cerebellar surface", "nonlinear inner surface vtk missing")
    return r
