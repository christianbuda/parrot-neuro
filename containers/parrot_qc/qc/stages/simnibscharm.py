"""SimNIBS charm QC: final tissue segmentation + tetrahedral head mesh."""
import json

from ..checks import StageResult, PASS, FAIL
from .. import render2d, render3d
from ._common import load_nifti, n_labels

NAME = "simnibscharm"
TITLE = "SimNIBS charm — tissues & mesh"


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("simnibscharm")
    tissues = d / "final_tissues.nii.gz"
    if not tissues.exists():
        return r.skip("no final_tissues.nii.gz")

    img = load_nifti(r, tissues, "final_tissues readable", ndim=3)
    if img is not None:
        r.ok("tissue classes", f"{n_labels(img)} labels")
        if ctx.t1_path().exists():
            ctx.add_figure(r, "charm_tissues", "Final tissue segmentation on T1",
                           lambda p: render2d.roi_overlay(ctx.t1_path(), tissues, p,
                                                          "final_tissues", cmap="tab20"))

    # scalp surface with fiducials, one camera per fiducial (placement sanity)
    scalp = ctx.sfile("surfaces", "charm_scalp.ply")
    fid = ctx.stage_dir("scalplandmarks") / "fiducials.json"
    if scalp.exists() and fid.exists():
        def _fids(p):
            mesh = render3d.load_surface(scalp)
            fiducials = json.loads(fid.read_text())
            render3d.snapshot_fiducials(mesh, fiducials, p, "scalp + fiducials")
        ctx.add_figure(r, "charm_scalp_fiducials", "Scalp + fiducials (one view per landmark)", _fids)
    else:
        r.notes.append("scalp+fiducials figure skipped (charm_scalp.ply or fiducials.json missing)")

    msh = d / "subject.msh"
    if not msh.exists():
        r.warn("head mesh", "subject.msh missing")
        return r
    # count elements cheaply via meshio (subject.msh can be large; just read header/cells)
    try:
        import meshio
        m = meshio.read(str(msh))
        n_tet = sum(len(c.data) for c in m.cells if c.type == "tetra")
        n_tri = sum(len(c.data) for c in m.cells if c.type == "triangle")
        r.add(PASS if n_tet > 10000 else FAIL, "head mesh elements",
              f"{n_tet} tetra, {n_tri} tri, {len(m.points)} nodes")
    except Exception as e:  # noqa: BLE001
        r.fail("head mesh", f"unreadable: {e}")
    return r
