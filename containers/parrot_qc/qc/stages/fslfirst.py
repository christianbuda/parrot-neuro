"""FSL FIRST QC: subcortical structure surfaces (.vtk)."""
import matplotlib.pyplot as plt

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render3d

NAME = "fslfirst"
TITLE = "FSL FIRST — subcortical surfaces"
DESCRIPTION = ("Subcortical structures from FSL FIRST. Each labelled structure should be smooth and sit deep inside the cortex (shown translucent) in its correct bilateral position, without intersecting the cortical surface or its neighbours. If a surface looks wrong (rough, misshapen, mislocated), the error is purely cosmetic — it does not affect the leadfield — and most likely reflects input T1 quality rather than a pipeline fault.")

# 7 bilateral structures + brainstem
_EXPECTED = 15


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("fslfirst")
    vtks = sorted(d.glob("FSL-*_first.vtk")) if d.exists() else []
    if not vtks:
        return r.skip("no FSL-*_first.vtk")

    r.add(PASS if len(vtks) >= _EXPECTED else WARN, "structure count",
          f"{len(vtks)} / {_EXPECTED} expected")

    empty = 0
    for v in vtks:
        try:
            m = render3d.load_surface(v)
            if m.n_points == 0:
                empty += 1
        except Exception as e:  # noqa: BLE001
            r.fail(f"{v.name}", f"unreadable: {e}")
    r.add(FAIL if empty else PASS, "non-empty surfaces",
          f"{len(vtks) - empty}/{len(vtks)} non-empty")

    # Render from the co-registered world-space first_*.ply (the raw fslfirst .vtk
    # is in a different frame, so it wouldn't line up with the cortex hemispheres).
    # Each structure a distinct colour + legend; translucent pial hemispheres give
    # placement context (the structures should sit deep inside the cortex).
    surf_dir = ctx.stage_dir("surfaces")
    plys = sorted(surf_dir.glob("first_*.ply"))
    if plys:
        def _fig(p):
            cmap = plt.get_cmap("tab20")
            items = []
            for i, fp in enumerate(plys):
                items.append({"mesh": render3d.load_surface(fp),
                              "color": tuple(cmap(i % 20)[:3]), "opacity": 1.0,
                              "label": fp.stem.replace("first_", "")})
            for h in ("freesurfer_lh_pial.ply", "freesurfer_rh_pial.ply"):
                hp = surf_dir / h
                if hp.exists():
                    items.append({"mesh": render3d.load_surface(hp),
                                  "color": "lightgray", "opacity": 0.10})
            render3d.snapshot_meshes(items, p, title="FSL FIRST", legend=True,
                                     views=("left", "anterior", "superior"))
        ctx.add_figure(r, "fslfirst_3d",
                       "Subcortical structures in cortex (3D, coloured per structure)", _fig)
    return r
