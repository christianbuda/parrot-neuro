"""FSL FIRST QC: subcortical structure surfaces (.vtk)."""
from ..checks import StageResult, PASS, WARN, FAIL
from .. import render3d

NAME = "fslfirst"
TITLE = "FSL FIRST — subcortical surfaces"

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

    meshes = []
    empty = 0
    for v in vtks:
        try:
            m = render3d.load_surface(v)
            if m.n_points == 0:
                empty += 1
            else:
                meshes.append({"mesh": m, "color": "salmon", "opacity": 1.0})
        except Exception as e:  # noqa: BLE001
            r.fail(f"{v.name}", f"unreadable: {e}")
    r.add(FAIL if empty else PASS, "non-empty surfaces",
          f"{len(vtks) - empty}/{len(vtks)} non-empty")

    if meshes:
        ctx.add_figure(r, "fslfirst_3d", "Subcortical surfaces (3D)",
                       lambda p: render3d.snapshot_meshes(meshes, p, "FSL FIRST"))
    return r
