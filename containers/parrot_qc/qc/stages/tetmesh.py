"""Tetmesh QC: the CGAL FEM mesh — element count, quality, label cross-section."""
import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render2d, render3d

NAME = "tetmesh"
TITLE = "Tetrahedral FEM mesh"


def _int_cell_key(grid):
    for k in grid.cell_data:
        if np.asarray(grid.cell_data[k]).dtype.kind in "iu":
            return k
    return None


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("tetmesh")
    vtu = d / "tetrahedral_mesh.vtu"
    mesh = d / "tetrahedral_mesh.mesh"
    src = vtu if vtu.exists() else (mesh if mesh.exists() else None)
    if src is None:
        return r.skip("no tetrahedral_mesh.{vtu,mesh}")

    import pyvista as pv
    try:
        grid = pv.read(str(vtu)) if vtu.exists() else render3d.load_surface(mesh)
    except Exception as e:  # noqa: BLE001
        r.fail("mesh readable", f"{e}")
        return r

    # The .vtu mixes boundary triangles (VTK type 5) with tetrahedra (type 10);
    # assess only the tets (triangles legitimately have zero volume).
    VTK_TETRA = 10
    tet_mask = np.asarray(grid.celltypes) == VTK_TETRA
    n_tet = int(tet_mask.sum())
    r.add(PASS if n_tet > 10000 else WARN, "element count",
          f"{n_tet} tetrahedra, {grid.n_points} nodes ({grid.n_cells} cells incl. facets)")

    # element volumes -> degenerate/inverted detection (tets only)
    try:
        vol = np.asarray(grid.compute_cell_sizes(length=False, area=False,
                                                 volume=True).cell_data["Volume"])
        # Winding can make ALL signed volumes negative -- that's a sign
        # convention, not degeneracy. Only near-zero |volume| is a bad element.
        absvol = np.abs(vol[tet_mask]) if n_tet else np.abs(vol)
        tiny = max(1e-9, float(np.median(absvol)) * 1e-6)
        n_bad = int((absvol <= tiny).sum())
        r.add(PASS if n_bad == 0 else FAIL, "non-degenerate elements",
              f"{n_bad}/{n_tet} tets with |volume| <= {tiny:.2g} mm³")
        ctx.add_figure(r, "tetmesh_volume_hist", "Element volume distribution",
                       lambda p: render2d.histogram(absvol, p,
                                                    "tet volumes", "|volume| (mm³)", logy=True))
    except Exception as e:  # noqa: BLE001
        r.warn("element volumes", f"could not compute: {e}")

    # label cross-section
    lk = _int_cell_key(grid)
    if lk is not None:
        labels = np.asarray(grid.cell_data[lk])
        r.add(PASS, "tissue labels", f"{np.unique(labels).size} classes (cell array '{lk}')")
    ctx.add_figure(r, "tetmesh_clip", "Mesh cross-section by tissue label",
                   lambda p: render3d.snapshot_clip(grid, p, scalars=lk,
                                                    title="tet mesh labels"))
    return r
