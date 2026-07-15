"""Tetmesh QC: the CGAL FEM mesh — element count, quality, label cross-section."""
import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render2d, render3d

NAME = "tetmesh"
TITLE = "Tetrahedral FEM mesh"
DESCRIPTION = ("The CGAL tetrahedral FEM mesh. The planar section should show clean, well-shaped tetrahedra with tissue labels in the right places, and no inverted or zero-volume elements (see the volume histogram). A few slivers (tiny but non-zero tets) are expected and warn rather than fail; they are usually one localized cluster at a thin tissue interface and do not measurably affect the leadfield.")

# Sliver fraction above which the mesh is considered systemically bad rather than
# carrying the handful of slivers CGAL's time-limited perturb/exude always leaves.
SLIVER_FRAC_FAIL = 1e-5


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

    # element volumes -> inverted / degenerate / sliver detection (tets only).
    # The three conditions below are disjoint and graded separately: they are
    # different failure modes with different consequences for the FEM solve.
    try:
        if n_tet == 0:
            raise ValueError("no tetrahedra to measure")
        vol = np.asarray(grid.compute_cell_sizes(length=False, area=False,
                                                 volume=True).cell_data["Volume"])[tet_mask]
        absvol = np.abs(vol)

        # Winding can make ALL signed volumes negative -- that's a sign convention,
        # not degeneracy. A *mix* of signs is genuine inversion, which corrupts the
        # stiffness assembly; the minority sign is the inverted population.
        n_inv = min(int((vol > 0).sum()), int((vol < 0).sum()))
        r.add(PASS if n_inv == 0 else FAIL, "element orientation",
              f"{n_inv}/{n_tet} inverted tets" + ("" if n_inv == 0 else " (mixed winding)"))

        # Exactly-zero tets contribute nothing to the stiffness matrix -> always fatal.
        n_zero = int((absvol == 0).sum())
        r.add(PASS if n_zero == 0 else FAIL, "degenerate elements",
              f"{n_zero}/{n_tet} tets with exactly zero volume")

        # Slivers (tiny but non-zero). cell_radius_edge_ratio -- the mesher's only
        # shape criterion -- provably does not exclude slivers, and perturb/exude run
        # under a time limit, so a small surviving population is expected rather than
        # a defect: a localized cluster degrades conditioning but the solve absorbs it.
        # Only a systemic population means the mesh itself is bad, so grade by fraction.
        tiny = max(1e-9, float(np.median(absvol)) * 1e-6)
        n_sliver = int(((absvol <= tiny) & (absvol > 0)).sum())
        frac = n_sliver / n_tet
        r.add(PASS if n_sliver == 0 else (WARN if frac <= SLIVER_FRAC_FAIL else FAIL),
              "sliver elements",
              f"{n_sliver}/{n_tet} ({frac:.2g}) tets with 0 < |volume| <= {tiny:.2g} mm³")

        # log-x: volumes span ~11 orders of magnitude, so a sliver is invisible in a
        # linear bin next to the median. The threshold marker shows what the sliver
        # check graded on -- anything left of the line is a counted sliver.
        ctx.add_figure(r, "tetmesh_volume_hist", "Element volume distribution",
                       lambda p: render2d.histogram(absvol, p,
                                                    "tet volumes", "|volume| (mm³)",
                                                    logy=True, logx=True, vline=tiny,
                                                    vline_label=f"sliver threshold ({tiny:.2g})"))
    except Exception as e:  # noqa: BLE001
        r.warn("element volumes", f"could not compute: {e}")

    # label cross-section: a single planar cut showing the interior tetrahedra,
    # coloured by tissue label with a legend from the tetmesh labels.txt.
    lk = _int_cell_key(grid)
    clip_grid = grid
    scal_key = lk
    palette = clim = legend_items = None
    if lk is not None:
        # Section only the tetrahedra (drop the boundary-facet triangles, whose marker
        # labels aren't tissues). The .vtu tet label encodes tissue + 10*group
        # (range 1..39); the tissue is recovered by (label-1)%10+1 -- verified to
        # reproduce the .mesh medit:ref tissue counts the solver uses.
        clip_grid = grid.extract_cells(tet_mask) if n_tet else grid
        labels = np.asarray(clip_grid.cell_data[lk])
        tissue = ((labels - 1) % 10) + 1
        clip_grid.cell_data["tissue"] = tissue
        scal_key = "tissue"
        uniq = np.unique(tissue)
        r.add(PASS, "tissue labels", f"{uniq.size} tissues (from cell array '{lk}')")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from ._common import read_label_table
        table = dict(read_label_table(ctx.stage_dir("tetmesh") / "labels.txt"))
        cmap0 = plt.get_cmap("tab10")
        hi = int(uniq.max())
        colors = [tuple(cmap0(i % 10)[:3]) for i in range(hi + 1)]  # indexed by tissue id
        palette = ListedColormap(colors)
        clim = (0.5, hi + 0.5)
        legend_items = [[table[int(v)], colors[int(v)]] for v in uniq if int(v) in table]
    ctx.add_figure(r, "tetmesh_clip", "Mesh planar section by tissue",
                   lambda p: render3d.snapshot_volume_clip(clip_grid, p, scalars=scal_key,
                                                           normal="x", cmap=palette or "tab20",
                                                           clim=clim, legend_items=legend_items,
                                                           title="tet mesh tissues"))
    return r
