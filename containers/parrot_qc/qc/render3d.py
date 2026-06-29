"""3D QC snapshots via pyvista, offscreen (Xvfb + Mesa software GL, no GPU).

Used for geometry QC the 2D overlays can't show: BEM surface nesting, the
tetrahedral mesh exterior, electrode placement on the scalp, and dipole clouds.
Each snapshot is a 3-panel (left / front / top) montage so a reviewer sees the
whole head at a glance. Mesh readers cover the real on-disk formats: .ply / .vtk
/ .vtu / .stl natively, .mesh (medit) via meshio.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np                        # noqa: E402
import pyvista as pv                      # noqa: E402
import vtk                                # noqa: E402

# VTK is noisy when it probes EGL/GLX before falling back to OSMesa software
# rendering (which is what we use); silence those non-fatal warnings.
vtk.vtkObject.GlobalWarningDisplayOff()

_XVFB = False
_WIN = (1200, 440)


def _ensure_display():
    global _XVFB
    if not _XVFB:
        try:
            pv.start_xvfb()
        except Exception:
            pass  # may already have a display; rendering try/except will catch real failures
        _XVFB = True


def polydata(verts, faces):
    """Build a pyvista surface from vertex (N,3) + triangular face (M,3) arrays."""
    f = np.asarray(faces, dtype=np.int64)
    flat = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel()
    return pv.PolyData(np.asarray(verts, dtype=float), flat)


def load_surface(path):
    """Read a surface/volume mesh into a pyvista object."""
    p = str(path)
    if p.endswith(".mesh"):
        import meshio
        m = meshio.read(p)
        cells = m.cells_dict
        if "triangle" in cells:
            tri = cells["triangle"]
            faces = np.hstack([np.full((len(tri), 1), 3), tri]).ravel()
            return pv.PolyData(m.points, faces)
        # tetra mesh -> extract exterior surface
        grid = pv.from_meshio(m)
        return grid.extract_surface()
    return pv.read(p)


def _add_views(plotter, draw_fn, title=None):
    """Populate a (1,3) plotter with left / front / top cameras of the same scene."""
    cams = ["yz", "xz", "xy"]
    for i, cam in enumerate(cams):
        plotter.subplot(0, i)
        draw_fn(plotter)
        plotter.camera_position = cam
        if i == 0 and title:
            plotter.add_text(title, font_size=9)


def snapshot_meshes(items, out, title=None):
    """items: list of dicts {mesh: PolyData, color, opacity, style?}.
    Renders a 3-view montage and screenshots to `out`."""
    _ensure_display()
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=_WIN, border=False)

    def draw(p):
        for it in items:
            p.add_mesh(
                it["mesh"], color=it.get("color", "lightgray"),
                opacity=it.get("opacity", 1.0), style=it.get("style", "surface"),
                smooth_shading=True, show_scalar_bar=False,
            )

    _add_views(pl, draw, title)
    pl.screenshot(out)
    pl.close()


def snapshot_points(points, out, scalars=None, ref_mesh=None, title=None,
                    cmap="viridis", point_size=6.0, ref_color="wheat", ref_opacity=0.12):
    """3-view montage of a point cloud (dipoles/electrodes), optionally inside a
    translucent reference surface (e.g. scalp) for spatial context."""
    _ensure_display()
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    if scalars is not None:
        cloud["s"] = np.asarray(scalars)
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=_WIN, border=False)

    def draw(p):
        if ref_mesh is not None:
            p.add_mesh(ref_mesh, color=ref_color, opacity=ref_opacity, show_scalar_bar=False)
        p.add_mesh(
            cloud, scalars="s" if scalars is not None else None,
            cmap=cmap, point_size=point_size, render_points_as_spheres=True,
            show_scalar_bar=False,
        )

    _add_views(pl, draw, title)
    pl.screenshot(out)
    pl.close()


def snapshot_fiducials(scalp_mesh, fiducials, out, title=None):
    """One panel per fiducial, camera facing it from outside the head, the focused
    fiducial drawn as a large red sphere (others small black) with its name written
    on the panel. fiducials: {name: [x,y,z]}."""
    _ensure_display()
    names = list(fiducials.keys())
    pts = {k: np.asarray(v, dtype=float) for k, v in fiducials.items()}
    center = np.asarray(scalp_mesh.center, dtype=float)
    n = max(1, len(names))
    pl = pv.Plotter(off_screen=True, shape=(1, n), window_size=(330 * n, 380), border=False)
    for i, name in enumerate(names):
        pl.subplot(0, i)
        pl.add_mesh(scalp_mesh, color="wheat", opacity=0.5, show_scalar_bar=False,
                    smooth_shading=True)
        for k, p in pts.items():
            focused = k == name
            pl.add_mesh(pv.Sphere(radius=7 if focused else 3.5, center=p),
                        color="red" if focused else "black")
        f = pts[name]
        d = f - center
        d = d / (np.linalg.norm(d) + 1e-9)
        pl.camera_position = [tuple(f + d * 260), tuple(f), (0, 0, 1)]
        pl.add_text(name, font_size=13)
    pl.screenshot(out)
    pl.close()


def snapshot_clip(grid_or_path, out, scalars=None, title=None, cmap="tab20"):
    """Cross-section of a volume mesh coloured by a cell scalar (e.g. tet labels)."""
    _ensure_display()
    grid = grid_or_path if isinstance(grid_or_path, pv.DataSet) else pv.read(str(grid_or_path))
    sliced = grid.slice_orthogonal()
    pl = pv.Plotter(off_screen=True, window_size=(700, 600), border=False)
    pl.add_mesh(sliced, scalars=scalars, cmap=cmap, show_scalar_bar=True)
    if title:
        pl.add_text(title, font_size=9)
    pl.view_isometric()
    pl.screenshot(out)
    pl.close()
