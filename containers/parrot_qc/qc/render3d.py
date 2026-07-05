"""3D QC snapshots via pyvista, offscreen (Xvfb + Mesa software GL, no GPU).

Used for geometry QC the 2D overlays can't show: BEM surface nesting, the
tetrahedral mesh interior, electrode/dipole clouds, tractography. Each snapshot
is a multi-panel montage of named anatomical views so a reviewer sees the whole
head at a glance. Mesh readers cover the real on-disk formats: .ply / .vtk / .vtu
/ .stl natively, .mesh (medit) via meshio.

Camera views use the subject world frame (RAS: +x=right, +y=anterior, +z=superior),
so "anterior" is a true face-on view regardless of image orientation. Every added
capability (named views, discrete legends, zoom-to-structure, direction arrows,
PBR/metallic lighting, streamlines) degrades gracefully under OSMesa: a feature
that fails to render is wrapped so the figure still comes out (add_figure then
downgrades a hard failure to a WARN).
"""
from __future__ import annotations

import gzip
import os
import tempfile

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np                        # noqa: E402
import pyvista as pv                      # noqa: E402
import vtk                                # noqa: E402

# VTK is noisy when it probes EGL/GLX before falling back to OSMesa software
# rendering (which is what we use); silence those non-fatal warnings.
vtk.vtkObject.GlobalWarningDisplayOff()

_XVFB = False
_PANEL = (400, 440)  # per-view panel size; window scales with the number of views

# Named cameras: (offset direction of the camera from the scene centre, view-up).
# The camera looks back toward the centre along -offset. RAS world frame.
_CAMERAS = {
    "anterior":  ((0, 1, 0), (0, 0, 1)),    # face-on, from the front
    "posterior": ((0, -1, 0), (0, 0, 1)),   # back of the head
    "left":      ((-1, 0, 0), (0, 0, 1)),   # left side
    "right":     ((1, 0, 0), (0, 0, 1)),    # right side
    "superior":  ((0, 0, 1), (0, 1, 0)),    # top-down, anterior points up
    "inferior":  ((0, 0, -1), (0, 1, 0)),   # bottom-up
}
_DEFAULT_VIEWS = ("left", "anterior", "superior")


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


# --- camera / montage plumbing ----------------------------------------------

def _bounds_of(focus):
    """focus may be a pyvista mesh (use its bounds) or an explicit 6-tuple."""
    if focus is None:
        return None
    if hasattr(focus, "bounds"):
        return list(focus.bounds)
    return list(focus)


def _set_view(pl, view, focus_bounds=None):
    """Point the current subplot's camera at a named anatomical view and fit it to
    either the whole scene or a focus structure's bounds (zoom-to-structure).

    pyvista's view_vector(v) places the camera on the +v side looking back at the
    focal point, so we pass +d (the camera-offset direction) directly -- e.g.
    anterior = camera on +y = a true face-on view."""
    d, up = _CAMERAS.get(view, _CAMERAS["anterior"])
    pl.view_vector((d[0], d[1], d[2]), viewup=up)
    pl.reset_camera(bounds=focus_bounds)  # bounds=None fits all actors


def _multiview(draw, out, views, title=None, legend_items=None, focus_bounds=None,
               panel=_PANEL, view_labels=True):
    """Render `draw(plotter)` from each named view into a 1xN montage screenshot.

    legend_items: optional list of [text, color] drawn as a colour key on the last
    panel. focus_bounds: optional 6-tuple to zoom the cameras onto a structure."""
    _ensure_display()
    views = list(views)
    n = max(1, len(views))
    pl = pv.Plotter(off_screen=True, shape=(1, n), window_size=(panel[0] * n, panel[1]),
                    border=False)
    for i, view in enumerate(views):
        pl.subplot(0, i)
        draw(pl)
        _set_view(pl, view, focus_bounds)
        if title and i == 0:
            pl.add_text(title, font_size=9)
        if view_labels:
            pl.add_text(view, font_size=8, position="lower_left")
        if legend_items and i == n - 1:
            try:
                pl.add_legend(labels=legend_items, bcolor="white", border=True,
                              size=(0.33, 0.33), loc="lower right")
            except Exception:  # noqa: BLE001 - legend is a nicety, never fail the figure
                pass
    pl.screenshot(out)
    pl.close()


def _legend_from_items(items):
    return [[it["label"], it.get("color", "lightgray")] for it in items if it.get("label")]


# --- public snapshot API ----------------------------------------------------

def snapshot_meshes(items, out, title=None, views=_DEFAULT_VIEWS, legend=False, focus=None):
    """items: list of dicts {mesh, color, opacity, style?, label?, pbr?, metallic?,
    roughness?, smooth?}. Renders a named-view montage. `legend=True` builds a colour
    key from the items' `label`s; `focus` (a mesh or bounds tuple) zooms the cameras."""
    def draw(p):
        for it in items:
            p.add_mesh(
                it["mesh"], color=it.get("color", "lightgray"),
                opacity=it.get("opacity", 1.0), style=it.get("style", "surface"),
                smooth_shading=it.get("smooth", True), show_scalar_bar=False,
                pbr=it.get("pbr", False), metallic=it.get("metallic", 0.0),
                roughness=it.get("roughness", 0.5),
            )
    legend_items = _legend_from_items(items) if legend else None
    _multiview(draw, out, views, title, legend_items, _bounds_of(focus))


def snapshot_points(points, out, scalars=None, ref_mesh=None, title=None,
                    cmap="viridis", point_size=6.0, ref_color="wheat", ref_opacity=0.12,
                    views=_DEFAULT_VIEWS, vectors=None, arrow_scale=5.0, arrow_max=1500,
                    legend_items=None, clim=None, ref_style="surface", focus=None):
    """Named-view montage of a point cloud (dipoles/electrodes/sensitivity),
    optionally inside a translucent reference surface (e.g. scalp) and optionally
    with direction arrows (`vectors`, one per point; subsampled to `arrow_max`)."""
    pts = np.asarray(points, dtype=float)
    cloud = pv.PolyData(pts)
    if scalars is not None:
        cloud["s"] = np.asarray(scalars)

    glyphs = None
    if vectors is not None and len(vectors) == len(pts):
        # Arrows only where the vector is non-zero/finite (callers zero-out sources
        # whose orientation is trivial, e.g. surface normals), subsampled AFTER that
        # filter so sparse oriented sources aren't skipped by the stride.
        v = np.asarray(vectors, dtype=float)
        norm = np.linalg.norm(v, axis=1)
        keep = np.where(np.isfinite(norm) & (norm > 1e-6))[0]
        if keep.size:
            step = max(1, keep.size // arrow_max)
            sel = keep[::step]
            vc = pv.PolyData(pts[sel])
            vc["v"] = v[sel]
            try:
                glyphs = vc.glyph(orient="v", scale=False, factor=arrow_scale, geom=pv.Arrow())
            except Exception:  # noqa: BLE001
                glyphs = None

    def draw(p):
        if ref_mesh is not None:
            p.add_mesh(ref_mesh, color=ref_color, opacity=ref_opacity,
                       show_scalar_bar=False, style=ref_style)
        p.add_mesh(
            cloud, scalars="s" if scalars is not None else None,
            cmap=cmap, point_size=point_size, render_points_as_spheres=True,
            show_scalar_bar=False, clim=clim,
        )
        if glyphs is not None:
            p.add_mesh(glyphs, color="black", show_scalar_bar=False)

    # focus=True (or a mesh/bounds) zooms the cameras onto the cloud / structure
    fb = _bounds_of(cloud if focus is True else focus)
    _multiview(draw, out, views, title, legend_items, fb)


def snapshot_fiducials(scalp_mesh, fiducials, out, title=None):
    """One panel per fiducial, camera facing it from outside the head. The scalp is
    opaque with a specular finish and a raking light so gyral ridges/bumps read
    (they anchor the true landmark position); the focused fiducial is a small red
    sphere, the others smaller black. fiducials: {name: [x,y,z]}."""
    _ensure_display()
    names = list(fiducials.keys())
    pts = {k: np.asarray(v, dtype=float) for k, v in fiducials.items()}
    center = np.asarray(scalp_mesh.center, dtype=float)
    n = max(1, len(names))
    pl = pv.Plotter(off_screen=True, shape=(1, n), window_size=(330 * n, 380), border=False)
    for i, name in enumerate(names):
        pl.subplot(0, i)
        # Matte-ish tan with modest specular; pyvista's default light kit already
        # gives enough directional shading for gyral relief. (An extra raking light
        # + near-white wheat blew the exposure out to almost pure white.)
        pl.add_mesh(scalp_mesh, color="tan", opacity=1.0, show_scalar_bar=False,
                    smooth_shading=True, specular=0.2, specular_power=12,
                    ambient=0.15, diffuse=0.7)
        for k, p in pts.items():
            focused = k == name
            pl.add_mesh(pv.Sphere(radius=4.0 if focused else 2.0, center=p),
                        color="red" if focused else "black")
        f = pts[name]
        d = f - center
        d = d / (np.linalg.norm(d) + 1e-9)
        pl.camera_position = [tuple(f + d * 360), tuple(f), (0, 0, 1)]  # pull back (less zoom)
        pl.add_text(name, font_size=13)
    pl.screenshot(out)
    pl.close()


def snapshot_clip(grid_or_path, out, scalars=None, title=None, cmap="tab20",
                  legend_items=None):
    """Orthogonal cross-section of a volume mesh coloured by a cell scalar."""
    _ensure_display()
    grid = grid_or_path if isinstance(grid_or_path, pv.DataSet) else pv.read(str(grid_or_path))
    sliced = grid.slice_orthogonal()
    pl = pv.Plotter(off_screen=True, window_size=(700, 600), border=False)
    pl.add_mesh(sliced, scalars=scalars, cmap=cmap, show_scalar_bar=legend_items is None)
    if legend_items:
        try:
            pl.add_legend(labels=legend_items, bcolor="white", border=True, size=(0.3, 0.4))
        except Exception:  # noqa: BLE001
            pass
    if title:
        pl.add_text(title, font_size=9)
    pl.view_isometric()
    pl.screenshot(out)
    pl.close()


def snapshot_volume_clip(grid_or_path, out, scalars=None, title=None, cmap="tab20",
                         normal="x", legend_items=None, clim=None):
    """Single planar cut of a volume mesh: clip away one half and view the exposed
    tetrahedra on the cut plane (the "bumps" of the interior), coloured by label."""
    _ensure_display()
    grid = grid_or_path if isinstance(grid_or_path, pv.DataSet) else pv.read(str(grid_or_path))
    clipped = grid.clip(normal=normal)
    # Flat shading (lighting off) keeps the tissue colours bright; light-grey edges
    # keep the tetrahedra visible without the black edge-mesh darkening the whole cut.
    # Wide window: the (tall) sagittal section fits to height and centres, leaving a
    # clear right margin for the legend instead of it landing on the head.
    pl = pv.Plotter(off_screen=True, window_size=(1500, 700), border=False)
    pl.add_mesh(clipped, scalars=scalars, cmap=cmap, clim=clim, show_scalar_bar=legend_items is None,
                show_edges=True, edge_color=(0.4, 0.4, 0.4), line_width=0.2, lighting=False)
    if title:
        pl.add_text(title, font_size=9)
    pl.camera_position = "yz" if normal == "x" else ("xz" if normal == "y" else "xy")
    if legend_items:
        try:
            # Head faces +y (right of frame) in this sagittal view, so the occiput
            # (left) side is clear -- put the legend there.
            pl.add_legend(labels=legend_items, bcolor="white", border=True, size=(0.2, 0.5),
                          loc="center left")
        except Exception:  # noqa: BLE001
            pass
    pl.screenshot(out)
    pl.close()


def snapshot_streamlines(tck_path, out, brain_mesh=None, title=None, max_lines=8000,
                         views=_DEFAULT_VIEWS, seed=0):
    """Direction-coloured render of a (subsampled) tractogram inside a translucent
    brain. Handles gzipped .tck; streamlines are in world mm, matching the .ply
    surfaces. Raises if there are no usable streamlines (caller downgrades to WARN)."""
    import nibabel as nib

    p = str(tck_path)
    tmp = None
    try:
        if p.endswith(".gz"):
            tmp = tempfile.NamedTemporaryFile(suffix=".tck", delete=False)
            with gzip.open(p, "rb") as fh:
                tmp.write(fh.read())
            tmp.close()
            load_path = tmp.name
        else:
            load_path = p
        tract = nib.streamlines.load(load_path)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    lines = tract.streamlines
    n = len(lines)
    if n == 0:
        raise ValueError("empty tractogram")
    idx = np.random.default_rng(seed).choice(n, size=min(max_lines, n), replace=False)

    pts, conn, rgb = [], [], []
    offset = 0
    for i in idx:
        s = np.asarray(lines[i], dtype=float)
        if len(s) < 2:
            continue
        m = len(s)
        pts.append(s)
        conn.append(np.hstack([[m], np.arange(offset, offset + m)]))
        d = np.abs(s[-1] - s[0])
        d = d / (np.linalg.norm(d) + 1e-9)          # DEC-style direction colour
        rgb.append(np.tile((d * 255).astype(np.uint8), (m, 1)))
        offset += m
    if not pts:
        raise ValueError("no multi-point streamlines")

    poly = pv.PolyData()
    poly.points = np.vstack(pts)
    poly.lines = np.hstack(conn)
    poly["rgb"] = np.vstack(rgb)

    def draw(p):
        if brain_mesh is not None:
            p.add_mesh(brain_mesh, color="white", opacity=0.1, show_scalar_bar=False)
        p.add_mesh(poly, scalars="rgb", rgb=True, show_scalar_bar=False, line_width=1)

    _multiview(draw, out, views, title)
