"""Dipoles QC: sampled source positions/orientations per spacing."""
import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render3d
from .electrodes import _scalp_mesh

NAME = "dipoles"
TITLE = "Dipoles — source sampling"
DESCRIPTION = ("Sampled source dipoles, coloured by orientation model (surface-normal vs volumetric gradient/principal-axis/random). Sources should fill the grey matter and subcortical/cerebellar volumes uniformly at the target spacing; arrows show the volumetric sources' preferential directions.")


def _check_spacing(ctx, r, sdir, tag):
    pos_f = sdir / "dipole_positions.npy"
    if not pos_f.exists():
        r.warn(f"{tag} positions", "dipole_positions.npy missing")
        return
    pos = np.load(pos_f)
    ok = pos.ndim == 2 and pos.shape[1] == 3 and np.isfinite(pos).all()
    r.add(PASS if ok else FAIL, f"{tag} positions", f"{len(pos)} dipoles, shape={pos.shape}")
    n = len(pos)

    dirs_f = sdir / "dipole_directions.npy"
    if dirs_f.exists():
        dirs = np.load(dirs_f)
        norms = np.linalg.norm(dirs, axis=1)
        unit = np.allclose(norms[np.isfinite(norms)], 1.0, atol=1e-2)
        same = len(dirs) == n
        r.add(PASS if (unit and same and np.isfinite(dirs).all()) else WARN,
              f"{tag} directions",
              f"shape={dirs.shape}, |dir| mean={np.nanmean(norms):.3f} (expect ~1)")

    vol_f = sdir / "dipole_volume.npy"
    if vol_f.exists():
        vol = np.load(vol_f)
        neg = int((vol < 0).sum())
        zeros = int((vol == 0).sum())
        ok = np.isfinite(vol).all() and neg == 0 and len(vol) == n
        r.add(PASS if ok else WARN, f"{tag} per-dipole volume",
              f"min={np.nanmin(vol):.3g}, max={np.nanmax(vol):.3g} mm³, "
              f"zeros={zeros}, negative={neg}")

    # 3D scatter coloured by orientation type -- the source model, not the region:
    #   N = surface normal (fixed; cortex/cerebellum/hippocampus surfaces)
    #   G = gradient, P = principal axis, R = random  (all volumetric)
    # Direction arrows show each source's preferential orientation.
    scal = None
    legend_items = None
    cmap = "tab10"
    clim = None
    ot = None
    ot_f = sdir / "orient_type.npy"
    if ot_f.exists():
        try:
            ot = np.load(ot_f, allow_pickle=True).astype(str)
            if len(ot) == n:
                names = {"N": "surface normal", "G": "gradient",
                         "P": "principal axis", "R": "random", "U": "unassigned"}
                palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
                present = [c for c in ["N", "G", "P", "R", "U"] if np.any(ot == c)]
                idx = {c: i for i, c in enumerate(present)}
                scal = np.array([idx.get(c, 0) for c in ot], dtype=float)
                cmap = palette[:len(present)]
                clim = (-0.5, len(present) - 0.5)
                legend_items = [[names.get(c, c), palette[i]] for i, c in enumerate(present)]
        except Exception:  # noqa: BLE001
            scal = None

    # Arrows only for the volumetric sources (G/P/R): their preferential direction is
    # the non-trivial one. Surface-normal (N) sources are zeroed out so their (obvious,
    # and cloud-burying) normals don't clutter the figure.
    dirs_for_arrows = None
    if dirs_f.exists():
        dirs_for_arrows = np.load(dirs_f).astype(float).copy()
        if ot is not None and len(ot) == len(dirs_for_arrows):
            dirs_for_arrows[ot == "N"] = 0.0
    scalp = _scalp_mesh(ctx)
    ctx.add_figure(r, f"dipoles_{tag}", f"Dipole cloud ({tag}) — coloured by orientation type",
                   lambda p: render3d.snapshot_points(
                       pos, p, scalars=scal, ref_mesh=scalp, ref_opacity=0.15,
                       views=("left", "anterior", "superior"), vectors=dirs_for_arrows,
                       arrow_scale=8.0, arrow_max=2500, point_size=2.5, cmap=cmap, clim=clim,
                       legend_items=legend_items, title=f"dipoles {tag}"))


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("dipoles")
    spacings = sorted(d.glob("spacing*mm")) if d.exists() else []
    if not spacings:
        return r.skip("no dipoles/spacing*mm")
    for sdir in spacings:
        _check_spacing(ctx, r, sdir, sdir.name.replace("spacing", ""))
    return r
