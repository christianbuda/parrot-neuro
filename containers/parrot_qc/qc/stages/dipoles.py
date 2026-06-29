"""Dipoles QC: sampled source positions/orientations per spacing."""
import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render3d
from .electrodes import _scalp_mesh

NAME = "dipoles"
TITLE = "Dipoles — source sampling"


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

    # 3D scatter, coloured by aggregated atlas label if available
    scal = None
    agg = sdir / "aggregated_dipole_labels.npy"
    if agg.exists():
        try:
            scal = np.load(agg).astype(float)
            if len(scal) != n:
                scal = None
        except Exception:  # noqa: BLE001
            scal = None
    scalp = _scalp_mesh(ctx)
    ctx.add_figure(r, f"dipoles_{tag}", f"Dipole cloud ({tag})",
                   lambda p: render3d.snapshot_points(pos, p, scalars=scal, ref_mesh=scalp,
                                                      title=f"dipoles {tag}", point_size=5,
                                                      cmap="tab20"))


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("dipoles")
    spacings = sorted(d.glob("spacing*mm")) if d.exists() else []
    if not spacings:
        return r.skip("no dipoles/spacing*mm")
    for sdir in spacings:
        _check_spacing(ctx, r, sdir, sdir.name.replace("spacing", ""))
    return r
