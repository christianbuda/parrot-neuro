"""Dipoles QC: sampled source positions/orientations per spacing."""
import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render3d

NAME = "dipoles"
TITLE = "Dipoles — source sampling"
DESCRIPTION = ("Sampled source dipoles, coloured by anatomical compartment (cerebrum / "
               "cerebellum / hippocampus / subcortical). Sources should fill the grey matter "
               "and subcortical/cerebellar volumes uniformly at the target spacing; arrows show "
               "the volumetric sources' preferential directions.")

# per-source subdir name -> compartment (for colouring the aggregated cloud)
_SOURCE_COMPARTMENT = {
    "freesurfer_lh_middle": "cerebrum", "freesurfer_rh_middle": "cerebrum",
    "cereb_inner_processed": "cerebellum",
    "hippunfold_L_dentate_middle": "hippocampus", "hippunfold_R_dentate_middle": "hippocampus",
    "hippunfold_L_hipp_middle": "hippocampus", "hippunfold_R_hipp_middle": "hippocampus",
}


def _key(p):
    return (round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3))


def _compartment_labels(sdir, pos):
    """Compartment per aggregated dipole. The aggregated cloud is the exact
    concatenation of the per-source dipole_positions (surfaces/<mesh>/ + volumetric/),
    so we match each position back to its source by coordinate. Returns str array."""
    lut = {}
    surf = sdir / "surfaces"
    if surf.exists():
        for d in surf.iterdir():
            comp = _SOURCE_COMPARTMENT.get(d.name)
            f = d / "dipole_positions.npy"
            if comp and f.exists():
                for p in np.load(f):
                    lut[_key(p)] = comp
    vf = sdir / "volumetric" / "dipole_positions.npy"
    if vf.exists():
        for p in np.load(vf):
            lut[_key(p)] = "subcortical"
    return np.array([lut.get(_key(q), "other") for q in pos])


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

    # 3D scatter coloured by anatomical compartment (cerebrum / cerebellum /
    # hippocampus / subcortical), recovered by matching each aggregated dipole back
    # to the per-source file it was concatenated from. Direction arrows show the
    # volumetric sources' preferential orientation (surface normals are omitted --
    # obvious and cloud-burying).
    comp = _compartment_labels(sdir, pos)
    _COLORS = {"cerebrum": "#1f77b4", "cerebellum": "#ff7f0e", "hippocampus": "#2ca02c",
               "subcortical": "#d62728", "other": "#7f7f7f"}
    present = [c for c in ("cerebrum", "cerebellum", "hippocampus", "subcortical", "other")
               if np.any(comp == c)]
    idx = {c: i for i, c in enumerate(present)}
    scal = np.array([idx[c] for c in comp], dtype=float)
    cmap = [_COLORS[c] for c in present]
    clim = (-0.5, len(present) - 0.5)
    legend_items = [[c, _COLORS[c]] for c in present]

    ot = np.load(ot_f, allow_pickle=True).astype(str) if (ot_f := sdir / "orient_type.npy").exists() else None
    dirs_for_arrows = None
    if dirs_f.exists():
        dirs_for_arrows = np.load(dirs_f).astype(float).copy()
        if ot is not None and len(ot) == len(dirs_for_arrows):
            dirs_for_arrows[ot == "N"] = 0.0

    ctx.add_figure(r, f"dipoles_{tag}", f"Dipole cloud ({tag}) — coloured by compartment",
                   lambda p: render3d.snapshot_points(
                       pos, p, scalars=scal, ref_mesh=None, focus=True,
                       views=("left", "anterior", "superior"), vectors=dirs_for_arrows,
                       arrow_scale=8.0, arrow_max=2500, point_size=1.5, cmap=cmap, clim=clim,
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
