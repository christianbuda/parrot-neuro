"""Electrodes QC: 10-5 electrode placement + scalp fiducials.

Coordinates live in landmarks_10-5-full.csv (headerless: name,x,y,z);
selected_landmarks_10-5-full.json is the *list of names* kept after excluding
ears/eyes; fiducials.json maps NAS/LPA/RPA/IN -> [x,y,z].
"""
import csv
import json

import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render3d

NAME = "electrodes"
TITLE = "Electrodes & fiducials"
DESCRIPTION = ("10-5 electrode positions + fiducials on the scalp. Electrodes should sit on the scalp in a regular cap layout; selected (red) exclude ears/eyes; fiducials should match NAS/LPA/RPA.")


def _scalp_mesh(ctx):
    """The reference scalp for the 3D overlays (electrodes, dipoles, leadfields,
    artifacts all share this). Prefer the SimNIBS charm scalp -- it's the clean
    surface the forward model actually uses (electrode placement, artifact warp,
    MNI reg). The MNE dense scalp (bem/*-scalp.npy) is a QC-only fallback and, on
    MP2RAGE, mkheadsurf wraps the residual background noise into a box."""
    charm = ctx.stage_dir("surfaces") / "charm_scalp.ply"
    if charm.exists():
        try:
            return render3d.load_surface(charm)
        except Exception:  # noqa: BLE001 - fall back to the MNE scalp below
            pass
    for backend in ("fastsurfer", "freesurfer"):
        bem = ctx.stage_dir(backend) / "bem"
        vs, fs = bem / "vertices-scalp.npy", bem / "faces-scalp.npy"
        if vs.exists() and fs.exists():
            return render3d.polydata(np.load(vs), np.load(fs))
    return None


def _read_csv_coords(path):
    """Headerless name,x,y,z -> {name: [x,y,z]} (rows that don't parse are skipped)."""
    coords = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) >= 4:
                try:
                    coords[row[0].strip()] = [float(row[1]), float(row[2]), float(row[3])]
                except ValueError:
                    continue
    return coords


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    edir = ctx.stage_dir("electrodes")
    fid = ctx.stage_dir("scalplandmarks") / "fiducials.json"
    if not edir.exists() and not fid.exists():
        return r.skip("no electrodes/ or scalplandmarks/")

    # fiducials
    if fid.exists():
        try:
            fids = json.loads(fid.read_text())
            keys = {k.lower() for k in fids}
            need = {"nas", "lpa", "rpa"}
            r.add(PASS if need <= keys else WARN, "fiducials",
                  f"{len(fids)} points; NAS/LPA/RPA={'yes' if need <= keys else 'missing'}")
        except Exception as e:  # noqa: BLE001
            r.fail("fiducials", f"unreadable: {e}")
    else:
        r.warn("fiducials", "fiducials.json missing")

    # full 10-5 coordinates
    coords = {}
    full = edir / "landmarks_10-5-full.csv"
    if full.exists():
        try:
            coords = _read_csv_coords(full)
            r.add(PASS if len(coords) >= 50 else WARN, "10-5 electrodes (full)",
                  f"{len(coords)} positions")
        except Exception as e:  # noqa: BLE001
            r.fail("10-5 electrodes (full)", f"unreadable: {e}")

    # selected names (ears/eyes excluded)
    selected = []
    sel = edir / "selected_landmarks_10-5-full.json"
    if sel.exists():
        try:
            data = json.loads(sel.read_text())
            selected = data if isinstance(data, list) else list(data)
            r.add(PASS if selected else WARN, "selected electrodes",
                  f"{len(selected)} of {len(coords)} kept")
        except Exception as e:  # noqa: BLE001
            r.fail("selected electrodes", f"unreadable: {e}")
    else:
        r.warn("selected electrodes", "selected_landmarks json missing")

    # 3D scatter on scalp, selected highlighted
    if coords:
        names = list(coords)
        pts = np.asarray([coords[k] for k in names], dtype=float)
        finite = np.isfinite(pts).all()
        r.add(PASS if finite else FAIL, "electrode coordinates",
              f"{len(pts)} positions, finite={finite}")
        if finite:
            sel_set = set(selected)
            scal = np.asarray([1.0 if k in sel_set else 0.0 for k in names]) if selected else None
            palette = ["#3b4cc0", "#b40426"]   # excluded (blue) / selected (red)
            legend_items = [["excluded", palette[0]], ["selected", palette[1]]] if selected else None
            scalp = _scalp_mesh(ctx)
            ctx.add_figure(r, "electrodes_3d", "Electrodes on scalp (selected = red)",
                           lambda p: render3d.snapshot_points(
                               pts, p, scalars=scal, ref_mesh=scalp,
                               views=("left", "anterior", "superior"), title="electrodes",
                               point_size=6, cmap=palette if selected else "coolwarm",
                               clim=(-0.5, 1.5) if selected else None,
                               legend_items=legend_items))
    return r
