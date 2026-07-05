"""Leadfields QC (final stage): forward-solution matrix sanity + sensitivity map.

Each leadfield maps source activity to electrode potentials; its layout is
(n_electrodes, 3 * n_dipoles) with columns dipole-major, xyz-interleaved
([d0x,d0y,d0z, d1x,...] -- see make_leadfield_duneuro.py). We check it loads, is
finite, is not all-zero, has a plausible magnitude spread, and that its source
dimension matches the dipole count. The headline visual is a 3D sensitivity map:
each dipole coloured by the norm of its 3-column block (how strongly the whole
cap sees it) -- deep sources should be dimmer than superficial ones.
"""
import re

import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL, fmt_range
from .. import render2d, render3d
from .electrodes import _scalp_mesh

NAME = "leadfields"
TITLE = "Leadfields — forward solution"
DESCRIPTION = ("The forward-solution leadfield(s). Each should be finite, non-zero, and have the expected (n_elec, 3*n_dip) shape; the sensitivity map should show superficial sources brighter than deep ones.")


def _dipole_positions(ctx, spacing):
    if spacing is None:
        return None
    f = ctx.stage_dir("dipoles") / f"spacing{spacing}mm" / "dipole_positions.npy"
    if f.exists():
        try:
            return np.load(f)
        except Exception:  # noqa: BLE001
            return None
    return None


def _per_dipole_sensitivity(L, ndip):
    """Frobenius norm of each dipole's (n_elec, 3) column block -> (ndip,)."""
    Lr = L.reshape(L.shape[0], ndip, 3)
    return np.sqrt((Lr ** 2).sum(axis=(0, 2)))


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("leadfields")
    lfs = sorted(d.glob("processed_*-leadfield.npy")) if d.exists() else []
    if not lfs:
        return r.skip("no processed_*-leadfield.npy")

    pick = None  # (L, tag, positions) for the sensitivity figure
    for lf in lfs:
        tag = lf.name.replace("processed_", "").replace("-leadfield.npy", "")
        try:
            L = np.load(lf)
        except Exception as e:  # noqa: BLE001
            r.fail(f"leadfield {tag}", f"unreadable: {e}")
            continue
        if L.ndim != 2:
            r.fail(f"leadfield {tag}", f"not 2D (shape={L.shape})")
            continue
        finite = np.isfinite(L).all()
        nonzero = np.any(L != 0)
        detail = f"shape={L.shape}, {fmt_range(L)}"
        if not finite:
            detail += ", NON-FINITE"
        if not nonzero:
            detail += ", ALL-ZERO"
        r.add(PASS if (finite and nonzero) else FAIL, f"leadfield {tag}", detail)

        # expected layout: (n_elec, 3 * n_dip)
        m = re.search(r"(\d+\.\d+)mm", tag)
        spacing = m.group(1) if m else None
        positions = _dipole_positions(ctx, spacing)
        if positions is not None:
            ndip = len(positions)
            ok = L.shape[1] == 3 * ndip
            r.add(PASS if ok else WARN, f"source count {tag}",
                  f"{ndip} dipoles -> expect {3 * ndip} cols, got {L.shape[1]}")
            if ok and pick is None and finite and nonzero:
                pick = (L, tag, positions)

        if pick is None and finite and nonzero:
            pick = (L, tag, None)  # fallback for the histogram only

    if pick is not None:
        L, tag, positions = pick
        ctx.add_figure(r, "lf_magnitude_hist", "Leadfield magnitude distribution",
                       lambda p: render2d.histogram(np.abs(L).ravel(), p,
                                                    f"|leadfield| ({tag})", "|gain|", logy=True))
        if positions is not None:
            sens = _per_dipole_sensitivity(L, len(positions))
            log_sens = np.log10(sens + sens[sens > 0].min() * 1e-3) if np.any(sens > 0) else sens
            scalp = _scalp_mesh(ctx)
            ctx.add_figure(r, "lf_sensitivity_3d",
                           "Per-dipole sensitivity (cap's total gain, log scale)",
                           lambda p: render3d.snapshot_points(
                               positions, p, scalars=log_sens, ref_mesh=scalp, ref_opacity=0.15,
                               views=("left", "anterior", "superior"),
                               title=f"sensitivity {tag}", point_size=3, cmap="inferno"))
    return r
