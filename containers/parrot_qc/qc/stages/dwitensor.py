"""DWI tensor QC (optional): FA + eigenvalues from the DTI fit in mesh (T1) space."""
import numpy as np
import nibabel as nib

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render2d
from ._common import load_nifti

NAME = "dwitensor"
TITLE = "DWI tensor — FA / eigenvalues"
DESCRIPTION = ("The DTI fit (FA + eigenvalues) in mesh/T1 space. FA should be high in white-matter tracts (CC, CST) and low in grey matter/CSF, with ordered non-negative eigenvalues. FA>1 in noisy/CSF voxels is normal least-squares fit noise.")


def _pick(d, pattern):
    hits = sorted(d.glob(pattern))
    return hits[0] if hits else None


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("dwitensor")
    if not d.exists():
        return r.skip("no dwitensor/ (no DWI)")

    fa = _pick(d, "*space-T1*param-fa.nii.gz") or _pick(d, "*param-fa.nii.gz")
    if fa is None:
        return r.skip("no FA map (no DWI tensor)")

    img = load_nifti(r, fa, "FA readable", ndim=3)
    if img is not None:
        v = np.asanyarray(img.dataobj).astype(np.float64)
        nz = v[v != 0]
        lo, hi = (float(nz.min()), float(nz.max())) if nz.size else (0.0, 0.0)
        # FA in noisy/CSF voxels routinely exceeds 1 with a least-squares tensor
        # fit -- that is expected, not an error. Only flag a clearly broken map
        # (non-finite, negative, or absurd values); report the range as info.
        finite = bool(np.isfinite(nz).all()) if nz.size else False
        broken = (not nz.size) or (not finite) or lo < -1e-6 or hi > 3.0
        r.add(FAIL if broken else PASS, "FA range",
              f"min={lo:.3f}, max={hi:.3f} (values >1 are normal DTI fit noise)")
        is_t1 = "space-T1" in fa.name
        if is_t1 and ctx.t1_path().exists():
            ctx.add_figure(r, "fa_on_t1", "FA on T1",
                           lambda p: render2d.stat_overlay(ctx.t1_path(), fa, p, "FA",
                                                          cmap="hot", vmax=1.0))
        ctx.add_figure(r, "fa_hist", "FA histogram (non-zero)",
                       lambda p: render2d.histogram(nz, p, "FA", "FA"))

    eigvals = _pick(d, "*space-T1*param-eigvals.nii.gz") or _pick(d, "*param-eigvals.nii.gz")
    if eigvals is not None:
        ev = np.asanyarray(nib.load(str(eigvals)).dataobj).astype(np.float64)
        if ev.ndim == 4 and ev.shape[-1] == 3:
            mask = ev[..., 0] != 0
            l1, l2, l3 = ev[..., 0][mask], ev[..., 1][mask], ev[..., 2][mask]
            ordered = np.mean((l1 >= l2) & (l2 >= l3)) if mask.any() else 0.0
            nonneg = np.mean(l3 >= -1e-9) if mask.any() else 0.0
            r.add(PASS if ordered > 0.98 else WARN, "eigenvalue ordering",
                  f"{ordered*100:.1f}% voxels λ1≥λ2≥λ3")
            r.add(PASS if nonneg > 0.95 else WARN, "eigenvalue non-negativity",
                  f"{nonneg*100:.1f}% voxels λ3≥0")
    return r
