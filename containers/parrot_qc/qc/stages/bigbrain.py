"""BigBrain QC: subject<-BigBrain warped staining (registration sanity).

The BigBrain 100um staining volume is nonlinearly warped into subject space and
later averaged per-parcel into a neural-density weighting for the leadfield. A
healthy warp fills essentially the whole cortex; a failed subject<->BigBrain
affine (the registration is a hard cross-contrast fit and can lock onto a
local minimum -- typically a near-identity linear part plus a ~10 cm z-shift)
leaves the warped staining covering only a fraction of the brain. That silently
corrupts the neural density (uncovered voxels default to a flat weight, saturated
ones to zero), which the leadfield stage only partially sees as "dead sources".
So we check it here at the source: what fraction of the subject brain the warped
staining actually covers, plus the affine's scale/translation as a diagnostic.
"""
import numpy as np
import nibabel as nib

from ..checks import StageResult, PASS, WARN
from .. import render2d
from ._common import load_nifti

NAME = "bigbrain"
TITLE = "BigBrain — warped staining"
DESCRIPTION = ("The BigBrain 100um staining warped into subject space (source of the per-dipole neural-density weighting). It should cover essentially the whole cortex; partial coverage means the subject<->BigBrain registration failed and the neural density is unreliable.")

# Fraction of the subject brain the warped staining must cover. Cohort stats
# (227 LEMON subjects): healthy warps cover 99.7-99.8%; the five failures cover
# 3-30%. 95% sits far below every good subject and far above every failure.
COVERAGE_WARN_FRAC = 0.95


def _affine_diagnostic(r, mat_path):
    """Report the affine's linear scale (det) + translation -- the failure
    signature is a near-identity/degenerate linear part with a large translation."""
    if not mat_path.exists():
        return
    try:
        from scipy.io import loadmat
        m = loadmat(str(mat_path))
        p = m["AffineTransform_float_3_3"].squeeze()
        A = p[:9].reshape(3, 3)
        t = p[9:12]
        det = float(np.linalg.det(A))
        r.add(PASS, "affine",
              f"det={det:.3f}, translation=[{t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}] mm")
    except Exception as e:  # noqa: BLE001 - diagnostic only, never fail the stage
        r.add(PASS, "affine", f"unreadable ({type(e).__name__})")


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("bigbrain")
    stain = d / "subject_full16_100um_2009b_sym.nii.gz"
    if not stain.exists():
        return r.skip("no subject_full16_100um_2009b_sym.nii.gz")

    stain_img = load_nifti(r, stain, "warped staining", ndim=3)
    brain = d / "T1_stripped_N4corrected.nii.gz"  # the moving image -> brain mask

    if stain_img is not None and brain.exists():
        bm = np.asanyarray(nib.load(str(brain)).dataobj) > 0
        st = np.asanyarray(stain_img.dataobj) > 0
        n_brain = int(bm.sum())
        coverage = (st & bm).sum() / n_brain if n_brain else 0.0
        status = PASS if coverage >= COVERAGE_WARN_FRAC else WARN
        detail = f"warped staining covers {coverage * 100:.1f}% of the subject brain"
        if coverage < COVERAGE_WARN_FRAC:
            detail += " -- registration likely failed; neural density is unreliable"
        r.add(status, "brain coverage", detail)

        ctx.add_figure(r, "bigbrain_on_t1", "Warped BigBrain staining on T1 (coverage)",
                       lambda p: render2d.stat_overlay(
                           ctx.t1_path() if ctx.t1_path().exists() else brain,
                           stain, p, title="warped BigBrain staining", cmap="hot"))

    _affine_diagnostic(r, d / "transform_files" / "0GenericAffine.mat")
    return r
