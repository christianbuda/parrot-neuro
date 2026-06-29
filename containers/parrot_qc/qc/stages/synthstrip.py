"""SynthStrip QC: brain extraction mask sanity + overlay."""
import numpy as np

from ..checks import StageResult, PASS, WARN
from .. import render2d
from ._common import load_nifti, voxel_volume_ml

NAME = "synthstrip"
TITLE = "SynthStrip — brain extraction"


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("synthstrip")
    mask = d / "T1_stripped_mask.nii.gz"
    if not mask.exists():
        return r.skip("no T1_stripped_mask.nii.gz")

    img = load_nifti(r, mask, "brain mask readable", ndim=3)
    if img is not None:
        data = np.asanyarray(img.dataobj)
        vol_ml = int((data > 0).sum() * voxel_volume_ml(img))
        # generous adult whole-brain (incl. some CSF/cerebellum) bound
        ok = 800 <= vol_ml <= 2200
        r.add(PASS if ok else WARN, "brain volume", f"{vol_ml} mL (expect ~900–1800)")
        if ctx.t1_path().exists():
            ctx.add_figure(r, "synthstrip_mask", "Brain mask edge on T1",
                           lambda p: render2d.contours_overlay(ctx.t1_path(), mask, p,
                                                              "brain mask", colors="red"))
    # secondary noCSF mask, if present
    nocsf = d / "T1_noCSF_stripped_mask.nii.gz"
    if nocsf.exists():
        load_nifti(r, nocsf, "noCSF mask readable", ndim=3)
    return r
