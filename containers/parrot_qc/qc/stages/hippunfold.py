"""HippUnfold QC: hippocampal subfield segmentation + midthickness surfaces."""
import nibabel as nib

from ..checks import StageResult, PASS, FAIL
from .. import render2d
from ._common import load_nifti

NAME = "hippunfold"
TITLE = "HippUnfold — hippocampus"


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("hippunfold")
    if not d.exists():
        return r.skip("no hippunfold/ tree")

    # volumetric subfield dseg in T1w space (one per hemisphere)
    dsegs = sorted((d / "anat").glob("*space-T1w*subfields*dseg.nii.gz")) if (d / "anat").exists() else []
    if not dsegs:
        dsegs = sorted(d.glob("**/*space-T1w*subfields*dseg.nii.gz"))
    if not dsegs:
        r.warn("subfield dseg", "no space-T1w subfields dseg found")
    for ds in dsegs:
        hemi = "L" if "hemi-L" in ds.name else ("R" if "hemi-R" in ds.name else "?")
        load_nifti(r, ds, f"subfield dseg ({hemi})", ndim=3)
    if dsegs and ctx.t1_path().exists():
        ctx.add_figure(r, "hipp_subfields_on_t1", "Hippocampal subfields on T1",
                       lambda p: render2d.roi_overlay(ctx.t1_path(), dsegs[0], p,
                                                      "subfields (one hemi)"))

    # midthickness surfaces
    surfs = sorted((d / "surf").glob("*space-T1w*label-hipp_midthickness.surf.gii")) if (d / "surf").exists() else []
    for sp in surfs:
        hemi = "L" if "hemi-L" in sp.name else ("R" if "hemi-R" in sp.name else "?")
        try:
            g = nib.load(str(sp))
            nv = g.darrays[0].data.shape[0]
            r.add(PASS if nv > 0 else FAIL, f"midthickness surf ({hemi})", f"{nv} verts")
        except Exception as e:  # noqa: BLE001
            r.fail(f"midthickness surf ({hemi})", f"unreadable: {e}")
    if not surfs:
        r.warn("midthickness surfaces", "none found")
    return r
