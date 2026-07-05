"""HippUnfold QC: hippocampal subfield segmentation + midthickness surfaces."""
import nibabel as nib

from ..checks import StageResult, PASS, FAIL
from .. import render2d, render3d
from ._common import load_nifti

# HippUnfold multihist7 subfield labels (used for the 2D overlay legend).
_SUBFIELDS = [(1, "Sub"), (2, "CA1"), (3, "CA2"), (4, "CA3"),
              (5, "CA4"), (6, "DG"), (7, "SRLM"), (8, "Cyst")]

NAME = "hippunfold"
TITLE = "HippUnfold — hippocampus"
DESCRIPTION = ("Hippocampal subfield segmentation + midthickness surfaces per hemisphere. Subfields should form the expected medial-temporal ribbon; the L/R surfaces should be smooth and complete and sit in the hippocampal formation.")


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
        ctx.add_figure(r, "hipp_subfields_on_t1", "Hippocampal subfields on T1 (zoomed)",
                       lambda p: render2d.label_overlay(ctx.t1_path(), dsegs[0], p,
                                                        _SUBFIELDS, "subfields (one hemi)",
                                                        crop=True, crop_pad=10))

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

    # 3D of the (tiny) hippocampal midthickness surfaces, zoomed, L/R coloured, with
    # a translucent brain for anatomical context (the co-registered world-space plys).
    surf_dir = ctx.stage_dir("surfaces")
    lp, rp = surf_dir / "hippunfold_L_hipp_middle.ply", surf_dir / "hippunfold_R_hipp_middle.ply"
    if lp.exists() or rp.exists():
        def _render(p):
            items, foc = [], None
            for fp, col, lab in ((lp, "#1f77b4", "L hippocampus"), (rp, "#d62728", "R hippocampus")):
                if fp.exists():
                    m = render3d.load_surface(fp)
                    items.append({"mesh": m, "color": col, "opacity": 1.0, "label": lab})
                    foc = m if foc is None else foc.merge(m)
            for h in ("freesurfer_lh_pial.ply", "freesurfer_rh_pial.ply"):
                hp = surf_dir / h
                if hp.exists():
                    items.append({"mesh": render3d.load_surface(hp), "color": "lightgray",
                                  "opacity": 0.08})
            # L=blue / R=red is obvious from the view labels; no 2-entry legend.
            render3d.snapshot_meshes(items, p, title="hippocampus (blue=L, red=R)", focus=foc,
                                     views=("left", "anterior", "superior"))
        ctx.add_figure(r, "hipp_surfaces_3d", "Hippocampal surfaces (3D, zoomed)", _render)
    return r
