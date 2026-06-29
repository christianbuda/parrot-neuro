"""Atlas QC: multi-resolution subject parcellations (cortical Schaefer + subcortical)."""
import numpy as np

from ..checks import StageResult, PASS, WARN
from .. import render2d
from ._common import load_nifti, n_labels

NAME = "atlas"
TITLE = "Atlas — multi-resolution parcellation"

_RES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("atlas")
    if not d.exists():
        return r.skip("no atlas/ tree")

    present = 0
    for n in _RES:
        f = d / f"atlas{n}.nii.gz"
        if not f.exists():
            r.warn(f"atlas{n}", "missing")
            continue
        present += 1
        img = load_nifti(r, f, f"atlas{n} readable", ndim=3)
        if img is not None:
            nl = n_labels(img)
            # cortical n + subcortical/cerebellar -> should exceed n
            r.add(PASS if nl >= n else WARN, f"atlas{n} label count",
                  f"{nl} labels (>= {n} cortical expected)")
    if present == 0:
        return r.skip("no atlasN volumes found")

    agg = d / "atlas_aggregated.nii.gz"
    if agg.exists():
        img = load_nifti(r, agg, "aggregated atlas readable", ndim=3)
        if img is not None:
            r.add(PASS, "aggregated regions", f"{n_labels(img)} labels")

    a100 = d / "atlas100.nii.gz"
    if a100.exists() and ctx.t1_path().exists():
        ctx.add_figure(r, "atlas100_on_t1", "Atlas (100) on T1",
                       lambda p: render2d.roi_overlay(ctx.t1_path(), a100, p,
                                                      "atlas100", cmap="gist_ncar"))
    return r
