"""Atlas QC: multi-resolution subject parcellations (cortical Schaefer + subcortical)."""
import numpy as np

from ..checks import StageResult, PASS, WARN
from .. import render2d
from ._common import load_nifti, n_labels

NAME = "atlas"
TITLE = "Atlas — multi-resolution parcellation"
DESCRIPTION = ("The aggregated parcellation (connectivity nodes) on the cleaned T1. It should tile the cortical ribbon following anatomy, with no large gaps and no spill into white matter or CSF.")

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

    # Show only the aggregated parcellation (the ~121-node connectivity atlas) on the
    # cleaned T1. A per-region legend is impractical at 121 nodes, so the colorbar is
    # off; the point is coverage + segmentation-follows-anatomy, not region identity.
    if agg.exists() and ctx.t1_path().exists():
        ctx.add_figure(r, "atlas_aggregated_on_t1",
                       "Aggregated parcellation on T1 (connectivity nodes)",
                       lambda p: render2d.roi_overlay(ctx.t1_path(), agg, p,
                                                      "aggregated atlas", cmap="gist_ncar"))
    return r
