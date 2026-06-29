"""Ingest QC: the standardized raw T1 (and optional T2) the whole pipeline builds on."""
import numpy as np

from ..checks import StageResult, PASS, WARN, fmt_range
from .. import render2d
from ._common import load_nifti

NAME = "ingest"
TITLE = "Ingest — raw T1 / T2"


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    t1 = ctx.sfile("raw", "T1.nii.gz")
    if not t1.exists():
        return r.skip("no raw/T1.nii.gz")

    img = load_nifti(r, t1, "T1 readable & 3D", ndim=3)
    if img is not None:
        zooms = img.header.get_zooms()[:3]
        sane = all(0.3 <= z <= 2.0 for z in zooms)
        r.add(PASS if sane else WARN, "T1 voxel size",
              "x".join(f"{z:.2f}" for z in zooms) + " mm")
        data = np.asanyarray(img.dataobj).astype(np.float64)
        p1, p99 = np.percentile(data, [1, 99])
        r.add(PASS if p99 > p1 else WARN, "T1 intensity spread",
              f"p1={p1:.1f}, p99={p99:.1f} ({fmt_range(data)})")
        ctx.add_figure(r, "ingest_t1", "Ingested T1 (mosaic)",
                       lambda p: render2d.mosaic(t1, p, "T1"))

    t2 = ctx.sfile("raw", "T2.nii.gz")
    if t2.exists():
        load_nifti(r, t2, "T2 readable & 3D", ndim=3)
        ctx.add_figure(r, "ingest_t2", "Ingested T2 (mosaic)",
                       lambda p: render2d.mosaic(t2, p, "T2"))
    return r
