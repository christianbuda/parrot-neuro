"""Ordered registry of per-stage QC modules.

Order mirrors the reconstruction pipeline so the report reads top-to-bottom like
the run itself. Each module exposes NAME, TITLE and run(ctx) -> StageResult.
"""
from . import (
    ingest,
    fastsurfer,
    hippunfold,
    simnibscharm,
    fslfirst,
    synthstrip,
    cerebellum,
    surfaces,
    atlas,
    tissuelabels,
    electrodes,
    dipoles,
    tetmesh,
    dwitensor,
    anisotropy,
    connectivity,
    leadfields,
)

STAGES = [
    ingest,
    fastsurfer,
    hippunfold,
    simnibscharm,
    fslfirst,
    synthstrip,
    cerebellum,
    surfaces,
    atlas,
    tissuelabels,
    electrodes,
    dipoles,
    tetmesh,
    dwitensor,
    anisotropy,
    connectivity,
    leadfields,
]
