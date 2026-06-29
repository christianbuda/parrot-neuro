"""Per-subject QC context: path resolution + figure bookkeeping.

Centralises (a) where each pipeline stage writes its outputs in the derivatives
tree (the real on-disk names, which differ from the legend in places -- surfaces
are .ply, BEM lives under fastsurfer/, leadfields are processed_*-leadfield.npy)
and (b) the create-figure-with-graceful-fallback helper so stage modules stay
short and a single broken plot degrades to a WARN instead of crashing the report.
"""
from __future__ import annotations

import traceback
from pathlib import Path

from .checks import Check, StageResult, WARN


class Context:
    def __init__(self, deriv: str, subject: str):
        self.deriv = Path(deriv)
        self.subject = subject                 # bare id, e.g. "010002"
        self.subj = f"sub-{subject}"
        self.out_dir = self.deriv / "qc" / self.subj
        self.fig_dir = self.out_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # --- path helpers -------------------------------------------------------
    def stage_dir(self, stage: str) -> Path:
        """<derivatives>/<stage>/sub-<id> for the per-subject stages."""
        return self.deriv / stage / self.subj

    def sfile(self, stage: str, *parts) -> Path:
        return self.stage_dir(stage).joinpath(*parts)

    def t1_path(self) -> Path:
        """The ingested T1 -- the background for most 2D overlays."""
        return self.sfile("raw", "T1.nii.gz")

    # --- figure helper ------------------------------------------------------
    def add_figure(self, result: StageResult, stem: str, caption: str, render_fn):
        """Render one figure via render_fn(abs_png_path); record it, or downgrade
        a rendering failure to a WARN check so it never aborts the report."""
        rel = f"figures/{stem}.png"
        abspath = self.out_dir / rel
        try:
            render_fn(str(abspath))
        except Exception as e:  # noqa: BLE001 - QC must be robust to any plot failure
            result.checks.append(
                Check(f"render: {caption}", WARN, f"{type(e).__name__}: {e}")
            )
            traceback.print_exc()
            return
        if abspath.exists():
            result.figures.append((caption, rel))
        else:
            result.checks.append(Check(f"render: {caption}", WARN, "no output produced"))
