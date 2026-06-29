"""QC result model + small reusable check helpers.

A QC run produces, per pipeline stage, a `StageResult`: a list of `Check`s (each
pass/warn/fail) and a list of figures (caption -> PNG written under the subject's
qc/figures/). A stage with no outputs reports `present=False` and renders as a
muted "not produced" row -- never a failure. This keeps one report valid across
FastSurfer-only, HCP, no-DWI and full runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# Status vocabulary. SKIP = "not produced / not applicable"; it never worsens an
# aggregate (a skipped optional stage must not turn a subject red).
PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
_SEVERITY = {SKIP: -1, PASS: 0, WARN: 1, FAIL: 2}


def worst(statuses) -> str:
    """Aggregate status: the most severe non-skip; SKIP only if everything is skip."""
    real = [s for s in statuses if s != SKIP]
    if not real:
        return SKIP
    return max(real, key=lambda s: _SEVERITY[s])


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class StageResult:
    name: str
    title: str
    present: bool = True
    checks: List[Check] = field(default_factory=list)
    # figures: (caption, html-relative png path e.g. "figures/foo.png")
    figures: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # --- mutators used by stage modules -------------------------------------
    def ok(self, name, detail=""):
        self.checks.append(Check(name, PASS, detail))

    def warn(self, name, detail=""):
        self.checks.append(Check(name, WARN, detail))

    def fail(self, name, detail=""):
        self.checks.append(Check(name, FAIL, detail))

    def add(self, status, name, detail=""):
        self.checks.append(Check(name, status, detail))

    def skip(self, reason="not produced"):
        """Mark the whole stage as not-produced/not-applicable."""
        self.present = False
        if reason:
            self.notes.append(reason)
        return self

    @property
    def status(self) -> str:
        if not self.present:
            return SKIP
        if not self.checks:
            return SKIP
        return worst(c.status for c in self.checks)


# --- generic numeric helpers (return a status string) -----------------------

def status_in_range(value, lo, hi) -> str:
    return PASS if (lo <= value <= hi) else FAIL


def status_finite(arr) -> str:
    a = np.asarray(arr)
    return PASS if np.isfinite(a).all() else FAIL


def fmt_range(arr) -> str:
    a = np.asarray(arr, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return "all non-finite"
    return f"min={finite.min():.4g}, max={finite.max():.4g}, mean={finite.mean():.4g}"
