"""Status vocabulary shared by the metric/flag/report modules.

Deliberately mirrors ``containers/parrot_qc/qc/checks.py``'s PASS/WARN/FAIL/SKIP
vocabulary (same look-and-feel across the project's QC surfaces), but is its
own copy -- the Docker QC package is baked into a separate image and isn't
importable from here.
"""
from __future__ import annotations

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
_SEVERITY = {SKIP: -1, PASS: 0, WARN: 1, FAIL: 2}


def worst(statuses) -> str:
    """Aggregate status: the most severe non-skip; SKIP only if everything is skip."""
    real = [s for s in statuses if s != SKIP]
    if not real:
        return SKIP
    return max(real, key=lambda s: _SEVERITY[s])
