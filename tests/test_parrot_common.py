"""Unit tests for containers/_shared/parrot_common.

This package is the one place code shared by several container images lives, so a change
here silently reaches four images at once. It is also the only container code that is pure
enough to test on the host (no ANTs, no DUNEuro, no derivatives tree), which is why it gets
tests while the rest of containers/ does not.

Only the numpy/scipy half is covered; meshops needs real trimesh surfaces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "containers" / "_shared"))

from parrot_common import thresholds  # noqa: E402
from parrot_common.geometry import fov_escape_depth, scalp_residual  # noqa: E402


def _flat_scalp(z=0.0, n=21):
    """A flat 'subject scalp' patch in the z=`z` plane; odd n puts a vertex at x=y=0."""
    x, y = np.meshgrid(np.linspace(-50, 50, n), np.linspace(-50, 50, n))
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def test_thresholds_are_ordered():
    assert thresholds.REG_RESIDUAL_WARN_MM < thresholds.REG_RESIDUAL_FAIL_MM
    assert 0.0 < thresholds.REG_MIN_EVALUATED_FRAC < 1.0


def test_residual_ignores_template_below_the_subject_scalp_floor():
    """The bug this package was extracted for: orphan neck vertices dominating the mean.

    Ten template vertices 10 mm above a flat scalp, plus one 100 mm below where the subject
    has no surface at all. The unclipped mean would be ~18 mm; the clipped one stays at 10.
    """
    subj = _flat_scalp(z=0.0)
    tmpl = np.column_stack([np.zeros(10), np.zeros(10), np.full(10, 10.0)])
    tmpl = np.vstack([tmpl, [0.0, 0.0, -100.0]])

    err, stats = scalp_residual(tmpl, subj)

    assert err == pytest.approx(10.0)
    assert stats["n_evaluated"] == 10
    assert stats["n_template_vertices"] == 11
    assert stats["evaluated_fraction"] == pytest.approx(10 / 11)


def test_residual_margin_excludes_the_band_just_above_the_floor():
    """Vertices within REG_FOV_MARGIN_MM of the floor are dropped too: an FOV-cut scalp is
    unreliable right at the cut, not only below it."""
    subj = _flat_scalp(z=0.0)
    tmpl = np.array([[0.0, 0.0, thresholds.REG_FOV_MARGIN_MM - 0.1],
                     [0.0, 0.0, thresholds.REG_FOV_MARGIN_MM + 0.1]])

    _, stats = scalp_residual(tmpl, subj)

    assert stats["n_evaluated"] == 1


def test_residual_keeps_everything_when_the_subject_spans_the_template():
    """A subject with full neck coverage must measure exactly as it did before clipping."""
    subj = np.vstack([_flat_scalp(z=0.0), _flat_scalp(z=-200.0)])
    tmpl = np.column_stack([np.zeros(5), np.zeros(5), np.linspace(-100, 50, 5)])

    _, stats = scalp_residual(tmpl, subj)

    assert stats["evaluated_fraction"] == 1.0


def test_fov_escape_depth_sign_marks_inside_and_outside():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0])
    pts = np.array([[5.0, 5.0, 5.0],      # inside
                    [-3.0, 5.0, 5.0],     # 3 mm past the low x face
                    [5.0, 5.0, 17.0]])    # 7 mm past the high z face

    depth = fov_escape_depth(pts, lo, hi)

    assert depth[0] < 0
    assert depth[1] == pytest.approx(3.0)
    assert depth[2] == pytest.approx(7.0)


def test_residual_reports_inf_when_nothing_survives_the_clip():
    """A catastrophically wrong affine must reach the caller's fraction guard, not raise."""
    subj = _flat_scalp(z=0.0)
    tmpl = np.array([[0.0, 0.0, -500.0], [0.0, 0.0, -400.0]])

    err, stats = scalp_residual(tmpl, subj)

    assert np.isinf(err)
    assert stats["evaluated_fraction"] == 0.0
    assert stats["evaluated_fraction"] < thresholds.REG_MIN_EVALUATED_FRAC
