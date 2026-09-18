"""Point/surface measurements shared across images. numpy + scipy only (no trimesh)."""
import numpy as np
from scipy.spatial import cKDTree

from .thresholds import REG_FOV_MARGIN_MM


def scalp_residual(mapped_tmpl, subj_verts, margin_mm=REG_FOV_MARGIN_MM):
    """Mean template->subject scalp distance, over the z-band where both surfaces exist.

    `mapped_tmpl` is the template scalp already carried into subject space by the affine.
    Vertices below the subject scalp's own floor have nothing to match (see
    thresholds.REG_FOV_MARGIN_MM) and are excluded.

    Returns (mean_mm, stats); stats records what was excluded so the gate stays auditable.
    """
    mapped_tmpl = np.asarray(mapped_tmpl, dtype=float)
    subj_verts = np.asarray(subj_verts, dtype=float)
    z_min = float(subj_verts[:, 2].min())
    keep = mapped_tmpl[:, 2] >= z_min + margin_mm
    # An affine wrong enough to put the whole template below the floor leaves nothing to
    # measure. Report inf rather than raising, so the caller rejects it through its normal
    # evaluated-fraction guard instead of an IndexError out of np.percentile.
    d = cKDTree(subj_verts).query(mapped_tmpl[keep], k=1)[0] if keep.any() else None
    stats = {
        "n_template_vertices": int(len(mapped_tmpl)),
        "n_evaluated": int(keep.sum()),
        "evaluated_fraction": float(keep.mean()),
        "subject_scalp_z_min_mm": z_min,
        "fov_margin_mm": float(margin_mm),
        "residual_p90_mm": float(np.percentile(d, 90)) if d is not None else float("inf"),
    }
    return (float(np.mean(d)) if d is not None else float("inf")), stats


def fov_escape_depth(vertices, lo, hi):
    """Per-vertex distance outside the [lo, hi] image bounding box; <= 0 means inside.

    Watershed/charm surfaces can be fitted past the edge of the acquired T1, where there is
    no data to fit to, so several stages flag it.
    """
    v = np.asarray(vertices, dtype=float)
    return np.maximum(np.asarray(lo, float) - v, v - np.asarray(hi, float)).max(1)
