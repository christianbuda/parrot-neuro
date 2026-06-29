"""Shared helpers for stage QC modules (NIfTI sanity, label counting, etc.)."""
from __future__ import annotations

import numpy as np
import nibabel as nib

from ..checks import PASS, WARN, FAIL


def load_nifti(result, path, name, ndim=None, last_dim=None):
    """Load a NIfTI/MGZ, record a sanity check (exists/readable/finite/shape),
    return the nibabel image or None. Continuous data with NaN/Inf -> FAIL."""
    if not path.exists():
        result.fail(name, "missing")
        return None
    try:
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
    except Exception as e:  # noqa: BLE001
        result.fail(name, f"unreadable: {e}")
        return None

    status = PASS
    msgs = [f"shape={tuple(img.shape)}"]
    # effective dimensionality ignoring trailing singleton axes: many pipeline
    # volumes are saved as (X,Y,Z,1), which is still a 3D field.
    eff = list(img.shape)
    while ndim is not None and len(eff) > ndim and eff[-1] == 1:
        eff.pop()
    if ndim is not None and len(eff) != ndim:
        status = WARN
        msgs.append(f"expected {ndim}D")
    if last_dim is not None and (len(img.shape) < 1 or img.shape[-1] != last_dim):
        status = WARN
        msgs.append(f"expected last dim {last_dim}")
    if data.dtype.kind == "f" and not np.isfinite(data).all():
        status = FAIL
        msgs.append("non-finite values present")
    result.add(status, name, ", ".join(msgs))
    return img


def voxel_volume_ml(img) -> float:
    zooms = img.header.get_zooms()[:3]
    return float(np.prod(zooms)) / 1000.0  # mm^3 -> mL


def n_labels(img) -> int:
    data = np.asanyarray(img.dataobj)
    return int(np.unique(data[data != 0]).size)


def first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    return None
