"""2D QC figures: slice mosaics and volumetric overlays via nilearn + matplotlib.

All work in the subject's own anatomical space (not MNI); nilearn happily plots
any affine, picking cut coordinates from the image bounds. We render label/mask
geometry as overlays on the T1 here (segmentation-follows-anatomy QC); genuine
surface/mesh geometry is handled in render3d.py.

The T1 background is passed through `_masked_bg`: MP2RAGE (MPRAGEised) T1s keep a
low-but-nonzero speckle in the air that nilearn's autoscaling stretches into
visible noise around the head. We zero the sub-tissue background (Otsu split) so
it renders black; on an already-clean conventional T1 this is a no-op.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.patches import Patch      # noqa: E402
import numpy as np                        # noqa: E402
import nibabel as nib                     # noqa: E402
from nilearn import plotting              # noqa: E402

_DPI = 95


# --- background cleanup ------------------------------------------------------

def _otsu(values) -> float:
    """Otsu threshold on a 1D array (256-bin histogram). Splits the background/
    noise population from foreground tissue. Returns 0.0 on degenerate input."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0 or v.min() == v.max():
        return 0.0
    hist, edges = np.histogram(v, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(np.float64)
    total = w.sum()
    wsum = np.cumsum(w)
    msum = np.cumsum(w * centers)
    mtot = msum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mb = msum / wsum
        mf = (mtot - msum) / (total - wsum)
        var = wsum * (total - wsum) * (mb - mf) ** 2
    k = int(np.nanargmax(var))
    return float(centers[k])


_BG_CACHE: dict = {}


def _masked_bg(path):
    """Return a background-cleaned copy of the T1 as a nibabel image (cached by
    path+mtime). Voxels below the Otsu split are zeroed so the extracranial noise
    renders black; clean T1s are effectively unchanged."""
    path = str(path)
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        key = (path, 0)
    if key in _BG_CACHE:
        return _BG_CACHE[key]
    img = nib.load(path)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    # Otsu over the WHOLE volume (not just >0): the background population anchors
    # the low class. On a clean T1 that population is the ~0 air, so the split
    # sits just above it and only zeros the already-dark background (no-op-ish);
    # on MP2RAGE it sits between the nonzero air noise and tissue, zeroing the
    # noise. (Computing it over >0 only would split within tissue and eat CSF.)
    finite = data[np.isfinite(data)]
    if finite.size and finite.min() != finite.max():
        thr = _otsu(finite)
        out = np.where(data > thr, data, 0.0).astype(np.float32)
    else:
        out = data
    # Build from the affine only: reusing img.header would re-apply its scl_slope/
    # scl_inter to the already-scaled data (asanyarray applied it once) -> double
    # scaling. The affine preserves orientation/geometry.
    masked = nib.Nifti1Image(out, img.affine)
    _BG_CACHE[key] = masked
    return masked


def _display_figure(disp):
    """The matplotlib Figure behind a nilearn display (for adding a legend)."""
    try:
        return next(iter(disp.axes.values())).ax.figure
    except Exception:  # noqa: BLE001
        return None


def add_patch_legend(disp, entries, ncol=None, fontsize=6):
    """Attach a discrete colour key to a nilearn display. entries: list of
    (label, color). Guarded -- a legend is a nicety, never fail the figure."""
    if not entries:
        return
    fig = _display_figure(disp)
    if fig is None:
        return
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in entries]
    ncol = ncol or min(len(entries), 6)
    try:
        fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=fontsize,
                   frameon=False, labelcolor="white", handlelength=1.0,
                   columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))
    except Exception:  # noqa: BLE001
        pass


# --- spatial overlays --------------------------------------------------------

def mosaic(bg, out, title=None, cmap="gray"):
    """Multi-slice anatomical mosaic (axial+coronal+sagittal)."""
    disp = plotting.plot_anat(_masked_bg(bg), display_mode="mosaic", title=title, cmap=cmap)
    disp.savefig(out, dpi=_DPI)
    disp.close()


def roi_overlay(bg, roi, out, title=None, alpha=0.55, cmap="tab20", legend=None):
    """Integer label / mask volume overlaid (filled) on a background volume.
    `legend`: optional list of (label, color) drawn as a discrete key."""
    disp = plotting.plot_roi(
        str(roi), bg_img=_masked_bg(bg), display_mode="mosaic", title=title,
        alpha=alpha, cmap=cmap, black_bg=False,
    )
    add_patch_legend(disp, legend)
    disp.savefig(out, dpi=_DPI)
    disp.close()


def contours_overlay(bg, label, out, title=None, colors="red", linewidths=0.6):
    """Outline of a label/mask volume drawn on a background (e.g. brain mask edge)."""
    disp = plotting.plot_anat(_masked_bg(bg), display_mode="mosaic", title=title)
    disp.add_contours(str(label), colors=colors, linewidths=linewidths)
    disp.savefig(out, dpi=_DPI)
    disp.close()


def stat_overlay(bg, stat, out, title=None, cmap="hot", vmax=None, threshold=1e-6):
    """Continuous map (e.g. FA) overlaid on a background volume."""
    disp = plotting.plot_stat_map(
        str(stat), bg_img=_masked_bg(bg), display_mode="mosaic", title=title,
        cmap=cmap, vmax=vmax, threshold=threshold, colorbar=True, black_bg=False,
    )
    disp.savefig(out, dpi=_DPI)
    disp.close()


def rgb_mosaic(rgb_img, out, title=None, n=21, gamma=0.6, gain=1.6):
    """Render a 3-channel RGB NIfTI (e.g. the DEC map) as an axial slice grid.

    nilearn has no native RGB display, so we tile axial slices ourselves. The raw
    DEC magnitudes are small (dominated by low-FA voxels), so we gamma-brighten and
    gain the colours to make the CC (red) / CST (blue) visible.
    rgb_img: nibabel image with last axis = 3 (values in [0,1])."""
    img = rgb_img if hasattr(rgb_img, "dataobj") else nib.load(str(rgb_img))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 5:                     # (x,y,z,1,3) -> (x,y,z,3)
        data = data[:, :, :, 0, :]
    data = np.clip(data, 0, 1)
    data = np.clip((data ** gamma) * gain, 0, 1)   # brighten for visibility
    z = data.shape[2]
    lo, hi = int(z * 0.2), int(z * 0.85)
    idx = np.linspace(lo, hi, n).astype(int)
    cols = 7
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.7))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, k in zip(np.ravel(axes), idx):
        ax.imshow(np.transpose(data[:, :, k, :], (1, 0, 2))[::-1], origin="upper")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)


# --- non-spatial plots ------------------------------------------------------

def heatmap(matrix, out, title=None, log=False, cmap="magma"):
    m = np.asarray(matrix, dtype=np.float64)
    if log:
        m = np.log1p(np.clip(m, 0, None))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(m, cmap=cmap, aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        ax.set_title(title)
    ax.set_xlabel("region"); ax.set_ylabel("region")
    fig.tight_layout(); fig.savefig(out, dpi=_DPI); plt.close(fig)


def histogram(values, out, title=None, xlabel="", bins=60, logy=False):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(v, bins=bins)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(out, dpi=_DPI); plt.close(fig)


def scatter_xy(x, y, out, title=None, xlabel="", ylabel="", s=4):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.scatter(x, y, s=s, alpha=0.4)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(out, dpi=_DPI); plt.close(fig)
