"""2D QC figures: slice mosaics and volumetric overlays via nilearn + matplotlib.

All work in the subject's own anatomical space (not MNI); nilearn happily plots
any affine, picking cut coordinates from the image bounds. We render label/mask
geometry as overlays on the T1 here (segmentation-follows-anatomy QC); genuine
surface/mesh geometry is handled in render3d.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
import nibabel as nib                     # noqa: E402
from nilearn import plotting              # noqa: E402

_DPI = 95


def mosaic(bg, out, title=None, cmap="gray"):
    """Multi-slice anatomical mosaic (axial+coronal+sagittal)."""
    disp = plotting.plot_anat(str(bg), display_mode="mosaic", title=title, cmap=cmap)
    disp.savefig(out, dpi=_DPI)
    disp.close()


def roi_overlay(bg, roi, out, title=None, alpha=0.55, cmap="tab20"):
    """Integer label / mask volume overlaid (filled) on a background volume."""
    disp = plotting.plot_roi(
        str(roi), bg_img=str(bg), display_mode="mosaic", title=title,
        alpha=alpha, cmap=cmap, black_bg=False,
    )
    disp.savefig(out, dpi=_DPI)
    disp.close()


def contours_overlay(bg, label, out, title=None, colors="red", linewidths=0.6):
    """Outline of a label/mask volume drawn on a background (e.g. brain mask edge)."""
    disp = plotting.plot_anat(str(bg), display_mode="mosaic", title=title)
    disp.add_contours(str(label), colors=colors, linewidths=linewidths)
    disp.savefig(out, dpi=_DPI)
    disp.close()


def stat_overlay(bg, stat, out, title=None, cmap="hot", vmax=None, threshold=1e-6):
    """Continuous map (e.g. FA) overlaid on a background volume."""
    disp = plotting.plot_stat_map(
        str(stat), bg_img=str(bg), display_mode="mosaic", title=title,
        cmap=cmap, vmax=vmax, threshold=threshold, colorbar=True, black_bg=False,
    )
    disp.savefig(out, dpi=_DPI)
    disp.close()


def rgb_mosaic(rgb_img, out, title=None, n=21):
    """Render a 3-channel RGB NIfTI (e.g. the DEC map) as an axial slice grid.

    nilearn has no native RGB display, so we tile axial slices ourselves.
    rgb_img: nibabel image with last axis = 3 (values in [0,1])."""
    img = rgb_img if hasattr(rgb_img, "dataobj") else nib.load(str(rgb_img))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 5:                     # (x,y,z,1,3) -> (x,y,z,3)
        data = data[:, :, :, 0, :]
    data = np.clip(data, 0, 1)
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
