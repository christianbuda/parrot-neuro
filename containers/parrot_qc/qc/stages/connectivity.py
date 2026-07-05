"""Connectivity QC (optional): structural connectome matrices + parcellations."""
import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render2d, render3d

NAME = "connectivity"
TITLE = "Structural connectome"
DESCRIPTION = ("The structural connectome (weights/distances) and the tractogram it is built from. The matrix should be symmetric, non-negative and non-empty; streamlines should fill the white matter with anatomically sensible bundles.")

_RES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def _add_tractography(ctx, r):
    """Subsampled, direction-coloured render of the streamlines the connectome is
    built from (T1/mesh space), inside a translucent brain."""
    qr = ctx.stage_dir("qsirecon") / "dwi"
    hits = sorted(qr.glob("*space-T1*streamlines.tck*")) if qr.exists() else []
    if not hits:
        return
    tck = hits[0]
    brain_f = ctx.stage_dir("surfaces") / "freesurfer_BEM_brain.ply"
    brain = render3d.load_surface(brain_f) if brain_f.exists() else None
    ctx.add_figure(r, "tractography_3d", "Tractography (subsampled, direction-coloured)",
                   lambda p: render3d.snapshot_streamlines(tck, p, brain_mesh=brain,
                                                           title="tractography", max_lines=5000))


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("connectivity")
    if not d.exists():
        return r.skip("no connectivity/ (no DWI / no template)")

    _add_tractography(ctx, r)

    found = 0
    for n in _RES:
        wf = d / f"weights_{n}.txt"
        if not wf.exists():
            continue
        found += 1
        try:
            W = np.loadtxt(wf)
        except Exception as e:  # noqa: BLE001
            r.fail(f"weights_{n}", f"unreadable: {e}")
            continue
        square = W.ndim == 2 and W.shape[0] == W.shape[1]
        sym = square and np.allclose(W, W.T, atol=1e-6)
        nonneg = np.all(W >= 0)
        nonzero = np.any(W > 0)
        status = PASS if (square and sym and nonneg and nonzero) else (
            FAIL if not (square and nonzero) else WARN)
        r.add(status, f"weights_{n}",
              f"shape={W.shape}, symmetric={sym}, nonneg={bool(nonneg)}, "
              f"nonzero={bool(nonzero)}, density={np.mean(W > 0):.3f}")
        if not nonzero:
            r.notes.append(f"weights_{n} is all-zero (template fallback / empty tractogram?)")

    if found == 0:
        # parcellation may exist without matrices
        if list(d.glob("atlas*_connectivity.nii.gz")):
            r.warn("connectome matrices", "parcellation present but no weights_*.txt")
            return r
        return r.skip("no weights_*.txt")

    # heatmap + degree for the 100-region connectome
    w100 = d / "weights_100.txt"
    if w100.exists():
        W = np.loadtxt(w100)
        ctx.add_figure(r, "conn_heatmap", "Connectome (100) log-weights",
                       lambda p: render2d.heatmap(W, p, "weights_100 (log)", log=True))
        ctx.add_figure(r, "conn_degree", "Node strength distribution (100)",
                       lambda p: render2d.histogram(W.sum(axis=1), p,
                                                    "node strength", "Σ weights"))

    dist = d / "distances_100.txt"
    if dist.exists():
        D = np.loadtxt(dist)
        finite = D[np.isfinite(D) & (D > 0)]
        r.add(PASS if finite.size and finite.max() < 1000 else WARN, "distances_100",
              f"mean={finite.mean():.1f} mm, max={finite.max():.1f} mm" if finite.size else "empty")
    return r
