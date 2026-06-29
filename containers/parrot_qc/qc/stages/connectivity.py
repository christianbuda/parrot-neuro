"""Connectivity QC (optional): structural connectome matrices + parcellations."""
import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render2d

NAME = "connectivity"
TITLE = "Structural connectome"

_RES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("connectivity")
    if not d.exists():
        return r.skip("no connectivity/ (no DWI / no template)")

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
