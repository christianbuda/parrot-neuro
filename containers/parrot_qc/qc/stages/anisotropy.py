"""Anisotropy QC (optional): WM conductivity tensors fed to the FEM leadfield.

Confirms the tensors are SPD with bounded anisotropy, and -- reusing the logic of
utils/qc_anisotropy.py -- that the principal direction points the anatomically
correct way (corpus callosum L-R, brainstem S-I). Also renders the pipeline's DEC
map (CC should be red, cortico-spinal tract blue).
"""
import numpy as np
import nibabel as nib

from ..checks import StageResult, PASS, WARN, FAIL
from .. import render2d
from ._common import first_existing

NAME = "anisotropy"
TITLE = "WM anisotropic conductivity"
DESCRIPTION = ("White-matter conductivity tensors for the FEM. Tensors should be SPD with bounded anisotropy; the DEC map should show the corpus callosum red (L-R) and the cortico-spinal tract blue (S-I) -- the standard direction check.")

CC_LABELS = (251, 252, 253, 254, 255)
BRAINSTEM = 16
AXES = ("x (L-R)", "y (A-P)", "z (S-I)")


def _aseg(ctx):
    for backend in ("fastsurfer", "freesurfer"):
        p = first_existing(ctx.stage_dir(backend) / "mri" / "aseg.mgz",
                           ctx.stage_dir(backend) / "mri" / "aparc+aseg.mgz")
        if p:
            return p
    return None


def _dominant_axis(pev):
    if len(pev) == 0:
        return None
    return int(np.argmax(np.bincount(np.argmax(np.abs(pev), axis=1), minlength=3)))


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    d = ctx.stage_dir("anisotropy")
    tens_f = d / "conductivity_tensors.npy"
    if not tens_f.exists():
        return r.skip("no conductivity_tensors.npy (no DWI anisotropy)")

    tensors = np.load(tens_f)
    if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
        r.fail("tensor array", f"unexpected shape {tensors.shape}")
        return r
    w, v = np.linalg.eigh(tensors)
    spd = np.mean(w[:, 0] > 0)
    r.add(PASS if spd > 0.99 else FAIL, "tensors SPD",
          f"{spd*100:.1f}% with all eigenvalues > 0")
    ratio = w[:, 2] / np.maximum(w[:, 0], 1e-12)
    r.add(PASS if np.nanmax(ratio) <= 12 else WARN, "anisotropy ratio",
          f"max λmax/λmin = {np.nanmax(ratio):.2f} (clamp ~10)")
    aniso = ratio > 1.05
    r.add(PASS, "anisotropic fraction",
          f"{int(aniso.sum())}/{len(tensors)} tets anisotropic")

    # principal-direction-by-region (reuses qc_anisotropy approach)
    pev = v[:, :, 2]
    mesh_f = ctx.stage_dir("tetmesh") / "tetrahedral_mesh.mesh"
    wm_f = d / "wm_element_indices.npy"
    aseg = _aseg(ctx)
    if mesh_f.exists() and wm_f.exists() and aseg is not None:
        try:
            import meshio
            m = meshio.read(str(mesh_f))
            tetra = m.cells_dict["tetra"].astype(np.int64)
            wm_idx = np.load(wm_f)
            centroids = m.points[tetra[wm_idx]].mean(axis=1)
            ai = nib.load(str(aseg))
            data = np.asanyarray(ai.dataobj)
            inv = np.linalg.inv(ai.affine)
            vox = np.rint(centroids @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
            shape = np.array(data.shape[:3])
            inb = np.all((vox >= 0) & (vox < shape), axis=1)
            lab = np.zeros(len(centroids), dtype=np.int64)
            i, j, k = vox[inb].T
            lab[inb] = data[i, j, k]
            cc = aniso & np.isin(lab, CC_LABELS)
            bs = aniso & (lab == BRAINSTEM)
            cc_axis, bs_axis = _dominant_axis(pev[cc]), _dominant_axis(pev[bs])
            r.add(PASS if cc_axis == 0 else WARN, "corpus callosum direction",
                  f"dominant {AXES[cc_axis] if cc_axis is not None else 'n/a'} (expect L-R)")
            r.add(PASS if bs_axis == 2 else WARN, "brainstem direction",
                  f"dominant {AXES[bs_axis] if bs_axis is not None else 'n/a'} (expect S-I)")
        except Exception as e:  # noqa: BLE001
            r.warn("direction-by-region", f"could not evaluate: {e}")

    dec = d / "dec_principal_direction.nii.gz"
    if dec.exists():
        ctx.add_figure(r, "dec_map", "DEC map (CC red, CST blue)",
                       lambda p: render2d.rgb_mosaic(dec, p, "principal direction (DEC)"))
    return r
