"""QC the WM anisotropic-conductivity tensors: do the fibre directions point the
anatomically correct way?

The numeric checks elsewhere confirm the tensors are SPD with the right
magnitudes/ratios, but not that the principal eigenvector points the right way
in the brain -- a silent axis/handedness/component-order flip would pass all of
those yet give a plausible-but-wrong leadfield. This script eigendecomposes the
*actual* conductivity tensors fed to DUNEuro and checks the principal direction
against a FreeSurfer aseg:

  * Corpus callosum (aseg 251-255): fibres run left-right  -> |e_x| (RAS x) dominant
  * Brain-stem      (aseg 16):       fibres run superior-inferior -> |e_z| dominant

It also writes a directionally-encoded colour volume (DEC map) on the aseg grid
(R=|e_x| L-R, G=|e_y| A-P, B=|e_z| S-I, scaled by anisotropy) for eyeballing in
a viewer: the corpus callosum should be red, the cortico-spinal tract blue.

All frames are scanner-RAS: MRtrix tensors, the mesh, and the aseg affine all
share it, so no extra rotation is applied (that is exactly what we are testing).

Self-contained (numpy + nibabel + meshio); run it inside the forward_solvers
image, e.g.:
    python3 utils/qc_anisotropy.py \
        --tensors    derivatives/anisotropy/sub-ID/conductivity_tensors.npy \
        --wm_indices derivatives/anisotropy/sub-ID/wm_element_indices.npy \
        --mesh_path  derivatives/tetmesh/sub-ID/tetrahedral_mesh.mesh \
        --aseg       derivatives/fastsurfer/sub-ID/mri/aseg.mgz \
        --dec_out    derivatives/anisotropy/sub-ID/dec_principal_direction.nii.gz
"""
import argparse
import numpy as np
import nibabel as nib
import meshio

CC_LABELS = (251, 252, 253, 254, 255)   # FreeSurfer corpus callosum
BRAINSTEM_LABEL = 16
AXES = ('x (L-R)', 'y (A-P)', 'z (S-I)')


def tet_centroids_mm(mesh_path, wm_idx):
    """Centroids (mm, scanner-RAS) of the white-matter tetrahedra."""
    mesh = meshio.read(mesh_path)
    points = mesh.points.astype(np.float64)            # mesh points are in mm
    tetra = mesh.cells_dict['tetra'].astype(np.int64)
    return points[tetra[wm_idx]].mean(axis=1)


def principal_eigenvectors(tensors):
    """(M,3,3) -> principal eigenvector (M,3) and anisotropy ratio lambda_max/lambda_min."""
    w, v = np.linalg.eigh(tensors)            # ascending
    pev = v[:, :, 2]                          # eigenvector of largest eigenvalue
    ratio = w[:, 2] / np.maximum(w[:, 0], 1e-12)
    return pev, ratio


def sample_labels(aseg, pts_mm):
    data = np.asanyarray(aseg.dataobj)
    inv = np.linalg.inv(aseg.affine)
    vox = np.rint(pts_mm @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
    shape = np.array(data.shape[:3])
    inb = np.all((vox >= 0) & (vox < shape), axis=1)
    lab = np.zeros(len(pts_mm), dtype=np.int64)
    i, j, k = vox[inb].T
    lab[inb] = data[i, j, k]
    return lab


def report(name, pev):
    if len(pev) == 0:
        print(f"  {name}: no anisotropic tets found"); return
    comp = np.abs(pev).mean(axis=0)
    dom = np.bincount(np.argmax(np.abs(pev), axis=1), minlength=3) / len(pev)
    print(f"  {name}: n={len(pev)}")
    print(f"     mean |eigvec| per axis: x={comp[0]:.3f}  y={comp[1]:.3f}  z={comp[2]:.3f}")
    print(f"     dominant-axis fraction: x={dom[0]:.2f}  y={dom[1]:.2f}  z={dom[2]:.2f}  "
          f"-> mostly {AXES[int(np.argmax(dom))]}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tensors', required=True, help='anisotropy/.../conductivity_tensors.npy')
    p.add_argument('--wm_indices', required=True, help='anisotropy/.../wm_element_indices.npy')
    p.add_argument('--mesh_path', required=True, help='CGAL tetrahedral_mesh.mesh (points in mm)')
    p.add_argument('--aseg', required=True, help='FreeSurfer aseg (.mgz/.nii.gz) with CC + brainstem labels')
    p.add_argument('--min_ratio', type=float, default=1.05,
                   help='exclude near-isotropic tets (lambda_max/lambda_min below this)')
    p.add_argument('--dec_out', default=None, help='optional output path for the DEC colour NIfTI')
    args = p.parse_args()

    tensors = np.load(args.tensors)
    wm_idx = np.load(args.wm_indices)
    centroids_mm = tet_centroids_mm(args.mesh_path, wm_idx)

    pev, ratio = principal_eigenvectors(tensors)
    aniso = ratio > args.min_ratio                              # drop isotropic-fallback tets
    print(f"WM tets: {len(tensors)} total, {int(aniso.sum())} anisotropic (ratio>{args.min_ratio})")

    aseg = nib.load(args.aseg)
    labels = sample_labels(aseg, centroids_mm)

    print("\n--- principal direction by region (eigenvectors of the conductivity tensors) ---")
    cc = aniso & np.isin(labels, CC_LABELS)
    bs = aniso & (labels == BRAINSTEM_LABEL)
    report("corpus callosum  (expect x / L-R)", pev[cc])
    report("brain-stem       (expect z / S-I)", pev[bs])
    report("all anisotropic WM", pev[aniso])

    if args.dec_out:
        shape = np.asanyarray(aseg.dataobj).shape[:3]
        dec = np.zeros(shape + (3,), dtype=np.float32)
        inv = np.linalg.inv(aseg.affine)
        vox = np.rint(centroids_mm[aniso] @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
        col = np.abs(pev[aniso]) * np.clip((ratio[aniso] - 1) / 4.0, 0, 1)[:, None]
        inb = np.all((vox >= 0) & (vox < np.array(shape)), axis=1)
        v = vox[inb]
        dec[v[:, 0], v[:, 1], v[:, 2]] = col[inb]
        nib.Nifti1Image(dec, aseg.affine).to_filename(args.dec_out)
        print(f"\nwrote DEC colour volume -> {args.dec_out}  (R=L-R, G=A-P, B=S-I; CC should be red)")


if __name__ == '__main__':
    main()
