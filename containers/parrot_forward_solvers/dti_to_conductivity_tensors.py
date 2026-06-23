"""DTI -> per-WM-tetrahedron anisotropic conductivity tensors (CGAL FEM).

Reads the subject diffusion tensor (cerebral, from dwitensor/; optionally the
warped cerebellar template tensor from the cerebellum stage), samples it at the
centroid of every White-Matter tetrahedron of the CGAL mesh, and turns each
diffusion tensor into a conductivity tensor with the *shape-preserving
orthotropic* model:

    sigma_i = sigma_iso * lambda_i / (lambda_1 lambda_2 lambda_3)^(1/3)

i.e. conductivity keeps all three DTI eigenvalue *ratios* (Tuch's sigma = k*D
linear hypothesis) and shares the DTI eigenvectors, while the geometric mean of
the three conductivities is pinned to the isotropic WM value the mesh already
uses (sigma_iso, read from the mesh's own conductivities.txt -- 0.348 S/m for
the ITIS table). Equivalently, in log-eigenvalue space we center the spread to a
zero mean (geometric mean = sigma_iso) and optionally cap it.

Guards:
  * FA-gate: tets whose tensor FA < --fa-threshold fall back to isotropic
    sigma_iso*I (CSF/grey partial-volume or mislabeled WM).
  * Ratio clamp: cap the final sigma_max/sigma_min at --max-anisotropy-ratio
    (default ~10, a robustness ceiling against low-SNR / crossing-fibre tensor
    failures -- NOT a physiological down-correction; see the project notes).
  * Out-of-FOV / zero tensors -> isotropic.

Mode (--anisotropy-mode):
  * shape (default): the per-voxel shape above, ratio capped AT --max-anisotropy-ratio.
  * fixed: drop the per-voxel shape; force a transversely-isotropic tensor at the
    ITIS ratio (0.733/0.231 ~= 3.17) aligned to the DTI principal eigenvector --
    orientation from DTI, magnitudes fixed. Same code path (clamp *to* the ratio).

Output (anisotropy/sub-<ID>/):
  conductivity_tensors.npy   (M, 3, 3) float64  -- one SPD conductivity tensor per WM tet
  wm_element_indices.npy     (M,)      int64    -- element index of each WM tet in the mesh
Storing only WM tets (not all N_elements) keeps memory small and matches the
hybrid label scheme the duneuro config uses (shared labels for isotropic tissues,
unique labels only for WM tets).

Tensors are read in MRtrix component order (D11 D22 D33 D12 D13 D23 =
xx yy zz xy xz yz) -- the order both dwitensor and the cerebellar warp emit.
"""
import os
import argparse
import numpy as np
import nibabel as nib

from mesh_io import read_mesh, read_conductivities, read_tissues

# MRtrix 6-vector order is (xx, yy, zz, xy, xz, yz). Map each component to its
# (row, col) in the symmetric 3x3 matrix.
_MRTRIX_IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def mrtrix6_to_mat(d6):
    """(..., 6) MRtrix tensor components -> (..., 3, 3) symmetric matrices."""
    M = np.zeros(d6.shape[:-1] + (3, 3), dtype=np.float64)
    for k, (i, j) in enumerate(_MRTRIX_IDX):
        M[..., i, j] = d6[..., k]
        M[..., j, i] = d6[..., k]
    return M


def world_mm_to_voxel(affine, pts_mm):
    """Map world-mm points (M,3) to nearest integer voxel indices (M,3)."""
    inv = np.linalg.inv(affine)
    vox = pts_mm @ inv[:3, :3].T + inv[:3, 3]
    return np.rint(vox).astype(np.int64)


def sample_volume(img, pts_mm, ncomp):
    """Nearest-neighbour sample an (X,Y,Z[,ncomp]) volume at world-mm points.

    Returns (values (M,ncomp), valid (M,)) where valid is False outside the FOV.
    Nearest-neighbour (no interpolation) keeps every sampled diffusion tensor a
    genuine SPD voxel value -- no log-Euclidean swelling. Tet size (~2 mm) is
    comparable to the DTI voxel, so this is adequate; trilinear log-Euclidean
    sampling is a possible future refinement.
    """
    data = np.asanyarray(img.dataobj)
    vox = world_mm_to_voxel(img.affine, pts_mm)
    shape = data.shape[:3]
    inb = np.all((vox >= 0) & (vox < np.array(shape)), axis=1)
    out = np.zeros((len(pts_mm), ncomp), dtype=np.float64)
    i, j, k = vox[inb].T
    out[inb] = data[i, j, k].reshape(-1, ncomp)
    valid = inb.copy()
    valid[inb] = np.abs(out[inb]).sum(axis=1) > 0  # zero tensor == no data
    return out, valid


def fractional_anisotropy(eigvals):
    """FA from eigenvalues (M,3), any order. 0 (isotropic) .. 1 (linear)."""
    md = eigvals.mean(axis=1, keepdims=True)
    num = ((eigvals - md) ** 2).sum(axis=1)
    den = (eigvals ** 2).sum(axis=1)
    return np.sqrt(1.5 * num / np.maximum(den, 1e-30))


def conductivity_eigenvalues(eigvals, mode, sigma_iso, max_ratio):
    """Map DTI eigenvalues (M,3, ascending) -> conductivity eigenvalues (M,3).

    Works in log space so the geometric mean is exactly sigma_iso (volume
    constraint) by construction. eigvals are ascending (lambda3<=lambda2<=lambda1),
    so column 2 is the principal axis.
    """
    l = np.log(np.maximum(eigvals, 1e-12))
    l = l - l.mean(axis=1, keepdims=True)  # geometric mean of exp(l) == 1

    if mode == 'shape':
        # cap sigma_max/sigma_min = exp(l_max - l_min) at max_ratio
        spread = l[:, 2] - l[:, 0]  # ln(lambda1/lambda3) >= 0
        s = np.minimum(1.0, np.log(max_ratio) / np.maximum(spread, 1e-9))
        l = l * s[:, None]
    elif mode == 'fixed':
        # transversely isotropic at the ITIS ratio, geometric mean 1, principal
        # axis = column 2 (largest DTI eigenvalue). Orientation from DTI only.
        lnR = np.log(3.17)
        l = np.full_like(l, -lnR / 3.0)
        l[:, 2] = 2.0 * lnR / 3.0
    else:
        raise ValueError(f"unknown --anisotropy-mode '{mode}' (expected shape|fixed)")

    return sigma_iso * np.exp(l)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject', required=True)
    p.add_argument('--output_dir', required=True, help='derivatives root (e.g. /derivatives)')
    p.add_argument('--mesh_path', required=True, help='CGAL .mesh (points in mm)')
    p.add_argument('--tissue_names', required=True, help='mesh labels.txt (<label>,<name>)')
    p.add_argument('--conductivities_path', required=True, help='mesh conductivities.txt (<label>,<S/m>)')
    p.add_argument('--cerebral_dti', required=True,
                   help='subject DTI tensor in T1/mesh space (MRtrix order, e.g. dwitensor/.../space-T1_model-dti_tensor.nii.gz)')
    p.add_argument('--cerebellar_dti', default=None,
                   help='optional warped cerebellar template DTI (MRtrix order, cerebellum/.../nonlinear_DTI.nii.gz)')
    p.add_argument('--cerebellum_mask', default=None,
                   help='optional cerebellum mask (>0); WM tets inside it are sampled from --cerebellar_dti')
    p.add_argument('--wm_tissue_name', default='Brain (White Matter)',
                   help='tissue name carrying white matter in labels.txt')
    p.add_argument('--anisotropy-mode', dest='mode', choices=['shape', 'fixed'], default='shape')
    p.add_argument('--max-anisotropy-ratio', dest='max_ratio', type=float, default=10.0,
                   help='robustness ceiling on sigma_max/sigma_min (shape mode)')
    p.add_argument('--fa-threshold', dest='fa_threshold', type=float, default=0.15,
                   help='below this FA the tet falls back to isotropic sigma_iso*I')
    args = p.parse_args()

    # --- mesh + WM identity + isotropic anchor -------------------------------
    nodes, tetra, labels = read_mesh(args.mesh_path)
    names = read_tissues(args.tissue_names)
    cond = read_conductivities(args.conductivities_path)
    try:
        wm_label = names.index(args.wm_tissue_name.strip().lower())
    except ValueError:
        raise ValueError(f"'{args.wm_tissue_name}' not found in {args.tissue_names}: {names}")
    sigma_iso = float(cond[wm_label])
    print(f"WM label = {wm_label} ('{args.wm_tissue_name}'), sigma_iso = {sigma_iso:.4f} S/m")

    wm = np.where(labels == wm_label)[0]
    n_wm = len(wm)
    print(f"{n_wm} white-matter tetrahedra of {len(tetra)} total")
    # centroids in world mm (read_mesh returns nodes in meters)
    centroids_mm = nodes[tetra[wm]].mean(axis=1) * 1000.0

    # --- route tets to cerebral vs cerebellar DTI ----------------------------
    use_cereb = np.zeros(n_wm, dtype=bool)
    if args.cerebellar_dti and args.cerebellum_mask:
        mvals, _ = sample_volume(nib.load(args.cerebellum_mask), centroids_mm, 1)
        use_cereb = mvals[:, 0] > 0.5
        print(f"{int(use_cereb.sum())} WM tets routed to the cerebellar template DTI")
    elif args.cerebellar_dti and not args.cerebellum_mask:
        print('[WARN] --cerebellar_dti given without --cerebellum_mask; ignoring cerebellar DTI.')

    # --- sample diffusion tensors --------------------------------------------
    D6 = np.zeros((n_wm, 6), dtype=np.float64)
    valid = np.zeros(n_wm, dtype=bool)

    cer = ~use_cereb
    D6[cer], valid[cer] = sample_volume(nib.load(args.cerebral_dti), centroids_mm[cer], 6)
    if use_cereb.any():
        D6[use_cereb], valid[use_cereb] = sample_volume(
            nib.load(args.cerebellar_dti), centroids_mm[use_cereb], 6)

    # --- diffusion -> conductivity -------------------------------------------
    D = mrtrix6_to_mat(D6)
    eigvals, eigvecs = np.linalg.eigh(D)          # ascending; eigvecs[m,:,i] <-> eigvals[m,i]
    fa = fractional_anisotropy(eigvals)
    sigma_eig = conductivity_eigenvalues(eigvals, args.mode, sigma_iso, args.max_ratio)

    # Sigma = V diag(sigma_eig) V^T
    tensors = np.einsum('mij,mj,mkj->mik', eigvecs, sigma_eig, eigvecs)

    # isotropic fallback: no data, low FA, or degenerate (non-positive) eigenvalues
    iso = (~valid) | (fa < args.fa_threshold) | (eigvals[:, 0] <= 0)
    tensors[iso] = sigma_iso * np.eye(3)
    print(f"{int(iso.sum())} WM tets isotropic "
          f"(no-data {int((~valid).sum())}, FA<{args.fa_threshold} {int((fa < args.fa_threshold).sum())})")

    aniso = ~iso
    if aniso.any():
        # report realised conductivity anisotropy on the kept tets
        se = np.sort(sigma_eig[aniso], axis=1)
        ratio = se[:, 2] / np.maximum(se[:, 0], 1e-12)
        print(f"anisotropic tets: {int(aniso.sum())}; sigma ratio median {np.median(ratio):.2f} "
              f"max {ratio.max():.2f}; geo-mean(sigma) median "
              f"{np.median(np.exp(np.log(np.maximum(sigma_eig[aniso],1e-12)).mean(1))):.4f} S/m")

    out_dir = os.path.join(args.output_dir, f'anisotropy/sub-{args.subject}')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'conductivity_tensors.npy'), tensors)
    np.save(os.path.join(out_dir, 'wm_element_indices.npy'), wm)
    print(f"wrote {tensors.shape} conductivity tensors -> {out_dir}")


if __name__ == '__main__':
    main()
