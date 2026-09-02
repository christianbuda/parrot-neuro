#!/usr/bin/env python3
"""Stage published brain *templates* as a Parrot-ready BIDS dataset.

Unlike the cohort stagers (``lemon.py``, ``hcp.py``) this one has no subject loop over a
directory tree: it stages two fixed, hand-picked template heads as pseudo-subjects, so
Parrot can produce subject-independent reference / fallback forward models.

  sub-MNI09b   ICBM152 2009b nonlinear asymmetric, 0.5 mm  (T1w + T2w + PDw)
  sub-OMM1     Oxford-MM-1 v1, 1 mm                        (T1w + FLAIR, T1-only recon)

A third candidate, the IIT Human Brain Atlas, was **dropped**: its ``IITmean_t1`` is
skull-stripped (brain only), and charm / BEM / electrode placement / the leadfield all
need an intact head, so it cannot yield a forward model.

Neither template ships raw DWI (OMM has only a *fitted* tensor; IIT's "HARDI" file is
28 spherical-harmonic ODF coefficients), and there is no ``.bval``/``.bvec`` anywhere, so
QSIPrep/QSIRecon cannot run for either. No ``dwi/`` folder is staged. Instead, OMM's WM
anisotropy is obtained by **injecting its fitted tensor straight into the derivatives
tree** at the path the orchestrator's ``anisotropy`` stage gates on -- see
``inject_omm_tensor`` below. That stage is gated purely on the file existing (there is no
``HAS_DWI`` check), so writing it before launching is all that is needed.

Copies here are **byte-faithful**: MNI's ``.nii`` is gzipped and OMM's ``.nii.gz`` copied
verbatim, with no nibabel round-trip. The cohort stagers use ``common.clean_voxel_size``
to snap a float32 voxel-size artifact, but both templates already have exact voxel sizes
(0.5 and 1.0), so cleaning would be a pure no-op on the geometry while needlessly
rewriting MNI's int16 + ``scl_slope``/``scl_inter`` intensity scaling. ``assert_clean_
geometry`` asserts the precondition that lets us skip it.

Runs INSIDE the parrot_mri_reconstruction image (the host has no nibabel), with the
template source root mounted read-only at /src and the target BIDS root read-write at
/dst. Launch via bin/stage.sh.
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

from common import write_dataset_description, write_participants_tsv

SRC_ROOT = Path("/src")  # dir holding mni_icbm152_nlin_asym_09b/ and Oxford-MM-1/
DST_ROOT = Path("/dst")  # BIDS root; Parrot's derivatives/ is nested underneath

MNI_DIR = "mni_icbm152_nlin_asym_09b"
OMM_DIR = "Oxford-MM-1"

MNI_URL = "https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/"
OMM_URL = "https://git.fmrib.ox.ac.uk/fsl/oxford-mm-templates"

SUBJECTS = ["sub-MNI09b", "sub-OMM1"]

# Orchestrator override columns, IN POSITIONAL ORDER (parsed by position:
# col4=skip-T2-registration, col5=no-neck, col6=mp2rage).
OVERRIDE_COLS = ["skip_t2_registration", "no_neck", "mp2rage"]
# Both templates bottom out at z = -72 mm, i.e. no usable neck coverage, so charm must
# get --noneck. Neither T1 is an MP2RAGE UNI, so no MPRAGEise.
DEFAULT_OVERRIDE = {"skip_t2_registration": False, "no_neck": True, "mp2rage": False}
SUBJECT_OVERRIDES: dict[str, dict[str, bool]] = {}


# =============================================================================
# byte-faithful copying
# =============================================================================

def assert_clean_geometry(path: Path) -> nib.Nifti1Image:
    """Assert a source volume needs no header hygiene, so a byte copy is safe.

    The cohort stagers snap float32 voxel-size noise (1.0000009 -> 1.0) because
    FastSurfer's surf-stage conform rejects vox_size > 1.0. Templates are published on
    exact grids, so instead of rewriting the file we *check* the precondition -- and fail
    loudly if a future source ever violates it, rather than silently shipping a header
    FastSurfer will choke on.
    """
    img = nib.load(path)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    if not np.allclose(zooms, np.round(zooms, 4), atol=0, rtol=0):
        raise ValueError(f"{path.name}: voxel sizes {zooms} are not exact; needs cleaning")
    if zooms.max() > 1.0:
        raise ValueError(f"{path.name}: voxel size {zooms.max()} > 1.0 breaks FastSurfer conform")
    if img.ndim != 3:
        raise ValueError(f"{path.name}: expected a 3D anatomical, got shape {img.shape}")
    return img


def copy_nii(src: Path, dst: Path) -> None:
    """Copy a NIfTI byte-faithfully, gzipping a bare ``.nii`` on the way.

    No nibabel round-trip: the header (including MNI's int16 ``scl_slope``/``scl_inter``
    intensity scaling) reaches the destination bit-for-bit.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        shutil.copyfile(src, dst)
    else:
        with open(src, "rb") as fh_in, gzip.open(dst, "wb") as fh_out:
            shutil.copyfileobj(fh_in, fh_out, length=16 << 20)


def write_sidecar(dst_nii: Path, source: Path, url: str, description: str) -> None:
    """Write the BIDS .json sidecar recording provenance for a staged volume."""
    meta = {
        "Sources": [str(source)],
        "SourceDatasetURL": url,
        "Description": description,
        "GeneratedBy": [{"Name": "parrot stage.sh templates", "Description": __doc__.splitlines()[0]}],
    }
    dst_nii.with_name(dst_nii.name.replace(".nii.gz", ".json")).write_text(json.dumps(meta, indent=2))


def stage_volume(src: Path, dst: Path, url: str, description: str) -> None:
    assert_clean_geometry(src)
    copy_nii(src, dst)
    write_sidecar(dst, src, url, description)
    print(f"  {dst.name:32s} <- {src.name}")


# =============================================================================
# subjects
# =============================================================================

def stage_mni09b() -> None:
    """ICBM152 2009b hires: T1w + T2w (charm) + PDw (archival)."""
    sub, src, out = "sub-MNI09b", SRC_ROOT / MNI_DIR, DST_ROOT / "sub-MNI09b" / "anat"
    print(f"\n=== staging {sub} (ICBM152 2009b, 0.5 mm) ===")
    stage_volume(src / "mni_icbm152_t1_tal_nlin_asym_09b_hires.nii",
                 out / f"{sub}_T1w.nii.gz", MNI_URL, "ICBM152 2009b nonlinear asymmetric T1w template, 0.5 mm")
    stage_volume(src / "mni_icbm152_t2_tal_nlin_asym_09b_hires.nii",
                 out / f"{sub}_T2w.nii.gz", MNI_URL, "ICBM152 2009b nonlinear asymmetric T2w template, 0.5 mm")
    # PDw is archival: Parrot auto-discovers only T1w/T2w, so this is never consumed.
    stage_volume(src / "mni_icbm152_pd_tal_nlin_asym_09b_hires.nii",
                 out / f"{sub}_PDw.nii.gz", MNI_URL, "ICBM152 2009b nonlinear asymmetric PDw template, 0.5 mm (archival)")


def stage_omm1() -> None:
    """Oxford-MM-1: T1w (whole head) + FLAIR (archival) + DTI sourcedata.

    The second OMM contrast is a T2-FLAIR, not a T2. charm keys on CSF signal, which
    FLAIR nulls by construction, so feeding it would actively mislead the segmentation.
    Naming it ``_FLAIR`` keeps the recon T1-only: Parrot auto-discovers T1w/T2w only and
    no longer consults FLAIR at all.
    """
    sub, src, out = "sub-OMM1", SRC_ROOT / OMM_DIR, DST_ROOT / "sub-OMM1" / "anat"
    print(f"\n=== staging {sub} (Oxford-MM-1, 1 mm) ===")
    stage_volume(src / "OMM-1_T1_head.nii.gz",
                 out / f"{sub}_T1w.nii.gz", OMM_URL, "Oxford-MM-1 whole-head T1 template, 1 mm")
    stage_volume(src / "OMM-1_T2_FLAIR_head.nii.gz",
                 out / f"{sub}_FLAIR.nii.gz", OMM_URL,
                 "Oxford-MM-1 whole-head T2-FLAIR template, 1 mm (archival; NOT a T2 -- FLAIR nulls CSF, which charm needs)")

    # sourcedata: the fitted tensor we inject downstream + the shipped derived maps we
    # validate the conversion against.
    sd = DST_ROOT / "sourcedata" / "omm1_dti"
    sd.mkdir(parents=True, exist_ok=True)
    for name in ("OMM-1_DTI_tensor.nii.gz", "OMM-1_DTI_FA.nii.gz", "OMM-1_DTI_L1.nii.gz",
                 "OMM-1_DTI_L2.nii.gz", "OMM-1_DTI_L3.nii.gz", "OMM-1_DTI_V1.nii.gz",
                 "OMM-1_DTI_mask_average.nii.gz"):
        shutil.copyfile(src / name, sd / name)
    (sd / "README.md").write_text(
        "# Oxford-MM-1 fitted DTI (sourcedata)\n\n"
        f"Copied verbatim from `{src}` ({OMM_URL}).\n\n"
        "`OMM-1_DTI_tensor.nii.gz` is a *fitted* tensor -- there is no raw DWI for this\n"
        "template, so QSIPrep/QSIRecon cannot run. It is converted (FSL -> MRtrix component\n"
        "order, image -> scanner-RAS frame) and injected into\n"
        "`derivatives/dwitensor/sub-OMM1/` by `utils/staging/templates.py`, which is where the\n"
        "orchestrator's `anisotropy` stage picks it up. The other files are the shipped\n"
        "eigenvalue/eigenvector/FA maps, kept as validation references for that conversion.\n"
    )
    print(f"  sourcedata/omm1_dti/          <- tensor + FA/L1/L2/L3/V1/mask references")


# =============================================================================
# OMM DTI tensor -> injected derivatives/dwitensor
# =============================================================================

# FSL stores the lower triangle in the order (xx, xy, xz, yy, yz, zz); the Parrot
# consumer (containers/parrot_forward_solvers/dti_to_conductivity_tensors.py) wants
# MRtrix order (xx, yy, zz, xy, xz, yz). This index maps the former onto the latter.
FSL_TO_MRTRIX = [0, 3, 5, 1, 2, 4]

# --- structure-tensor frame test, tuning constants (see decide_tensor_frame) ---
ST_FA_MIN = 0.35          # white-matter core only
ST_ERODE_ITER = 3         # stay off the brain edge, where FA gradients are meaningless
ST_GRAD_SIGMA = 1.5       # voxels; pre-smoothing before the FA gradient
ST_TENSOR_SIGMA = 2.0     # voxels; the structure tensor's averaging window
ST_TUBE_PERP = 0.40       # lam1/lam0: both perpendicular directions must be sampled (tube, not sheet)
ST_TUBE_AXIS = 0.40       # lam2/lam1: the axis eigenvalue must be clearly the smallest
ST_OBLIQUE = 0.35         # |axis_x| and |axis_z| must BOTH exceed this (see below)
ST_MIN_VOXELS = 2000      # too few oblique tube voxels -> the test has no power
ST_MIN_MARGIN = 0.02      # absolute mean |V1.axis| difference required
ST_MIN_TSTAT = 10.0       # and it must be this many standard errors from zero


def fsl6_to_mrtrix6(d6: np.ndarray) -> np.ndarray:
    """(..., 6) FSL tensor components -> (..., 6) MRtrix component order."""
    return d6[..., FSL_TO_MRTRIX]


def mrtrix6_to_mat(d6: np.ndarray) -> np.ndarray:
    """(..., 6) MRtrix components (xx, yy, zz, xy, xz, yz) -> (..., 3, 3) symmetric."""
    M = np.zeros(d6.shape[:-1] + (3, 3), dtype=np.float64)
    for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
        M[..., i, j] = d6[..., k]
        M[..., j, i] = d6[..., k]
    return M


def mat_to_mrtrix6(M: np.ndarray) -> np.ndarray:
    """(..., 3, 3) symmetric -> (..., 6) MRtrix components."""
    return np.stack([M[..., i, j] for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]], axis=-1)


def fa_from_mat(M: np.ndarray) -> np.ndarray:
    """Fractional anisotropy of a batch of symmetric 3x3 tensors (frame-invariant)."""
    ev = np.linalg.eigvalsh(M)
    md = ev.mean(-1, keepdims=True)
    num = np.sqrt((3.0 / 2.0) * ((ev - md) ** 2).sum(-1))
    den = np.sqrt((ev ** 2).sum(-1))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def verify_component_order(d6_fsl: np.ndarray, src: Path) -> None:
    """Assert the FSL->MRtrix reindex reproduces the shipped eigenvalue maps.

    Eigenvalues are invariant under the frame question below, so this isolates and
    validates the *component order* on its own.
    """
    d6 = fsl6_to_mrtrix6(d6_fsl)
    ev = np.linalg.eigvalsh(mrtrix6_to_mat(d6))  # ascending
    for shipped_name, ours in (("L1", ev[..., 2]), ("L2", ev[..., 1]), ("L3", ev[..., 0])):
        shipped = np.asarray(nib.load(src / f"OMM-1_DTI_{shipped_name}.nii.gz").dataobj, dtype=np.float64)
        err = np.abs(shipped - ours).max()
        print(f"    {shipped_name}: max|shipped - reordered| = {err:.3e}")
        if err > 1e-8:
            raise ValueError(
                f"FSL->MRtrix reindex {FSL_TO_MRTRIX} does not reproduce {shipped_name} "
                f"(max err {err:.3e}); the source component order is not what we assume."
            )


def decide_tensor_frame(mats: np.ndarray, fa: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> bool:
    """Decide whether the tensor components need a frame change, and return True if so.

    FSL expresses tensor components in the **image** frame; the Parrot consumer reads
    them in **scanner-RAS**. OMM's affine direction matrix is diag(-1, 1, 1), so if the
    image-frame reading holds, the RAS tensor is ``R S R^T`` with ``R = diag(-1, 1, 1)``
    -- i.e. ``xy`` and ``xz`` flip sign.

    Getting this wrong is *silent*: a reflection preserves every eigenvalue, so FA/MD/L1
    checks pass either way, and it even preserves axis-aligned tracts, so the usual
    corpus-callosum (L-R) and corticospinal (S-I) sanity checks pass too. Left-right
    symmetry tests are degenerate here as well -- the ambiguity **is** the mirror.

    So we decide it against a reference whose frame is unambiguous: the local geometry of
    the FA map itself. Fibre bundles are tubular, so FA is near-constant along the bundle
    axis and falls off across it; the FA gradient is therefore perpendicular to the axis,
    and the axis is the *smallest*-eigenvalue eigenvector of the FA structure tensor
    ``G_sigma * (grad FA)(grad FA)^T``. That comes purely from spatial derivatives mapped
    through the affine, so its frame is fixed by definition.

    We then score both hypotheses by mean ``|V1 . axis|`` over voxels where the axis is
    well defined (tubular, not sheet-like) *and* clearly oblique -- ``|axis_x|`` and
    ``|axis_z|`` both substantial, because an axis-aligned voxel is invariant under the
    flip and carries no information.
    """
    print("\n  --- component-frame test (structure tensor of FA) ---")
    A = affine[:3, :3]

    # 1. FA structure tensor, with gradients expressed in WORLD axes:
    #    grad_world = A^-T grad_voxel  ->  (g_vox @ inv(A)) row-wise.
    fa_s = ndimage.gaussian_filter(np.where(mask, fa, 0.0), ST_GRAD_SIGMA)
    g_vox = np.stack(np.gradient(fa_s), axis=-1)
    g = g_vox @ np.linalg.inv(A)
    S = np.empty(fa.shape + (3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(i, 3):
            Sij = ndimage.gaussian_filter(g[..., i] * g[..., j], ST_TENSOR_SIGMA)
            S[..., i, j] = Sij
            S[..., j, i] = Sij

    # 2. Candidate voxels: WM core, safely inside the brain (edge gradients are artefact).
    interior = ndimage.binary_erosion(mask, iterations=ST_ERODE_ITER)
    cand = interior & (fa > ST_FA_MIN)
    idx = np.nonzero(cand)
    lam, vec = np.linalg.eigh(S[idx])          # ascending eigenvalues
    lam0, lam1, lam2 = lam[:, 2], lam[:, 1], lam[:, 0]   # largest -> smallest
    axis = vec[:, :, 0]                         # eigenvector of the SMALLEST eigenvalue

    with np.errstate(divide="ignore", invalid="ignore"):
        tube = (np.nan_to_num(lam1 / lam0) > ST_TUBE_PERP) & (np.nan_to_num(lam2 / lam1) < ST_TUBE_AXIS)
    oblique = (np.abs(axis[:, 0]) > ST_OBLIQUE) & (np.abs(axis[:, 2]) > ST_OBLIQUE)
    sel = tube & oblique
    n = int(sel.sum())
    print(f"    candidate WM voxels {cand.sum()} -> tubular {int(tube.sum())} -> oblique {n}")
    if n < ST_MIN_VOXELS:
        raise ValueError(f"only {n} oblique tubular voxels (< {ST_MIN_VOXELS}); frame test has no power")

    # 3. Score both hypotheses. The flipped V1 is exactly diag(-1,1,1) @ V1, so the two
    #    scores are a paired comparison on the same voxels.
    axis_s = axis[sel]
    v1 = np.linalg.eigh(mats[idx[0][sel], idx[1][sel], idx[2][sel]])[1][:, :, 2]
    dot_plain = np.abs((v1 * axis_s).sum(-1))
    v1_flip = v1 * np.array([-1.0, 1.0, 1.0])
    dot_flip = np.abs((v1_flip * axis_s).sum(-1))

    d = dot_flip - dot_plain
    margin, sem = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    tstat = margin / sem if sem > 0 else np.inf
    print(f"    mean |V1.axis|  as-is (scanner-RAS) = {dot_plain.mean():.4f}")
    print(f"    mean |V1.axis|  flipped (xy,xz)     = {dot_flip.mean():.4f}")
    print(f"    paired margin (flipped - as-is)     = {margin:+.4f}  (SEM {sem:.4f}, t = {tstat:+.1f}, n = {n})")

    if abs(margin) < ST_MIN_MARGIN or abs(tstat) < ST_MIN_TSTAT:
        raise ValueError(
            f"frame test INCONCLUSIVE: margin {margin:+.4f} (need |margin| >= {ST_MIN_MARGIN} "
            f"and |t| >= {ST_MIN_TSTAT}). Refusing to guess -- investigate before injecting."
        )
    flip = margin > 0
    print(f"    DECISION: {'FLIP xy/xz (components were in the image frame)' if flip else 'NO FLIP (components already scanner-RAS)'}")

    # Secondary, more general read of the same data: the strict discriminability
    # condition is |axis_x| large AND the axis not purely x-aligned. Reported only as a
    # consistency check on the (narrower) |x| & |z| criterion actually used above.
    gen = (np.abs(axis[:, 0]) > ST_OBLIQUE) & (np.hypot(axis[:, 1], axis[:, 2]) > ST_OBLIQUE) & tube
    if gen.sum() >= ST_MIN_VOXELS:
        a2 = axis[gen]
        w = np.linalg.eigh(mats[idx[0][gen], idx[1][gen], idx[2][gen]])[1][:, :, 2]
        m2 = float((np.abs(((w * np.array([-1.0, 1.0, 1.0])) * a2).sum(-1)) - np.abs((w * a2).sum(-1))).mean())
        print(f"    cross-check on the general oblique set (n = {int(gen.sum())}): margin {m2:+.4f}"
              f" -> {'FLIP' if m2 > 0 else 'NO FLIP'} ({'agrees' if (m2 > 0) == flip else 'DISAGREES'})")
        if (m2 > 0) != flip:
            raise ValueError("the two oblique-voxel criteria disagree on the frame; refusing to guess")
    return flip


def inject_omm_tensor() -> None:
    """Convert OMM's fitted DTI tensor and write it where ``anisotropy`` looks for it.

    Space is the easy half: the tensor shares its affine exactly with
    ``OMM-1_T1_head.nii.gz``, which we stage byte-faithfully as the BIDS T1w, and which
    ``ingest.py`` copies unchanged into ``raw/sub-OMM1/T1.nii.gz``. Mesh space *is* that
    world space, so the spatial transform is the identity -- the same situation as the
    HCP path, which likewise writes ``space-T1`` directly. What is left is the two
    *component* conventions: order (``verify_component_order``) and frame
    (``decide_tensor_frame``).
    """
    print("\n=== injecting sub-OMM1 DTI tensor into derivatives/dwitensor ===")
    src = DST_ROOT / "sourcedata" / "omm1_dti"
    t1 = DST_ROOT / "sub-OMM1" / "anat" / "sub-OMM1_T1w.nii.gz"

    timg = nib.load(src / "OMM-1_DTI_tensor.nii.gz")
    t1img = nib.load(t1)
    if not np.allclose(timg.affine, t1img.affine, atol=0, rtol=0):
        raise ValueError("OMM tensor affine differs from the staged T1w affine; identity assumption broken")
    if timg.shape != t1img.shape + (6,):
        raise ValueError(f"unexpected tensor shape {timg.shape} (expected {t1img.shape + (6,)})")

    d6_fsl = np.asarray(timg.dataobj, dtype=np.float64)
    print("  component ORDER: assuming FSL (xx,xy,xz,yy,yz,zz) -> MRtrix via", FSL_TO_MRTRIX)
    verify_component_order(d6_fsl, src)

    d6 = fsl6_to_mrtrix6(d6_fsl)
    mats = mrtrix6_to_mat(d6)

    fa = np.asarray(nib.load(src / "OMM-1_DTI_FA.nii.gz").dataobj, dtype=np.float64)
    mask = np.asarray(nib.load(src / "OMM-1_DTI_mask_average.nii.gz").dataobj) > 0.5
    if decide_tensor_frame(mats, fa, mask, timg.affine):
        # R S R^T with R = diag(-1, 1, 1): only the off-diagonals touching x change sign.
        d6[..., 3] *= -1.0   # xy
        d6[..., 4] *= -1.0   # xz
        mats = mrtrix6_to_mat(d6)

    # Sanitize exactly as bin/make_dwitensor.sh does: a single non-finite component would
    # poison the batched eigh downstream, so zero the whole voxel.
    bad = ~np.isfinite(d6).all(axis=-1)
    if bad.any():
        d6[bad] = 0.0
        mats[bad] = 0.0
    print(f"  non-finite voxels zeroed: {int(bad.sum())}")

    # FA recomputed from what we are about to write must match the shipped map.
    fa_ours = fa_from_mat(mats)
    fa_err = float(np.abs(fa_ours - fa)[mask].max())
    print(f"  max|FA(recomputed) - FA(shipped)| inside mask = {fa_err:.3e}")
    if fa_err > 1e-5:
        raise ValueError(f"recomputed FA disagrees with the shipped map (max err {fa_err:.3e})")

    out = DST_ROOT / "derivatives" / "dwitensor" / "sub-OMM1"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "sub-OMM1_space-T1_model-dti_tensor.nii.gz"
    nib.save(nib.Nifti1Image(d6.astype(np.float32), timg.affine, timg.header), dst)
    (out / "sub-OMM1_space-T1_model-dti_tensor.json").write_text(json.dumps({
        "Sources": ["sourcedata/omm1_dti/OMM-1_DTI_tensor.nii.gz"],
        "SourceDatasetURL": OMM_URL,
        "Description": ("Oxford-MM-1 fitted DTI tensor, injected in place of a dwitensor-stage fit "
                        "(the template ships no raw DWI, so QSIPrep/QSIRecon cannot run). Converted "
                        "from FSL component order to MRtrix (xx,yy,zz,xy,xz,yz) and into the "
                        "scanner-RAS frame; spatial transform is the identity because the tensor "
                        "shares its world frame with the T1 the mesh is built in."),
        "GeneratedBy": [{"Name": "parrot stage.sh templates", "Description": "utils/staging/templates.py"}],
    }, indent=2))
    print(f"  wrote {dst}")


# =============================================================================

def main() -> None:
    if sys.argv[1:]:
        print(f"[WARN] templates.py stages a fixed pair {SUBJECTS}; ignoring args {sys.argv[1:]}")
    stage_mni09b()
    stage_omm1()
    write_dataset_description(
        DST_ROOT,
        "Brain templates (staged for Parrot)",
        source_url=[MNI_URL, OMM_URL],
    )
    write_participants_tsv(
        DST_ROOT,
        SUBJECTS,
        override_cols=OVERRIDE_COLS,
        subject_overrides=SUBJECT_OVERRIDES,
        default_override=DEFAULT_OVERRIDE,
    )
    # Last, so a failed frame test leaves a complete, usable BIDS dataset behind and only
    # blocks the (optional) anisotropy extra.
    inject_omm_tensor()
    print("\nDone.")


if __name__ == "__main__":
    main()
