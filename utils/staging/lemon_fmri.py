#!/usr/bin/env python3
"""Import LEMON preprocessed resting-state fMRI into ``derivatives/fMRI``.

Per subject this writes two things under ``<dataset>/derivatives/fMRI/sub-<ID>/``:

  1. **The preprocessed BOLD volume**, copied verbatim
     (``..._space-native_desc-preproc_bold.nii.gz``) plus its ``confounds.txt`` as a
     provenance record.
  2. **Parcellated region time series** for every Schaefer resolution (100..1000),
     for BOTH atlas forms the Parrot pipeline produces:
       * ``desc-full`` -- the full label atlas (``atlas/sub-<ID>/atlas{N}.nii.gz``:
         Schaefer cortex + subcortical/cerebellar/hippocampal labels, raw ids), and
       * ``desc-conn`` -- the connectome-node atlas
         (``connectivity/sub-<ID>/atlas{N}_connectivity.nii.gz``: renumbered 0..M so
         row *i* == structural-connectome node *i*).

Why a registration step is needed
----------------------------------
LEMON's ``native`` BOLD is resampled into LEMON's own reoriented, FreeSurfer-conformed
``mp2rage_brain`` frame, which is NOT the raw-acquisition frame Parrot's atlases live in
(``lemon.py`` copies the raw T1w verbatim; the atlas affine matches it exactly). The two
frames differ by a rigid reorientation that LEMON never shipped as a transform (no
``.mat/.xfm/.warp`` in the preprocessed tree), so a plain world-affine resample lands
labels on the wrong tissue (verified NCC ~= 0.11). We recover the reorientation with a
per-subject **rigid** registration between LEMON's ``mp2rage_brain`` (BOLD frame) and
Parrot's raw T1w (atlas frame) -- same MP2RAGE, so it is near-exact -- and apply it to
resample each atlas onto the BOLD grid.

Why no denoising here
---------------------
LEMON's ``native`` output is already analysis-ready for connectivity (Mendes et al. 2019):
nuisance/motion confounds regressed (aCompCor + GLM; re-regressing the shipped
``confounds.txt`` explains ~3% variance), **band-pass filtered 0.01-0.1 Hz** -- i.e. already
the conventional resting-state FC band -- then mean-centered + variance-normalized (z-scored),
first 5 dummy volumes dropped (652 vols, TR = 1.4 s). So extraction is just a region average;
no filtering, confound regression, or standardisation is needed here. Pearson correlation of
these series downstream gives the functional connectome.

An optional ``--highpass`` (default 0 = OFF) applies an ADDITIONAL high-pass. Note LEMON
already high-passed at 0.01 Hz, so the old 0.01 Hz default merely double-filtered the same
cutoff (a few-percent extra roll-off at 0.01-0.03 Hz, no drift to remove) -- hence disabled
by default now. Leave it off unless you deliberately want a stricter high-pass than LEMON's.

Runs INSIDE the parrot_mri_reconstruction image (it has antspyx -- see
mni_registration.py -- plus numpy/scipy/nibabel; nilearn is NOT required). Launch via
bin/stage.sh with <src_dir> = the LEMON MRI_Preprocessed_Derivetives folder and
<bids_out_dir> = the LEONARDO bids root (which holds sub-*/anat, derivatives/atlas,
derivatives/connectivity):

    ./bin/stage.sh lemon_fmri \\
        /srv/.../MRI_MPILMBB_LEMON/MRI_Preprocessed_Derivetives \\
        /srv/nfs-data/sisko/christian/LEONARDO/bids \\
        [sub-010002 ...] [--force] [--highpass 0] [--tr 1.4] [--min-voxels 5]

Omit the subject list to import every subject with a ``native`` BOLD in <src_dir>.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

SRC_ROOT = Path("/src")                            # LEMON MRI_Preprocessed_Derivetives
DST_ROOT = Path("/dst")                            # LEONARDO bids root
FMRI_DERIV = DST_ROOT / "derivatives" / "fMRI"     # this stage's output tree

ATLAS_DIR = DST_ROOT / "derivatives" / "atlas"        # full label atlases (atlas{N}.nii.gz)
CONN_DIR = DST_ROOT / "derivatives" / "connectivity"  # renumbered node atlases

RESOLUTIONS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
TASK = "rest"

# Registration QC. A correct within-subject same-MP2RAGE rigid fit scores ~0.99; ANTs' MI
# optimiser is multithread-nondeterministic and occasionally lands in a bad basin (NCC < 0),
# so we take the best of a few seeded attempts and refuse to emit time series below the gate
# (a clean bimodal split: good ~0.99 vs failed ~0). Below the gate we keep the BOLD but skip
# the parcellation, flagging the subject, rather than silently writing misaligned series.
REG_ATTEMPTS = 3
REG_NCC_GATE = 0.5
REG_NCC_GOOD = 0.9  # early-exit: a fit this good needs no further attempts

# Two atlas variants: (key, description). Their per-subject files/labels are resolved below;
# they differ in filename template, label-file format, and node numbering.
VARIANTS = ("full", "conn")


def discover_subjects() -> list[str]:
    """All sub-<ID> ids that have a preprocessed ``native`` BOLD in the source folder."""
    return sorted({p.parents[1].name for p in SRC_ROOT.glob("sub-*/func/*_native.nii.gz")})


def find_bold(sub: str) -> Path | None:
    """The subject's preprocessed native-space BOLD (task-rest); None if absent."""
    hits = sorted((SRC_ROOT / sub / "func").glob("*_task-rest_*_native.nii.gz"))
    return hits[0] if hits else None


def load_full_labels(path: Path) -> tuple[list[int], list[str]]:
    """Parse a full-atlas ``id,name`` label file. Returns (ids, names) with id 0 dropped."""
    ids, names = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sid, name = line.split(",", 1)  # names never contain a comma
        if int(sid) == 0:               # 0 = Unknown / background
            continue
        ids.append(int(sid))
        names.append(name)
    return ids, names


def load_conn_labels(path: Path) -> tuple[list[int], list[str]]:
    """Parse a connectivity ``labels_{N}.txt`` (one name per line, line index == node id).

    Returns (ids, names) for nodes 1..M (index 0 == Unknown is dropped), so row order
    matches the structural-connectome node order.
    """
    names_all = path.read_text().splitlines()  # index 0 = Unknown
    ids = list(range(1, len(names_all)))
    names = names_all[1:]
    return ids, names


def variant_paths(sub: str, variant: str, n: int) -> tuple[Path, Path]:
    """(atlas_volume, label_file) for one subject/variant/resolution."""
    if variant == "full":
        d = ATLAS_DIR / sub
        return d / f"atlas{n}.nii.gz", d / f"atlas{n}_labels.txt"
    d = CONN_DIR / sub
    return d / f"atlas{n}_connectivity.nii.gz", d / f"labels_{n}.txt"


def anat_pair(sub: str) -> tuple[Path, Path]:
    """(mp2rage_brain [BOLD frame, /src], raw T1w [atlas frame, /dst]) for one subject."""
    return (SRC_ROOT / sub / "anat" / f"{sub}_ses-01_acq-mp2rage_brain.nii.gz",
            DST_ROOT / sub / "anat" / f"{sub}_acq-mp2rage_T1w.nii.gz")


def _rigid_ncc(fixed, reg) -> float:
    """Within-brain correlation of a warped moving image against the fixed brain."""
    f = fixed.numpy()
    mask = f > 0
    if mask.sum() <= 1000:
        return float("nan")
    return float(np.corrcoef(reg["warpedmovout"].numpy()[mask], f[mask])[0, 1])


def compute_canonical_init(workdir: Path, max_scan: int = 12):
    """Find one subject whose from-scratch rigid fit is clearly good; return (mat_path, sub).

    The atlas->BOLD reorientation LEMON applied is ~identical across subjects (they share the
    same frame conventions -- only head position differs), so one good subject's transform is
    an excellent INITIALISATION that pulls ANTs out of the bad basin that occasionally traps
    the from-scratch MI search (verified: it rescues subjects that score ~0 from scratch to
    ~0.99). Scans the FULL discovered cohort (not the requested subset), so it still works on a
    repair rerun that only *requests* the hard subjects. Returns (None, None) if no reference
    passes within ``max_scan`` candidates (not expected -- most subjects register fine).
    """
    import ants
    import shutil

    for sub in discover_subjects()[:max_scan]:
        brain, t1w = anat_pair(sub)
        if not (brain.exists() and t1w.exists()):
            continue
        fixed = ants.image_read(str(brain))
        reg = ants.registration(fixed=fixed, moving=ants.image_read(str(t1w)),
                                type_of_transform="Rigid", random_seed=42)
        if _rigid_ncc(fixed, reg) >= REG_NCC_GOOD:
            mat = workdir / "canonical_init.mat"
            shutil.copy(reg["fwdtransforms"][0], mat)
            return mat, sub
    return None, None


def register_to_atlas_frame(mp2rage_brain: Path, raw_t1w: Path, canonical: Path | None = None):
    """Rigid-register the atlas-frame T1w into the BOLD-frame anatomical.

    fixed = LEMON mp2rage_brain (BOLD world), moving = Parrot raw T1w (atlas world). Returns
    (fwdtransforms, ncc, n_attempts): ``ncc`` is the within-brain correlation of the warped
    T1w against the brain (alignment QC; ~0.99 == good). Strategy: best of K from-scratch MI
    attempts (retries guard against ANTs' nondeterministic bad-basin failures), then -- only if
    those still miss the QC gate -- one more attempt seeded with the run's ``canonical`` init,
    which reliably rescues the ~few subjects the from-scratch search cannot align.
    """
    import ants

    fixed = ants.image_read(str(mp2rage_brain))
    moving = ants.image_read(str(raw_t1w))

    best_fwd, best_ncc, used = None, float("-inf"), 0
    for k in range(REG_ATTEMPTS):
        used = k + 1
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid",
                                random_seed=42 + k)
        ncc = _rigid_ncc(fixed, reg)
        if np.isfinite(ncc) and ncc > best_ncc:
            best_fwd, best_ncc = reg["fwdtransforms"], ncc
        if best_ncc >= REG_NCC_GOOD:
            break

    if best_ncc < REG_NCC_GATE and canonical is not None:
        used += 1
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid",
                                initial_transform=str(canonical), random_seed=42)
        ncc = _rigid_ncc(fixed, reg)
        if np.isfinite(ncc) and ncc > best_ncc:
            best_fwd, best_ncc = reg["fwdtransforms"], ncc

    return best_fwd, best_ncc, used


def save_native_to_t1w_transform(fwdtransforms, out_path: Path) -> None:
    """Save the rigid transform that resamples native-frame images into the Parrot T1w frame.

    The pipeline registers fixed=mp2rage_brain (native/BOLD frame), moving=raw T1w, so its
    ``fwdtransforms`` warp the T1w *into* native space. We save the **inverse**, so the written
    ``.mat`` maps native -> T1w and applies directly (no invert flag), e.g. to carry the BOLD
    (or any native-frame image) into the Parrot T1/atlas frame:

        antsApplyTransforms -d 3 -i <native_image> -r <T1w> -t <this.mat> -o <out_in_T1w>

    (Verified: the inverted transform applied plainly reaches within-brain NCC ~0.99.)
    """
    import ants

    inv = ants.invert_ants_transform(ants.read_transform(fwdtransforms[0]))
    ants.write_transform(inv, str(out_path))


def fwd_from_saved_transform(xfm_path: Path, workdir: Path):
    """Reconstruct the atlas->BOLD (T1w->native) warp from a saved native->T1w transform.

    ``save_native_to_t1w_transform`` stored the INVERSE of the registration's forward warp
    (native->T1w); inverting it back recovers the original ``fwd`` (T1w->native, exact for a
    rigid), so a rerun can resample atlases onto the BOLD grid *without* paying for the ANTs
    registration again. Returns a one-element transform list (a temp ``.mat`` in ``workdir``).
    """
    import ants

    fwd = ants.invert_ants_transform(ants.read_transform(str(xfm_path)))
    p = workdir / "reused_fwd.mat"
    ants.write_transform(fwd, str(p))
    return [str(p)]


def prior_registration_ncc(json_path: Path) -> float:
    """Within-brain NCC recorded in an existing sidecar (nan if absent/unreadable).

    Only used to carry a reused transform's original QC score into the refreshed sidecar.
    """
    try:
        meta = json.loads(json_path.read_text())
        return float(meta["registration"]["within_brain_ncc"])
    except (OSError, KeyError, ValueError, TypeError):
        return float("nan")


def atlas_on_bold_grid(atlas_path: Path, bold_ref, fwdtransforms, workdir: Path,
                       tag: str) -> np.ndarray:
    """Resample an atlas label volume onto the BOLD grid via the rigid transform.

    Returns an int label array in the BOLD voxel order (nibabel-loaded, so it aligns
    index-for-index with the nibabel-loaded BOLD). ``genericLabel`` interpolation keeps
    labels crisp; we round-trip through a temp NIfTI so ANTs/nibabel axis conventions match.
    """
    import ants
    import nibabel as nib

    moving = ants.image_read(str(atlas_path))
    warped = ants.apply_transforms(
        fixed=bold_ref, moving=moving, transformlist=fwdtransforms,
        interpolator="genericLabel",
    )
    tmp = workdir / f"atlas_on_bold_{tag}.nii.gz"
    ants.image_write(warped, str(tmp))
    arr = np.asarray(nib.load(str(tmp)).dataobj).astype(np.int32)
    tmp.unlink(missing_ok=True)
    return arr


def extract_region_timeseries(bold: np.ndarray, brain: np.ndarray, atlas: np.ndarray,
                              ids: list[int], min_voxels: int):
    """Region-mean time series over an atlas on the BOLD grid.

    ``bold`` (X,Y,Z,T) float32, ``brain`` (X,Y,Z) bool finite-mask, ``atlas`` (X,Y,Z) int.
    Returns (ts, nvox): ts (n_regions, T) float32 with NaN rows for regions that have
    fewer than ``min_voxels`` in-brain voxels (e.g. cerebellum/brainstem outside the EPI
    FOV); nvox (n_regions,) the valid-voxel count per region. Row order follows ``ids``.
    """
    T = bold.shape[-1]
    ts = np.full((len(ids), T), np.nan, dtype=np.float32)
    nvox = np.zeros(len(ids), dtype=np.int32)
    for r, rid in enumerate(ids):
        m = (atlas == rid) & brain
        n = int(m.sum())
        nvox[r] = n
        if n >= min_voxels:
            ts[r] = bold[m].mean(axis=0)  # (n_vox, T) -> (T,)
    return ts, nvox


def highpass_rows(ts: np.ndarray, tr: float, cutoff: float) -> np.ndarray:
    """Zero-phase Butterworth high-pass each finite region series along time (axis=1)."""
    from scipy.signal import butter, filtfilt

    if cutoff is None or cutoff <= 0:
        return ts
    b, a = butter(2, cutoff, btype="high", fs=1.0 / tr)
    out = ts.copy()
    good = np.isfinite(ts).all(axis=1)  # NaN rows (empty regions) left untouched
    if good.any():
        out[good] = filtfilt(b, a, ts[good], axis=1).astype(np.float32)
    return out


def build_optim_nodes(nvox_by_res: dict[int, np.ndarray], min_voxels: int) -> dict[str, np.ndarray]:
    """Optimization node mask for the connectome-node (conn) atlas, per resolution.

    The optimization node set is the connectome axis restricted to rows with a usable
    BOLD signal (``nvox_{N} >= min_voxels``) -- exactly the non-NaN rows of the desc-conn
    time series. Everything is expressed on the connectome ROW axis (0..M-1), which is the
    subject-independent comparison frame: ``weights_{N}`` row i == conn ``ts_{N}`` row i ==
    connectome node i. Subjects are optimized independently on their own kept rows, but
    ``optim_to_conn_{N}`` carries each optim node back to this shared axis so fitted
    parameters can be compared across subjects (scatter into a length-M vector).

    Returns npz arrays per resolution N:
      * ``keep_{N}``          (M,) bool  -- nvox_{N} >= min_voxels
      * ``optim_to_conn_{N}`` (K,) int32 -- optim node k -> connectome row (shared axis)
      * ``conn_to_optim_{N}`` (M,) int32 -- connectome row -> optim node, -1 if dropped
    """
    out: dict[str, np.ndarray] = {}
    for n, nvox in nvox_by_res.items():
        keep = nvox >= min_voxels
        optim_to_conn = np.flatnonzero(keep).astype(np.int32)
        conn_to_optim = np.full(keep.size, -1, dtype=np.int32)
        conn_to_optim[optim_to_conn] = np.arange(optim_to_conn.size, dtype=np.int32)
        out[f"keep_{n}"] = keep
        out[f"optim_to_conn_{n}"] = optim_to_conn
        out[f"conn_to_optim_{n}"] = conn_to_optim
    return out


def backfill_optim_nodes(sub: str, min_voxels: int, force: bool) -> None:
    """Write the desc-optim_nodes npz from an already-staged desc-conn npz.

    The mask derives entirely from the saved ``nvox_{N}`` counts, so this needs no BOLD,
    registration, or time-series re-extraction -- a near-instant pass for backfilling the
    already-staged cohort.
    """
    out_dir = FMRI_DERIV / sub
    conn_path = out_dir / f"{sub}_task-{TASK}_atlas-schaefer_desc-conn_timeseries.npz"
    optim_path = out_dir / f"{sub}_task-{TASK}_atlas-schaefer_desc-optim_nodes.npz"
    if not conn_path.exists():
        print(f"  {sub}: no desc-conn npz -> skipped")
        return
    if optim_path.exists() and not force:
        print(f"  {sub}: optim nodes exist -> skip (use --force)")
        return
    z = np.load(conn_path)
    nvox_by_res = {n: z[f"nvox_{n}"] for n in RESOLUTIONS if f"nvox_{n}" in z.files}
    if not nvox_by_res:
        print(f"  {sub}: desc-conn has no nvox arrays -> skipped")
        return
    arrays = build_optim_nodes(nvox_by_res, min_voxels)
    np.savez_compressed(optim_path, **arrays)
    n = max(nvox_by_res)  # report the finest resolution present
    print(f"  {sub}: wrote {optim_path.name} "
          f"(kept {int(arrays[f'keep_{n}'].sum())}/{nvox_by_res[n].size} @{n})")


def stage_subject(sub: str, tr: float, highpass: float, min_voxels: int,
                  variants: tuple[str, ...], force: bool, canonical: Path | None = None,
                  transforms_only: bool = False, reuse_transform: bool = False) -> None:
    """Copy the BOLD and write region time series for one subject."""
    import ants
    import nibabel as nib

    print(f"\n=== {sub} ===")
    bold_src = find_bold(sub)
    if bold_src is None:
        print("  BOLD ABSENT -> skipped")
        return
    confounds_src = bold_src.with_name(bold_src.name.replace("_native.nii.gz", "_confounds.txt"))

    out_dir = FMRI_DERIV / sub
    bold_dst = out_dir / f"{sub}_task-{TASK}_space-native_desc-preproc_bold.nii.gz"
    json_path = out_dir / f"{sub}_task-{TASK}_timeseries.json"
    xfm_path = out_dir / f"{sub}_from-native_to-T1w_xfm.mat"
    npz_paths = {v: out_dir / f"{sub}_task-{TASK}_atlas-schaefer_desc-{v}_timeseries.npz"
                 for v in variants}

    # Transforms-only backfill: just (re)compute + save the native->T1w rigid transform,
    # skipping the BOLD copy and the (expensive) time-series extraction.
    if transforms_only:
        mp2rage_brain, raw_t1w = anat_pair(sub)
        if not (mp2rage_brain.exists() and raw_t1w.exists()):
            print("  no recon inputs -> transform SKIPPED")
            return
        if xfm_path.exists() and not force:
            print("  transform exists -> skip (use --force to overwrite)")
            return
        fwd, ncc, n_reg = register_to_atlas_frame(mp2rage_brain, raw_t1w, canonical)
        if not np.isfinite(ncc) or ncc < REG_NCC_GATE:
            print(f"  registration FAILED (NCC {ncc:.3f} < {REG_NCC_GATE}) -> no transform")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        save_native_to_t1w_transform(fwd, xfm_path)
        print(f"  saved {xfm_path.name} (NCC {ncc:.3f}, {n_reg} attempt(s))")
        return

    if all(p.exists() for p in npz_paths.values()) and bold_dst.exists() and not force:
        print("  outputs exist -> skip (use --force to overwrite)")
        return

    # --- inputs for the time-series extraction (recon must exist) -------------------------
    # The fMRI folder + BOLD copy are created only after a successful registration (below),
    # so un-reconstructed or registration-failed subjects leave no fMRI output at all.
    mp2rage_brain, raw_t1w = anat_pair(sub)
    missing = [str(p) for p in (mp2rage_brain, raw_t1w, ATLAS_DIR / sub, CONN_DIR / sub)
               if not p.exists()]
    if missing:
        print(f"  WARNING no recon inputs ({missing[0]} ...) -> SKIPPED (no fMRI folder)")
        return

    # --- 2. recover the BOLD<->atlas frame transform (rigid) -----------------------------
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        # The transform is fixed subject geometry; only the time series change on a rerun. With
        # --reuse-transform we invert the saved native->T1w back to the atlas->BOLD warp and skip
        # the expensive ANTs registration entirely (falls back to registering if no xfm exists).
        if reuse_transform and xfm_path.exists():
            fwd = fwd_from_saved_transform(xfm_path, workdir)
            ncc, n_reg = prior_registration_ncc(json_path), 0
            print(f"  reusing saved transform {xfm_path.name} (prior NCC {ncc:.3f}); "
                  f"registration skipped")
        else:
            fwd, ncc, n_reg = register_to_atlas_frame(mp2rage_brain, raw_t1w, canonical)
            print(f"  rigid registration: within-brain NCC = {ncc:.3f} ({n_reg} attempt(s))")
            if not np.isfinite(ncc) or ncc < REG_NCC_GATE:
                print(f"  WARNING registration FAILED (NCC {ncc:.3f} < {REG_NCC_GATE}) -> "
                      f"SKIPPED (no fMRI folder). Inspect this subject.")
                return
            # Save the native->T1w rigid transform (spatial convenience; the time series use the
            # atlas resampled into the pristine native BOLD below, so the BOLD is never resampled).
            out_dir.mkdir(parents=True, exist_ok=True)
            save_native_to_t1w_transform(fwd, xfm_path)

        # Registration OK -> create the folder and stage the verbatim BOLD + confounds record.
        # Both are copied verbatim (never change), so skip the rewrite if already present -- this
        # is what keeps a --force rerun from recopying every subject's 3.5 GB BOLD needlessly.
        out_dir.mkdir(parents=True, exist_ok=True)
        if not bold_dst.exists():
            shutil.copyfile(bold_src, bold_dst)
            print(f"  BOLD copied -> {bold_dst.name}")
        else:
            print(f"  BOLD present -> kept {bold_dst.name}")
        conf_dst = out_dir / f"{sub}_task-{TASK}_confounds.txt"
        if confounds_src.exists() and not conf_dst.exists():
            shutil.copyfile(confounds_src, conf_dst)

        # BOLD as arrays (float32 to keep the 4D volume ~3.5 GB, not 7) + finite brain mask.
        bold_img = nib.load(str(bold_src))
        bold = np.asarray(bold_img.dataobj, dtype=np.float32)   # (X,Y,Z,T)
        brain = np.isfinite(bold[..., 0])                       # LEMON masks background to NaN
        T = bold.shape[-1]

        # ANTs reference carrying the BOLD grid geometry (first volume; intensities unused).
        ref_path = workdir / "bold_ref.nii.gz"
        nib.save(nib.Nifti1Image(bold[..., 0], bold_img.affine), ref_path)  # clean 3D header
        bold_ref = ants.image_read(str(ref_path))

        # --- 3-5. per variant / resolution: resample atlas, region-average, high-pass -----
        meta_variants = {}
        conn_nvox: dict[int, np.ndarray] = {}  # per-res nvox for the optim-node mask (conn only)
        for variant in variants:
            arrays: dict[str, np.ndarray] = {}
            per_res = {}
            for n in RESOLUTIONS:
                atlas_path, label_path = variant_paths(sub, variant, n)
                if not atlas_path.exists() or not label_path.exists():
                    print(f"  [{variant} {n}] atlas/labels missing -> skipped")
                    continue
                ids, names = (load_full_labels(label_path) if variant == "full"
                              else load_conn_labels(label_path))
                atlas = atlas_on_bold_grid(atlas_path, bold_ref, fwd, workdir, f"{variant}{n}")
                ts, nvox = extract_region_timeseries(bold, brain, atlas, ids, min_voxels)
                ts = highpass_rows(ts, tr, highpass)
                arrays[f"ts_{n}"] = ts
                arrays[f"ids_{n}"] = np.asarray(ids, dtype=np.int32)
                arrays[f"labels_{n}"] = np.asarray(names)  # unicode array (no pickle on load)
                arrays[f"nvox_{n}"] = nvox
                if variant == "conn":
                    conn_nvox[n] = nvox
                empty = int((nvox < min_voxels).sum())
                per_res[str(n)] = {"n_regions": len(ids), "n_empty": empty}
                print(f"  [{variant} {n}] {len(ids)} regions, {empty} empty -> ts {ts.shape}")
            if arrays:
                np.savez_compressed(npz_paths[variant], **arrays)
                meta_variants[variant] = per_res

    # --- optim node mask (conn atlas only): the non-NaN connectome rows ------------------
    optim_path = out_dir / f"{sub}_task-{TASK}_atlas-schaefer_desc-optim_nodes.npz"
    wrote_optim = False
    if "conn" in variants and conn_nvox:
        np.savez_compressed(optim_path, **build_optim_nodes(conn_nvox, min_voxels))
        wrote_optim = True
        print(f"  wrote {optim_path.name}")

    # --- 6. sidecar ----------------------------------------------------------------------
    meta = {
        "task": TASK,
        "tr_sec": tr,
        "n_timepoints": T,
        "space": ("LEMON fMRI native (structural-coregistered to LEMON's mp2rage_brain). "
                  "NOT Parrot's atlas T1 frame -- atlases were rigidly registered into this "
                  "frame before extraction (see registration.within_brain_ncc)."),
        "temporal_filtering": ("LEMON preprocessing (Mendes et al. 2019) already band-passed "
                               "0.01-0.1 Hz and mean-centered + variance-normalized. No filtering "
                               "applied here by default."),
        "extra_highpass_hz": highpass,  # 0 = off (default); >0 = ADDITIONAL high-pass over LEMON's
        "extra_highpass_filter": ("Butterworth order-2, zero-phase (scipy filtfilt)"
                                  if highpass and highpass > 0 else None),
        "denoising_note": ("LEMON-preprocessed (Mendes et al. 2019): motion/nuisance confounds "
                           "regressed (aCompCor + GLM), band-pass 0.01-0.1 Hz, mean-centered + "
                           "variance-normalized, first 5 dummy volumes dropped. No further confound "
                           "regression, filtering, or standardisation here (unless extra_highpass_hz>0)."),
        "min_voxels": min_voxels,
        "registration": {
            "method": "antspyx Rigid (MI), moving=raw T1w (atlas frame), fixed=mp2rage_brain",
            "within_brain_ncc": round(ncc, 4) if np.isfinite(ncc) else None,
            "attempts": n_reg,  # 0 == reused a previously saved transform (registration skipped)
            "reused_saved_transform": n_reg == 0,
        },
        "native_to_t1w_transform": {
            "file": xfm_path.name,
            "usage": ("Carries native-frame images into the Parrot T1/atlas frame. Apply "
                      "directly (no invert flag): antsApplyTransforms -d 3 -i <native_img> "
                      "-r <T1w> -t " + xfm_path.name + " -o <out_in_T1w>. Not needed for the "
                      "time series (the atlas is instead resampled into the pristine native BOLD)."),
        },
        "variants": {
            "full": "full Schaefer + subcortical/cerebellar/hippocampal label atlas (raw ids)",
            "conn": "renumbered connectome-node atlas; row i == structural-connectome node i",
        },
        "resolutions": meta_variants,
        "arrays_note": ("Per variant npz keyed by resolution: ts_{N} (n_regions, n_timepoints) "
                        "float32; ids_{N} region label ids; labels_{N} region names (row order); "
                        "nvox_{N} valid in-brain voxels per region (NaN row if < min_voxels). "
                        "Load: z = np.load(path)."),
        "optim_nodes": ({
            "file": optim_path.name,
            "min_voxels": min_voxels,
            "description": (
                "Connectome-node subset used for TVB optimization (conn atlas only): rows with a "
                "usable BOLD signal (nvox_{N} >= min_voxels), i.e. the non-NaN rows of desc-conn. "
                "Arrays per resolution N on the connectome ROW axis (0..M-1; weights_{N} row i == "
                "conn ts_{N} row i == connectome node i): keep_{N} (M,) bool; optim_to_conn_{N} "
                "(K,) optim node -> connectome row (subject-independent comparison axis); "
                "conn_to_optim_{N} (M,) connectome row -> optim node, -1 if dropped. "
                "Optimize per subject on kept rows: ts_{N}[keep], weights_{N}[ix_(keep,keep)]; "
                "apply any GLOBAL normalization AFTER masking. Dipole -> optim node: "
                "conn_to_optim_{N}[full_to_reduced_{N}[dipole_label] - 1] (the -1: weights/ts drop "
                "connectivity node 0 = Unknown). Compare fitted params across subjects by "
                "scattering into a length-M vector at optim_to_conn_{N}."),
        } if wrote_optim else None),
        "source_bold": bold_src.name,
    }
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {', '.join(p.name for p in npz_paths.values() if p.exists())} + sidecar")


def write_dataset_description() -> None:
    """Minimal BIDS-derivative dataset_description.json for derivatives/fMRI."""
    FMRI_DERIV.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": "LEMON preprocessed fMRI (BOLD + Schaefer region time series)",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "parrot-neuro staging: lemon_fmri.py"}],
        "SourceDatasets": [{"URL": "MPI-Leipzig Mind-Brain-Body LEMON (MRI_Preprocessed)"}],
    }
    (FMRI_DERIV / "dataset_description.json").write_text(json.dumps(desc, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import LEMON fMRI: copy BOLD + extract Schaefer region time series.")
    ap.add_argument("subjects", nargs="*", help="sub-<ID> ids (default: all in <src_dir>)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--tr", type=float, default=1.4,
                    help="repetition time in seconds (default: 1.4; BOLD header TR is unreliable)")
    ap.add_argument("--highpass", type=float, default=0.0,
                    help="OPTIONAL extra high-pass cutoff in Hz applied on top of LEMON's own "
                         "0.01-0.1 Hz band-pass (default: 0 = off; LEMON already high-passed at "
                         "0.01 Hz, so 0.01 here just double-filters). Set >0 only for a stricter band.")
    ap.add_argument("--min-voxels", type=int, default=5,
                    help="regions with fewer in-brain voxels get a NaN row (default: 5)")
    ap.add_argument("--variants", default="full,conn",
                    help="comma list of atlas variants to extract (default: full,conn)")
    ap.add_argument("--transforms-only", action="store_true",
                    help="only (re)compute + save the native->T1w rigid transform per subject "
                         "(skip BOLD copy and time-series extraction); for backfilling transforms")
    ap.add_argument("--reuse-transform", action="store_true",
                    help="on rerun, reuse each subject's saved native->T1w transform and SKIP the "
                         "(expensive) ANTs registration -- only re-extracts the time series (e.g. "
                         "to re-stage with a different --highpass). Use together with --force to "
                         "overwrite the existing npz; the verbatim BOLD copy is kept in place.")
    ap.add_argument("--optim-nodes-only", action="store_true",
                    help="only (re)write the desc-optim_nodes npz from each subject's existing "
                         "desc-conn nvox (no BOLD/registration/extraction); fast backfill")
    args = ap.parse_args()

    subjects = args.subjects or discover_subjects()

    # Fast path: derive the optim-node mask from already-staged nvox; no ANTs, no canonical init.
    if args.optim_nodes_only:
        print(f"Backfilling optim-node masks for {len(subjects)} subject(s) -> {FMRI_DERIV}")
        for sub in subjects:
            backfill_optim_nodes(sub, args.min_voxels, args.force)
        print("\nDone.")
        return

    variants = tuple(v for v in args.variants.split(",") if v in VARIANTS)
    print(f"Staging fMRI for {len(subjects)} subject(s) -> {FMRI_DERIV} "
          f"(variants: {', '.join(variants)}; TR={args.tr}s; high-pass={args.highpass}Hz)")

    # One canonical rigid init for the whole run (rescues the ~few subjects the from-scratch
    # registration cannot align). Lives in a run-level tempdir that outlives the per-subject ones.
    with tempfile.TemporaryDirectory() as run_tmp:
        # --reuse-transform does no registration at all, so skip the canonical-init scan
        # (which itself registers ~12 subjects just to seed the bad-basin fallback).
        if args.reuse_transform:
            canonical, ref = None, None
            print("canonical registration init: skipped (--reuse-transform)")
        else:
            canonical, ref = compute_canonical_init(Path(run_tmp))
            print(f"canonical registration init: from {ref}" if canonical
                  else "canonical registration init: NONE found (fallback disabled)")
        for sub in subjects:
            stage_subject(sub, args.tr, args.highpass, args.min_voxels, variants, args.force,
                          canonical, args.transforms_only, args.reuse_transform)

    write_dataset_description()
    print("\nDone.")


if __name__ == "__main__":
    main()
