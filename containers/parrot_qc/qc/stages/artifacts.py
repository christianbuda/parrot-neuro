"""EEG artifact-source QC: eyes + face/neck muscle sources and their geometry-only leadfields.

The `artifacts` stage adds extra-brain physiological noise sources -- eyes (sampled natively in the
subject's Eye_balls compartment) and muscle (HArtMuT template positions warped into the subject) --
and solves geometry-only artifact leadfields stackable with the brain leadfield. This validates:
  * the subject<->MNI affine (artifacts/registration/) and its scalp-overlap self-check,
  * the artifact dipole sets + artifactsources.json (counts, neck coverage),
  * the eye leadfield and the muscle leadfield (subject-mesh solve OR the canned HArtMuT fallback),
and renders the source positions on the head plus sample EOG/EMG cap topographies.

Optional stage: when it wasn't run (no network egress for the template fetch, or a subject without
the inputs) this reports `skip`, never `fail` -- like the other optional stages.
"""
import json

import numpy as np

from ..checks import StageResult, PASS, WARN, FAIL, fmt_range
from .. import render3d
from .electrodes import _scalp_mesh, _read_csv_coords

NAME = "artifacts"
TITLE = "EEG artifact sources (eyes + muscle)"
DESCRIPTION = ("Extra-brain EEG artifact sources (eyes + face/neck muscle) and their geometry-only leadfields. Eyes should sit in the orbits and muscle around the face/neck; the EOG topography should be a frontal dipolar pattern and the EMG a peripheral ring.")

EYE_LF = "processed_duneuro_artifact-eyes-CGAL-leadfield.npy"
MUSCLE_LF = "processed_duneuro_artifact-muscle-CGAL-leadfield.npy"
MUSCLE_FALLBACK_LF = "processed_hartmut_muscle-leadfield.npy"


def _load(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:  # noqa: BLE001
        return None


def _check_leadfield(r, name, path, n_src_expected):
    """Existence + finiteness + (n_elec, 3*n_src) shape check; returns L or None."""
    if not path.exists():
        return None
    L = _load(path)
    if L is None or L.ndim != 2:
        r.fail(name, f"unreadable or not 2D ({None if L is None else L.shape})")
        return None
    finite = bool(np.isfinite(L).all())
    nonzero = bool(np.any(L != 0))
    detail = f"shape={L.shape}, {fmt_range(L)}"
    ok = finite and nonzero
    if n_src_expected is not None:
        shape_ok = L.shape[1] == 3 * n_src_expected
        ok = ok and shape_ok
        detail += f", expect {3 * n_src_expected} cols"
    if not finite:
        detail += ", NON-FINITE"
    if not nonzero:
        detail += ", ALL-ZERO"
    r.add(PASS if ok else FAIL, name, detail)
    return L if (finite and nonzero) else None


def _oriented_potential(L, orientations):
    """Signed per-electrode potential for unit dipole moments along `orientations`, summed over
    sources -> (n_elec,). For eyes this projects each source onto its corneo-retinal axis, giving
    the recognisable *dipolar* EOG topography rather than a magnitude blob."""
    n_src = L.shape[1] // 3
    o = np.asarray(orientations, dtype=float)[:n_src]
    Lr = L.reshape(L.shape[0], n_src, 3)               # (n_elec, n_src, 3)
    return np.einsum("esc,sc->e", Lr, o)               # sum_s L_s . orient_s


def _total_footprint(L):
    """Per-electrode total sensitivity across ALL sources/orientations (row L2 norm) -> (n_elec,).
    Shows a source group's whole spatial footprint (eyes -> frontal; muscle -> bilateral
    temporal/facial/neck ring) instead of one focal source."""
    return np.linalg.norm(L, axis=1)


def _source_footprint(L):
    """Per-SOURCE total coupling to the whole cap (norm over electrodes+orientations) -> (n_src,).
    The per-electrode counterpart is _total_footprint; this one exposes individual bad sources."""
    n_src = L.shape[1] // 3
    return np.linalg.norm(L.reshape(L.shape[0], n_src, 3), axis=(0, 2))


def _check_source_outliers(r, name, L):
    """Flag near-singular source columns -> (footprints, max/p99 ratio).

    A point dipole's potential grows as 1/r^2, so a source that ends up a millimetre from an
    electrode captures that channel almost entirely and swamps every other source once the noise
    generator multiplies through. place_artifact_dipoles.py drops such sources by distance; this
    checks the *result*. Healthy cohort range is 1.3-2.5x."""
    fp = _source_footprint(L)
    p99 = np.percentile(fp, 99)
    ratio = float(fp.max() / p99) if p99 > 0 else float("inf")
    n_hot = int((fp > 5 * p99).sum())
    st = PASS if ratio < 5 else (WARN if ratio < 20 else FAIL)
    r.add(st, f"{name} source outliers",
          f"max/p99 = {ratio:.1f}x (source {int(fp.argmax())}), {n_hot} source(s) above 5x p99; "
          f"median {np.median(fp):.3g}, p99 {p99:.3g}, max {fp.max():.3g}")
    return fp, ratio


SNAP_TOLERANCE_MM = 1.5   # see _check_electrode_clearance
CLEARANCE_FAIL_MM = 3.0   # below this the 1/r^2 growth is steep enough to distort the column


def _check_electrode_clearance(r, name, positions, elec, gap):
    """Distance from each source to the nearest electrode -- the upstream cause of hot columns.

    Measured on the geometry the solver used, which is NOT quite the geometry placement produced:
    make_leadfield_artifacts snaps every source to the nearest valid-tissue tet centroid, and that
    can shave a fraction of a millimetre off the clearance place_artifact_dipoles guaranteed.
    Allowing SNAP_TOLERANCE_MM keeps this check on real regressions instead of warning on every
    subject; what actually matters (the resulting footprint) is checked by _check_source_outliers.
    """
    if positions is None or elec is None or gap is None:
        return
    from scipy.spatial import cKDTree
    d, _ = cKDTree(elec).query(positions, k=1)
    st = (PASS if d.min() >= gap - SNAP_TOLERANCE_MM
          else WARN if d.min() >= CLEARANCE_FAIL_MM else FAIL)
    r.add(st, f"{name} electrode clearance",
          f"closest source {d.min():.2f} mm from an electrode (target {gap:g} mm at placement, "
          f"{SNAP_TOLERANCE_MM:g} mm snap tolerance); {int((d < gap).sum())} below target, "
          f"{int((d < CLEARANCE_FAIL_MM).sum())} under {CLEARANCE_FAIL_MM:g} mm; "
          f"median {np.median(d):.1f} mm")


def _electrode_positions(ctx, n_rows):
    """Montage positions in leadfield-row order (landmarks_10-5-full.csv), or None on mismatch."""
    csv = ctx.stage_dir("electrodes") / "landmarks_10-5-full.csv"
    if not csv.exists():
        return None
    coords = _read_csv_coords(csv)  # insertion (file) order == leadfield row order
    pts = np.array(list(coords.values()), dtype=float)
    return pts if len(pts) == n_rows else None


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    adip = ctx.stage_dir("artifacts/dipoles")
    if not adip.exists():
        return r.skip("artifact stage not produced")

    # --- subject<->MNI registration --------------------------------------------------------------
    reg = ctx.stage_dir("artifacts/registration")
    if (reg / "mni_to_subject_affine.npy").exists():
        qc_json = reg / "registration_qc.json"
        if qc_json.exists():
            try:
                info = json.loads(qc_json.read_text())
                overlap = float(info.get("mean_scalp_overlap_mm", float("nan")))
                # ~15-25 mm is the expected template-vs-subject scalp shape difference (the ray-cast
                # corrects the radial part); only a grossly large value indicates a bad registration.
                st = PASS if overlap <= 30 else WARN
                r.add(st, "MNI registration", f"scalp overlap {overlap:.1f} mm "
                      f"(via {info.get('selected_transformlist', '?')})")
            except Exception as e:  # noqa: BLE001
                r.warn("MNI registration", f"qc json unreadable: {e}")
        else:
            r.add(PASS, "MNI registration", "affine present (no qc json)")
    else:
        r.fail("MNI registration", "mni_to_subject_affine.npy missing")

    # --- artifactsources.json: counts + neck coverage --------------------------------------------
    n_eye = n_muscle = None
    neck_ok = True
    # Placement-time clearance target; absent from outputs predating the fix, in which
    # case the pipeline default is assumed and the check below reports the shortfall.
    gap_mm = 5.0
    src_json = adip / "artifactsources.json"
    if src_json.exists():
        try:
            src = json.loads(src_json.read_text())
            n_eye = src.get("eyes", {}).get("n_dipoles")
            mus = src.get("muscle", {})
            n_muscle = mus.get("n_kept")
            neck_ok = bool(mus.get("neck_coverage", True))
            near_el = mus.get("n_dropped_near_electrode")
            gap_mm = float(mus.get("min_electrode_distance_mm", gap_mm))
            drops = f"dropped {mus.get('n_dropped')}"
            if near_el is not None:
                drops += (f": {mus.get('n_dropped_raycast')} ray-miss, {near_el} within "
                          f"{mus.get('min_electrode_distance_mm')} mm of an electrode")
            r.add(PASS, "artifactsources.json",
                  f"eyes={n_eye}, muscle kept={n_muscle}/{mus.get('n_total')} "
                  f"({drops}), neck_coverage={neck_ok}")
            if not neck_ok:
                r.warn("muscle neck coverage",
                       "too few muscle sources survived the warp -> canned-leadfield fallback used")
        except Exception as e:  # noqa: BLE001
            r.warn("artifactsources.json", f"unreadable: {e}")
    else:
        r.warn("artifactsources.json", "missing")

    # --- dipole sets -----------------------------------------------------------------------------
    eye_pos = _load(adip / "eyes" / "dipole_positions.npy")
    mus_pos = _load(adip / "muscle" / "dipole_positions.npy")
    if eye_pos is not None:
        n_eye = len(eye_pos)
        r.add(PASS, "eye dipoles", f"{n_eye} sources")
    else:
        r.fail("eye dipoles", "dipole_positions.npy missing")
    if mus_pos is not None:
        n_muscle = len(mus_pos)
        r.add(PASS, "muscle dipoles", f"{n_muscle} sources")
    else:
        r.warn("muscle dipoles", "dipole_positions.npy missing")

    # --- leadfields ------------------------------------------------------------------------------
    lf_dir = ctx.stage_dir("leadfields")
    eye_L = _check_leadfield(r, "eye leadfield", lf_dir / EYE_LF, n_eye)

    muscle_L = None
    if (lf_dir / MUSCLE_LF).exists():
        muscle_L = _check_leadfield(r, "muscle leadfield (solved)", lf_dir / MUSCLE_LF, n_muscle)
    elif (lf_dir / MUSCLE_FALLBACK_LF).exists():
        muscle_L = _check_leadfield(r, "muscle leadfield (HArtMuT fallback)",
                                    lf_dir / MUSCLE_FALLBACK_LF, None)
    else:
        r.fail("muscle leadfield", "neither solved nor fallback leadfield present")

    # --- per-source sanity: clearance (cause) + outlier footprints (effect) -----------------------
    # The solve uses the SNAPPED positions when they were recorded; fall back to the placed ones.
    mus_solved = _load(adip / "muscle" / "dipole_positions_solved.npy")
    mus_used = mus_solved if mus_solved is not None else mus_pos
    # Geometry-only use, so read the montage directly rather than via _electrode_positions (which
    # gates on matching the leadfield row count).
    elec_csv = ctx.stage_dir("electrodes") / "landmarks_10-5-full.csv"
    elec_all = (np.array(list(_read_csv_coords(elec_csv).values()), dtype=float)
                if elec_csv.exists() else None)
    _check_electrode_clearance(r, "muscle", mus_used, elec_all, gap_mm)

    mus_fp = None
    if muscle_L is not None:
        mus_fp, _ = _check_source_outliers(r, "muscle leadfield", muscle_L)
    if eye_L is not None:
        _check_source_outliers(r, "eye leadfield", eye_L)

    # --- figures ---------------------------------------------------------------------------------
    scalp = _scalp_mesh(ctx)

    # 1. Artifact source positions on the head, coloured by group (0 = eye, 1 = muscle).
    clouds = []
    if eye_pos is not None:
        clouds.append((eye_pos, np.zeros(len(eye_pos))))
    if mus_pos is not None:
        clouds.append((mus_pos, np.ones(len(mus_pos))))
    if clouds:
        pts = np.vstack([c[0] for c in clouds])
        grp = np.concatenate([c[1] for c in clouds])
        # These sources sit inside/behind the scalp (eyes in the orbits, muscle on the
        # face/neck), so this is the one overlay that keeps a translucent scalp -- an
        # opaque one would hide the sources entirely.
        ctx.add_figure(r, "artifact_dipoles_3d",
                       "Artifact source positions (blue = eyes, yellow = muscle)",
                       lambda p: render3d.snapshot_points(pts, p, scalars=grp, ref_mesh=scalp,
                                                          ref_opacity=0.2,
                                                          views=("anterior", "left", "superior"),
                                                          title="artifact sources", point_size=5,
                                                          cmap="cividis"))

    # 2. EOG topography: signed potential from the corneo-retinal axis, summed over both eyes ->
    #    the recognisable dipolar frontal EOG pattern (diverging colour scale).
    # NB: this map is PER-PEAK NORMALISED, the EMG map below is log-absolute -- the two are NOT
    # amplitude-comparable. These are leadfield GEOMETRY (source->cap coupling), not artifact size:
    # the realised artifact is leadfield x source moment, and the moments come from the (future)
    # amplitude generator. Per source, eyes and muscles couple to the cap comparably; EOG only looks
    # "small" here because it is focal (2 coherent ocular clusters) and normalised. Don't read
    # relative artifact magnitude off these figures.
    if eye_L is not None:
        elec = _electrode_positions(ctx, eye_L.shape[0])
        axes = _load(adip / "eyes" / "dipole_preferential_direction.npy")
        if elec is not None and axes is not None and len(axes) == eye_L.shape[1] // 3:
            pot = _oriented_potential(eye_L, axes)
            pot = pot / (np.abs(pot).max() + 1e-30)       # normalise for a symmetric diverging map
            ctx.add_figure(r, "eog_topography",
                           "EOG topography (corneo-retinal projection, both eyes)",
                           lambda p: render3d.snapshot_points(elec, p, scalars=pot, ref_mesh=scalp,
                                                             views=("anterior", "left", "superior"),
                                                             title="EOG", point_size=12,
                                                             cmap="coolwarm", clim=(-1, 1),
                                                             scalar_bar=True,
                                                             scalar_bar_title="signed potential (norm.)"))
        elif elec is not None:  # no stored axes -> fall back to the group footprint
            fp = _total_footprint(eye_L)
            ctx.add_figure(r, "eog_topography", "Eye sensitivity footprint (all eye sources)",
                           lambda p: render3d.snapshot_points(elec, p, scalars=fp, ref_mesh=scalp,
                                                             views=("anterior", "left", "superior"),
                                                             title="EOG", point_size=12,
                                                             cmap="inferno", scalar_bar=True,
                                                             scalar_bar_title="sensitivity (a.u.)"))

    # 3. EMG footprint: per-electrode total sensitivity over ALL muscle sources -> the bilateral
    #    temporal/facial/neck ring. Log scale, because the superficial neck muscles have far larger
    #    gains than the facial ones and on a linear scale would saturate the map to a focal blob.
    if muscle_L is not None:
        elec = _electrode_positions(ctx, muscle_L.shape[0])
        if elec is not None:
            fp = _total_footprint(muscle_L)
            fp = np.log10(fp + fp[fp > 0].min() * 1e-3) if np.any(fp > 0) else fp
            ctx.add_figure(r, "emg_footprint",
                           "Muscle sensitivity footprint (all muscle sources, log scale)",
                           lambda p: render3d.snapshot_points(elec, p, scalars=fp, ref_mesh=scalp,
                                                             views=("anterior", "left", "superior"),
                                                             title="EMG", point_size=12,
                                                             cmap="inferno", scalar_bar=True,
                                                             scalar_bar_title="log10 sensitivity"))

    # 4. Per-SOURCE strength: each muscle source coloured by how hard it drives the whole cap,
    #    as log10 of its footprint over the median. Decades, not a ratio to the p99: the healthy
    #    spread is itself ~2 decades (deep facial vs superficial neck), so a p99-normalised scale
    #    renders everything black -- the exact failure that made emg_footprint unreadable. The top
    #    of the scale follows the data, so a source that slipped under an electrode both stands out
    #    as an isolated bright dot AND visibly compresses everything else.
    if mus_fp is not None and mus_used is not None and len(mus_used) == len(mus_fp):
        p50 = np.median(mus_fp)
        z = np.log10(mus_fp / p50, out=np.full(len(mus_fp), -1.0), where=mus_fp > 0) if p50 > 0 else mus_fp
        ctx.add_figure(r, "muscle_source_strength",
                       f"Per-source cap coupling, log10(footprint / median); peak "
                       f"{10 ** float(z.max()):.0f}x the median. An isolated bright dot is a "
                       "source too close to an electrode.",
                       lambda p: render3d.snapshot_points(mus_used, p, scalars=z, ref_mesh=scalp,
                                                          ref_opacity=0.2,
                                                          views=("anterior", "left", "superior"),
                                                          title="source strength", point_size=6,
                                                          cmap="inferno",
                                                          clim=(-1, max(2.0, float(z.max()))),
                                                          scalar_bar=True,
                                                          scalar_bar_title="log10(fp / median)"))
    return r
