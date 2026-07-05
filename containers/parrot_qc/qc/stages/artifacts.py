"""EEG artifact-source QC: eyes + face/neck muscle sources and their geometry-only leadfields.

The `artifacts` stage adds extra-brain physiological noise sources -- eyes (sampled natively in the
subject's Eye_balls compartment) and muscle (HArtMuT template positions warped into the subject) --
and solves geometry-only artifact leadfields stackable with the brain leadfield. This validates:
  * the subject<->MNI affine (registration/) and its scalp-overlap self-check,
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
    adip = ctx.stage_dir("artifactdipoles")
    if not adip.exists():
        return r.skip("artifact stage not produced")

    # --- subject<->MNI registration --------------------------------------------------------------
    reg = ctx.stage_dir("registration")
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
    src_json = adip / "artifactsources.json"
    if src_json.exists():
        try:
            src = json.loads(src_json.read_text())
            n_eye = src.get("eyes", {}).get("n_dipoles")
            mus = src.get("muscle", {})
            n_muscle = mus.get("n_kept")
            neck_ok = bool(mus.get("neck_coverage", True))
            r.add(PASS, "artifactsources.json",
                  f"eyes={n_eye}, muscle kept={n_muscle}/{mus.get('n_total')} "
                  f"(dropped {mus.get('n_dropped')}), neck_coverage={neck_ok}")
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
        ctx.add_figure(r, "artifact_dipoles_3d",
                       "Artifact source positions (blue = eyes, yellow = muscle)",
                       lambda p: render3d.snapshot_points(pts, p, scalars=grp, ref_mesh=scalp,
                                                          title="artifact sources", point_size=5,
                                                          cmap="cividis"))

    # 2. EOG topography: signed potential from the corneo-retinal axis, summed over both eyes ->
    #    the recognisable dipolar frontal EOG pattern (diverging colour scale).
    if eye_L is not None:
        elec = _electrode_positions(ctx, eye_L.shape[0])
        axes = _load(adip / "eyes" / "dipole_preferential_direction.npy")
        if elec is not None and axes is not None and len(axes) == eye_L.shape[1] // 3:
            pot = _oriented_potential(eye_L, axes)
            pot = pot / (np.abs(pot).max() + 1e-30)       # normalise for a symmetric diverging map
            ctx.add_figure(r, "eog_topography",
                           "EOG topography (corneo-retinal projection, both eyes)",
                           lambda p: render3d.snapshot_points(elec, p, scalars=pot, ref_mesh=scalp,
                                                             title="EOG", point_size=12,
                                                             cmap="coolwarm"))
        elif elec is not None:  # no stored axes -> fall back to the group footprint
            fp = _total_footprint(eye_L)
            ctx.add_figure(r, "eog_topography", "Eye sensitivity footprint (all eye sources)",
                           lambda p: render3d.snapshot_points(elec, p, scalars=fp, ref_mesh=scalp,
                                                             title="EOG", point_size=12,
                                                             cmap="inferno"))

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
                                                             title="EMG", point_size=12,
                                                             cmap="inferno"))
    return r
