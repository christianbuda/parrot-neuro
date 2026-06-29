"""Tissue-labels QC: electrical (and acoustic) conductivity label fields + tables."""
from ..checks import StageResult, PASS, WARN, FAIL
from .. import render2d
from ._common import load_nifti, n_labels, first_existing

NAME = "tissuelabels"
TITLE = "Tissue labels — electrical / acoustic"


def _check_table(r, path, name):
    if not path.exists():
        r.warn(name, "missing")
        return
    try:
        rows = [ln for ln in path.read_text().splitlines() if ln.strip()]
        r.add(PASS if len(rows) > 1 else WARN, name, f"{len(rows)} rows")
    except Exception as e:  # noqa: BLE001
        r.fail(name, f"unreadable: {e}")


def run(ctx) -> StageResult:
    r = StageResult(NAME, TITLE)
    elec = ctx.stage_dir("tissuelabels") / "electrical"
    if not elec.exists():
        return r.skip("no tissuelabels/electrical tree")

    # prefer sim4life if present, else simnibs (mirrors the mesher's preference)
    vol = first_existing(elec / "sim4life.nii.gz", elec / "simnibs.nii.gz")
    if vol is None:
        r.fail("electrical label field", "no sim4life/simnibs volume")
    else:
        img = load_nifti(r, vol, f"electrical labels ({vol.stem})", ndim=3)
        if img is not None:
            r.add(PASS, "electrical tissue classes", f"{n_labels(img)} labels")
            if ctx.t1_path().exists():
                ctx.add_figure(r, "electrical_labels", "Electrical tissue labels on T1",
                               lambda p: render2d.roi_overlay(ctx.t1_path(), vol, p,
                                                              "electrical", cmap="tab20"))
    _check_table(r, elec / "simnibs_conductivities.txt", "conductivity table")
    _check_table(r, elec / "simnibs_labels.txt", "label table")

    aco = ctx.stage_dir("tissuelabels") / "acoustic"
    aco_vol = first_existing(aco / "sim4life.nii.gz", aco / "simnibs.nii.gz") if aco.exists() else None
    if aco_vol is not None:
        load_nifti(r, aco_vol, "acoustic label field", ndim=3)
        for prop in ("density", "speed_of_sound", "attenuation_constant", "nonlinearity_parameter"):
            _check_table(r, aco / f"simnibs_{prop}.txt", f"acoustic {prop}")
    else:
        r.notes.append("acoustic labels not produced (AEGEUS add-on)")
    return r
