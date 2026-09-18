"""Numbers that more than one image gates on. Change here, rebuild both images."""

# MNI->subject affine, template->subject scalp residual (mni_registration.py writes it,
# parrot_qc re-measures it independently). A correct affine lands the NYhead template scalp
# on the subject scalp to ~2-3 mm: 2.1-2.6 mm across the 10 AEGEUS subjects, 1.3 mm on the
# MNI09b template head. A mirrored (RAS/LPS) affine reads 17-22 mm.
REG_RESIDUAL_WARN_MM = 4.0
REG_RESIDUAL_FAIL_MM = 6.0

# The residual is measured only in the z-band where both surfaces exist. NYhead carries a long
# neck (down to z = -185 mm); a head built from an FOV-cropped T1 (the MNI152 templates stop at
# z ~= -73 mm) has no surface there, and those orphan vertices snap 70-110 mm away -- enough on
# their own to turn a 1.3 mm fit into an 8.8 mm "failure".
REG_FOV_MARGIN_MM = 5.0
# Below this evaluated fraction the restricted residual is not meaningful (a badly shifted affine
# could hide behind a handful of well-placed vertices), so the caller rejects instead of reporting.
REG_MIN_EVALUATED_FRAC = 0.5
