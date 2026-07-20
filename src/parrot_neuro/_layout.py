"""Single source of truth for the Parrot derivatives layout.

Every on-disk stage directory name and the mapping to QC stage names lives here,
and nowhere else in this package -- ``_paths.py`` composes filenames on top of
these constants, and ``_loaders.py`` composes readers on top of ``_paths.py``.

The names below are the *real* on-disk directory names, which are underscore-free
and differ in places from ``bin/legend_of_files.txt`` (e.g. ``simnibscharm``, not
``simnibs_charm``). They mirror the QC package's own layout knowledge in
``containers/parrot_qc/qc/context.py`` and ``qc/stages/__init__.py`` -- that
package is baked into the frozen ``parrot_qc`` Docker image and cannot be imported
here, so this is a deliberate (documented) duplication. **If the pipeline renames a
stage folder, update this module and keep it in sync with ``qc/stages/``.**
"""
from __future__ import annotations

# --- stage directory names (relative to <derivatives>) ----------------------
RAW = "raw"
FASTSURFER = "fastsurfer"
FREESURFER = "freesurfer"
HIPPUNFOLD = "hippunfold"
SIMNIBS = "simnibscharm"
FSLFIRST = "fslfirst"
SYNTHSTRIP = "synthstrip"
CEREBELLUM = "cerebellum"
BIGBRAIN = "bigbrain"
SURFACES = "surfaces"
ATLAS = "atlas"
TISSUE = "tissuelabels"
SCALP = "scalplandmarks"
ELECTRODES = "electrodes"
DIPOLES = "dipoles"
TETMESH = "tetmesh"
LEADFIELDS = "leadfields"
DWITENSOR = "dwitensor"
ANISOTROPY = "anisotropy"
CONNECTIVITY = "connectivity"
ARTIFACTS = "artifacts"
QSIPREP = "qsiprep"
QSIRECON = "qsirecon"
EEG = "EEG"
FMRI = "fMRI"
QC = "qc"

# --- dir stage -> QC stage name ---------------------------------------------
# qc_report.json["stages"] is a list of {name, status, ...}. The QC module names
# mostly match the dir names, with a few exceptions (notably raw -> "ingest").
# Loaders pass a *dir* stage constant; this maps it to the QC entry to check.
QC_STAGE = {
    RAW: "ingest",
    FASTSURFER: "fastsurfer",
    HIPPUNFOLD: "hippunfold",
    SIMNIBS: "simnibscharm",
    FSLFIRST: "fslfirst",
    SYNTHSTRIP: "synthstrip",
    CEREBELLUM: "cerebellum",
    BIGBRAIN: "bigbrain",
    SURFACES: "surfaces",
    ATLAS: "atlas",
    TISSUE: "tissuelabels",
    ELECTRODES: "electrodes",
    SCALP: "electrodes",  # scalp landmarks are validated under the electrodes QC stage
    DIPOLES: "dipoles",
    TETMESH: "tetmesh",
    DWITENSOR: "dwitensor",
    ANISOTROPY: "anisotropy",
    CONNECTIVITY: "connectivity",
    LEADFIELDS: "leadfields",
    ARTIFACTS: "artifacts",
}
