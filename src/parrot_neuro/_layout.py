"""Stage names used by Subject's explicit accessors and physical presence probes.

The QC manifest writer owns producer filename discovery; accessors request its
registered logical names. Keep these names aligned with qc/manifests.py and
qc/context.py when changing producer layouts. The legend is documentation only.
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
EEG = "EEG"
FMRI = "fMRI"
QC = "qc"

# --- dir stage -> QC stage name ---------------------------------------------
# qc_report.json["stages"] is a list of {name, status, ...}. The QC module names
# mostly match the dir names, with a few exceptions (notably raw -> "ingest").
# ``Subject.qc_status`` accepts a dir stage constant and maps it to the QC entry.
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
