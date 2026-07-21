"""``Subject`` -- a lightweight facade over one subject's Parrot derivatives.

Initialize with the BIDS *dataset* root and a subject id; the class resolves the
derivatives tree, exposes subject metadata, discovers the variable outputs, and
hands back paths (``s.path.*``) or loaded objects (``s.load.*``) for essentially
every reconstructed file.

    >>> s = Subject("/srv/.../parrot_LEMON", "010002")
    >>> s.path.leadfield("openmeeg-4.0mm")          # -> pathlib.Path
    >>> lf = s.load.leadfield("openmeeg-4.0mm")      # -> np.ndarray
    >>> s.available_leadfields(), s.atlas_resolutions(), s.has_dwi

The path/load layout knowledge lives in :mod:`._layout` (+ :mod:`._paths`); see the
header of ``_layout.py`` for the relationship to the QC package's ``context.py``.
"""
from __future__ import annotations

import csv
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

from . import _layout as L
from ._loaders import SubjectLoaders
from ._paths import SubjectPaths

_DEGRADED = {"warn", "fail"}


class Subject:
    def __init__(
        self,
        bids_root: str | Path,
        subject_id: str,
        *,
        cache: bool = False,
        warn_on_qc: bool = True,
    ):
        self.bids_root = Path(bids_root)
        self.deriv = self.bids_root / "derivatives"
        if not self.deriv.is_dir():
            raise FileNotFoundError(
                f"No derivatives tree at {self.deriv} -- pass the BIDS *dataset* root "
                f"(the folder containing 'derivatives/'), not the derivatives dir itself."
            )
        # accept "010002", "sub-010002", or an int-like id
        self.subject = str(subject_id)
        if self.subject.startswith("sub-"):
            self.subject = self.subject[len("sub-") :]
        self.subj = f"sub-{self.subject}"

        self.warn_on_qc = warn_on_qc
        self._load_cache: Optional[dict[Path, Any]] = {} if cache else None
        self._warned_stages: set[str] = set()

        self.path = SubjectPaths(self)
        self.load = SubjectLoaders(self)

    def __repr__(self) -> str:
        return f"Subject({self.subject!r}, deriv={self.deriv}, backend={self.surface_backend})"

    # --- backend + metadata -------------------------------------------------
    @cached_property
    def surface_backend(self) -> str:
        """``'freesurfer'`` if a FreeSurfer surface recon is present, else
        ``'fastsurfer'`` (mirrors the probe in run_reconstruction.sh). Note this is
        the *recon* backend; the ``surfaces/`` ``.ply`` files are always
        ``freesurfer_``-prefixed regardless."""
        if (self.deriv / L.FREESURFER / self.subj / "surf" / "lh.white").exists():
            return L.FREESURFER
        return L.FASTSURFER

    @cached_property
    def participants_row(self) -> Optional[dict[str, str]]:
        """This subject's row from ``<bids_root>/participants.tsv`` as a dict
        (per-subject override columns included), or ``None`` if unavailable."""
        tsv = self.bids_root / "participants.tsv"
        if not tsv.exists():
            return None
        with open(tsv, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("participant_id") == self.subj:
                    return row
        return None

    # --- QC awareness -------------------------------------------------------
    @cached_property
    def qc(self) -> Optional[dict]:
        """The parsed ``qc/<subj>/qc_report.json`` (or ``None`` if absent)."""
        report = self.deriv / L.QC / self.subj / "qc_report.json"
        if not report.exists():
            return None
        import json

        with open(report) as f:
            return json.load(f)

    @cached_property
    def _qc_status_map(self) -> dict[str, str]:
        rep = self.qc
        if not rep:
            return {}
        return {st["name"]: st.get("status") for st in rep.get("stages", []) if "name" in st}

    def qc_status(self, stage: str) -> Optional[str]:
        """QC status (``pass``/``warn``/``fail``/``skip``/``None``) for a stage.
        Accepts either a QC stage name (e.g. ``"ingest"``) or a derivatives dir
        name (e.g. ``"raw"``)."""
        name = self._qc_status_map.get(stage)
        if name is not None or stage in self._qc_status_map:
            return name
        return self._qc_status_map.get(L.QC_STAGE.get(stage, ""))

    def _maybe_warn_qc(self, dir_stage: str) -> None:
        """Called by loaders before reading; warn once per stage if it QC'd badly."""
        if not self.warn_on_qc or dir_stage in self._warned_stages:
            return
        qc_name = L.QC_STAGE.get(dir_stage, dir_stage)
        status = self._qc_status_map.get(qc_name)
        if status in _DEGRADED:
            self._warned_stages.add(dir_stage)
            warnings.warn(
                f"{self.subj}: stage '{qc_name}' QC status is '{status}'; "
                f"the output you are loading may be unreliable. "
                f"See {self.deriv / L.QC / self.subj / 'index.html'}. "
                f"(build the Subject with warn_on_qc=False to silence.)",
                stacklevel=3,
            )

    # --- optional-stage flags (probe dir existence) -------------------------
    @property
    def has_dwi(self) -> bool:
        return (self.deriv / L.QSIPREP / self.subj).is_dir() or (
            self.deriv / L.DWITENSOR / self.subj
        ).is_dir()

    @property
    def has_anisotropy(self) -> bool:
        return self.path.conductivity_tensors().exists()

    @property
    def has_artifacts(self) -> bool:
        return self.path.artifact_dipoles_dir().is_dir()

    @property
    def has_eeg(self) -> bool:
        return (self.deriv / L.EEG / self.subj).is_dir()

    @property
    def has_fmri(self) -> bool:
        return (self.deriv / L.FMRI / self.subj).is_dir()

    @property
    def has_optim_nodes(self) -> bool:
        """Whether the fMRI-derived optimization node mask (desc-optim_nodes) exists."""
        return self.path.optim_nodes().exists()

    # --- discovery of variable outputs (glob) -------------------------------
    def available_leadfields(self) -> list[str]:
        """Discovered leadfield keys, e.g. ``['duneuroCGAL-2.0mm', 'openmeeg-4.0mm', ...]``.
        Pass one to ``s.path.leadfield`` / ``s.load.leadfield``."""
        d = self.path.stage_dir(L.LEADFIELDS)
        keys = [
            p.name[len("processed_") : -len("-leadfield.npy")]
            for p in d.glob("processed_*-leadfield.npy")
        ]
        return sorted(keys)

    def atlas_resolutions(self) -> list[int]:
        """Cortical-subregion counts of the available atlases, e.g. ``[100, 200, ...]``."""
        d = self.path.stage_dir(L.ATLAS)
        out = []
        for p in d.glob("atlas*.nii.gz"):
            stem = p.name[len("atlas") : -len(".nii.gz")]
            if stem.isdigit():
                out.append(int(stem))
        return sorted(out)

    def dipole_spacings(self) -> list[float]:
        """Available dipole spacings in mm, e.g. ``[2.0, 3.0, 4.0]``."""
        d = self.path.stage_dir(L.DIPOLES)
        out = []
        for p in d.glob("spacing*mm"):
            try:
                out.append(float(p.name[len("spacing") : -len("mm")]))
            except ValueError:
                continue
        return sorted(out)
