"""``Subject`` -- a lightweight facade over one subject's Parrot derivatives.

Initialize with the BIDS *dataset* root and a subject id. ``s.outputs`` inventories
registered outputs; ``s.path`` resolves original files and ``s.load`` reads them.
Existing datasets need manifest population separately before output access.
Missing optional manifests do not block other outputs.

    >>> s = Subject("/srv/.../parrot_LEMON", "010002")
    >>> s.path.leadfield("openmeeg-4.0mm")          # -> pathlib.Path
    >>> lf = s.load.leadfield("openmeeg-4.0mm")      # -> np.ndarray
    >>> s.available_leadfields(), s.atlas_resolutions(), s.has_dwi

Omit ``bids_root`` to use the ``PARROT_BIDS_ROOT`` environment variable:
``Subject(subject_id="010002")``. An explicit root always takes precedence.
The environment is read at construction time; no user configuration file is needed.

Loaded anatomy, metadata, dipoles and leadfields are cached lazily and mutable:
repeated loads share objects, so copy before modifying them. EEG/BOLD and large
distances are not retained by default. Use ``clear_cache()`` to release objects.
Volume ``use_cache=False`` bypasses both lookup and insertion. There is no
constructor-wide cache switch.

``has_*`` checks physical presence only (stage directories for DWI/artifacts/EEG/
fMRI, specific files for anisotropy/rest optimization nodes). It does not assert
registration, completeness, or QC. QC warn/fail stages are summarized once at
initialization; absent QC remains None.
"""
from __future__ import annotations

import csv
import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any

from . import _layout as L
from ._loaders import SubjectLoaders
from ._outputs import OutputInventory
from ._paths import SubjectPaths

_DEGRADED = {"warn", "fail"}


class Subject:
    def __init__(
        self,
        bids_root: str | Path | None = None,
        subject_id: str | None = None,
        *,
        warn_on_qc: bool = True,
        electrodes=None,
    ):
        if subject_id is None:
            raise TypeError("subject_id is required")
        if bids_root is None:
            bids_root = os.environ.get("PARROT_BIDS_ROOT")
            if not bids_root:
                raise ValueError("Pass bids_root or set the PARROT_BIDS_ROOT environment variable.")
        self.bids_root = Path(bids_root).expanduser()
        self.deriv = self.bids_root / "derivatives"
        if not self.deriv.is_dir():
            raise FileNotFoundError(
                f"No derivatives tree at {self.deriv} -- pass the BIDS *dataset* root "
                f"(the folder containing 'derivatives/'), not the derivatives dir itself."
            )
        # accept "010002", "sub-010002", or an int-like id
        self.subject = str(subject_id).removeprefix("sub-")
        self.subj = f"sub-{self.subject}"

        self.warn_on_qc = warn_on_qc
        self._load_cache: dict[Any, Any] = {}
        self._electrodes = None
        self.outputs = OutputInventory(self.deriv, self.subj)

        self.path = SubjectPaths(self)
        self.load = SubjectLoaders(self)
        if electrodes is not None:
            self.set_electrodes(electrodes)
        degraded = [f"{name}: {status}" for name, status in self._qc_status_map.items()
                    if status in _DEGRADED]
        if warn_on_qc and degraded:
            warnings.warn(f"{self.subj} QC — {', '.join(degraded)}. "
                          f"See {self.deriv / L.QC / self.subj / 'index.html'}",
                          stacklevel=2)

    def clear_cache(self):
        """Release loaded objects. Previously returned mutable objects remain valid."""
        self._load_cache.clear()

    @property
    def electrode_names(self) -> list[str]:
        """Model montage in leadfield row order; recorded EEG is independent."""
        return list(self._electrodes or self.load.electrodes_selected())

    def set_electrodes(self, names=None):
        """Select/reorder model electrodes; None restores the reconstruction montage.

        Load again after changing selection: previously returned arrays do not change.
        """
        if names is not None:
            if isinstance(names, str):
                raise ValueError("Electrodes must be a sequence of names")
            names = tuple(names)
            full = self.load.electrodes_selected()
            if not names or len(set(names)) != len(names):
                raise ValueError("Electrode selection must be nonempty and contain no duplicates")
            unknown = set(names) - set(full)
            if unknown:
                raise ValueError(f"Unknown model electrodes: {sorted(unknown)}")
        self._electrodes = names
        for key in list(self._load_cache):
            if isinstance(key, tuple) and key[0] in {"leadfield", "electrodes"}:
                del self._load_cache[key]

    def __repr__(self) -> str:
        return f"Subject({self.subject!r}, deriv={self.deriv}, backend={self.surface_backend})"

    # --- backend + metadata -------------------------------------------------
    @cached_property
    def surface_backend(self) -> str | None:
        """Present surface-reconstruction backend, or None when neither exists."""
        if (self.deriv / L.FREESURFER / self.subj / "surf" / "lh.white").exists():
            return L.FREESURFER
        if (self.deriv / L.FASTSURFER / self.subj / "surf" / "lh.white").exists():
            return L.FASTSURFER
        return None

    @cached_property
    def participants_row(self) -> dict[str, str] | None:
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
    def qc(self) -> dict | None:
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

    def qc_status(self, stage: str) -> str | None:
        """QC status (``pass``/``warn``/``fail``/``skip``/``None``) for a stage.
        Accepts either a QC stage name (e.g. ``"ingest"``) or a derivatives dir
        name (e.g. ``"raw"``)."""
        name = self._qc_status_map.get(stage)
        if name is not None or stage in self._qc_status_map:
            return name
        return self._qc_status_map.get(L.QC_STAGE.get(stage, ""))

    # --- optional-stage flags (probe dir existence) -------------------------
    @property
    def has_dwi(self) -> bool:
        return (self.deriv / L.QSIPREP / self.subj).is_dir() or (
            self.deriv / L.DWITENSOR / self.subj
        ).is_dir()

    @property
    def has_anisotropy(self) -> bool:
        return (self.deriv / L.ANISOTROPY / self.subj / "conductivity_tensors.npy").is_file()

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
        return (self.deriv / L.FMRI / self.subj /
                f"{self.subj}_task-rest_atlas-schaefer_desc-optim_nodes.npz").is_file()

    # --- discovery of variable outputs (glob) -------------------------------
    def available_leadfields(self) -> list[str]:
        """Discovered leadfield keys, e.g. ``['duneuroCGAL-2.0mm', 'openmeeg-4.0mm', ...]``.
        Pass one to ``s.path.leadfield`` / ``s.load.leadfield``."""
        return sorted({e["parameters"]["key"] for e in self.outputs.find("leadfield")})

    def atlas_resolutions(self) -> list[int]:
        """Cortical-subregion counts of the available atlases, e.g. ``[100, 200, ...]``."""
        out = []
        for entry in self.outputs.find(L.ATLAS):
            p = Path(entry["parameters"]["name"])
            if not p.name.startswith("atlas") or not p.name.endswith(".nii.gz"):
                continue
            stem = p.name[len("atlas") : -len(".nii.gz")]
            if stem.isdigit():
                out.append(int(stem))
        return sorted(out)

    def dipole_spacings(self) -> list[float]:
        """Available dipole spacings in mm, e.g. ``[2.0, 3.0, 4.0]``."""
        out = set()
        for entry in self.outputs.find(L.DIPOLES):
            p = Path(entry["parameters"]["name"]).parts[0]
            try:
                out.add(float(p[len("spacing") : -len("mm")]))
            except ValueError:
                continue
        return sorted(out)

    def dipole_blocks(self, spacing: float) -> list[str]:
        """Registered block identifiers; order is not aggregated dipole order."""
        prefix = f"spacing{float(spacing)}mm/"
        return sorted({e["parameters"]["name"][len(prefix):].removesuffix("/dipole_traceback.npy")
                       for e in self.outputs.find(L.DIPOLES)
                       if e["parameters"]["name"].startswith(prefix)
                       and e["parameters"]["name"].endswith("/dipole_traceback.npy")})
