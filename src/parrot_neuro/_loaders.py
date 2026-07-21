"""The ``s.load`` namespace: same vocabulary as ``s.path``, but returns loaded
objects instead of paths.

Each loader delegates to the matching :class:`~parrot_neuro._paths.SubjectPaths`
method for the path, then applies the reader the rest of the repo already uses for
that file type (``np.load`` / ``nibabel`` / ``trimesh`` / ``json`` / ``np.loadtxt``).
``nibabel`` and ``trimesh`` are imported lazily inside the readers so importing
``parrot_neuro`` stays cheap for path-only use.

Two behaviours are threaded through :meth:`_read`:
  * **QC-awareness** -- if the file's owning stage QC'd as ``warn``/``fail``, a
    :func:`warnings.warn` fires (unless the Subject was built with ``warn_on_qc=False``).
  * **opt-in caching** -- when the Subject was built with ``cache=True``, loaded
    objects are memoized per resolved path (so a big leadfield is read once).
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from . import _layout as L

if TYPE_CHECKING:
    from .subject import Subject


# --- readers (heavy deps imported lazily) -----------------------------------
def _read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _read_npz(path: Path):
    """Return the NpzFile handle; index it by key (no ``allow_pickle``)."""
    return np.load(path)


def _read_volume(path: Path):
    """Return the nibabel image (keeps the affine; call ``.get_fdata()`` yourself)."""
    import nibabel as nib

    return nib.load(str(path))


def _read_mesh(path: Path):
    import trimesh

    return trimesh.load_mesh(str(path), process=False)


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return _json.load(f)


def _read_table(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def _read_electrodes(path: Path) -> dict[str, np.ndarray]:
    """Parse the ``name, x, y, z`` 10-5 montage CSV into ``{name: array([x, y, z])}``."""
    out: dict[str, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            toks = [t.strip() for t in line.split(",")]
            if len(toks) < 4 or not toks[0]:
                continue
            out[toks[0]] = np.array([float(t) for t in toks[1:4]])
    return out


def _read_lines(path: Path) -> list[str]:
    """One entry per line, verbatim (e.g. connectivity ``labels_{N}.txt``)."""
    return path.read_text().splitlines()


def _read_label_table(path: Path) -> dict[int, str]:
    """Parse an ``id name ...`` LUT/labels table into ``{id: name}``. Skips comment
    and non-numeric-leading lines (mirrors the FreeSurfer-LUT parsing used in the
    QC package and the old Subject class)."""
    out: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            toks = line.replace(",", " ").split()
            if len(toks) < 2 or not toks[0].lstrip("-").isdigit():
                continue
            out[int(toks[0])] = toks[1]
    return out


class SubjectLoaders:
    def __init__(self, subject: "Subject"):
        self._s = subject

    # --- the one place QC-warning + caching live ----------------------------
    def _read(self, path: Path, reader: Callable[[Path], Any], stage: str | None = None) -> Any:
        if stage is not None:
            self._s._maybe_warn_qc(stage)
        cache = self._s._load_cache
        if cache is not None and path in cache:
            return cache[path]
        obj = reader(path)
        if cache is not None:
            cache[path] = obj
        return obj

    # --- generic escape hatches ---------------------------------------------
    def npy(self, stage: str, *parts: str) -> np.ndarray:
        return self._read(self._s.path.sfile(stage, *parts), _read_npy, stage)

    def volume(self, stage: str, *parts: str):
        return self._read(self._s.path.sfile(stage, *parts), _read_volume, stage)

    def mesh(self, stage: str, *parts: str):
        return self._read(self._s.path.sfile(stage, *parts), _read_mesh, stage)

    def json(self, stage: str, *parts: str) -> Any:
        return self._read(self._s.path.sfile(stage, *parts), _read_json, stage)

    def table(self, stage: str, *parts: str) -> np.ndarray:
        return self._read(self._s.path.sfile(stage, *parts), _read_table, stage)

    # --- anatomy / volumes --------------------------------------------------
    def t1(self):
        return self._read(self._s.path.t1(), _read_volume, L.RAW)

    def t2(self):
        return self._read(self._s.path.t2(), _read_volume, L.RAW)

    def t1_stripped(self):
        return self._read(self._s.path.t1_stripped(), _read_volume, L.SYNTHSTRIP)

    def t1_mask(self):
        return self._read(self._s.path.t1_mask(), _read_volume, L.SYNTHSTRIP)

    def final_tissues(self):
        return self._read(self._s.path.final_tissues(), _read_volume, L.SIMNIBS)

    def tissue_labels(self, kind: str = "electrical", source: str = "simnibs"):
        return self._read(self._s.path.tissue_labels(kind, source), _read_volume, L.TISSUE)

    # --- atlas --------------------------------------------------------------
    def atlas(self, res: int):
        return self._read(self._s.path.atlas(res), _read_volume, L.ATLAS)

    def atlas_lut(self, res: int) -> dict[int, str]:
        return self._read(self._s.path.atlas_lut(res), _read_label_table, L.ATLAS)

    def atlas_aggregated(self):
        return self._read(self._s.path.atlas_aggregated(), _read_volume, L.ATLAS)

    # --- surfaces -----------------------------------------------------------
    def surface(self, name: str):
        return self._read(self._s.path.surface(name), _read_mesh, L.SURFACES)

    def cortex(self, hemi: str, layer: str = "middle"):
        return self._read(self._s.path.cortex(hemi, layer), _read_mesh, L.SURFACES)

    def bem(self, layer: str):
        return self._read(self._s.path.bem(layer), _read_mesh, L.SURFACES)

    def scalp(self):
        return self._read(self._s.path.scalp(), _read_mesh, L.SURFACES)

    def vertex_attr(self, name: str) -> np.ndarray:
        return self._read(self._s.path.vertex_attr(name), _read_npy, L.SURFACES)

    # --- forward model ------------------------------------------------------
    def dipoles(self, spacing: float) -> "DipoleSet":
        """Bundle the core per-dipole arrays for a spacing into a small dataclass.
        Per-atlas region labels are separate -- see :meth:`dipole_labels`."""
        p = self._s.path
        r = lambda name: self._read(p.dipole_file(name, spacing), _read_npy, L.DIPOLES)
        return DipoleSet(
            spacing=spacing,
            positions=r("dipole_positions"),
            directions=r("dipole_directions"),
            volume=r("dipole_volume"),
            neural_density=r("dipole_neural_density"),
            orient_type=r("orient_type"),
        )

    def dipole_labels(self, res: int, spacing: float) -> np.ndarray:
        return self._read(
            self._s.path.dipole_file(f"{res}Parcels_dipole_labels", spacing), _read_npy, L.DIPOLES
        )

    def electrodes(self) -> dict[str, np.ndarray]:
        """The full 10-5 montage as ``{electrode_name: array([x, y, z])}``."""
        return self._read(self._s.path.electrodes_csv(), _read_electrodes, L.ELECTRODES)

    def electrodes_selected(self) -> Any:
        return self._read(self._s.path.electrodes_selected(), _read_json, L.ELECTRODES)

    def fiducials(self) -> Any:
        return self._read(self._s.path.fiducials(), _read_json, L.SCALP)

    # NOTE: no load.tetmesh() -- the CGAL tetrahedral mesh is a *volumetric* mesh
    # (.mesh/.vtu) that trimesh cannot read; use ``s.path.tetmesh(ext)`` and read it
    # with meshio (the repo's volume-mesh reader), which is not a core dependency.

    # --- leadfields ---------------------------------------------------------
    def leadfield(self, key: str) -> np.ndarray:
        return self._read(self._s.path.leadfield(key), _read_npy, L.LEADFIELDS)

    # --- DWI (optional) -----------------------------------------------------
    def dwi_tensor(self, space: str = "T1"):
        return self._read(self._s.path.dwi_tensor(space), _read_volume, L.DWITENSOR)

    def dwi_param(self, param: str, space: str = "T1"):
        return self._read(self._s.path.dwi_param(param, space), _read_volume, L.DWITENSOR)

    # --- connectivity -------------------------------------------------------
    # `fmri_aligned=True` restricts a structural array to the fMRI-usable node set
    # (the mask lives in the fMRI folder; see `fmri_nodes`). The mask is applied FIRST,
    # so any global normalization you do downstream is self-consistent -- never normalize
    # the full connectome and then slice. The fMRI side needs no such flag: the desc-conn
    # time series *is* the reference axis and its own NaN rows define the mask.
    def weights(self, n: int, normalized: bool = False, fmri_aligned: bool = False) -> np.ndarray:
        W = self._read(self._s.path.weights(n, normalized), _read_table, L.CONNECTIVITY)
        if fmri_aligned:
            keep = self.fmri_keep(n)
            W = W[np.ix_(keep, keep)]
        return W

    def distances(self, n: int, fmri_aligned: bool = False) -> np.ndarray:
        D = self._read(self._s.path.distances(n), _read_table, L.CONNECTIVITY)
        if fmri_aligned:
            keep = self.fmri_keep(n)
            D = D[np.ix_(keep, keep)]
        return D

    def connectivity_labels(self, n: int, fmri_aligned: bool = False) -> list[str]:
        """Connectome node names aligned with the ``weights_{n}`` / ``distances_{n}`` rows
        (length M; the leading Unknown node is dropped, matching the matrices). With
        ``fmri_aligned=True`` the names are restricted to the fMRI-usable nodes (length K)."""
        names = self._read(self._s.path.connectivity_labels(n), _read_lines, L.CONNECTIVITY)[1:]
        if fmri_aligned:
            return [nm for nm, k in zip(names, self.fmri_keep(n)) if k]
        return names

    # --- fMRI-derived node alignment (mask over the connectome axis) ---------
    # The mask encodes BOLD coverage but indexes the connectome ROW axis, so it bridges
    # the structural and functional sides. Consumers: the `fmri_aligned=` flags above and
    # the dipole->node converter below.
    def fmri_nodes(self, n: int, task: str = "rest") -> "FmriNodes":
        """The node-alignment maps for resolution ``n`` (from the desc-optim_nodes npz):
        ``keep`` (M,) bool, ``to_conn`` (K,) optim node -> connectome row (the shared
        cross-subject axis; scatter fitted params here), ``from_conn`` (M,) connectome row
        -> optim node (``-1`` if dropped)."""
        z = self._read(self._s.path.optim_nodes(task), _read_npz, None)
        return FmriNodes(
            keep=np.asarray(z[f"keep_{n}"]),
            to_conn=np.asarray(z[f"optim_to_conn_{n}"]),
            from_conn=np.asarray(z[f"conn_to_optim_{n}"]),
        )

    def fmri_keep(self, n: int, task: str = "rest") -> np.ndarray:
        """Boolean mask (M,) over connectome rows: True where the BOLD signal is usable
        (``nvox >= min_voxels``), i.e. the non-NaN rows of the desc-conn time series."""
        return self.fmri_nodes(n, task).keep

    def dipole_node_labels(self, res: int, spacing: float, task: str = "rest") -> np.ndarray:
        """Per-dipole *optimization-node* index (N,), ``-1`` where the dipole's connectome
        node has no usable BOLD signal (dropped from the aligned set). Composes the
        full->connectome (``full_to_reduced``) and connectome->optim (``from_conn``) maps, so
        index k refers to ``weights(res, fmri_aligned=True)`` row k -- ready to build the
        source->dipole projection in the aligned space. (This returns a *different id space*
        from :meth:`dipole_labels`, which is why it is a separate method, not a flag.)"""
        from_conn = self.fmri_nodes(res, task).from_conn
        f2r = self._read(self._s.path.full_to_reduced(res), _read_npy, L.CONNECTIVITY)
        labels = self.dipole_labels(res, spacing)
        conn_row = f2r[labels] - 1  # reduced label id -> connectome row (Unknown 0 -> -1)
        node = np.full(conn_row.shape, -1, dtype=np.int64)
        valid = conn_row >= 0  # dipoles in Unknown (none today) never index from_conn[-1]
        node[valid] = from_conn[conn_row[valid]]
        return node

    # --- anisotropy (optional) ----------------------------------------------
    def conductivity_tensors(self) -> np.ndarray:
        return self._read(self._s.path.conductivity_tensors(), _read_npy, L.ANISOTROPY)

    def wm_element_indices(self) -> np.ndarray:
        return self._read(self._s.path.wm_element_indices(), _read_npy, L.ANISOTROPY)

    # --- artifacts (optional) -----------------------------------------------
    def artifact_affine(self, direction: str = "mni_to_subject") -> np.ndarray:
        return self._read(self._s.path.artifact_affine(direction), _read_npy, L.ARTIFACTS)

    def artifact_sources(self) -> Any:
        return self._read(self._s.path.artifact_sources(), _read_json, L.ARTIFACTS)

    def artifact_dipole(self, name: str, kind: str = "eyes") -> np.ndarray:
        return self._read(self._s.path.artifact_dipole_file(name, kind), _read_npy, L.ARTIFACTS)

    # --- staged inputs (optional) -------------------------------------------
    def eeg(self, task: str = "eyesclosed"):
        """NpzFile of splice-free segments; keys ``seg_000``, ``seg_001``, ..."""
        return self._read(self._s.path.eeg(task), _read_npz, None)

    def fmri_timeseries(self, variant: str = "full", task: str = "rest"):
        """NpzFile of Schaefer time series; keys ``ts_<n>``, ``ids_<n>``, ..."""
        return self._read(self._s.path.fmri_timeseries(variant, task), _read_npz, None)


# --- small value type for the dipole bundle ---------------------------------
from dataclasses import dataclass  # noqa: E402  (kept near its only consumer)


@dataclass
class DipoleSet:
    """The core per-dipole arrays at one spacing (positions in mm)."""

    spacing: float
    positions: np.ndarray
    directions: np.ndarray
    volume: np.ndarray
    neural_density: np.ndarray
    orient_type: np.ndarray

    def __len__(self) -> int:
        return len(self.positions)


@dataclass
class FmriNodes:
    """fMRI-derived node alignment for one resolution (all on the connectome row axis).

    ``keep`` (M,) selects the fMRI-usable nodes; ``to_conn`` (K,) maps an aligned node back
    to its connectome row -- the subject-independent axis to scatter fitted params onto for
    cross-subject comparison; ``from_conn`` (M,) is the inverse, ``-1`` for dropped nodes.
    """

    keep: np.ndarray
    to_conn: np.ndarray
    from_conn: np.ndarray

    def __len__(self) -> int:
        return int(self.keep.sum())
