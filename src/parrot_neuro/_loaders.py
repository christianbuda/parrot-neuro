"""The ``s.load`` namespace: same vocabulary as ``s.path``, but returns loaded
objects instead of paths.

Each loader delegates to the matching :class:`~parrot_neuro._paths.SubjectPaths`
method for the path, then applies the reader the rest of the repo already uses for
that file type (``np.load`` / ``nibabel`` / ``trimesh`` / ``json`` / ``np.loadtxt``).
``nibabel`` and ``trimesh`` are imported lazily inside the readers so importing
``parrot_neuro`` stays cheap for path-only use.

Small metadata, anatomy, dipoles and transformed leadfields are cached lazily.
EEG/BOLD and large distances are not retained by default. Cached objects are
mutable and shared: copy before modifying. ``use_cache=False`` bypasses both
lookup and insertion; ``Subject.clear_cache()`` releases retained objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import json as _json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _layout as L

if TYPE_CHECKING:
    from .subject import Subject


# --- readers (heavy deps imported lazily) -----------------------------------
def _read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    """Load all arrays and close the archive before returning."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


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

    def _read(self, path: Path, reader: Callable[[Path], Any], stage: str | None = None,
              *, use_cache=True, key=None) -> Any:
        cache = self._s._load_cache
        key = (reader, path) if key is None else key
        if use_cache and key in cache:
            return cache[key]
        obj = reader(path)
        if use_cache:
            cache[key] = obj
        return obj

    # --- generic escape hatches ---------------------------------------------
    def npy(self, stage: str, *parts: str) -> np.ndarray:
        return self._read(self._s.path.sfile(stage, *parts), _read_npy, stage)

    def volume(self, stage: str, *parts: str, use_cache=True):
        return self._read(self._s.path.sfile(stage, *parts), _read_volume, stage, use_cache=use_cache)

    def mesh(self, stage: str, *parts: str):
        return self._read(self._s.path.sfile(stage, *parts), _read_mesh, stage)

    def json(self, stage: str, *parts: str) -> Any:
        return self._read(self._s.path.sfile(stage, *parts), _read_json, stage)

    def table(self, stage: str, *parts: str) -> np.ndarray:
        return self._read(self._s.path.sfile(stage, *parts), _read_table, stage)

    # --- anatomy / volumes --------------------------------------------------
    def t1(self, *, use_cache=True):
        return self._read(self._s.path.t1(), _read_volume, L.RAW, use_cache=use_cache)

    def t2(self, *, use_cache=True):
        return self._read(self._s.path.t2(), _read_volume, L.RAW, use_cache=use_cache)

    def t1_stripped(self, *, use_cache=True):
        return self._read(self._s.path.t1_stripped(), _read_volume, L.SYNTHSTRIP, use_cache=use_cache)

    def t1_mask(self, *, use_cache=True):
        return self._read(self._s.path.t1_mask(), _read_volume, L.SYNTHSTRIP, use_cache=use_cache)

    def final_tissues(self, *, use_cache=True):
        return self._read(self._s.path.final_tissues(), _read_volume, L.SIMNIBS, use_cache=use_cache)

    def tissue_labels(self, kind: str = "electrical", source: str = "simnibs", *, use_cache=True):
        return self._read(self._s.path.tissue_labels(kind, source), _read_volume, L.TISSUE, use_cache=use_cache)

    # --- atlas --------------------------------------------------------------
    def atlas(self, res: int, *, use_cache=True):
        return self._read(self._s.path.atlas(res), _read_volume, L.ATLAS, use_cache=use_cache)

    def atlas_lut(self, res: int) -> dict[int, str]:
        return self._read(self._s.path.atlas_lut(res), _read_label_table, L.ATLAS)

    def atlas_aggregated(self, *, use_cache=True):
        return self._read(self._s.path.atlas_aggregated(), _read_volume, L.ATLAS, use_cache=use_cache)

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
        key = ("dipoles", float(spacing))
        if key in self._s._load_cache:
            return self._s._load_cache[key]
        r = lambda name: self._read(p.dipole_file(name, spacing), _read_npy, L.DIPOLES)
        result = DipoleSet(
            spacing=spacing,
            positions=r("dipole_positions"),
            directions=r("dipole_directions"),
            volume=r("dipole_volume"),
            neural_density=r("dipole_neural_density"),
            orient_type=r("orient_type"),
        )
        self._s._load_cache[key] = result
        return result

    def dipole_block(self, spacing: float, block: str, name: str, *, use_cache=None):
        if use_cache is None:
            use_cache = name != "distance_matrix"
        return self._read(self._s.path.dipole_block(spacing, block, name), _read_npy,
                          use_cache=use_cache)

    def dipole_labels(self, res: int, spacing: float) -> np.ndarray:
        return self._read(
            self._s.path.dipole_file(f"{res}Parcels_dipole_labels", spacing), _read_npy, L.DIPOLES
        )

    def electrodes(self) -> dict[str, np.ndarray]:
        """Selected model positions in T1 world mm, in leadfield row order."""
        def selected(path):
            full = self.electrode_montage()
            return {name: full[name] for name in self._s.electrode_names}
        return self._read(self._s.path.electrodes_csv(), selected, key=("electrodes",))

    def electrode_montage(self) -> dict[str, np.ndarray]:
        """Full CSV geometry for matching recorded channels, independent of selection."""
        return self._read(self._s.path.electrodes_csv(), _read_electrodes)

    def electrodes_selected(self) -> Any:
        """Reconstruction-selected names; this is the allowed model montage."""
        def read(path):
            names = _read_json(path)
            if (not isinstance(names, list) or not names
                    or not all(isinstance(n, str) for n in names)
                    or len(names) != len(set(names))
                    or set(names) - set(self.electrode_montage())):
                raise ValueError(f"Invalid reconstruction-selected montage: {path}")
            return names
        return self._read(self._s.path.electrodes_selected(), read, key=("montage",))

    def fiducials(self) -> Any:
        return self._read(self._s.path.fiducials(), _read_json, L.SCALP)

    # NOTE: no load.tetmesh() -- the CGAL tetrahedral mesh is a *volumetric* mesh
    # (.mesh/.vtu) that trimesh cannot read; use ``s.path.tetmesh(ext)`` and read it
    # with meshio (the repo's volume-mesh reader), which is not a core dependency.

    # --- leadfields ---------------------------------------------------------
    def leadfield(self, key: str) -> np.ndarray:
        """(electrodes, source columns), average-referenced over selected rows.

        Brain columns are dipole-major xyz; artifact columns retain their stored
        source ordering/units. Paths always identify the original file.
        """
        def selected(path):
            original = _read_npy(path)
            full = list(self.electrode_montage())
            if original.ndim != 2 or original.shape[0] != len(full):
                raise ValueError(f"Leadfield rows disagree with full CSV montage: {path}")
            rows = [full.index(name) for name in self._s.electrode_names]
            result = original[rows].astype(np.result_type(original.dtype, np.float32), copy=True)
            result -= result.mean(axis=0, keepdims=True)
            return result
        return self._read(self._s.path.leadfield(key), selected, key=("leadfield", key))

    # --- DWI (optional) -----------------------------------------------------
    def dwi_tensor(self, space: str = "T1", *, use_cache=True):
        return self._read(self._s.path.dwi_tensor(space), _read_volume, L.DWITENSOR, use_cache=use_cache)

    def dwi_param(self, param: str, space: str = "T1", *, use_cache=True):
        return self._read(self._s.path.dwi_param(param, space), _read_volume, L.DWITENSOR, use_cache=use_cache)

    # --- connectivity -------------------------------------------------------
    # `fmri_aligned=True` restricts a structural array to the fMRI-usable node set
    # (the mask lives in the fMRI folder; see `fmri_nodes`). The mask is applied FIRST,
    # so any global normalization you do downstream is self-consistent -- never normalize
    # the full connectome and then slice. The fMRI side needs no such flag: the desc-conn
    # time series *is* the reference axis and its own NaN rows define the mask.
    def weights(self, n: int, normalized: bool = False, fmri_aligned: bool = False, task: str = "rest") -> np.ndarray:
        W = self._read(self._s.path.weights(n, normalized), _read_table, L.CONNECTIVITY)
        if fmri_aligned:
            keep = self.fmri_keep(n, task)
            if W.shape != (len(keep), len(keep)):
                raise ValueError("Structural weights and fMRI mask axes disagree")
            W = W[np.ix_(keep, keep)]
        return W

    def distances(self, n: int, fmri_aligned: bool = False, task: str = "rest") -> np.ndarray:
        D = self._read(self._s.path.distances(n), _read_table, L.CONNECTIVITY, use_cache=False)
        if fmri_aligned:
            keep = self.fmri_keep(n, task)
            if D.shape != (len(keep), len(keep)):
                raise ValueError("Structural distances and fMRI mask axes disagree")
            D = D[np.ix_(keep, keep)]
        return D

    def connectivity_labels(self, n: int, fmri_aligned: bool = False, task: str = "rest") -> list[str]:
        """Connectome node names aligned with the ``weights_{n}`` / ``distances_{n}`` rows
        (length M; the leading Unknown node is dropped, matching the matrices). With
        ``fmri_aligned=True`` the names are restricted to the fMRI-usable nodes (length K)."""
        names = self._read(self._s.path.connectivity_labels(n), _read_lines, L.CONNECTIVITY)[1:]
        if fmri_aligned:
            keep = self.fmri_keep(n, task)
            if len(names) != len(keep):
                raise ValueError("Connectivity labels and fMRI mask lengths disagree")
            return [nm for nm, k in zip(names, keep) if k]
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
        key = ("fmri_nodes", n, task)
        if key in self._s._load_cache:
            return self._s._load_cache[key]
        z = self._read(self._s.path.optim_nodes(task), _read_npz, None)
        result = FmriNodes(
            keep=np.asarray(z[f"keep_{n}"]),
            to_conn=np.asarray(z[f"optim_to_conn_{n}"]),
            from_conn=np.asarray(z[f"conn_to_optim_{n}"]),
        )
        if result.keep.ndim != 1 or result.keep.dtype != np.bool_:
            raise ValueError(f"Invalid fMRI keep mask for atlas {n}, task {task}")
        expected = np.full(result.keep.shape, -1, dtype=int)
        expected[result.keep] = np.arange(result.keep.sum())
        if (not np.array_equal(result.to_conn, np.flatnonzero(result.keep))
                or not np.array_equal(result.from_conn, expected)):
            raise ValueError(f"Inconsistent fMRI node maps for atlas {n}, task {task}")
        self._s._load_cache[key] = result
        return result

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
        key = ("dipole_node_labels", res, float(spacing), task)
        if key in self._s._load_cache:
            return self._s._load_cache[key]
        f2r = self._read(self._s.path.full_to_reduced(res), _read_npy, L.CONNECTIVITY)
        labels = self.dipole_labels(res, spacing)
        conn_row = f2r[labels] - 1  # reduced label id -> connectome row (Unknown 0 -> -1)
        node = np.full(conn_row.shape, -1, dtype=np.int64)
        valid = conn_row >= 0  # dipoles in Unknown (none today) never index from_conn[-1]
        node[valid] = from_conn[conn_row[valid]]
        self._s._load_cache[key] = node
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
    def eeg(self, task: str = "eyesclosed", *, use_cache=False) -> dict[str, np.ndarray]:
        """Eagerly loaded splice-free segments; keys ``seg_000``, ``seg_001``, ..."""
        return self._read(self._s.path.eeg(task), _read_npz, use_cache=use_cache)

    def eeg_metadata(self, task: str = "eyesclosed") -> dict:
        return self._read(self._s.path.eeg_metadata(task), _read_json)

    def fmri_timeseries(self, variant: str = "full", task: str = "rest", *, use_cache=False) -> dict[str, np.ndarray]:
        """Eagerly loaded Schaefer time series; keys ``ts_<n>``, ``ids_<n>``, ..."""
        return self._read(self._s.path.fmri_timeseries(variant, task), _read_npz, use_cache=use_cache)



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
