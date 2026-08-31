"""Build a miniature but *structurally exact* Parrot derivatives tree.

Why this exists
---------------
A real Parrot subject is ~800 MB of leadfields and tens of thousands of dipoles:
untestable in CI and far too slow to iterate against. Everything the
``parrot_neuro`` API does, though, is shape- and layout-bound, not size-bound --
so a 58-dipole subject with the *real* filenames, the *real* npz keys, the *real*
dtypes and the *real* cross-file invariants exercises ~90% of the code paths in
milliseconds.

Ground truth for the layout is the measured structure of
``/srv/.../parrot_LEMON/derivatives`` (see the module-level tables below); every
filename here was taken from a real tree, not guessed.

The whole tree is **67 kB in 68 files** and builds in ~30 ms, so it is cheap
enough to rebuild per test (which is what ``tests/conftest.py`` does -- later
phases write a projector cache and run folders *into* this tree, and a shared
copy would leak between tests).

Dimensions (all overridable as keyword arguments)
------------------------------------------------
==========================  =======  =================================================
quantity                    value    note
==========================  =======  =================================================
electrodes                        8  real 10-5 names, ``name, x, y, z`` CSV, no header
dipoles                          58  15 volumetric + 25 + 18 on two surface blocks
dipole spacing                2.0mm  a single spacing -> ``dipoles/<subj>/spacing2.0mm/``
atlas resolution                100  files are ``100Parcels_dipole_labels.npy`` etc.
connectome nodes (M)              6  ``weights_100.txt`` is 6x6; ``labels_100.txt`` 7 lines
optimization nodes (K)            4  2 connectome nodes dropped for lack of BOLD
brain leadfields                  2  ``(8, 174) = (n_elec, 3 * n_dip)``
eye / muscle artifact src      5 / 7  ``(8, 15)`` and ``(8, 21)``
==========================  =======  =================================================

The leadfield keys carry the spacing (``duneuroCGAL-2.0mm``), so overriding
``spacing=`` moves the dipole folder *and* renames the leadfields consistently:
``Subject.dipole_spacings()`` and ``Subject.available_leadfields()`` agree.

What is and is not checkable by hand
------------------------------------
The **integer bookkeeping** is: block offsets and sizes, the traceback masks,
``full_to_reduced`` / ``reduced_to_full``, ``keep`` / ``conn_to_optim`` /
``optim_to_conn``, and the round-robin dipole -> node assignment. All of those
follow from the dimension table with no randomness.

Everything *numeric* -- positions, directions, volumes, neural density,
leadfields, connectivity weights, time series -- is an ``rng`` draw. Nothing
about a projected matrix ``M`` can be derived on paper; assert it against a
naive oracle recomputed from the same files instead.

The deliberately non-alphabetical block order
---------------------------------------------
In a real tree ``dipole_traceback.npy`` is a **boolean mask over the aggregated
dipole axis**, and the blocks' contiguous runs are *not* in directory order.
Measured on ``sub-010002``::

    volumetric, freesurfer_lh_middle, freesurfer_rh_middle, cereb_inner_processed,
    hippunfold_L_dentate, hippunfold_R_dentate, hippunfold_L_hipp, hippunfold_R_hipp

Two things are wrong about a directory listing: ``volumetric`` is **first**, not
last, and the surface blocks are **not sorted among themselves** either
(``cereb_inner_processed`` sorts before ``freesurfer_*`` but comes after it).

The fixture reproduces *both* halves of the trap with three blocks in this
aggregated order::

    volumetric                      offsets   0 - 14
    surfaces/freesurfer_lh_middle   offsets  15 - 39
    surfaces/cereb_inner_processed  offsets  40 - 57

so that all three plausible wrong orderings are detectably wrong:

* ``sorted(surfaces) + [volumetric]`` -- the reference bug on ``eeg-bold-fit``
  (``optimization/forward.py::get_electric_signals``);
* ``list(surfaces.iterdir()) + [volumetric]`` -- the same bug, filesystem-order
  flavour;
* ``[volumetric] + sorted(surfaces)`` -- the *naive fix* of the reference bug,
  which gets ``volumetric`` right and the two surface blocks swapped.

The only correct source of block order is ``dipole_traceback.npy``
(sort the blocks by ``flatnonzero(mask)[0]``).

Geometry scale: density, not size
---------------------------------
Source smoothing uses ``sigma = 1.5 * spacing`` against each block's
``distance_matrix.npy``, so what decides whether the smoothing is a real
operator or a rounding error is the ratio ``inter-dipole distance / sigma`` --
**not** the absolute size of the head.

Block radii are therefore *derived from* ``spacing`` (see ``_ball_radius`` /
``_shell_radius``): a random sample of ``n`` points in the block gets a median
nearest-neighbour distance of about ``spacing``, which puts the effective
neighbour count (``exp`` of the row entropy of the normalized kernel) at
**8.6 / 8.7 / 4.7** for volumetric / lh_middle / cereb respectively, against
**7.2 - 18.9** measured on real blocks (7.2 ``hippunfold_L_dentate``, 12.3
``hippunfold_R_hipp``, 18.9 ``freesurfer_lh_middle``, all at spacing 4.0 mm).

The consequence is that the fixture's *brain* is a ~2 cm object while its
electrodes stay on a realistic 85 mm sphere. That is deliberate: the leadfield
is a random matrix anyway, and matching the dipole density is what gives the
projector tests teeth. An earlier version used head-scale dipole positions, and
its smoothing matrix was numerically indistinguishable from the identity (1.13
effective neighbours) -- which made *every* smoothing bug cost ~1% of ``M``.
Measured cost of each deliberate projector bug, ``||dM||/||M||``, before -> after
this rescaling::

    sorted(surfaces)+[volumetric]  12.5% -> 34.4%   S transposed           2.5% ->  7.6%
    [volumetric]+sorted(surfaces)      - -> 21.9%   column- not row-norm   1.5% -> 10.4%
    R dipoles not re-drawn         38.8% -> 44.4%   volume weight omitted  1.3% ->  7.6%
    smoothing dropped (S = I)       5.8% -> 57.7%   sigma = spacing        3.1% -> 18.8%

so a projector-equivalence test at any sane tolerance now catches all of them.
Blocks are offset from each other so their positions do not interpenetrate;
that offset is cosmetic (each block's distance matrix is internal).

Invariants guaranteed
---------------------
* ``agg[traceback_mask] == block_array`` for ``dipole_positions``,
  ``dipole_directions`` (per-block ``dipole_normals`` / ``dipole_preferential_direction``),
  ``dipole_volume`` and ``orient_type``; the masks partition the aggregated axis exactly.
* Each block's ``distance_matrix.npy`` is the *true* pairwise Euclidean distance
  matrix of that block's positions (symmetric, zero diagonal, triangle
  inequality), so Gaussian source-smoothing computed from it is meaningful maths.
* ``orient_type`` contains free (``'R'``) dipoles plus ``'P'``/``'G'`` in the
  volumetric block and pure normals (``'N'``) on **every** surface block --
  the real mix, and the reason orientation-seed tests have anything to re-draw.
* ``full_to_reduced_100[label] - 1`` sends every dipole's parcel label into
  ``[0, M)``; ``conn_to_optim_100`` sends the M connectome rows onto ``[0, K)``
  with ``-1`` on the dropped rows. Hence
  ``Subject.load.dipole_node_labels(100, 2.0)`` takes values in ``{-1} u [0, K)``
  and **does** contain ``-1`` (the valid-mask logic is exercised, not skipped).
* Every connectome node owns at least one dipole -- including the two dropped
  from the optimization set, which is what produces those ``-1``s.
* ``weights_100.txt`` / ``weights_invnodevol_100.txt`` / ``distances_100.txt`` are
  symmetric with a zero diagonal.
* The ``desc-conn_timeseries`` ``ts_100`` rows are NaN on exactly the connectome
  rows that ``keep_100`` drops -- the invariant asserted in
  ``tests/test_subject.py::test_fmri_aligned_loaders``.
* Leadfields are average-referenced (columns sum to ~0 over electrodes), as the
  pipeline's ``processed_*`` leadfields are, and are scaled to the real
  ``mean|entry| ~ 2.5e-7`` so an absolute-amplitude assertion is not nonsense.
* Byte-level determinism: same ``seed`` -> byte-identical files. ``np.savez``
  pins its zip timestamps to 1980-01-01 and nibabel gzips with ``mtime=0``, so
  this holds for the ``.npz`` and ``.nii.gz`` outputs too.

Seed dependence of a projected ``M`` (read before writing a per-column test)
---------------------------------------------------------------------------
Re-drawing the ``orient_type == 'R'`` directions changes only the columns of
``M`` whose nodes own volumetric dipoles. Here the volumetric block owns the
first ``M // 2`` connectome nodes, so with the default dimensions **only optim
columns 0 and 1 move across orientation seeds**; columns 2-3 come from the two
surface blocks, are all ``'N'``, and are bit-identical across seeds. This is
faithful -- real surface blocks are all ``'N'`` too.

A "no fixed seed" test must therefore assert on the whole ``M`` (or on columns
0-1), never per-column over all K columns.

A second atlas resolution -- only in the fMRI npz files
------------------------------------------------------
``desc-optim_nodes.npz`` and ``desc-conn_timeseries.npz`` carry keys for **two**
resolutions (``100`` and ``200``) with *different* node counts (M=6/K=4 and
M=9/K=6), so a consumer that hardcodes or mis-selects the resolution suffix gets
a length mismatch instead of silently working. Only resolution 100 has a full
atlas / connectome / dipole-label set: ``Subject.atlas_resolutions()`` returns
``[100]``. Adding a second complete resolution is future work -- say so rather
than assuming it is there.

Deliberately *not* written (nothing in the API reads them, and a plausible-looking
fake would be worse than an honest absence): the DWI stages, anisotropy, tetmesh,
the ``surfaces/`` and BEM ``.ply`` meshes, ``raw/T2``, the ``eyesopen`` EEG task,
``atlas<n>_connectivity.nii.gz``, ``assignments_<n>.txt`` and the binary
``artifacts/registration/.../ants_affine.mat``. Their absence is itself realistic --
``has_dwi`` and ``has_anisotropy`` report ``False`` here, which is what an
optional-stage test wants. Add one here when a test needs it; never fake it at the
call site.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- dimensions --------------------------------------------------------------
SUBJECT_ID = "000001"

#: Montage order of the 8 fake electrodes (real 10-5 names).
ELECTRODE_NAMES: tuple[str, ...] = ("Fp1", "Fp2", "T7", "Cz", "T8", "Pz", "O1", "O2")
#: Approximate 10-5 directions on the unit sphere (RAS: +x right, +y anterior, +z superior).
_ELECTRODE_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "Fp1": (-0.309, 0.891, 0.334),
    "Fp2": (0.309, 0.891, 0.334),
    "T7": (-1.0, 0.0, 0.0),
    "Cz": (0.0, 0.0, 1.0),
    "T8": (1.0, 0.0, 0.0),
    "Pz": (0.0, -0.719, 0.695),
    "O1": (-0.309, -0.891, 0.334),
    "O2": (0.309, -0.891, 0.334),
}
HEAD_RADIUS_MM = 85.0

#: The recorded EEG covers only a *subset* of the montage, in its own order --
#: the 59-of-345 LEMON situation in miniature, so name-matching code is tested.
EEG_CHANNEL_NAMES: tuple[str, ...] = ("Fp2", "Cz", "O1", "Fp1", "O2")
EEG_SFREQ = 250.0
EEG_SEGMENT_LENGTHS: tuple[int, ...] = (64, 48)
EEG_TASK = "eyesclosed"

SPACING = 2.0
N_VOLUMETRIC = 15
#: Surface block names in **aggregated** order -- deliberately not sorted, and
#: deliberately not the order a directory listing gives. See the module docstring.
SURFACE_BLOCKS: tuple[str, ...] = ("freesurfer_lh_middle", "cereb_inner_processed")
N_SURFACE = 25  # surfaces/freesurfer_lh_middle
N_CEREBELLUM = 18  # surfaces/cereb_inner_processed
#: Full aggregated block order, relative to the spacing directory.
BLOCK_ORDER: tuple[str, ...] = ("volumetric", *(f"surfaces/{b}" for b in SURFACE_BLOCKS))

#: Block centres (mm). Cosmetic only -- smoothing is per block, so the offsets
#: never enter any distance matrix. They exist so the blocks do not interpenetrate.
_BLOCK_CENTRES: dict[str, tuple[float, float, float]] = {
    "volumetric": (0.0, 6.0, 0.0),
    "surfaces/freesurfer_lh_middle": (-26.0, 4.0, 10.0),
    "surfaces/cereb_inner_processed": (0.0, -30.0, -20.0),
}

ATLAS_RES = 100
#: A second resolution, present ONLY in the fMRI npz files (see the docstring).
SECOND_ATLAS_RES = 200
N_CONN_NODES = 6  # M
N_OPTIM_NODES = 4  # K
#: How much the second resolution's node counts differ from the first.
_SECOND_RES_EXTRA_CONN = 3
_SECOND_RES_EXTRA_OPTIM = 2

#: Full-atlas ids of the M connectome nodes (sparse, like the real 8237-long map).
FULL_ATLAS_IDS: tuple[int, ...] = (2, 4, 6, 8, 10, 12)

#: Leadfield keys carry the dipole spacing, so a ``spacing=`` override renames
#: them too. ``LEADFIELD_KEYS`` is the tuple at the default spacing.
LEADFIELD_KEY_TEMPLATES: tuple[str, ...] = (
    "duneuroCGAL-{spacing}mm",
    "duneuroCGAL_anisotropic-{spacing}mm",
)
ARTIFACT_LEADFIELD_KEYS: tuple[str, ...] = (
    "duneuro_artifact-eyes-CGAL",
    "duneuro_artifact-muscle-CGAL",
)


def leadfield_keys(spacing: float = SPACING) -> tuple[str, ...]:
    """Brain-leadfield keys at ``spacing`` -- the single source of truth for both
    the constants and the filenames the writer emits."""
    return tuple(t.format(spacing=spacing) for t in LEADFIELD_KEY_TEMPLATES)


LEADFIELD_KEYS: tuple[str, ...] = leadfield_keys(SPACING)

#: Real ``processed_*`` leadfield entries average ``|.| ~ 2.5e-7``; a zero-mean
#: normal with this sigma reproduces that (``E|X| = sigma * sqrt(2/pi)``).
LEADFIELD_SIGMA = 3.1e-7
#: Artifact sources sit right under the electrodes, so their leadfields are much
#: larger than the brain's.
ARTIFACT_LEADFIELD_SIGMA = 3.1e-6

N_EYE_SOURCES = 5
N_MUSCLE_SOURCES = 7

N_TRS = 20
FMRI_TASK = "rest"

#: QC stage names, in the order the real ``qc_report.json`` lists them.
QC_STAGES: tuple[str, ...] = (
    "ingest", "fastsurfer", "hippunfold", "simnibscharm", "fslfirst", "synthstrip",
    "cerebellum", "bigbrain", "surfaces", "atlas", "tissuelabels", "electrodes",
    "dipoles", "tetmesh", "connectivity", "leadfields", "artifacts",
)

_SUBCORTICAL_NAMES = ("Left-Caudate", "Left-Putamen", "Left-Pallidum",
                      "Left-Thalamus", "Left-Amygdala")
_CORTICAL_NAME = "17Networks_LH_VisCent_ExStr_{i}"


# --- the resolved fixture ----------------------------------------------------
@dataclass(frozen=True)
class _Blocks:
    """Per-block bookkeeping. ``names`` is in *aggregated* order (volumetric first)."""

    names: tuple[str, ...]  # relative to the spacing dir, e.g. "surfaces/freesurfer_lh_middle"
    sizes: tuple[int, ...]

    @property
    def offsets(self) -> tuple[int, ...]:
        out, acc = [], 0
        for n in self.sizes:
            out.append(acc)
            acc += n
        return tuple(out)

    def mask(self, i: int, n_total: int) -> np.ndarray:
        m = np.zeros(n_total, dtype=bool)
        m[self.offsets[i] : self.offsets[i] + self.sizes[i]] = True
        return m


def build_fake_subject(
    root: str | Path,
    subject_id: str = SUBJECT_ID,
    *,
    seed: int = 0,
    n_volumetric: int = N_VOLUMETRIC,
    n_surface: int = N_SURFACE,
    n_cerebellum: int = N_CEREBELLUM,
    n_conn_nodes: int = N_CONN_NODES,
    n_optim_nodes: int = N_OPTIM_NODES,
    atlas_res: int = ATLAS_RES,
    spacing: float = SPACING,
    electrode_names: tuple[str, ...] = ELECTRODE_NAMES,
    eeg_channel_names: tuple[str, ...] = EEG_CHANNEL_NAMES,
    n_eye_sources: int = N_EYE_SOURCES,
    n_muscle_sources: int = N_MUSCLE_SOURCES,
    n_trs: int = N_TRS,
) -> Path:
    """Write the fake derivatives tree under ``root`` and return the **BIDS root**.

    The returned path is the folder that *contains* ``derivatives/``, so it can be
    handed straight to :class:`parrot_neuro.Subject`::

        bids = build_fake_subject(tmp_path)
        s = Subject(bids, "000001")

    ``seed`` fixes every random array; two builds with the same arguments produce
    byte-identical files. See the module docstring for the dimension table, the
    deliberate block ordering and the guaranteed invariants.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    subject = str(subject_id).removeprefix("sub-")
    subj = f"sub-{subject}"
    deriv = root / "derivatives"

    block_sizes = (n_volumetric, n_surface, n_cerebellum)
    blocks = _Blocks(BLOCK_ORDER, block_sizes)
    n_surface_total = n_surface + n_cerebellum
    n_dip = sum(block_sizes)
    n_elec = len(electrode_names)

    if min(block_sizes) < 1:
        raise ValueError("every block must contain at least one dipole")
    if not 1 <= n_optim_nodes <= n_conn_nodes:
        # n_optim_nodes == 0 would make dipole_node_labels all -1, i.e. an
        # all-zero projector -- a degenerate tree no test wants by accident.
        raise ValueError("n_optim_nodes must be between 1 and n_conn_nodes")
    missing = set(eeg_channel_names) - set(electrode_names)
    if missing:
        raise ValueError(f"eeg_channel_names must be a subset of the montage; extra: {missing}")

    # --- node bookkeeping ----------------------------------------------------
    # Split the connectome nodes between the blocks: the volumetric block owns
    # the "subcortical" nodes, the two surface blocks share the "cortical" ones.
    n_vol_nodes = n_conn_nodes // 2
    vol_nodes = list(range(n_vol_nodes))
    surf_nodes = list(range(n_vol_nodes, n_conn_nodes))
    if not vol_nodes or not surf_nodes:
        raise ValueError("n_conn_nodes must be >= 2 so both blocks own nodes")
    if n_volumetric < len(vol_nodes) or n_surface_total < len(surf_nodes):
        raise ValueError("each connectome node must own at least one dipole")

    # Drop nodes from the optimization set alternating between the two groups, so
    # `dipole_node_labels` carries -1 in *both* halves (see the module docstring).
    pool = [x for pair in zip(reversed(vol_nodes), reversed(surf_nodes)) for x in pair]
    pool += [x for x in reversed(vol_nodes + surf_nodes) if x not in pool]
    dropped = sorted(pool[: n_conn_nodes - n_optim_nodes])
    keep, conn_to_optim, optim_to_conn = _optim_split(n_conn_nodes, dropped)

    # Round-robin dipole -> node inside each half: every node owns >= 1 dipole and
    # the counts are near-equal, which keeps the bookkeeping checkable by hand.
    node_of_dipole = np.concatenate([
        np.array([vol_nodes[i % len(vol_nodes)] for i in range(n_volumetric)]),
        np.array([surf_nodes[i % len(surf_nodes)] for i in range(n_surface_total)]),
    ]).astype(np.int64)

    full_ids = _full_atlas_ids(n_conn_nodes)
    node_names = _node_names(vol_nodes, surf_nodes)

    # --- geometry ------------------------------------------------------------
    # Radii are derived from `spacing` so the smoothing kernel (sigma = 1.5 *
    # spacing) reaches a realistic number of neighbours; see the module docstring.
    vol_pos = _sample_ball(rng, n_volumetric, _ball_radius(n_volumetric, spacing))
    vol_pos += np.array(_BLOCK_CENTRES["volumetric"])

    surf_pos, surf_dirs = [], []
    for name, n_blk in zip(BLOCK_ORDER[1:], block_sizes[1:]):
        # a surface block is a spherical patch: outward-radial normals, as a real
        # mid-surface has, and the same density rule in 2D.
        d = _sample_shell(rng, n_blk, left_only=name.endswith("lh_middle"))
        surf_pos.append(d * _shell_radius(n_blk, spacing) + np.array(_BLOCK_CENTRES[name]))
        surf_dirs.append(d)
    positions = np.concatenate([vol_pos, *surf_pos])

    vol_dirs = _unit(rng.standard_normal((n_volumetric, 3)))
    directions = np.concatenate([vol_dirs, *surf_dirs])

    volume = rng.uniform(8.0, 18.0, size=n_dip)
    neural_density = rng.uniform(0.1, 0.5, size=n_dip)

    # Free ('R') and fixed ('G'/'P') orientations in the volumetric block, pure
    # normals on every surface block -- the real mix, and the reason
    # orientation-seed tests have anything to re-draw.
    vol_orient = np.array([("R", "R", "P", "R", "G")[i % 5] for i in range(n_volumetric)])
    surf_orient = np.array(["N"] * n_surface_total)
    orient_type = np.concatenate([vol_orient, surf_orient]).astype("<U1")

    parcel_labels = full_ids[node_of_dipole].astype(np.int64)

    # --- write ---------------------------------------------------------------
    _write_participants(root, subj)
    _write_raw(deriv, subj, rng)
    _write_atlas(deriv, subj, atlas_res, full_ids, node_names)
    _write_electrodes(deriv, subj, electrode_names)
    _write_dipoles(
        deriv, subj, spacing, blocks, atlas_res,
        positions=positions, directions=directions, volume=volume,
        neural_density=neural_density, orient_type=orient_type,
        parcel_labels=parcel_labels, node_of_dipole=node_of_dipole,
    )
    _write_leadfields(deriv, subj, rng, spacing, n_elec, n_dip,
                      n_eye_sources, n_muscle_sources)
    _write_connectivity(deriv, subj, rng, atlas_res, n_conn_nodes, full_ids, node_names)
    _write_fmri(deriv, subj, rng, atlas_res, keep, conn_to_optim, optim_to_conn,
                full_ids, node_names, n_trs, n_conn_nodes, n_optim_nodes)
    _write_eeg(deriv, subj, rng, eeg_channel_names)
    _write_artifacts(deriv, subj, rng, n_eye_sources, n_muscle_sources)
    _write_qc(deriv, subj)
    return root


# --- geometry helpers --------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _ball_radius(n: int, spacing: float) -> float:
    """Radius of a ball holding ``n`` random points at median nearest-neighbour
    distance ~= ``spacing``.

    A Poisson sample of density ``lambda`` has mean nearest-neighbour distance
    ``0.554 * lambda**(-1/3)`` in 3D, so "one point per ``(2*spacing)**3``" lands
    the nearest neighbour at roughly ``spacing``. That in turn puts ~7-13
    dipoles inside the ``sigma = 1.5 * spacing`` smoothing kernel, matching the
    effective neighbour counts measured on real blocks.
    """
    return 2.0 * spacing * (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)


def _shell_radius(n: int, spacing: float) -> float:
    """Same rule in 2D, for ``n`` points spread over (half) a spherical shell."""
    return 2.0 * spacing * float(np.sqrt(n / (2.0 * np.pi)))


def _sample_ball(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """Uniform points inside a ball of the given radius (mm)."""
    d = _unit(rng.standard_normal((n, 3)))
    r = radius * rng.uniform(0.0, 1.0, size=(n, 1)) ** (1 / 3)
    return d * r


def _sample_shell(
    rng: np.random.Generator, n: int, *, left_only: bool = False
) -> np.ndarray:
    """``n`` unit vectors -- the outward normals of a spherical patch."""
    d = _unit(rng.standard_normal((n, 3)))
    if left_only:  # a left-hemisphere surface block lives at x < 0
        d[:, 0] = -np.abs(d[:, 0])
    return d


def _spiral_direction(i: int, n: int) -> tuple[float, float, float]:
    """Golden-angle point on the upper unit sphere -- the fallback position for an
    electrode name that is not one of the built-in 10-5 landmarks."""
    z = 1.0 - (i + 0.5) / n
    r = float(np.sqrt(max(1.0 - z * z, 0.0)))
    phi = float(np.pi * (1.0 + 5.0**0.5) * i)
    return (r * float(np.cos(phi)), r * float(np.sin(phi)), float(z))


def _full_atlas_ids(n_conn_nodes: int) -> np.ndarray:
    """Sparse full-atlas ids, one per connectome node (id 0 stays for Unknown)."""
    if n_conn_nodes <= len(FULL_ATLAS_IDS):
        return np.array(FULL_ATLAS_IDS[:n_conn_nodes], dtype=np.int64)
    return np.arange(1, n_conn_nodes + 1, dtype=np.int64) * 2


def _node_names(vol_nodes: list[int], surf_nodes: list[int]) -> list[str]:
    names: list[str] = []
    for j in range(len(vol_nodes)):
        names.append(_SUBCORTICAL_NAMES[j] if j < len(_SUBCORTICAL_NAMES)
                     else f"Left-Subcortical-{j}")
    for j in range(len(surf_nodes)):
        names.append(_CORTICAL_NAME.format(i=j + 1))
    return names


def _optim_split(m: int, dropped: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``keep`` / ``conn_to_optim`` / ``optim_to_conn`` for ``m`` connectome rows."""
    keep = np.ones(m, dtype=bool)
    keep[dropped] = False
    conn_to_optim = np.full(m, -1, dtype=np.int32)
    conn_to_optim[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    optim_to_conn = np.flatnonzero(keep).astype(np.int32)
    return keep, conn_to_optim, optim_to_conn


def _symmetric(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    """A symmetric, zero-diagonal, non-negative matrix."""
    a = rng.uniform(0.0, scale, size=(n, n))
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    return a


# --- writers -----------------------------------------------------------------
def _dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _write_participants(root: Path, subj: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\tskip_t2_registration\tno_neck\tmp2rage\n"
        f"{subj}\t42.5\tF\tfalse\tfalse\ttrue\n"
    )


def _write_raw(deriv: Path, subj: str, rng: np.random.Generator) -> None:
    """A 4x4x4 T1 -- enough for ``s.load.t1().affine`` and any nibabel round-trip."""
    import nibabel as nib

    d = deriv / "raw" / subj
    d.mkdir(parents=True, exist_ok=True)
    data = rng.integers(0, 1000, size=(4, 4, 4)).astype(np.int16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), d / "T1.nii.gz")
    _dump_json(d / "T1.json", {"Modality": "MR", "Note": "synthetic test fixture"})


def _write_atlas(
    deriv: Path, subj: str, res: int, full_ids: np.ndarray, node_names: list[str]
) -> None:
    """``atlas<res>.nii.gz`` exists so ``Subject.atlas_resolutions()`` discovers it."""
    import nibabel as nib

    d = deriv / "atlas" / subj
    d.mkdir(parents=True, exist_ok=True)
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol.reshape(-1)[: len(full_ids)] = full_ids  # one voxel per node, deterministic
    nib.save(nib.Nifti1Image(vol, np.eye(4)), d / f"atlas{res}.nii.gz")
    lines = [f"{i}\t{nm}\t0 0 0 255" for i, nm in zip(full_ids, node_names)]
    (d / f"atlas{res}_LUT.txt").write_text("\n".join(lines) + "\n")


def _write_electrodes(deriv: Path, subj: str, names: tuple[str, ...]) -> None:
    d = deriv / "electrodes" / subj
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, nm in enumerate(names):
        v = np.array(_ELECTRODE_DIRECTIONS.get(nm) or _spiral_direction(i, len(names)), float)
        # repr() of a *python* float: full round-trip precision, and no numpy
        # scalar wrapper (numpy 2 reprs are `np.float64(...)`).
        x, y, z = (float(c) for c in v / np.linalg.norm(v) * HEAD_RADIUS_MM)
        lines.append(f"{nm}, {x!r}, {y!r}, {z!r}")
    (d / "landmarks_10-5-full.csv").write_text("\n".join(lines) + "\n")
    _dump_json(d / "selected_landmarks_10-5-full.json", list(names))


def _write_dipoles(
    deriv: Path,
    subj: str,
    spacing: float,
    blocks: _Blocks,
    atlas_res: int,
    *,
    positions: np.ndarray,
    directions: np.ndarray,
    volume: np.ndarray,
    neural_density: np.ndarray,
    orient_type: np.ndarray,
    parcel_labels: np.ndarray,
    node_of_dipole: np.ndarray,
) -> None:
    d = deriv / "dipoles" / subj / f"spacing{spacing}mm"
    d.mkdir(parents=True, exist_ok=True)
    n_dip = len(positions)

    _save(d / "dipole_positions.npy", positions)
    _save(d / "dipole_directions.npy", directions)
    _save(d / "dipole_volume.npy", volume)
    _save(d / "dipole_neural_density.npy", neural_density)
    _save(d / "orient_type.npy", orient_type)
    _save(d / f"{atlas_res}Parcels_dipole_labels.npy", parcel_labels)
    # The aggregated-atlas label is a coarser id space than the Schaefer parcels;
    # nothing in the API interprets it, so node index + 1 is a faithful stand-in.
    _save(d / "aggregated_dipole_labels.npy", (node_of_dipole + 1).astype(np.int64))

    for i, name in enumerate(blocks.names):
        mask = blocks.mask(i, n_dip)
        bd = d / name
        bd.mkdir(parents=True, exist_ok=True)
        bpos = positions[mask]
        _save(bd / "dipole_traceback.npy", mask)
        _save(bd / "dipole_positions.npy", bpos)
        _save(bd / "dipole_volume.npy", volume[mask])
        _save(bd / "orient_type.npy", orient_type[mask])
        # The true pairwise distance matrix: symmetric with a zero diagonal, so a
        # Gaussian smoothing kernel built from it is real geometry, not noise.
        dm = np.linalg.norm(bpos[:, None, :] - bpos[None, :, :], axis=-1)
        np.fill_diagonal(dm, 0.0)
        _save(bd / "distance_matrix.npy", 0.5 * (dm + dm.T))
        if name == "volumetric":
            _save(bd / "dipole_preferential_direction.npy", directions[mask])
            _save(bd / "dipole_labels.npy", parcel_labels[mask])
            # NOTE: volumetric/ deliberately has no <res>Parcels_dipole_labels.npy
            # -- the real tree doesn't either.
        else:
            _save(bd / "dipole_normals.npy", directions[mask])
            _save(bd / "sampled_vertices.npy", np.arange(int(mask.sum()), dtype=np.int64) * 7)
            _save(bd / f"{atlas_res}Parcels_dipole_labels.npy", parcel_labels[mask])


def _write_leadfields(
    deriv: Path,
    subj: str,
    rng: np.random.Generator,
    spacing: float,
    n_elec: int,
    n_dip: int,
    n_eye: int,
    n_muscle: int,
) -> None:
    """``processed_<key>-leadfield.npy`` at ``(n_elec, 3 * n_src)``, average-referenced.

    Filenames come from :func:`leadfield_keys` / :data:`ARTIFACT_LEADFIELD_KEYS`, so
    the constants and the tree can never drift.
    """
    d = deriv / "leadfields" / subj
    d.mkdir(parents=True, exist_ok=True)

    def _ref(a: np.ndarray) -> np.ndarray:
        return a - a.mean(axis=0, keepdims=True)

    iso_key, aniso_key = leadfield_keys(spacing)
    base = _ref(rng.normal(0.0, LEADFIELD_SIGMA, size=(n_elec, 3 * n_dip)))
    _save(d / f"processed_{iso_key}-leadfield.npy", base)
    # The anisotropic variant is the isotropic one perturbed: highly correlated but
    # distinct, exactly as the real pair is.
    aniso = _ref(base + rng.normal(0.0, 0.1 * LEADFIELD_SIGMA, size=base.shape))
    _save(d / f"processed_{aniso_key}-leadfield.npy", aniso)

    eyes_key, muscle_key = ARTIFACT_LEADFIELD_KEYS
    for key, n_src in ((eyes_key, n_eye), (muscle_key, n_muscle)):
        _save(d / f"processed_{key}-leadfield.npy",
              _ref(rng.normal(0.0, ARTIFACT_LEADFIELD_SIGMA, size=(n_elec, 3 * n_src))))


def _write_connectivity(
    deriv: Path,
    subj: str,
    rng: np.random.Generator,
    res: int,
    m: int,
    full_ids: np.ndarray,
    node_names: list[str],
) -> None:
    d = deriv / "connectivity" / subj
    d.mkdir(parents=True, exist_ok=True)

    np.savetxt(d / f"weights_{res}.txt", _symmetric(rng, m, 5000.0))
    np.savetxt(d / f"weights_invnodevol_{res}.txt", _symmetric(rng, m, 1.0))
    np.savetxt(d / f"distances_{res}.txt", _symmetric(rng, m, 120.0))

    # labels_<res>.txt has M+1 lines: the leading Unknown node is dropped from the
    # matrices (Subject.load.connectivity_labels slices it off).
    (d / f"labels_{res}.txt").write_text("\n".join(["Unknown", *node_names]) + "\n")

    # full_to_reduced: full atlas id -> reduced id, 0 == Unknown, -1 == not in the
    # connectome. Length is (max full id + 1) plus a couple of unused trailing slots,
    # mirroring the real sparse map.
    f2r = np.full(int(full_ids.max()) + 3, -1, dtype=np.int64)
    f2r[0] = 0
    f2r[full_ids] = np.arange(1, m + 1, dtype=np.int64)
    _save(d / f"full_to_reduced_{res}.npy", f2r)

    r2f = np.concatenate([[0], full_ids]).astype(np.int64)
    _save(d / f"reduced_to_full_{res}.npy", r2f)


def _write_fmri(
    deriv: Path,
    subj: str,
    rng: np.random.Generator,
    res: int,
    keep: np.ndarray,
    conn_to_optim: np.ndarray,
    optim_to_conn: np.ndarray,
    full_ids: np.ndarray,
    node_names: list[str],
    n_trs: int,
    n_conn_nodes: int,
    n_optim_nodes: int,
) -> None:
    d = deriv / "fMRI" / subj
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{subj}_task-{FMRI_TASK}_atlas-schaefer"

    # A SECOND resolution with different M and K, so a consumer that picks the
    # wrong `_<res>` suffix hits a length mismatch instead of silently working.
    # Only the primary resolution has an atlas/connectome/dipole-label set.
    res2 = SECOND_ATLAS_RES if res != SECOND_ATLAS_RES else res + 100
    m2 = n_conn_nodes + _SECOND_RES_EXTRA_CONN
    k2 = min(n_optim_nodes + _SECOND_RES_EXTRA_OPTIM, m2)
    keep2, c2o2, o2c2 = _optim_split(m2, list(range(k2, m2)))
    ids2 = np.arange(1, m2 + 1, dtype=np.int64) * 2
    names2 = [f"Res{res2}-Node-{j}" for j in range(m2)]

    nodes: dict[str, np.ndarray] = {}
    series: dict[str, np.ndarray] = {}
    for r, kp, c2o, o2c, ids, names in (
        (res, keep, conn_to_optim, optim_to_conn, full_ids, node_names),
        (res2, keep2, c2o2, o2c2, ids2, names2),
    ):
        nodes[f"keep_{r}"] = kp
        nodes[f"optim_to_conn_{r}"] = o2c
        nodes[f"conn_to_optim_{r}"] = c2o
        ts = rng.standard_normal((len(kp), n_trs)).astype(np.float32)
        ts[~kp] = np.nan  # the invariant: NaN rows are exactly the dropped nodes
        series[f"ts_{r}"] = ts
        series[f"ids_{r}"] = ids.astype(np.int32)
        series[f"labels_{r}"] = np.array(names)
        series[f"nvox_{r}"] = np.where(kp, 120, 3).astype(np.int32)

    np.savez(d / f"{stem}_desc-optim_nodes.npz", **nodes)
    np.savez(d / f"{stem}_desc-conn_timeseries.npz", **series)
    _dump_json(d / f"{subj}_task-{FMRI_TASK}_timeseries.json",
               {"RepetitionTime": 1.4, "Atlas": "schaefer", "Note": "synthetic test fixture"})


def _write_eeg(
    deriv: Path, subj: str, rng: np.random.Generator, channel_names: tuple[str, ...]
) -> None:
    d = deriv / "EEG" / subj
    d.mkdir(parents=True, exist_ok=True)
    segs = {
        f"seg_{i:03d}": rng.standard_normal((len(channel_names), n)).astype(np.float32)
        for i, n in enumerate(EEG_SEGMENT_LENGTHS)
    }
    np.savez(d / f"{subj}_task-{EEG_TASK}_eeg.npz", **segs)
    _dump_json(
        d / f"{subj}_task-{EEG_TASK}_eeg.json",
        {
            "condition": EEG_TASK,
            "condition_code": "EC",
            "sampling_frequency": EEG_SFREQ,
            "n_channels": len(channel_names),
            "channel_names": list(channel_names),
            "units": "V",
            "n_segments": len(EEG_SEGMENT_LENGTHS),
            "segment_lengths_samples": list(EEG_SEGMENT_LENGTHS),
            "total_samples": int(sum(EEG_SEGMENT_LENGTHS)),
            "notes": "Synthetic test fixture; channel_names is a subset of the montage.",
        },
    )


#: Artifact source geometry: eye sources sit in the orbits (anterior, inferior),
#: muscle sources on the lower/lateral scalp -- so an "eyes are frontal" sanity
#: check on the positions is meaningful rather than accidental.
_ORBIT_CENTRES: tuple[tuple[float, float, float], ...] = (
    (-30.0, 68.0, -18.0),  # Eye_left
    (30.0, 68.0, -18.0),   # Eye_right
)
_MUSCLE_RADIUS_MM = 78.0


def _write_artifacts(
    deriv: Path, subj: str, rng: np.random.Generator, n_eye: int, n_muscle: int
) -> None:
    reg = deriv / "artifacts" / "registration" / subj
    reg.mkdir(parents=True, exist_ok=True)
    # A mild affine plus its exact inverse, so round-tripping a point is a no-op.
    a = np.eye(4)
    a[:3, :3] += 0.05 * rng.standard_normal((3, 3))
    a[:3, 3] = rng.normal(0.0, 2.0, size=3)
    _save(reg / "mni_to_subject_affine.npy", a)
    _save(reg / "subject_to_mni_affine.npy", np.linalg.inv(a))
    _dump_json(reg / "registration_qc.json", {"status": "pass", "method": "antspyx-affine"})

    dip = deriv / "artifacts" / "dipoles" / subj

    side = np.array([i % 2 for i in range(n_eye)])
    eye_pos = np.array(_ORBIT_CENTRES)[side] + _sample_ball(rng, n_eye, radius=8.0)
    eye_labels = np.array([("Eye_left", "Eye_right")[i] for i in side])

    # lower/lateral scalp: |z| pushed negative, radius just inside the electrodes
    md = _sample_shell(rng, n_muscle)
    md[:, 2] = -np.abs(md[:, 2])
    muscle_pos = _unit(md) * _MUSCLE_RADIUS_MM
    muscle_names = ("Muscle_DepressorLabii_left", "Muscle_Temporalis_right")
    muscle_labels = np.array([muscle_names[i % 2] for i in range(n_muscle)])

    for kind, n, pos, labels in (
        ("eyes", n_eye, eye_pos, eye_labels),
        ("muscle", n_muscle, muscle_pos, muscle_labels),
    ):
        kd = dip / kind
        kd.mkdir(parents=True, exist_ok=True)
        _save(kd / "dipole_positions.npy", pos)
        _save(kd / "dipole_preferential_direction.npy", _unit(rng.standard_normal((n, 3))))
        _save(kd / "dipole_labels.npy", labels)
        _save(kd / "dipole_volume.npy", rng.uniform(50.0, 200.0, size=n))
        _save(kd / "orient_type.npy", np.array(["A"] * n, dtype="<U1"))

    _dump_json(dip / "artifactsources.json", {
        "eyes": {"n_dipoles": n_eye},
        "muscle": {"n_kept": n_muscle, "n_dropped": 0, "n_total": n_muscle,
                   "kept_fraction": 1.0, "neck_coverage": True},
    })
    _dump_json(dip / "solve_groups.json", [
        {"name": "eyes", "dipoles_dir": f"artifacts/dipoles/{subj}/eyes",
         "valid_tissues": ["Eye (Aqueous Humor)"], "out_tag": "_artifact-eyes-CGAL"},
        {"name": "muscle", "dipoles_dir": f"artifacts/dipoles/{subj}/muscle",
         "valid_tissues": ["Muscle", "Skin"], "out_tag": "_artifact-muscle-CGAL"},
    ])


def _write_qc(deriv: Path, subj: str) -> None:
    """All stages ``pass`` -- QC-aware loaders must stay silent on this fixture."""
    _dump_json(deriv / "qc" / subj / "qc_report.json", {
        "subject": subj,
        "generated": "2000-01-01T00:00:00",
        "overall_status": "pass",
        "stages": [
            {"name": nm, "title": nm, "present": True, "status": "pass",
             "notes": [], "checks": [], "figures": []}
            for nm in QC_STAGES
        ],
    })
