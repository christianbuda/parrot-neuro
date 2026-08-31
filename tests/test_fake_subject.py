"""Self-test for the synthetic derivatives tree.

Everything here goes through the **public** ``Subject`` API, so this file does
double duty: it proves ``build_fake_subject`` produces a valid Subject, and it
pins the invariants that the (not-yet-written) ``parrot_neuro.simulation`` tests
will rely on. If one of these fails, the fixture is wrong -- not the test.
"""
from __future__ import annotations

import filecmp
import warnings
from pathlib import Path

import numpy as np
import pytest

from fixtures.build_fake_subject import (
    ARTIFACT_LEADFIELD_KEYS,
    ATLAS_RES,
    BLOCK_ORDER,
    EEG_CHANNEL_NAMES,
    EEG_SEGMENT_LENGTHS,
    EEG_SFREQ,
    EEG_TASK,
    ELECTRODE_NAMES,
    LEADFIELD_KEYS,
    N_CONN_NODES,
    N_EYE_SOURCES,
    N_MUSCLE_SOURCES,
    N_OPTIM_NODES,
    N_CEREBELLUM,
    N_SURFACE,
    N_VOLUMETRIC,
    SECOND_ATLAS_RES,
    SPACING,
    SUBJECT_ID,
    SURFACE_BLOCKS,
    build_fake_subject,
    leadfield_keys,
)
from parrot_neuro import Subject

N_DIP = N_VOLUMETRIC + N_SURFACE + N_CEREBELLUM
#: Aggregated order -- volumetric FIRST, and the surface blocks NOT sorted among
#: themselves. Both halves of the trap; see the fixture docstring.
BLOCKS = BLOCK_ORDER
#: Per-block sizes, in aggregated order.
BLOCK_SIZES = (N_VOLUMETRIC, N_SURFACE, N_CEREBELLUM)


# --- it is a valid Subject ---------------------------------------------------
def test_subject_constructs_and_discovers(fake_subject: Subject):
    s = fake_subject
    assert s.subj == f"sub-{SUBJECT_ID}"
    assert s.dipole_spacings() == [SPACING]
    assert s.atlas_resolutions() == [ATLAS_RES]
    assert s.available_leadfields() == sorted(LEADFIELD_KEYS + ARTIFACT_LEADFIELD_KEYS)
    assert s.participants_row["participant_id"] == s.subj


def test_optional_stage_flags(fake_subject: Subject):
    s = fake_subject
    assert (s.has_eeg, s.has_fmri, s.has_optim_nodes, s.has_artifacts) == (True,) * 4
    # deliberately absent stages must report False, not blow up
    assert (s.has_dwi, s.has_anisotropy) == (False, False)


def test_qc_is_clean_so_loaders_stay_silent(fake_subject: Subject):
    assert fake_subject.qc["overall_status"] == "pass"
    assert fake_subject.qc_status("dipoles") == "pass"
    assert fake_subject.qc_status("raw") == "pass"  # dir name -> QC name ("ingest")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any QC warning becomes a failure
        fake_subject.load.leadfield(LEADFIELD_KEYS[0])


def test_t1_loads(fake_subject: Subject):
    img = fake_subject.load.t1()
    assert img.affine.shape == (4, 4) and img.shape == (4, 4, 4)


def test_electrodes(fake_subject: Subject):
    el = fake_subject.load.electrodes()
    assert list(el) == list(ELECTRODE_NAMES)
    pos = np.array(list(el.values()))
    assert pos.shape == (len(ELECTRODE_NAMES), 3)
    assert np.allclose(np.linalg.norm(pos, axis=1), 85.0)  # all on the head sphere


def test_leadfield_shapes_and_average_reference(fake_subject: Subject):
    n_elec = len(ELECTRODE_NAMES)
    for key in LEADFIELD_KEYS:
        lf = fake_subject.load.leadfield(key)
        assert lf.shape == (n_elec, 3 * N_DIP) and lf.dtype == np.float64
        assert np.allclose(lf.sum(axis=0), 0.0)  # average-referenced, as the real ones are
    assert fake_subject.load.leadfield(ARTIFACT_LEADFIELD_KEYS[0]).shape == (
        n_elec, 3 * N_EYE_SOURCES
    )
    assert fake_subject.load.leadfield(ARTIFACT_LEADFIELD_KEYS[1]).shape == (
        n_elec, 3 * N_MUSCLE_SOURCES
    )


# --- dipole block structure: the whole point of the fixture ------------------
def test_dipole_bundle_shapes_and_dtypes(fake_subject: Subject):
    d = fake_subject.load.dipoles(SPACING)
    assert len(d) == N_DIP
    assert d.positions.shape == (N_DIP, 3) and d.positions.dtype == np.float64
    assert d.directions.shape == (N_DIP, 3)
    assert np.allclose(np.linalg.norm(d.directions, axis=1), 1.0)
    assert d.volume.shape == (N_DIP,) and (d.volume > 0).all()
    assert d.neural_density.shape == (N_DIP,)
    assert d.orient_type.dtype == np.dtype("<U1")
    assert set(np.unique(d.orient_type)) == {"R", "P", "G", "N"}
    assert (d.orient_type == "R").sum() > 0  # free orientations exist to re-draw


def _block_starts(base) -> dict[str, int]:
    """Block -> its first index on the aggregated axis, read from the traceback masks.

    This is the *only* correct way to recover block order; every test below that
    cares about ordering derives it from here rather than from a directory listing.
    """
    return {b: int(np.flatnonzero(np.load(base / b / "dipole_traceback.npy"))[0])
            for b in BLOCKS}


def test_block_order_is_volumetric_first_and_surfaces_unsorted(fake_subject: Subject):
    """The aggregated order is ``volumetric`` first AND the surface blocks unsorted.

    Both halves matter. The reference implementation on ``eeg-bold-fit``
    (``optimization/forward.py::get_electric_signals``) builds the block list as
    ``surfaces/* + [volumetric]`` and so gets the first half wrong; the obvious
    naive repair, ``[volumetric] + sorted(surfaces)``, gets the second half wrong.
    A fixture that only trapped the first would bless that repair.
    """
    base = fake_subject.path.dipole_dir(SPACING)
    starts = _block_starts(base)
    assert starts == dict(zip(BLOCKS, np.cumsum((0,) + BLOCK_SIZES[:-1])))

    true_order = sorted(starts, key=starts.get)
    surf = [b for b in BLOCKS if b.startswith("surfaces/")]
    # every plausible directory-derived ordering disagrees with the truth on disk
    assert true_order != sorted(surf) + ["volumetric"]          # the reference bug
    assert true_order != ["volumetric"] + sorted(surf)          # the naive repair
    assert true_order != sorted(BLOCKS)                         # plain alphabetical
    assert true_order != [f"surfaces/{p.name}" for p in (base / "surfaces").iterdir()] + [
        "volumetric"
    ]                                                           # filesystem order


def test_wrong_block_orders_change_a_projected_matrix(fake_subject: Subject):
    """A wrong block order is not merely a different list -- it moves the numbers.

    Builds the plan's ``M = L . blockdiag(S) . E`` on the fixture and confirms each
    wrong ordering perturbs it by a margin no tolerance would hide. Without this the
    ordering test above could pass on a fixture whose blocks are interchangeable.
    """
    s = fake_subject
    base = s.path.dipole_dir(SPACING)
    starts = _block_starts(base)
    d = s.load.dipoles(SPACING)
    node = s.load.dipole_node_labels(ATLAS_RES, SPACING)
    k = int(node.max()) + 1
    sigma = 1.5 * SPACING

    def build(order):
        raw = s.load.leadfield(LEADFIELD_KEYS[0])
        lf = (raw.reshape(raw.shape[0], -1, 3) * d.directions[None]).sum(-1)
        e = np.zeros((N_DIP, k))
        e[node >= 0, node[node >= 0]] = 1.0
        se, off = np.zeros((N_DIP, k)), 0
        for b in order:
            mask = np.load(base / b / "dipole_traceback.npy")
            dm = np.load(base / b / "distance_matrix.npy")
            w = np.exp(-(dm ** 2) / (2 * sigma ** 2)) * d.volume[None, mask]
            w /= w.sum(axis=1, keepdims=True)
            se[off:off + len(dm)] = w @ e[off:off + len(dm)]
            off += len(dm)
        return lf @ se

    true_order = sorted(starts, key=starts.get)
    surf = [b for b in BLOCKS if b.startswith("surfaces/")]
    m = build(true_order)
    assert np.linalg.matrix_rank(m) == k  # non-degenerate, so the test can bite
    for wrong in (sorted(surf) + ["volumetric"], ["volumetric"] + sorted(surf)):
        rel = np.linalg.norm(build(wrong) - m) / np.linalg.norm(m)
        assert rel > 0.05, f"{wrong} is only {rel:.1%} off -- the fixture has no teeth"


def test_traceback_masks_partition_and_agree_with_aggregated(fake_subject: Subject):
    base = fake_subject.path.dipole_dir(SPACING)
    d = fake_subject.load.dipoles(SPACING)
    covered = np.zeros(N_DIP, dtype=bool)
    for b in BLOCKS:
        m = np.load(base / b / "dipole_traceback.npy")
        assert m.dtype == np.bool_ and m.shape == (N_DIP,)
        assert not (covered & m).any()  # blocks do not overlap
        covered |= m
        assert np.array_equal(d.positions[m], np.load(base / b / "dipole_positions.npy"))
        assert np.array_equal(d.volume[m], np.load(base / b / "dipole_volume.npy"))
        assert np.array_equal(d.orient_type[m], np.load(base / b / "orient_type.npy"))
        dirs = "dipole_preferential_direction" if b == "volumetric" else "dipole_normals"
        assert np.array_equal(d.directions[m], np.load(base / b / f"{dirs}.npy"))
    assert covered.all()  # ...and they cover everything


def test_distance_matrices_are_true_pairwise_distances(fake_subject: Subject):
    base = fake_subject.path.dipole_dir(SPACING)
    for b in BLOCKS:
        pos = np.load(base / b / "dipole_positions.npy")
        dm = np.load(base / b / "distance_matrix.npy")
        assert dm.shape == (len(pos), len(pos)) and dm.dtype == np.float64
        expected = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        assert np.allclose(dm, expected)
        assert np.array_equal(dm, dm.T) and np.allclose(np.diag(dm), 0.0)


def test_volumetric_block_has_no_parcel_labels(fake_subject: Subject):
    """Faithful to the real tree: only surface blocks carry <res>Parcels labels."""
    base = fake_subject.path.dipole_dir(SPACING)
    assert not (base / "volumetric" / f"{ATLAS_RES}Parcels_dipole_labels.npy").exists()
    for b in SURFACE_BLOCKS:
        assert (base / "surfaces" / b / f"{ATLAS_RES}Parcels_dipole_labels.npy").exists()


# --- label / node bookkeeping ------------------------------------------------
def test_dipole_labels_map_onto_every_connectome_node(fake_subject: Subject):
    s = fake_subject
    labels = s.load.dipole_labels(ATLAS_RES, SPACING)
    assert labels.shape == (N_DIP,) and labels.dtype == np.int64
    f2r = s.load.npy("connectivity", f"full_to_reduced_{ATLAS_RES}.npy")
    rows = f2r[labels] - 1  # the connectome's off-by-one: id 0 == Unknown
    assert rows.min() >= 0 and rows.max() < N_CONN_NODES
    # every connectome node owns at least one dipole (no empty region)
    assert set(np.unique(rows)) == set(range(N_CONN_NODES))


def test_dipole_node_labels_span_optim_axis_and_contain_dropped(fake_subject: Subject):
    dn = fake_subject.load.dipole_node_labels(ATLAS_RES, SPACING)
    assert dn.shape == (N_DIP,)
    assert set(np.unique(dn)) <= {-1} | set(range(N_OPTIM_NODES))
    assert (dn == -1).any(), "the valid-mask logic must be exercised"
    assert set(np.unique(dn[dn >= 0])) == set(range(N_OPTIM_NODES))
    # dropped nodes exist in BOTH blocks, so neither half of the axis is trivially valid
    assert (dn[:N_VOLUMETRIC] == -1).any() and (dn[N_VOLUMETRIC:] == -1).any()


def test_fmri_node_alignment(fake_subject: Subject):
    s = fake_subject
    nodes = s.load.fmri_nodes(ATLAS_RES)
    assert nodes.keep.shape == (N_CONN_NODES,) and nodes.keep.dtype == np.bool_
    assert nodes.to_conn.dtype == np.int32 and nodes.from_conn.dtype == np.int32
    assert len(nodes) == N_OPTIM_NODES and nodes.to_conn.shape == (N_OPTIM_NODES,)
    # M -> K -> M is the identity on kept nodes
    assert np.array_equal(nodes.from_conn[nodes.to_conn], np.arange(N_OPTIM_NODES))
    assert (nodes.from_conn[~nodes.keep] == -1).all()


def test_connectivity_matrices(fake_subject: Subject):
    s = fake_subject
    for W in (s.load.weights(ATLAS_RES), s.load.weights(ATLAS_RES, normalized=True),
              s.load.distances(ATLAS_RES)):
        assert W.shape == (N_CONN_NODES, N_CONN_NODES)
        assert np.array_equal(W, W.T) and np.allclose(np.diag(W), 0.0)
    names = s.load.connectivity_labels(ATLAS_RES)
    assert len(names) == N_CONN_NODES and "Unknown" not in names
    assert s.path.connectivity_labels(ATLAS_RES).read_text().splitlines()[0] == "Unknown"


def test_fmri_aligned_slicing_matches_test_subject_conventions(fake_subject: Subject):
    """The same invariants ``tests/test_subject.py::test_fmri_aligned_loaders`` asserts
    against real data, so the fixture is a drop-in stand-in for that test."""
    s = fake_subject
    keep = s.load.fmri_keep(ATLAS_RES)
    W = s.load.weights(ATLAS_RES, fmri_aligned=True)
    assert W.shape == (N_OPTIM_NODES, N_OPTIM_NODES)
    assert np.array_equal(W, s.load.weights(ATLAS_RES)[np.ix_(keep, keep)])
    assert len(s.load.connectivity_labels(ATLAS_RES, fmri_aligned=True)) == N_OPTIM_NODES
    ts = np.asarray(s.load.fmri_timeseries("conn")[f"ts_{ATLAS_RES}"])
    assert ts.dtype == np.float32 and ts.shape[0] == N_CONN_NODES
    # NaN rows are exactly the dropped nodes -- the real cross-file invariant
    assert np.array_equal(~np.isnan(ts).any(axis=1), keep)


# --- staged inputs -----------------------------------------------------------
def test_eeg_segments_and_sidecar(fake_subject: Subject):
    s = fake_subject
    npz = s.load.eeg(EEG_TASK)
    assert sorted(npz.files) == [f"seg_{i:03d}" for i in range(len(EEG_SEGMENT_LENGTHS))]
    for key, n in zip(sorted(npz.files), EEG_SEGMENT_LENGTHS):
        assert npz[key].shape == (len(EEG_CHANNEL_NAMES), n)
        assert npz[key].dtype == np.float32
    import json

    meta = json.loads(s.path.eeg(EEG_TASK).with_suffix(".json").read_text())
    assert meta["sampling_frequency"] == EEG_SFREQ
    assert meta["channel_names"] == list(EEG_CHANNEL_NAMES)
    # the recorded montage is a strict, differently-ordered subset of the electrodes
    assert set(meta["channel_names"]) < set(ELECTRODE_NAMES)
    assert meta["channel_names"] != [n for n in ELECTRODE_NAMES if n in EEG_CHANNEL_NAMES]


def test_artifact_sources(fake_subject: Subject):
    s = fake_subject
    meta = s.load.artifact_sources()
    assert meta["eyes"]["n_dipoles"] == N_EYE_SOURCES
    assert meta["muscle"]["n_kept"] == N_MUSCLE_SOURCES and meta["muscle"]["neck_coverage"]
    for kind, n in (("eyes", N_EYE_SOURCES), ("muscle", N_MUSCLE_SOURCES)):
        assert s.load.artifact_dipole("dipole_positions", kind).shape == (n, 3)
        assert s.load.artifact_dipole("dipole_preferential_direction", kind).shape == (n, 3)
        assert s.load.artifact_dipole("dipole_volume", kind).shape == (n,)
        assert (s.load.artifact_dipole("orient_type", kind) == "A").all()
        assert s.load.artifact_dipole("dipole_labels", kind).dtype.kind == "U"


def test_artifact_affines_round_trip(fake_subject: Subject):
    a = fake_subject.load.artifact_affine("mni_to_subject")
    b = fake_subject.load.artifact_affine("subject_to_mni")
    assert a.shape == b.shape == (4, 4)
    assert np.allclose(a @ b, np.eye(4))


# --- determinism -------------------------------------------------------------
def _all_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_same_seed_is_byte_identical(tmp_path: Path):
    a = build_fake_subject(tmp_path / "a", seed=0)
    b = build_fake_subject(tmp_path / "b", seed=0)
    names = _all_files(a)
    assert names and names == _all_files(b)
    match, mismatch, errors = filecmp.cmpfiles(a, b, names, shallow=False)
    assert (mismatch, errors) == ([], [])


def test_different_seed_changes_the_data_but_not_the_layout(tmp_path: Path):
    a = build_fake_subject(tmp_path / "a", seed=0)
    b = build_fake_subject(tmp_path / "b", seed=1)
    assert _all_files(a) == _all_files(b)
    sa, sb = Subject(a, SUBJECT_ID), Subject(b, SUBJECT_ID)
    assert not np.array_equal(sa.load.dipoles(SPACING).positions,
                              sb.load.dipoles(SPACING).positions)
    # ...while the label bookkeeping is structural, hence seed-independent
    assert np.array_equal(sa.load.dipole_node_labels(ATLAS_RES, SPACING),
                          sb.load.dipole_node_labels(ATLAS_RES, SPACING))


def test_dimensions_are_overridable(tmp_path: Path):
    root = build_fake_subject(tmp_path / "small", "000042", seed=3,
                              n_volumetric=6, n_surface=9, n_cerebellum=4,
                              n_conn_nodes=4, n_optim_nodes=3, n_trs=5)
    s = Subject(root, "000042")
    n_dip = 6 + 9 + 4
    assert len(s.load.dipoles(SPACING)) == n_dip
    assert s.load.weights(ATLAS_RES).shape == (4, 4)
    assert s.load.leadfield(LEADFIELD_KEYS[0]).shape == (len(ELECTRODE_NAMES), 3 * n_dip)
    dn = s.load.dipole_node_labels(ATLAS_RES, SPACING)
    assert (dn == -1).any() and set(np.unique(dn[dn >= 0])) == {0, 1, 2}
    # the block structure survives the override, in the same aggregated order
    base = s.path.dipole_dir(SPACING)
    assert _block_starts(base) == dict(zip(BLOCKS, (0, 6, 15)))


def test_spacing_override_keeps_dipoles_and_leadfields_consistent(tmp_path: Path):
    """The leadfield key carries the spacing, so overriding it must move both.

    Getting this wrong yields a tree where ``dipole_spacings() == [4.0]`` but the
    only leadfield is named ``-2.0mm`` -- a projector cache keyed on
    ``(leadfield, atlas, spacing)`` would then be silently self-contradictory.
    """
    root = build_fake_subject(tmp_path / "sp", spacing=4.0)
    s = Subject(root, SUBJECT_ID)
    assert s.dipole_spacings() == [4.0]
    assert s.available_leadfields() == sorted(leadfield_keys(4.0) + ARTIFACT_LEADFIELD_KEYS)
    # and the pairing actually resolves through the public API
    assert s.load.leadfield(leadfield_keys(4.0)[0]).shape[1] == 3 * N_DIP


def test_second_atlas_resolution_present_only_in_the_fmri_npz(fake_subject: Subject):
    """Both fMRI npz files carry two resolutions with *different* node counts.

    A consumer that hardcodes or mis-selects the ``_<res>`` suffix then gets a length
    mismatch instead of silently working on the wrong axis. Only the first resolution
    has a full atlas/connectome/dipole-label set.
    """
    s = fake_subject
    z = np.load(s.path.optim_nodes())
    assert {f"{k}_{r}" for k in ("keep", "optim_to_conn", "conn_to_optim")
            for r in (ATLAS_RES, SECOND_ATLAS_RES)} == set(z.files)
    assert z[f"keep_{SECOND_ATLAS_RES}"].shape[0] != z[f"keep_{ATLAS_RES}"].shape[0]
    assert (z[f"optim_to_conn_{SECOND_ATLAS_RES}"].shape[0]
            != z[f"optim_to_conn_{ATLAS_RES}"].shape[0])
    ts = s.load.fmri_timeseries("conn")
    assert ts[f"ts_{SECOND_ATLAS_RES}"].shape[0] == z[f"keep_{SECOND_ATLAS_RES}"].shape[0]
    # ...but only the first resolution is a complete parcellation
    assert s.atlas_resolutions() == [ATLAS_RES]


def test_rejects_impossible_dimensions(tmp_path: Path):
    for kwargs in (
        {"n_optim_nodes": 99},          # more optim nodes than connectome nodes
        {"n_optim_nodes": 0},           # would make every dipole_node_label -1 (M == 0)
        {"n_cerebellum": 0},            # an empty block breaks the traceback partition
        {"n_volumetric": 1},            # fewer dipoles than the nodes that must own them
        {"eeg_channel_names": ("Fp1", "NotAnElectrode")},
    ):
        with pytest.raises(ValueError):
            build_fake_subject(tmp_path / f"bad{abs(hash(str(kwargs)))}", **kwargs)
