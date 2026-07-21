"""Smoke tests for parrot_neuro.Subject against a real derivatives tree.

These are integration-style: they need a reconstructed subject on disk. Point the
tests at one via the PARROT_TEST_BIDS / PARROT_TEST_SUBJECT env vars, else they
fall back to the local LEMON reference and skip cleanly if it is absent.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

from parrot_neuro import Subject

BIDS = Path(os.environ.get("PARROT_TEST_BIDS", "/srv/nfs-data/sisko/christian/parrot_LEMON"))
SUBJECT = os.environ.get("PARROT_TEST_SUBJECT", "010002")

pytestmark = pytest.mark.skipif(
    not (BIDS / "derivatives").is_dir(),
    reason=f"no reference derivatives at {BIDS}/derivatives (set PARROT_TEST_BIDS)",
)


@pytest.fixture
def s() -> Subject:
    return Subject(BIDS, SUBJECT)


def test_init_and_metadata(s: Subject):
    assert s.subj == f"sub-{s.subject}"
    assert s.deriv == BIDS / "derivatives"
    assert s.surface_backend in {"fastsurfer", "freesurfer"}
    # accepts a sub- prefixed id too
    assert Subject(BIDS, f"sub-{SUBJECT}").subject == SUBJECT


def test_wrong_root_raises():
    # passing the derivatives dir instead of the dataset root should be caught
    with pytest.raises(FileNotFoundError):
        Subject(BIDS / "derivatives", SUBJECT)


def test_path_layer_is_cheap_and_correct(s: Subject):
    assert s.path.t1().exists()
    assert s.path.cortex("lh").name == "freesurfer_lh_middle.ply"
    assert s.path.leadfield("openmeeg-4.0mm").name == "processed_openmeeg-4.0mm-leadfield.npy"


def test_discovery(s: Subject):
    lfs = s.available_leadfields()
    assert lfs and all("-leadfield" not in k for k in lfs)
    assert s.atlas_resolutions()[0] == 100
    assert 2.0 in s.dipole_spacings()


def test_optional_stage_flags_are_bool(s: Subject):
    for flag in (s.has_dwi, s.has_anisotropy, s.has_artifacts, s.has_eeg, s.has_fmri,
                 s.has_optim_nodes):
        assert isinstance(flag, bool)


def test_fmri_aligned_loaders(s: Subject):
    """fMRI-aligned structural loaders: masked matrices are square and equal the full matrix
    sliced by keep; labels/dipole indices align to the same K nodes."""
    if not s.has_optim_nodes:
        pytest.skip("no desc-optim_nodes for this subject")
    import numpy as np

    n = 1000
    nodes = s.load.fmri_nodes(n)
    keep = nodes.keep
    k = int(keep.sum())
    assert len(nodes) == k and nodes.to_conn.shape == (k,)
    W = s.load.weights(n, fmri_aligned=True)
    # masked == full sliced by keep (mask-first invariant)
    assert np.array_equal(W, s.load.weights(n)[np.ix_(keep, keep)])
    assert W.shape == (k, k)
    assert s.load.distances(n, fmri_aligned=True).shape == (k, k)
    assert s.load.weights(n, normalized=True, fmri_aligned=True).shape == (k, k)
    assert len(s.load.connectivity_labels(n, fmri_aligned=True)) == k
    # the fMRI is its own reference: its non-NaN rows are exactly the keep mask
    ts = np.asarray(s.load.fmri_timeseries("conn")[f"ts_{n}"])
    assert np.array_equal(~np.isnan(ts).any(axis=1), keep)
    # dipole -> node index: valid indices land in [0, k); dropped-node dipoles are -1
    sp = s.dipole_spacings()[0]
    dn = s.load.dipole_node_labels(n, sp)
    assert dn.shape == s.load.dipole_labels(n, sp).shape
    valid = dn >= 0
    assert dn[valid].min() >= 0 and dn[valid].max() < k


def test_loaders_return_expected_types(s: Subject):
    assert s.load.t1().affine.shape == (4, 4)
    lf = s.load.leadfield(s.available_leadfields()[0])
    assert lf.ndim == 2
    assert s.load.cortex("lh").vertices.shape[1] == 3
    assert isinstance(s.load.electrodes_selected(), (list, dict))
    el = s.load.electrodes()
    assert isinstance(el, dict) and next(iter(el.values())).shape == (3,)


def test_cache_returns_same_object():
    s = Subject(BIDS, SUBJECT, cache=True)
    key = s.available_leadfields()[0]
    assert s.load.leadfield(key) is s.load.leadfield(key)
    # without caching, a fresh read each time
    s2 = Subject(BIDS, SUBJECT)
    assert s2.load.leadfield(key) is not s2.load.leadfield(key)


def test_missing_optional_stage_raises_clearly(s: Subject):
    # a file that does not exist must raise (not return silently)
    with pytest.raises((FileNotFoundError, OSError)):
        s.load.npy("anisotropy", "does_not_exist.npy")


def test_qc_warning_fires_on_degraded_stage(s: Subject):
    # inject a degraded status and confirm a loader warns once
    s._qc_status_map["leadfields"] = "fail"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s.load.leadfield(s.available_leadfields()[0])
    assert any("leadfields" in str(x.message) for x in w)


def test_qc_warning_silenced_when_disabled():
    s = Subject(BIDS, SUBJECT, warn_on_qc=False)
    s._qc_status_map["leadfields"] = "fail"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s.load.leadfield(s.available_leadfields()[0])
    assert not any("leadfields" in str(x.message) for x in w)
