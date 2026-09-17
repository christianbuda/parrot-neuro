"""Tests for parrot_neuro.simulation.params.NMMParams, against the synthetic
subject fixture (fake_subject: atlas 100, M=6 connectome nodes, K=4 optim nodes;
see tests/fixtures/build_fake_subject.py's docstring).

Written before params.py per the plan's "tests first" phasing (STEP1-SPEC.md).
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fixtures.build_fake_legacy_fit import build_fake_legacy_fit
from parrot_neuro.simulation import NMMParams, get_spec
from parrot_neuro.simulation import params as _params_mod

MODEL = "heterogeneous_jr_wc"
ATLAS = 100
M = 6  # fake_subject's connectome node count at atlas 100


@pytest.fixture(autouse=True)
def _no_template_dir(monkeypatch):
    """template_data/nmm_params/ genuinely doesn't exist in this repo, but don't
    let a stray PARROT_TEMPLATE_DATA_DIR in the environment change that."""
    monkeypatch.delenv("PARROT_TEMPLATE_DATA_DIR", raising=False)


# --- small duck-typed BoldFitConfig stand-in (see params.py::from_fit) ------
@dataclasses.dataclass
class _LP:
    name: str
    low: float
    high: float
    location: str = "dynamics"
    init: float | None = None


@dataclasses.dataclass
class _FakeCfg:
    subject: object
    atlas: int
    fmri_task: str
    learnable_params: tuple
    output_dir: str
    num_epochs: int = 20
    optimize: str = "both"

    def to_dict(self) -> dict:
        return {
            "subject": {"bids_root": str(self.subject.bids_root), "subject": self.subject.subject},
            "atlas": self.atlas,
            "fmri_task": self.fmri_task,
            "output_dir": str(self.output_dir),
            "learnable_params": [dataclasses.asdict(lp) for lp in self.learnable_params],
            "num_epochs": self.num_epochs,
            "optimize": self.optimize,
        }


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


def _assert_no_absolute_paths(obj):
    home = str(Path.home())
    for s in _walk_strings(obj):
        assert not s.startswith("/"), s
        assert "/srv/" not in s, s
        assert home not in s, s


# --- 1. save/load round-trip -------------------------------------------------
def test_save_load_round_trip(tmp_path):
    p = NMMParams.defaults(MODEL, ATLAS, M)
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    p = p.override(A=a, G=0.42)

    npz_path = p.save(tmp_path / "sub-000001_atlas-100_desc-nmmparams")
    assert npz_path.exists()
    assert npz_path.with_suffix(".json").exists()

    loaded = NMMParams.load(npz_path)
    assert loaded.model == p.model
    assert loaded.model_version == p.model_version
    assert loaded.atlas == p.atlas
    assert loaded.n_nodes == p.n_nodes
    assert loaded.node_space == p.node_space
    assert loaded.bounds == p.bounds
    assert loaded.locations == p.locations
    assert loaded.fixed_params == p.fixed_params
    assert loaded.structural_params == p.structural_params
    assert loaded.fit_config == p.fit_config
    assert loaded.provenance == p.provenance
    for name in p.values:
        np.testing.assert_array_equal(loaded.values[name], p.values[name])
        assert loaded.values[name].dtype == np.float64
    assert np.isnan(loaded.values["A"][2])


# --- 2. load() is portable ---------------------------------------------------
def test_load_is_portable(tmp_path, monkeypatch):
    p = NMMParams.defaults(MODEL, ATLAS, M)
    stem = tmp_path / "orig" / "params"
    p.save(stem)
    abs_stem = str(stem)

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    loaded = NMMParams.load(abs_stem)
    assert loaded.model == p.model

    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.move(f"{stem}.npz", moved / "params.npz")
    shutil.move(f"{stem}.json", moved / "params.json")
    loaded2 = NMMParams.load(moved / "params")
    assert loaded2.n_nodes == p.n_nodes
    for name in p.values:
        np.testing.assert_array_equal(loaded2.values[name], p.values[name])


# --- 3. sidecar completeness --------------------------------------------------
def test_sidecar_completeness():
    p = NMMParams.defaults(MODEL, ATLAS, M)
    spec = get_spec(MODEL)
    for b in spec.learnable:
        assert b.name in p.bounds
        assert p.bounds[b.name] == (b.low, b.high)
        assert p.locations[b.name] == b.location

    values_names = set(p.values)
    fixed_names = set(p.fixed_params)
    structural_names = set(p.structural_params)
    assert values_names | fixed_names | structural_names == spec.full_key_set()
    assert values_names.isdisjoint(fixed_names)
    assert values_names.isdisjoint(structural_names)
    assert fixed_names.isdisjoint(structural_names)


# --- 4/5. M<->K scatter ------------------------------------------------------
def test_to_optim_axis_identity_on_kept_nodes(fake_subject):
    p = NMMParams.defaults(MODEL, ATLAS, M)
    nodes = fake_subject.load.fmri_nodes(ATLAS, "rest")
    k_vals = np.array([10.0, 20.0, 30.0, 40.0])
    full = np.full(M, np.nan)
    full[nodes.to_conn] = k_vals
    p = p.override(A=full)

    got = p.to_optim_axis(fake_subject, task="rest")
    np.testing.assert_array_equal(got["A"], k_vals)
    assert got["G"].shape == (1,)
    np.testing.assert_array_equal(got["G"], p.values["G"])


def test_to_optim_axis_raises_on_nan_kept_node(fake_subject):
    p = NMMParams.defaults(MODEL, ATLAS, M)
    nodes = fake_subject.load.fmri_nodes(ATLAS, "rest")
    a = np.full(M, 3.25)
    a[nodes.to_conn[0]] = np.nan
    p = p.override(A=a)

    with pytest.raises(ValueError):
        p.to_optim_axis(fake_subject, task="rest")

    got = p.to_optim_axis(fake_subject, task="rest", allow_nan=True)
    assert np.isnan(got["A"][0])


# --- 6. fallback chain --------------------------------------------------------
def test_for_subject_defaults_only(fake_subject):
    p = NMMParams.for_subject(fake_subject, ATLAS)
    assert p.provenance["kind"] == "defaults"
    assert p.provenance["chain"] == ["defaults"]
    assert all(np.all(np.isfinite(v)) for v in p.values.values())


def test_for_subject_single_fit(fake_subject):
    fit_p = NMMParams.defaults(MODEL, ATLAS, M)
    fit_p.save(fake_subject.path.nmm_params("myfit", ATLAS))

    p = NMMParams.for_subject(fake_subject, ATLAS)
    assert p.provenance["kind"] == "subject_fit"
    assert p.provenance["chain"] == ["subject_fit"]
    for name in p.values:
        assert p.provenance["coverage"][name] == {"subject_fit": M if p.locations[name] == "dynamics" else 1}


def test_for_subject_partial_fit_is_mixed(fake_subject):
    # a value clearly different from the model's own default (3.25), so a
    # merged entry unambiguously tells you which source filled it
    FIT_VALUE = 9.99
    nodes = fake_subject.load.fmri_nodes(ATLAS, "rest")
    a = np.full(M, FIT_VALUE)
    nan_idx = [int(nodes.to_conn[0]), int(nodes.to_conn[1])]
    a[nan_idx] = np.nan
    fit_p = NMMParams.defaults(MODEL, ATLAS, M).override(A=a)
    fit_p.save(fake_subject.path.nmm_params("myfit", ATLAS))

    merged = NMMParams.for_subject(fake_subject, ATLAS)
    assert merged.provenance["kind"] == "mixed"
    assert merged.provenance["chain"] == ["subject_fit", "defaults"]
    assert merged.provenance["coverage"]["A"] == {"subject_fit": M - 2, "defaults": 2}
    # a fully-covered param never needed defaults -> no "defaults" entry for it
    assert merged.provenance["coverage"]["c_ee"] == {"subject_fit": M}
    assert np.all(np.isfinite(merged.values["A"]))
    spec = get_spec(MODEL)
    default_a = spec.resolved_init(next(b for b in spec.learnable if b.name == "A"))
    for i in range(M):
        if i in nan_idx:
            assert merged.values["A"][i] == pytest.approx(default_a)
        else:
            assert merged.values["A"][i] == pytest.approx(FIT_VALUE)


def test_for_subject_multiple_fits_requires_selection(fake_subject):
    p1 = NMMParams.defaults(MODEL, ATLAS, M).override(A=1.0)
    p1.save(fake_subject.path.nmm_params("fit1", ATLAS))
    p2 = NMMParams.defaults(MODEL, ATLAS, M).override(A=2.0)
    p2.save(fake_subject.path.nmm_params("fit2", ATLAS))

    with pytest.raises(ValueError) as exc:
        NMMParams.for_subject(fake_subject, ATLAS)
    assert "fit1" in str(exc.value) and "fit2" in str(exc.value)

    selected = NMMParams.for_subject(fake_subject, ATLAS, fit="fit2")
    assert np.all(selected.values["A"] == 2.0)


# --- 7. template no-op --------------------------------------------------------
def test_template_returns_none_when_absent():
    assert NMMParams.template(MODEL, ATLAS) is None


# --- 8. from_legacy ------------------------------------------------------------
def test_from_legacy(tmp_path, fake_subject, fake_bids_root):
    legacy_dir = build_fake_legacy_fit(
        tmp_path / "legacy", bids_root=fake_bids_root, subject_id="000001",
        atlas=ATLAS, n_optim_nodes=4, num_epochs=20,
    )
    p = NMMParams.from_legacy(legacy_dir / "optimized_params.npz", subject=fake_subject)

    assert p.n_nodes == M
    assert p.atlas == ATLAS
    assert p.values["A"].shape == (M,)
    assert p.values["G"].shape == (1,)
    assert p.locations["G"] == "coupling"
    assert p.locations["A"] == "dynamics"

    keep = fake_subject.load.fmri_nodes(ATLAS, "rest").keep
    np.testing.assert_array_equal(np.isfinite(p.values["A"]), keep)

    assert p.provenance["kind"] == "legacy_fit"
    assert p.provenance["epochs_run"] == 20
    assert p.provenance["stopped_early"] is False
    assert p.provenance["final_loss_eeg"] is not None
    assert p.provenance["final_loss_bold"] is not None
    assert p.provenance["eeg_loss_ratio"] is not None
    assert p.provenance["bold_loss_ratio"] is not None

    _assert_no_absolute_paths(p.fit_config)
    assert "bids_root" not in p.fit_config["subject"]
    assert p.fit_config["subject"]["dataset"] == Path(fake_bids_root).name


def test_from_legacy_stopped_early(tmp_path, fake_subject, fake_bids_root):
    legacy_dir = build_fake_legacy_fit(
        tmp_path / "legacy", bids_root=fake_bids_root, subject_id="000001",
        atlas=ATLAS, n_optim_nodes=4, num_epochs=20, n_logged_epochs=10,
    )
    p = NMMParams.from_legacy(legacy_dir / "optimized_params.npz", subject=fake_subject)
    assert p.provenance["epochs_run"] == 10
    assert p.provenance["stopped_early"] is True


@pytest.mark.skipif(
    not __import__("os").environ.get("PARROT_TEST_LEGACY_FIT"),
    reason="set PARROT_TEST_LEGACY_FIT to a real fit dir to run this",
)
def test_from_legacy_real_fit():
    import os

    d = Path(os.environ["PARROT_TEST_LEGACY_FIT"])
    npz = d / "optimized_params.npz"
    assert npz.exists(), f"{d} has no optimized_params.npz"
    p = NMMParams.from_legacy(npz)
    assert p.provenance["kind"] == "legacy_fit"
    assert p.n_nodes > 0


# --- 9. provenance allowlist ---------------------------------------------------
def test_provenance_allowlist_exact():
    expected = {
        "kind", "chain", "coverage", "subject", "dataset", "group", "optimize",
        "epochs_run", "stopped_early", "final_loss_eeg", "final_loss_bold",
        "eeg_loss_ratio", "bold_loss_ratio", "tvboptim", "parrot_neuro",
        "git_commit", "date",
    }
    assert _params_mod._PROVENANCE_ALLOWED_KEYS == expected


def test_save_raises_on_disallowed_provenance_key(tmp_path):
    p = NMMParams.defaults(MODEL, ATLAS, M)
    bad = dataclasses.replace(p, provenance={**p.provenance, "wandb_sweep": "some/sweep/path"})
    with pytest.raises(ValueError):
        bad.save(tmp_path / "bad")


# --- 10. no absolute paths anywhere in a saved sidecar -------------------------
def test_from_fit_sidecar_has_no_absolute_paths(tmp_path, fake_subject, fake_bids_root):
    learnable = (
        _LP("A", 2.0, 5.0, "dynamics", init=3.25),
        _LP("G", 0.0, 5.0, "coupling", init=0.1),
    )
    nested_output_dir = Path(fake_bids_root) / "nmmfit_out" / "000001_both"
    cfg = _FakeCfg(
        subject=fake_subject, atlas=ATLAS, fmri_task="rest",
        learnable_params=learnable, output_dir=str(nested_output_dir),
    )
    nodes = fake_subject.load.fmri_nodes(ATLAS, "rest")
    values_optim_axis = {
        "A": np.array([1.0, 2.0, 3.0, 4.0]),
        "G": np.array([0.2]),
    }
    assert len(values_optim_axis["A"]) == len(nodes.to_conn)

    p = NMMParams.from_fit(
        values_optim_axis, cfg, loss_history_eeg=[1e-4, 5e-5, 2e-5],
        loss_history_bold=[0.4, 0.3, 0.2],
    )
    assert p.provenance["kind"] == "subject_fit"
    assert p.provenance["epochs_run"] == 3
    assert p.provenance["stopped_early"] is True  # 3 < cfg.num_epochs (20)

    npz_path = p.save(tmp_path / "fit_out" / "params")
    sidecar = json.loads(npz_path.with_suffix(".json").read_text())
    _assert_no_absolute_paths(sidecar)
    assert sidecar["fit_config"]["subject"]["dataset"] == Path(fake_bids_root).name
    assert "bids_root" not in sidecar["fit_config"]["subject"]
    assert sidecar["fit_config"]["output_dir"] == "nmmfit_out/000001_both"


# --- 11. override --------------------------------------------------------------
def test_override(fake_subject):
    p = NMMParams.defaults(MODEL, ATLAS, M)

    scalar = p.override(A=7.0)
    assert np.all(scalar.values["A"] == 7.0)
    assert scalar.values["A"].shape == (M,)

    arr = p.override(A=np.arange(float(M)))
    np.testing.assert_array_equal(arr.values["A"], np.arange(float(M)))

    with pytest.raises(ValueError):
        p.override(A=np.arange(float(M - 1)))

    with pytest.raises(ValueError):
        p.override(not_a_param=1.0)

    # original untouched
    assert np.all(p.values["A"] == p.values["A"][0])
    assert p.values["A"][0] != 7.0


# --- 12. defaults() has no NaN, full coverage -----------------------------------
def test_defaults_no_nan_full_coverage():
    p = NMMParams.defaults(MODEL, ATLAS, M)
    for v in p.values.values():
        assert np.all(np.isfinite(v))
    for frac in p.coverage.values():
        assert frac == 1.0


# --- 13. describe() --------------------------------------------------------------
def test_describe_mentions_kind_and_coverage():
    p = NMMParams.defaults(MODEL, ATLAS, M)
    s = p.describe()
    assert "defaults" in s
    assert "coverage=" in s


# --- Definition-of-done: import hygiene -----------------------------------------
def test_parrot_neuro_does_not_import_simulation():
    out = subprocess.run(
        [sys.executable, "-c",
         "import parrot_neuro, sys; assert 'parrot_neuro.simulation' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr


def test_params_module_is_jax_and_tvboptim_free():
    out = subprocess.run(
        [sys.executable, "-c",
         "import parrot_neuro.simulation.params, sys; "
         "assert 'jax' not in sys.modules; assert 'tvboptim' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
