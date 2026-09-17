"""Build a miniature ``optimized_params.npz`` + sibling ``config.json`` pair --
the pre-``NMMParams`` fit output format, so ``NMMParams.from_legacy`` has
something to migrate.

This is a **transcription**, not a sample: no real ``optimized_params.npz`` exists
anywhere on this machine (searched). The key set and field names are copied from
the two writers, at the commit this step was built against
(``origin/eeg-bold-fit`` ref ``0721b71``):

    npz keys / bookkeeping:
        examples/eeg_bold_fit_cli.py:186-191
            optimized = train.extract_learnable_values(result.diff_params, cfg.learnable_params)
            np.savez(out_dir / "optimized_params.npz", **optimized,
                      loss_eeg=..., loss_bold=..., eeg_loss_ratio=..., bold_loss_ratio=...,
                      combined_loss_ratio=...)
        i.e. the npz mixes one array per learnable param (K-axis; (1,) for a
        coupling param like G) with 5 bookkeeping entries.

    config.json field set:
        src/parrot_neuro/optimization/config.py:143 (class BoldFitConfig),
        :328 (to_dict), :352 (save) -- every ``BoldFitConfig`` dataclass field,
        dumped field-by-field (``learnable_params`` as a list of
        ``{name, low, high, location, init}`` dicts -- see ``config.py``'s
        ``LearnableParam``).

    Learnable-param defaults (name/low/high/location/init):
        src/parrot_neuro/optimization/config.py's DEFAULT_LEARNABLE_PARAMS
        (P, c_ee, A, B, a, b, mu on "dynamics"; G on "coupling").

If the real format ever changes, or a real ``optimized_params.npz`` turns up,
prefer it -- this fixture is a best-effort stand-in, not a spec.

``output_dir`` in the generated config.json is deliberately an absolute path
OUTSIDE ``bids_root`` (the real CLI's default, ``eeg_bold_fit_res/...``, is a
plain relative path from wherever the script ran -- never nested under the
BIDS dataset) -- this exercises ``NMMParams.from_legacy``'s "not under
bids_root -> basename" sanitize branch, not the "relative to bids_root" one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_LEARNABLE_PARAMS: tuple[dict, ...] = (
    {"name": "P", "low": 0.0, "high": 2.0, "location": "dynamics", "init": 0.0},
    {"name": "c_ee", "low": 6.0, "high": 20.0, "location": "dynamics", "init": 12.0},
    {"name": "A", "low": 2.0, "high": 5.0, "location": "dynamics", "init": 3.25},
    {"name": "B", "low": 12.0, "high": 35.0, "location": "dynamics", "init": 22.0},
    {"name": "a", "low": 0.04, "high": 0.2, "location": "dynamics", "init": 0.1},
    {"name": "b", "low": 0.02, "high": 0.1, "location": "dynamics", "init": 0.05},
    {"name": "mu", "low": 0.1, "high": 0.4, "location": "dynamics", "init": 0.22},
    {"name": "G", "low": 0.0, "high": 5.0, "location": "coupling", "init": 0.1},
)

_LEGACY_BOOKKEEPING_KEYS = ("loss_eeg", "loss_bold", "eeg_loss_ratio", "bold_loss_ratio",
                            "combined_loss_ratio")


def build_fake_legacy_fit(
    out_dir: str | Path,
    *,
    bids_root: str | Path,
    subject_id: str = "000001",
    atlas: int = 100,
    n_optim_nodes: int = 4,
    fmri_task: str = "rest",
    num_epochs: int = 20,
    n_logged_epochs: int | None = None,
    seed: int = 0,
) -> Path:
    """Write ``optimized_params.npz`` + ``config.json`` into ``out_dir``.

    ``n_optim_nodes`` should match the target subject's real K at ``atlas`` (the
    synthetic fixture's K=4 at atlas=100) so ``from_legacy``'s K->M scatter has a
    real ``optim_to_conn`` map to use.

    ``n_logged_epochs`` (default: ``num_epochs``, i.e. no early stop) lets a
    caller simulate early stopping: pass something smaller than ``num_epochs`` so
    the loss histories are shorter than the configured budget.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_logged = num_epochs if n_logged_epochs is None else n_logged_epochs
    subject = str(subject_id).removeprefix("sub-")

    npz_arrays = {}
    for lp in DEFAULT_LEARNABLE_PARAMS:
        shape = (1,) if lp["location"] == "coupling" else (n_optim_nodes,)
        npz_arrays[lp["name"]] = rng.uniform(lp["low"], lp["high"], size=shape)

    loss_eeg = np.linspace(1e-5, 3e-6, n_logged)
    loss_bold = np.linspace(0.5, 0.12, n_logged)
    eeg_ratio = float(loss_eeg[-1] / loss_eeg[0]) if n_logged else float("nan")
    bold_ratio = float(loss_bold[-1] / loss_bold[0]) if n_logged else float("nan")
    combined_ratio = eeg_ratio + bold_ratio

    np.savez(
        out_dir / "optimized_params.npz",
        **npz_arrays,
        loss_eeg=loss_eeg, loss_bold=loss_bold,
        eeg_loss_ratio=eeg_ratio, bold_loss_ratio=bold_ratio,
        combined_loss_ratio=combined_ratio,
    )

    # Realistic legacy layout: NOT under bids_root (see module docstring).
    output_dir = f"/tmp/eeg_bold_fit_res/atlas-{atlas}/{subject}_both"

    cfg = {
        "subject": {"bids_root": str(bids_root), "subject": subject},
        "atlas": atlas, "spacing": "2.0", "leadfield_label": "duneuroCGAL",
        "fs": 250.0, "chunk_length": 500, "fmin": 1.0, "fmax": 15.0,
        "conduction_speed": 3.0,
        "gamma_weight": 0.0, "gamma_fmin": 15.0, "gamma_fmax": 40.0,
        "fmri_task": fmri_task,
        "bold_bandpass_low": 0.01, "bold_bandpass_high": 0.1, "bold_bandpass_order": 4,
        "eeg_task": "eyesclosed",
        "output_dir": output_dir,
        "t0": 0.0, "dt": 1.0, "t1_eeg": 2500.0, "t1_bold": 700_000.0, "t1_warmup": 30_000.0,
        "eeg_settle_ms": 500.0, "eeg_stride_ms": 4.0, "tr_ms": 1400.0,
        "bold_downsample_ms": 4.0, "bold_skip_trs": 8, "base_sigma": 0.048, "noise_seed": 69,
        "solver_block_size": 1400,
        "learnable_params": list(DEFAULT_LEARNABLE_PARAMS),
        "learning_rate": 0.01, "learning_rate_bold": None, "grad_clip_norm": 1.0,
        "num_epochs": num_epochs, "bold_every": 2, "print_params_every": 10,
        "optimize": "both",
        "early_stop_window": 20, "early_stop_patience": None, "early_stop_min_delta": 0.001,
        "bold_fc_weight": 0.5, "bold_dfc_weight": 0.5,
        "dfc_window_trs": 6, "dfc_step_trs": 1, "dfc_kmin": 1, "dfc_n_bins": 25, "dfc_sigma": 0.05,
        "bold_psd_weight": 0.0, "bold_psd_nperseg_trs": 32, "bold_psd_noverlap_trs": 16,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return out_dir
