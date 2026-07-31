"""Central configuration: all paths and run-level knobs live here.

Keep this module dependency-light (stdlib only). In particular it must NOT
import ``jax`` (or anything that imports jax), because ``apply_jax_env()``
sets environment variables that only take effect if they are exported
*before* jax is first imported. The workbench notebook therefore does, as its
very first cell::

    from parrot_neuro.optimization import config
    config.apply_jax_env()          # must run before any jax import
    # ... only now import parrot_neuro.optimization.forward / .model / .signal (jax)
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a hard parrot_neuro dependency at config-import time
    from parrot_neuro import Subject

# --- Subject / forward-model data -------------------------------------------
# Subject identity (a parrot_neuro.Subject) now lives on BoldFitConfig itself
# (see below) rather than as a module-level path — there is no sane "default"
# subject anymore, only whichever one the driver script points at.

# --- Forward-model selection ------------------------------------------------
ATLAS = 100                     # connectivity parcellation (100 or 1000)
SPACING = "2.0"                 # dipole spacing in mm; must be a string
LEADFIELD_LABEL = "duneuroCGAL"  # which solver/mesh leadfield to load

# --- Signal / spectral parameters -------------------------------------------
FS = 250                        # target EEG sampling rate (Hz) the sim matches
TIME_STEPS = 500                # samples per analysis chunk (== CHUNK_LENGTH)
CHUNK_LENGTH = 500              # EEG chunk length in samples
FMIN = 1.                      # PSD-loss lower band edge (Hz)
FMAX = 15.0                     # PSD-loss upper band edge (Hz)
GAMMA_FMIN = 15.0                # optional gamma-band log-PSD term lower edge (Hz)
GAMMA_FMAX = 40.0                # optional gamma-band log-PSD term upper edge (Hz)

# Conduction speed (m/s) turning tract lengths into delays: delays = L / SPEED.
CONDUCTION_SPEED = 3.0

# Simulated-BOLD bandpass, applied before any FC/FCD comparison -- matches the
# 0.01-0.1 Hz band typical resting-state fMRI preprocessing already bandpassed
# the *empirical* BOLD to (see connectivity.filter_sim_bold: only the simulated
# side needs filtering here).
BOLD_BANDPASS_LOW = 0.01        # Hz
BOLD_BANDPASS_HIGH = 0.1        # Hz
BOLD_BANDPASS_ORDER = 4         # Butterworth order

# --- JAX runtime environment ------------------------------------------------
# Single GPU index (PCI_BUS_ID order, cf. nvtop) for an UNMANAGED environment
# (e.g. a shared workstation) where nothing has already scoped
# CUDA_VISIBLE_DEVICES. Override via $PARROT_CUDA_DEVICE if "3" isn't free.
CUDA_DEVICE = os.environ.get("PARROT_CUDA_DEVICE", "3")
# Default assumes $HOME is a normal, roomy filesystem -- not true everywhere
# (e.g. LEONARDO's $HOME is small and quota'd), so this is overridable too.
JAX_CACHE_DIR = os.environ.get("PARROT_JAX_CACHE_DIR", os.path.expanduser("~/.cache/jax"))
JAX_ENABLE_X64 = True           # float64 — needed for stiff TVB dynamics


def apply_jax_env() -> None:
    """Export JAX/CUDA environment variables.

    Call this once, before importing jax anywhere. Idempotent.

    Uses ``setdefault`` for ``CUDA_VISIBLE_DEVICES``: under a scheduler (e.g.
    SLURM with ``--gres=gpu:N``), that variable is already set to the job's
    *allocated* device(s) before this runs — overwriting it with the
    workstation default would point at a device outside the job's cgroup
    (wrong GPU, or none at all).

    Also forces the ``platform`` (direct cudaMalloc/cudaFree, no arena) GPU
    allocator with upfront preallocation off. Verified empirically (2026-07-29):
    the BOLD simulator's one-off ~23GiB allocation (atlas=1000, t1_bold=320s)
    reliably OOMs under JAX's DEFAULT allocator (BFC: an arena that grows
    incrementally and can fail a single large request to internal
    fragmentation even with tens of GiB nominally free) — reproduced on an
    otherwise-idle, 93GiB-free GPU. Setting *only*
    ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` does NOT fix this (still BFC, just
    without the upfront grab); ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` is the
    part that actually matters, and needs both set together. Both use
    ``setdefault`` so an explicit override (e.g. a differently-tuned SLURM
    script) still wins.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", CUDA_DEVICE)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = JAX_CACHE_DIR
    os.environ["JAX_ENABLE_X64"] = str(JAX_ENABLE_X64)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


@dataclass(frozen=True)
class LearnableParam:
    """One model parameter to expose to the optimizer as a
    ``SigmoidBoundedParameter`` over ``[low, high]``.

    ``name`` must be a key of ``HeterogeneousModel.DEFAULT_PARAMS`` (when
    ``location="dynamics"``) or of the coupling's ``DEFAULT_PARAMS`` (when
    ``location="coupling"`` — the model has exactly one coupling, "delayed",
    so this always means ``coupling.delayed.<name>``; today that's just
    ``G``). Dynamics parameters are broadcast per-node (shape
    ``(num_nodes,)``); coupling parameters are global scalars (shape
    ``(1,)``).

    ``init`` defaults to the model's own ``DEFAULT_PARAMS`` value when left
    as ``None`` — usually the right choice, since those are already sensible
    physiological defaults.
    """

    name: str
    low: float
    high: float
    location: str = "dynamics"  # "dynamics" | "coupling"
    init: float | None = None


#: Default set of learnable parameters for the alternating EEG+BOLD fit: a
#: working selection spanning the Jansen-Rit cortical (A, B, a, b, mu),
#: Wilson-Cowan subcortical (P, c_ee), and global coupling (G) parameters.
#: This is a prototyping default, not a settled choice — which parameters are
#: worth optimizing is still being explored, so expect it to change. Add/remove
#: entries — or pass a whole new tuple — to change what the optimizer is allowed
#: to touch; ``network.py`` and ``train.py`` both build off this same list, so
#: they can never silently disagree about what's learnable.
DEFAULT_LEARNABLE_PARAMS: tuple[LearnableParam, ...] = (
    LearnableParam("P", 0.0, 2.0, "dynamics", init=0.0),
    LearnableParam("c_ee", 6.0, 20.0, "dynamics", init=12.0),
    LearnableParam("A", 2.0, 5.0, "dynamics", init=3.25),
    LearnableParam("B", 12.0, 35.0, "dynamics", init=22.0),
    LearnableParam("a", 0.04, 0.2, "dynamics", init=0.1),
    LearnableParam("b", 0.02, 0.1, "dynamics", init=0.05),
    LearnableParam("mu", 0.1, 0.4, "dynamics", init=0.22),
    LearnableParam("G", 0.0, 5.0, "coupling", init=0.1),
)

@dataclass
class BoldFitConfig:
    """One place to feed a subject/run into the alternating EEG+BOLD fit
    (``parrot_neuro.optimization.pipeline.run_bold_fit``). Field defaults (other than ``subject``,
    which has none — there's no sane default subject) mirror the plain
    module-level constants above so a bare ``BoldFitConfig(subject=...)``
    reproduces the same atlas/spacing those already point at — override
    whichever fields differ for your subject/experiment.
    """

    # --- subject / forward model ---
    subject: "Subject"
    atlas: int = ATLAS
    spacing: str = SPACING
    leadfield_label: str = LEADFIELD_LABEL
    fs: float = FS
    chunk_length: int = CHUNK_LENGTH
    fmin: float = FMIN
    fmax: float = FMAX
    conduction_speed: float = CONDUCTION_SPEED

    # --- optional EEG gamma-band term (see train.make_eeg_loss_fn) ---
    # 0 (default) = off, preserving the old main-band-only EEG loss. > 0 adds a
    # log(PSD) MSE term over [gamma_fmin, gamma_fmax) alongside the main
    # normalized-linear PSD MSE over [fmin, fmax) -- log-space because gamma
    # power is orders of magnitude smaller than the main band's and would be
    # swamped by a linear/normalized comparison.
    gamma_weight: float = 0.0
    gamma_fmin: float = GAMMA_FMIN
    gamma_fmax: float = GAMMA_FMAX

    # --- BOLD target + connectome region alignment ---
    # Empirical BOLD + the missing-region mask are both derived from the
    # subject's own fMRI derivatives (see optimization.connectivity) rather than
    # passed in as separate paths — this just selects which run to read.
    fmri_task: str = "rest"
    # Bandpass applied to *simulated* BOLD only, right before FC/FCD -- see
    # connectivity.filter_sim_bold. Empirical BOLD is already filtered upstream.
    bold_bandpass_low: float = BOLD_BANDPASS_LOW
    bold_bandpass_high: float = BOLD_BANDPASS_HIGH
    bold_bandpass_order: int = BOLD_BANDPASS_ORDER

    # --- empirical EEG ---
    eeg_task: str = "eyesclosed"  # which subject.load.eeg(...) recording to fit

    # --- output ---
    output_dir: Path = Path("eeg_bold_fit_res")

    # --- simulation horizons ---
    t0: float = 0.0
    dt: float = 1.0
    t1_eeg: float = 2_500.0     # ms; short horizon for the EEG PSD loss
    t1_bold: float = 900_000.0  # ms; long horizon for the BOLD FC loss (42 TRs at TR=1400ms)
    # One-time BOLD warm-up solve duration (ms), separate from t1_bold -- see
    # train.build_simulators' docstring. None (default) = old behaviour, warm
    # up for the full t1_bold (expensive/OOM-prone for a long t1_bold at a
    # large atlas). Set to something with margin over both your dynamics'
    # settling time and the HRF kernel duration (20s by default) -- e.g.
    # 30_000 -- to shrink the warm-up's memory/time without touching t1_bold
    # or the amount of BOLD signal available to the loss.
    t1_warmup: float | None = None
    eeg_settle_ms: float = 500.0  # discard as transient before computing the EEG loss
    eeg_stride_ms: float = 4.0    # subsample post-settle states at this period (1000/stride = eff. fs)
    tr_ms: float = 1400.0  # fixed by the dataset (LEMON), not tunable -- adjust the rest around it
    bold_downsample_ms: float = 4.0
    # 8 TRs (11.2s) burn-in -- lowered from 20 (28s, ~48% of the 42-TR budget at this tr_ms) to
    # leave enough TRs for the "dfc" loss to have more than a handful of windows. Assumes the
    # network settles within ~8 TRs at this dt/base_sigma -- worth confirming against
    # plot_node_activity/plot_bold_learning if you change dt or the noise level.
    bold_skip_trs: int = 8
    base_sigma: float = 0.048  # noise std on JR voltages / WC proportions
    noise_seed: int = 69

    # GPU-memory/wall-time trade for the integration scan's backward pass (see
    # network.build_network's docstring for the full accounting). None (default)
    # keeps every step's state live for the backward pass -- O(n_steps) memory,
    # dominated by the long BOLD horizon. An int K checkpoints the scan in
    # blocks of K steps (jax.checkpoint): O(n_steps/K + K) memory for ~1.3-1.7x
    # more compute; K ~ sqrt(n_steps) is the rule-of-thumb optimum (e.g. ~565
    # for the default t1_bold=900_000ms at dt=1.0ms). Exact gradient either way
    # -- this is a pure memory/time trade, not an approximation.
    solver_block_size: int | None = None

    # --- which parameters the optimizer is allowed to touch ---
    learnable_params: tuple[LearnableParam, ...] = DEFAULT_LEARNABLE_PARAMS

    # --- optimization ---
    learning_rate: float = 1e-2
    # None (default) = reuse learning_rate for the BOLD step too (old behaviour).
    # Now that EEG and BOLD each get their own Adam state (see train.run_alternating_fit),
    # they can also use different step sizes -- set this to tune the BOLD step
    # independently (e.g. if it's cheap/fast to plateau at the shared rate, or
    # too unstable at it given how much more expensive/noisy a BOLD step is).
    learning_rate_bold: float | None = None
    grad_clip_norm: float = 1.0
    num_epochs: int = 2
    bold_every: int = 1
    print_params_every: int = 10
    # Which loss(es) actually take gradient steps. "both" (default) is the
    # alternating fit; "eeg" or "bold" fits against only that target (the other
    # simulator/loss is still built, for diagnostics, but never gets a gradient
    # step and its loss history stays empty).
    optimize: str = "both"  # "eeg" | "bold" | "both"

    # --- early stopping (see train.is_loss_stalled) ---
    # None (default) = old behaviour, always run all num_epochs. Set an int to
    # stop once every actively-optimized loss's relative trend over the last
    # `early_stop_patience` overlapping `early_stop_window`-epoch windows has
    # stayed >= -early_stop_min_delta (flat or increasing, not still dropping).
    early_stop_window: int = 20
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 1e-3

    # --- BOLD loss: weighted combination of static FC + dynamic FC (FCD) ---
    # Both terms are always computed (from the SAME simulated trajectory, so
    # neither doubles the cost of the expensive BOLD forward pass -- see
    # train.make_bold_loss_fn) and combined as
    # bold_fc_weight * fc_loss + bold_dfc_weight * dfc_loss. Set either to 0 to
    # recover the old single-mode ("fc"-only or "dfc"-only) behaviour.
    bold_fc_weight: float = 0.5
    bold_dfc_weight: float = 0.5
    # Short window + dense overlap (step=1) is a necessity, not a choice: with only
    # t1_bold=900s of simulated BOLD (kept short for training cost) and a fixed
    # tr_ms=1400, there's no room for literature-standard 30-60s FCD windows -- these
    # values instead maximize how many (highly overlapping, not independent) window
    # snapshots survive within that budget. At bold_skip_trs=8 this gives
    # n_windows=(34-6)//1+1=29 -> 29*28/2=406 off-diagonal FCD values to compare.
    # Revisit if t1_bold changes.
    dfc_window_trs: int = 6    # sliding-window length (TRs) for the dFC term = 8.4s at tr_ms=1400
    dfc_step_trs: int = 1      # sliding-window stride (TRs)
    # k_min=1 keeps every off-diagonal FCD entry (including immediately-adjacent,
    # heavily-overlapping windows); raise it to drop near-diagonal entries that
    # are highly correlated by construction (step_trs << window_trs) rather than
    # by dynamics.
    dfc_kmin: int = 1
    # 25, not the more common 100 -- ~406 raw FCD values can't support 100 histogram
    # bins without mostly resolving noise (worth revisiting alongside dfc_window_trs).
    dfc_n_bins: int = 25
    dfc_sigma: float = 0.05    # Gaussian-kernel width (correlation units) for the soft histogram

    # --- optional BOLD spectral-shape term (see connectivity.bold_psd_band) ---
    # 0 (default) = off. > 0 adds a Welch-PSD MSE term, restricted+normalized to
    # [bold_bandpass_low, bold_bandpass_high], to the combined BOLD loss above --
    # fc_vector's time-averaged correlation has no sensitivity at all to each
    # signal's own temporal/spectral shape (only to which regions co-fluctuate).
    bold_psd_weight: float = 0.0
    # Welch-segment length/overlap, in TRs -- shared between simulated and
    # empirical so their PSDs land on the same frequency-bin grid despite very
    # different total recording lengths (same "necessity given the short
    # simulated horizon" rationale as dfc_window_trs/dfc_step_trs above).
    bold_psd_nperseg_trs: int = 32
    bold_psd_noverlap_trs: int = 16

    def __post_init__(self):
        if self.optimize not in ("eeg", "bold", "both"):
            raise ValueError(f"optimize must be 'eeg', 'bold', or 'both', got {self.optimize!r}")
        for name in ("bold_fc_weight", "bold_dfc_weight", "bold_psd_weight", "gamma_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")

        # The EEG loss compares the simulator's post-settle, strided output
        # directly against the empirical PSD's frequency bins — the two only
        # line up if they have the same number of samples. Changing t1_eeg,
        # eeg_settle_ms, eeg_stride_ms or chunk_length independently silently
        # breaks this and surfaces later as a cryptic jax shape-mismatch deep
        # inside a jitted loss function; check it up front instead.
        n_eeg_samples = (self.t1_eeg - self.eeg_settle_ms) / self.eeg_stride_ms
        if n_eeg_samples != self.chunk_length:
            raise ValueError(
                f"(t1_eeg - eeg_settle_ms) / eeg_stride_ms = {n_eeg_samples} must equal "
                f"chunk_length ({self.chunk_length}) — the simulated EEG segment and the "
                "empirical PSD it's compared against must have the same length. Adjust "
                "t1_eeg, eeg_settle_ms, eeg_stride_ms, or chunk_length so they agree."
            )

    def to_dict(self) -> dict:
        """JSON-serializable dict of this run's exact configuration.

        Built field-by-field (not a bare ``dataclasses.asdict(self)``) because
        two fields need special handling: ``subject`` is a ``parrot_neuro.Subject``,
        not a dataclass -- ``asdict`` would fall back to ``copy.deepcopy``-ing
        the whole object (including any populated internal load caches), which
        is wasteful and not guaranteed to succeed; reduced instead to its
        identifying ``(bids_root, subject id)`` pair. ``output_dir`` (a
        ``Path``) becomes a plain string. ``learnable_params`` is fine to
        ``asdict`` -- each ``LearnableParam`` holds only plain scalars.
        """
        d = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name == "subject":
                value = {"bids_root": str(value.bids_root), "subject": value.subject}
            elif f.name == "output_dir":
                value = str(value)
            elif f.name == "learnable_params":
                value = [dataclasses.asdict(lp) for lp in value]
            d[f.name] = value
        return d

    def save(self, out_dir: str | Path | None = None) -> Path:
        """Write this run's full configuration to ``<out_dir>/config.json``
        (default: ``self.output_dir``) -- so a results folder always says
        exactly what hyperparameters produced it, without cross-referencing a
        script version or commit hash. Call this right after constructing the
        config (before ``pipeline.fit``) so even a crashed/OOM'd run leaves
        behind a record of what was attempted.

        ``default=str`` in the ``json.dump`` call is a defensive fallback for
        any value ``to_dict`` didn't anticipate (e.g. a stray ``Path`` or
        numpy scalar) -- everything already-anticipated is a plain
        str/int/float/bool/list/dict by the time it gets there.
        """
        out_dir = Path(out_dir) if out_dir is not None else Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "config.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path
