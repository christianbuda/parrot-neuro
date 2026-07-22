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
FMAX = 40.0                     # PSD-loss upper band edge (Hz)

# Conduction speed (m/s) turning tract lengths into delays: delays = L / SPEED.
CONDUCTION_SPEED = 3.0

# --- JAX runtime environment ------------------------------------------------
CUDA_DEVICE = "1"               # single GPU index (PCI_BUS_ID order, cf. nvtop)
JAX_CACHE_DIR = os.path.expanduser("~/.cache/jax")  # per-user, always writable
JAX_ENABLE_X64 = True           # float64 — needed for stiff TVB dynamics


def apply_jax_env() -> None:
    """Export JAX/CUDA environment variables.

    Call this once, before importing jax anywhere. Idempotent.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    os.environ["JAX_COMPILATION_CACHE_DIR"] = JAX_CACHE_DIR
    os.environ["JAX_ENABLE_X64"] = str(JAX_ENABLE_X64)


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

    # --- BOLD target + connectome region alignment ---
    # Empirical BOLD + the missing-region mask are both derived from the
    # subject's own fMRI derivatives (see optimization.connectivity) rather than
    # passed in as separate paths — this just selects which run to read.
    fmri_task: str = "rest"

    # --- empirical EEG ---
    eeg_task: str = "eyesclosed"  # which subject.load.eeg(...) recording to fit

    # --- output ---
    output_dir: Path = Path("eeg_bold_fit_res")

    # --- simulation horizons ---
    t0: float = 0.0
    dt: float = 1.0
    t1_eeg: float = 2_500.0     # ms; short horizon for the EEG PSD loss
    t1_bold: float = 60_000.0  # ms; long horizon for the BOLD FC loss (>=83 TRs at TR=720ms)
    eeg_settle_ms: float = 500.0  # discard as transient before computing the EEG loss
    eeg_stride_ms: float = 4.0    # subsample post-settle states at this period (1000/stride = eff. fs)
    tr_ms: float = 720.0
    bold_downsample_ms: float = 4.0
    bold_skip_trs: int = 20
    base_sigma: float = 0.048  # noise std on JR voltages / WC proportions
    noise_seed: int = 69

    # --- which parameters the optimizer is allowed to touch ---
    learnable_params: tuple[LearnableParam, ...] = DEFAULT_LEARNABLE_PARAMS

    # --- optimization ---
    learning_rate: float = 1e-3
    grad_clip_norm: float = 1.0
    num_epochs: int = 200
    bold_every: int = 1
    print_params_every: int = 10

    def __post_init__(self):
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
