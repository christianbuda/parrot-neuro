"""Jax-free description of parrot's neural-mass model(s).

Single source of truth *inside this package* for the numbers that live in
`HeterogeneousModel.DEFAULT_PARAMS` and `DEFAULT_LEARNABLE_PARAMS` on branch
`eeg-bold-fit` (a jax module `params.py` cannot import -- see STEP1-SPEC.md).
Step 4's `models.py` will build `HeterogeneousModel` FROM this spec instead of
restating the values; the drift guard then compares this spec against
`origin/eeg-bold-fit`'s model.

Values transcribed verbatim from `origin/eeg-bold-fit` (ref 0721b71):
    src/parrot_neuro/optimization/model.py::HeterogeneousModel.DEFAULT_PARAMS
    src/parrot_neuro/optimization/config.py::DEFAULT_LEARNABLE_PARAMS

`G` (coupling) is not in `DEFAULT_PARAMS` -- it lives on `DelayedLinearCoupling`,
which is out of scope for this transcription. Its only known default in our
reference material is `DEFAULT_LEARNABLE_PARAMS`'s own `init=0.1`, so
`coupling_defaults` uses that value (see STEP1-SPEC.md's "Coupling params" note).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class Bound:
    name: str
    low: float
    high: float
    location: str = "dynamics"  # "dynamics" | "coupling"
    init: float | None = None   # None -> the spec's own default for `name`


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: int
    default_params: Mapping[str, float]     # dynamics defaults, incl. structural masks
    coupling_defaults: Mapping[str, float]  # coupling param defaults (today: just G)
    learnable: tuple[Bound, ...]            # the DEFAULT learnable selection
    structural_params: tuple[str, ...]      # supplied by the network builder, not fit

    def resolved_init(self, b: Bound) -> float:
        """``b.init`` if set, else this spec's own default for ``b.name``."""
        if b.init is not None:
            return b.init
        table = self.coupling_defaults if b.location == "coupling" else self.default_params
        return table[b.name]

    def full_key_set(self) -> frozenset[str]:
        """Every parameter name this model defines, dynamics + coupling."""
        return frozenset(self.default_params) | frozenset(self.coupling_defaults)


_HETEROGENEOUS_JR_WC_DEFAULT_PARAMS: Mapping[str, float] = MappingProxyType({
    # --- heterogeneity masks (structural -- see ModelSpec.structural_params) ---
    "mask_cortical": 0.0,
    "mask_subcortical": 0.0,
    # --- network params ---
    "cortex_coupling_scale": 1,
    "tau_axon_JR": 15.0,
    "tau_axon_WC": 1.0,
    # --- Jansen-Rit params (ms scale) ---
    "A": 3.25,
    "B": 22.0,
    "a": 0.1,
    "b": 0.05,
    "v0": 5.52,
    "nu_max": 0.0025,
    "r": 0.56,
    "J": 135.0,
    "a_1": 1.0,
    "a_2": 0.8,
    "a_3": 0.25,
    "a_4": 0.25,
    "mu": 0.22,
    # --- Wilson-Cowan params ---
    "c_ee": 12.0, "c_ei": 4.0, "c_ie": 13.0, "c_ii": 11.0,
    "tau_e": 10.0, "tau_i": 10.0,
    "a_e": 1.2, "b_e": 2.8, "c_e": 1.0, "theta_e": 0.0,
    "a_i": 1.0, "b_i": 4.0, "c_i": 1.0, "theta_i": 0.0,
    "r_e": 1.0, "r_i": 1.0, "k_e": 1.0, "k_i": 1.0,
    "P": 0.0, "Q": 0.0,
    "alpha_e": 1.0, "alpha_i": 1.0,
    "shift_sigmoid": 1.0,
})

_HETEROGENEOUS_JR_WC_COUPLING_DEFAULTS: Mapping[str, float] = MappingProxyType({
    "G": 0.1,
})

_HETEROGENEOUS_JR_WC_LEARNABLE: tuple[Bound, ...] = (
    Bound("P", 0.0, 2.0, "dynamics", init=0.0),
    Bound("c_ee", 6.0, 20.0, "dynamics", init=12.0),
    Bound("A", 2.0, 5.0, "dynamics", init=3.25),
    Bound("B", 12.0, 35.0, "dynamics", init=22.0),
    Bound("a", 0.04, 0.2, "dynamics", init=0.1),
    Bound("b", 0.02, 0.1, "dynamics", init=0.05),
    Bound("mu", 0.1, 0.4, "dynamics", init=0.22),
    Bound("G", 0.0, 5.0, "coupling", init=0.1),
)

MODEL_SPECS: dict[str, ModelSpec] = {
    "heterogeneous_jr_wc": ModelSpec(
        name="heterogeneous_jr_wc",
        version=1,
        default_params=_HETEROGENEOUS_JR_WC_DEFAULT_PARAMS,
        coupling_defaults=_HETEROGENEOUS_JR_WC_COUPLING_DEFAULTS,
        learnable=_HETEROGENEOUS_JR_WC_LEARNABLE,
        structural_params=("mask_cortical", "mask_subcortical"),
    ),
}


def get_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; known models: {sorted(MODEL_SPECS)}") from None
