"""parrot_neuro.optimization — subject-specific TVB neural-mass -> EEG(+BOLD)
forward simulation & parameter fitting.

Prototyping subpackage extracted from the ``eeg.ipynb`` / ``eeg_bold_new.py``
notebooks. Import ``config`` and call ``config.apply_jax_env()`` before
importing any jax-backed submodule (everything except ``config`` itself).

Deliberately NOT imported by the top-level ``parrot_neuro`` package: importing
``parrot_neuro`` (the ``Subject`` facade) must stay lightweight and jax-free.
The heavy stack (jax, tvboptim, optax, equinox) is only pulled in when you
explicitly ``from parrot_neuro.optimization import ...`` — install it with the
``parrot-neuro[optim]`` extra.

- EEG-only PSD fitting: ``config``, ``data``, ``forward``, ``model``, ``signal``.
- Alternating EEG(PSD) + BOLD(FC) fitting adds: ``connectivity`` (SC + empirical
  BOLD + missing-region bookkeeping), ``network`` (assembles the JR/WC model),
  ``train`` (loss functions + the alternating loop), ``viz`` (diagnostic
  plots), and ``pipeline`` (the ``build_context``/``fit`` entry points that
  compose all of the above — see ``pipeline``'s docstring for a full example).
"""
from __future__ import annotations

from . import config

__all__ = [
    "config",
    "forward",
    "model",
    "signal",
    "data",
    "connectivity",
    "network",
    "train",
    "viz",
    "pipeline",
]
