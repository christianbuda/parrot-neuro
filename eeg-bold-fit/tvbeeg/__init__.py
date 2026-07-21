"""tvbeeg — subject-specific TVB neural-mass -> EEG(+BOLD) forward simulation & fitting.

Prototyping package extracted from the ``eeg.ipynb`` / ``eeg_bold_new.py``
notebooks. Import ``config`` and call ``config.apply_jax_env()`` before
importing any jax-backed submodule (everything except ``config`` itself).

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
    "extract",
    "connectivity",
    "network",
    "train",
    "viz",
    "pipeline",
]
