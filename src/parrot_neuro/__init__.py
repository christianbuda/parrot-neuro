"""parrot_neuro -- a small local-dev API over the Parrot pipeline outputs.

Currently this exposes :class:`Subject`, a facade that resolves and loads a
subject's reconstructed derivatives (surfaces, atlases, dipoles, leadfields,
tissue labels, DWI, artifacts, staged EEG/fMRI, ...).
"""
from .subject import Subject

__all__ = ["Subject"]
__version__ = "0.1.0"
