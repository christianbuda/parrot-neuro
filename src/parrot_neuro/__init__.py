"""parrot_neuro -- a small local-dev API over the Parrot pipeline outputs.

Currently this exposes :class:`Subject`, a facade that resolves and loads a
subject's reconstructed derivatives (surfaces, atlases, dipoles, leadfields,
tissue labels, DWI, artifacts, staged EEG/fMRI, ...).
"""
from .subject import Subject

# NOTE: do NOT import the `optimization` subpackage here. `import parrot_neuro`
# must stay lightweight and jax-free (core deps: numpy/nibabel/trimesh only).
# The TVB/EEG simulation stack lives under `parrot_neuro.optimization` and pulls
# in jax/tvboptim/optax/equinox — import it explicitly where you need it.
__all__ = ["Subject"]
__version__ = "0.1.0"
