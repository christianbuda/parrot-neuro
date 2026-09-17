"""parrot_neuro.simulation -- the NMM simulation API (in progress).

Names only here, no heavy imports -- mirrors ``optimization/__init__.py``.
``params.py`` is numpy+stdlib; later modules (models/network/engine) pull in
jax and are never imported from this file. See
``~/.claude/plans/i-want-to-plan-enumerated-walrus.md`` for the full design.
"""
from __future__ import annotations

from ._modelspec import Bound, ModelSpec, get_spec
from .params import NMMParams

__all__ = ["NMMParams", "Bound", "ModelSpec", "get_spec"]
