"""Synthetic test data builders.

Currently one: :func:`build_fake_subject.build_fake_subject`, which writes a
miniature-but-structurally-exact Parrot derivatives tree. Read that module's
docstring before writing a test against it -- it is the spec for what the fake
tree guarantees.
"""
from __future__ import annotations

from .build_fake_subject import build_fake_subject

__all__ = ["build_fake_subject"]
