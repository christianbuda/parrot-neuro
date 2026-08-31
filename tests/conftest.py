"""Shared pytest fixtures.

The synthetic subject tree is rebuilt **per test**, deliberately. Later phases
write *into* it -- the projector cache lands in
``derivatives/simulations/sub-<id>/projectors/`` and simulation runs get their own
folders -- and the plan's test inventory includes "cache invalidation: touching a
source file rebuilds; otherwise reuses" and "same config reuses the folder". A
session-scoped tree would let those tests leak into each other and become
order-dependent. Building costs ~30 ms against a suite that takes seconds, so
isolation is the better trade.

``fake_bids_root_readonly`` is the session-scoped escape hatch for tests that only
*read*; use it only when a test provably writes nothing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fixtures.build_fake_subject import SUBJECT_ID, build_fake_subject


@pytest.fixture
def fake_bids_root(tmp_path: Path) -> Path:
    """A fresh synthetic BIDS root (the folder containing ``derivatives/``).

    Writable and private to the calling test.
    """
    return build_fake_subject(tmp_path / "fake_bids", SUBJECT_ID, seed=0)


@pytest.fixture
def fake_subject(fake_bids_root: Path):
    """A :class:`parrot_neuro.Subject` over a fresh synthetic tree."""
    from parrot_neuro import Subject

    return Subject(fake_bids_root, SUBJECT_ID)


@pytest.fixture(scope="session")
def fake_bids_root_readonly(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped synthetic tree, shared by every test that asks for it.

    Only for tests that write nothing -- anything that mutates the tree (or that
    triggers code which caches into it) must use :func:`fake_bids_root` instead.
    """
    return build_fake_subject(tmp_path_factory.mktemp("fake_bids_ro"), SUBJECT_ID, seed=0)
