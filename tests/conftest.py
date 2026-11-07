"""Producer manifest helper available to synthetic Subject contract tests."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "containers/parrot_qc"))
from qc.manifests import write_manifest


@pytest.fixture
def register():
    def write(root, step, **kwargs):
        return write_manifest(root / "derivatives", "010002", step, "test/producer:1", **kwargs)
    return write
