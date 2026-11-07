"""Exercise image selection and SIF cache reuse without Docker or a cluster."""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def manifest(expression):
    return subprocess.check_output(
        ["bash", "-c", f'source bin/images.sh; {expression}'],
        cwd=REPO, text=True,
    ).strip()


def executable(path, body):
    path.write_text("#!/bin/bash\nset -eu\n" + body)
    path.chmod(0o755)


@pytest.fixture
def tools(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("CALL_LOG", str(tmp_path / "calls"))
    monkeypatch.setenv("OUT", str(tmp_path / "sifs"))
    monkeypatch.setenv("DEST", "")
    monkeypatch.setenv("FORCE", "0")
    executable(bindir / "sudo", 'exec "$@"\n')
    executable(bindir / "apptainer", '''
printf '%s\n' "$*" >> "$CALL_LOG"
touch "$3"
''')
    return bindir


def test_local_sif_build_uses_pin_and_does_not_reuse_latest(tmp_path, tools):
    ref = manifest('echo "$IMG_QSIRECON"')
    name = manifest('image_cache_name "$IMG_QSIRECON"')
    out = tmp_path / "sifs"
    out.mkdir()
    (out / "qsirecon_latest.sif").write_text("old image")
    executable(tools / "docker", '''
printf '%s\n' "$*" >> "$CALL_LOG"
if [[ "$*" == *--format* ]]; then echo sha256:localimageid; fi
''')
    cmd = ["bash", "hpc/leonardo/build_sif_local.sh", "qsirecon"]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    calls = (tmp_path / "calls").read_text()
    assert f"image inspect {ref}" in calls
    assert "docker-daemon://sha256:localimageid" in calls
    assert (out / f"{name}.sif").exists()
    assert (out / "qsirecon_latest.sif").read_text() == "old image"
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    assert (tmp_path / "calls").read_text() == calls


def test_local_sif_build_remote_fallback_uses_digest(tmp_path, tools):
    executable(tools / "docker", 'exit 1\n')
    subprocess.run(
        ["bash", "hpc/leonardo/build_sif_local.sh", "hippunfold"],
        cwd=REPO, check=True, capture_output=True,
    )
    ref = manifest('echo "$IMG_HIPPUNFOLD"')
    assert f"docker://{ref}" in (tmp_path / "calls").read_text()


def test_parrot_builds_publish_pinned_version_latest_and_extra_tag(tmp_path, tools):
    executable(tools / "docker", 'printf "%s\\n" "$*" >> "$CALL_LOG"\n')
    subprocess.run(
        ["bash", "bin/build.sh", "--push", "--version", "test", "qc"],
        cwd=REPO, check=True, capture_output=True,
    )
    calls = (tmp_path / "calls").read_text()
    pinned = manifest('echo "$PARROT_VERSION"')
    assert (f"-t christianbuda/parrot_qc:{pinned} -t christianbuda/parrot_qc:latest "
            "-t christianbuda/parrot_qc:test") in calls
    assert f"push christianbuda/parrot_qc:{pinned}" in calls
    assert "push christianbuda/parrot_qc:latest" in calls
    assert "push christianbuda/parrot_qc:test" in calls


def test_prepull_uses_shared_pins_and_preserves_cache_on_failure(tmp_path, tools):
    executable(tools / "skopeo", 'echo sha256:registrytag\n')
    cache = tmp_path / "cache"
    cmd = ["bash", "hpc/leonardo/prepull_sifs.sh", str(cache)]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    refs = manifest('printf "%s\\n" "${ALL_IMAGES[@]}"').splitlines()
    calls = (tmp_path / "calls").read_text()
    for ref in refs:
        assert f"docker://{ref}" in calls
    assert len(list(cache.glob("*.sif"))) == len(refs)
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    assert (tmp_path / "calls").read_text() == calls

    # A failed forced refresh must not destroy the previously usable cache.
    executable(tools / "apptainer", 'touch "$3"; exit 1\n')
    result = subprocess.run(
        cmd, cwd=REPO, env={**os.environ, "FORCE": "1"}, capture_output=True,
    )
    assert result.returncode == 1
    assert len(list(cache.glob("*.sif"))) == len(refs)
    assert not list(cache.glob("*.part"))
