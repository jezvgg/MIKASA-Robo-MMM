"""A dirty snapshot must never be labelled with an unrelated Git commit."""

import subprocess

from utils.collection.provenance import committed_snapshot
from utils.collection.source_storage import file_sha256


def test_revision_binding_checks_contents_and_untracked_sources(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    p = tmp_path / "scene.py"
    p.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "scene.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    original = {"scene.py": file_sha256(p)}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    assert committed_snapshot(tmp_path, original) == head
    p.write_text("VALUE = 2\n")
    assert committed_snapshot(tmp_path, {"scene.py": file_sha256(p)}) is None
    # A past recording can truthfully resolve to its committed old contents.
    assert committed_snapshot(tmp_path, original) == head
    assert committed_snapshot(tmp_path, {"untracked.py": file_sha256(p)}) is None
