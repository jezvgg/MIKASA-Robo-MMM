"""Resolve retained numerical recordings without changing recorded commands."""

import hashlib
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numerical_source(episode):
    path = Path(episode.get("numeric_source_h5", episode["source_h5"]))
    if "numeric_source_h5" in episode:
        if file_sha256(path) != episode["numeric_source_sha256"]:
            raise ValueError(f"Retained numerical recording changed: {path}")
        if file_sha256(path.with_suffix(".json")) != episode["numeric_metadata_sha256"]:
            raise ValueError(f"Retained recording metadata changed: {path}")
    return path
