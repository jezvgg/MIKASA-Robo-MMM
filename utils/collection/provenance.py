"""Bind recorded file hashes to real Git revisions by checking committed blobs."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

from .source_storage import file_sha256


def complete_provenance(provenance, signature):
    """Validate all component bindings, including the full recorded manifest."""
    files = signature.get("source_files", {})
    if not files:
        return False
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    commits = provenance.get("component_commits", {})
    return (
        all(isinstance(commits.get(k), str) and re.fullmatch(r"[0-9a-f]{40}", commits[k])
            for k in ("robot", "environment", "planner", "converter"))
        and commits["robot"] == signature.get("profile", {}).get("robot_reference", {}).get("commit")
        and provenance.get("source_signature_sha256") == signature.get("code_sha256") == fingerprint
        and provenance.get("source_file_count") == len(files)
        and bool(provenance.get("exporter_source_files"))
    )


def dataset_provenance_complete(metadata):
    """A merge must bind its own code and every contributing conversion."""
    signature = metadata.get("source_run", {}).get("signature", {})
    if not complete_provenance(metadata.get("provenance", {}), signature):
        return False
    if "conversion_runs" not in metadata:
        return "aggregation" not in metadata
    runs = metadata["conversion_runs"]
    by_id = {run["id"]: run for run in runs}
    return (
        bool(runs) and len(by_id) == len(runs)
        and all(complete_provenance(run.get("provenance", {}), signature) for run in runs)
        and all(e.get("conversion_id") in by_id for e in metadata["episodes"])
    )


def committed_snapshot(repo, files):
    """Return HEAD only when every recorded byte hash matches that Git tree."""
    if not files or any("\n" in p or "\r" in p for p in files):
        return None
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    paths = sorted(files)
    result = subprocess.run(["git", "cat-file", "--batch"], cwd=repo,
                            input="".join(f"{head}:{p}\n" for p in paths).encode(),
                            capture_output=True, check=True)
    data, offset = result.stdout, 0
    for path in paths:
        end = data.index(b"\n", offset)
        header = data[offset:end].split()
        if header[-1] == b"missing" or len(header) != 3 or header[1] != b"blob":
            return None
        size = int(header[2])
        payload = data[end + 1:end + 1 + size]
        if hashlib.sha256(payload).hexdigest() != files[path]:
            return None
        offset = end + 1 + size + 1
    if offset != len(data):
        raise ValueError("Unexpected Git batch output")
    return head


def export_provenance(repo, signature):
    repo = Path(repo)
    files = signature["source_files"]
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if fingerprint != signature["code_sha256"]:
        raise ValueError("Recorded source fingerprint does not match its file manifest")
    source_commit = committed_snapshot(repo, files)
    exporter_files = {
        str(p.relative_to(repo)): file_sha256(p)
        for p in (repo / "utils/collection").glob("*.py")
        if not p.name.startswith("test_")
    }
    return dict(
        binding="Recorded source snapshot matched committed file contents at export time; not a claim about collection-time HEAD",
        source_signature_sha256=fingerprint,
        source_file_count=len(files),
        component_commits=dict(
            robot=signature["profile"]["robot_reference"]["commit"],
            environment=source_commit, planner=source_commit,
            converter=committed_snapshot(repo, exporter_files)),
        exporter_source_files=exporter_files,
    )
