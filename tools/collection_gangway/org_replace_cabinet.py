"""Owner 2026-10-26 «добавь в организацию как зальешь вместо прошлой версии»: replace the org repo's cabinetsearch/
(CabinetSearch AG profile 16, four compartments) with the five-compartment cabinetsearch5-ag-1000ep.

Run after hf_folder_upload.py has put every new file under cabinetsearch/ (overwriting the same paths, verified by
sha256). This script makes ONE commit: it deletes the old folder's files that the new dataset does not have, and
replaces the README table row with README_ROW. Then it checks that cabinetsearch/ on the Hub is exactly the new
dataset (paths and sizes). The token comes from HF_TOKEN_PATH; this script never reads it.

    python org_replace_cabinet.py SRC_DIR README_ROW_FILE
"""
import os, sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

REPO, PREFIX = "MIKASA-Robo-MMM/MIKASA-Robo-MMM", "cabinetsearch/"
src, row = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).read_text().strip()
api = HfApi()
new = {PREFIX + str(p.relative_to(src)): p.stat().st_size for p in src.rglob("*")
       if p.is_file() and p.relative_to(src).parts[0] != ".cache" and p.name != ".gitattributes"}
hub = {f.path: f.size for f in api.list_repo_tree(REPO, path_in_repo=PREFIX.rstrip("/"), repo_type="dataset",
                                                 recursive=True) if getattr(f, "size", None) is not None}
wrong = [r for r, s in new.items() if hub.get(r) != s]
if wrong:
    sys.exit(f"{len(wrong)} new files missing or different on the Hub, e.g. {wrong[:3]}: run hf_folder_upload.py first")
stale = sorted(set(hub) - set(new))
readme = Path(hf_hub_download(REPO, "README.md", repo_type="dataset", cache_dir="/dev/shm/hf-org-readme")).read_text()
lines = readme.split("\n")
old = [i for i, l in enumerate(lines) if l.startswith("| `cabinetsearch/` |")]
assert len(old) == 1, old
lines[old[0]] = row
ops = [CommitOperationDelete(path_in_repo=r) for r in stale]
ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="\n".join(lines).encode()))
c = api.create_commit(REPO, repo_type="dataset", operations=ops,
                      commit_message=f"cabinetsearch/: five compartments (CabinetSearch5 AG profile 7) replaces the "
                                     f"four-compartment version; {len(stale)} old files removed, README row updated")
print("commit", c.oid, "| removed", len(stale), "old files")
hub = {f.path: f.size for f in api.list_repo_tree(REPO, path_in_repo=PREFIX.rstrip("/"), repo_type="dataset",
                                                 recursive=True) if getattr(f, "size", None) is not None}
print("FINAL:", "cabinetsearch/ == new dataset" if hub == new else f"MISMATCH {len(set(hub) ^ set(new))} paths",
      f"({len(new)} files, {sum(new.values()) / 1e9:.2f} GB)")
