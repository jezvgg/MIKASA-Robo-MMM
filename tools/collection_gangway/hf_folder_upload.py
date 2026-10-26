#!/usr/bin/env python3
"""Upload a local dataset folder into a subfolder of a Hub dataset repo, straight from the server, in verified portions.

usage: hf_folder_upload.py --src ~/datasets/samedrawer-ag-1000ep --repo MIKASA-Robo-MMM/MIKASA-Robo-MMM --prefix samedrawer/

The token is read by huggingface_hub from HF_TOKEN_PATH (e.g. a file in /dev/shm the owner wrote); this script never
reads, prints or copies it. Skipped: .cache/ (huggingface_hub download metadata) and the source's own .gitattributes
(the target repo has its own). Each portion is one commit; after it, every file is checked on the Hub: size, and the
LFS sha256 or the git blob id for small files. Progress is kept in ~/sd-collect/org-upload/<repo>__<prefix>.json, so a
rerun continues. HF_HUB_DISABLE_XET=1 (the Xet upload hung on the first commit on 2026-10-18).
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402

SKIP_DIRS = {".cache"}
SKIP_FILES = {".gitattributes"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def hashes(path):
    """(sha256, git blob sha1) of a file."""
    sha, size = hashlib.sha256(), path.stat().st_size
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            sha.update(chunk); blob.update(chunk)
    return sha.hexdigest(), blob.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prefix", required=True, help="folder in the repo, e.g. samedrawer/; \"\" for the repo root")
    ap.add_argument("--portion-gb", type=float, default=1.0)
    args = ap.parse_args()
    src = Path(os.path.expanduser(args.src)).resolve()
    prefix = args.prefix.strip("/") + "/" if args.prefix.strip("/") else ""  # "" = the repo root
    state_dir = Path(os.path.expanduser("~/sd-collect/org-upload")); state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / f"{args.repo.replace('/', '__')}__{prefix.strip('/').replace('/', '__')}.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"done": {}}
    files = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if not p.is_file() or rel.parts[0] in SKIP_DIRS or str(rel) in SKIP_FILES:
            continue
        files.append((p, prefix + str(rel)))
    total = sum(p.stat().st_size for p, _ in files)
    log(f"{len(files)} files, {total / 1e9:.2f} GB from {src} -> {args.repo}/{prefix}; {len(state['done'])} already verified")
    api = HfApi()
    todo = [(p, r) for p, r in files if r not in state["done"]]
    cap = args.portion_gb * 1e9
    portions, cur, size = [], [], 0
    for p, r in todo:
        s = p.stat().st_size
        if cur and size + s > cap:
            portions.append(cur); cur, size = [], 0
        cur.append((p, r)); size += s
    if cur:
        portions.append(cur)
    sent = sum(v["size"] for v in state["done"].values())
    for n, portion in enumerate(portions, 1):
        meta = {}
        for p, r in portion:
            sha, blob = hashes(p)
            meta[r] = dict(size=p.stat().st_size, sha256=sha, blob=blob)
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in portion]
        for attempt in range(5):
            try:
                api.create_commit(args.repo, repo_type="dataset", operations=ops,
                                  commit_message=f"Add {prefix} portion {n}/{len(portions)} ({len(ops)} files)")
                break
            except Exception as e:
                log(f"commit retry {attempt + 1}: {type(e).__name__}: {str(e)[:200]}")
                time.sleep(30 * (attempt + 1))
        else:
            sys.exit(f"commit failed for portion {n}")
        info = {f.path: f for f in api.get_paths_info(args.repo, list(meta), repo_type="dataset", expand=True)}
        for r, m in meta.items():
            f = info.get(r)
            if f is None or f.size != m["size"]:
                sys.exit(f"hub check failed (missing/size): {r}")
            if f.lfs is not None:
                if f.lfs.sha256 != m["sha256"]:
                    sys.exit(f"hub sha256 mismatch: {r}")
            elif f.blob_id != m["blob"]:
                sys.exit(f"hub blob mismatch: {r}")
            state["done"][r] = dict(size=m["size"], sha256=m["sha256"])
        state_file.write_text(json.dumps(state))
        sent += sum(m["size"] for m in meta.values())
        log(f"portion {n}/{len(portions)} verified: {len(ops)} files; {sent / 1e9:.2f}/{total / 1e9:.2f} GB on the Hub")
    hub = {f.path: f.size for f in api.list_repo_tree(args.repo, path_in_repo=prefix.rstrip("/") or None, repo_type="dataset",
                                                     recursive=True) if getattr(f, "size", None) is not None}
    missing = [r for p, r in files if hub.get(r) != p.stat().st_size]
    log(f"FINAL: {len(files)} files under {prefix}, {len(missing)} missing or different on the Hub")
    if missing:
        print("\n".join(missing[:50])); sys.exit(1)


if __name__ == "__main__":
    main()
