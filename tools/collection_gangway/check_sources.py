import glob, hashlib, json, os
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
from huggingface_hub import HfApi
R = Path(os.path.expanduser("~/sd-collect/runs-ag"))
api = HfApi()
for repo, pat in (("nurtayev-d/cabinetsearch-ag-1000ep", "cabinet_search-ag-train-*"), ("nurtayev-d/depthrecall-ag-1000ep", "depth_recall-ag6-train-*")):
    batches = sorted(p.name for p in R.glob(pat) if p.is_dir())
    hub = {}
    for f in api.list_repo_tree(repo, path_in_repo="sources", repo_type="dataset", recursive=True, expand=True):
        if getattr(f, "size", None) is not None:
            hub[f.path] = f
    missing, wrong, h5n, h5bad = [], [], 0, []
    for b in batches:
        for p in (R / b).rglob("*"):
            if not p.is_file():
                continue
            key = f"sources/{b}/{p.relative_to(R / b)}"
            f = hub.get(key)
            if f is None:
                missing.append(key); continue
            if f.size != p.stat().st_size:
                wrong.append(key); continue
            if p.suffix == ".h5":
                h5n += 1
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for c in iter(lambda: fh.read(1 << 22), b""):
                        h.update(c)
                if f.lfs is None or f.lfs.sha256 != h.hexdigest():
                    h5bad.append(key)
    print(repo, "batches", len(batches), "hub files under sources", len(hub), "missing", len(missing), missing[:3], "size-different", len(wrong), wrong[:3], "h5 checked", h5n, "h5 sha mismatch", len(h5bad), h5bad[:3], flush=True)
