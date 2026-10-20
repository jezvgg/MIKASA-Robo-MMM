#!/usr/bin/env python3
"""RunPod volume (S3) or an ssh host -> Mac staging -> Hugging Face, one verified portion at a time.

The Mac has ~8 GB free, so a dataset never sits here whole: files are grouped into portions of
<= --portion-gb, each portion is downloaded, checked against S3 (size + md5 == single-part ETag),
committed to the HF repo in one commit, checked again on the Hub (size + sha256 of the LFS
object, git blob sha1 for small files), and only then deleted locally. One portion is prefetched
while the previous one uploads; uploads themselves are strictly sequential. State is a JSON file,
so a rerun resumes after the last verified portion.

usage: pipe.py --prefix mikasa-runs/.../seasondish-try-2/ --repo nurtayev-d/seasondish-try-2
               [--extra LOCALFILE:PATH_IN_REPO ...] [--portion-gb 2]
       pipe.py --ssh gangway:sd-collect/runs/final/samedrawer-1000ep-b --repo nurtayev-d/... --portion-gb 0.5

With --ssh HOST:DIR (DIR relative to the remote home) the files are listed with their sizes and
sha256 on the host, a portion is pulled as one tar stream, and every file's sha256 is checked
against the host's before the commit; the rest (portions, Hub checks, state) is the same.
"""
import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

AWS = [str(Path.home() / ".local/bin/aws"), "--profile", os.environ.get("RUNPOD_S3_PROFILE", "runpod"),
       "--endpoint-url", os.environ.get("RUNPOD_S3_ENDPOINT", "")]   # the RunPod volume's S3 endpoint (--prefix mode only)
BUCKET = os.environ.get("RUNPOD_S3_BUCKET", "")                     # the volume id (--prefix mode only)
LAST = ("meta/", "README.md")          # committed last, so the repo only looks complete when it is


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def s3_list(prefix):
    out = subprocess.run(AWS + ["s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", prefix,
                                "--output", "json"], check=True, capture_output=True, text=True).stdout
    items = json.loads(out or "{}").get("Contents", [])
    return [dict(key=o["Key"], size=o["Size"], etag=o["ETag"].strip('"')) for o in items]


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = path.stat().st_size
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk); sha.update(chunk); blob.update(chunk)
    return md5.hexdigest(), sha.hexdigest(), blob.hexdigest()


def ssh_list(host, root):
    """[{key, size, sha256}] for every file under ~/root on the host (key = path relative to it)."""
    script = (f'cd "$HOME/{root}" && find . -type f -printf "%s %P\\n" | sort -k2 && echo "--sha--" && '
              f'find . -type f -printf "%P\\0" | sort -z | xargs -0 sha256sum')
    out = subprocess.run(["ssh", "-n", host, script], check=True, capture_output=True, text=True).stdout
    sizes_txt, sha_txt = out.split("--sha--\n", 1)
    sizes = {}
    for line in sizes_txt.splitlines():
        size, key = line.split(" ", 1)
        sizes[key] = int(size)
    items = []
    for line in sha_txt.splitlines():
        digest, key = line.split("  ", 1)
        items.append(dict(key=key, size=sizes[key], sha256_remote=digest))
    assert len(items) == len(sizes), "listing and hashing disagree"
    return items


def ssh_download(portion, stage, host, root):
    """One tar stream per portion; every file checked against the host's sha256."""
    stage.mkdir(parents=True, exist_ok=True)
    names = "".join(it["key"] + "\n" for it in portion).encode()
    for attempt in range(5):
        pull = subprocess.Popen(["ssh", host, f'cd "$HOME/{root}" && tar -cf - --files-from=-'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        unpack = subprocess.Popen(["tar", "-xf", "-", "-C", str(stage)], stdin=pull.stdout)
        pull.stdin.write(names); pull.stdin.close(); pull.stdout.close()
        if unpack.wait() == 0 and pull.wait() == 0:
            break
        log(f"retry tar pull ({attempt + 1})")
        time.sleep(10 * (attempt + 1))
    else:
        raise SystemExit("ssh tar pull failed")
    for it in portion:
        dst = stage / it["key"]
        if not dst.exists() or dst.stat().st_size != it["size"]:
            raise SystemExit(f"size mismatch after pull: {it['key']}")
        _, sha, blob = hashes(dst)
        if sha != it["sha256_remote"]:
            raise SystemExit(f"sha256 mismatch against the host: {it['key']}")
        it.update(local=str(dst), sha256=sha, blob=blob)


def download(portion, stage, prefix):
    for it in portion:
        dst = stage / it["key"][len(prefix):]
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            r = subprocess.run(AWS + ["s3", "cp", f"s3://{BUCKET}/{it['key']}", str(dst),
                                      "--only-show-errors"], capture_output=True, text=True)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size == it["size"]:
                break
            log(f"retry download {it['key']} ({r.stderr.strip()[:200]})")
            time.sleep(10 * (attempt + 1))
        else:
            raise SystemExit(f"download failed: {it['key']}")
        md5, sha, blob = hashes(dst)
        if "-" not in it["etag"] and md5 != it["etag"]:
            raise SystemExit(f"md5 mismatch against S3 ETag: {it['key']}")
        it.update(local=str(dst), sha256=sha, blob=blob)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prefix", help="S3 prefix on the RunPod volume")
    src.add_argument("--ssh", help="HOST:DIR, DIR relative to the remote home")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--portion-gb", type=float, default=2.0)
    ap.add_argument("--extra", nargs="*", default=[], help="LOCALFILE:PATH_IN_REPO, committed last")
    ap.add_argument("--work", default=str(Path(__file__).parent / "work"))
    ap.add_argument("--path-prefix", default="", help="prepended to every path in the repo, e.g. sources/batch-001/")
    args = ap.parse_args()
    if args.ssh:
        host, root = args.ssh.split(":", 1)
        root = root.strip("/")
        prefix = ""
    else:
        prefix = args.prefix.rstrip("/") + "/"
    work = Path(args.work) / args.repo.replace("/", "__")
    if args.path_prefix:                    # one state file per prefix, so a batch upload never reads another's
        work = work / args.path_prefix.strip("/").replace("/", "__")

    def rp(it):
        """The file's path in the repo."""
        return args.path_prefix + it["key"][len(prefix):]
    work.mkdir(parents=True, exist_ok=True)
    state_file = work / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"done": {}}

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)

    items = ssh_list(host, root) if args.ssh else s3_list(prefix)
    total = sum(i["size"] for i in items)
    where = f"{host}:~/{root}" if args.ssh else f"s3://{BUCKET}/{prefix}"
    log(f"{len(items)} objects, {total / 1e9:.2f} GB under {where}; "
        f"{len(state['done'])} already verified on the Hub")
    todo = [i for i in items if rp(i) not in state["done"]]
    todo.sort(key=lambda i: (i["key"][len(prefix):].startswith(LAST), i["key"]))
    cap = args.portion_gb * 1e9
    portions, cur, cur_size = [], [], 0
    for it in todo:
        if cur and (cur_size + it["size"] > cap
                    or it["key"][len(prefix):].startswith(LAST) != cur[-1]["key"][len(prefix):].startswith(LAST)):
            portions.append(cur); cur, cur_size = [], 0
        cur.append(it); cur_size += it["size"]
    if cur:
        portions.append(cur)

    q = queue.Queue(maxsize=1)

    def prefetch():
        try:
            for n, portion in enumerate(portions, 1):
                stage = work / f"stage-{n:04d}"
                if args.ssh:
                    ssh_download(portion, stage, host, root)
                else:
                    download(portion, stage, prefix)
                q.put((n, stage, portion))
            q.put(None)
        except BaseException as e:                      # surface download failures to the main loop
            q.put(e)

    threading.Thread(target=prefetch, daemon=True).start()
    sent = sum(v["size"] for v in state["done"].values())
    while (got := q.get()) is not None:
        if isinstance(got, BaseException):
            raise SystemExit(f"prefetch failed: {got}")
        n, stage, portion = got
        size = sum(i["size"] for i in portion)
        ops = [CommitOperationAdd(path_in_repo=rp(i), path_or_fileobj=i["local"])
               for i in portion]
        for attempt in range(5):
            try:
                api.create_commit(args.repo, repo_type="dataset", operations=ops,
                                  commit_message=f"Add portion {n}/{len(portions)} ({len(ops)} files)")
                break
            except Exception as e:                                  # network hiccups: retry
                log(f"commit retry {attempt + 1}: {type(e).__name__}: {str(e)[:200]}")
                time.sleep(30 * (attempt + 1))
        else:
            raise SystemExit(f"commit failed for portion {n}")
        paths = [rp(i) for i in portion]
        info = {p.path: p for p in api.get_paths_info(args.repo, paths, repo_type="dataset", expand=True)}
        for i, p in zip(portion, paths):
            f = info.get(p)
            if f is None or f.size != i["size"]:
                raise SystemExit(f"hub check failed (missing/size): {p}")
            if f.lfs is not None:
                if f.lfs.sha256 != i["sha256"]:
                    raise SystemExit(f"hub sha256 mismatch: {p}")
            elif f.blob_id != i["blob"]:
                raise SystemExit(f"hub blob mismatch: {p}")
            state["done"][p] = dict(size=i["size"], sha256=i["sha256"])
        state_file.write_text(json.dumps(state))
        shutil.rmtree(stage)
        sent += size
        log(f"portion {n}/{len(portions)} verified: {len(ops)} files, {size / 1e9:.2f} GB; "
            f"{sent / 1e9:.2f}/{total / 1e9:.2f} GB on the Hub")

    extras = []
    for spec in args.extra:
        local, path_in_repo = spec.split(":", 1)
        extras.append((Path(local), path_in_repo))
    if extras:
        api.create_commit(args.repo, repo_type="dataset",
                          operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(l))
                                      for l, p in extras],
                          commit_message="Add dataset card and validation seeds")
        info = {p.path: p for p in api.get_paths_info(args.repo, [p for _, p in extras],
                                                      repo_type="dataset", expand=True)}
        for local, p in extras:
            if p not in info or info[p].size != local.stat().st_size:
                raise SystemExit(f"hub check failed for extra {p}")
        log(f"extras verified: {[p for _, p in extras]}")

    hub = {f.path: f.size for f in api.list_repo_tree(args.repo, repo_type="dataset", recursive=True)
           if hasattr(f, "size") and f.size is not None and not f.path.endswith(".gitattributes")}
    overridden = {p for _, p in extras}          # replaced on purpose by --extra, checked above
    missing = [rp(i) for i in items if rp(i) not in overridden and hub.get(rp(i)) != i["size"]]
    log(f"FINAL: {len(items)} source objects, {len(missing)} missing or different on the Hub")
    if missing:
        print("\n".join(missing[:50]))
        sys.exit(1)


if __name__ == "__main__":
    main()
