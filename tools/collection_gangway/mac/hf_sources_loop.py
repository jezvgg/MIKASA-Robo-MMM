#!/usr/bin/env python3
"""Upload each finished collection batch's source files (trajectory .h5 + json + logs) to the dataset's HF repo
under sources/<batch>/, then delete the batch's .h5 on the server. Owner 2026-10-19: «заливай h5 на hf в то же репо».

usage: hf_sources_loop.py --task cabinet_search --repo nurtayev-d/cabinetsearch-ag-1000ep [--once]

A batch is ready once its ledger exists (its gate, export and retention are done). The upload is
tools/hf_upload_pipe.py --ssh, which verifies every file on the Hub (size, LFS sha256 or git blob) and exits 0 only
if nothing is missing; only then are the batch's .h5 unlinked on the server (manifest
driver/deletions/<batch>-h5-on-hf.json). Done batches are remembered in tools/work/<repo>/sources-done.json.
--keep uploads without deleting (the default deletes after verification).
Runs until the task's driver has stopped and every ledgered batch is uploaded (or once with --once).
Run with the interpreter that has huggingface_hub >= 1.0 (anaconda env `mikasa`), HF_HUB_DISABLE_XET=1.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
HOST, RUNS, DRIVER = "gangway", "sd-collect/runs-ag", "sd-collect/driver"

DELETE = r'''
import datetime, json, os, sys
from pathlib import Path
batch, repo = sys.argv[1], sys.argv[2]
R = Path(os.path.expanduser("~/sd-collect/runs-ag")).resolve(); A = Path(os.path.expanduser("~/sd-collect/driver"))
run = (R / batch).resolve()
assert run.parent == R and run.is_dir(), run
gone = []
for f in sorted(run.rglob("*.h5")):
    gone.append(dict(path=str(f.relative_to(R)), bytes=f.stat().st_size)); f.unlink()
(A / "deletions").mkdir(exist_ok=True)
(A / "deletions" / f"{batch}-h5-on-hf.json").write_text(json.dumps(dict(
    rule="owner 2026-10-19: source h5 uploaded to the dataset repo (verified), then deleted here",
    repo=repo, path_in_repo=f"sources/{batch}/", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    files=gone), indent=1) + "\n")
print(len(gone), sum(g["bytes"] for g in gone))
'''

STATUS = r'''
import glob, json, os, sys
task, prefix = sys.argv[1], sys.argv[2]
A = os.path.expanduser("~/sd-collect/driver")
ledgers = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{A}/ledger/{task}-{prefix}train-*.json"))
try:
    pid = int(open(f"{A}/pids/{task}.pid").read()); os.kill(pid, 0); alive = True
except Exception:
    alive = False
print(json.dumps(dict(ledgers=ledgers, driver_alive=alive)))
'''


def ssh_py(code, *args, timeout=600):
    r = subprocess.run(["ssh", HOST, "python3", "-", *args], input=code, capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-400:])
    return r.stdout.strip()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--prefix", default="ag-", help="batch prefix (profile generation), e.g. ag6- for DR profile 6")
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--keep", action="store_true", help="upload only; keep the .h5 on the server (owner 2026-10-19 «теперь можно не удалять», +300 GB)")
    args = ap.parse_args()
    work = HERE / "work" / args.repo.replace("/", "__")
    work.mkdir(parents=True, exist_ok=True)
    done_file = work / "sources-done.json"
    done = set(json.loads(done_file.read_text())) if done_file.exists() else set()
    env = dict(os.environ, HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    while True:
        try:
            st = json.loads(ssh_py(STATUS, args.task, args.prefix, timeout=120))
        except Exception as e:                      # the gateway drops now and then: try again later
            log(f"status failed: {e}"); time.sleep(args.every); continue
        todo = [b for b in st["ledgers"] if b not in done]
        for batch in todo:
            r = subprocess.run([sys.executable, str(HERE / "hf_upload_pipe.py"), "--ssh", f"{HOST}:{RUNS}/{batch}",
                                "--repo", args.repo, "--path-prefix", f"sources/{batch}/", "--portion-gb", "0.5"],
                               env=env, capture_output=True, text=True)
            tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
            if r.returncode != 0:
                log(f"{batch}: upload failed ({tail[0][:200]}); retry next round"); break
            if args.keep:
                done.add(batch); done_file.write_text(json.dumps(sorted(done)))
                log(f"{batch}: on HF ({tail[0][:120]}); h5 kept on the server"); continue
            try:
                n, size = ssh_py(DELETE, batch, args.repo).split()
            except Exception as e:
                log(f"{batch}: uploaded, h5 delete failed: {e}"); break
            done.add(batch); done_file.write_text(json.dumps(sorted(done)))
            log(f"{batch}: on HF ({tail[0][:120]}); deleted {n} h5, {int(size) / 2**30:.2f} GiB")
        if args.once or (not st["driver_alive"] and all(b in done for b in st["ledgers"])):
            log(f"finished: {len(done)} batches"); return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
