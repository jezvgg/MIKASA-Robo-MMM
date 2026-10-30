#!/usr/bin/env bash
# Host memory of the MemER high level's data loading on 4 GPUs, per dataloader worker count:
# vla/memer/train_high.sh on the current SFT export ($SFT_ROOT, $DATASETS) with WORKERS set to each
# of $WORKER_COUNTS in turn, no checkpoint (save steps past the end), stopped after $STEPS logged
# steps or $MAX_MIN minutes. Every 10 s it records the container's anonymous memory (cgroup
# memory.stat anon) and the RSS and PSS of the run's processes, rank processes and their dataloader
# workers apart, plus GPU use. A guard stops a run whose container anon memory passes $GUARD_GIB:
# the container's limit is its memory.max (728 GiB on 2026-10-27), and past it the pod is killed,
# as on 2026-10-27 13:03. torchrun starts every rank in a session of its own, so the run is
# followed as the process tree under the launch's PID, not as a process group.
# usage: nohup vla/memer/probe_loader_mem.sh [work dir, default ~/loader-probe] > log 2>&1 < /dev/null &
# Result: <work dir>/REPORT.txt; per run w<N>.mem.csv (memory), w<N>.gpu.csv, w<N>.log (training).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=${1:-$HOME/loader-probe}
SFT_ROOT=${SFT_ROOT:-$HOME/datasets/memer_sft}
DATASETS=${DATASETS:-memer_samedrawer,memer_seasondish,memer_cabinetsearch,memer_depthrecall}
WORKER_COUNTS=${WORKER_COUNTS:-4 2 3}; STEPS=${STEPS:-60}; GUARD_GIB=${GUARD_GIB:-600}; MAX_MIN=${MAX_MIN:-45}
mkdir -p "$W"
REPORT=$W/REPORT.txt
say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$REPORT"; }

tree() {  # tree PID: PID and all its descendants
  local p; echo "$1"; for p in $(ps -o pid= --ppid "$1" 2>/dev/null); do tree "$p"; done
}
stop_tree() {  # stop_tree PID: TERM to the whole tree, KILL to what is left 40 s later
  local pids; pids=$(tree "$1"); kill -TERM $pids 2>/dev/null; sleep 40; kill -KILL $pids $(tree "$1") 2>/dev/null
}

sampler() {  # sampler PID CSV: one line every 10 s while the tree lives; stops the tree past the guard
  python3 - "$1" "$2" "$GUARD_GIB" <<'PY'
import os, signal, sys, time
root, out, guard = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
G = 1 << 30
def anon():
    for line in open("/sys/fs/cgroup/memory.stat"):
        if line.startswith("anon "):
            return int(line.split()[1]) / G
def tree():
    parent = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                parent[int(p)] = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
            except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
                pass
    if root not in parent:
        return {}
    kids = {}
    for p, pp in parent.items():
        kids.setdefault(pp, []).append(p)
    found, todo = {}, [root]
    while todo:
        p = todo.pop()
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            rss = pss = 0.0
            for line in open(f"/proc/{p}/smaps_rollup"):
                if line.startswith("Rss:"): rss = int(line.split()[1]) * 1024 / G
                elif line.startswith("Pss:"): pss = int(line.split()[1]) * 1024 / G
            found[p] = (parent[p], cmd, rss, pss)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        todo += kids.get(p, [])
    return found
def stop(pids):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in pids:
            try: os.kill(p, sig)
            except ProcessLookupError: pass
        if sig == signal.SIGTERM: time.sleep(30)
with open(out, "w") as f:
    f.write("time,container_anon_gib,ranks,ranks_rss,ranks_pss,workers,workers_rss,workers_pss,max_worker_pss\n")
    t0 = time.time(); stopped = False
    while True:
        ps = tree()
        if not ps:
            break
        ranks = {p: v for p, v in ps.items() if "train_qwen.py" in v[1] and "torchrun" in ps.get(v[0], (0, ""))[1]}
        workers = {p: v for p, v in ps.items() if v[0] in ranks}
        a = anon()
        f.write(f"{time.time() - t0:.0f},{a:.1f},{len(ranks)},{sum(v[2] for v in ranks.values()):.1f},"
                f"{sum(v[3] for v in ranks.values()):.1f},{len(workers)},{sum(v[2] for v in workers.values()):.1f},"
                f"{sum(v[3] for v in workers.values()):.1f},{max([v[3] for v in workers.values()] or [0]):.1f}\n")
        f.flush()
        if a > guard and not stopped:
            f.write(f"GUARD: container anon {a:.0f} GiB > {guard:.0f}; stopping the run\n"); f.flush()
            stop(list(ps)); stopped = True
        time.sleep(10)
PY
}

say "loader memory probe: $DATASETS from $SFT_ROOT, workers per GPU: $WORKER_COUNTS, $STEPS steps or $MAX_MIN min each, guard $GUARD_GIB GiB (container limit $(( $(cat /sys/fs/cgroup/memory.max) >> 30 )) GiB, anon now $(awk '/^anon /{printf "%.0f", $2 / 2^30}' /sys/fs/cgroup/memory.stat) GiB)"
for w in $WORKER_COUNTS; do
  log=$W/w$w.log out=$W/out_w$w
  rm -rf "$out"
  start=$(date +%s)
  WORKERS="$w" DATASETS="$DATASETS" SAVE_ONLY_MODEL=1 \
    "$REPO/vla/memer/train_high.sh" "$SFT_ROOT" "$out" 10000000 10000000 > "$log" 2>&1 &
  pid=$!
  sampler "$pid" "$W/w$w.mem.csv" &
  spid=$!
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 10 > "$W/w$w.gpu.csv" 2>/dev/null &
  gpid=$!
  steps() { tr '\r' '\n' < "$log" | grep -c "'loss':"; }
  first=""
  while kill -0 "$pid" 2>/dev/null && [ "$(steps)" -lt "$STEPS" ] && [ $(( $(date +%s) - start )) -lt $((MAX_MIN * 60)) ]; do
    [ -z "$first" ] && [ "$(steps)" -ge 1 ] && first=$(( $(date +%s) - start ))
    sleep 15
  done
  stop_tree "$pid"
  wait "$spid" 2>/dev/null; kill "$gpid" 2>/dev/null; wait "$gpid" 2>/dev/null
  rm -rf "$out"
  python3 - "$W/w$w.mem.csv" "$W/w$w.gpu.csv" "$log" "$w" "${first:-none}" <<'PY' | tee -a "$REPORT"
import csv, re, sys
mem, gpu, log, w = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = [r for r in csv.DictReader(l for l in open(mem) if not l.startswith("GUARD"))]
guard = [l.strip() for l in open(mem) if l.startswith("GUARD")]
text = open(log, errors="replace").read().replace("\r", "\n")
loss = re.findall(r"'loss': ([0-9.e+-]+)", text)
its = re.findall(r"(\d+\.\d+)s/it", text)
util = [int(l.split(", ")[3].split()[0]) for l in open(gpu) if l.count(", ") >= 3][-48:]  # the last 2 minutes
peak = lambda k: max(float(r[k]) for r in rows) if rows else 0
print(f"workers {w}/GPU: first step after {sys.argv[5]} s; {len(loss)} steps logged; step time {its[-1] if its else '?'} s; "
      f"GPU mean util, last 2 min {sum(util) / max(len(util), 1):.0f}%"
      + (f"; {guard[0]}" if guard else "") + ("; Traceback in the log" if "Traceback" in text else ""))
print(f"  peak: container anon {peak('container_anon_gib'):.0f} GiB; ranks RSS {peak('ranks_rss'):.0f} / PSS {peak('ranks_pss'):.0f} GiB; "
      f"{max(int(r['workers']) for r in rows) if rows else 0} workers RSS {peak('workers_rss'):.0f} / PSS {peak('workers_pss'):.0f} GiB "
      f"(largest worker PSS {peak('max_worker_pss'):.1f} GiB)")
for r in rows[::max(len(rows) // 8, 1)] + rows[-1:]:
    print(f"    t={int(r['time']):5d}s anon {float(r['container_anon_gib']):6.0f}  ranks PSS {float(r['ranks_pss']):5.0f}  "
          f"workers {r['workers']:>2} PSS {float(r['workers_pss']):6.0f} (max {float(r['max_worker_pss']):4.1f})")
PY
done
say "finished"
