#!/usr/bin/env python3
"""Container memory log, so that a rise toward the pod's limit can be traced to its program afterwards
(on 2026-10-27 only the machine's total was logged, other users included). Every INTERVAL s it
appends one line to OUT_DIR/<date>.log: the cgroup's memory.current and the memory.stat entries that
matter (anon and shmem cannot be reclaimed; file is page cache, which the kernel drops under
pressure), then the five largest program groups by anonymous RSS (RssAnon; PSS is not readable here
outside one's own process tree). A group is every process running the same script or module, so a
rank and its forked loader workers count together; pages they still share count once per process.
usage: nohup setsid python3 mem_log.py OUT_DIR [interval s, 60] > /dev/null 2>&1 < /dev/null &"""
import os
import sys
import time

G = 1 << 30
KEYS = ("anon", "shmem", "file", "file_dirty", "kernel", "slab_reclaimable")


def label(pid):  # the script or module a process runs, else its program name
    argv = [a.decode(errors="replace") for a in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if a]
    for i, a in enumerate(argv):
        if a == "-m" and i + 1 < len(argv):
            return argv[i + 1]
        if a.endswith((".py", ".sh")):
            return os.path.basename(a)
    return os.path.basename(argv[0].split()[0]) if argv else None


def rss_anon(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) * 1024
    return 0


out_dir = sys.argv[1]
every = float(sys.argv[2]) if len(sys.argv) > 2 else 60
os.makedirs(out_dir, exist_ok=True)
while True:
    stat = dict(line.split() for line in open("/sys/fs/cgroup/memory.stat"))
    groups = {}
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != os.getpid():
            try:
                name, b = label(p), rss_anon(p)
                if name:
                    g = groups.setdefault(name, [0, 0])
                    g[0] += b
                    g[1] += 1
            except OSError:  # gone, or not ours to read
                pass
    top = sorted(groups.items(), key=lambda kv: -kv[1][0])[:5]
    line = (f"{time.strftime('%F %T')} current {int(open('/sys/fs/cgroup/memory.current').read()) / G:.1f} "
            + " ".join(f"{k} {int(stat.get(k, 0)) / G:.1f}" for k in KEYS) + " GiB | "
            + "; ".join(f"{name} x{n} {b / G:.1f}" for name, (b, n) in top))
    with open(os.path.join(out_dir, time.strftime("%F") + ".log"), "a") as f:
        f.write(line + "\n")
    time.sleep(every)
