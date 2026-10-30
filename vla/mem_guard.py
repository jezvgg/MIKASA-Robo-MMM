#!/usr/bin/env python3
"""Stop a run before its container runs out of memory: past the cgroup's memory.max the gangway pod
is killed whole, with everything in it (2026-10-27, twice). While process ROOT lives, this reads the
container's anonymous memory every 2 s; past the guard (default 80% of memory.max) it sends TERM to
ROOT and all its descendants, KILL 5 s later, prints one GUARD line and exits 3. torchrun starts
every rank in a session of its own, so the run is followed as ROOT's process tree, not a group.
usage: mem_guard.py ROOT_PID [guard GiB]"""
import os
import signal
import sys
import time

G = 1 << 30


def anon():
    for line in open("/sys/fs/cgroup/memory.stat"):
        if line.startswith("anon "):
            return int(line.split()[1]) / G
    return 0.0


def limit():
    raw = open("/sys/fs/cgroup/memory.max").read().strip()
    if raw != "max":
        return int(raw) / G
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024 / G


def stat(pid):  # (state, parent) of a live process, None when it is gone
    try:
        fields = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return fields[0], int(fields[1])
    except (OSError, IndexError, ValueError):
        return None


def tree(root):  # root and all its descendants, this guard left out
    kids = {}
    for p in os.listdir("/proc"):
        s = p.isdigit() and stat(int(p))
        if s:
            kids.setdefault(s[1], []).append(int(p))
    found, todo = [], [root]
    while todo:
        p = todo.pop()
        found.append(p)
        todo += kids.get(p, [])
    return [p for p in found if p != os.getpid()]


root = int(sys.argv[1])
guard = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8 * limit()
while (s := stat(root)) and s[0] != "Z":
    a = anon()
    if a > guard:
        pids = tree(root)
        print(f"[{time.strftime('%F %T')}] GUARD: container anon memory {a:.0f} GiB > {guard:.0f} GiB "
              f"(limit {limit():.0f} GiB); stopping {len(pids)} processes under {root}", flush=True)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            # a rank whose torchrun died first is reparented, so the first snapshot is kept
            for p in pids + [p for p in tree(root) if p not in pids]:
                try:
                    os.kill(p, sig)
                except ProcessLookupError:
                    pass
            if sig == signal.SIGTERM:
                time.sleep(5)
        sys.exit(3)
    time.sleep(2)
