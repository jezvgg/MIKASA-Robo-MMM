"""Write our normalization bounds into MemoryVLA's cached dataset statistics.

MemoryVLA (from OpenVLA) hard-codes BOUNDS_Q99: every action channel is mapped from its task's
[q01, q99] to [-1, 1] and clipped, in training and at inference. On our data that clips real
motion (artifacts/base-normalization.md): DepthRecall's base turn has q01 = q99 = 0 and is never
produced. The authors' code reads q01/q99 from `dataset_statistics_<hash>.json` (cached next to
the RLDS on the first run) for normalization, the neutral action after an episode's end and
un-normalization, so the bounds are changed there and the code is left alone:

  base (channels 11, 12): q01 = -1, q99 = +1, the action contract's limits (fixed bounds, as the
      BEHAVIOR baselines, BRS and RoboCasa treat a wheeled base); stop stays 0;
  arm, head, torso (0-6, 8-10): q01 = min, q99 = max of the task (as OpenVLA-OFT for absolute
      joints); the gripper (7) is not normalized and is left as is.

usage: set_norm_bounds.py <dataset_statistics_*.json> [--check]   (--check only verifies)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = (11, 12)
JOINTS = tuple(range(7)) + (8, 9, 10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stats", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    data = json.loads(args.stats.read_text())
    keys = [k for k in data if isinstance(data[k], dict) and "action" in data[k]] or [None]
    for key in keys:
        act = data[key]["action"] if key else data["action"]
        want_lo = list(act["q01"]); want_hi = list(act["q99"])
        for i in BASE:
            want_lo[i], want_hi[i] = -1.0, 1.0
        for i in JOINTS:
            want_lo[i], want_hi[i] = act["min"][i], act["max"][i]
        if args.check:
            ok = all(abs(a - b) < 1e-9 for a, b in zip(act["q01"] + act["q99"], want_lo + want_hi))
            print(f"{key or 'action'}: {'bounds set' if ok else 'NOT set'}")
            if not ok:
                raise SystemExit(1)
            continue
        print(f"{key or 'action'}: q01 {[round(x, 3) for x in act['q01']]} -> {[round(x, 3) for x in want_lo]}")
        print(f"{' ' * len(key or 'action')}  q99 {[round(x, 3) for x in act['q99']]} -> {[round(x, 3) for x in want_hi]}")
        act["q01"], act["q99"] = want_lo, want_hi
    if not args.check:
        args.stats.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
