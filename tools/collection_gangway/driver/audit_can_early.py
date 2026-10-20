"""CabinetSearch gate (profile 16, 2026-10-18): the can stays where it spawned until the push starts.

usage: audit_can_early.py <run root>  -> <run>-can-early.json next to this script

For every oracle success: the step of the first "push start" event (events.jsonl, logged by
cabinet_search_planner right before the deliberate push), and over env_states/actors/cube from
step 0 to that step the largest horizontal displacement from the step-0 position and the largest
tilt of the can's own z axis from its step-0 direction. Pass: <= 0.01 m and <= 20 deg.
Why: the owner's E5 review (2026-10-18) saw the can knocked over before the push in published
episodes; a replay of all 1000 by the DR/SeasonDish session found 7.5 % moving >= 12 frames
before the end, 62 of them tipping > 45 deg, while the success check passes them.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

MAX_XY_M, MAX_TILT_DEG = 0.01, 20.0
root = Path(sys.argv[1])
A = Path(__file__).parent
run = json.loads((root / "run.json").read_text())


def z_axis(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)


rows = []
for seed in run["seeds"]:
    d = root / "oracle" / str(seed)
    if not (d / "result.json").exists() or json.loads((d / "result.json").read_text()).get("status") != "success":
        continue
    events = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines() if line.strip()]
    starts = [e["step"] for e in events if e.get("message") == "push start"]
    with h5py.File(d / "trajectory.h5") as f:
        cube = np.asarray(f["traj_0"]["env_states"]["actors"]["cube"][:], dtype=np.float64)
    end = starts[0] if starts else len(cube) - 1
    seg = cube[: end + 1]
    xy = np.linalg.norm(seg[:, :2] - seg[0, :2], axis=1)
    zs = z_axis(seg[:, 3:7])
    tilt = np.degrees(np.arccos(np.clip(zs @ zs[0], -1.0, 1.0)))
    rows.append(dict(seed=seed, push_start=starts[0] if starts else None, frames=int(len(cube)),
                     max_xy_m=round(float(xy.max()), 5), max_tilt_deg=round(float(tilt.max()), 2),
                     passed=bool(starts and xy.max() <= MAX_XY_M and tilt.max() <= MAX_TILT_DEG)))
out = A / f"{root.name}-can-early.json"
out.write_text(json.dumps(dict(run=str(root), max_xy_m=MAX_XY_M, max_tilt_deg=MAX_TILT_DEG, episodes=rows), indent=1) + "\n")
print(len(rows), "successes;", sum(r["passed"] for r in rows), "passed;",
      [(r["seed"], r["max_xy_m"], r["max_tilt_deg"]) for r in rows if not r["passed"]], flush=True)
