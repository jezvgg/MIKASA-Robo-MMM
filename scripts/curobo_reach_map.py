"""Reach map of ds_fetch from one stand, by cuRobo's batched scan (`robots/fetch/curobo_client.scan`).

    MIKASA_CUROBO_SOCKET=... (or MIKASA_CUROBO_PY=...) PYTHONPATH=. python scripts/curobo_reach_map.py OUT.json

Which gripper poses in front of the base the arm reaches with the torso free, on a grid of
x (ahead) and y (sideways) at the heights the tasks use, for a side grasp (approach level,
from the base towards the point) and a top grasp (approach straight down). Kinematics
only, then with self-collision; the scene is not in it. No simulator needed.
"""
import json
import sys
import time

import numpy as np

from robots.fetch import curobo_client

HEIGHTS = (0.60, 0.92, 1.20, 1.48)      # floor-level shelf, counter top, mid, the Depth Recall shelf
XS = np.round(np.arange(0.20, 1.2001, 0.025), 3)
YS = np.round(np.arange(-0.70, 0.7001, 0.025), 3)


def side(p):
    """Approach (gripper x) level and pointing from the base at the point; fingers close sideways."""
    a = np.array([p[0], p[1], 0.0])
    a /= np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, a)
    T = np.eye(4)
    T[:3, :3] = np.stack([a, y, z], axis=1)
    T[:3, 3] = p
    return T


def top(p):
    """Approach straight down, fingers closing along the base's y."""
    T = np.eye(4)
    T[:3, :3] = np.stack([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], axis=1)
    T[:3, 3] = p
    return T


def main(out):
    pts = [(x, y, h) for h in HEIGHTS for x in XS for y in YS]
    report = dict(heights=list(HEIGHTS), xs=XS.tolist(), ys=YS.tolist(), maps={}, timing={})
    for name, make in (("side", side), ("top", top)):
        goals = np.stack([make(np.array(p)) for p in pts])
        for coll in (False, True):
            t0 = time.perf_counter()
            ok, _ = curobo_client.scan(goals, [(0.0, 0.0, 0.0)], coll=coll)
            wall = time.perf_counter() - t0
            grid = ok.reshape(len(HEIGHTS), len(XS), len(YS))
            key = f"{name}_{'selfcoll' if coll else 'kinematic'}"
            report["maps"][key] = grid.astype(int).tolist()
            report["timing"][key] = dict(goals=len(goals), wall_s=round(wall, 3))
            frac = grid.mean(axis=(1, 2))
            print(key, f"{len(goals)} goals in {wall:.2f} s;", "reachable by height:",
                  {h: round(float(f), 3) for h, f in zip(HEIGHTS, frac)}, flush=True)
    report["server"] = {k: v for k, v in curobo_client.STATS.items()}
    with open(out, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main(sys.argv[1])
