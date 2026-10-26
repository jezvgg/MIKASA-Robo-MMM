"""Re-score E1 of already gated CabinetSearch batches with the rug rule, without rendering again: the stored
per-frame `visible` of <batch>-views-combined.json, the base xy of validated/<seed>/trajectory.h5 and the rug
centre of the episode ticket (RandomState(seed), fifth draw). A rug-phase frame with the base within
home_radius (0.20 m) of the rug centre counts as reached. Writes driver/<batch>-views-e1.json.

    python e1_recompute.py N_COMPARTMENTS BATCH [BATCH ...]
"""
import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np

H = Path.home() / "sd-collect"
HOME_XY, HOME_JITTER, HOME_RADIUS = np.array([1.80, -1.90]), np.array([0.25, 0.08]), 0.20
RULE = ("E1 rug rule (owner 2026-10-25): in CabinetSearch rug phases a frame with the base within home_radius "
        "of the rug centre counts the rug as reached; visible_fraction_raw = camera visibility as audited")


def rug(seed, n):
    rs = np.random.RandomState(seed)
    rs.randint(n)
    rs.uniform(-1.0, 1.0, size=2)
    rs.uniform(-1.0, 1.0, size=3)
    rs.randint(3)
    return HOME_XY + rs.uniform(-1.0, 1.0, size=2) * HOME_JITTER


n = int(sys.argv[1])
for batch in sys.argv[2:]:
    src = H / "driver" / f"{batch}-views-combined.json"
    d = json.loads(src.read_text()) if src.exists() else json.loads(gzip.open(str(src) + ".gz").read())
    out = []
    for e in d["episodes"]:
        seed = e["seed"]
        with h5py.File(H / "runs-ag" / batch / "validated" / str(seed) / "trajectory.h5") as f:
            xy = f["traj_0"]["qpos"][:, :2]
        r = rug(seed, n)
        vis = []
        for row in e["frames"]:
            reached = row["phase"].startswith("home_") and float(np.linalg.norm(xy[min(row["step"], len(xy) - 1)] - r)) <= HOME_RADIUS
            vis.append(bool(row["visible"]) or reached)
        out.append(dict(seed=seed, visible_fraction_raw=e["visible_fraction"], visible_fraction=float(np.mean(vis))))
    (H / "driver" / f"{batch}-views-e1.json").write_text(json.dumps(dict(rule=RULE, source=src.name, episodes=out), indent=1) + "\n")
    v = [x["visible_fraction"] for x in out]
    print(batch, len(out), f"E1 min {min(v):.3f}, >=0.95 {sum(x >= 0.95 for x in v)}/{len(v)}", flush=True)
