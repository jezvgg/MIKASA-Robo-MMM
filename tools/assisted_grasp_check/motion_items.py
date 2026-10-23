"""Checklist motion items (DATASET_CHECKLIST D2-D7) for oracle trajectories saved by
scripts/collection_episode.py --trajectory-out, with the checklist's own functions
(utils/collection/checklist.py: motion_metrics, pause_check) on the 10 Hz export view (stride 2)
and D7's switch parity on the 20 Hz original.

    PYTHONPATH=. python tools/assisted_grasp_check/motion_items.py utils/collection/cabinet_search_profile.json traj/*.npz
"""
import json
import sys

import numpy as np

from utils.collection.checklist import motion_metrics, pause_check

profile = json.load(open(sys.argv[1]))
rows = []
for path in sys.argv[2:]:
    z = np.load(path, allow_pickle=True)
    a20, q20, names = z["action"], z["qpos"], [str(n) for n in z["joints"]]
    a, q = a20[0::2], q20[0::2]
    m = motion_metrics(a, q, names, fps=10)
    p = pause_check(a, profile)
    sw = np.flatnonzero(np.diff(a20[:, 7]) != 0) + 1
    row = {
        "file": path.split("/")[-1], "success": bool(z["success"]), "frames_10hz": len(a),
        "D2": bool(max(m["roll_span"]) <= np.pi + 1e-5 and min(m["roll_min"] + m["roll_command_min"]) >= -np.pi - 1e-5
                   and max(m["roll_max"] + m["roll_command_max"]) <= np.pi + 1e-5),
        "D3_close_signs": m["close_sign_combinations"],
        "D4": bool(m["max_continuous_turn_rad"] <= np.pi + 1e-5), "max_turn_rad": round(m["max_continuous_turn_rad"], 3),
        "D5_reverse_frames": m["reverse_frames"], "D5_direction_changes": m["base_direction_changes"],
        "D6": bool(p["passed"]), "identical_pairs": round(p["raw"]["identical_action_fraction"], 4),
        "longest_pause": p["raw"]["longest_constant_action_frames"],
        "D7_odd_switches": int((sw % 2 == 1).sum()),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
