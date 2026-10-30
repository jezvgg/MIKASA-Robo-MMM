"""10 Hz replay with pair-averaged actions instead of the held even action (owner 2026-10-26: «вместо того что
каждое действие держится 2 шага, сделай среднее для 20 гц. И проверь так»).

For every oracle success of a campaign run dir: reset with the seed, check the initial state against the oracle
recording, then execute a10[k] = mean(a20[2k], a20[2k+1]), each held for two control steps (a final unpaired
action is kept as it is), with the pipeline's own execute_actions. Success as in the validated phase: every step
completed and success at the end. MODE all averages all 13 components; MODE base averages only the base
velocities (11, 12) and holds the even action for the rest. Nothing in the run dir is changed; the result goes to
<run>/averaged-<mode>.json.

    cd <repo of the run> && PYTHONPATH=. python avg_replay.py RUN_DIR MODE [JOBS]
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np


def averaged(a20, mode):
    n = len(a20) // 2
    held = a20[::2].copy()
    mean = held.copy()
    mean[:n] = 0.5 * (a20[0:2 * n:2] + a20[1:2 * n:2])
    out = mean if mode == "all" else held
    if mode == "base":
        out[:, 11:13] = mean[:, 11:13]
    gripper_split = int(np.sum(a20[0:2 * n:2, 7] != a20[1:2 * n:2, 7]))
    return out.astype(np.float32), held, gripper_split


def one(args):
    root, seed, mode = args
    import h5py
    from utils.collection.client import execute_actions
    from utils.collection.pipeline import episode_path, initial_state_matches
    from utils.collection.profile import make_env
    from utils.mikasa.seeding import seed_everything

    run = json.loads((root / "run.json").read_text())
    seed_everything(seed)
    env = make_env(run)
    try:
        env.reset(seed=seed)
        with h5py.File(episode_path(root, "oracle", seed), "r") as h5:
            a20 = h5["traj_0"]["actions"][:]
            if not initial_state_matches(h5["traj_0"]["env_states"], env.unwrapped.get_state_dict()):
                return dict(seed=seed, status="initial_state_mismatch", success=False)
        actions, held, split = averaged(a20, mode)
        r = execute_actions(env, actions, repeat=2)
        ok = bool(r["completed"] and r["success"] and r["status"] == "completed")
        diff = np.abs(actions - held).max(0)
        return dict(seed=seed, success=ok, status="success" if ok else (r["status"] if r["status"] != "completed" else "missed"),
                    source_steps=len(a20), steps=r["control_steps"], gripper_pairs_split=split,
                    max_abs_change_vs_held=[round(float(x), 5) for x in diff])
    finally:
        env.close()


if __name__ == "__main__":
    root, mode = Path(sys.argv[1]).resolve(), sys.argv[2]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    assert mode in ("all", "base"), mode
    attempts = json.loads((root / "attempts.json").read_text())
    seeds = sorted(int(s) for s in attempts["oracle_successful_seeds"])
    with mp.get_context("spawn").Pool(min(jobs, len(seeds))) as pool:
        rows = pool.map(one, [(root, s, mode) for s in seeds])
    held = {int(s): r["status"] for s, r in attempts["results"].get("validated", {}).items()}
    out = dict(rule=__doc__.split("\n\n")[0], mode=mode, oracle_attempts=len(attempts["attempted_seeds"]),
               oracle_successes=len(seeds), averaged_successes=sum(r["success"] for r in rows),
               held_successes=sum(v == "success" for v in held.values()), episodes=rows)
    (root / f"averaged-{mode}.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(r["seed"], r["status"], "| held:", held.get(r["seed"]), "| gripper pairs split", r.get("gripper_pairs_split"))
    print(f"{root.name} mode={mode}: oracle {len(seeds)}/{out['oracle_attempts']}, "
          f"averaged 10 Hz {out['averaged_successes']}/{len(seeds)}, held 10 Hz {out['held_successes']}/{len(seeds)}")
