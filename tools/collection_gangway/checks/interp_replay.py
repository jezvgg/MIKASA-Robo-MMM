"""20 Hz replay of the 10 Hz targets with the second step interpolated (owner 2026-10-26: «вместо двух одинаковых
действий/целей, получаемых из 10гц, мы второе действие/цель считаем как среднее между двумя из 10»).

The 10 Hz targets are the dataset's a10[k] = a20[2k] of the oracle recording. Each is executed for two control
steps: a10[k], then (a10[k] + a10[k+1]) / 2; the last one, with no next target, is held for both. MODE all
interpolates all 13 components; MODE nogrip holds the gripper (component 7) and interpolates the rest. Success as
in the validated phase: every step completed and success at the end. The robot qpos of every step is compared with
the held 10 Hz replay (validated/<seed>/trajectory.h5). Nothing in the run dir is changed; the result goes to
<run>/interp-<mode>.json.

    cd <repo of the run> && PYTHONPATH=. python interp_replay.py RUN_DIR MODE [JOBS]
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np


def interpolated(a10, mode):
    nxt = np.concatenate([a10[1:], a10[-1:]])
    mid = 0.5 * (a10 + nxt)
    if mode == "nogrip":
        mid[:, 7] = a10[:, 7]
    out = np.empty((2 * len(a10), a10.shape[1]), dtype=np.float32)
    out[0::2], out[1::2] = a10, mid
    return out


def one(args):
    root, seed, mode = args
    import h5py
    from utils.collection.client import as_numpy, check_actions, scalar_bool
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
        actions = interpolated(a20[::2], mode)
        check_actions(actions)
        robot = env.unwrapped.agent.robot
        qpos = [as_numpy(robot.get_qpos())[0]]
        status, success = "completed", False
        for action in actions:
            _, _, terminated, truncated, info = env.step(action)
            qpos.append(as_numpy(robot.get_qpos())[0])
            success = scalar_bool(info["success"])
            if scalar_bool(truncated):
                status = "truncated"; break
            if scalar_bool(terminated) and (not success or scalar_bool(info.get("fail", False))):
                status = "terminated"; break
        ok = bool(status == "completed" and success and len(qpos) == len(actions) + 1)
        held = None
        vp = episode_path(root, "validated", seed)
        if vp.exists():
            with h5py.File(vp, "r") as h5:
                held = h5["traj_0"]["qpos"][:]
        dev = None
        if held is not None:
            n = min(len(held), len(qpos))
            dev = float(np.abs(np.asarray(qpos[:n]) - held[:n]).max())
        return dict(seed=seed, success=ok, status="success" if ok else (status if status != "completed" else "missed"),
                    source_steps=len(a20), steps=len(qpos) - 1, gripper_mid_steps=int(np.sum((actions[1::2, 7] != actions[0::2, 7]))),
                    max_qpos_diff_vs_held_rad=dev)
    finally:
        env.close()


if __name__ == "__main__":
    root, mode = Path(sys.argv[1]).resolve(), sys.argv[2]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    assert mode in ("all", "nogrip"), mode
    attempts = json.loads((root / "attempts.json").read_text())
    seeds = sorted(int(s) for s in attempts["oracle_successful_seeds"])
    with mp.get_context("spawn").Pool(min(jobs, len(seeds))) as pool:
        rows = pool.map(one, [(root, s, mode) for s in seeds])
    held = {int(s): r["status"] for s, r in attempts["results"].get("validated", {}).items()}
    out = dict(rule=__doc__.split("\n\n")[0], mode=mode, oracle_attempts=len(attempts["attempted_seeds"]),
               oracle_successes=len(seeds), interp_successes=sum(r["success"] for r in rows),
               held_successes=sum(v == "success" for v in held.values()), episodes=rows)
    (root / f"interp-{mode}.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(r["seed"], r["status"], "| held:", held.get(r["seed"]), "| qpos diff vs held", r.get("max_qpos_diff_vs_held_rad"),
              "| gripper mid steps", r.get("gripper_mid_steps"))
    print(f"{root.name} mode={mode}: oracle {len(seeds)}/{out['oracle_attempts']}, interpolated {out['interp_successes']}/{len(seeds)}, "
          f"held {out['held_successes']}/{len(seeds)}")
