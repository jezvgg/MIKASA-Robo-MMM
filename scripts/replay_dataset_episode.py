"""Replay one recorded episode's actions, as the collection's paired replay does, optionally with assisted grasp.

    PYTHONPATH=. python scripts/replay_dataset_episode.py season_dish actions.npy 7762000 [--assisted-grasp]

``actions.npy`` holds an episode's 10 Hz absolute targets (the LeRobot ``action`` column of
our published datasets, frame order). The env is built from the task's pinned profile,
reset with the episode's seed, and every target is held for two 20 Hz control steps by
`utils.collection.client.execute_actions` — the same call and the same repeat the
collection's replay validation used, so with the rule off the recorded outcome must come
back. With ``--assisted-grasp`` the same open-loop actions run under the weld rule: the
difference is what the rule changes for motions made under friction.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PROFILES = Path(__file__).resolve().parents[1] / "utils" / "collection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=("season_dish", "same_drawer", "cabinet_search"))
    ap.add_argument("actions", type=Path)
    ap.add_argument("seed", type=int)
    ap.add_argument("--assisted-grasp", action="store_true")
    args = ap.parse_args()

    import gymnasium as gym

    import my_scenes  # noqa: F401
    from utils.collection.client import execute_actions
    from utils.collection.contract import ACTION_REPEAT
    from utils.mikasa.seeding import seed_everything

    profile = json.loads((PROFILES / f"{args.task}_profile.json").read_text())
    kwargs = dict(profile["env_kwargs"], obs_mode="state")
    if args.assisted_grasp:
        kwargs["assisted_grasp"] = True
    actions = np.load(args.actions).astype(np.float32)
    seed_everything(args.seed)
    env = gym.make(profile["env_id"], **kwargs)
    base = env.unwrapped
    try:
        env.reset(seed=args.seed)
        result = execute_actions(env, actions, repeat=ACTION_REPEAT)
        info = result["info"]
        flag = lambda k: bool(np.asarray(info[k].cpu() if hasattr(info[k], "cpu") else info[k]).reshape(-1)[0]) \
            if k in info else None
        print(f"replay task={args.task} seed={args.seed} assisted_grasp={args.assisted_grasp} "
              f"status={result['status']} success={flag('success')} success_once={flag('success_once')} "
              f"steps={result['control_steps']} of={len(actions) * ACTION_REPEAT}", flush=True)
        ag = base._ag
        if ag is not None:
            kinds = [k for _, k, _ in ag.events]
            print(f"assisted_grasp establish={kinds.count('establish')} release={kinds.count('release')} "
                  f"events={ag.events}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
