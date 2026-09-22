"""Execute a recorded action stream through the actual 12D/RGB policy boundary."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from .client import policy_metadata, run_policy_episode
from .contract import CAMERAS, read_json, write_json
from .profile import make_env


def qualify(root, seed, output):
    from utils.mikasa.seeding import seed_everything

    run = read_json(root / "run.json")
    with h5py.File(root / "validated" / str(seed) / "trajectory.h5", "r") as h5:
        actions = h5["traj_0/actions"][::2]

    class RecordedPolicy:
        cursor = 0
        inputs_checked = 0
        def get_server_metadata(self):
            return {"mikasa_data": policy_metadata()}
        def infer(self, observation):
            expected = {"observation.state", "prompt"} | {f"observation.images.{name}" for name in CAMERAS}
            if set(observation) != expected or observation["observation.state"].shape != (12,):
                raise AssertionError("The actual client exposed unsupported policy inputs")
            result = actions[self.cursor:self.cursor+1]
            self.cursor += 1
            self.inputs_checked += 1
            return {"actions": result}

    seed_everything(seed)
    env = make_env(run, rgb=True)
    policy = RecordedPolicy()
    try:
        obs, info = env.reset(seed=seed)
        result = run_policy_episode(env, policy, obs,
            env.unwrapped.get_language_instruction()[0], max_policy_steps=len(actions),
            execute_horizon=1, stop_on_success=False)
        result.update(seed=seed, policy_inputs_checked=policy.inputs_checked,
            policy_metadata=policy_metadata(), source_sha256=run["signature"]["code_sha256"],
            interpretation="Recorded-action interface check; no learned policy or memory-performance claim.")
        write_json(output, result)
        print(result)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.run.resolve(), args.seed, args.output.resolve())


if __name__ == "__main__":
    main()
