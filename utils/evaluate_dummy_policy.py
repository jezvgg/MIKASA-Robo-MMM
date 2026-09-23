"""Evaluate the fixed-action DummyPolicy on the DSFetch tray task.

Run a CPU smoke evaluation with:
    python -m utils.evaluate_dummy_policy --num-episodes 10
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import mani_skill  # noqa: F401
import my_scenes  # noqa: F401
import numpy as np

from utils.dummy_policy import DummyPolicy

ENV_ID = "MyRoboCasa_TakeItBackTray-v1"


def scalar_bool(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def evaluate(
    num_episodes: int = 10,
    start_seed: int = 0,
    seed_step: int = 1,
    max_steps: int = 1000,
) -> tuple[int, int]:
    hits = 0

    for episode in range(num_episodes):
        seed = start_seed + episode * seed_step
        env = gym.make(
            ENV_ID,
            num_envs=1,
            obs_mode="state",
            control_mode="pd_joint_pos",
            robot_uids="ds_fetch",
            sim_backend="cpu",
            render_backend="cpu",
        )
        try:
            policy = DummyPolicy()
            if policy.action.shape != env.action_space.shape:
                raise ValueError(
                    f"policy action {policy.action.shape} does not match "
                    f"environment action space {env.action_space.shape}"
                )
            obs, _ = env.reset(seed=seed)
            policy.reset()
            success = False

            for step in range(1, max_steps + 1):
                obs, _, terminated, truncated, info = env.step(policy.act(obs))
                success = success or scalar_bool(info["success"])
                if scalar_bool(terminated) or scalar_bool(truncated):
                    break
            hits += int(success)
            print(
                f"episode={episode + 1}/{num_episodes} seed={seed} "
                f"success={success} steps={step}",
                flush=True,
            )
        finally:
            env.close()

    print(f"success={hits}/{num_episodes}")
    return hits, num_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    if args.num_episodes < 1 or args.max_steps < 1:
        parser.error("num-episodes and max-steps must be positive")
    evaluate(args.num_episodes, args.start_seed, args.seed_step, args.max_steps)


if __name__ == "__main__":
    main()
