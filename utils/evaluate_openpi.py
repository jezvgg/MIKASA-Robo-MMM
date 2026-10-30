"""Run closed-loop Fetch evaluation against a separately served OpenPI policy."""

from __future__ import annotations

import argparse

import gymnasium as gym
import mani_skill  # noqa: F401
import my_scenes  # noqa: F401
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

ENV_ID = "MyRoboCasa_TakeIt-v1"
CAMERAS = ("left_base_camera_link", "fetch_hand", "right_base_camera_link")
ACTION_REPEAT = 2  # dataset actions are 10 Hz; simulator runs at 20 Hz


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def camera_frame(obs: dict, camera: str) -> np.ndarray:
    frame = to_numpy(obs["sensor_data"][camera]["rgb"])[0]
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB from {camera}, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if frame.size and frame.max() <= 1.0:
            frame *= 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def policy_observation(obs: dict, prompt: str) -> dict:
    qpos = to_numpy(obs["agent"]["qpos"])[0]
    state = qpos[3:].astype(np.float32)
    if state.shape != (12,):
        raise ValueError(f"expected 12D qpos[3:] state, got {state.shape}")
    return {
        "observation/image": camera_frame(obs, CAMERAS[0]),
        "observation/hand_image": camera_frame(obs, CAMERAS[1]),
        "observation/side_image": camera_frame(obs, CAMERAS[2]),
        "observation/state": state,
        "prompt": prompt,
    }


def scalar_bool(value) -> bool:
    return bool(to_numpy(value).reshape(-1)[0])


def evaluate(args) -> tuple[int, int]:
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    hits = 0
    for episode in range(args.num_episodes):
        seed = args.start_seed + episode * args.seed_step
        env = gym.make(
            args.env_id,
            num_envs=1,
            obs_mode="rgb",
            render_mode="rgb_array",
            control_mode="pd_joint_pos",
            robot_uids="ds_fetch",
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
        )
        try:
            if tuple(env.action_space.shape) != (13,):
                raise ValueError(
                    f"expected 13D Fetch action space, got {env.action_space.shape}"
                )
            obs, _ = env.reset(seed=seed)
            policy.reset()
            success = False
            step = 0
            done = False

            while step < args.max_steps and not done:
                result = policy.infer(policy_observation(obs, args.prompt))
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.ndim == 3 and actions.shape[0] == 1:
                    actions = actions[0]
                if actions.ndim != 2 or actions.shape[1] != 13:
                    raise ValueError(
                        f"expected (horizon, 13) action chunk, got {actions.shape}"
                    )
                if not np.isfinite(actions).all():
                    raise ValueError("policy returned non-finite actions")

                for action in actions:
                    for _ in range(ACTION_REPEAT):
                        obs, _, terminated, truncated, info = env.step(action)
                        step += 1
                        success = success or scalar_bool(info["success"])
                        done = (
                            scalar_bool(terminated)
                            or scalar_bool(truncated)
                            or step >= args.max_steps
                        )
                        if done:
                            break
                    if done:
                        break

            hits += int(success)
            print(
                f"episode={episode + 1}/{args.num_episodes} seed={seed} "
                f"success={success} steps={step}",
                flush=True,
            )
        finally:
            env.close()

    print(f"success={hits}/{args.num_episodes}")
    return hits, args.num_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument(
        "--prompt", required=True, help="must match the training task instruction"
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--sim-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--render-backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if args.num_episodes < 1 or args.max_steps < 1 or args.port < 1:
        parser.error("num-episodes, max-steps, and port must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
