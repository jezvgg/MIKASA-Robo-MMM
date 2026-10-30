"""Short real Fetch rollouts with explicit memory reset and 10-Hz ingestion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from utils.policies.data import CAMERAS, FetchDataset

ENVIRONMENTS = {"takeitback": "MyRoboCasa_TakeItBack-v1", "veggie": "MyRoboCasa_FridgeVeggies-v1"}
ACTION_REPEAT = 2


def finite_actions(value):
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != 13 or len(actions) == 0:
        raise ValueError(f"expected nonempty (H,13) actions, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise FloatingPointError("non-finite policy actions")
    return actions


class RoboMMEClient:
    def __init__(self, host, port):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.frames = 0

    def reset(self, session):
        # Upstream openpi-client.reset() is a no-op: use the fork's wire command.
        reply = self.client.infer({"reset": True})
        if reply.get("reset_finished") is not True:
            raise RuntimeError(f"server did not confirm real memory reset: {reply}")
        self.frames = 0

    def ingest(self, observation):
        reply = self.client.infer({"add_buffer": True,
            "images": observation["observation/image"][None, None],
            "state": observation["observation/state"][None]})
        if reply.get("add_buffer_finished") is not True:
            raise RuntimeError("server did not confirm history ingestion")
        self.frames += 1

    def infer(self, observation):
        return finite_actions(self.client.infer(observation)["actions"])


class HamletClient:
    def __init__(self, host, port):
        from gr00t.policy.server_client import PolicyClient
        import zmq
        self.client = PolicyClient(host=host, port=port, timeout_ms=180000)
        self.client.socket.setsockopt(zmq.RCVTIMEO, 180000)
        self.client.socket.setsockopt(zmq.SNDTIMEO, 180000)
        self.client.socket.setsockopt(zmq.LINGER, 0)

    def reset(self, session):
        # Memory is keyed by session; reset_memory on the first actual query.
        reply = self.client.reset(options={"session_id": session})
        if not reply.get("memory_reset") or reply.get("remaining_sessions") != 0:
            raise RuntimeError("HAMLET server did not clear session caches")
        self.session, self.first, self.frames = session, True, 0

    def ingest(self, observation):
        # Native HAMLET caches one observation per policy query (stride 16).
        pass

    def infer(self, observation):
        images = [observation[key] for key in (
            "observation/image", "observation/hand_image", "observation/side_image")]
        raw = {"video": {name: image[None, None] for name, image in zip(CAMERAS, images)},
               "state": {"fetch_state": observation["observation/state"][None, None]},
               "language": {"instruction": [[observation["prompt"]]]}}
        actions, info = self.client.get_action(raw, options={
            "session_ids": [self.session], "reset_memory": [self.first]})
        self.first = False
        self.frames += 1
        return finite_actions(actions["fetch_action"])


def evaluate(args):
    import gymnasium as gym
    import my_scenes  # noqa: F401
    from utils.evaluate_openpi import policy_observation, scalar_bool

    if args.backend == "hamlet" and args.execution_horizon != 16:
        raise ValueError("native HAMLET memory_stride=16 requires execution_horizon=16")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite rollout evidence: {args.output}")
    if workspace := os.environ.get("POLICY_WORKSPACE"):
        from utils.policies.workspace import contained, require_space
        contained(Path(workspace), args.output)
        require_space(Path(workspace), 2_000_000_000)
    source = FetchDataset(args.root)
    client = (RoboMMEClient if args.backend == "robomme" else HamletClient)(args.host, args.port)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task in source.tasks:
        prompt = source.instructions[(task, 0)]
        for episode in range(args.episodes):
            seed = args.start_seed + episode
            session = f"{args.backend}:{task}:{args.output.stem}:{seed}"
            env = gym.make(ENVIRONMENTS[task], num_envs=1, obs_mode="rgb",
                render_mode="rgb_array", control_mode="pd_joint_pos", robot_uids="ds_fetch",
                sim_backend="physx_cpu", render_backend=args.render_backend, assisted_grasp=True)
            try:
                action_space = getattr(env.unwrapped, "single_action_space", env.action_space)
                if tuple(action_space.shape) != (13,):
                    raise ValueError("simulator does not expose the canonical 13D Fetch action")
                if env.unwrapped.control_freq != 20:
                    raise ValueError("ACTION_REPEAT=2 requires 20-Hz simulator control")
                obs, _ = env.reset(seed=seed)
                client.reset(session)
                client.ingest(policy_observation(obs, prompt))
                step, done, success, queries = 0, False, False, 0
                last_shape = None
                while step < args.max_steps and not done:
                    actions = client.infer(policy_observation(obs, prompt))
                    queries += 1
                    last_shape = list(actions.shape)
                    for action in actions[:args.execution_horizon]:
                        repeats = 0
                        for _ in range(ACTION_REPEAT):
                            obs, _, terminated, truncated, info = env.step(action)
                            step += 1
                            repeats += 1
                            success |= scalar_bool(info["success"])
                            done = scalar_bool(terminated) or scalar_bool(truncated) or step >= args.max_steps
                            if done:
                                break
                        if repeats == ACTION_REPEAT:
                            client.ingest(policy_observation(obs, prompt))
                        if done:
                            break
                result = {"task": task, "env_id": ENVIRONMENTS[task], "seed": seed,
                          "simulator_steps": step, "queries": queries, "last_action_shape": last_shape,
                          "history_observations": client.frames, "session": session,
                          "real_reset_requested": True, "success_once": success,
                          "control_hz": 20, "data_hz": 10, "action_repeat": ACTION_REPEAT,
                          "history_stride_data_steps": 16 if args.backend == "hamlet" else 1}
                results.append(result)
                args.output.write_text(json.dumps({"backend": args.backend, "rollouts": results}, indent=2))
                print(json.dumps(result), flush=True)
            finally:
                env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("robomme", "hamlet"), required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=2, help="per task, allowing cross-episode reset tests")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--render-backend", default="sapien_cuda")
    args = parser.parse_args()
    if min(args.execution_horizon, args.episodes, args.max_steps, args.port) <= 0:
        parser.error("horizon, episodes, steps and port must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
