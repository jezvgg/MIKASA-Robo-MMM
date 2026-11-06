"""Closed-loop Fetch rollouts of a served policy on the six task datasets' environments.

    python -m utils.policies.evaluate --backend memer --root ~/datasets/memer --output out.json \
        --seeds-dir HF_SEEDS --episodes 100 [--tasks samedrawer takeitback] [--shard 0/4] [--video 2] [--resume]

Tasks are the dataset folders under --root, as utils/policies/data.py reads them: our four
(samedrawer, seasondish, cabinetsearch, depthrecall) and jezv's two (takeitback, veggie). The
env of a task is the one its dataset's meta/source_rlds_metadata.json records: ours from the
collection profile (kwargs, camera check and motion parameters of utils/collection/profile.py),
jezv's from his env kwargs, reset with reconfigure=True as his recording was. Every episode:
seed_everything(seed), a fresh env, reset(seed=seed); the prompt is the dataset's instruction.

Timing: policies run at the datasets' 10 Hz, the simulator at 20 Hz. A 10 Hz target becomes
two control steps: the target, then the average of it and the next target (--execution
midpoint; the gripper keeps its target; a chunk's last executed target averages with the
chunk's next one, and is held when the chunk has none) or the target twice (--execution hold).
The policy sees the frame after each pair. An episode ends at the first success, at a task
fail (terminated without success) or at the limit: twice the task dataset's longest episode,
4 x its 10 Hz frames in 20 Hz steps (--max-steps overrides).

Seeds: the validation seeds of all six tasks are on HF, MIKASA-Robo-MMM/MIKASA-Robo-MMM
<task>/validation_seeds.json; --seeds-dir reads that layout. --seeds TASK=FILE (a JSON list, or
a dict with "episode_seeds" or "seeds") sets or overrides one task. The first --episodes seeds
run; a task without a file runs --start-seed onward. Seeds of the dataset's own training
episodes are refused.

Backends:
  memer     MemER: the high level (vla/memer/serve_high.py at --high-port) names the subtask,
            the low level (`python -m mikasa_pi05 serve memer_lowlevel_4xh100` at --port) acts
            on it; the high level runs at every low-level query, as RoboMME's MemER eval
            (examples/robomme/eval.py) does
  robomme   RoboMME's pi0.5 memory variants (memory reset and history ingestion on the server)
  hamlet    HAMLET (GR00T), native memory stride 16
  recorded  the dataset's first --episodes on their own seeds with their own actions, played to
            the end and judged at the last frame (checks the envs, the seeds and the execution)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from utils.policies.data import CAMERAS, FetchDataset

ENVIRONMENTS = {"takeitback": "MyRoboCasa_TakeItBack-v1", "veggie": "MyRoboCasa_FridgeVeggies-v1"}
OUR_TASKS = ("samedrawer", "seasondish", "cabinetsearch", "depthrecall")
TASKS = OUR_TASKS + tuple(ENVIRONMENTS)
ACTION_REPEAT = 2
GRIPPER = 7  # utils/collection/contract.py ACTION_NAMES
EXECUTIONS = ("midpoint", "hold")
HIGH_AUTHKEY_ENV = "MEMER_HIGH_AUTHKEY"  # vla/memer/serve_high.py reads the same variable
REPO = Path(__file__).resolve().parents[2]


def finite_actions(value):
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != 13 or len(actions) == 0:
        raise ValueError(f"expected nonempty (H,13) actions, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise FloatingPointError("non-finite policy actions")
    return actions


def control_steps(actions, count, execution):
    """The 20 Hz control steps of a chunk's first `count` 10 Hz targets."""
    actions = finite_actions(actions)
    targets = actions[:count]
    if execution == "hold":
        return np.repeat(targets, ACTION_REPEAT, axis=0)
    if execution != "midpoint":
        raise ValueError(f"unknown execution {execution!r}")
    following = actions[1:len(targets) + 1]
    if len(following) < len(targets):  # no next target in the chunk: the last one is held
        following = np.concatenate([following, actions[-1:]])
    middle = (targets + following) * np.float32(0.5)
    middle[:, GRIPPER] = targets[:, GRIPPER]
    steps = np.empty((ACTION_REPEAT * len(targets), actions.shape[1]), dtype=np.float32)
    steps[0::2], steps[1::2] = targets, middle
    return steps


def episode_limit(lengths):
    """20 Hz steps of twice the longest 10 Hz episode."""
    return 2 * ACTION_REPEAT * max(lengths)


def read_seeds(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        value = value.get("episode_seeds", value.get("seeds"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a list of seeds or a dict with episode_seeds/seeds")
    seeds = [int(seed) for seed in value]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: duplicate seeds")
    return seeds


def scalar_bool(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def robomme_observation(obs, prompt):
    from utils.evaluate_openpi import policy_observation

    return policy_observation(obs, prompt)


def mikasa_observation(obs, prompt):
    from utils.collection.client import policy_observation

    return policy_observation(obs, prompt)


class TaskSource:
    """One task's dataset: instruction, episode lengths, training seeds and env recipe."""

    def __init__(self, root, task):
        self.task = task
        self.dataset = FetchDataset(root, tasks=(task,))
        texts = set(self.dataset.instructions.values())
        if len(texts) != 1:
            raise ValueError(f"{task}: expected one instruction, got {sorted(texts)}")
        self.instruction = texts.pop()
        self.lengths = [episode.length for episode in self.dataset.episodes]
        self.meta = json.loads((Path(root) / task / "meta/source_rlds_metadata.json").read_text())
        seeds = [int(seed) for seed in self.meta["episode_seeds"]]
        indices = self.meta.get("episode_indices") or list(range(len(seeds)))
        self.seed_of = dict(zip((int(index) for index in indices), seeds))
        self.training_seeds = set(seeds)

    @property
    def env_id(self):
        return self.meta["env_info"]["env_id"] if self.task in ENVIRONMENTS else self.meta["profile"]["env_id"]


def make_env(source, *, max_episode_steps, render_backend=None):
    """A fresh env of the task as its dataset was recorded, and the options of its reset."""
    import gymnasium as gym
    import my_scenes  # noqa: F401

    if source.task in ENVIRONMENTS:
        info = source.meta["env_info"]
        if info["env_id"] != ENVIRONMENTS[source.task]:
            raise ValueError(f"{source.task}: dataset env {info['env_id']}, expected {ENVIRONMENTS[source.task]}")
        kwargs = dict(info["env_kwargs"], render_backend=render_backend or "sapien_cuda")
        env = gym.make(info["env_id"], max_episode_steps=max_episode_steps, **kwargs)
        options = {"reconfigure": True}  # his recording's reset: without it every seed keeps one scene
    else:
        from utils.collection.profile import profile_env_kwargs, verify_env

        profile = source.meta["profile"]
        kwargs = profile_env_kwargs(profile, obs_mode="rgb")
        if render_backend not in (None, kwargs.get("render_backend")):
            raise ValueError("Rendering backend differs from the dataset's profile")
        env = gym.make(profile["env_id"], max_episode_steps=max_episode_steps, **kwargs)
        options = None
    try:
        if source.task not in ENVIRONMENTS:
            verify_env(env, profile)
            # The tasks read the planner's motion parameters at reset (profile.experiment_env).
            env.unwrapped.motion_parameters = dict(profile["planner"].get("motion_parameters", {}))
        action_space = getattr(env.unwrapped, "single_action_space", env.action_space)
        if tuple(action_space.shape) != (13,):
            raise ValueError("simulator does not expose the canonical 13D Fetch action")
        if env.unwrapped.control_freq != 20:
            raise ValueError("two control steps per target require 20-Hz simulator control")
    except Exception:
        env.close()
        raise
    return env, options


class RoboMMEClient:
    observation = staticmethod(robomme_observation)

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
    observation = staticmethod(robomme_observation)

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


def high_authkey() -> bytes:
    return os.environ.get(HIGH_AUTHKEY_ENV, "memer-high").encode()


def encode_frame(observation):
    """The cameras of one observation as raw bytes: no numpy pickle crosses the two venvs."""
    return {key: (np.ascontiguousarray(value).tobytes(), tuple(value.shape))
            for key, value in observation.items() if key.startswith("observation.images.")}


class MemerClient:
    """MemER: the high level names the current subtask from the frames, the low level acts on it.

    Every 10 Hz frame is kept and goes to the high level at the next query, which runs the model
    once, at the newest frame; until its first valid subtask the low level gets the instruction.
    """

    observation = staticmethod(mikasa_observation)

    def __init__(self, host, port, high_host, high_port):
        from multiprocessing.connection import Client
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        from utils.collection.client import policy_metadata

        self.low = WebsocketClientPolicy(host=host, port=port)
        low = self.low.get_server_metadata()
        if low.get("mikasa_data") != policy_metadata():
            raise ValueError("low-level server uses a different robot or timing contract")
        self.high = Client((high_host, high_port), authkey=high_authkey())
        self.metadata = {"low": {key: low[key] for key in ("checkpoint", "action_horizon", "prompt") if key in low},
                         "high": self.call("metadata")}
        self.frames = 0

    def call(self, command, **payload):
        self.high.send({"cmd": command, **payload})
        reply = self.high.recv()
        if "error" in reply:
            raise RuntimeError(f"MemER high level: {reply['error']}")
        return reply

    def reset(self, session):
        self.session, self.instruction, self.subtask = session, None, None
        self.pending, self.frames, self.trace = [], 0, []
        self.high_calls = self.parse_failures = self.instruction_prompts = 0
        self.memory = []

    def ingest(self, observation):
        if self.instruction is None:
            self.instruction = observation["prompt"]
            self.call("reset", instruction=self.instruction)
        self.pending.append(encode_frame(observation))
        self.frames += 1

    def infer(self, observation):
        reply = self.call("step", frames=self.pending)
        self.pending = []
        self.high_calls += 1
        self.memory = reply["memory_indices_after"]
        subtask = reply["current_subtask"]
        if not reply["parse_ok"] or not subtask:
            self.parse_failures += 1
        elif subtask != self.subtask:
            self.trace.append({"frame": reply["timestep"], "subtask": subtask})
            self.subtask = subtask
        if self.subtask is None:
            self.instruction_prompts += 1
        request = dict(observation, prompt=self.subtask or self.instruction)
        return finite_actions(self.low.infer(request)["actions"])

    def extra(self):
        return {"subtasks": self.trace, "high_calls": self.high_calls, "parse_failures": self.parse_failures,
                "instruction_prompts": self.instruction_prompts, "memory_frames": self.memory}


class RecordedClient:
    """A dataset episode's own 10 Hz actions, `horizon` per query, until they run out."""

    observation = staticmethod(mikasa_observation)

    def __init__(self, horizon):
        self.horizon, self.actions, self.frames = horizon, None, 0

    def load(self, actions):
        self.actions = finite_actions(actions)

    def reset(self, session):
        self.position, self.frames = 0, 0

    def ingest(self, observation):
        self.frames += 1

    def infer(self, observation):
        if self.position >= len(self.actions):
            return None
        chunk = self.actions[self.position:]  # the rest: a midpoint needs the next target
        self.position += self.horizon
        return chunk


def camera_frames(obs):
    def rgb(camera):
        value = obs["sensor_data"][camera]["rgb"]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)[0]

    return {camera: rgb(camera) for camera in CAMERAS}


def save_video(path, frames):
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in frames:
        hand = np.repeat(np.repeat(frame["fetch_hand"], 2, axis=0), 2, axis=1)
        rows.append(np.concatenate([frame["left_base_camera_link"], frame["right_base_camera_link"], hand], axis=1))
    imageio.mimsave(path, rows, fps=10, macro_block_size=1)


def run_episode(env, obs, client, prompt, *, max_steps, horizon, execution, stop_on_success=True, video=None):
    """Query, execute `horizon` targets of the chunk, query again; the policy sees every 10 Hz frame."""
    def observe(obs):
        if video is not None:
            video.append(camera_frames(obs))
        return client.observation(obs, prompt)

    client.ingest(observe(obs))
    steps = queries = 0
    success = success_once = False
    status, last_shape = None, None
    while status is None:
        actions = client.infer(client.observation(obs, prompt))
        if actions is None:  # a recorded episode has run out
            status = "replayed"
            break
        queries += 1
        last_shape = list(np.shape(actions))
        for index, control in enumerate(control_steps(actions, horizon, execution)):
            obs, _, terminated, truncated, info = env.step(control)
            steps += 1
            success = scalar_bool(info["success"])
            success_once |= success
            if stop_on_success and success:
                status = "success"
            elif stop_on_success and scalar_bool(terminated):
                status = "fail"
            elif scalar_bool(truncated) or steps >= max_steps:
                status = "timeout"
            if status:
                break
            if index % ACTION_REPEAT == ACTION_REPEAT - 1:
                client.ingest(observe(obs))
    return {"status": status, "success": success if not stop_on_success else success_once,
            "success_once": success_once, "simulator_steps": steps, "queries": queries,
            "last_action_shape": last_shape, "history_observations": client.frames}


def summarize(rollouts):
    tasks = {}
    for rollout in rollouts:
        task = tasks.setdefault(rollout["task"], {"episodes": 0, "successes": 0, "status": {}})
        task["episodes"] += 1
        task["successes"] += int(rollout["success"])
        task["status"][rollout["status"]] = task["status"].get(rollout["status"], 0) + 1
    for task in tasks.values():
        task["success_rate"] = task["successes"] / task["episodes"]
    return tasks


def code_provenance():
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def task_seeds(args):
    files = {}
    if args.seeds_dir:
        for task in TASKS:
            if (path := args.seeds_dir / task / "validation_seeds.json").exists():
                files[task] = path
    for item in args.seeds:
        task, _, path = item.partition("=")
        if task not in TASKS or not path:
            raise ValueError(f"--seeds expects TASK=FILE with TASK in {TASKS}, got {item!r}")
        files[task] = Path(path)
    return files


def plan_rollouts(args, sources):
    """(task, seed, dataset episode or None) for every rollout, and each task's settings.

    Each task's seeds are dealt to the shards in turn (a task starting one shard after the
    previous task), and within a shard the tasks take turns, so every shard and every partial
    result covers all of them in equal parts.
    """
    files = task_seeds(args)
    tasks, settings = {}, {}
    for task, source in sources.items():
        plan = tasks[task] = []
        max_steps = args.max_steps or episode_limit(source.lengths)
        settings[task] = {"env_id": source.env_id, "instruction": source.instruction,
                          "dataset_episodes": len(source.lengths), "longest_dataset_frames": max(source.lengths),
                          "max_steps": max_steps,
                          "max_steps_rule": "--max-steps" if args.max_steps else "2 x longest dataset episode"}
        if args.backend == "recorded":
            episodes = source.dataset.episodes[:args.episodes]
            plan += [(task, source.seed_of[episode.source_index], episode) for episode in episodes]
            settings[task].update(seeds="the dataset's own", episodes=len(plan))
            continue
        if task in files:
            seeds = read_seeds(files[task])
            if len(seeds) < args.episodes:
                raise ValueError(f"{task}: {files[task]} holds {len(seeds)} seeds, --episodes is {args.episodes}")
            seeds, settings[task]["seeds"] = seeds[:args.episodes], str(files[task].resolve())
        else:
            seeds = list(range(args.start_seed, args.start_seed + args.episodes))
            settings[task]["seeds"] = f"--start-seed {args.start_seed}"
        overlap = sorted(set(seeds) & source.training_seeds)
        if overlap:
            raise ValueError(f"{task}: {len(overlap)} seeds of training episodes, e.g. {overlap[:5]}")
        plan += [(task, seed, None) for seed in seeds]
        settings[task]["episodes"] = len(plan)
    shard, shards = map(int, args.shard.split("/"))
    mine = [[item for index, item in enumerate(items) if (index + offset) % shards == shard]
            for offset, items in enumerate(tasks.values())]
    return [item for turn in itertools.zip_longest(*mine) for item in turn if item is not None], settings


def make_client(args):
    if args.backend == "memer":
        return MemerClient(args.host, args.port, args.high_host, args.high_port)
    if args.backend == "recorded":
        return RecordedClient(args.execution_horizon)
    return (RoboMMEClient if args.backend == "robomme" else HamletClient)(args.host, args.port)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def evaluate(args):
    from utils.mikasa.seeding import seed_everything

    if args.backend == "hamlet" and args.execution_horizon != 16:
        raise ValueError("native HAMLET memory_stride=16 requires execution_horizon=16")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite rollout evidence: {args.output}")
    if workspace := os.environ.get("POLICY_WORKSPACE"):
        from utils.policies.workspace import contained, require_space
        contained(Path(workspace), args.output)
        require_space(Path(workspace), 2_000_000_000)
    tasks = args.tasks or [task for task in TASKS if (args.root / task / "meta/info.json").exists()]
    sources = {task: TaskSource(args.root, task) for task in tasks}
    plan, settings = plan_rollouts(args, sources)
    header = {"backend": args.backend, "execution": args.execution, "execution_horizon": args.execution_horizon,
              "stop_on_success": args.backend != "recorded", "shard": args.shard, "root": str(args.root.resolve()),
              "tasks": settings}
    results = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if {key: previous.get(key) for key in header} != header:
            raise ValueError(f"{args.output} was written with other settings; not resuming into it")
        results = previous["rollouts"]
    done = {(result["task"], result["seed"]) for result in results}
    client = make_client(args)
    header["code"], header["client"] = code_provenance(), getattr(client, "metadata", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    videos = {task: sum(result["task"] == task for result in results) for task in tasks}
    for task, seed, episode in plan:
        if (task, seed) in done:
            continue
        source = sources[task]
        session = f"{args.backend}:{task}:{args.output.stem}:{seed}"
        if episode is not None:
            client.load(source.dataset.episode_rows(episode.global_index)["action"])
        frames = [] if videos[task] < args.video else None
        videos[task] += 1
        seed_everything(seed)
        env, options = make_env(source, max_episode_steps=settings[task]["max_steps"],
                                render_backend=args.render_backend)
        start = time.time()
        try:
            obs, _ = env.reset(seed=seed, options=options)
            client.reset(session)
            outcome = run_episode(env, obs, client, source.instruction, max_steps=settings[task]["max_steps"],
                                  horizon=args.execution_horizon, execution=args.execution,
                                  stop_on_success=args.backend != "recorded", video=frames)
        finally:
            env.close()
        result = {"task": task, "env_id": source.env_id, "seed": seed, "session": session,
                  **({"episode_index": episode.source_index} if episode is not None else {}),
                  **outcome, "seconds": round(time.time() - start, 1), "real_reset_requested": True,
                  "control_hz": 20, "data_hz": 10, "action_repeat": ACTION_REPEAT,
                  "history_stride_data_steps": 16 if args.backend == "hamlet" else 1,
                  **(client.extra() if hasattr(client, "extra") else {})}
        if frames is not None:
            video = args.output.with_name(f"{args.output.stem}-videos") / f"{task}-{seed}.mp4"
            save_video(video, frames)
            result["video"] = str(video)
        results.append(result)
        write_json(args.output, {**header, "rollouts": results, "summary": summarize(results)})
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("memer", "robomme", "hamlet", "recorded"), required=True)
    parser.add_argument("--root", required=True, type=Path, help="one dataset folder per task")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, help="default: every task folder under --root")
    parser.add_argument("--seeds-dir", type=Path, help="<task>/validation_seeds.json per task, as on HF")
    parser.add_argument("--seeds", action="append", default=[], metavar="TASK=FILE")
    parser.add_argument("--episodes", type=int, default=2, help="per task, allowing cross-episode reset tests")
    parser.add_argument("--start-seed", type=int, default=0, help="for a task without --seeds")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="the policy server (MemER: the low level)")
    parser.add_argument("--high-host", default="127.0.0.1")
    parser.add_argument("--high-port", type=int, default=8100, help="MemER's high level")
    parser.add_argument("--execution", choices=EXECUTIONS, default="midpoint")
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--max-steps", type=int, help="20 Hz steps; default twice the dataset's longest episode")
    parser.add_argument("--render-backend", help="default: the dataset's profile (ours), sapien_cuda (jezv's)")
    parser.add_argument("--shard", default="0/1", help="k/n: every n-th rollout from the k-th")
    parser.add_argument("--video", type=int, default=0, help="save the first N episodes of each task")
    parser.add_argument("--resume", action="store_true", help="continue an --output written with the same settings")
    args = parser.parse_args()
    if min(args.execution_horizon, args.episodes, args.port, args.high_port) <= 0 or (
            args.max_steps is not None and args.max_steps <= 0):
        parser.error("horizon, episodes, steps and ports must be positive")
    shard, _, shards = args.shard.partition("/")
    if not (shard.isdigit() and shards.isdigit() and int(shard) < int(shards)):
        parser.error("--shard expects k/n with 0 <= k < n")
    evaluate(args)


if __name__ == "__main__":
    main()
