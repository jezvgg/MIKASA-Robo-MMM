"""Closed-loop SameDrawer evaluation of a served pi0.5 policy with the first-frame input.

Runs in the simulator venv from the repository root:

    PYTHONPATH=. python vla/pi05_first_frame/eval_samedrawer.py \
        --server 127.0.0.1:8000 --out results/run1 [--seeds validation] [--shard 0/4]

Every episode: seed_everything(seed), env.reset(seed=seed); the policy wrapper keeps
the observation returned by reset and sends that camera image as
`observation.first_frame` with every request. Actions run through the repository's
own `run_policy_episode` (10 Hz targets, each held for two 20 Hz control steps).

    --first-frame black   ablation: send a black image instead of the real first frame
    --policy recorded     replay the dataset's own 10 Hz actions through the same loop
                          (needs --seeds train:N); checks seeds, timing and execution
    --summarize DIR...    merge episodes.jsonl files (e.g. shards) into one summary

Results: <out>/episodes.jsonl (one line per seed, resumable), <out>/summary.json,
<out>/videos/<seed>.mp4 for the first --video episodes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Signatures of the code and engine that produced the dataset and the validation seeds
# (mikasa-runs/collection-2026-09-28/same_drawer-validation-001/run.json).
EXPECTED_CODE_SHA256 = "a39d1c4297901b185be6d5c590212f240fdfaed8fe2080ea5eecbf0c0c36417b"
EXPECTED_ENGINE_SHA256 = "74ecf1d0b2eab020567939c0cec3c170623af4c0c46b9dcfab8ad7c6602dace2"
DEFAULT_HORIZON_STEPS = 800  # 1600 control steps at 20 Hz = the task's max_episode_steps
VIDEO_CAMERAS = ("left_base_camera_link", "right_base_camera_link", "fetch_hand")


def load_profile_with_torch_build():
    """The repository's load_profile, tolerating only torch's local CUDA build tag.

    The profile pins torch==2.14.0. The PyPI build targets CUDA 13 (driver >= 580); on
    older drivers setup_server.sh installs the same release built for CUDA 12.6, whose
    version reads 2.14.0+cu126. Physics runs on the CPU; torch only receives rendered
    pixels from the GPU, so the build tag does not change the simulation.
    """
    from utils.collection import profile as profile_module

    real_version = importlib.metadata.version

    def version(name):
        value = real_version(name)
        return value.split("+")[0] if name == "torch" else value

    importlib.metadata.version = version
    try:
        return profile_module.load_profile("same_drawer")
    finally:
        importlib.metadata.version = real_version


def check_torch_cuda():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            f"torch {torch.__version__} cannot use CUDA with this driver; the sapien_cuda renderer hands "
            "camera images to torch on the GPU. Install torch 2.14.0 built for this driver "
            "(scripts/setup_server.sh picks +cu126 when the driver is older than 580)."
        )


class FirstFramePolicy:
    """Adds the reset observation's image to every request; reset() before each episode."""

    def __init__(self, policy, mode: str = "reset"):
        self.policy = policy
        self.mode = mode
        meta = policy.get_server_metadata()
        self.key = meta["first_frame"]["key"]
        self.camera = meta["first_frame"]["camera"]
        self.first = None
        self.requests = 0

    def get_server_metadata(self):
        return self.policy.get_server_metadata()

    def reset(self):
        self.first = None
        self.requests = 0

    def infer(self, observation: dict) -> dict:
        if self.first is None:
            frame = observation[f"observation.images.{self.camera}"]
            self.first = np.zeros_like(frame) if self.mode == "black" else np.array(frame, copy=True)
        self.requests += 1
        return self.policy.infer({**observation, self.key: self.first})


class RecordedPolicy:
    """Serves one recorded episode's 10 Hz actions in order (no model)."""

    def __init__(self, metadata: dict, execute_horizon: int):
        self.metadata = metadata
        self.execute_horizon = execute_horizon
        self.actions = None
        self.cursor = 0

    def get_server_metadata(self):
        return self.metadata

    def load(self, actions: np.ndarray):
        self.actions, self.cursor = np.asarray(actions, dtype=np.float32), 0

    def infer(self, observation: dict) -> dict:
        chunk = self.actions[self.cursor:self.cursor + self.execute_horizon]
        if not len(chunk):
            chunk = self.actions[-1:]
        self.cursor += self.execute_horizon
        return {"actions": chunk}


def make_recording_env(env):
    """Records drawer openings at every control step, and camera frames while `capture` is set."""
    import gymnasium as gym

    class Recorder(gym.Wrapper):
        capture = False

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.frames, self.max_open, self.steps, self.qpos = [], None, 0, []
            self._capture(obs)
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.steps += 1
            amounts = self.env.unwrapped.drawer_open_amounts()[0].cpu().numpy()
            self.max_open = amounts if self.max_open is None else np.maximum(self.max_open, amounts)
            if self.steps % 2 == 0:
                self._capture(obs)
            return obs, reward, terminated, truncated, info

        def _capture(self, obs):
            # One entry per 10 Hz policy frame, aligned with the dataset's frame_index.
            self.qpos.append(_np(self.env.unwrapped.agent.robot.get_qpos())[0].astype(np.float64))
            if self.capture:
                self.frames.append({c: _np(obs["sensor_data"][c]["rgb"])[0] for c in VIDEO_CAMERAS})

    return Recorder(env)


def _np(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def connect_policy_server(client_module, host: str, port: int, timeout: float = 1800):
    """Wait for a server that may still be loading and compiling.

    openpi's client retries only on ConnectionRefusedError. `localhost` can resolve to ::1
    first, and in a container without IPv6 that attempt fails with EAFNOSUPPORT, which
    socket.create_connection reports instead of the IPv4 refusal; any OSError means "not
    up yet" here. Prefer 127.0.0.1 to avoid the IPv6 attempt altogether.
    """
    deadline = time.time() + timeout
    while True:
        try:
            return client_module.WebsocketClientPolicy(host, port)
        except OSError as error:
            if time.time() > deadline:
                raise
            print(f"waiting for the policy server at {host}:{port} ({error})", flush=True)
            time.sleep(5)


def episode_record(env, seed: int, result: dict, elapsed: float, reference=None) -> dict:
    base = env.unwrapped
    flags = {name: bool(_np(getattr(base, name))[0]) for name in (
        "closed_done", "apple_done", "wrong_drawer_touched", "sequence_violated", "apple_retention_violated")}
    final_open = _np(base.drawer_open_amounts())[0]
    return {
        "seed": seed,
        "target_drawer": int(_np(base.target_drawer)[0]),
        "success": bool(result["success"]),
        "status": result["status"],
        "control_steps": int(result["control_steps"]),
        "policy_requests": int(result["policy_requests"]),
        "clipped_targets": int(result["clipped_targets"]),
        **flags,
        "final_open_amounts": [round(float(x), 4) for x in final_open],
        "max_open_amounts": [round(float(x), 4) for x in (env.max_open if env.max_open is not None else final_open)],
        "seconds": round(elapsed, 1),
        **({"tracking": tracking(env.qpos, reference)} if reference is not None else {}),
    }


def save_video(path: Path, frames: list[dict], first: np.ndarray | None):
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in frames:
        hand = np.repeat(np.repeat(frame["fetch_hand"], 2, axis=0), 2, axis=1)
        panels = [frame["left_base_camera_link"], frame["right_base_camera_link"], hand]
        if first is not None:
            panels = [first] + panels
        rows.append(np.concatenate(panels, axis=1))
    imageio.mimsave(path, rows, fps=10, macro_block_size=1)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(records: list[dict]) -> dict:
    n = len(records)
    ok = sum(r["success"] for r in records)
    lo, hi = wilson(ok, n)
    by_drawer = {}
    for d in sorted({r["target_drawer"] for r in records}):
        rs = [r for r in records if r["target_drawer"] == d]
        by_drawer[str(d)] = {"n": len(rs), "success": sum(r["success"] for r in rs)}

    def opened_after_apple(r):
        # Drawers opened beyond 5 cm; the target must be among them for a correct choice.
        return [i for i, x in enumerate(r["max_open_amounts"]) if x > 0.05]

    stages = {
        "closed_cue_drawer": sum(r["closed_done"] for r in records),
        "apple_placed": sum(r["apple_done"] for r in records),
        "success": ok,
        "wrong_drawer_touched": sum(r["wrong_drawer_touched"] for r in records),
        "sequence_violated": sum(r["sequence_violated"] for r in records),
        "apple_knocked_off": sum(r["apple_retention_violated"] for r in records),
        "apple_placed_and_target_reopened": sum(
            r["apple_done"] and r["final_open_amounts"][r["target_drawer"]] >= 0.10 for r in records),
        "apple_placed_and_other_drawer_opened": sum(
            r["apple_done"] and any(i != r["target_drawer"] for i in opened_after_apple(r)) for r in records),
    }
    tracked = [r["tracking"]["followed_demo_for_s"] for r in records if "tracking" in r]
    return {
        "episodes": n,
        "success": ok,
        **({"followed_demo_for_s": tracked} if tracked else {}),
        "success_rate": ok / n if n else 0.0,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "by_target_drawer": by_drawer,
        "stages": stages,
        "status": {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})},
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_seeds(spec: str, dataset_dir: Path) -> tuple[list[int], dict[int, int]]:
    """Returns the seeds and, for training episodes, seed -> episode index."""
    if spec == "validation":
        return [int(s) for s in json.loads((dataset_dir / "validation_seeds.json").read_text())["seeds"]], {}
    if spec.startswith(("train:", "episodes:")):
        import pyarrow.parquet as pq

        pairs = []
        for path in sorted((dataset_dir / "meta/episodes").glob("*/*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "episode_seed"]).to_pydict()
            pairs += zip(table["episode_index"], table["episode_seed"])
        pairs = sorted(pairs)
        if spec.startswith("train:"):
            pairs = pairs[: int(spec.split(":", 1)[1])]
        else:
            wanted = [int(e) for e in spec.split(":", 1)[1].split(",")]
            seed_of = dict(pairs)
            pairs = [(e, seed_of[e]) for e in wanted]
        return [int(s) for _, s in pairs], {int(s): int(e) for e, s in pairs}
    if Path(spec).is_file():
        data = json.loads(Path(spec).read_text())
        return [int(s) for s in (data["seeds"] if isinstance(data, dict) else data)], {}
    return [int(s) for s in spec.split(",")], {}


ARM_STATE_DIMS = [2, 4, 5, 6, 7, 8, 9]  # arm joints inside observation.state (qpos[3:])
TRACK_ARM_RAD = 0.10   # ~6 degrees on any arm joint
TRACK_BASE_M = 0.10    # 10 cm of base position


def recorded_states(dataset_dir: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """The recorded observation.state (qpos[3:]) and global_state (base x, y, yaw) per 10 Hz frame."""
    import pyarrow.parquet as pq

    for path in sorted((dataset_dir / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "global_state"],
                              filters=[("episode_index", "=", episode)]).to_pydict()
        if table["episode_index"]:
            order = np.argsort(table["frame_index"])
            return (np.asarray(table["observation.state"], dtype=np.float64)[order],
                    np.asarray(table["global_state"], dtype=np.float64)[order])
    raise ValueError(f"Episode {episode} not found")


def tracking(live_qpos: list, recorded: tuple[np.ndarray, np.ndarray]) -> dict:
    """How long the live run stays on the recorded trajectory of the same seed."""
    state, base = recorded
    live = np.stack(live_qpos)
    n = min(len(live), len(state))
    arm = np.abs(live[:n, 3:][:, ARM_STATE_DIMS] - state[:n][:, ARM_STATE_DIMS]).max(axis=1)
    drift = np.linalg.norm(live[:n, :2] - base[:n, :2], axis=1)
    off = np.flatnonzero((arm > TRACK_ARM_RAD) | (drift > TRACK_BASE_M))
    at = {f"{t}s": [round(float(arm[t * 10]), 3), round(float(drift[t * 10]), 3)]
          for t in (1, 2, 5, 10, 20, 30, 45) if t * 10 < n}
    return {
        "followed_demo_for_s": round(float(off[0]) / 10, 1) if len(off) else round(n / 10, 1),
        "demo_length_s": round(len(state) / 10, 1),
        "max_arm_err_rad_and_base_err_m_at": at,
    }


def recorded_actions(dataset_dir: Path, episode: int) -> np.ndarray:
    import pyarrow.parquet as pq

    for path in sorted((dataset_dir / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action"],
                              filters=[("episode_index", "=", episode)]).to_pydict()
        if table["episode_index"]:
            order = np.argsort(table["frame_index"])
            return np.asarray(table["action"], dtype=np.float32)[order]
    raise ValueError(f"Episode {episode} not found")


def environment_info() -> dict:
    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES"),
    }
    try:
        info["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False).stdout.strip().splitlines()
    except FileNotFoundError:
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="127.0.0.1:8000", help="host:port of `python -m mikasa_pi05 serve`")
    parser.add_argument("--out", type=Path, help="results directory")
    parser.add_argument("--dataset-dir", type=Path, default=Path(os.environ.get("SAMEDRAWER_DATASET_DIR", ".")))
    parser.add_argument("--seeds", default="validation",
                        help="validation | train:N (first N episodes) | episodes:I,J,... | seed list | json file")
    parser.add_argument("--shard", default="0/1", help="i/n: evaluate every n-th seed starting at i")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--replan-steps", type=int, default=5, help="10 Hz actions executed per request")
    parser.add_argument("--max-policy-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--first-frame", choices=("reset", "black"), default="reset")
    parser.add_argument("--policy", choices=("server", "recorded"), default="server")
    parser.add_argument("--video", type=int, default=0, help="save videos of the first N episodes")
    parser.add_argument("--allow-signature-mismatch", action="store_true")
    parser.add_argument("--summarize", type=Path, nargs="+", help="merge result directories and exit")
    args = parser.parse_args()

    if args.summarize:
        records = {r["seed"]: r for d in args.summarize for r in read_jsonl(d / "episodes.jsonl")}
        print(json.dumps(summarize(sorted(records.values(), key=lambda r: r["seed"])), indent=2))
        return
    if args.out is None:
        parser.error("--out is required")

    check_torch_cuda()
    from utils.collection.client import policy_metadata, run_policy_episode
    from utils.collection.profile import make_env, runtime_signature
    from utils.mikasa.seeding import seed_everything

    profile = load_profile_with_torch_build()
    signature = runtime_signature(profile)
    mismatch = []
    if signature["code_sha256"] != EXPECTED_CODE_SHA256:
        mismatch.append(f"code {signature['code_sha256'][:12]} != {EXPECTED_CODE_SHA256[:12]}")
    if signature["engine_sha256"] != EXPECTED_ENGINE_SHA256:
        mismatch.append(f"engine {signature['engine_sha256'][:12]} != {EXPECTED_ENGINE_SHA256[:12]}")
    if mismatch and not args.allow_signature_mismatch:
        raise SystemExit("Simulator differs from the one that recorded the dataset: " + "; ".join(mismatch))

    seeds, episode_of_seed = resolve_seeds(args.seeds, args.dataset_dir)
    shard, shards = (int(x) for x in args.shard.split("/"))
    seeds = seeds[shard::shards][: args.max_episodes]

    if args.policy == "recorded":
        if not episode_of_seed:
            parser.error("--policy recorded needs --seeds train:N or episodes:I,J,...")
        recorded = RecordedPolicy(
            {"mikasa_data": policy_metadata(),
             "first_frame": {"key": "observation.first_frame", "camera": "left_base_camera_link", "frame": 0}},
            args.replan_steps)
        policy = FirstFramePolicy(recorded, args.first_frame)
    else:
        from openpi_client import websocket_client_policy

        host, port = args.server.rsplit(":", 1)
        if host in ("localhost", "127.0.0.1", "::1"):
            # websockets >= 15 routes through http(s)_proxy from the environment, local
            # addresses included; a lab proxy refuses them and the client fails at once.
            for name in ("no_proxy", "NO_PROXY"):
                os.environ[name] = ",".join(filter(None, [os.environ.get(name), "localhost", "127.0.0.1", "::1"]))
        policy = FirstFramePolicy(connect_policy_server(websocket_client_policy, host, int(port)), args.first_frame)
    server_meta = policy.get_server_metadata()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps({
        "argv": sys.argv, "seeds": seeds, "server_metadata": server_meta,
        "code_sha256": signature["code_sha256"], "engine_sha256": signature["engine_sha256"],
        "signature_mismatch": mismatch, "environment": environment_info(),
    }, indent=2, default=str))
    log = args.out / "episodes.jsonl"
    done = {r["seed"] for r in read_jsonl(log)}

    env = make_recording_env(make_env({"signature": signature, "purpose": "validation"}, rgb=True))
    instruction = env.unwrapped.get_language_instruction()[0]
    tasks_file = args.dataset_dir / "meta/tasks.parquet"
    if tasks_file.exists():
        import pyarrow.parquet as pq

        trained_on = pq.read_table(tasks_file).to_pandas().index.tolist()
        if instruction not in trained_on:
            raise SystemExit(f"Env instruction differs from the dataset task: {instruction!r} vs {trained_on}")
    try:
        for number, seed in enumerate(seeds):
            if seed in done:
                continue
            start = time.time()
            env.capture = number < args.video
            seed_everything(seed)
            obs, _ = env.reset(seed=seed)
            policy.reset()
            if args.policy == "recorded":
                recorded.load(recorded_actions(args.dataset_dir, episode_of_seed[seed]))
            result = run_policy_episode(
                env, policy, obs, instruction,
                max_policy_steps=args.max_policy_steps,
                execute_horizon=args.replan_steps,
                stop_on_success=True,
                clip_actions=args.policy == "server",
            )
            reference = recorded_states(args.dataset_dir, episode_of_seed[seed]) if seed in episode_of_seed else None
            record = episode_record(env, seed, result, time.time() - start, reference)
            with log.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if number < args.video:
                save_video(args.out / "videos" / f"{seed}.mp4", env.frames, policy.first)
    finally:
        env.close()

    summary = summarize(read_jsonl(log))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
