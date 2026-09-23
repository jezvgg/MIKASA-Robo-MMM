#!/usr/bin/env python3
"""Streaming ManiSkill HDF5 -> LeRobot v3.0 converter.

Same output layout as mani_skill.trajectory.convert_to_lerobot. RGB is encoded
per episode; dataframes are retained until chunked Parquet output. Exact action
and state quantiles are computed from Parquet afterward, using O(frames * dims)
float32 memory.

Usage:
    uv run python -m utils.convert_to_lerobot_stream --traj-path M.h5 \
        --output-dir DIR --task-name "..." --fps 10
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from mani_skill.trajectory.convert_to_lerobot import (
    create_video_from_frames,
    process_episode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
QUANTILES = (
    (0.01, "q01"),
    (0.10, "q10"),
    (0.50, "q50"),
    (0.90, "q90"),
    (0.99, "q99"),
)


@dataclass
class Args:
    traj_path: str
    output_dir: str
    fps: int = 10
    task_name: str = "Unknown task"
    chunks_size: int = 100
    robot_type: str = "ds_fetch"


def iter_episodes(h5_file: Path):
    with h5py.File(h5_file, "r") as f:
        keys = sorted(
            (k for k in f.keys() if k.startswith("traj_")),
            key=lambda k: int(k[5:]),
        )
        first = f[keys[0]]
        rgb_cameras = []
        if "obs/sensor_data" in first:
            for cam in first["obs/sensor_data"]:
                if "rgb" in first[f"obs/sensor_data/{cam}"]:
                    rgb_cameras.append(cam)
        state_dim = None
        flat_state = False
        if "obs/agent" in first and "qpos" in first["obs/agent"]:
            state_dim = first["obs/agent"]["qpos"].shape[1]
        elif "obs" in first and isinstance(first["obs"], h5py.Dataset):
            # obs_mode="state" is stored as one flattened (T+1, D) dataset.
            state_dim = first["obs"].shape[1]
            flat_state = True
        for key in keys:
            traj = f[key]
            ep = {"actions": traj["actions"][:]}
            for cam in rgb_cameras:
                ep[f"rgb_{cam}"] = traj[f"obs/sensor_data/{cam}/rgb"][: len(ep["actions"])]
            if state_dim:
                if flat_state:
                    ep["robot_state"] = traj["obs"][: len(ep["actions"])]
                else:
                    ep["robot_state"] = traj["obs/agent/qpos"][: len(ep["actions"])]
            for name in ("rewards", "success", "terminated", "truncated"):
                if name in traj:
                    ep[name] = traj[name][:]
            yield ep, rgb_cameras, state_dim


class Moments:
    """Running mean/std/min/max accumulator."""

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0
        self.min = None
        self.max = None

    def add(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        self.n += int(x.size)
        self.s += float(x.sum())
        self.s2 += float(np.square(x).sum())
        lo, hi = float(x.min()), float(x.max())
        self.min = lo if self.min is None else min(self.min, lo)
        self.max = hi if self.max is None else max(self.max, hi)

    def summary(self):
        mean = self.s / self.n
        std = float(np.sqrt(max(self.s2 / self.n - mean * mean, 0.0)))
        return {"mean": mean, "std": std, "min": self.min, "max": self.max, "n": self.n}


def scalar_stats_float(s, n):
    return {"mean": [s["mean"]], "std": [s["std"]],
            "max": [float(s["max"])], "min": [float(s["min"])], "count": [n]}


def scalar_stats_int(s, n):
    return {"mean": [s["mean"]], "std": [s["std"]],
            "max": [int(s["max"])], "min": [int(s["min"])], "count": [n]}


def vec_stats(moms, dim):
    return {
        "mean": [moms[i].summary()["mean"] for i in range(dim)],
        "std": [moms[i].summary()["std"] for i in range(dim)],
        "max": [moms[i].summary()["max"] for i in range(dim)],
        "min": [moms[i].summary()["min"] for i in range(dim)],
        "count": [moms[0].summary()["n"] // dim],
    }


def quantile_stats(values: np.ndarray) -> dict[str, list[float]]:
    quantiles = np.quantile(
        values, [q for q, _ in QUANTILES], axis=0, method="linear"
    )
    return {name: quantiles[i].tolist() for i, (_, name) in enumerate(QUANTILES)}


def load_trajectory_metadata(traj_path: Path) -> tuple[dict, Path]:
    metadata_path = traj_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"trajectory metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata.get("episodes"), list):
        raise ValueError(f"{metadata_path} has no episodes list")
    return metadata, metadata_path


def load_camera_configs(
    metadata: dict, metadata_path: Path, camera_sizes: dict[str, tuple[int, int]]
) -> dict[str, dict]:
    """Load and validate camera settings from trajectory metadata."""
    configs = metadata.get("camera_configs")
    if not isinstance(configs, dict):
        raise ValueError(f"{metadata_path} has no camera_configs mapping")

    missing = set(camera_sizes) - configs.keys()
    if missing:
        raise ValueError(f"{metadata_path} lacks camera configs: {sorted(missing)}")
    for camera, (width, height) in camera_sizes.items():
        config = configs[camera]
        if not isinstance(config, dict):
            raise ValueError(f"invalid camera config for {camera} in {metadata_path}")
        if (config.get("width"), config.get("height")) != (width, height):
            raise ValueError(
                f"{camera} config resolution does not match RGB frames: "
                f"{config.get('height')}x{config.get('width')} vs {height}x{width}"
            )
    return {camera: configs[camera] for camera in camera_sizes}


def vector_quantiles(
    data_dir: Path, feature_dims: dict[str, int], total_frames: int
) -> dict[str, dict[str, list[float]]]:
    """Compute exact per-dimension LeRobot quantiles from written Parquet chunks."""
    # ponytail: exact quantiles use ~4 * frames * vector_dims bytes; use disk-backed
    # selection if future datasets outgrow RAM.
    values = {
        key: np.empty((total_frames, dim), dtype=np.float32)
        for key, dim in feature_dims.items()
    }
    offset = 0
    for path in sorted(data_dir.rglob("*.parquet")):
        table = pq.read_table(path, columns=list(feature_dims))
        count = table.num_rows
        for key, dim in feature_dims.items():
            column = table[key].combine_chunks()
            if column.null_count:
                raise ValueError(f"null values in {key} at {path}")
            values[key][offset : offset + count] = column.values.to_numpy(
                zero_copy_only=False
            ).reshape(count, dim)
        offset += count

    if offset != total_frames:
        raise ValueError(
            f"expected {total_frames} frames in {data_dir}, found {offset}"
        )

    result = {}
    for key, matrix in values.items():
        result[key] = {
            name: [
                float(np.quantile(matrix[:, i], q, method="linear"))
                for i in range(matrix.shape[1])
            ]
            for q, name in QUANTILES
        }
    return result


def main(args: Args):
    input_path = Path(args.traj_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    base_path = Path(args.output_dir)
    source_metadata, metadata_path = load_trajectory_metadata(input_path)
    source_episodes = source_metadata["episodes"]
    with h5py.File(input_path, "r") as source_h5:
        n_probe = sum(key.startswith("traj_") for key in source_h5.keys())
    if len(source_episodes) != n_probe:
        raise ValueError(
            f"{metadata_path} has {len(source_episodes)} episodes, "
            f"but {input_path} has {n_probe} trajectories"
        )

    it = iter_episodes(input_path)
    first_ep, rgb_cameras, state_dim = next(it)
    action_dim = first_ep["actions"].shape[1]
    camera_sizes = {}
    for cam in rgb_cameras:
        frames = first_ep[f"rgb_{cam}"]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"expected RGB frames (T, H, W, 3) for {cam}, got {frames.shape}"
            )
        camera_sizes[cam] = (frames.shape[2], frames.shape[1])  # width, height
    camera_configs = (
        load_camera_configs(source_metadata, metadata_path, camera_sizes)
        if camera_sizes
        else {}
    )

    a_mom = [Moments() for _ in range(action_dim)]
    s_mom = [Moments() for _ in range(state_dim)] if state_dim else []
    cam_mom = {cam: [Moments() for _ in range(3)] for cam in rgb_cameras}
    ts_mom, fi_mom, ei_mom, ix_mom, ti_mom = Moments(), Moments(), Moments(), Moments(), Moments()

    episode_lengths = []
    episode_states = []
    episode_metadata = []
    dfs = []
    global_index = 0
    total_frames = 0
    ep_idx = -1

    def handle(ep_data, source_episode):
        nonlocal ep_idx, global_index, total_frames
        ep_idx += 1
        df = process_episode(
            ep_data, ep_idx, state_dim is not None, args.fps,
            task_index=0, task_name=args.task_name,
        )
        length = len(df)
        df["index"] = range(global_index, global_index + length)
        global_index += length
        total_frames += length

        chunk_idx = ep_idx // args.chunks_size
        for cam in rgb_cameras:
            frames = ep_data[f"rgb_{cam}"]
            width, height = camera_sizes[cam]
            if frames.shape[1:3] != (height, width):
                raise ValueError(
                    f"{cam} resolution changed: expected {height}x{width}, "
                    f"got {frames.shape[1]}x{frames.shape[2]}"
                )
            video_path = (base_path / "videos" / f"observation.images.{cam}" /
                          f"chunk-{chunk_idx:03d}" / f"file-{ep_idx:03d}.mp4")
            create_video_from_frames(frames, video_path, args.fps, width, height)
            sample = frames[:: max(1, length // 20)]
            pix = (sample.astype(np.float32) / 255.0).reshape(-1, 3)
            for c in range(3):
                cam_mom[cam][c].add(pix[:, c])

        actions = ep_data["actions"].astype(np.float64)
        for i in range(action_dim):
            a_mom[i].add(actions[:, i])
        state = None
        if state_dim and "robot_state" in ep_data:
            state = ep_data["robot_state"].astype(np.float64)
            for i in range(state_dim):
                s_mom[i].add(state[:, i])
        ts_mom.add(df["timestamp"].values)
        fi_mom.add(df["frame_index"].values)
        ei_mom.add(df["episode_index"].values)
        ix_mom.add(df["index"].values)
        ti_mom.add(df["task_index"].values)

        episode_lengths.append(length)
        dfs.append(df)
        episode_states.append({
            "actions": {"min": actions.min(0).tolist(), "max": actions.max(0).tolist(),
                        "mean": actions.mean(0).tolist(), "std": actions.std(0).tolist(),
                        **quantile_stats(actions), "count": [length]},
            "state": ({"min": state.min(0).tolist(), "max": state.max(0).tolist(),
                       "mean": state.mean(0).tolist(), "std": state.std(0).tolist(),
                       **quantile_stats(state), "count": [length]}
                      if state is not None else None),
        })

        rewards = np.asarray(ep_data.get("rewards", []), dtype=np.float64).reshape(-1)
        reward_sum = source_episode.get("reward_sum")
        reward_mean = source_episode.get("reward_mean")
        if rewards.size:
            if reward_sum is None:
                reward_sum = float(rewards.sum())
            if reward_mean is None:
                reward_mean = float(rewards.mean())
        success_values = np.asarray(ep_data.get("success", [])).reshape(-1)
        success = source_episode.get("success")
        success_once = source_episode.get("success_once")
        if success is None and success_values.size:
            success = bool(success_values[-1])
        if success_once is None and success_values.size:
            success_once = bool(success_values.any())
        terminated = source_episode.get("terminated")
        truncated = source_episode.get("truncated")
        if terminated is None:
            values = np.asarray(ep_data.get("terminated", [])).reshape(-1)
            if values.size:
                terminated = bool(values[-1])
        if truncated is None:
            values = np.asarray(ep_data.get("truncated", [])).reshape(-1)
            if values.size:
                truncated = bool(values[-1])
        elapsed_steps = source_episode.get("elapsed_steps")
        if elapsed_steps is None:
            elapsed_steps = length

        details = dict(source_episode)
        details.update({
            "episode_index": ep_idx,
            "source_episode_id": source_episode.get(
                "source_episode_id", source_episode.get("episode_id")
            ),
            "episode_length": length,
            "elapsed_steps": int(elapsed_steps),
            "duration_s": length / args.fps,
            "success": None if success is None else bool(success),
            "success_once": None if success_once is None else bool(success_once),
            "reward_sum": None if reward_sum is None else float(reward_sum),
            "reward_mean": None if reward_mean is None else float(reward_mean),
            "terminated": None if terminated is None else bool(terminated),
            "truncated": None if truncated is None else bool(truncated),
        })
        episode_metadata.append(details)

    # pre-create directories for the full episode count
    num_chunks = (n_probe + args.chunks_size - 1) // args.chunks_size
    (base_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for c in range(num_chunks):
        (base_path / "data" / f"chunk-{c:03d}").mkdir(parents=True, exist_ok=True)
        for cam in rgb_cameras:
            (base_path / "videos" / f"observation.images.{cam}" /
             f"chunk-{c:03d}").mkdir(parents=True, exist_ok=True)

    # process the first episode, then the rest
    handle(first_ep, source_episodes[0])
    del first_ep
    for source_index, (ep_data, _cams, _sd) in enumerate(it, start=1):
        handle(ep_data, source_episodes[source_index])
        del ep_data
        if (ep_idx + 1) % 50 == 0:
            logger.info(f"episodes processed: {ep_idx + 1}/{n_probe}")

    logger.info(f"Processed {n_probe} episodes; writing parquets/meta")

    for chunk_idx in range(num_chunks):
        start, end = chunk_idx * args.chunks_size, min(
            (chunk_idx + 1) * args.chunks_size, ep_idx + 1)
        combined = pd.concat(dfs[start:end], ignore_index=True)
        combined["task"] = combined["task"].astype("string")
        fields = []
        for col in combined.columns:
            if col == "task":
                fields.append(pa.field("task", pa.string()))
            elif col in ("action", "observation.state"):
                fields.append(pa.field(col, pa.list_(pa.float32())))
            elif col == "timestamp":
                fields.append(pa.field(col, pa.float32()))
            elif col in ("frame_index", "episode_index", "index", "task_index"):
                fields.append(pa.field(col, pa.int64()))
        pq.write_table(
            pa.Table.from_pandas(combined, schema=pa.schema(fields)),
            base_path / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet")
    del dfs, combined

    ep_rows = []
    for i, st in enumerate(episode_states):
        chunk_idx = chunk_of_i(i, args.chunks_size)
        details = episode_metadata[i]
        em = {
            "episode_index": i, "data/chunk_index": chunk_idx, "data/file_index": 0,
            "dataset_from_index": sum(episode_lengths[:i]),
            "dataset_to_index": sum(episode_lengths[: i + 1]),
            "tasks": [args.task_name], "length": episode_lengths[i],
            "source_episode_id": details.get(
                "source_episode_id", details.get("episode_id")
            ),
            "episode_seed": details.get("episode_seed"),
            "elapsed_steps": details["elapsed_steps"],
            "duration_s": details["duration_s"],
            "success": details["success"],
            "success_once": details["success_once"],
            "reward_sum": details["reward_sum"],
            "reward_mean": details["reward_mean"],
            "terminated": details["terminated"],
            "truncated": details["truncated"],
            "control_mode": details.get("control_mode"),
            "meta/episodes/chunk_index": chunk_idx, "meta/episodes/file_index": 0,
            "stats/action/min": st["actions"]["min"],
            "stats/action/max": st["actions"]["max"],
            "stats/action/mean": st["actions"]["mean"],
            "stats/action/std": st["actions"]["std"],
            "stats/action/count": st["actions"]["count"],
        }
        for _, name in QUANTILES:
            em[f"stats/action/{name}"] = st["actions"][name]
        if st["state"]:
            em.update({
                "stats/observation.state/min": st["state"]["min"],
                "stats/observation.state/max": st["state"]["max"],
                "stats/observation.state/mean": st["state"]["mean"],
                "stats/observation.state/std": st["state"]["std"],
                "stats/observation.state/count": st["state"]["count"],
            })
            for _, name in QUANTILES:
                em[f"stats/observation.state/{name}"] = st["state"][name]
        for cam in rgb_cameras:
            p = f"videos/observation.images.{cam}"
            em[f"{p}/chunk_index"] = chunk_idx
            em[f"{p}/file_index"] = i
            em[f"{p}/from_timestamp"] = 0.0
            em[f"{p}/to_timestamp"] = episode_lengths[i] / args.fps
        ep_rows.append(em)
    pd.DataFrame(ep_rows).to_parquet(
        base_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    pd.DataFrame({"task_index": [0]}, index=[args.task_name]).to_parquet(
        base_path / "meta" / "tasks.parquet", index=True)

    source_rlds_metadata = dict(source_metadata)
    source_rlds_metadata.update({
        "env_id": source_metadata.get("env_id")
        or source_metadata.get("env_info", {}).get("env_id"),
        "task_name": args.task_name,
        "num_episodes": len(episode_metadata),
        "episode_lengths": episode_lengths,
        "episode_durations_s": [details["duration_s"] for details in episode_metadata],
        "success_once": [details["success_once"] for details in episode_metadata],
        "episode_seeds": [details.get("episode_seed") for details in episode_metadata],
        "episodes": episode_metadata,
    })
    if any(details["reward_sum"] is not None for details in episode_metadata):
        source_rlds_metadata["reward_sums"] = [
            details["reward_sum"] for details in episode_metadata
        ]
        source_rlds_metadata["reward_means"] = [
            details["reward_mean"] for details in episode_metadata
        ]
    (base_path / "meta" / "source_rlds_metadata.json").write_text(
        json.dumps(source_rlds_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del episode_states, episode_metadata, ep_rows

    features = {
        "action": {"dtype": "float32", "shape": [action_dim],
                   "names": [f"action_{i}" for i in range(action_dim)],
                   "fps": float(args.fps)},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": float(args.fps)},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(args.fps)},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(args.fps)},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(args.fps)},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(args.fps)},
        "task": {"dtype": "string", "shape": [1], "names": None, "fps": float(args.fps)},
    }
    if state_dim:
        features["observation.state"] = {
            "dtype": "float32", "shape": [state_dim],
            "names": [f"joint_{i}" for i in range(state_dim)], "fps": float(args.fps)}
    for cam in rgb_cameras:
        image_width, image_height = camera_sizes[cam]
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [image_height, image_width, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": float(args.fps), "video.height": image_height,
                     "video.width": image_width, "video.channels": 3,
                     "video.codec": "mp4v", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False,
                     "camera_config": camera_configs[cam]},
        }

    data_mb = int(sum(f.stat().st_size for f in (base_path / "data").rglob("*.parquet")) / 1048576)
    info = {
        "codebase_version": "v3.0", "robot_type": args.robot_type,
        "total_episodes": ep_idx + 1, "total_frames": total_frames,
        "total_tasks": 1, "total_videos": (ep_idx + 1) * len(rgb_cameras),
        "total_chunks": num_chunks, "chunks_size": args.chunks_size,
        "fps": args.fps, "data_files_size_in_mb": data_mb,
        "splits": {"train": f"0:{ep_idx + 1}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (base_path / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    stats = {"action": vec_stats(a_mom, action_dim)}
    quantile_dims = {"action": action_dim}
    if state_dim:
        stats["observation.state"] = vec_stats(s_mom, state_dim)
        quantile_dims["observation.state"] = state_dim
    for key, quantiles in vector_quantiles(
        base_path / "data", quantile_dims, total_frames
    ).items():
        stats[key].update(quantiles)
    for cam, chs in cam_mom.items():
        stats[f"observation.images.{cam}"] = {
            "mean": [[ch.summary()["mean"]] for ch in chs],
            "std": [[ch.summary()["std"]] for ch in chs],
            "max": [[ch.summary()["max"]] for ch in chs],
            "min": [[ch.summary()["min"]] for ch in chs],
            "count": [[total_frames]],
        }
    stats["timestamp"] = scalar_stats_float(ts_mom.summary(), total_frames)
    stats["frame_index"] = scalar_stats_int(fi_mom.summary(), ep_idx + 1)
    stats["episode_index"] = scalar_stats_int(ei_mom.summary(), ep_idx + 1)
    stats["index"] = scalar_stats_int(ix_mom.summary(), total_frames)
    stats["task_index"] = scalar_stats_int(ti_mom.summary(), ep_idx + 1)
    (base_path / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    logger.info(f"Conversion completed: {ep_idx + 1} episodes, {total_frames} frames, "
                f"{num_chunks} chunks -> {base_path}")
    return 0


def chunk_of_i(i, chunks_size):
    return i // chunks_size


if __name__ == "__main__":
    import sys
    sys.exit(main(tyro.cli(Args)))
