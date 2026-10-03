"""Combine chunk-sized LeRobot v3 streams without keeping RGB HDF5 files."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from utils.convert_to_lerobot_stream import vector_quantiles


class Moments:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sum2 = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not values.size:
            return
        self.n += values.size
        self.sum += float(values.sum())
        self.sum2 += float(np.square(values).sum())
        lo, hi = float(values.min()), float(values.max())
        self.minimum = lo if self.minimum is None else min(self.minimum, lo)
        self.maximum = hi if self.maximum is None else max(self.maximum, hi)

    def stats(self, count):
        mean = self.sum / self.n
        std = float(np.sqrt(max(self.sum2 / self.n - mean * mean, 0.0)))
        return {
            "mean": [mean],
            "std": [std],
            "max": [self.maximum],
            "min": [self.minimum],
            "count": [count],
        }


def vector_stats(moments, count):
    return {
        "mean": [m.sum / m.n for m in moments],
        "std": [
            float(np.sqrt(max(m.sum2 / m.n - (m.sum / m.n) ** 2, 0.0)))
            for m in moments
        ],
        "max": [m.maximum for m in moments],
        "min": [m.minimum for m in moments],
        "count": [count],
    }


def scalar_image_stats(sources, camera, total_frames):
    n = 0
    sums = np.zeros(3, dtype=np.float64)
    sums2 = np.zeros(3, dtype=np.float64)
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for source in sources:
        stats = json.loads((source / "meta" / "stats.json").read_text())[
            f"observation.images.{camera}"
        ]
        count = int(stats["count"][0][0])
        mean = np.asarray(stats["mean"], dtype=np.float64).reshape(3)
        std = np.asarray(stats["std"], dtype=np.float64).reshape(3)
        sums += mean * count
        sums2 += (std * std + mean * mean) * count
        minimum = np.minimum(minimum, np.asarray(stats["min"], dtype=np.float64).reshape(3))
        maximum = np.maximum(maximum, np.asarray(stats["max"], dtype=np.float64).reshape(3))
        n += count
    mean = sums / n
    std = np.sqrt(np.maximum(sums2 / n - mean * mean, 0.0))
    return {
        "mean": [[float(x)] for x in mean],
        "std": [[float(x)] for x in std],
        "max": [[float(x)] for x in maximum],
        "min": [[float(x)] for x in minimum],
        "count": [[total_frames]],
    }


def copy_batches(sources, output):
    first_info = json.loads((sources[0] / "meta" / "info.json").read_text())
    cameras = [
        key.removeprefix("observation.images.")
        for key, feature in first_info["features"].items()
        if feature.get("dtype") == "video"
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    (output / "videos").mkdir(exist_ok=True)
    (output / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    frame_offset = 0
    episode_offset = 0
    episode_rows = []
    source_episode_rows = []
    all_lengths = []
    all_durations = []
    all_success_once = []
    all_seeds = []

    for chunk_idx, source in enumerate(sources):
        info = json.loads((source / "meta" / "info.json").read_text())
        data_path = source / "data" / "chunk-000" / "file-000.parquet"
        data = pd.read_parquet(data_path)
        data["episode_index"] = data["episode_index"].astype(np.int64) + episode_offset
        data["index"] = data["index"].astype(np.int64) + frame_offset
        target_data = output / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet"
        target_data.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(target_data, index=False)

        episodes = pd.read_parquet(
            source / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        for row in episodes.to_dict("records"):
            local_episode = int(row["episode_index"])
            row["episode_index"] = local_episode + episode_offset
            row["data/chunk_index"] = chunk_idx
            row["dataset_from_index"] = int(row["dataset_from_index"]) + frame_offset
            row["dataset_to_index"] = int(row["dataset_to_index"]) + frame_offset
            row["meta/episodes/chunk_index"] = chunk_idx
            for camera in cameras:
                prefix = f"videos/observation.images.{camera}"
                row[f"{prefix}/chunk_index"] = chunk_idx
            episode_rows.append(row)

        source_metadata = json.loads(
            (source / "meta" / "source_rlds_metadata.json").read_text()
        )
        for details in source_metadata["episodes"]:
            details = dict(details)
            old_id = details.get("episode_id", len(source_episode_rows))
            details["source_episode_id"] = details.get("source_episode_id", old_id)
            details["episode_id"] = len(source_episode_rows)
            details["episode_index"] = len(source_episode_rows)
            source_episode_rows.append(details)
            all_lengths.append(int(details["episode_length"]))
            all_durations.append(float(details["duration_s"]))
            all_success_once.append(details.get("success_once"))
            all_seeds.append(details.get("episode_seed"))

        for camera in cameras:
            src_dir = source / "videos" / f"observation.images.{camera}" / "chunk-000"
            dst_dir = output / "videos" / f"observation.images.{camera}" / f"chunk-{chunk_idx:03d}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for local_episode in range(int(info["total_episodes"])):
                src = src_dir / f"file-{local_episode:03d}.mp4"
                if not src.exists():
                    raise FileNotFoundError(src)
                shutil.copy2(src, dst_dir / src.name)

        frame_offset += int(info["total_frames"])
        episode_offset += int(info["total_episodes"])

    episodes = pd.DataFrame(episode_rows)
    episodes.to_parquet(
        output / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    shutil.copy2(sources[0] / "meta" / "tasks.parquet", output / "meta" / "tasks.parquet")
    if (sources[0] / "README.md").exists():
        shutil.copy2(sources[0] / "README.md", output / "README.md")

    metadata = json.loads(
        (sources[0] / "meta" / "source_rlds_metadata.json").read_text()
    )
    metadata.update(
        num_episodes=episode_offset,
        episode_lengths=all_lengths,
        episode_durations_s=all_durations,
        success_once=all_success_once,
        episode_seeds=all_seeds,
        episodes=source_episode_rows,
    )
    if "reward_sums" in metadata:
        metadata["reward_sums"] = [e.get("reward_sum") for e in source_episode_rows]
        metadata["reward_means"] = [e.get("reward_mean") for e in source_episode_rows]
    (output / "meta" / "source_rlds_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    info = dict(first_info)
    info.update(
        total_episodes=episode_offset,
        total_frames=frame_offset,
        total_videos=episode_offset * len(cameras),
        total_chunks=len(sources),
        splits={"train": f"0:{episode_offset}"},
    )
    info["data_files_size_in_mb"] = int(
        sum(path.stat().st_size for path in (output / "data").rglob("*.parquet"))
        / 1048576
    )
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info, cameras, frame_offset, episode_offset


def recompute_stats(output, info, cameras, sources, total_frames, total_episodes):
    action_dim = int(info["features"]["action"]["shape"][0])
    state_dim = info["features"].get("observation.state", {}).get("shape", [0])[0]
    global_dim = info["features"].get("global_state", {}).get("shape", [0])[0]
    vector_columns = {"action": action_dim}
    if state_dim:
        vector_columns["observation.state"] = state_dim
    if global_dim:
        vector_columns["global_state"] = global_dim
    moments = {key: [Moments() for _ in range(dim)] for key, dim in vector_columns.items()}
    scalar_columns = {
        "timestamp": total_frames,
        "frame_index": total_episodes,
        "episode_index": total_episodes,
        "index": total_frames,
        "task_index": total_episodes,
    }
    scalar_moments = {key: Moments() for key in scalar_columns}
    for path in sorted((output / "data").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        for key, dim in vector_columns.items():
            values = np.asarray(frame[key].tolist(), dtype=np.float32)
            for i in range(dim):
                moments[key][i].add(values[:, i])
        for key in scalar_columns:
            scalar_moments[key].add(frame[key].to_numpy())

    stats = {
        key: vector_stats(moments[key], total_frames) for key in vector_columns
    }
    quantiles = vector_quantiles(output / "data", vector_columns, total_frames)
    for key, values in quantiles.items():
        stats[key].update(values)
    for camera in cameras:
        stats[f"observation.images.{camera}"] = scalar_image_stats(
            sources, camera, total_frames
        )
    for key, count in scalar_columns.items():
        stats[key] = scalar_moment_stats(scalar_moments[key], count)
    (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")


def scalar_moment_stats(moment, count):
    return moment.stats(count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    info, cameras, total_frames, total_episodes = copy_batches(args.sources, args.output_dir)
    recompute_stats(
        args.output_dir, info, cameras, args.sources, total_frames, total_episodes
    )
    print(
        f"combined {total_episodes} episodes, {total_frames} frames, "
        f"{len(args.sources)} chunks -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
