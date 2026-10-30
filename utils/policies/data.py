"""Immutable two-root LeRobot-v3 access with explicit episode provenance.

No action-coordinate conversion is performed: Fetch's 13 channels already mix
absolute joint targets and normalized base velocities. Importing this module
requires neither a model backend nor LeRobot itself.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

CAMERAS = ("left_base_camera_link", "fetch_hand", "right_base_camera_link")
TASKS = ("takeitback", "veggie")
STATE_DIM = 12
ACTION_DIM = 13
FPS = 10


@dataclass(frozen=True)
class Episode:
    task: str
    source_index: int
    global_index: int
    length: int
    metadata: dict


class FetchDataset:
    """Frame-proportional union, preserving every frame and episode boundary."""

    def __init__(self, root: str | Path, tasks: tuple[str, ...] = TASKS):
        import pyarrow.parquet as pq

        self.root = Path(root).resolve()
        self.tasks = tuple(tasks)
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be nonempty and unique")
        self.info = {}
        self.instructions = {}
        self.episodes = []
        self.ends = []
        total = 0
        for task in self.tasks:
            base = self.root / task
            info = json.loads((base / "meta/info.json").read_text())
            if float(info["fps"]) != FPS:
                raise ValueError(f"{task}: expected {FPS} Hz, got {info['fps']}")
            for key, dim in (("action", ACTION_DIM), ("observation.state", STATE_DIM)):
                if info["features"][key]["shape"] != [dim]:
                    raise ValueError(f"{task}: unexpected {key} shape")
            for camera in CAMERAS:
                if f"observation.images.{camera}" not in info["features"]:
                    raise ValueError(f"{task}: missing {camera}")
            self.info[task] = info
            table = pq.read_table(base / "meta/tasks.parquet").to_pandas()
            if "task_index" not in table:
                raise ValueError(f"{task}: missing task_index")
            text_columns = [c for c in table.columns if c != "task_index"]
            for index, row in table.iterrows():
                text = str(row[text_columns[0]]) if text_columns else str(index)
                self.instructions[(task, int(row.task_index))] = text
            rows = []
            for path in sorted((base / "meta/episodes").rglob("*.parquet")):
                rows.extend(pq.read_table(path).to_pylist())
            rows.sort(key=lambda row: int(row["episode_index"]))
            if len(rows) != info["total_episodes"]:
                raise ValueError(f"{task}: episode count mismatch")
            if len({int(r['episode_index']) for r in rows}) != len(rows):
                raise ValueError(f"{task}: duplicate episode indices")
            for row in rows:
                length = int(row["length"])
                if length < 1:
                    raise ValueError("empty episode")
                self.episodes.append(Episode(task, int(row["episode_index"]), len(self.episodes), length, row))
                total += length
                self.ends.append(total)
            if sum(int(r["length"]) for r in rows) != info["total_frames"]:
                raise ValueError(f"{task}: frame count mismatch")

    def __len__(self):
        return self.ends[-1]

    def locate(self, index: int) -> tuple[Episode, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.ends, index)
        start = self.ends[episode_index - 1] if episode_index else 0
        return self.episodes[episode_index], index - start

    def data_path(self, episode: Episode) -> Path:
        row = episode.metadata
        relative = self.info[episode.task]["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        return self.root / episode.task / relative

    def video_path(self, episode: Episode, camera: str) -> Path:
        key = f"observation.images.{camera}"
        row = episode.metadata
        relative = self.info[episode.task]["video_path"].format(
            video_key=key, chunk_index=row[f"videos/{key}/chunk_index"],
            file_index=row[f"videos/{key}/file_index"],
        )
        return self.root / episode.task / relative

    @lru_cache(maxsize=8)
    def episode_rows(self, global_episode: int):
        import pyarrow.parquet as pq

        episode = self.episodes[global_episode]
        table = pq.read_table(
            self.data_path(episode),
            filters=[("episode_index", "=", episode.source_index)],
        )
        rows = table.to_pydict()
        if len(rows["action"]) != episode.length:
            raise ValueError(f"{episode.task}/{episode.source_index}: wrong row count")
        if rows["frame_index"] != list(range(episode.length)):
            raise ValueError("episode frames must be in chronological order")
        return rows

    @lru_cache(maxsize=6)
    def decoder(self, path: str):
        from torchcodec.decoders import VideoDecoder

        return VideoDecoder(path, device="cpu", num_ffmpeg_threads=1)

    def image(self, episode: Episode, frame: int, camera: str) -> np.ndarray:
        key = f"observation.images.{camera}"
        offset = float(episode.metadata[f"videos/{key}/from_timestamp"])
        video_frame = int(round(offset * FPS)) + frame
        tensor = self.decoder(str(self.video_path(episode, camera))).get_frame_at(video_frame).data
        return tensor.permute(1, 2, 0).numpy().copy()

    def sample(self, index: int, horizon: int, *, images: bool = True) -> dict:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        episode, frame = self.locate(index)
        rows = self.episode_rows(episode.global_index)
        indices = np.minimum(np.arange(frame, frame + horizon), episode.length - 1)
        actions = np.asarray(rows["action"], dtype=np.float32)[indices]
        state = np.asarray(rows["observation.state"][frame], dtype=np.float32)
        if state.shape != (STATE_DIM,) or actions.shape != (horizon, ACTION_DIM):
            raise ValueError("invalid Fetch sample shapes")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("non-finite Fetch training sample")
        sample = {
            "state": state, "actions": actions,
            "prompt": self.instructions[(episode.task, int(rows["task_index"][frame]))],
            "episode_index": episode.global_index, "frame_index": frame,
            "task": episode.task,
        }
        if images:
            sample["images"] = {camera: self.image(episode, frame, camera) for camera in CAMERAS}
        return sample

    def manifest(self) -> dict:
        result = {"root": str(self.root), "fps": FPS, "state_dim": STATE_DIM,
                  "action_dim": ACTION_DIM, "cameras": list(CAMERAS),
                  "sampling": "one anchor per exported 10-Hz frame; frame-proportional task mixture",
                  "terminal_actions": "repeat final action inside the same episode", "tasks": {}}
        for task in self.tasks:
            episodes = [e for e in self.episodes if e.task == task]
            result["tasks"][task] = {"episodes": len(episodes), "frames": sum(e.length for e in episodes)}
        result["total_frames"] = len(self)
        result["total_episodes"] = len(self.episodes)
        return result


def epoch_plan(num_samples: int, batch_size: int, epochs: int = 1) -> dict:
    """Final JAX batch needs an explicit validity mask; never drop the tail."""
    if min(num_samples, batch_size, epochs) < 1:
        raise ValueError("positive sample, batch and epoch counts required")
    steps = (num_samples + batch_size - 1) // batch_size
    return {"samples_per_epoch": num_samples, "batch_size": batch_size,
            "epochs": epochs, "steps_per_epoch": steps, "total_steps": steps * epochs,
            "final_batch_valid": num_samples - (steps - 1) * batch_size,
            "tail_policy": "partial batch or explicit masked padding; no drop_last"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    dataset = FetchDataset(args.root)
    report = dataset.manifest()
    report["epoch_plan"] = epoch_plan(len(dataset), args.batch_size)
    report["samples"] = []
    for task in dataset.tasks:
        first = next(e for e in dataset.episodes if e.task == task)
        start = dataset.ends[first.global_index - 1] if first.global_index else 0
        for index in (start, start + first.length - 1):
            sample = dataset.sample(index, 20, images=args.check_images)
            report["samples"].append({"index": index, "task": task, "prompt": sample["prompt"],
                "state_shape": list(sample["state"].shape), "actions_shape": list(sample["actions"].shape),
                "image_shapes": {k: list(v.shape) for k, v in sample.get("images", {}).items()}})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
