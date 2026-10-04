"""SameDrawer LeRobot v3.0 reader that adds the episode's first frame to every sample.

openpi's pinned LeRobot (0.1.0) reads only the v2.1 layout, while the SameDrawer
dataset is v3.0: per-camera mp4 files that each hold many episodes back to back,
and parquet files indexed by `meta/episodes`. This module reads that layout
directly. The numeric columns (~80 MB) stay in memory; images are decoded from
the mp4 files with PyAV, which ships its own FFmpeg.

Each sample uses the keys the evaluation client sends, so training and
evaluation go through the same openpi input transforms:

    observation.state                    (12,) float32, qpos[3:] in qpos order
    observation.images.<camera>          HxWx3 uint8 at frame t
    observation.first_frame              HxWx3 uint8, frame 0 of the same episode
    prompt                               the task instruction
    actions                              (action_horizon, 13) float32, targets for frames t..t+H-1

At the end of an episode the action chunk repeats the last action, as LeRobot's
`delta_timestamps` does for openpi's other datasets.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

CAMERAS = ("left_base_camera_link", "right_base_camera_link", "fetch_hand")
FIRST_FRAME_KEY = "observation.first_frame"
STATE_KEY = "observation.state"
IMAGE_PREFIX = "observation.images."


@dataclasses.dataclass(frozen=True)
class _VideoRef:
    path: str
    from_timestamp: float


class _VideoFile:
    """One open mp4; returns RGB frames by presentation time."""

    def __init__(self, path: str, fps: float):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "SLICE"
        self.tolerance = 0.5 / fps

    def frame_at(self, timestamp: float) -> np.ndarray:
        offset = int(timestamp / self.stream.time_base)
        self.container.seek(offset, stream=self.stream, backward=True, any_frame=False)
        for frame in self.container.decode(self.stream):
            if frame.time is None:
                continue
            if abs(frame.time - timestamp) <= self.tolerance:
                return frame.to_ndarray(format="rgb24")
            if frame.time > timestamp + self.tolerance:
                break
        raise ValueError(f"No frame at {timestamp:.3f} s in {self.container.name}")

    def close(self) -> None:
        self.container.close()


class SameDrawerDataset:
    """Map-style dataset over all frames of the selected episodes."""

    def __init__(
        self,
        root: str | Path,
        action_horizon: int,
        *,
        first_frame_camera: str = "left_base_camera_link",
        episodes: list[int] | None = None,
        load_images: bool = True,
        max_open_files: int = 8,
    ):
        self.root = Path(root)
        info = json.loads((self.root / "meta/info.json").read_text())
        if info["codebase_version"] != "v3.0":
            raise ValueError(f"Expected a LeRobot v3.0 dataset, got {info['codebase_version']}")
        if first_frame_camera not in CAMERAS:
            raise ValueError(f"Unknown camera {first_frame_camera!r}")
        self.fps = float(info["fps"])
        self.action_horizon = action_horizon
        self.first_frame_camera = first_frame_camera
        self.load_images = load_images
        self.max_open_files = max_open_files

        data = _read_parquets(sorted((self.root / "data").glob("*/*.parquet")),
                              ["observation.state", "action", "episode_index", "frame_index", "index", "task_index"])
        order = np.argsort(data["index"])
        if not np.array_equal(data["index"][order], np.arange(len(order))):
            raise ValueError("Dataset frame index is not contiguous")
        self.state = np.stack(data["observation.state"][order]).astype(np.float32)
        self.action = np.stack(data["action"][order]).astype(np.float32)
        self.episode_of_frame = data["episode_index"][order].astype(np.int64)
        self.frame_in_episode = data["frame_index"][order].astype(np.int64)
        task_of_frame = data["task_index"][order].astype(np.int64)

        tasks = pq.read_table(self.root / "meta/tasks.parquet").to_pandas()
        self.tasks = {int(row["task_index"]): str(name) for name, row in tasks.iterrows()}

        meta = _read_parquets(sorted((self.root / "meta/episodes").glob("*/*.parquet")), None)
        self.episode_range: dict[int, tuple[int, int]] = {}
        self.episode_seed: dict[int, int] = {}
        self.videos: dict[tuple[int, str], _VideoRef] = {}
        for i, ep in enumerate(meta["episode_index"]):
            ep = int(ep)
            self.episode_range[ep] = (int(meta["dataset_from_index"][i]), int(meta["dataset_to_index"][i]))
            if "episode_seed" in meta:
                self.episode_seed[ep] = int(meta["episode_seed"][i])
            for camera in CAMERAS:
                key = f"videos/{IMAGE_PREFIX}{camera}"
                path = self.root / info["video_path"].format(
                    video_key=IMAGE_PREFIX + camera,
                    chunk_index=int(meta[f"{key}/chunk_index"][i]),
                    file_index=int(meta[f"{key}/file_index"][i]),
                )
                self.videos[ep, camera] = _VideoRef(str(path), float(meta[f"{key}/from_timestamp"][i]))

        selected = sorted(self.episode_range) if episodes is None else sorted(set(episodes))
        missing = set(selected) - set(self.episode_range)
        if missing:
            raise ValueError(f"Episodes not in dataset: {sorted(missing)[:10]}")
        self.episodes = selected
        self.rows = np.concatenate([np.arange(*self.episode_range[ep]) for ep in selected])
        for ep in selected:
            start, end = self.episode_range[ep]
            if not (np.all(self.episode_of_frame[start:end] == ep)
                    and np.array_equal(self.frame_in_episode[start:end], np.arange(end - start))):
                raise ValueError(f"Episode {ep} frames are not contiguous")
        self.prompt_of_frame = task_of_frame

        self._files: dict[str, _VideoFile] = {}
        self._first_frames: dict[int, np.ndarray] = {}
        self._pid = os.getpid()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index) -> dict:
        row = int(self.rows[index.__index__()])
        episode = int(self.episode_of_frame[row])
        frame = int(self.frame_in_episode[row])
        _, end = self.episode_range[episode]
        chunk = np.minimum(np.arange(row, row + self.action_horizon), end - 1)
        sample = {
            STATE_KEY: self.state[row].copy(),
            "actions": self.action[chunk],  # fancy indexing copies; DeltaActions edits in place
            "prompt": self.tasks[int(self.prompt_of_frame[row])],
        }
        if self.load_images:
            for camera in CAMERAS:
                sample[IMAGE_PREFIX + camera] = self.image(episode, frame, camera)
            sample[FIRST_FRAME_KEY] = self.first_frame(episode)
        return sample

    def image(self, episode: int, frame: int, camera: str) -> np.ndarray:
        ref = self.videos[episode, camera]
        return self._file(ref.path).frame_at(ref.from_timestamp + frame / self.fps)

    def first_frame(self, episode: int) -> np.ndarray:
        if episode not in self._first_frames:
            self._first_frames[episode] = self.image(episode, 0, self.first_frame_camera)
        return self._first_frames[episode]

    def _file(self, path: str) -> _VideoFile:
        if os.getpid() != self._pid:  # never share decoder state across processes
            self._files, self._first_frames, self._pid = {}, {}, os.getpid()
        if path not in self._files:
            if len(self._files) >= self.max_open_files:
                oldest = next(iter(self._files))
                self._files.pop(oldest).close()
            self._files[path] = _VideoFile(path, self.fps)
        else:
            self._files[path] = self._files.pop(path)  # most recently used goes last
        return self._files[path]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"], state["_first_frames"] = {}, {}
        return state


def _read_parquets(paths: list[Path], columns: list[str] | None) -> dict[str, np.ndarray]:
    if not paths:
        raise FileNotFoundError("No parquet files found")
    tables = [pq.read_table(p, columns=columns) for p in paths]
    out = {}
    for name in tables[0].column_names:
        if name.startswith("stats/"):
            continue
        out[name] = np.concatenate([t.column(name).to_numpy(zero_copy_only=False) for t in tables])
    return out
