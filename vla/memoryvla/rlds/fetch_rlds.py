"""RLDS (TFDS) datasets of the six Fetch tasks for MemoryVLA's OXE data loader.

Laid out after RoboMME/MemoryVLA's `rlds_dataset_builder/robomme` (pin f7befab), the
builder the MemoryVLA authors wrote for a benchmark like ours: one RLDS step per stored
frame, the main camera as `image` and the wrist camera as `wrist_image` (JPEG), the
episode's own instruction. What differs is only what our data is:

* three cameras (the owner's choice, 2026-10-21): `image` the left base camera, `wrist_image`
  fetch_hand and `right_image` the right base camera, for MemoryVLA's three-view mode
  (`--load_bi_wrist`: primary + its left_wrist and right_wrist slots, patches/0001);

* the source is a LeRobot v3 dataset (parquet + h264 videos), read frame for frame at its
  stored 10 Hz; images keep their stored size (main 256x256, wrist 128x128), the OXE loader
  resizes them for the model;
* `action` is the stored 13D Fetch command (arm 7, gripper 1, body 3, base forward/yaw
  velocity), not RoboMME's 7D EEF delta; `state` keeps the stored 12D joints, which the
  MIKASA config of the authors' `mikasa-robo` branch does not load.

One builder per task (`fetch_samedrawer`, ...); the source root is `$FETCH_LEROBOT_ROOT`
(default ~/datasets). `FETCH_RLDS_MAX_EPISODES` limits a build for a size test.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

PRIMARY = "observation.images.left_base_camera_link"
WRIST = "observation.images.fetch_hand"
RIGHT = "observation.images.right_base_camera_link"
# The labelled copies (vla/memer/label_subtasks.py): the same frames and videos as the HF
# datasets, with the task text of the environments' current instructions (salt and pepper
# named by their shakers' colours since 2026-10-22), which every policy trains on.
SOURCES = {
    "fetch_samedrawer": "memer/samedrawer",
    "fetch_seasondish": "memer/seasondish",
    "fetch_cabinetsearch": "memer/cabinetsearch",
    "fetch_depthrecall": "memer/depthrecall",
    "fetch_takeitback": "memer/takeitback",
    "fetch_veggie": "memer/veggie",
}


def source_root(name: str) -> Path:
    return Path(os.environ.get("FETCH_LEROBOT_ROOT", Path.home() / "datasets")) / SOURCES[name]


def episodes_table(root: Path) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))) \
        .sort_values("episode_index").reset_index(drop=True)


def video_frames(root: Path, row, key: str) -> list[np.ndarray]:
    """The episode's frames of one camera, by the episode's own [from, to) timestamps."""
    col = f"videos/{key}"
    path = root / "videos" / key / f"chunk-{int(row[col + '/chunk_index']):03d}" / f"file-{int(row[col + '/file_index']):03d}.mp4"
    start, stop = float(row[col + "/from_timestamp"]), float(row[col + "/to_timestamp"])
    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start - 1e-3:
                continue
            if t >= stop - 1e-3:
                break
            out.append(frame.to_ndarray(format="rgb24"))
    return out


class FetchRLDS(tfds.core.GeneratorBasedBuilder):
    """Base builder; subclasses only name the source."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "LeRobot v3 -> RLDS, one step per stored 10 Hz frame."}

    def _info(self) -> tfds.core.DatasetInfo:
        info = json.loads((source_root(self.name) / "meta/info.json").read_text())
        main, wrist, right = (info["features"][k]["shape"] for k in (PRIMARY, WRIST, RIGHT))
        return self.dataset_info_from_configs(features=tfds.features.FeaturesDict({
            "steps": tfds.features.Dataset({
                "observation": tfds.features.FeaturesDict({
                    "image": tfds.features.Image(shape=tuple(main), dtype=np.uint8, encoding_format="jpeg",
                                                 doc="Main camera (left base) RGB."),
                    "wrist_image": tfds.features.Image(shape=tuple(wrist), dtype=np.uint8, encoding_format="jpeg",
                                                       doc="Wrist camera (fetch_hand) RGB."),
                    "right_image": tfds.features.Image(shape=tuple(right), dtype=np.uint8, encoding_format="jpeg",
                                                       doc="Right base camera RGB."),
                    "state": tfds.features.Tensor(shape=(12,), dtype=np.float32, doc="Stored 12D joint state."),
                }),
                "action": tfds.features.Tensor(shape=(13,), dtype=np.float32, doc="Stored 13D Fetch command."),
                "discount": tfds.features.Scalar(dtype=np.float32, doc="Discount, 1."),
                "reward": tfds.features.Scalar(dtype=np.float32, doc="1 on the final step."),
                "is_first": tfds.features.Scalar(dtype=np.bool_, doc="First step of the episode."),
                "is_last": tfds.features.Scalar(dtype=np.bool_, doc="Last step of the episode."),
                "is_terminal": tfds.features.Scalar(dtype=np.bool_, doc="Last step (demos)."),
                "language_instruction": tfds.features.Text(doc="The episode's instruction."),
            }),
            "episode_metadata": tfds.features.FeaturesDict({
                "file_path": tfds.features.Text(doc="Source dataset and episode index."),
                "episode_index": tfds.features.Scalar(dtype=np.int64, doc="Episode index in the source."),
            }),
        }))

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        root = source_root(self.name)
        episodes = episodes_table(root)
        limit = int(os.environ.get("FETCH_RLDS_MAX_EPISODES", "0")) or len(episodes)
        cols = ["episode_index", "frame_index", "action", "observation.state"]
        data = pd.concat(pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
        for _, row in episodes.head(limit).iterrows():
            e = int(row["episode_index"])
            d = data[data.episode_index == e].sort_values("frame_index")
            main, wrist, right = (video_frames(root, row, k) for k in (PRIMARY, WRIST, RIGHT))
            n = int(row["length"])
            if not (len(d) == len(main) == len(wrist) == len(right) == n):
                raise ValueError(f"{root.name} episode {e}: {len(d)} rows, "
                                 f"{len(main)}/{len(wrist)}/{len(right)} frames, length {n}")
            task = str(list(row["tasks"])[0])
            actions = np.stack(d["action"].to_numpy()).astype(np.float32)
            states = np.stack(d["observation.state"].to_numpy()).astype(np.float32)
            steps = [{
                "observation": {"image": main[i], "wrist_image": wrist[i], "right_image": right[i],
                                "state": states[i]},
                "action": actions[i],
                "discount": 1.0,
                "reward": float(i == n - 1),
                "is_first": i == 0,
                "is_last": i == n - 1,
                "is_terminal": i == n - 1,
                "language_instruction": task,
            } for i in range(n)]
            key = f"{SOURCES[self.name]}/episode_{e}"
            yield key, {"steps": steps, "episode_metadata": {"file_path": key, "episode_index": e}}


class FetchSamedrawer(FetchRLDS):
    pass


class FetchSeasondish(FetchRLDS):
    pass


class FetchCabinetsearch(FetchRLDS):
    pass


class FetchDepthrecall(FetchRLDS):
    pass


class FetchTakeitback(FetchRLDS):
    pass


class FetchVeggie(FetchRLDS):
    pass


BUILDERS = {cls.name: cls for cls in (FetchSamedrawer, FetchSeasondish, FetchCabinetsearch,
                                      FetchDepthrecall, FetchTakeitback, FetchVeggie)}

