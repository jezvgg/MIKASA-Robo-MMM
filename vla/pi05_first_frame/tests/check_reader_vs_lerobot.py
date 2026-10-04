"""Compare mikasa_pi05's v3.0 reader with LeRobot's own (run in a LeRobot >= 0.4 venv).

    PYTHONPATH=vla/pi05_first_frame/src python vla/pi05_first_frame/tests/check_reader_vs_lerobot.py DATASET_DIR

Checks, on random frames plus episode starts and ends: state and action equal exactly,
each camera image within a small pixel tolerance (both decode the same h264 stream, the
YUV->RGB conversion may differ by rounding), and the image belongs to the same frame
(a neighbouring frame differs much more than the tolerance).
"""

from __future__ import annotations

import sys

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mikasa_pi05.dataset import CAMERAS, IMAGE_PREFIX, SameDrawerDataset

TOLERANCE = 2.0  # mean absolute difference, 0..255


def main(root: str, count: int = 120) -> None:
    ours = SameDrawerDataset(root, 10, max_open_files=256)
    theirs = LeRobotDataset("local/samedrawer", root=root, video_backend="pyav")
    rng = np.random.default_rng(0)
    indices = set(rng.integers(0, len(ours), count).tolist())
    for episode in ours.episodes[:5] + ours.episodes[-3:]:
        start, end = ours.episode_range[episode]
        indices |= {start, start + 1, end - 1}
    worst = 0.0
    for index in sorted(indices):
        a = ours[index]
        b = theirs[index]
        np.testing.assert_array_equal(a["observation.state"], b["observation.state"].numpy())
        np.testing.assert_array_equal(a["actions"][0], b["action"].numpy())
        for camera in CAMERAS:
            key = IMAGE_PREFIX + camera
            ref = (b[key].numpy().transpose(1, 2, 0) * 255).round()
            diff = np.abs(a[key].astype(np.float64) - ref).mean()
            worst = max(worst, diff)
            if diff > TOLERANCE:
                raise SystemExit(f"index {index} {camera}: mean |diff| {diff:.2f} > {TOLERANCE}")
    print(f"{len(indices)} frames match; worst mean |diff| {worst:.3f} of 255")


if __name__ == "__main__":
    main(sys.argv[1])
