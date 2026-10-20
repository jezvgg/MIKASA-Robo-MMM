"""Live simulator render vs dataset frames at reset, for training episodes.

Training images were re-rendered from saved states and stored as h264; evaluation
renders live. This measures how far apart the two are on this machine.

    # 1. simulator venv, repository root:
    PYTHONPATH=. $SIM_VENV/bin/python vla/pi05_first_frame/tests/render_parity.py dump OUT.npz 8
    # 2. openpi venv:
    $OPENPI_DIR/.venv/bin/python vla/pi05_first_frame/tests/render_parity.py compare OUT.npz

`compare` prints, per camera, the mean absolute difference to the same episode's frame 0
and, as a scale, to that episode's frame 1 and to another episode's frame 0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

CAMERAS = ("left_base_camera_link", "right_base_camera_link", "fetch_hand")


def dump(out: str, count: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval_samedrawer import check_torch_cuda, load_profile_with_torch_build, resolve_seeds
    from utils.collection.profile import make_env, runtime_signature
    from utils.mikasa.seeding import seed_everything

    check_torch_cuda()
    dataset_dir = Path(os.environ["SAMEDRAWER_DATASET_DIR"])
    seeds, episode_of_seed = resolve_seeds(f"train:{count}", dataset_dir)
    env = make_env({"signature": runtime_signature(load_profile_with_torch_build()), "purpose": "validation"},
                   rgb=True)
    frames = {c: [] for c in CAMERAS}
    for seed in seeds:
        seed_everything(seed)
        obs, _ = env.reset(seed=seed)
        for camera in CAMERAS:
            frames[camera].append(obs["sensor_data"][camera]["rgb"][0].cpu().numpy())
    env.close()
    np.savez_compressed(out, episodes=np.array([episode_of_seed[s] for s in seeds]),
                        **{c: np.stack(v) for c, v in frames.items()})
    print(f"wrote {out}: {len(seeds)} episodes")


def compare(path: str) -> None:
    from mikasa_pi05.dataset import SameDrawerDataset

    data = np.load(path)
    ds = SameDrawerDataset(os.environ["SAMEDRAWER_DATASET_DIR"], 10)
    episodes = data["episodes"].tolist()
    for camera in CAMERAS:
        same, next_frame, other = [], [], []
        for i, episode in enumerate(episodes):
            live = data[camera][i].astype(np.float64)
            same.append(np.abs(live - ds.image(episode, 0, camera)).mean())
            next_frame.append(np.abs(live - ds.image(episode, 1, camera)).mean())
            other.append(np.abs(live - ds.image(episodes[(i + 1) % len(episodes)], 0, camera)).mean())
        print(f"{camera:24s} live vs frame0 {np.mean(same):6.3f} (max {np.max(same):.3f}) | "
              f"vs frame1 {np.mean(next_frame):6.3f} | vs other episode {np.mean(other):6.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], int(sys.argv[3]))
    else:
        compare(sys.argv[2])
