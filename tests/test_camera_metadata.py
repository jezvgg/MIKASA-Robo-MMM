import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np

from utils.merge_replays import main as merge_replays
from utils.replay_rgb import camera_config_metadata, episode_outcome_metadata


def test_camera_config_metadata_is_json_safe():
    config = SimpleNamespace(
        uid="head_camera",
        width=256,
        height=256,
        fov=1.5,
        intrinsic=None,
        near=0.01,
        far=100,
        pose=SimpleNamespace(
            p=np.array([[1.0, 2.0, 3.0]]),
            q=np.array([[1.0, 0.0, 0.0, 0.0]]),
        ),
        entity_uid="head_camera_link",
        mount=None,
        shader_config=SimpleNamespace(
            shader_pack="minimal",
            texture_names={"Color": ["rgb"]},
            shader_pack_config={},
        ),
    )

    metadata = camera_config_metadata(config)
    json.dumps(metadata)
    assert metadata["width"] == metadata["height"] == 256
    assert metadata["fov"] == 1.5
    assert metadata["pose"]["position"] == [1.0, 2.0, 3.0]


def test_episode_outcome_uses_full_rate_rewards():
    metadata = episode_outcome_metadata(
        rewards=[1.0, 2.0, 3.0, 4.0],
        success=[False, True, False, False],
        terminated=[False, False, False, True],
        truncated=[False, False, False, False],
    )
    assert metadata == {
        "reward_sum": 10.0,
        "reward_mean": 2.5,
        "success": False,
        "success_once": True,
        "terminated": True,
        "truncated": False,
    }


def test_merge_replays_preserves_camera_configs(tmp_path, monkeypatch):
    source = tmp_path / "replays"
    output = tmp_path / "merged"
    configs = {"fetch_hand": {"width": 128, "height": 128, "fov": 2.0}}
    for seed in (0, 1):
        run = source / f"seed{seed}"
        run.mkdir(parents=True)
        with h5py.File(run / "trajectory.h5", "w") as file:
            file.create_group("traj_0")
        (run / "trajectory.json").write_text(
            json.dumps(
                {
                    "env_info": {"env_id": "test"},
                    "commit_info": None,
                    "episodes": [{"episode_id": seed + 10, "episode_seed": seed}],
                    "camera_configs": configs,
                }
            )
        )

    monkeypatch.setattr(
        sys, "argv", ["merge_replays", "--src", str(source), "--out", str(output)]
    )
    merge_replays()
    merged = json.loads((output / "trajectory.json").read_text())
    assert merged["camera_configs"] == configs
    assert [episode["episode_id"] for episode in merged["episodes"]] == [0, 1]
    assert [episode["source_episode_id"] for episode in merged["episodes"]] == [10, 11]
