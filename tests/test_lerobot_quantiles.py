import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils import convert_to_lerobot_stream as converter
from utils.convert_to_lerobot_stream import Args, iter_episodes, main, vector_quantiles


def test_vector_quantiles_reads_all_parquet_chunks(tmp_path):
    data_dir = tmp_path / "data"
    for chunk, actions, states in (
        (0, [[0, 10], [1, 11]], [[-2, 8], [2, 9]]),
        (1, [[2, 12], [3, 13]], [[4, 10], [8, 11]]),
    ):
        path = data_dir / f"chunk-{chunk:03d}" / "file-000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "action": pa.array(actions, type=pa.list_(pa.float32())),
                    "observation.state": pa.array(
                        states, type=pa.list_(pa.float32())
                    ),
                }
            ),
            path,
        )

    result = vector_quantiles(
        data_dir, {"action": 2, "observation.state": 2}, total_frames=4
    )
    probabilities = (0.01, 0.10, 0.50, 0.90, 0.99)
    for key, rows in (
        ("action", [[0, 10], [1, 11], [2, 12], [3, 13]]),
        ("observation.state", [[-2, 8], [2, 9], [4, 10], [8, 11]]),
    ):
        expected = np.quantile(np.asarray(rows), probabilities, axis=0, method="linear")
        for i, name in enumerate(("q01", "q10", "q50", "q90", "q99")):
            np.testing.assert_allclose(result[key][name], expected[i])


def test_iter_episodes_uses_numeric_trajectory_order(tmp_path):
    source = tmp_path / "trajectory.h5"
    with h5py.File(source, "w") as file:
        for episode_id in (10, 2):
            episode = file.create_group(f"traj_{episode_id}")
            episode.create_dataset(
                "actions", data=np.full((1, 13), episode_id, dtype=np.float32)
            )

    episodes = [episode["actions"][0, 0] for episode, _, _ in iter_episodes(source)]
    assert episodes == [2, 10]


def test_iter_episodes_splits_flat_ds_fetch_state(tmp_path):
    source = tmp_path / "trajectory.h5"
    flat_state = np.arange(2 * 30, dtype=np.float32).reshape(2, 30)
    with h5py.File(source, "w") as file:
        episode = file.create_group("traj_0")
        episode.create_dataset("actions", data=np.zeros((2, 13), dtype=np.float32))
        episode.create_dataset("obs", data=flat_state)

    episode, _, state_dim = next(iter_episodes(source))
    assert state_dim == 12
    np.testing.assert_array_equal(episode["global_state"], flat_state[:, :3])
    np.testing.assert_array_equal(episode["robot_state"], flat_state[:, 3:15])


def test_converter_preserves_camera_sizes_and_writes_video_ranges(
    tmp_path, monkeypatch
):
    source = tmp_path / "trajectory.h5"
    output = tmp_path / "lerobot"
    actions = np.arange(3 * 13, dtype=np.float32).reshape(3, 13)
    states = np.arange(3 * 15, dtype=np.float32).reshape(3, 15)
    camera_sizes = {
        "fetch_hand": (128, 128),
        "head_left": (256, 256),
        "side_aux": (96, 160),
    }
    encoded_sizes = {}

    def record_video(frames, path, fps, width, height):
        camera = path.parts[-3].removeprefix("observation.images.")
        assert frames.shape[1:3] == (height, width)
        encoded_sizes[camera] = (width, height)

    monkeypatch.setattr(converter, "create_video_from_frames", record_video)
    camera_configs = {}
    with h5py.File(source, "w") as file:
        episode = file.create_group("traj_0")
        episode.create_dataset("actions", data=actions)
        episode.create_dataset("rewards", data=np.array([1.0, 2.0, 3.0]))
        episode.create_dataset("success", data=np.array([False, False, True]))
        episode.create_dataset("terminated", data=np.array([False, False, True]))
        episode.create_dataset("truncated", data=np.array([False, False, False]))
        episode.create_group("obs/agent").create_dataset("qpos", data=states)
        sensor_data = episode.create_group("obs/sensor_data")
        for camera, (height, width) in camera_sizes.items():
            sensor_data.create_group(camera).create_dataset(
                "rgb", data=np.zeros((3, height, width, 3), dtype=np.uint8)
            )
            camera_configs[camera] = {
                "uid": camera,
                "width": width,
                "height": height,
                "fov": 1.5,
                "intrinsic": None,
                "near": 0.01,
                "far": 100.0,
                "pose": {
                    "position": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "entity_uid": "head_camera_link",
                "mount_name": None,
                "shader_config": {
                    "shader_pack": "minimal",
                    "texture_names": {},
                    "shader_pack_config": {},
                },
            }
    source_metadata = {
        "env_info": {"env_id": "test-env"},
        "episodes": [
            {
                "episode_id": 7,
                "episode_seed": 42,
                "elapsed_steps": 3,
                "success": True,
                "success_once": True,
                "reward_sum": 6.0,
                "reward_mean": 2.0,
                "control_mode": "pd_joint_pos",
                "reset_kwargs": {"seed": 42},
            }
        ],
        "camera_configs": camera_configs,
    }
    source.with_suffix(".json").write_text(json.dumps(source_metadata))

    assert main(Args(str(source), str(output), task_name="test")) == 0
    assert encoded_sizes == {
        camera: (width, height)
        for camera, (height, width) in camera_sizes.items()
    }
    stats = json.loads((output / "meta" / "stats.json").read_text())
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["robot_type"] == "ds_fetch"
    episode = pq.read_table(
        output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()
    assert episode["source_episode_id"] == [7]
    assert episode["episode_seed"] == [42]
    assert episode["elapsed_steps"] == [3]
    assert episode["duration_s"] == [0.3]
    assert episode["success"] == [True]
    assert episode["success_once"] == [True]
    assert episode["reward_sum"] == [6.0]
    assert episode["reward_mean"] == [2.0]
    assert episode["terminated"] == [True]
    assert episode["truncated"] == [False]
    data = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    assert data["observation.state"] == states[:, 3:].tolist()
    assert data["global_state"] == states[:, :3].tolist()
    for key, values in (
        ("action", actions),
        ("observation.state", states[:, 3:]),
        ("global_state", states[:, :3]),
    ):
        expected = np.quantile(
            values, (0.01, 0.10, 0.50, 0.90, 0.99), axis=0, method="linear"
        )
        for i, name in enumerate(("q01", "q10", "q50", "q90", "q99")):
            np.testing.assert_allclose(stats[key][name], expected[i])
            np.testing.assert_allclose(episode[f"stats/{key}/{name}"][0], expected[i])
    assert stats["action"]["count"] == [3]
    assert stats["observation.state"]["count"] == [3]
    assert stats["global_state"]["count"] == [3]
    assert episode["stats/observation.state/count"] == [[3]]
    assert episode["stats/global_state/count"] == [[3]]
    assert info["features"]["observation.state"]["shape"] == [12]
    assert info["features"]["global_state"]["shape"] == [3]
    assert info["features"]["global_state"]["names"] == [
        "x_base", "y_base", "psi_base"
    ]

    source_rlds = json.loads(
        (output / "meta" / "source_rlds_metadata.json").read_text()
    )
    assert source_rlds["env_id"] == "test-env"
    assert source_rlds["num_episodes"] == 1
    assert source_rlds["episode_lengths"] == [3]
    assert source_rlds["episode_durations_s"] == [0.3]
    assert source_rlds["episode_seeds"] == [42]
    assert source_rlds["success_once"] == [True]
    assert source_rlds["reward_sums"] == [6.0]
    assert source_rlds["episodes"][0]["reset_kwargs"] == {"seed": 42}

    for camera, (height, width) in camera_sizes.items():
        feature = info["features"][f"observation.images.{camera}"]
        assert feature["shape"] == [height, width, 3]
        assert feature["info"]["video.height"] == height
        assert feature["info"]["video.width"] == width
        assert feature["info"]["camera_config"] == camera_configs[camera]
        video = f"videos/observation.images.{camera}"
        assert episode[f"{video}/chunk_index"] == [0]
        assert episode[f"{video}/file_index"] == [0]
        assert episode[f"{video}/from_timestamp"] == [0.0]
        assert episode[f"{video}/to_timestamp"] == [0.3]
