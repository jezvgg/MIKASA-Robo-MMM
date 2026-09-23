import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.convert_to_lerobot_stream import Args, main, vector_quantiles


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


def test_converter_writes_vector_quantiles(tmp_path):
    source = tmp_path / "trajectory.h5"
    output = tmp_path / "lerobot"
    actions = np.arange(3 * 13, dtype=np.float32).reshape(3, 13)
    states = np.arange(3 * 15, dtype=np.float32).reshape(3, 15)
    with h5py.File(source, "w") as file:
        episode = file.create_group("traj_0")
        episode.create_dataset("actions", data=actions)
        episode.create_dataset("obs", data=states)

    assert main(Args(str(source), str(output), task_name="test", robot_type="ds_fetch")) == 0
    stats = json.loads((output / "meta" / "stats.json").read_text())
    episode = pq.read_table(
        output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pydict()
    for key, values in (("action", actions), ("observation.state", states)):
        expected = np.quantile(
            values, (0.01, 0.10, 0.50, 0.90, 0.99), axis=0, method="linear"
        )
        for i, name in enumerate(("q01", "q10", "q50", "q90", "q99")):
            np.testing.assert_allclose(stats[key][name], expected[i])
            np.testing.assert_allclose(
                episode[f"stats/{key}/{name}"][0], expected[i]
            )
