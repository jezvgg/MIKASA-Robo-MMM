"""Reproduce statistics loss in the upstream runtime metadata view."""

import json

import numpy as np
import pytest

from utils.collection.normalization_stats import (
    QUANTILES,
    read_episode_metadata,
    verify_normalization,
)


def metadata_fixture(root):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    shapes = {"action": [13], "observation.state": [12]}
    stats = {
        feature: {
            **{
                name: [0.0] * shape[0]
                for name in (*QUANTILES, "mean", "std", "min", "max")
            },
            "count": [8],
        }
        for feature, shape in shapes.items()
    }
    paths = root / "meta/episodes/chunk-000"
    paths.mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "features": {k: {"shape": v} for k, v in shapes.items()},
            }
        )
    )
    (root / "meta/stats.json").write_text(json.dumps(stats))
    records = [
        {
            "episode_index": episode,
            **{
                f"stats/{feature}/{q}": [float(episode)] * shape[0]
                for feature, shape in shapes.items()
                for q in QUANTILES
            },
        }
        for episode in [1, 0]
    ]
    file = paths / "file-000.parquet"
    pq.write_table(pa.Table.from_pylist(records), file)
    return file


def test_original_parquet_retains_quantiles_filtered_by_lerobot_runtime(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.utils import load_episodes

    metadata_fixture(tmp_path)
    runtime = load_episodes(tmp_path)
    assert not any(name.startswith("stats/") for name in runtime.column_names)
    full = read_episode_metadata(tmp_path)
    assert full["episode_index"].tolist() == [0, 1]
    np.testing.assert_array_equal(full["stats/action/q01"].iloc[1], np.ones(13))
    assert verify_normalization(tmp_path)["episode_quantile_columns"] == 10


@pytest.mark.parametrize("damage", ["missing_column", "null_value"])
def test_missing_episode_quantiles_fail_before_video_loading(tmp_path, damage):
    file = metadata_fixture(tmp_path)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(file)
    if damage == "missing_column":
        table = table.drop(["stats/action/q50"])
    else:
        records = table.to_pylist()
        records[0]["stats/action/q50"] = None
        table = pa.Table.from_pylist(records, schema=table.schema)
    pq.write_table(table, file)
    with pytest.raises(ValueError, match="G4"):
        verify_normalization(tmp_path)
