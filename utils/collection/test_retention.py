"""Retention must preserve physics data and fail closed on corrupted inputs."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from utils.collection import retention
from utils.collection.source_storage import file_sha256, numerical_source


def test_compaction_preserves_raw_states_actions_and_source(tmp_path):
    source = tmp_path / "render.h5"
    target = tmp_path / "numeric.h5"
    actions = np.arange(52, dtype=np.float32).reshape(4, 13)
    with h5py.File(source, "w") as f:
        f.attrs["origin"] = "physical replay"
        f["traj_0/actions"] = actions
        f["traj_0/qpos"] = np.arange(75, dtype=np.float32).reshape(5, 15)
        f["traj_0/env_states/drawer"] = np.linspace(0, .2, 5)
        f["traj_0/obs/agent/qpos"] = f["traj_0/qpos"][:]
        f["traj_0/obs/sensor_data/wrist/rgb"] = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        f["schema_scalar"] = 7
    source.with_suffix(".json").write_text(json.dumps({"seed": 17}))
    digest = file_sha256(source)
    retention.compact_recording(source, target)
    assert file_sha256(source) == digest
    with h5py.File(target) as f:
        np.testing.assert_array_equal(f["traj_0/actions"], actions)
        assert "traj_0/env_states/drawer" in f
        assert "traj_0/obs/agent/qpos" in f
        assert "traj_0/obs/sensor_data" not in f
        assert f["schema_scalar"][()] == 7
        assert f.attrs["origin"] == "physical replay"


def test_modified_numeric_recording_is_rejected(tmp_path):
    p = tmp_path / "numeric.h5"
    p.write_bytes(b"verified numerical recording")
    p.with_suffix(".json").write_text('{}')
    episode = dict(source_h5="unused", numeric_source_h5=str(p),
                   numeric_source_sha256=file_sha256(p),
                   numeric_metadata_sha256=file_sha256(p.with_suffix(".json")))
    assert numerical_source(episode) == p
    p.write_bytes(b"changed numerical recording")
    with pytest.raises(ValueError, match="numerical recording changed"):
        numerical_source(episode)


def test_pruning_rejects_physical_source_even_if_other_checks_pass(tmp_path, monkeypatch):
    p = tmp_path / "oracle/9/trajectory.h5"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"physical source must survive")
    metadata = dict(source_root=str(tmp_path), episodes=[dict(scene_seed=9, source_h5=str(p))])
    (tmp_path / "source_h5_metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "retention.json").write_text('{}')
    monkeypatch.setattr(retention, "verify", lambda root: None)
    monkeypatch.setattr(retention, "numerical_source", lambda episode: Path("verified"))
    with pytest.raises(ValueError, match="restricted"):
        retention.prune(tmp_path)
    assert p.read_bytes() == b"physical source must survive"
