import json
from types import SimpleNamespace

import pytest

from utils.trajectory_metadata import (
    patch_trajectory_metadata,
    resolve_sim_backend,
    validate_rlds_metadata,
    validate_trajectory_metadata,
)


def _trajectory_metadata():
    return {
        "env_info": {"env_id": "test-env", "env_kwargs": {"sim_backend": "auto"}},
        "episodes": [
            {
                "episode_id": 0,
                "episode_seed": 102,
                "reset_kwargs": {"seed": [200]},
            }
        ],
    }


def test_resolve_sim_backend_returns_physical_backend():
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            backend=SimpleNamespace(sim_backend="physx_cuda")
        )
    )

    assert resolve_sim_backend(env) == "physx_cuda"


def test_resolve_sim_backend_rejects_unresolved_auto():
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(backend=SimpleNamespace(sim_backend="auto"))
    )

    with pytest.raises(ValueError, match="could not resolve a physical sim_backend"):
        resolve_sim_backend(env)


def test_patch_trajectory_metadata_repairs_seed_and_records_backend(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(_trajectory_metadata()))

    patch_trajectory_metadata(
        path,
        "physx_cpu",
        episode_seed=200,
        requested_sim_backend="auto",
    )
    metadata = json.loads(path.read_text())
    episode = metadata["episodes"][0]

    assert metadata["sim_backend"] == "physx_cpu"
    assert metadata["requested_sim_backend"] == "auto"
    assert metadata["env_info"]["env_kwargs"]["sim_backend"] == "physx_cpu"
    assert episode["episode_seed"] == episode["source_seed"] == 200
    assert episode["reset_kwargs"]["seed"] == 200
    assert episode["sim_backend"] == "physx_cpu"
    assert validate_trajectory_metadata(metadata, path) == "physx_cpu"


def test_patch_trajectory_metadata_accepts_cli_backend_alias(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(_trajectory_metadata()))

    patch_trajectory_metadata(
        path, "physx_cpu", requested_sim_backend="cpu"
    )
    metadata = json.loads(path.read_text())

    assert metadata["sim_backend"] == "physx_cpu"
    assert metadata["requested_sim_backend"] == "cpu"


def test_patch_trajectory_metadata_rejects_backend_or_seed_mismatch(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(_trajectory_metadata()))

    with pytest.raises(ValueError, match="requested sim_backend='physx_cuda'"):
        patch_trajectory_metadata(
            path, "physx_cpu", requested_sim_backend="physx_cuda"
        )

    with pytest.raises(ValueError, match="does not match expected episode_seed 201"):
        patch_trajectory_metadata(path, "physx_cpu", episode_seed=201)


def test_trajectory_metadata_rejects_duplicate_or_unverifiable_seeds():
    metadata = _trajectory_metadata()
    metadata["sim_backend"] = "physx_cpu"
    metadata["episodes"][0].update(
        episode_seed=200, source_seed=200, sim_backend="physx_cpu"
    )
    metadata["episodes"].append(
        {
            "episode_id": 1,
            "episode_seed": 200,
            "source_seed": 200,
            "sim_backend": "physx_cpu",
        }
    )
    with pytest.raises(ValueError, match="duplicate episode_seed 200"):
        validate_trajectory_metadata(metadata, "test-trajectory")

    metadata["episodes"] = [{"episode_id": 0, "episode_seed": 200}]
    with pytest.raises(ValueError, match="no recorded reset_kwargs.seed or source_seed"):
        validate_trajectory_metadata(metadata, "test-trajectory")


def test_rlds_metadata_requires_complete_consistent_g5_arrays():
    episode = {
        "episode_seed": 200,
        "episode_length": 12,
        "duration_s": 0.6,
        "success": True,
        "success_once": True,
        "reward_sum": 2.0,
        "terminated": True,
        "truncated": False,
        "sim_backend": "physx_cpu",
        "source_seed": 200,
        "reset_kwargs": {"seed": 200},
    }
    metadata = {
        "sim_backend": "physx_cpu",
        "episodes": [episode],
        "episode_seeds": [200],
        "episode_lengths": [12],
        "episode_durations_s": [0.6],
        "success_once": [True],
        "reward_sums": [2.0],
    }

    assert validate_rlds_metadata(metadata, "test-rlds") == "physx_cpu"
    del metadata["reward_sums"]
    with pytest.raises(ValueError, match="reward_sums with one value per episode"):
        validate_rlds_metadata(metadata, "test-rlds")
