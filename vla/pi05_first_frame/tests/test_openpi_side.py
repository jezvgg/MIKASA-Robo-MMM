"""Checks of the openpi-side pipeline on the real dataset (run in the openpi venv).

    SAMEDRAWER_DATASET_DIR=... PI05_WORK=... OPENPI_DATA_HOME=... \
        $OPENPI_DIR/.venv/bin/python -m pytest vla/pi05_first_frame/tests -q

Norm stats must exist (python -m mikasa_pi05 norm-stats pi05_sd_ff_4xh100).
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SAMEDRAWER_DATASET_DIR"), reason="SAMEDRAWER_DATASET_DIR is not set"
)


@pytest.fixture(scope="module")
def config():
    from mikasa_pi05.configs import get_config

    return get_config("pi05_sd_ff_4xh100")


@pytest.fixture(scope="module")
def data_config(config):
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        pytest.skip("norm stats missing")
    return data_config


@pytest.fixture(scope="module")
def dataset(config, data_config):
    return data_config.dataset_factory(data_config, config.model.action_horizon, config.model)


def _indices(dataset, n=24, seed=0):
    rng = np.random.default_rng(seed)
    starts = [dataset.episode_range[e][0] for e in dataset.episodes[:3]]
    ends = [dataset.episode_range[e][1] - 1 for e in dataset.episodes[:3]]
    return sorted({*rng.integers(0, len(dataset), n).tolist(), *starts, *ends})


def test_first_frame_is_frame_zero_of_same_episode(dataset):
    for index in _indices(dataset, n=8):
        sample = dataset[index]
        row = int(dataset.rows[index])
        episode = int(dataset.episode_of_frame[row])
        expected = dataset.image(episode, 0, dataset.first_frame_camera)
        np.testing.assert_array_equal(sample["observation.first_frame"], expected)


def test_action_chunk_is_clamped_to_episode(dataset):
    episode = dataset.episodes[0]
    start, end = dataset.episode_range[episode]
    sample = dataset[end - 1 - start]
    np.testing.assert_array_equal(sample["actions"], np.repeat(dataset.action[end - 1][None], 10, axis=0))
    sample = dataset[0]
    np.testing.assert_array_equal(sample["actions"], dataset.action[start:start + 10])


def test_transforms_round_trip(dataset, data_config):
    """Delta -> normalize -> unnormalize -> absolute returns the recorded actions."""
    from openpi import transforms as _transforms

    inputs = [*data_config.data_transforms.inputs,
              _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)]
    outputs = [_transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
               *data_config.data_transforms.outputs]
    for index in _indices(dataset):
        sample = dataset[index]
        expected = sample["actions"].copy()
        item = copy.deepcopy(sample)
        for step in inputs:
            item = step(item)
        assert set(item["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb"}
        out = {"state": item["state"], "actions": item["actions"]}
        for step in outputs:
            out = step(out)
        np.testing.assert_allclose(out["actions"], expected, atol=1e-5)


def test_prompt_with_state_fits_token_budget(dataset, data_config, config):
    from openpi import transforms as _transforms

    steps = [*data_config.data_transforms.inputs,
             _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
             *data_config.model_transforms.inputs]
    longest = 0
    for index in _indices(dataset, n=64, seed=1):
        item = dataset[index]
        for step in steps:
            item = step(item)
        used = int(np.asarray(item["tokenized_prompt_mask"]).sum())
        longest = max(longest, used)
        assert set(item["image"]) == set(config.model.image_keys)
        assert all(img.shape == (224, 224, 3) for img in item["image"].values())
    assert longest < config.model.max_token_len, f"prompt uses {longest} of {config.model.max_token_len} tokens"


def test_model_spec_has_four_images(config):
    observation, actions = config.model.inputs_spec(batch_size=2)
    assert list(observation.images) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb"]
    assert actions.shape == (2, 10, 32)
