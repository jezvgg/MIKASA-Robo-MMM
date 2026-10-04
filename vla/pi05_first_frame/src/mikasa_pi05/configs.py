"""openpi TrainConfigs for the SameDrawer first-frame baseline.

Paths come from the environment (see scripts/server.env.example):
    PI05_WORK                  work directory: assets/samedrawer/ (norm stats) and checkpoints/
    SAMEDRAWER_DATASET_DIR     local copy of nurtayev-d/samedrawer-1000ep (LeRobot v3.0)
No variable starts with MIKASA_: the simulator's pinned profile refuses those.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import pathlib

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi import transforms as _transforms
from typing_extensions import override

from mikasa_pi05 import transforms as sd
from mikasa_pi05.dataset import CAMERAS, FIRST_FRAME_KEY, SameDrawerDataset

WORK = pathlib.Path(os.environ.get("PI05_WORK", "~/mikasa-pi05")).expanduser()
HF_REPO_ID = "nurtayev-d/samedrawer-1000ep"
HF_REVISION = "d126ebae7e8aa4217c2a61a5b08214fc75f1bdac"
FIRST_FRAME_CAMERA = "left_base_camera_link"
ACTION_HORIZON = 10
BASE_WEIGHTS = "gs://openpi-assets/checkpoints/pi05_base/params"
# One set of norm stats over all 1000 episodes, shared by every config (also the overfit one).
NORM_STATS_DIR = WORK / "assets" / "samedrawer"

# Must equal utils.collection.client.policy_metadata(); run_policy_episode refuses a mismatch.
MIKASA_DATA = {
    "robot": "ds_fetch",
    "control_mode": "pd_joint_pos",
    "control_fps": 20,
    "policy_fps": 10,
    "action_repeat": 2,
    "action_dim": sd.ACTION_DIM,
    "state_dim": sd.STATE_DIM,
    "cameras": {
        "left_base_camera_link": [256, 256, 3],
        "right_base_camera_link": [256, 256, 3],
        "fetch_hand": [128, 128, 3],
    },
}
POLICY_METADATA = {
    "mikasa_data": MIKASA_DATA,
    "first_frame": {"key": FIRST_FRAME_KEY, "camera": FIRST_FRAME_CAMERA, "frame": 0},
    "action_horizon": ACTION_HORIZON,
}


def dataset_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("SAMEDRAWER_DATASET_DIR", WORK / "data/samedrawer-1000ep")).expanduser()


def make_dataset(data_config, action_horizon, model_config, *, root, first_frame_camera, episodes, load_images=True):
    del data_config, model_config
    return SameDrawerDataset(
        root,
        action_horizon,
        first_frame_camera=first_frame_camera,
        episodes=list(episodes) if episodes is not None else None,
        load_images=load_images,
    )


@dataclasses.dataclass(frozen=True)
class SameDrawerDataConfig(_config.DataConfigFactory):
    repo_id: str = HF_REPO_ID  # also the asset id of the norm stats
    assets: _config.AssetsConfig = dataclasses.field(
        default_factory=lambda: _config.AssetsConfig(assets_dir=str(NORM_STATS_DIR)))
    # Defaults to $SAMEDRAWER_DATASET_DIR when unset.
    dataset_dir: str | None = None
    first_frame_camera: str = FIRST_FRAME_CAMERA
    # Train on these episode indices only (overfit checks); None = all 1000.
    episodes: tuple[int, ...] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        if self.first_frame_camera not in CAMERAS:
            raise ValueError(f"Unknown camera {self.first_frame_camera!r}")
        data_transforms = _transforms.Group(
            inputs=[sd.SameDrawerInputs(), _transforms.DeltaActions(sd.DELTA_ACTION_MASK)],
            outputs=[_transforms.AbsoluteActions(sd.DELTA_ACTION_MASK), sd.SameDrawerOutputs()],
        )
        factory = functools.partial(
            make_dataset,
            root=str(self.dataset_dir or dataset_dir()),
            first_frame_camera=self.first_frame_camera,
            episodes=self.episodes,
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
            dataset_factory=factory,
        )


def _model(**kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=ACTION_HORIZON,
        discrete_state_input=True,
        image_keys=sd.IMAGE_KEYS,
        **kwargs,
    )


def _train_config(name: str, **kwargs) -> _config.TrainConfig:
    defaults = dict(
        name=name,
        model=_model(),
        data=SameDrawerDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader(BASE_WEIGHTS),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        policy_metadata=POLICY_METADATA,
        checkpoint_base_dir=str(WORK / "checkpoints"),
        wandb_enabled=False,
    )
    defaults.update(kwargs)
    return _config.TrainConfig(**defaults)


# Training episodes 0, 1, 2 and 5 have the cue in drawers 0, 1, 2 and 3 (seeds 9000000, 9000001,
# 9000006, 9000004): one episode per answer, so the overfit run must use the first frame too.
OVERFIT_EPISODES = (0, 1, 2, 5)

CONFIGS = {
    # Pipeline check on a small GPU: tiny Gemma (width 64), full-size SigLIP, no base weights.
    "pi05_sd_ff_dummy": _train_config(
        "pi05_sd_ff_dummy",
        model=_model(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        batch_size=4,
        num_workers=2,
        num_train_steps=50,
        save_interval=25,
        keep_period=None,
        log_interval=5,
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=5, peak_lr=1e-4, decay_steps=50, decay_lr=1e-5),
    ),
    # 1x H100 debug: real model, full fine-tune, short run to measure step time/memory.
    "pi05_sd_ff_1xh100": _train_config(
        "pi05_sd_ff_1xh100",
        batch_size=32,
        num_workers=8,
        num_train_steps=3_000,
        save_interval=1_000,
        keep_period=None,
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=5e-5, decay_steps=3_000, decay_lr=5e-6),
    ),
    # 1x H100 overfit check (~30 min with evaluation): one episode per cue drawer, ~2,230 frames,
    # ~7 passes in 500 steps of 32; the policy must then reproduce those episodes' actions.
    "pi05_sd_ff_overfit": _train_config(
        "pi05_sd_ff_overfit",
        data=SameDrawerDataConfig(episodes=OVERFIT_EPISODES),
        batch_size=32,
        num_workers=8,
        num_train_steps=500,
        save_interval=500,
        keep_period=None,
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=50, peak_lr=5e-5, decay_steps=500, decay_lr=5e-6),
    ),
    # 4x H100 baseline: ~6.8 epochs of the 560,900 frames.
    "pi05_sd_ff_4xh100": _train_config(
        "pi05_sd_ff_4xh100",
        batch_size=128,
        fsdp_devices=4,
        num_workers=16,
        num_train_steps=30_000,
        save_interval=2_000,
        keep_period=5_000,
        ema_decay=0.999,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6
        ),
    ),
}


# Configs meant for a single GPU with a small disk (lab server: 100 GB) save params-only checkpoints.
PARAMS_ONLY_CHECKPOINTS = {"pi05_sd_ff_dummy", "pi05_sd_ff_overfit", "pi05_sd_ff_1xh100"}


def get_config(name: str) -> _config.TrainConfig:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config {name!r}; choose from {sorted(CONFIGS)}")
    return CONFIGS[name]
