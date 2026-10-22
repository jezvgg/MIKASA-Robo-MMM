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

import flax.nnx as nnx
import openpi.models.pi0_config as pi0_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi import transforms as _transforms
from typing_extensions import override

from mikasa_pi05 import transforms as sd
from mikasa_pi05.dataset import CAMERAS, FIRST_FRAME_KEY, ConcatDataset, SameDrawerDataset

WORK = pathlib.Path(os.environ.get("PI05_WORK", "~/mikasa-pi05")).expanduser()
HF_REPO_ID = "nurtayev-d/samedrawer-1000ep"
HF_REVISION = "d126ebae7e8aa4217c2a61a5b08214fc75f1bdac"
FIRST_FRAME_CAMERA = "left_base_camera_link"
ACTION_HORIZON = 10
BASE_WEIGHTS = "gs://openpi-assets/checkpoints/pi05_base/params"
# One set of norm stats over all 1000 episodes, shared by every config with ACTION_HORIZON (also
# the overfit one). Arm actions are deltas from the current joints, so a longer chunk has wider
# action stats and gets its own directory (H50_NORM_STATS_DIR).
NORM_STATS_DIR = WORK / "assets" / "samedrawer"
H50_NORM_STATS_DIR = WORK / "assets" / "samedrawer-h50"

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


def _model(action_horizon: int = ACTION_HORIZON, **kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=action_horizon,
        discrete_state_input=True,
        image_keys=kwargs.pop("image_keys", sd.IMAGE_KEYS),
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
        # SigLIP stays at its pretrained weights (as in RoboMME's and the lab's pi0.5 baselines);
        # openpi keeps frozen params in bf16 and computes no gradients or optimizer state for them.
        freeze_filter=nnx_utils.PathRegex(".*img.*"),
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
        # A params-only snapshot every 1000 steps for the progress SR (scripts/sr_monitor.sh);
        # the full checkpoint (optimizer state, for --resume) every FULL_CHECKPOINT_EVERY steps.
        save_interval=1_000,
        keep_period=None,
        ema_decay=0.999,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6
        ),
    ),
}

# The 4x H100 baseline with action chunks of 50 (openpi's default horizon) instead of 10.
# Evaluate it with --replan-steps 25: SR_EVAL_ARGS and FINAL_EVAL_ARGS of scripts/run_4xh100.sh.
CONFIGS["pi05_sd_ff_h50_4xh100"] = dataclasses.replace(
    CONFIGS["pi05_sd_ff_4xh100"],
    name="pi05_sd_ff_h50_4xh100",
    model=_model(action_horizon=50),
    data=SameDrawerDataConfig(
        assets=_config.AssetsConfig(assets_dir=str(H50_NORM_STATS_DIR))),
    policy_metadata={**POLICY_METADATA, "action_horizon": 50},
)


# --- MemER's low-level policy (Sridhar et al., arXiv 2510.20328, §3.3 + Table 5) -----------------
# pi0.5 fine-tuned from the public pi0.5-DROID checkpoint on (current images, state, the frame's
# subtask) -> action chunk, one policy over all tasks. The memer repo has no low-level trainer
# ("start from openpi, put the subtask in the prompt"), so this is openpi's pi0.5 fine-tune with:
#   Table 5: batch 128, 18,000 steps, AdamW b1 0.9 b2 0.95, weight decay 0, clip 1, cosine with
#            warmup ratio 0.033 (594 steps), every parameter trained;
#   §4:      chunks of 15 actions (pi05_droid's horizon);
#   openpi defaults where Table 5 says nothing: peak LR 2.5e-5 decaying to 2.5e-6 and EMA 0.99
#            (CosineDecaySchedule/TrainConfig, as pi05_droid_finetune leaves them), weight decay
#            1e-10 (openpi's stand-in for 0);
#   ours: the three Fetch cameras in pi0.5's own three slots, our 13D actions with their own norm
#         stats (DROID's stats are for DROID's actions), the six tasks' labelled copies
#         (vla/memer/label_subtasks.py) under $MEMER_DATA_ROOT (default ~/datasets/memer).
MEMER_TASKS = ("samedrawer", "seasondish", "cabinetsearch", "depthrecall", "takeitback", "veggie")
MEMER_ACTION_HORIZON = 15
MEMER_STEPS = 18_000
MEMER_NORM_STATS_DIR = WORK / "assets" / "memer-lowlevel"
DROID_WEIGHTS = "gs://openpi-assets/checkpoints/pi05_droid/params"
MEMER_POLICY_METADATA = {
    "mikasa_data": MIKASA_DATA,
    "prompt": "MemER subtask (meta/subtasks.parquet)",
    "action_horizon": MEMER_ACTION_HORIZON,
}


def memer_data_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("MEMER_DATA_ROOT", "~/datasets/memer")).expanduser()


def make_memer_dataset(data_config, action_horizon, model_config, *, root, tasks, episodes=None, load_images=True):
    del data_config, model_config   # `episodes`: the same indices of every task; None = all
    return ConcatDataset([
        SameDrawerDataset(pathlib.Path(root) / task, action_horizon, first_frame=False, prompt="subtask",
                          episodes=None if episodes is None else list(episodes), load_images=load_images)
        for task in tasks
    ])


@dataclasses.dataclass(frozen=True)
class MemerLowLevelDataConfig(_config.DataConfigFactory):
    repo_id: str = "memer-lowlevel"  # the asset id of the norm stats
    assets: _config.AssetsConfig = dataclasses.field(
        default_factory=lambda: _config.AssetsConfig(assets_dir=str(MEMER_NORM_STATS_DIR)))
    data_root: str | None = None
    tasks: tuple[str, ...] = MEMER_TASKS
    # Train on these episode indices of every task only (debug runs hold the rest out); None = all.
    episodes: tuple[int, ...] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        data_transforms = _transforms.Group(
            inputs=[sd.SameDrawerInputs(first_frame=False), _transforms.DeltaActions(sd.DELTA_ACTION_MASK)],
            outputs=[_transforms.AbsoluteActions(sd.DELTA_ACTION_MASK), sd.SameDrawerOutputs()],
        )
        factory = functools.partial(make_memer_dataset, root=str(self.data_root or memer_data_root()),
                                    tasks=self.tasks, episodes=self.episodes)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
            dataset_factory=factory,
        )


def _memer_config(name: str, **kwargs) -> _config.TrainConfig:
    defaults = dict(
        model=_model(action_horizon=MEMER_ACTION_HORIZON, image_keys=sd.CURRENT_IMAGE_KEYS),
        data=MemerLowLevelDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader(DROID_WEIGHTS),
        policy_metadata=MEMER_POLICY_METADATA,
        freeze_filter=nnx.Nothing(),   # Table 5 freezes nothing (openpi's default)
        num_train_steps=MEMER_STEPS,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=round(0.033 * MEMER_STEPS), peak_lr=2.5e-5, decay_steps=MEMER_STEPS, decay_lr=2.5e-6),
    )
    defaults.update(kwargs)
    return _train_config(name, **defaults)


CONFIGS["memer_lowlevel_4xh100"] = _memer_config(
    "memer_lowlevel_4xh100", batch_size=128, fsdp_devices=4, num_workers=16, save_interval=1_000, keep_period=None)
# One-GPU smoke: the same model, data and optimizer; batch 4, 3 steps, no EMA copy.
CONFIGS["memer_lowlevel_smoke_1xh100"] = _memer_config(
    "memer_lowlevel_smoke_1xh100", batch_size=4, num_workers=4, num_train_steps=3, save_interval=3,
    keep_period=None, ema_decay=None)
# One-GPU debug (owner 2026-10-22, vla/memer/debug_1gpu.sh): all six tasks, episodes 0-979 of each
# (980-999 held out for tests/check_memer_action_error.py), batch 32, 2000 steps, peak LR 1e-4 (4x
# the full run's) for a quick learning signal, no EMA. Not the paper's recipe; a pipeline check.
MEMER_DEBUG_TRAIN_EPISODES = tuple(range(980))
CONFIGS["memer_lowlevel_debug_1xh100"] = _memer_config(
    "memer_lowlevel_debug_1xh100", data=MemerLowLevelDataConfig(episodes=MEMER_DEBUG_TRAIN_EPISODES),
    batch_size=32, num_workers=6, num_train_steps=2_000, save_interval=2_000, keep_period=None, ema_decay=None,
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=1e-4, decay_steps=2_000, decay_lr=1e-5))


# Steps between full checkpoints (with optimizer state) for configs that snapshot more often.
# The dummy config uses the same scheme so the whole flow can be checked on a small GPU.
FULL_CHECKPOINT_EVERY = {"pi05_sd_ff_4xh100": 5_000, "pi05_sd_ff_h50_4xh100": 5_000, "pi05_sd_ff_dummy": 50,
                         "memer_lowlevel_4xh100": 6_000}

# Configs meant for a single GPU with a small disk (lab server: 100 GB) save params-only checkpoints.
PARAMS_ONLY_CHECKPOINTS = {"pi05_sd_ff_dummy", "pi05_sd_ff_overfit", "pi05_sd_ff_1xh100", "memer_lowlevel_smoke_1xh100",
                           "memer_lowlevel_debug_1xh100"}


def get_config(name: str) -> _config.TrainConfig:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config {name!r}; choose from {sorted(CONFIGS)}")
    return CONFIGS[name]
