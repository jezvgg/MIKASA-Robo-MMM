"""OpenPI transforms and training config for the project Fetch/LeRobot contract."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from typing_extensions import override

from openpi import transforms
from openpi.models import model as model_lib
from openpi.training import config as openpi_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "pi05_fetch"
ACTION_DIM = 13
STATE_DIM = 12
MODEL_DIM = 32


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb_hwc(value) -> np.ndarray:
    image = _as_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"expected a single 3D RGB image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"expected RGB channels, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size and image.max() <= 1.0:
            image *= 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class FetchInputs(transforms.DataTransformFn):
    """Map dataset/eval keys to Pi0.5's observation structure."""

    def __call__(self, data: dict) -> dict:
        state = _as_numpy(data["observation/state"]).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"expected {STATE_DIM}D robot state, got {state.shape}")

        result = {
            "state": state,
            "image": {
                "base_0_rgb": _rgb_hwc(data["observation/image"]),
                "left_wrist_0_rgb": _rgb_hwc(data["observation/hand_image"]),
                "right_wrist_0_rgb": _rgb_hwc(data["observation/side_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.bool_(True),
                "left_wrist_0_rgb": np.bool_(True),
                "right_wrist_0_rgb": np.bool_(True),
            },
        }
        if "actions" in data:
            actions = _as_numpy(data["actions"]).astype(np.float32)
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"expected {ACTION_DIM}D actions, got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            result["prompt"] = str(prompt)
        return result


@dataclasses.dataclass(frozen=True)
class FetchOutputs(transforms.DataTransformFn):
    """Convert padded Pi0.5 output back to the environment's 13D action."""

    def __call__(self, data: dict) -> dict:
        actions = _as_numpy(data["actions"]).astype(np.float32)
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(
                f"expected at least {ACTION_DIM} action values, got {actions.shape}"
            )
        return {"actions": actions[..., :ACTION_DIM]}


@dataclasses.dataclass(frozen=True)
class FetchLeRobotDataConfig(openpi_config.DataConfigFactory):
    @override
    def create(
        self, assets_dirs: Path, model_config: model_lib.BaseModelConfig
    ) -> openpi_config.DataConfig:
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.left_base_camera_link",
                        "observation/hand_image": "observation.images.fetch_hand",
                        "observation/side_image": "observation.images.right_base_camera_link",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=transforms.Group(
                inputs=[FetchInputs()], outputs=[FetchOutputs()]
            ),
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


def make_train_config(
    repo_id: str,
    exp_name: str,
    *,
    action_horizon: int = 10,
    batch_size: int = 8,
    num_workers: int = 2,
    num_train_steps: int = 30_000,
    lora: bool = False,
) -> openpi_config.TrainConfig:
    """Build project config from OpenPI's official Pi0.5 fine-tuning defaults."""
    base = openpi_config.get_config("pi05_libero")
    model = dataclasses.replace(
        base.model,
        action_dim=MODEL_DIM,
        action_horizon=action_horizon,
        discrete_state_input=False,
    )
    if lora:
        model = dataclasses.replace(
            model,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )

    return dataclasses.replace(
        base,
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        data=FetchLeRobotDataConfig(
            repo_id=repo_id,
            base_config=openpi_config.DataConfig(prompt_from_task=True),
        ),
        freeze_filter=model.get_freeze_filter() if lora else base.freeze_filter,
        ema_decay=None if lora else base.ema_decay,
        assets_base_dir=str(REPO_ROOT / "logs/openpi/assets"),
        checkpoint_base_dir=str(REPO_ROOT / "logs/openpi/checkpoints"),
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        wandb_enabled=False,
    )


def self_check() -> None:
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    sample = {
        "observation/state": np.zeros(STATE_DIM, dtype=np.float32),
        "observation/image": frame,
        "observation/hand_image": np.zeros((3, 8, 12), dtype=np.float32),
        "observation/side_image": frame,
        "actions": np.zeros((10, ACTION_DIM), dtype=np.float32),
        "prompt": "place the cup on the tray",
    }
    model_input = FetchInputs()(sample)
    assert model_input["state"].shape == (STATE_DIM,)
    assert model_input["image"]["left_wrist_0_rgb"].shape == (8, 12, 3)
    assert model_input["actions"].shape == (10, ACTION_DIM)
    assert FetchOutputs()({"actions": np.zeros((10, MODEL_DIM))})["actions"].shape == (
        10,
        ACTION_DIM,
    )


if __name__ == "__main__":
    self_check()
    print("OpenPI Fetch transforms OK")
