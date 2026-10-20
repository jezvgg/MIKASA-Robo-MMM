"""openpi transforms for SameDrawer: three current cameras plus the episode's first frame.

The client (eval_samedrawer.py) and the dataset (dataset.py) send the same keys;
SameDrawerInputs maps them to model image slots and reorders the state to the
action order so that DeltaActions can subtract the matching joint positions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms

from mikasa_pi05.dataset import FIRST_FRAME_KEY, IMAGE_PREFIX, STATE_KEY

ACTION_DIM = 13
STATE_DIM = 12

# Dataset state = qpos[3:] in qpos order:
#   torso_lift, head_pan, shoulder_pan, head_tilt, shoulder_lift, upperarm_roll,
#   elbow_flex, forearm_roll, wrist_flex, wrist_roll, r_gripper_finger, l_gripper_finger
# Action order: 7 arm joints, gripper, head_pan, head_tilt, torso_lift, base forward, base yaw.
# After this permutation state[i] is the joint that action[i] targets for i in 0..6 and 8..10;
# state[7] and state[11] are the two finger positions (the gripper action is a normalized command).
STATE_TO_ACTION_ORDER = (2, 4, 5, 6, 7, 8, 9, 10, 1, 3, 0, 11)

# Joint targets become deltas from the current joint position; the gripper command and
# the base velocities stay absolute. Length 11 leaves the two base channels untouched.
DELTA_ACTION_MASK = transforms.make_bool_mask(7, -1, 3)

# Model image slots, in prefix-token order. Slot names keep pi0.5's convention: keys
# without "wrist" get the random crop/rotate augmentation in training.
IMAGE_SLOTS = {
    "base_0_rgb": IMAGE_PREFIX + "left_base_camera_link",
    "left_wrist_0_rgb": IMAGE_PREFIX + "fetch_hand",
    "right_wrist_0_rgb": IMAGE_PREFIX + "right_base_camera_link",
    "base_1_rgb": FIRST_FRAME_KEY,
}
IMAGE_KEYS = tuple(IMAGE_SLOTS)


def _image(value) -> np.ndarray:
    image = np.asarray(value)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected an HxWx3 uint8 image, got {image.shape} {image.dtype}")
    return image


@dataclasses.dataclass(frozen=True)
class SameDrawerInputs(transforms.DataTransformFn):
    """Wire format -> openpi model inputs. Images are optional only for the norm-stats pass."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data[STATE_KEY], dtype=np.float32)
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected a {STATE_DIM}-dim state, got {state.shape}")
        inputs = {"state": state[..., list(STATE_TO_ACTION_ORDER)]}
        if any(key in data for key in IMAGE_SLOTS.values()):
            inputs["image"] = {slot: _image(data[key]) for slot, key in IMAGE_SLOTS.items()}
            inputs["image_mask"] = {slot: np.True_ for slot in IMAGE_SLOTS}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"Expected {ACTION_DIM}-dim actions, got {actions.shape}")
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SameDrawerOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
