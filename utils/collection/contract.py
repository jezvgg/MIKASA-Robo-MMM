"""Data checks shared by the simulator pipeline and the offline exporter."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ROBOT = "ds_fetch"
CONTROL_MODE = "pd_joint_pos"
CONTROL_FPS = 20
POLICY_FPS = 10
ACTION_REPEAT = CONTROL_FPS // POLICY_FPS
CAMERAS = {
    "left_base_camera_link": (256, 256, 3),
    "right_base_camera_link": (256, 256, 3),
    "fetch_hand": (128, 128, 3),
}
ACTION_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "gripper",
    "head_pan_joint",
    "head_tilt_joint",
    "torso_lift_joint",
    "base_forward_velocity",
    "base_yaw_velocity",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    """Publish only complete metadata, including after an interrupted worker."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def check_actions(actions: np.ndarray) -> None:
    if actions.ndim != 2 or actions.shape[1] != 13 or not len(actions):
        raise ValueError(f"Expected nonempty (T, 13) actions, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Non-finite actions")
    # Arm and body targets are absolute radians/metres, NOT all in [-1, 1].
    if np.any(np.abs(actions[:, [7, 11, 12]]) > 1 + 1e-6):
        raise ValueError("Normalized gripper/base channels exceed [-1, 1]")


def held_actions(actions: np.ndarray) -> np.ndarray:
    """First target of each pair, held for two 20 Hz steps (10 Hz policy).

    An odd final source step is padded by one hold (0.05 s), never dropped.
    No averaging, clipping or delta-action summation is valid for this controller.
    Success must be measured again after executing the returned actions.
    """
    check_actions(actions)
    return np.repeat(actions[::ACTION_REPEAT], ACTION_REPEAT, axis=0)


def episode_summary(trajectory: h5py.Group) -> dict:
    """Summarize actual control-step flags; ever-success is not final success."""
    n = len(trajectory["actions"])
    successes = np.asarray(trajectory["success"][:], dtype=bool)
    if successes.shape != (n,):
        raise ValueError("Success history must have one flag per control step")
    return dict(control_steps=n, duration_seconds=n / CONTROL_FPS,
                reward_sum=float(trajectory["rewards"][:].sum()),
                success=bool(successes[-1]) if n else False,
                success_once=bool(successes.any()))


def recording_info(path: Path, *, stage: str) -> dict:
    """Reject mismatched, partial or failed recordings before consuming them."""
    meta = read_json(path.with_suffix(".json"))
    kwargs = meta["env_info"]["env_kwargs"]
    contract = meta["mikasa_data"]
    if kwargs.get("robot_uids") != ROBOT:
        raise ValueError("Recording uses a different robot")
    if kwargs.get("control_mode") != CONTROL_MODE:
        raise ValueError("Only pd_joint_pos recordings are supported")
    if kwargs.get("sim_config", {}).get("control_freq") != CONTROL_FPS:
        raise ValueError("Recording must explicitly specify control_freq=20")
    if contract.get("stage") != stage or contract.get("version") != 3:
        raise ValueError("Missing or incompatible collection contract")
    native = stage in {"oracle", "native", "native_rgb"}
    expected_timing = (CONTROL_FPS, 1) if native else (POLICY_FPS, ACTION_REPEAT)
    if (contract.get("policy_fps"), contract.get("action_repeat")) != expected_timing:
        raise ValueError("Recording has incompatible policy timing")
    if not contract.get("instruction"):
        raise ValueError("Missing language instruction")
    if len(meta["episodes"]) != 1:
        raise ValueError("Expected one episode per file")
    episode = meta["episodes"][0]
    if episode["episode_id"] != 0 or episode["control_mode"] != CONTROL_MODE:
        raise ValueError("Unexpected episode id or controller")
    with h5py.File(path, "r") as h5:
        if set(h5) != {"traj_0"}:
            raise ValueError("Incomplete single-episode H5")
        traj = h5["traj_0"]
        actions = traj["actions"][:]
        check_actions(actions)
        n = len(actions)
        if episode["elapsed_steps"] != n:
            raise ValueError("Action count disagrees with metadata")
        if traj["success"].shape != (n,) or not bool(traj["success"][-1]):
            raise ValueError("Episode did not finish successfully")
        if not episode.get("success", False):
            raise ValueError("Success metadata disagrees with H5")
        summary = episode_summary(traj)
        for key in ("success", "success_once"):
            if contract.get(key) != summary[key] or episode.get(key) != summary[key]:
                raise ValueError(f"{key} metadata is missing or disagrees with H5")

        def check_state(name, value):
            if isinstance(value, h5py.Dataset) and value.shape[0] != n + 1:
                raise ValueError(f"env_states/{name} needs T+1 samples")

        traj["env_states"].visititems(check_state)
        if contract.get("explicit_time_and_proprio", False):
            np.testing.assert_allclose(traj["timestamp"][:], np.arange(n+1) / CONTROL_FPS, atol=1e-9, rtol=0)
            qpos = traj["qpos"][:]
            if qpos.shape != (n+1, 15):
                raise ValueError("Expected explicit T+1 raw qpos")
            np.testing.assert_array_equal(traj["proprio"][:], qpos[:, 3:])
            np.testing.assert_array_equal(traj["global_state"][:], qpos[:, :3])
        if not native:
            if n % ACTION_REPEAT or not np.array_equal(actions, held_actions(actions)):
                raise ValueError("Actions were not validated with two-step holds")
        if stage in {"rgb", "native_rgb"}:
            qpos = traj["obs/agent/qpos"]
            if qpos.shape != (n + 1, 15):
                raise ValueError("Expected T+1 observations with 15 joint positions")
            if not np.isfinite(qpos[:]).all():
                raise ValueError("Non-finite joint positions")
            if set(traj["obs/sensor_data"]) != set(CAMERAS):
                raise ValueError("Camera rig differs from the collection contract")
            for camera, shape in CAMERAS.items():
                rgb = traj[f"obs/sensor_data/{camera}/rgb"]
                if rgb.shape != (n + 1, *shape) or rgb.dtype != np.uint8:
                    raise ValueError(f"Invalid RGB shape/dtype for {camera}")
    return meta
