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


#: How a 10 Hz target becomes two 20 Hz control steps (profile data.action_execution). "hold" repeats it.
#: "midpoint" (SeasonDish from profile 29, owner 2026-10-28 «20 герц же получаются из 10 усреднением» → «Усреднение»)
#: executes it, then the midpoint to the next target; the gripper holds its target, since a half-way gripper command
#: would start the fingers one step before the recorded switch (checklist D7). The last target, with no next, is held.
EXECUTIONS = ("hold", "midpoint")


def midpoint_actions(targets: np.ndarray) -> np.ndarray:
    """The 20 Hz steps of 10 Hz targets under "midpoint" execution.

    Example:
        >>> a = np.zeros((3, 13), np.float32); a[:, 0] = [0., 1., 3.]; a[:, 7] = [-1., 1., 1.]
        >>> m = midpoint_actions(a); m[:, 0].tolist(), m[:, 7].tolist()
        ([0.0, 0.5, 1.0, 2.0, 3.0, 3.0], [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    """
    targets = np.asarray(targets, dtype=np.float32)
    check_actions(targets)
    following = np.concatenate([targets[1:], targets[-1:]])
    middle = (targets + following) * np.float32(0.5)
    middle[:, 7] = targets[:, 7]
    steps = np.empty((2 * len(targets), targets.shape[1]), dtype=np.float32)
    steps[0::2], steps[1::2] = targets, middle
    return steps


def control_actions(actions: np.ndarray, execution: str = "hold", length: int | None = None) -> np.ndarray:
    """The 20 Hz steps a validated replay executes for a recording's 20 Hz `actions` (each pair's first is the 10 Hz
    target). `length` > the replay of `actions` pads it with the last target held, as the midpoint replay may run on
    after the source until the task's final hold succeeds (`pipeline.validate_one`).

    Example:
        >>> a = np.zeros((4, 13), np.float32); a[:, 0] = [0., 0., 2., 2.]; a[:, 7] = 1.
        >>> control_actions(a, "midpoint", length=6)[:, 0].tolist()
        [0.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    """
    if execution not in EXECUTIONS:
        raise ValueError(f"Unknown action execution: {execution}")
    targets = np.asarray(actions)[::ACTION_REPEAT]
    if length is not None and length > ACTION_REPEAT * len(targets):
        extra = (length - ACTION_REPEAT * len(targets) + ACTION_REPEAT - 1) // ACTION_REPEAT
        targets = np.concatenate([targets, np.repeat(targets[-1:], extra, axis=0)])
    if execution == "hold":
        return np.repeat(np.asarray(targets, dtype=np.float32), ACTION_REPEAT, axis=0) if length else held_actions(actions)
    return midpoint_actions(targets)


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
        if "truncated" not in traj or traj["truncated"].shape != (n,):
            raise ValueError("Recording needs one truncation flag per control step")
        if np.asarray(traj["truncated"][:], dtype=bool).any():
            raise ValueError("Truncated recordings are diagnostic only")
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
            execution = contract.get("action_execution", "hold")
            if n % ACTION_REPEAT or not np.array_equal(actions, control_actions(actions, execution)):
                raise ValueError(f"Actions were not validated with the {execution} execution of 10 Hz targets")
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
