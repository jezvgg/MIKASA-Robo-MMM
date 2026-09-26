"""Seeded task-level variation around the robot's unmodified rest keyframe."""

from __future__ import annotations

import numpy as np
import torch

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)
BODY_JOINTS = ("torso_lift_joint", "head_pan_joint", "head_tilt_joint")


def randomize_initial_joints(env, env_idx):
    """Call once at reset, after the task's object/base/answer draws.

    Select by joint name: qpos interleaves arm, torso and head. Preserve the
    task's base pose and finger opening. Sample uniformly inside the intersection
    of the requested rest-pose neighbourhood and the robot's existing limits;
    sampling the interval avoids a pile-up of clamped poses at a joint limit.
    Each environment uses its own episode RNG, including during partial resets.
    Configuration is part of the task dataclass and recorded runtime signature.
    """
    cfg, robot = env.cfg, env.agent.robot
    scales = [cfg.initial_arm_jitter_rad] * len(ARM_JOINTS) + [
        cfg.initial_torso_jitter_m,
        cfg.initial_head_jitter_rad,
        cfg.initial_head_jitter_rad,
    ]
    if not np.isfinite(scales).all() or min(scales) < 0:
        raise ValueError("Initial joint jitter must be finite and nonnegative")
    names = [joint.name for joint in robot.get_active_joints()]
    columns = [names.index(name) for name in ARM_JOINTS + BODY_JOINTS]
    qpos = robot.get_qpos()[env_idx].clone()
    limits = robot.get_qlimits()[env_idx][:, columns]
    scale = torch.as_tensor(scales, dtype=qpos.dtype, device=qpos.device)
    low = torch.maximum(qpos[:, columns] - scale, limits[..., 0])
    high = torch.minimum(qpos[:, columns] + scale, limits[..., 1])
    if bool(torch.any(low > high)):
        raise ValueError("Initial joint neighbourhood is outside robot limits")
    draws = np.stack(
        [
            env._batched_episode_rng[int(index)].uniform(0, 1, size=len(columns))
            for index in env_idx.tolist()
        ]
    )
    unit = torch.as_tensor(draws, dtype=qpos.dtype, device=qpos.device)
    qpos[:, columns] = low + unit * (high - low)
    robot.set_qpos(qpos)
    robot.set_qvel(torch.zeros_like(qpos))
