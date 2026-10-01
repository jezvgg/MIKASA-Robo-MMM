"""The concurrent base + torso + arm approach to a stand pose (the stand itself is chosen in `stands`)."""
from __future__ import annotations

import numpy as np

from .arm_path import JointLimiter
from .config import (ARM_END_PROGRESS, ELBOW_COL, ELBOW_LAG, LIFT_COL, LIFT_LEAD, TORSO_LEAD, ARM_MIN_TAIL_STEPS, ARM_START_PROGRESS, ARM_TAIL_TIMEOUT, ARM_YAW_BAND, ARM_YAW_FULL,
                     MAX_APPROACH_STEPS, READY_ARM_POSTURE, READY_RAMP_STEPS, STALL_MIN_MOVE, STALL_WINDOW)
from .gaze import base_xyyaw
from .servo import ServoState, base_action, polar, servo_command, wrap


def _contact_names(agent) -> list:
    """Names of non-robot bodies touching the robot (diagnostics for a blocked base)."""
    mine = {l.entity.name for l in agent.robot._objs[0].links}
    out = set()
    for c in agent.robot.scene.px.get_contacts():
        n0, n1 = c.bodies[0].entity.name, c.bodies[1].entity.name
        if (n0 in mine) != (n1 in mine):
            mine_b, other = (c.bodies[0], c.bodies[1]) if n0 in mine else (c.bodies[1], c.bodies[0])
            out.add(f'{mine_b.entity.name}~{other.entity.name}@{np.round(other.entity.pose.p, 2).tolist()}')
    return sorted(out)[:4]


def _log_servo(planner, i: int, pose: np.ndarray, goal: np.ndarray, v: float, w: float, arrived: bool) -> None:
    """Append one row (attempt, step, x, y, yaw, rho, alpha, beta, v, w, arrived) to planner.v61_servo_log."""
    rho, alpha, beta = polar(pose, goal)
    row = [getattr(planner, "v61_attempt", 0), i, *pose, rho, alpha, beta, v, w, int(arrived)]
    planner.v61_servo_log.append([round(float(x), 4) for x in row])


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 0.5 - 0.5 * np.cos(np.pi * x)


def _arm_progress(s: float) -> np.ndarray:
    """Per-joint unfold progress: the shoulder lift leads and the elbow trails, so the hand rises before it reaches out
    over the counter edge (a straight joint line sweeps it through the edge at a height below the counter top)."""
    p = np.full(7, s)
    p[LIFT_COL] = float(_smooth(LIFT_LEAD * s))
    p[ELBOW_COL] = float(np.clip((s - ELBOW_LAG) / (1.0 - ELBOW_LAG), 0.0, 1.0))
    return p


def drive_concurrent(planner, agent, goal: np.ndarray, arm1: np.ndarray, torso1: float, keep_arm: bool = False) -> bool:
    """Servo the base to `goal` while arm and torso move to (`arm1`, `torso1`) on the path progress.

    `keep_arm` starts the arm move from the current command instead of tucking to the ready posture first (a retry
    from a nearby stand: the arm does not fold and unfold again).

    Returns True when the base arrived within tolerance and the arm reached its target.
    """
    arm_init = agent.controller.controllers["arm"].qpos[0].cpu().numpy().astype(np.float64)
    arm0 = READY_ARM_POSTURE
    body0 = agent.controller.controllers["body"].qpos[0].cpu().numpy().astype(np.float64)
    pose0 = base_xyyaw(agent)
    rho0 = max(float(np.hypot(*(goal[:2] - pose0[:2]))), 0.3)
    yaw0 = max(abs(wrap(goal[2] - pose0[2])), 0.5)
    progress, arrived, tail = 0.0, False, 0
    state = ServoState()
    window = []
    last = getattr(agent.controller.controllers["arm"], "_target_qpos", None)   # the last command, not the lagging measurement
    limiter = JointLimiter(arm_init if last is None else last[0].cpu().numpy().astype(np.float64))
    if keep_arm:
        arm0 = limiter.q.copy()
    tucked = keep_arm
    for i in range(MAX_APPROACH_STEPS):
        pose = base_xyyaw(agent)
        v, w, done = servo_command(pose, goal, state)
        if tucked:
            window.append(pose[:2].copy())
        if not done and not arrived and abs(v) > 0.15 and len(window) > STALL_WINDOW:
            if float(np.hypot(*(window[-1] - window[-1 - STALL_WINDOW]))) < STALL_MIN_MOVE:
                planner.v6_reason = f'blocked at {np.round(pose, 2).tolist()} by {_contact_names(agent)}'
                return False   # commanded forward but the base does not move: blocked
        arrived = arrived or done
        frac = max(float(np.hypot(*(goal[:2] - pose[:2]))) / rho0, abs(wrap(goal[2] - pose[2])) / yaw0)
        progress = max(progress, 1.0 - frac)
        s = 1.0 if arrived else _smooth((progress - ARM_START_PROGRESS) / (ARM_END_PROGRESS - ARM_START_PROGRESS))
        s_gate = float(np.clip((ARM_YAW_FULL + ARM_YAW_BAND - abs(wrap(goal[2] - pose[2]))) / ARM_YAW_BAND, 0.0, 1.0))
        s = s if arrived else s * s_gate   # the arm only unfolds once the base is nearly facing the stand heading
        tucked = tucked or (i >= READY_RAMP_STEPS and float(np.max(np.abs(limiter.q - arm0))) < 0.02)
        s_arm = _arm_progress(s) if not arrived else np.ones(7)
        arm = limiter.update(arm0 + (arm1 - arm0) * s_arm if tucked else arm0)
        s_torso = _smooth(TORSO_LEAD * s) if not arrived else 1.0   # the torso rises first: the hand clears the counter edge before it reaches over
        body = np.array([0.0, 0.0, body0[2] + (torso1 - body0[2]) * s_torso])
        v, w = (0.0, 0.0) if (arrived or not tucked) else (v, w)   # tuck the arm before the base turns it into the cabinets
        _log_servo(planner, i, pose, goal, v, w, arrived)
        planner._step(planner._compose(arm, body, base_action(v, w)))
        if planner.truncated:
            planner.v6_reason = 'truncated'
            return False
        if arrived:
            tail += 1
            err = agent.controller.controllers["arm"].qpos[0].cpu().numpy() - arm1
            reached = float(np.max(np.abs(err)))
            if tail >= ARM_MIN_TAIL_STEPS and reached < 0.03:
                return True
            if tail >= ARM_TAIL_TIMEOUT:
                break   # the arm is held (a finger on the counter): give the stand up instead of idling
    planner.v6_reason = f'timeout, contacts {_contact_names(agent)}, base {np.round(base_xyyaw(agent), 2).tolist()} arrived={arrived} arm_err={float(np.max(np.abs(agent.controller.controllers["arm"].qpos[0].cpu().numpy() - arm1))):.3f}'
    return False
