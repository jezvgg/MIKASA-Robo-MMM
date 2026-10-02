"""The approach to a stand pose: the base drives with the arm tucked, then arm and torso unfold along the checked joint line."""
from __future__ import annotations

import numpy as np

from .arm_path import JointLimiter
from .config import (ARM_MIN_TAIL_STEPS, ARM_TAIL_SLACK, MAX_APPROACH_STEPS, READY_ARM_POSTURE, READY_RAMP_STEPS, STALL_MIN_MOVE, STALL_WINDOW,
                     UNFOLD_PEAK_STEP)
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


def unfold_steps(arm0: np.ndarray, arm1: np.ndarray) -> int:
    """Env steps of the unfold: long enough that the cosine ramp's peak joint speed stays at UNFOLD_PEAK_STEP, so the
    rate limiter does not bend the joint line that `arm_ik.unfold_is_free` has checked."""
    return int(np.ceil(0.5 * np.pi * float(np.max(np.abs(arm1 - arm0))) / UNFOLD_PEAK_STEP)) + 1


def drive_concurrent(planner, agent, goal: np.ndarray, arm1: np.ndarray, torso1: float) -> bool:
    """Servo the base to `goal` with the arm tucked, then unfold arm and torso to (`arm1`, `torso1`) along a straight joint line.

    The line is the one `arm_ik.unfold_is_free` checked for the stand pose, so the hand never sweeps through the counter
    edge while the base is still arriving. Returns True when the base arrived within tolerance and the arm reached its target.
    """
    arm_init = agent.controller.controllers["arm"].qpos[0].cpu().numpy().astype(np.float64)
    arm0 = READY_ARM_POSTURE
    body0 = agent.controller.controllers["body"].qpos[0].cpu().numpy().astype(np.float64)
    arrived, tail = False, 0
    tucked = float(np.max(np.abs(arm_init - arm0))) < 0.02   # a retry starts tucked: no 2 s wait for a ramp that is not needed
    steps = unfold_steps(arm0, arm1)
    state = ServoState()
    window = []
    last = getattr(agent.controller.controllers["arm"], "_target_qpos", None)   # the last command, not the lagging measurement
    limiter = JointLimiter(arm_init if last is None else last[0].cpu().numpy().astype(np.float64))
    for i in range(MAX_APPROACH_STEPS):
        pose = base_xyyaw(agent)
        at_via = state.via_done
        v, w, done = servo_command(pose, goal, state)
        if state.via_done and not at_via:
            window.clear()   # the base rested at the via point turning: that standstill is not a block
        if tucked:
            window.append(pose[:2].copy())
        if not done and not arrived and abs(v) > 0.15 and len(window) > STALL_WINDOW:
            if float(np.hypot(*(window[-1] - window[-1 - STALL_WINDOW]))) < STALL_MIN_MOVE:
                planner.v6_reason = f'blocked at {np.round(pose, 2).tolist()} by {_contact_names(agent)}'
                return False   # commanded forward but the base does not move: blocked
        arrived = arrived or done
        tucked = tucked or (i >= READY_RAMP_STEPS and float(np.max(np.abs(limiter.q - arm0))) < 0.02)
        s = _smooth(tail / steps) if arrived else 0.0   # the unfold starts once the base has arrived
        arm = limiter.update(arm0 + (arm1 - arm0) * s)
        body = np.array([0.0, 0.0, body0[2] + (torso1 - body0[2]) * s])
        v, w = (0.0, 0.0) if (arrived or not tucked) else (v, w)   # tuck the arm before the base turns it into the cabinets
        _log_servo(planner, i, pose, goal, v, w, arrived)
        planner._step(planner._compose(arm, body, base_action(v, w)))
        if planner.truncated:
            planner.v6_reason = 'truncated'
            return False
        if arrived:
            tail += 1
            err = agent.controller.controllers["arm"].qpos[0].cpu().numpy() - arm1
            if tail >= max(ARM_MIN_TAIL_STEPS, steps) and float(np.max(np.abs(err))) < 0.03:
                return True
            if tail >= steps + ARM_TAIL_SLACK:
                break   # the arm is held (a finger on the counter): give the stand up instead of idling
    planner.v6_reason = f'timeout, contacts {_contact_names(agent)}, base {np.round(base_xyyaw(agent), 2).tolist()} arrived={arrived} arm_err={float(np.max(np.abs(agent.controller.controllers["arm"].qpos[0].cpu().numpy() - arm1))):.3f}'
    return False
