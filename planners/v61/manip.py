"""Fixed-base manipulation of the v6 planner: grasp, carry over the tray, lower and release."""
from __future__ import annotations


import numpy as np
import sapien

from .config import (RETREAT_STEPS, RETREAT_UP, CARRY_LIFT, CUP_REST_STEPS, CUP_REST_V, CUP_REST_W, CUP_SETTLE_MAX, LOWER_RAMP_STEPS, LOWER_TOL, LOWER_TRIM_GAIN, LOWER_TRIM_STEPS, PLACE_EXTRA_HEIGHTS, GRIP_SETTLE_STEPS, GRIPPER_CLOSED, GRIPPER_OPEN, TRAY_DROP_GAP)


def _targets(agent):
    arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy().astype(np.float64).copy()
    body = agent.controller.controllers["body"].qpos[0].cpu().numpy().astype(np.float64).copy()
    return arm, body


def align_gripper_switch(planner, unwenv) -> None:
    """Make the next gripper switch land on an even env step (the dataset keeps every second step)."""
    if int(unwenv.elapsed_steps.reshape(-1)[0]) % 2 == 1:
        planner.idle_steps(t=1)


def _hold(planner, agent, steps: int, stop, targets=None) -> None:
    """Hold the arm and body targets (default: the controller's current ones) until `stop()` has held for GRIP_SETTLE_STEPS."""
    arm, body = targets if targets is not None else _targets(agent)
    settle = 0
    for _ in range(steps):
        planner._compose(arm, body, np.zeros(2))
        planner._step(planner._from_abs(planner._last_abs))
        if planner.truncated:
            return
        settle = settle + 1 if stop() else 0
        if settle >= GRIP_SETTLE_STEPS:
            return


def move_tcp(planner, pose: sapien.Pose, free_torso: bool, line_first: bool = True) -> bool:
    """TCP to `pose` with the base held: a joint line and the solver's screw/RRT, in the order `line_first` picks."""
    for by_line in ((True, False) if line_first else (False, True)):
        res = planner.static_manipulation(pose, n_init_qpos=100, disable_lift_joint=not free_torso, by_line=by_line)
        if res != -1:
            return True
        if planner.truncated:
            return False
    return False


def grasp_cup(planner, unwenv, agent, grasp_pose: sapien.Pose, pregrasp_pose: sapien.Pose) -> bool:
    """Move to the grasp pose, close the gripper, return True when the cup is held."""
    planner.gripper_state = GRIPPER_OPEN
    if not move_tcp(planner, pregrasp_pose, free_torso=True) or not move_tcp(planner, grasp_pose, free_torso=False):
        unwenv.reason = 'grasp: move_tcp failed'
        return False
    miss = float(np.linalg.norm(agent.tcp.pose.p[0].cpu().numpy() - grasp_pose.p))
    if miss > 0.06:
        unwenv.reason = f'grasp: tcp {miss:.3f} m from the grasp pose'
        return False
    align_gripper_switch(planner, unwenv)
    planner.gripper_state = GRIPPER_CLOSED
    grasping = lambda: bool(unwenv.agent.is_grasping(unwenv.cup).item())
    _hold(planner, agent, 20, grasping)
    planner.planner.update_from_simulation()
    if not grasping():
        unwenv.reason = 'grasp: closed gripper does not hold the cup'
    return grasping()


def carry_over_tray(planner, env, unwenv, place_pose: sapien.Pose) -> bool:
    """Attach the cup to the planning hand and move it over the tray (torso free, so it rises with the arm)."""
    from planners.oracle.oracle_common import hold_object_in_planner

    hold_object_in_planner(env, planner, unwenv, unwenv.cup, held=True, who="takeitback_tray")
    planner.gripper_state = GRIPPER_CLOSED
    tcp = unwenv.agent.tcp.pose
    lift = sapien.Pose(p=tcp.p[0].cpu().numpy() + np.array([0.0, 0.0, CARRY_LIFT]), q=tcp.q[0].cpu().numpy())
    move_tcp(planner, lift, free_torso=True, line_first=False)   # clear the counter before swinging over; a refusal is no reason to give up
    for extra in PLACE_EXTRA_HEIGHTS:   # a higher hover when the planner refuses the exact place pose; the torso lowers the cup afterwards
        hover = sapien.Pose(p=place_pose.p + np.array([0.0, 0.0, extra]), q=place_pose.q)
        if move_tcp(planner, hover, free_torso=True, line_first=False):
            return bool(unwenv.agent.is_grasping(unwenv.cup).item())
        if planner.truncated:
            break
    return False


def settle_cup(planner, unwenv) -> None:
    """Idle until the cup has been slower than the checker's rest limits for CUP_REST_STEPS steps, at most CUP_SETTLE_MAX."""
    calm = 0
    for _ in range(CUP_SETTLE_MAX):
        v = float(unwenv.cup.linear_velocity.norm())
        w = float(unwenv.cup.angular_velocity.norm())
        calm = calm + 1 if (v <= CUP_REST_V and w <= CUP_REST_W) else 0
        if calm >= CUP_REST_STEPS:
            return
        planner.idle_steps(t=1)
        if planner.truncated:
            return


def lower_and_release(planner, env, unwenv, agent) -> bool:
    """Lower the cup onto the tray with the torso, open the gripper, lower the torso back; True if released."""
    from planners.oracle.oracle_common import hold_object_in_planner

    arm, body = _targets(agent)
    tray_top = float(unwenv.tray.pose.p[0].cpu().numpy()[2] + unwenv.tray_half[2])
    target_z = tray_top + float(unwenv.cup_half[2]) + TRAY_DROP_GAP
    cup_z = float(unwenv.cup.pose.p[0].cpu().numpy()[2])
    torso0 = float(body[2])
    torso1 = float(np.clip(torso0 - (cup_z - target_z), 0.0, 0.386))
    for i in range(LOWER_RAMP_STEPS):
        b = body.copy()
        b[2] = torso0 + (torso1 - torso0) * ((i + 1) / LOWER_RAMP_STEPS)
        planner._compose(arm, b, np.zeros(2))
        planner._step(planner._from_abs(planner._last_abs))
        if planner.truncated:
            return False
    body = body.copy()
    body[2] = torso1   # from here on the command is the lowered torso, never the stale pre-lowering one
    for _ in range(LOWER_TRIM_STEPS):   # the torso lags its command: trim on the measured cup height until it is down
        err = float(unwenv.cup.pose.p[0].cpu().numpy()[2]) - target_z
        if err < LOWER_TOL:
            break
        body = body.copy()
        body[2] = float(np.clip(body[2] - LOWER_TRIM_GAIN * err, 0.0, 0.386))
        planner._compose(arm, body, np.zeros(2))
        planner._step(planner._from_abs(planner._last_abs))
        if planner.truncated:
            return False
    align_gripper_switch(planner, unwenv)
    planner.gripper_state = GRIPPER_OPEN
    _hold(planner, agent, 30, lambda: not bool(unwenv.agent.is_grasping(unwenv.cup).item()), targets=(arm, body))   # keep the commanded torso: re-reading the lagging measured one lifts the hand off the cup
    planner.planner.update_from_simulation()
    hold_object_in_planner(env, planner, unwenv, unwenv.cup, held=False, who="takeitback_tray")
    released = not bool(unwenv.agent.is_grasping(unwenv.cup).item())
    if released:   # lift the open hand off the cup so it can come to rest untouched
        top = float(np.clip(body[2] + RETREAT_UP, 0.0, 0.386))
        for i in range(RETREAT_STEPS):
            b = body.copy()
            b[2] = body[2] + (top - body[2]) * ((i + 1) / RETREAT_STEPS)
            planner._compose(arm, b, np.zeros(2))
            planner._step(planner._from_abs(planner._last_abs))
            if planner.truncated:
                break
    return released
