"""Stand-pose search (IK at a hypothetical base pose) and the concurrent base + torso + arm approach."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mplib
import numpy as np
import sapien

from .config import (ARM_END_PROGRESS, ARM_MIN_TAIL_STEPS, ARM_START_PROGRESS, BASE_RADIUS, COUNTER_CLEARANCE,
                     FLOOR_MARGIN, IK_SEEDS, MAX_APPROACH_STEPS, PLACE_HOVER, PREGRASP_GAP, READY_ARM_POSTURE, READY_RAMP_STEPS, STAND_HEADING, STAND_HEADING_TILT, ARM_YAW_BAND, ARM_YAW_FULL, STAND_TURN_WEIGHT, STAND_COUNTER_END_MARGIN, STAND_DX_WEIGHT, STAND_EXCLUDE_RADIUS, STALL_WINDOW, STALL_MIN_MOVE,
                     STAND_X_OFFSETS, STAND_Y_STEPS)
from .gaze import base_xyyaw
from .servo import base_action, servo_command, wrap


@dataclass(frozen=True)
class StandPlan:
    """A base pose from which the cup is grasped and the tray reached, with the arm configurations."""
    pose: np.ndarray            # (x, y, yaw) of the base
    grasp_pose: sapien.Pose
    pregrasp_pose: sapien.Pose
    place_pose: sapien.Pose     # TCP pose that carries the cup over the tray
    arm_pre: np.ndarray         # (7,) arm joints at the pregrasp pose
    torso_pre: float


def _joint_index(agent) -> dict:
    return {j.get_name(): i for i, j in enumerate(agent.robot.get_active_joints())}


def ik_at_base(planner, agent, base_pose: np.ndarray, target: sapien.Pose, seed_arm: np.ndarray):
    """IK of `target` (world TCP pose) with the base at `base_pose`; returns (arm7, torso) or None.

    The root joints of the folded planning qpos are set to the hypothetical base pose and held by the
    mask, so mplib solves the arm and torso for that base. Of the solutions the one closest to
    `seed_arm` is returned.
    """
    p = planner.planner
    cur = planner.robot.get_qpos().cpu().numpy()[0].astype(np.float64)
    cur_f = np.asarray(p.fold_qpos(cur), dtype=np.float64).copy()
    cur_f[:3] = base_pose
    mask = [True, True, True, False] + [False] * 11
    goal = p._transform_goal_to_wrt_base(mplib.Pose(p=target.p, q=target.q))
    status, sols = p.IK(goal, cur_f, mask, n_init_qpos=IK_SEEDS)
    if status != "Success" or sols is None or len(np.atleast_2d(sols)) == 0:
        return None
    idx = _joint_index(agent)
    arm_names = agent.controller.controllers["arm"].config.joint_names
    arm_cols = [idx[n] for n in arm_names]
    rows = np.atleast_2d(sols)
    limits = np.array([agent.robot.get_active_joints()[c].limits[0] for c in arm_cols], dtype=np.float64)
    best, best_cost = None, np.inf
    for r in rows:
        arm = np.asarray(r[arm_cols], dtype=np.float64)
        arm = seed_arm + (arm - seed_arm + np.pi) % (2.0 * np.pi) - np.pi   # same pose, branch nearest the seed
        in_limits = np.all((arm >= limits[:, 0] - 1e-3) | ~np.isfinite(limits[:, 0])) and np.all((arm <= limits[:, 1] + 1e-3) | ~np.isfinite(limits[:, 1]))
        cost = float(np.linalg.norm(arm - seed_arm))
        if in_limits and cost < best_cost:
            best, best_cost, torso = arm, cost, float(r[idx["torso_lift_joint"]])
    return None if best is None else (best, torso)


def _grasp_poses(agent, stand_xy: np.ndarray, cup_pos: np.ndarray, gap: float):
    approach = np.asarray(cup_pos, dtype=float) - np.array([stand_xy[0], stand_xy[1], 0.0])
    approach[2] = 0.0
    approach /= np.linalg.norm(approach)
    closing = np.cross(approach, np.array([0.0, 0.0, 1.0]))
    closing /= np.linalg.norm(closing)
    grasp = agent.build_grasp_pose(approach, closing, cup_pos)
    return grasp, grasp * sapien.Pose([0.0, 0.0, -gap])


def _place_pose(grasp: sapien.Pose, cup_pose: sapien.Pose, tray_pos, tray_half, cup_half) -> sapien.Pose:
    z = float(tray_pos[2] + tray_half[2] + cup_half[2] + PLACE_HOVER)
    target_cup = sapien.Pose(p=[float(tray_pos[0]), float(tray_pos[1]), z], q=cup_pose.q)
    return target_cup * (grasp.inv() * cup_pose).inv()


def find_stand(planner, unwenv, agent, noise_xy: np.ndarray, pregrasp_gap: float = PREGRASP_GAP,
               exclude: tuple = ()) -> Optional[StandPlan]:
    """The cheapest-turn stand pose from which pregrasp and place have IK; stands within STAND_EXCLUDE_RADIUS of `exclude` are skipped."""
    cup_pose = unwenv.cup.pose.sp
    cup_pos = unwenv.cup.pose.p[0].cpu().numpy()
    tray_pos = unwenv.tray.pose.p[0].cpu().numpy()
    front = float(unwenv.counter_pos[1] - unwenv.counter_size[1] / 2) - BASE_RADIUS - COUNTER_CLEARANCE
    x0, x1, y0, y1 = (float(v) for v in unwenv.floor_bounds)
    seed_arm = READY_ARM_POSTURE
    yaw0 = float(base_xyyaw(agent)[2])
    heading = STAND_HEADING + float(np.clip(wrap(yaw0 - STAND_HEADING), -STAND_HEADING_TILT, STAND_HEADING_TILT))
    mid_x = 0.5 * (float(cup_pos[0]) + float(tray_pos[0]))
    pose0 = base_xyyaw(agent)
    cands = []
    for iy, dy in enumerate(STAND_Y_STEPS):
        for ix, dx in enumerate(STAND_X_OFFSETS):
            xy = np.array([mid_x + dx, front - dy]) + noise_xy
            if not (x0 + FLOOR_MARGIN <= xy[0] <= x1 - FLOOR_MARGIN and y0 + FLOOR_MARGIN <= xy[1] <= y1 - FLOOR_MARGIN):
                continue
            if abs(xy[0] - float(unwenv.counter_pos[0])) > float(unwenv.counter_size[0]) / 2 - STAND_COUNTER_END_MARGIN:
                continue   # beside the counter end the arm sweeps into the neighbouring cabinet
            rel = xy - pose0[:2]
            alpha = abs(wrap(float(np.arctan2(rel[1], rel[0])) - float(pose0[2]))) if np.hypot(*rel) > 0.1 else 0.0
            cost = STAND_TURN_WEIGHT * (alpha + 0.5 * abs(wrap(heading - float(pose0[2])))) + STAND_DX_WEIGHT * abs(xy[0] - mid_x) + 0.05 * iy
            if all(np.hypot(*(xy - np.asarray(e))) > STAND_EXCLUDE_RADIUS for e in exclude):
                cands.append((cost, xy))
    cands.sort(key=lambda c: c[0])
    for _, xy in cands:
        pose = np.array([xy[0], xy[1], heading])
        grasp, pre = _grasp_poses(agent, xy, cup_pos, pregrasp_gap)
        place = _place_pose(grasp, cup_pose, tray_pos, unwenv.tray_half, unwenv.cup_half)
        sol_pre = ik_at_base(planner, agent, pose, pre, seed_arm)
        if sol_pre is None:
            continue
        if ik_at_base(planner, agent, pose, place, sol_pre[0]) is None:
            continue
        return StandPlan(pose, grasp, pre, place, sol_pre[0], sol_pre[1])
    return None


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


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 0.5 - 0.5 * np.cos(np.pi * x)


def drive_concurrent(planner, agent, goal: np.ndarray, arm1: np.ndarray, torso1: float) -> bool:
    """Servo the base to `goal` while arm and torso move to (`arm1`, `torso1`) on the path progress.

    Returns True when the base arrived within tolerance and the arm reached its target.
    """
    arm_init = agent.controller.controllers["arm"].qpos[0].cpu().numpy().astype(np.float64)
    arm0 = READY_ARM_POSTURE
    body0 = agent.controller.controllers["body"].qpos[0].cpu().numpy().astype(np.float64)
    pose0 = base_xyyaw(agent)
    rho0 = max(float(np.hypot(*(goal[:2] - pose0[:2]))), 0.3)
    yaw0 = max(abs(wrap(goal[2] - pose0[2])), 0.5)
    progress, arrived, tail = 0.0, False, 0
    window = []
    for i in range(MAX_APPROACH_STEPS):
        pose = base_xyyaw(agent)
        v, w, done = servo_command(pose, goal)
        if i >= READY_RAMP_STEPS:
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
        arm = arm_init + (arm0 - arm_init) * _smooth((i + 1) / READY_RAMP_STEPS) if i < READY_RAMP_STEPS else arm0 + (arm1 - arm0) * s
        body = np.array([0.0, 0.0, body0[2] + (torso1 - body0[2]) * s])
        v, w = (0.0, 0.0) if (arrived or i < READY_RAMP_STEPS) else (v, w)   # tuck the arm before the base turns it into the cabinets
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
    planner.v6_reason = f'timeout, base {np.round(base_xyyaw(agent), 2).tolist()} arrived={arrived} arm_err={float(np.max(np.abs(agent.controller.controllers["arm"].qpos[0].cpu().numpy() - arm1))):.3f}'
    return False
