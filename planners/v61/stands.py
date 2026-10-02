"""Stand-pose choice: of the base-cheapest feasible stands, the one with the cheapest arm chain as well."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sapien

from .arm_ik import ArmChain, chain_for_stand
from .config import (MARGIN_WEIGHT, ROLL_HARD, ROLL_SOFT, ROLL_WEIGHT, ARM_COST_WEIGHT, BASE_RADIUS, COUNTER_CLEARANCE, FLOOR_MARGIN, PLACE_HOVER, PAN_FREE, PAN_WEIGHT, PLACE_REACH_SOFT, PLACE_REACH_WEIGHT,
                     STAND_COUNTER_END_MARGIN, STAND_DX_WEIGHT, STAND_EXCLUDE_RADIUS, STAND_HEADING, STAND_HEADING_STEPS,
                     STAND_HEADING_TILT, STAND_TOPK, STRAIGHT_MAX_DX, BEARING_TILT, TORSO_SOFT, TORSO_WEIGHT, STAND_X_OFFSETS, STAND_Y_STEPS)
from .clearance import clear_ranked
from .gaze import base_xyyaw
from .predict import stand_cost
from .servo import wrap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StandPlan:
    """A base pose from which the cup is grasped and the tray reached, with the arm chain."""
    pose: np.ndarray            # (x, y, yaw) of the base, then the via point (x, y) of the drive when it needs one
    grasp_pose: sapien.Pose
    pregrasp_pose: sapien.Pose
    place_pose: sapien.Pose     # TCP pose that carries the cup over the tray
    arm_pre: np.ndarray         # (7,) arm joints at the pregrasp pose
    torso_pre: float
    total_cost: float = 0.0
    roll_travel: float = 0.0


def _place_pose(grasp: sapien.Pose, cup_pose: sapien.Pose, tray_pos, tray_half, cup_half) -> sapien.Pose:
    z = float(tray_pos[2] + tray_half[2] + cup_half[2] + PLACE_HOVER)
    target_cup = sapien.Pose(p=[float(tray_pos[0]), float(tray_pos[1]), z], q=cup_pose.q)
    return target_cup * (grasp.inv() * cup_pose).inv()


def _candidates(unwenv, agent, noise_xy: np.ndarray, exclude: tuple) -> list:
    """(base cost, xy, heading) of every stand on the grid, cheapest first."""
    cup_pos = unwenv.cup.pose.p[0].cpu().numpy()
    tray_pos = unwenv.tray.pose.p[0].cpu().numpy()
    front = float(unwenv.counter_pos[1] - unwenv.counter_size[1] / 2) - BASE_RADIUS - COUNTER_CLEARANCE
    x0, x1, y0, y1 = (float(v) for v in unwenv.floor_bounds)
    pose0 = base_xyyaw(agent)
    heading = STAND_HEADING + float(np.clip(wrap(float(pose0[2]) - STAND_HEADING), -STAND_HEADING_TILT, STAND_HEADING_TILT))
    mid_x = 0.5 * (float(cup_pos[0]) + float(tray_pos[0]))
    headings = sorted({STAND_HEADING + f * STAND_HEADING_TILT for f in STAND_HEADING_STEPS} | {heading})
    cands = []
    for iy, dy in enumerate(STAND_Y_STEPS):
        spots = [np.array([mid_x + dx, front - dy]) + noise_xy for dx in STAND_X_OFFSETS]
        sy = np.sin(heading)
        if sy > 0.3:   # the stand straight ahead of the base along the final heading: a drive without a turn back
            ty = front - dy + float(noise_xy[1]) - float(pose0[1])
            tx = float(pose0[0]) + ty / np.tan(heading)
            if ty > 0.05 and abs(tx - mid_x) <= STRAIGHT_MAX_DX:
                spots.append(np.array([tx, front - dy + float(noise_xy[1])]))
        for xy in spots:
            if not (x0 + FLOOR_MARGIN <= xy[0] <= x1 - FLOOR_MARGIN and y0 + FLOOR_MARGIN <= xy[1] <= y1 - FLOOR_MARGIN):
                continue
            if abs(xy[0] - float(unwenv.counter_pos[0])) > float(unwenv.counter_size[0]) / 2 - STAND_COUNTER_END_MARGIN:
                continue
            if not all(np.hypot(*(xy - np.asarray(e))) > STAND_EXCLUDE_RADIUS for e in exclude):
                continue
            bearing = float(np.arctan2(xy[1] - pose0[1], xy[0] - pose0[0]))   # facing the stand from the start: no turn back at the end
            bearing = STAND_HEADING + float(np.clip(wrap(bearing - STAND_HEADING), -BEARING_TILT, BEARING_TILT))
            for h in sorted(set(headings) | {bearing}):
                reach = max(0.0, float(np.hypot(*(xy - tray_pos[:2]))) - PLACE_REACH_SOFT)   # far stands fail to carry the cup over the tray
                cost = stand_cost(pose0, np.array([xy[0], xy[1], h])) + STAND_DX_WEIGHT * abs(xy[0] - mid_x) + 0.05 * iy + PLACE_REACH_WEIGHT * reach
                cands.append((cost, xy, h))
    cands.sort(key=lambda c: c[0])
    return cands


def find_stand(planner, unwenv, agent, noise_xy: np.ndarray, exclude: tuple = ()) -> Optional[StandPlan]:
    """The best stand with checked arm motions; when none exists, the best one without the collision checks."""
    return (_search(planner, unwenv, agent, noise_xy, exclude, True, ROLL_HARD, True) or _search(planner, unwenv, agent, noise_xy, exclude, True, None)
            or _search(planner, unwenv, agent, noise_xy, exclude, False, None))


def _search(planner, unwenv, agent, noise_xy: np.ndarray, exclude: tuple, checked: bool, roll_cap: Optional[float] = None, margin: bool = False) -> Optional[StandPlan]:
    """The stand minimising base cost + ARM_COST_WEIGHT x arm-chain cost among the STAND_TOPK cheapest feasible ones.

    A stand is feasible when an arm chain (pregrasp with a collision-free unfold from the ready posture, grasp with the torso
    held, place) exists for it; both closings of the gripper are tried and the cheaper chain wins. Stands within
    STAND_EXCLUDE_RADIUS of `exclude` are skipped.
    """
    planner.planner.update_from_simulation()   # the planning world must hold the scene as it is now before arm lines are checked
    cup_pose = unwenv.cup.pose.sp
    cup_pos = unwenv.cup.pose.p[0].cpu().numpy()
    tray_pos = unwenv.tray.pose.p[0].cpu().numpy()

    def place_for(grasp: sapien.Pose) -> sapien.Pose:
        return _place_pose(grasp, cup_pose, tray_pos, unwenv.tray_half, unwenv.cup_half)

    best: Optional[StandPlan] = None
    seen = 0
    for base_cost, xy, h, via in clear_ranked(planner, agent, _candidates(unwenv, agent, noise_xy, exclude)):   # drive free of the furniture first
        pose = np.array([xy[0], xy[1], h])
        chains = [c for c in (chain_for_stand(planner, agent, pose, cup_pos, place_for, flip, checked, roll_cap, margin) for flip in (False, True)) if c]
        if not chains:
            continue
        chain: ArmChain = min(chains, key=lambda c: c.cost)
        total = base_cost + ARM_COST_WEIGHT * chain.cost + PAN_WEIGHT * max(0.0, chain.pan - PAN_FREE) + TORSO_WEIGHT * max(0.0, chain.torso_pre - TORSO_SOFT) + ROLL_WEIGHT * max(0.0, chain.max_roll - ROLL_SOFT) + MARGIN_WEIGHT * chain.margin_miss
        logger.info("stand %s base %.2f arm %.2f roll %.2f maxroll %.2f pan %.2f flip=%s", np.round(pose, 2).tolist(), base_cost, chain.cost, chain.roll_travel, chain.max_roll, chain.pan, chain.flip)
        if best is None or total < best.total_cost:
            best = StandPlan(np.concatenate([pose, via]), chain.grasp_pose, chain.pre_pose, chain.place_pose, chain.arm_pre, chain.torso_pre, total, chain.roll_travel)
        seen += 1
        if seen >= STAND_TOPK:
            break
    return best
