"""Arm-configuration choice of the v6.1c planner: IK candidates ranked by joint travel, with the grasp IK the real motion will need.

For one hypothetical base pose `chain_for_stand` returns the pregrasp / grasp / place arm configurations of the cheapest
branch: the roll joints (upperarm, forearm, wrist) count `ARM_ROLL_WEIGHT` times a plain joint; both closings of the
gripper (a half turn about the approach axis, the same grasp) are tried; the
grasp must have IK with the torso held, as the real solver will need it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import mplib
import numpy as np
import sapien

from .config import (ARM_NON_ROLL_WEIGHT, ARM_ROLL_ABS_WEIGHT, ARM_ROLL_WEIGHT, GRASP_PROBE_GAP, IK_SEEDS, LOWER_MARGIN, PLACE_HOVER, PREGRASP_GAP,
                     PRE_CANDIDATES, READY_ARM_POSTURE, TRAY_DROP_GAP, UNFOLD_MARGIN, UNFOLD_SAMPLES)

logger = logging.getLogger(__name__)
ROLL_IDX = (2, 4, 6)          # upperarm_roll, forearm_roll, wrist_roll within the 7 arm joints
_WEIGHTS = np.array([ARM_ROLL_WEIGHT if i in ROLL_IDX else ARM_NON_ROLL_WEIGHT for i in range(7)])


@dataclass(frozen=True)
class ArmChain:
    """Arm configurations (7 joints + torso) at pregrasp, grasp and place for one stand pose and one closing sign."""
    flip: bool
    pre_pose: sapien.Pose
    grasp_pose: sapien.Pose
    place_pose: sapien.Pose
    arm_pre: np.ndarray
    torso_pre: float
    arm_grasp: np.ndarray
    arm_place: np.ndarray
    cost: float
    roll_travel: float          # rad of roll-joint travel ready -> pre -> grasp -> place
    pan: float = 0.0            # |shoulder pan| at the grasp: a large one means the stand is far off the approach axis


def joint_cost(arm: np.ndarray, seed: np.ndarray) -> float:
    """Weighted joint travel between two arm configurations, plus a small pull of the rolls towards zero."""
    return float(np.sum(_WEIGHTS * np.abs(arm - seed)) + ARM_ROLL_ABS_WEIGHT * np.sum(np.abs(arm[list(ROLL_IDX)])))


def _joint_index(agent) -> dict:
    return {j.get_name(): i for i, j in enumerate(agent.robot.get_active_joints())}


def _arm_cols(agent) -> list:
    idx = _joint_index(agent)
    return [idx[n] for n in agent.controller.controllers["arm"].config.joint_names]


def _folded_qpos(planner, agent, pose: np.ndarray, arm: np.ndarray, torso: float) -> np.ndarray:
    """Planning-model (folded) qpos: the current robot with base `pose` (world x, y, yaw), the given arm and torso."""
    p = planner.planner
    cur = planner.robot.get_qpos().cpu().numpy()[0].astype(np.float64)
    q = np.asarray(p.fold_qpos(cur), dtype=np.float64).copy()
    q[:3] = pose
    q[_arm_cols(agent)] = arm
    q[_joint_index(agent)["torso_lift_joint"]] = torso
    return q


def ik_candidates(planner, agent, pose: np.ndarray, target: sapien.Pose, seed_arm: np.ndarray,
                  torso: Optional[float] = None) -> list:
    """In-limit IK solutions (arm7, torso) of `target` at base `pose`, cheapest `joint_cost` from `seed_arm` first.

    `torso` None leaves the torso free; a value holds it there. Every solution is shifted by whole turns onto the
    branch of each joint nearest the seed.
    """
    p = planner.planner
    idx = _joint_index(agent)
    cur = planner.robot.get_qpos().cpu().numpy()[0].astype(np.float64)
    cur_f = np.asarray(p.fold_qpos(cur), dtype=np.float64).copy()
    cur_f[:3] = pose
    mask = [True, True, True, torso is not None] + [False] * 11
    if torso is not None:
        cur_f[idx["torso_lift_joint"]] = torso
    goal = p._transform_goal_to_wrt_base(mplib.Pose(p=target.p, q=target.q))
    status, sols = p.IK(goal, cur_f, mask, n_init_qpos=IK_SEEDS)
    if status != "Success" or sols is None or len(np.atleast_2d(sols)) == 0:
        return []
    cols = _arm_cols(agent)
    limits = np.array([agent.robot.get_active_joints()[c].limits[0] for c in cols], dtype=np.float64)
    out = []
    for r in np.atleast_2d(sols):
        arm = np.asarray(r[cols], dtype=np.float64)
        arm = seed_arm + (arm - seed_arm + np.pi) % (2.0 * np.pi) - np.pi
        lo_ok = ~np.isfinite(limits[:, 0]) | (arm >= limits[:, 0] - 1e-3)
        hi_ok = ~np.isfinite(limits[:, 1]) | (arm <= limits[:, 1] + 1e-3)
        if np.all(lo_ok & hi_ok):
            out.append((arm, float(r[idx["torso_lift_joint"]])))
    out.sort(key=lambda s: joint_cost(s[0], seed_arm))
    return out


def _collides(planner, agent, pose: np.ndarray, arm: np.ndarray, torso: float) -> bool:
    """True when the robot at base `pose` with this arm and torso touches the planning world."""
    return len(planner.planner.check_for_env_collision(_folded_qpos(planner, agent, pose, arm, torso))) > 0


def unfold_is_free(planner, agent, pose: np.ndarray, arm0: np.ndarray, arm1: np.ndarray, torso1: float) -> bool:
    """The straight joint line (arm and torso) the approach will run after arrival is collision-free at base `pose`
    and with the base UNFOLD_MARGIN further ahead (the base never stops exactly on the stand)."""
    torso0 = float(planner.robot.get_qpos().cpu().numpy()[0][_joint_index(agent)["torso_lift_joint"]])
    ahead = UNFOLD_MARGIN * np.array([np.cos(pose[2]), np.sin(pose[2]), 0.0])
    for shift in (np.zeros(3), ahead):
        for t in np.linspace(0.0, 1.0, UNFOLD_SAMPLES):
            if _collides(planner, agent, pose + shift, arm0 + (arm1 - arm0) * t, torso0 + (torso1 - torso0) * t):
                return False
    return True


def lowered_is_free(planner, agent, pose: np.ndarray, arm: np.ndarray, torso: float) -> bool:
    """The arm at the place pose, with the torso down by the hover height (the lowering of the cup), touches nothing."""
    drop = PLACE_HOVER - TRAY_DROP_GAP + LOWER_MARGIN
    return not _collides(planner, agent, pose, arm, max(0.0, torso - drop))


def grasp_poses(agent, stand_xy: np.ndarray, cup_pos: np.ndarray, flip: bool, gap: float = PREGRASP_GAP):
    """(grasp, pregrasp) TCP poses; `flip` turns the gripper half a turn about the approach axis (same grasp)."""
    approach = np.asarray(cup_pos, dtype=float) - np.array([stand_xy[0], stand_xy[1], 0.0])
    approach[2] = 0.0
    approach /= np.linalg.norm(approach)
    closing = np.cross(approach, np.array([0.0, 0.0, 1.0]))
    closing /= np.linalg.norm(closing)
    grasp = agent.build_grasp_pose(approach, -closing if flip else closing, cup_pos)
    return grasp, grasp * sapien.Pose([0.0, 0.0, -gap])


def chain_for_stand(planner, agent, pose: np.ndarray, cup_pos: np.ndarray, place_for, flip: bool,
                    checked: bool = True) -> Optional[ArmChain]:
    """Cheapest valid arm chain for base `pose` and closing sign `flip`, or None.

    `place_for(grasp)` returns the place TCP pose belonging to a grasp pose. With `checked` the unfold line from the
    ready posture to the pregrasp and the arm at the lowered place pose must be collision-free too.
    """
    grasp, pre = grasp_poses(agent, pose[:2], cup_pos, flip)
    place = place_for(grasp)
    ready = READY_ARM_POSTURE
    best: Optional[ArmChain] = None
    stage = {"grasp_ik": 0, "pre_ik": 0, "place_ik": 0}
    # grasp first, torso free: its torso is the one the locked-torso grasp motion then needs, so the pregrasp holds it
    # (mplib's IK refuses poses whose fingers touch the cup, so the reach is probed GRASP_PROBE_GAP short of the grasp)
    grasps = ik_candidates(planner, agent, pose, grasp * sapien.Pose([0.0, 0.0, -GRASP_PROBE_GAP]), ready)
    stage["grasp_ik"] = len(grasps)
    for arm_g, torso_g in grasps[:PRE_CANDIDATES]:
        pres = ik_candidates(planner, agent, pose, pre, arm_g, torso=torso_g)
        if not pres:
            continue
        stage["pre_ik"] += 1
        arm_pre = pres[0][0]
        if checked and not unfold_is_free(planner, agent, pose, ready, arm_pre, torso_g):
            continue
        places = ik_candidates(planner, agent, pose, place, arm_g)
        places = [p for p in places[:3] if not checked or lowered_is_free(planner, agent, pose, p[0], p[1])]
        if not places:
            continue
        stage["place_ik"] += 1
        arm_pl = places[0][0]
        path = ((ready, arm_pre), (arm_pre, arm_g), (arm_g, arm_pl))
        cost = sum(joint_cost(b, a) for a, b in path)
        roll = sum(float(np.sum(np.abs(b[list(ROLL_IDX)] - a[list(ROLL_IDX)]))) for a, b in path)
        if best is None or cost < best.cost:
            best = ArmChain(flip, pre, grasp, place, arm_pre, torso_g, arm_g, arm_pl, cost, roll, abs(float(arm_g[0])))
    logger.debug("chain at %s flip=%s: %s -> %s", np.round(pose, 2).tolist(), flip, stage, "ok" if best else "none")
    return best
