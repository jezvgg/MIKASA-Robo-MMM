"""Carry check: with the cup held, can the arm still reach the hover pose over the tray from a stand, robustly?

The stand search only proves a bare-hand place IK at the nominal base pose. The real carry starts from where the base
actually stopped and needs the torso to stay within its limit (and to have room to lower the cup), so the place IK is
re-solved here for base poses shifted by the arrival error and checked against the torso limits and the IK branch.
"""
from __future__ import annotations

import logging

import numpy as np
import sapien

from .arm_ik import ArmChain, _folded_qpos, ik_candidates
from .gaze import base_xyyaw
from .servo import base_action
from .config import (CARRY_ROLL_CLEARANCE, CARRY_ROLL_MAX, CARRY_ROLL_SPEED, CARRY_ROLL_STEP, ARRIVE_SHORT, ARRIVE_TOL, ARRIVE_YAW, CARRY_BRANCH_MAX, PLACE_HOVER, ROLL_HARD, TORSO_MAX, TRAY_DROP_GAP)

logger = logging.getLogger(__name__)


def _shifts(pose: np.ndarray, place_xy: np.ndarray) -> list:
    """Base poses to survive: nominal, short of the stand (away from the tray, the way arrival errors run), yawed, sideways."""
    away = pose[:2] - place_xy
    away = away / max(float(np.linalg.norm(away)), 1e-6)
    side = np.array([-away[1], away[0]])
    mk = lambda v, yaw=0.0: pose + np.array([v[0], v[1], yaw])
    return [pose, mk(ARRIVE_SHORT * away), mk(0.5 * ARRIVE_SHORT * away, ARRIVE_YAW), mk(0.5 * ARRIVE_SHORT * away, -ARRIVE_YAW), mk(ARRIVE_TOL * side), mk(-ARRIVE_TOL * side)]


def place_reachable(planner, agent, pose: np.ndarray, place_pose: sapien.Pose, arm_ref: np.ndarray) -> bool:
    """True when `place_pose` has an IK from base `pose` on the branch of `arm_ref`, torso in [drop, TORSO_MAX]."""
    drop = PLACE_HOVER - TRAY_DROP_GAP   # the torso must still be able to lower the cup by this much
    for arm, torso in ik_candidates(planner, agent, pose, place_pose, arm_ref, roll_cap=ROLL_HARD):
        if drop <= torso <= TORSO_MAX and float(np.max(np.abs(arm - arm_ref))) <= CARRY_BRANCH_MAX:
            return True
    return False


def carry_score(planner, agent, pose: np.ndarray, chain: ArmChain) -> tuple:
    """(nominal ok, share of the arrival-shifted base poses, incl. the nominal one, from which the carry is reachable)."""
    ok = [place_reachable(planner, agent, p, chain.place_pose, chain.arm_grasp) for p in _shifts(pose, chain.place_pose.p[:2])]
    logger.info("carry %s flip=%s nominal=%s shifted=%d/%d", np.round(pose, 2).tolist(), chain.flip, ok[0], sum(ok), len(ok))
    return ok[0], sum(ok) / len(ok)


def roll_in_reach(planner, agent, place_pose: sapien.Pose, arm: np.ndarray, body: np.ndarray) -> float:
    """With the cup held, roll the base forward along its heading until `place_pose` is reachable; returns the distance rolled.

    The base stops short of the stand by up to 0.1 m (the last centimetres are the arm's), which can put the hover pose
    out of reach (5806, 5807, 5708: from 38 IK solutions at the stand to none at the actual pose). Nothing is done when
    the pose is reachable already or when no roll within CARRY_ROLL_MAX reaches it without touching the planning world.
    """
    pose = base_xyyaw(agent)
    ahead = np.array([np.cos(pose[2]), np.sin(pose[2]), 0.0])
    if place_reachable(planner, agent, pose, place_pose, arm):
        return 0.0
    def contacts(p: np.ndarray) -> int:   # the held cup already touches the hand in the planning model: only new contacts count
        return len(planner.planner.check_for_env_collision(_folded_qpos(planner, agent, p, arm, float(body[2]))))

    base_hits = contacts(pose)
    tried = []
    for d in np.arange(CARRY_ROLL_STEP, CARRY_ROLL_MAX + 1e-9, CARRY_ROLL_STEP):
        reach = place_reachable(planner, agent, pose + d * ahead, place_pose, arm)
        hit = contacts(pose + (d + CARRY_ROLL_CLEARANCE) * ahead) > base_hits
        tried.append((round(float(d), 2), reach, hit))
        if reach and not hit:
            break
    else:
        logger.info("carry: place pose out of reach from %s and not within %.2f m ahead (d, reachable, collides): %s", np.round(pose, 3).tolist(), CARRY_ROLL_MAX, tried)
        return 0.0
    start = pose[:2].copy()
    for _ in range(300):   # bounded: a stuck base gives up
        if float(np.hypot(*(base_xyyaw(agent)[:2] - start))) >= d:
            break
        planner._step(planner._compose(arm, body, base_action(CARRY_ROLL_SPEED, 0.0)))
        if planner.truncated:
            return 0.0
    for _ in range(5):
        planner._step(planner._compose(arm, body, np.zeros(2)))
    planner.planner.update_from_simulation()
    rolled = float(np.hypot(*(base_xyyaw(agent)[:2] - start)))
    logger.info("carry: rolled %.3f m ahead (asked %.3f) to bring the place pose in reach", rolled, d)
    return rolled
