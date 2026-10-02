"""Clearance of a stand: the drive there with the arm tucked must not touch the furniture or spin the base round.

The tucked hand sticks 0.8 m out of the base at table height, so turning the base to face the counter close to it
pokes the hand into a cabinet door. `route` checks the predicted drive (the robot's meshes against the planning world,
which holds the doors, the dishwasher and the walls) and, when the direct drive collides, looks for a via point further
back along the final heading where the base turns first and then drives straight in. `clear_ranked` puts the stands
without such a route (or turning the base more than CLEAR_MAX_TURN) after the others.
"""
from __future__ import annotations

import heapq
import logging
from typing import Iterable, Iterator, Optional

import numpy as np

from .arm_ik import _collides, _joint_index
from .config import (CLEAR_AHEAD, CLEAR_MAX_TURN, CLEAR_PENALTY, CLEAR_SAMPLE_EVERY, PREDICT_DT, PREDICT_MAX_STEPS, READY_ARM_POSTURE, VIA_COST,
                     VIA_DISTANCES)
from .gaze import base_xyyaw
from .servo import ServoState, servo_command

logger = logging.getLogger(__name__)
_NO_VIA = np.empty(0)


def drive_trace(pose0: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """(n, 3) poses of the ideal-unicycle rollout of the servo from `pose0` to `goal`, every CLEAR_SAMPLE_EVERY steps."""
    pose, state, trace = np.array(pose0, dtype=float), ServoState(), []
    for i in range(PREDICT_MAX_STEPS):
        v, w, arrived = servo_command(pose, goal, state)
        if i % CLEAR_SAMPLE_EVERY == 0 or arrived:
            trace.append(pose.copy())
        if arrived:
            break
        pose = pose + np.array([v * np.cos(pose[2]), v * np.sin(pose[2]), w]) * PREDICT_DT
    return np.array(trace)


def _turn(trace: np.ndarray) -> float:
    return float(np.abs(np.diff(np.unwrap(trace[:, 2]))).sum()) if len(trace) > 1 else 0.0


def _is_free(planner, agent, trace: np.ndarray, torso: float) -> bool:
    """No pose of the trace (also CLEAR_AHEAD further along the heading) touches the planning world with the arm tucked."""
    for p in trace:
        for ahead in (0.0, CLEAR_AHEAD):
            if _collides(planner, agent, p + ahead * np.array([np.cos(p[2]), np.sin(p[2]), 0.0]), READY_ARM_POSTURE, torso):
                return False
    return True


def route(planner, agent, pose0: np.ndarray, goal: np.ndarray) -> Optional[np.ndarray]:
    """The via point (x, y) of a collision-free drive from `pose0` to `goal` that turns the base under CLEAR_MAX_TURN:
    an empty array for the direct drive, None when there is none."""
    torso = float(planner.robot.get_qpos().cpu().numpy()[0][_joint_index(agent)["torso_lift_joint"]])
    direct = drive_trace(pose0, goal)
    if _turn(direct) <= CLEAR_MAX_TURN and _is_free(planner, agent, direct, torso):
        return _NO_VIA
    heading = np.array([np.cos(goal[2]), np.sin(goal[2])])
    for d in VIA_DISTANCES:   # ponytail: via points only along the final heading, add side steps if a layout needs them
        via = goal[:2] - d * heading
        leg1, leg2 = drive_trace(pose0, np.array([*via, goal[2]])), drive_trace(np.array([*via, goal[2]]), goal)
        if _turn(leg1) + _turn(leg2) <= CLEAR_MAX_TURN and _is_free(planner, agent, leg1, torso) and _is_free(planner, agent, leg2, torso):
            logger.debug("stand %s: via %s", np.round(goal, 2).tolist(), np.round(via, 2).tolist())
            return via
    return None


def clear_ranked(planner, agent, cands: Iterable) -> Iterator:
    """(cost, xy, heading, via) of the candidates, cheapest first; a stand with no clear route is pushed back by CLEAR_PENALTY (lazy)."""
    pose0, held = base_xyyaw(agent), []
    for n, (cost, xy, h) in enumerate(cands):
        while held and held[0][0] <= cost:
            yield heapq.heappop(held)[2:]
        via = route(planner, agent, pose0, np.array([xy[0], xy[1], h]))
        if via is None:
            heapq.heappush(held, (cost + CLEAR_PENALTY, n, cost + CLEAR_PENALTY, xy, h, _NO_VIA))
        else:
            yield cost + (VIA_COST if len(via) else 0.0), xy, h, via
    while held:
        yield heapq.heappop(held)[2:]
