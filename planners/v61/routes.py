"""Collision check of the predicted base route with the arm tucked: open cabinet doors, dishwasher, stove and counter edges."""
from __future__ import annotations

import numpy as np

from .arm_ik import _collides, _joint_index
from .config import READY_ARM_POSTURE, ROUTE_STEP_M, ROUTE_STEP_YAW
from .predict import rollout


def route_samples(trace: np.ndarray) -> list:
    """Poses of the trace thinned to one per ROUTE_STEP_M of travel or ROUTE_STEP_YAW of turning, the last one included."""
    if len(trace) == 0:
        return []
    out, last = [trace[0]], trace[0]
    for q in trace[1:]:
        if np.hypot(*(q[:2] - last[:2])) >= ROUTE_STEP_M or abs(float(q[2] - last[2])) >= ROUTE_STEP_YAW:
            out.append(q)
            last = q
    if out[-1] is not trace[-1]:
        out.append(trace[-1])
    return out[1:]    # the start pose is where the base already is


def route_free(planner, agent, pose0: np.ndarray, goal: np.ndarray) -> bool:
    """True when the base driven from `pose0` to `goal` as the servo law predicts, arm tucked, touches nothing in the planning world."""
    torso = float(planner.robot.get_qpos().cpu().numpy()[0][_joint_index(agent)["torso_lift_joint"]])
    return not any(_collides(planner, agent, q, READY_ARM_POSTURE, torso) for q in route_samples(rollout(pose0, goal).trace))
