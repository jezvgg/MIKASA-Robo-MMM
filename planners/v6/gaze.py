"""Head gaze and per-step base-pose tracing, installed on the solver's single step funnel."""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from .config import GAZE_MAX_STEP, HEAD_PAN_LIMITS, HEAD_TILT_LIMITS


def heading(agent) -> np.ndarray:
    """Unit horizontal heading of the base (world frame)."""
    direction = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0].copy()
    direction[2] = 0.0
    return direction / np.linalg.norm(direction)


def base_xyyaw(agent) -> np.ndarray:
    """World (x, y, yaw) of the base link."""
    mat = agent.base_link.pose.sp.to_transformation_matrix()
    return np.array([mat[0, 3], mat[1, 3], np.arctan2(mat[1, 0], mat[0, 0])])


def head_look_at(agent, target_pos: np.ndarray) -> tuple[float, float]:
    """Head (pan, tilt) that points the head camera at `target_pos`, within the joint limits."""
    head = next(l for l in agent.robot.get_links() if l.get_name() == "head_camera_link")
    delta = np.asarray(target_pos, dtype=float).reshape(3) - np.asarray(head.pose.sp.p, dtype=float)
    pan = np.arctan2(delta[1], delta[0]) - base_xyyaw(agent)[2]
    pan = (pan + np.pi) % (2 * np.pi) - np.pi
    tilt = np.arctan2(-delta[2], np.hypot(delta[0], delta[1]))
    return float(np.clip(pan, *HEAD_PAN_LIMITS)), float(np.clip(tilt, *HEAD_TILT_LIMITS))


def install_gaze(planner, agent, target_pos: Callable[[], np.ndarray]) -> List[np.ndarray]:
    """Point the head at `target_pos()` on every env step; return the list the base trace fills.

    The head moves at most GAZE_MAX_STEP per step; the recorded action carries the executed head target.
    Every step also appends the base (x, y, yaw) to the returned trace.
    """
    head_slot = len(agent.controller.controllers["arm"].config.joint_names) + 1
    head = agent.controller.controllers["body"].qpos[0].cpu().numpy()[:2].astype(np.float64)
    raw_step = planner._guard.step
    trace: List[np.ndarray] = []

    def step(action, tape_entry=None):
        want = np.array(head_look_at(agent, target_pos()))
        head[:] = head + np.clip(want - head, -GAZE_MAX_STEP, GAZE_MAX_STEP)
        action = np.asarray(action, dtype=np.float64).copy()
        action[head_slot:head_slot + 2] = head
        if tape_entry is not None:
            tape_entry = np.asarray(tape_entry, dtype=np.float64).copy()
            tape_entry[head_slot:head_slot + 2] = head
        trace.append(base_xyyaw(agent))
        return raw_step(action, tape_entry=tape_entry)

    planner._guard.step = step
    return trace
