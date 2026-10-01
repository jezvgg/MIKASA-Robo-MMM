"""Forward-only pose servo for a differential base: drives to (x, y, yaw) turning while it drives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .monotone import MonotonePath, plan_path
from .config import (CLOSE_IN_MAX_SPEED, MONOTONE_FEEDBACK, NEAR_HOLD_RHO, MONOTONE_MAX_FAILED_PLANS, CLOSE_IN_LATCH_RHO, MONOTONE_REPLAN_EVERY, MONOTONE_LOOKAHEAD, MONOTONE_MIN_SPEED, ARRIVE_YAW_TOL, BASE_MAX_SPEED, BASE_MAX_SPEED_NORM, BASE_MAX_YAW_NORM,
                     BASE_MAX_YAW_RATE, CREEP_MIN_DISTANCE, CREEP_SPEED, NEAR_LATCH_RHO, NEAR_LATCH_PAR, NEAR_LATCH_PAR_RHO, YAW_TRIM_MIN_RATE, REVERSE_MAX_DISTANCE, REVERSE_SPEED, K_ALPHA, K_BETA, K_RHO, TURN_FIRST_ALPHA)


def wrap(a: float) -> float:
    """Angle wrapped to (-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def polar(pose: np.ndarray, goal: np.ndarray) -> tuple[float, float, float]:
    """Polar coordinates (rho, alpha, beta) of `goal` seen from `pose`: distance, bearing off the heading, final heading off the bearing."""
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    gamma = np.arctan2(dy, dx)
    return float(np.hypot(dx, dy)), wrap(gamma - pose[2]), wrap(goal[2] - gamma)


@dataclass
class ServoState:
    """Memory of one approach: once the base is close enough to the stand the position is latched for good."""
    latched: bool = False
    path: Optional[MonotonePath] = None
    planned: bool = False
    index: int = 0
    since_plan: int = 0
    failed_plans: int = 0
    close_in: bool = False       # once close in without a path, stay in the close-in mode (no polar-law spin past the stand)


def servo_command(pose: np.ndarray, goal: np.ndarray, state: ServoState | None = None) -> tuple[float, float, bool]:
    """(v m/s, w rad/s, arrived) of the pose-servo law towards `goal` (x, y, yaw).

    Far from the goal the polar law curves the path so the final heading is reached on arrival; when
    the goal is more than TURN_FIRST_ALPHA off the heading the base creeps while it turns, and a goal
    that is close and behind is reached in reverse. Once the base is within NEAR_LATCH_RHO of the goal
    (or abreast of it) the position is latched: the base stops and only trims the yaw, in one
    direction and without a polar bearing, which is noise at that range. The last centimetres are the arm's.
    """
    state = state if state is not None else ServoState()
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    rho = float(np.hypot(dx, dy))
    yaw_err = wrap(goal[2] - pose[2])
    e_par = float(dx * np.cos(pose[2]) + dy * np.sin(pose[2]))
    if rho < NEAR_LATCH_RHO or (abs(e_par) < NEAR_LATCH_PAR and rho < NEAR_LATCH_PAR_RHO):
        state.latched = True
    if state.latched:
        if abs(yaw_err) < ARRIVE_YAW_TOL:
            return 0.0, 0.0, True
        w = float(np.sign(yaw_err) * np.clip(max(YAW_TRIM_MIN_RATE, 2.0 * abs(yaw_err)), 0.0, BASE_MAX_YAW_RATE))
        return 0.0, w, False
    if not state.planned:
        state.planned = True
        state.path = _initial_path(pose, goal, rho)
    elif state.path is not None and state.since_plan >= MONOTONE_REPLAN_EVERY:
        fresh = plan_path(pose, goal, state.path.direction)
        if fresh is not None:
            state.path, state.index, state.since_plan, state.failed_plans = fresh, 0, 0, 0
        else:
            state.since_plan = 0
            if rho >= NEAR_HOLD_RHO:               # close in, a slightly off path is kept: the polar law is noise there
                state.failed_plans += 1
            if state.failed_plans >= MONOTONE_MAX_FAILED_PLANS:
                state.path = None                  # drifted out of the heading cone: fall back to the polar law
    if state.path is not None:
        state.since_plan += 1
        v, w = _track(pose, goal, state)
        if state.index >= len(state.path.theta) - 1 and rho < NEAR_HOLD_RHO:
            state.latched = True                   # end of the reference path: only the yaw is left to trim
        return v, w, False
    state.close_in = state.close_in or rho < NEAR_HOLD_RHO
    if state.close_in:
        if rho < CLOSE_IN_LATCH_RHO:
            state.latched = True
            return 0.0, 0.0, False
        return _close_in(e_par, yaw_err, state)
    gamma = np.arctan2(dy, dx)
    alpha = wrap(gamma - pose[2])
    if abs(alpha) > np.pi / 2 and rho < REVERSE_MAX_DISTANCE:
        a_rev = wrap(alpha + np.pi)
        w = K_ALPHA * a_rev
        return -float(min(REVERSE_SPEED, K_RHO * rho) * max(0.0, np.cos(a_rev)) ** 2), float(np.clip(w, -BASE_MAX_YAW_RATE, BASE_MAX_YAW_RATE)), False
    beta = wrap(goal[2] - gamma)
    w = K_ALPHA * alpha + K_BETA * beta
    if abs(alpha) > TURN_FIRST_ALPHA:
        v = CREEP_SPEED if rho > CREEP_MIN_DISTANCE else 0.0
    else:
        v = min(BASE_MAX_SPEED, K_RHO * rho) * max(0.0, np.cos(alpha)) ** 2
        v = max(v, min(CREEP_SPEED, rho))
    return float(v), float(np.clip(w, -BASE_MAX_YAW_RATE, BASE_MAX_YAW_RATE)), False


def _close_in(e_par: float, yaw_err: float, state: ServoState) -> tuple[float, float, bool]:
    """No path and the stand is near: trim the heading one way and close the along-track gap; the bearing is noise at this range."""
    if abs(e_par) < NEAR_LATCH_PAR and abs(yaw_err) < ARRIVE_YAW_TOL:
        state.latched = True
        return 0.0, 0.0, True
    w = 0.0 if abs(yaw_err) < ARRIVE_YAW_TOL else float(np.sign(yaw_err) * np.clip(max(YAW_TRIM_MIN_RATE, 2.0 * abs(yaw_err)), 0.0, BASE_MAX_YAW_RATE))
    v = 0.0 if abs(e_par) < NEAR_LATCH_PAR else float(np.clip(K_RHO * e_par, -REVERSE_SPEED, CLOSE_IN_MAX_SPEED))
    return v, w, False


def _initial_path(pose: np.ndarray, goal: np.ndarray, rho: float) -> Optional[MonotonePath]:
    """First feasible monotone path: forward, else (close goals only) in reverse."""
    path = plan_path(pose, goal, 1)
    if path is None and rho < REVERSE_MAX_DISTANCE:
        path = plan_path(pose, goal, -1)
    return path


def _track(pose: np.ndarray, goal: np.ndarray, state: ServoState) -> tuple[float, float]:
    """(v, w) that follows the monotone reference path; the yaw rate never points against the planned rotation."""
    path = state.path
    n = len(path.theta) - 1
    lo, hi = state.index, min(n, state.index + 40)
    dist = np.hypot(*(path.xy[lo:hi + 1] - pose[:2]).T)
    state.index = lo + int(np.argmin(dist))
    i = min(n, state.index + MONOTONE_LOOKAHEAD)
    remaining = path.length * (1.0 - state.index / n)
    speed = max(min(BASE_MAX_SPEED, K_RHO * remaining), MONOTONE_MIN_SPEED)
    w = path.curvature[state.index] * speed + MONOTONE_FEEDBACK * (path.theta[i] - (pose[2] - 2.0 * np.pi * np.round((pose[2] - path.theta[i]) / (2.0 * np.pi))))
    sgn = 1.0 if path.delta >= 0 else -1.0
    theta_now = pose[2] - 2.0 * np.pi * np.round((pose[2] - path.theta[i]) / (2.0 * np.pi))
    if (path.theta[-1] - theta_now) * sgn < 0:
        w = 0.0                                    # already past the goal heading: do not turn back
    w = sgn * max(0.0, sgn * w)
    scale = min(1.0, BASE_MAX_YAW_RATE / max(abs(w), 1e-9))
    return path.direction * speed * scale, w * scale


def base_action(v: float, w: float) -> np.ndarray:
    """Normalised [forward, yaw] action for velocities in m/s and rad/s."""
    return np.array([v / BASE_MAX_SPEED_NORM, w / BASE_MAX_YAW_NORM])
