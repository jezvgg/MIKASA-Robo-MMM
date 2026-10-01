"""Forward-only pose servo for a differential base: drives to (x, y, yaw) turning while it drives."""
from __future__ import annotations

import numpy as np

from .config import (ARRIVE_POS_TOL, ARRIVE_YAW_TOL, BASE_MAX_SPEED, BASE_MAX_SPEED_NORM, BASE_MAX_YAW_NORM,
                     BASE_MAX_YAW_RATE, CREEP_MIN_DISTANCE, CREEP_SPEED, REVERSE_MAX_DISTANCE, REVERSE_SPEED, K_ALPHA, K_BETA, K_RHO, TURN_FIRST_ALPHA)


def wrap(a: float) -> float:
    """Angle wrapped to (-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def servo_command(pose: np.ndarray, goal: np.ndarray) -> tuple[float, float, bool]:
    """(v m/s, w rad/s, arrived) of the polar pose-servo law towards `goal` (x, y, yaw).

    The base only drives forward. Far from the goal the law curves the path so the final heading
    is reached on arrival; when the goal is more than TURN_FIRST_ALPHA off the heading the base
    creeps forward while it turns, and close to the goal it only trims the heading.
    """
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    rho = float(np.hypot(dx, dy))
    yaw_err = wrap(goal[2] - pose[2])
    if rho < ARRIVE_POS_TOL:
        return 0.0, float(np.clip(2.0 * yaw_err, -BASE_MAX_YAW_RATE, BASE_MAX_YAW_RATE)), abs(yaw_err) < ARRIVE_YAW_TOL
    gamma = np.arctan2(dy, dx)
    alpha = wrap(gamma - pose[2])
    if abs(alpha) > np.pi / 2 and rho < REVERSE_MAX_DISTANCE:
        # goal is close and behind: back up (a short reverse beats an orbit); the yaw law is mirrored
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


def base_action(v: float, w: float) -> np.ndarray:
    """Normalised [forward, yaw] action for velocities in m/s and rad/s."""
    return np.array([v / BASE_MAX_SPEED_NORM, w / BASE_MAX_YAW_NORM])
