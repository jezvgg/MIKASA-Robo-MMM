"""Gentle release of the held cup (v6.1e): wait until the cup is down and upright, then open the fingers gradually."""
from __future__ import annotations

import logging

import numpy as np

from .config import GRIPPER_CLOSED, GRIPPER_OPEN, RELEASE_MAX_DZ, RELEASE_MAX_TILT, RELEASE_RAMP_STEPS, RELEASE_WAIT_MAX

logger = logging.getLogger(__name__)


def cup_tilt(unwenv) -> float:
    """Angle (rad) between the cup's axis and the vertical."""
    q = unwenv.cup.pose.q[0].cpu().numpy().astype(np.float64)   # wxyz
    return float(np.arccos(np.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0)))


def _step_hold(planner, arm: np.ndarray, body: np.ndarray) -> None:
    planner._compose(arm, body, np.zeros(2))
    planner._step(planner._from_abs(planner._last_abs))


def wait_cup_down(planner, unwenv, arm: np.ndarray, body: np.ndarray) -> None:
    """Hold the commanded arm and torso until the cup stopped descending and stands upright (at most RELEASE_WAIT_MAX steps)."""
    z = float(unwenv.cup.pose.p[0].cpu().numpy()[2])
    for i in range(RELEASE_WAIT_MAX):
        z_new = float(unwenv.cup.pose.p[0].cpu().numpy()[2])
        if abs(z_new - z) <= RELEASE_MAX_DZ and cup_tilt(unwenv) <= RELEASE_MAX_TILT:
            logger.debug('release: cup still after %d extra steps, tilt %.1f deg', i, np.degrees(cup_tilt(unwenv)))
            return
        z = z_new
        _step_hold(planner, arm, body)
        if planner.truncated:
            return
    logger.warning('release: cup not still after %d steps (tilt %.1f deg)', RELEASE_WAIT_MAX, np.degrees(cup_tilt(unwenv)))


def open_gradually(planner, arm: np.ndarray, body: np.ndarray) -> None:
    """Ramp the gripper command from closed to open over RELEASE_RAMP_STEPS steps, holding arm and torso."""
    for i in range(RELEASE_RAMP_STEPS):
        planner.gripper_state = GRIPPER_CLOSED + (GRIPPER_OPEN - GRIPPER_CLOSED) * (i + 1) / RELEASE_RAMP_STEPS
        _step_hold(planner, arm, body)
        if planner.truncated:
            return
    planner.gripper_state = GRIPPER_OPEN

