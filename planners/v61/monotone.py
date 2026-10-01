"""Monotone-heading path to a stand pose: the yaw only ever turns one way, from the start heading to the goal heading.

The heading follows theta(u) = theta0 + delta * u**p over the normalised arc length u. For a goal whose
bearing lies between the start and the goal heading, p and the length L are fixed by the displacement,
so the path hits the goal position and heading exactly. A goal outside that cone (the base would have
to turn past the final heading and back) has no such path: `plan_path` returns None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import MONOTONE_BEARING_SLACK, MONOTONE_MIN_CHORD, MONOTONE_SAMPLES

_U = (np.arange(MONOTONE_SAMPLES) + 0.5) / MONOTONE_SAMPLES
_U_EDGES = np.linspace(0.0, 1.0, MONOTONE_SAMPLES + 1)


def wrap(a: float) -> float:
    """Angle wrapped to (-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class MonotonePath:
    """Sampled reference path from the start pose to the goal pose."""
    xy: np.ndarray        # (N+1, 2)
    theta: np.ndarray     # (N+1,) heading along the path (continuous, from theta0 to theta0 + delta)
    curvature: np.ndarray  # (N+1,) d theta / d s, rad per metre of path (sign of delta)
    length: float
    direction: int        # +1 drives forward along the path, -1 in reverse
    delta: float          # total signed rotation


def _bearing_of(delta: float, p: float) -> float:
    """Direction of the displacement integral for exponent p, measured from theta0 along the turn, in [0, 2 pi)."""
    th = delta * _U ** p
    return float((np.arctan2(np.sin(th).sum(), np.cos(th).sum()) * np.sign(delta)) % (2.0 * np.pi))


def plan_path(pose: np.ndarray, goal: np.ndarray, direction: int = 1) -> Optional[MonotonePath]:
    """Monotone-heading path from `pose` to `goal` (the short way round), or None when the goal bearing is outside the heading cone."""
    delta = wrap(goal[2] - pose[2])
    d = goal[:2] - pose[:2]
    rho = float(np.hypot(*d))
    if rho < 1e-3:
        return None
    psi = wrap(float(np.arctan2(d[1], d[0])) - pose[2] + (0.0 if direction > 0 else np.pi))
    sgn = 1.0 if delta >= 0 else -1.0
    slack = np.deg2rad(MONOTONE_BEARING_SLACK)
    frac = float((psi * sgn + slack) % (2.0 * np.pi) - slack)   # bearing measured along the turn
    if frac < -slack or frac > abs(delta) + slack:
        return None
    if abs(delta) < np.deg2rad(1.0):
        p, delta_eff = 1.0, 1e-3 * sgn
    else:
        target = float(np.clip(frac, 0.02 * abs(delta), 0.98 * abs(delta)))
        lo, hi = np.log(0.03), np.log(30.0)
        for _ in range(40):                     # the bearing falls as p grows
            mid = 0.5 * (lo + hi)
            if _bearing_of(delta, np.exp(mid)) > target:
                lo = mid
            else:
                hi = mid
        p, delta_eff = float(np.exp(0.5 * (lo + hi))), delta
    th = pose[2] + delta_eff * _U ** p
    integ = np.array([np.cos(th).sum(), np.sin(th).sum()]) / MONOTONE_SAMPLES
    if float(np.hypot(*integ)) < MONOTONE_MIN_CHORD:
        return None                              # the heading sweeps so much that the path barely advances
    length = rho / float(np.hypot(*integ))
    th_e = pose[2] + delta_eff * _U_EDGES ** p
    ds = length / MONOTONE_SAMPLES
    steps = direction * ds * np.stack([np.cos(0.5 * (th_e[1:] + th_e[:-1])), np.sin(0.5 * (th_e[1:] + th_e[:-1]))], axis=1)
    xy = pose[:2] + np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])
    kappa = np.gradient(th_e, length * _U_EDGES)
    return MonotonePath(xy, th_e, kappa, float(length), direction, float(delta_eff))
