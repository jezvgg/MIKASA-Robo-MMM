"""Measure pre-release settling from consecutive physical poses at control ticks."""
from __future__ import annotations

import numpy as np


class ReleaseSettling:
    """Require four quiet intervals; a moving sample resets the entire window.

    Contact-constrained bodies can have nonzero solver velocities while their
    actual poses remain stationary. Measure displacement and rotation over each
    control interval instead, and independently require a quiet arm. No state is
    changed. The task success checker still uses its original velocity checks.
    """

    def __init__(self, pose, dt, *, required=4):
        self.previous = np.asarray(pose, dtype=float).copy()
        self.dt = float(dt)
        self.required = required
        self.stable_steps = 0
        self.linear_speed = float("inf")
        self.angular_speed = float("inf")
        self.arm_speed = float("inf")

    @property
    def ready(self):
        return self.stable_steps >= self.required

    def update(self, pose, arm_velocity):
        current = np.asarray(pose, dtype=float)
        self.linear_speed = float(np.linalg.norm(current[:3] - self.previous[:3]) / self.dt)
        qa = self.previous[3:] / np.linalg.norm(self.previous[3:])
        qb = current[3:] / np.linalg.norm(current[3:])
        self.angular_speed = float(2 * np.arccos(np.clip(abs(qa @ qb), 0., 1.)) / self.dt)
        self.arm_speed = float(np.max(np.abs(arm_velocity)))
        stable = (self.linear_speed < .01 and self.angular_speed < .5
                  and self.arm_speed < .05)
        self.stable_steps = self.stable_steps + 1 if stable else 0
        self.previous = current.copy()
        return self.ready
