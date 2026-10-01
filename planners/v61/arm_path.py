"""Rate- and acceleration-limited joint follower used for every arm motion the v6.1 servo loop commands."""
from __future__ import annotations

import numpy as np

from .config import ARM_MAX_ACCEL, ARM_MAX_STEP


class JointLimiter:
    """Moves a joint vector towards a (possibly moving) target without jumps.

    All joints are scaled together, so the joint-space path stays a straight line towards the target.
    The speed is capped by `ARM_MAX_STEP`, the speed change by `ARM_MAX_ACCEL`, and the speed falls with
    the square root of the remaining distance so the target is reached without overshoot.
    """

    def __init__(self, q: np.ndarray, max_step: float = ARM_MAX_STEP, max_accel: float = ARM_MAX_ACCEL) -> None:
        self.q = np.asarray(q, dtype=np.float64).copy()
        self.step_vec = np.zeros_like(self.q)
        self.max_step = max_step
        self.max_accel = max_accel

    def update(self, target: np.ndarray) -> np.ndarray:
        """Advance one env step towards `target`; returns the new command (a copy)."""
        err = np.asarray(target, dtype=np.float64) - self.q
        dist = float(np.max(np.abs(err)))
        if dist < 1e-9:
            want = np.zeros_like(err)
        else:
            speed = min(self.max_step, float(np.sqrt(2.0 * self.max_accel * dist)))
            want = err * (min(speed, dist) / dist)
        self.step_vec = self.step_vec + np.clip(want - self.step_vec, -self.max_accel, self.max_accel)
        self.q = self.q + self.step_vec
        return self.q.copy()


def _demo() -> None:
    lim = JointLimiter(np.zeros(2))
    tgt = np.array([1.0, -0.5])
    path = [lim.update(tgt) for _ in range(120)]
    steps = np.diff(np.array(path), axis=0)
    assert np.abs(steps).max() <= ARM_MAX_STEP + 1e-9
    assert np.abs(np.diff(steps, axis=0)).max() <= 2 * ARM_MAX_ACCEL + 1e-9
    assert np.allclose(path[-1], tgt, atol=5e-3), path[-1]


if __name__ == "__main__":
    _demo()
