"""Seeded, logged positional noise before collision-checked motion planning."""
from __future__ import annotations

import numpy as np
import sapien


class WaypointNoise:
    """Perturb goals, never recorded actions or the physical simulator state.

    Each call draws independently. A pose used for both a feasibility probe and
    execution must be sampled once and reused. Axes disabled for a goal stay exact.
    """

    def __init__(self, seed, amplitude_m, log):
        if not 0 <= amplitude_m <= 0.01:
            raise ValueError("Waypoint noise must be between 0 and 0.01 metres")
        self.seed = int(seed)
        self.amplitude_m = float(amplitude_m)
        self.rng = np.random.default_rng(self.seed)
        self.log = log
        self.index = 0
        log("waypoint noise configuration", seed=self.seed,
            amplitude_m=self.amplitude_m, distribution="uniform_independent_axes")

    def point(self, label, position, axes=(True, True, True)):
        original = np.asarray(position, dtype=np.float64).copy()
        if original.shape != (3,):
            raise ValueError("A waypoint must have three coordinates")
        offset = self.rng.uniform(-self.amplitude_m, self.amplitude_m, 3)
        offset *= np.asarray(axes, dtype=bool)
        result = original + offset
        self.log("waypoint noise", waypoint=label, sample_index=self.index,
                 original_m=original.tolist(), offset_m=offset.tolist(),
                 goal_m=result.tolist(), axes=list(axes))
        self.index += 1
        return result

    def pose(self, label, pose, axes=(True, True, True)):
        return sapien.Pose(self.point(label, pose.p, axes), pose.q)
