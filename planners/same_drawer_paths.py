"""Reserve wrist travel during the first upper-drawer contact."""

from contextlib import contextmanager

import numpy as np

from planners.oracle.roll_paths import RollPathPlanner


class DrawerPathPlanner(RollPathPlanner):
    """Keep the first push from spending the apple grasp's wrist-roll range.

    This narrows planning candidates and checks complete paths. Physical joint
    limits and measured roll history stay unchanged, including after release.
    """

    wrist_budget = 1.45

    def accepts(self, qpos, *, move_group=False):
        if not super().accepts(qpos, move_group=move_group):
            return False
        if not getattr(self._owner, "_drawer_initial_wrist_budget", False):
            return True
        index = self._owner._roll_indices[2]
        if move_group:
            index = list(self.move_group_joint_indices).index(index)
        return bool(np.all(np.abs(np.atleast_2d(qpos)[:, index]) <= self.wrist_budget))

    @contextmanager
    def _limits(self):
        with super()._limits():
            if getattr(self._owner, "_drawer_initial_wrist_budget", False):
                limits = self._planner.joint_limits.copy()
                index = self._owner._roll_indices[2]
                limits[index, 0] = max(limits[index, 0], -self.wrist_budget)
                limits[index, 1] = min(limits[index, 1], self.wrist_budget)
                self._planner.joint_limits = limits
            yield
