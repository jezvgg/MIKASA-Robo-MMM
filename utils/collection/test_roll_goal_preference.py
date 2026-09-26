"""Keep nearby roll solutions ahead of winding solutions without skipping collisions."""

from types import SimpleNamespace

import numpy as np

from planners.oracle.collection_solver import CollectionMotionPlanner
from robots.fetch.extand import FetchMotionPlanningSapienSolver


def test_roll_preference_tries_next_candidate_after_a_collision(monkeypatch):
    planner = object.__new__(CollectionMotionPlanner)
    planner._roll_indices = [8, 10, 12]
    planner._roll_low = np.array([0.68, -0.12, 0.0])
    planner._roll_high = np.array([0.70, -0.10, 0.02])
    planner.planner = SimpleNamespace(joint_limits=np.tile([-6.28, 6.28], (15, 1)))
    current = np.zeros(15)
    current[planner._roll_indices] = planner._roll_high
    goals = np.tile(current, (3, 1))
    # These are the measured handle probe's high-turn and two bounded solutions.
    goals[:, planner._roll_indices] = [
        [3.46, -2.97, 1.15],
        [-1.93, 2.47, 0.57],
        [0.195, 0.373, 1.15],
    ]
    attempted = []

    def inherited_line(self, pose, offered, *args, **kwargs):
        attempted.append(np.asarray(offered)[0])
        return None if len(attempted) == 1 else "executed"

    monkeypatch.setattr(
        FetchMotionPlanningSapienSolver, "_line_to_ik_goals", inherited_line
    )
    result = planner._line_to_ik_goals(
        None, goals, current, current, 0, 1, None, None, "test"
    )
    assert result == "executed"
    np.testing.assert_array_equal(attempted, goals[[2, 1]])
