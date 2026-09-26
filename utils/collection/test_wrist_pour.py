"""Kinematic counterexamples for one-joint pouring."""
import math
import numpy as np
from planners.oracle.wrist_pour import rotation, tilt_angles


def test_horizontal_wrist_reaches_165_in_both_directions():
    roots = tilt_angles([1, 0, 0], [0, 0, 1])
    assert any(x > 0 for x in roots) and any(x < 0 for x in roots)
    for delta in roots:
        z = (rotation([1, 0, 0], delta) @ [0, 0, 1])[2]
        assert np.isclose(math.degrees(math.acos(z)), 165)


def test_rejected_35_degree_grasp_cannot_pour_with_wrist_alone():
    p = math.radians(35)
    assert tilt_angles([math.cos(p), 0, -math.sin(p)], [0, 0, 1]) == []


def test_parallel_axis_cannot_change_tilt():
    assert tilt_angles([0, 0, 1], [0, 0, 1]) == []
