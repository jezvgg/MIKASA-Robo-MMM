"""The rest path cannot change a frozen joint or introduce winding."""
import numpy as np
from planners.oracle.rest_paths import monotone_knots


def test_curved_rest_stays_between_endpoints_without_joint_reversals():
    start = np.array([.2, -.4, 1.6, .3, -2.1])
    goal = np.array([.2, .8, -.2, .386, 0.])
    path = monotone_knots(start, goal, np.array([1., .5, 4., 2., 1.]))
    np.testing.assert_array_equal(path[0], start)
    np.testing.assert_allclose(path[-1], goal, atol=1e-15)
    assert np.all(path >= np.minimum(start, goal) - 1e-12)
    assert np.all(path <= np.maximum(start, goal) + 1e-12)
    assert np.all(np.diff(path, axis=0) * (goal-start) >= -1e-12)
    assert np.all(path[:, 0] == start[0])
