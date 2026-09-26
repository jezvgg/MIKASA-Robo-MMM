"""Check actual camera geometry and command speed, not just nonconstant columns."""

import numpy as np
import pytest
from planners.oracle.head_tracking import aim_angles, rotation, HeadTracker


@pytest.mark.parametrize(
    "pan,tilt,distance", [(-1.0, -0.2, 3.0), (0.6, 0.4, 1.0), (-0.3, 0.9, 0.5)]
)
def test_camera_ray_recovers_head_angles(pan, tilt, distance):
    # Neutral DSFetch head geometry and the midpoint of its offset cameras.
    origin = np.array([0.14253, 0, 0.057999])
    eye = np.array([-0.445, 0, 0.0225])
    pitch = 0.306
    ray = np.array([np.cos(pitch), 0, -np.sin(pitch)])
    target = rotation("z", pan)[:3, :3] @ (
        origin + rotation("y", tilt)[:3, :3] @ (eye + distance * ray)
    )
    actual = aim_angles(target, origin, eye, pitch, [[-1.56, 1.56], [-0.75, 1.44]])
    np.testing.assert_allclose(actual, [pan, tilt], atol=1e-9)


def test_slew_limit_and_joint_bounds_for_unreachable_target():
    angles = aim_angles(
        np.array([-2.0, 1.0, 3.0]),
        np.array([0.14, 0, 0.06]),
        np.array([-0.445, 0, 0.0225]),
        0.306,
        [[-1.56, 1.56], [-0.75, 1.44]],
    )
    assert -1.56 <= angles[0] <= 1.56
    assert -0.75 <= angles[1] <= 1.44
    tracker = object.__new__(HeadTracker)
    tracker.last_command = np.zeros(2)
    tracker.max_delta = 0.05
    tracker.desired = lambda: angles
    before = tracker.last_command.copy()
    command = tracker.command()
    assert np.max(np.abs(command - before)) <= 0.05 + 1e-12
    command[:] = 99
    assert np.max(np.abs(tracker.last_command)) <= 0.05 + 1e-12


def test_small_pose_fluctuations_do_not_make_head_dither():
    tracker = object.__new__(HeadTracker)
    tracker.last_command = np.array([0.2, 0.4])
    tracker.max_delta = 0.05
    start = tracker.last_command.copy()
    for offset in [0.001, -0.001, 0.004, -0.004]:
        tracker.desired = lambda: start + offset
        np.testing.assert_array_equal(tracker.command(), start)
    tracker.desired = lambda: start + 0.01
    np.testing.assert_allclose(tracker.command(), start + 0.01)
