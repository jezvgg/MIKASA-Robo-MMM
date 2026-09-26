import numpy as np
from planners.oracle.settling import ReleaseSettling


def test_continuous_motion_and_arm_motion_reset_stable_window():
    pose = np.array([0., 0., 1., 1., 0., 0., 0.])
    check = ReleaseSettling(pose, .05)
    for _ in range(3):
        assert not check.update(pose, np.zeros(7))
    pose[0] += .001
    assert not check.update(pose, np.zeros(7))
    assert check.stable_steps == 0
    for _ in range(3):
        assert not check.update(pose, np.zeros(7))
    assert not check.update(pose, np.full(7, .06))
    for _ in range(3):
        assert not check.update(pose, np.zeros(7))
    assert check.update(pose, np.zeros(7))


def test_quaternion_sign_is_same_pose_but_rotation_is_motion():
    pose = np.array([0., 0., 1., 1., 0., 0., 0.])
    check = ReleaseSettling(pose, .05)
    for _ in range(4):
        pose[3:] *= -1
        check.update(pose, np.zeros(7))
    assert check.ready
    pose[3:] = [np.cos(.03), 0., 0., np.sin(.03)]
    assert not check.update(pose, np.zeros(7))
    assert check.angular_speed > .5


def test_persistent_jitter_never_becomes_ready_by_timeout():
    pose = np.array([0., 0., 1., 1., 0., 0., 0.])
    check = ReleaseSettling(pose, .05)
    for i in range(20):
        pose[2] = 1. + .001 * (i % 2)
        assert not check.update(pose, np.zeros(7))
