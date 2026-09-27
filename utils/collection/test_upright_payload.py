"""The wrist compensation must preserve orientation and reject unreachable axes."""
import numpy as np
from planners.oracle.upright_payload import upright_wrist_candidates
from planners.oracle.wrist_pour import rotation


def test_compensation_recovers_many_grasp_axes_without_winding():
    rng = np.random.default_rng(24)
    for _ in range(100):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        expected = rng.uniform([-1.8,-2.8],[1.8,2.8])
        up = rotation([0,1,0],expected[0]) @ rotation([1,0,0],expected[1]) @ axis
        candidates = upright_wrist_candidates(axis,up,expected+.01,np.array([[-2.15,2.15],[-3.09,3.09]]))
        assert candidates
        np.testing.assert_allclose(candidates[0],expected,atol=1e-6)
        actual=rotation([0,1,0],candidates[0][0]) @ rotation([1,0,0],candidates[0][1]) @ axis
        np.testing.assert_allclose(actual,up,atol=1e-7)


def test_axis_that_two_wrist_joints_cannot_raise_is_refused():
    assert not upright_wrist_candidates([1,0,0],[0,1,0],[0,0],np.array([[-2,2],[-3,3]]))


def test_joint_limits_are_not_relaxed_for_uprightness():
    up = rotation([0,1,0],1.) @ np.array([0.,0.,1.])
    assert not upright_wrist_candidates([0,0,1],up,[0,0],np.array([[-.1,.1],[-.1,.1]]))
