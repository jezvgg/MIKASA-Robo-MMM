"""The wrist compensation must preserve orientation and reject unreachable axes."""
import numpy as np
from types import SimpleNamespace
from planners.oracle.roll_paths import RollPathPlanner
from planners.oracle.upright_payload import preview_roll_history
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


def test_future_pour_accounts_for_grasp_excursion_and_restores_preview():
    owner = SimpleNamespace(_roll_indices=[0], _roll_low=np.array([0.]),
                            _roll_high=np.array([0.]))
    planner = RollPathPlanner(SimpleNamespace(), owner)
    assert planner.accepts([[1.]])
    try:
        with preview_roll_history(owner, [[0.], [-2.8], [-1.5]]):
            assert not planner.accepts([[1.]])  # 3.8 radians across the full episode
            assert planner.accepts([[-.1]])
            raise RuntimeError('candidate refused')
    except RuntimeError:
        pass
    assert planner.accepts([[1.]])
    np.testing.assert_array_equal(owner._roll_low, [0.])


def test_transfer_guard_checks_initial_grasp_every_step_and_restores_callback():
    import pytest
    from planners.oracle.upright_payload import enforce_upright, PayloadTiltError
    angle=[14.9];held=[True];steps=[]
    def matrix():
        result=np.eye(4);result[:3,:3]=rotation([1,0,0],np.deg2rad(angle[0]));return result
    target=SimpleNamespace(pose=[SimpleNamespace(sp=SimpleNamespace(to_transformation_matrix=matrix))])
    task=SimpleNamespace(cfg=SimpleNamespace(pour_axis_body=[0,0,1]),
                         agent=SimpleNamespace(is_grasping=lambda _:np.array(held)))
    original=lambda action:steps.append(action)
    planner=SimpleNamespace(_step=original)
    with pytest.raises(PayloadTiltError):
        with enforce_upright(planner,task,target):
            planner._step('upright')
            angle[0]=15.1
            planner._step('excessive tilt')
    assert planner._step is original and len(steps)==2
    with pytest.raises(PayloadTiltError):
        with enforce_upright(planner,task,target):pytest.fail('must reject at entry')
    angle[0]=0.;held[0]=False
    with pytest.raises(PayloadTiltError):
        with enforce_upright(planner,task,target):pytest.fail('lost grasp')
    assert planner._step is original
