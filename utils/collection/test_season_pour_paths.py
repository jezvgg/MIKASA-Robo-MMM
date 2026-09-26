"""An infeasible pour must fail before any arm motion is executed."""
from types import SimpleNamespace
import numpy as np
import sapien
import torch

from planners.oracle.wrist_pour import pour_with_wrist


def setup_planner(line_status='Success', grasped=True):
    q = np.zeros(15)
    calls = []
    names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint',
             'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint',
             'wrist_roll_joint']
    indices = [5, 7, 8, 9, 10, 11, 12]
    joints = {n: SimpleNamespace(active_index=[i]) for n, i in zip(names, indices)}
    joints['torso_lift_joint'] = SimpleNamespace(active_index=[3])
    robot = SimpleNamespace(
        active_joints_map=joints,
        get_qpos=lambda: torch.tensor(q[None]),
        links_map={'wrist_roll_link': SimpleNamespace(pose=[SimpleNamespace(sp=sapien.Pose([0, 0, .2]))])},
    )
    target = SimpleNamespace(pose=[SimpleNamespace(sp=sapien.Pose([0, 0, .2]))])
    task = SimpleNamespace(
        agent=SimpleNamespace(robot=robot, is_grasping=lambda _: torch.tensor([grasped]),
            controller=SimpleNamespace(controllers={'arm': SimpleNamespace(config=SimpleNamespace(joint_names=names))})),
        cfg=SimpleNamespace(pour_axis_body=[0, 0, 1], pour_xy_radius=.1,
                            pour_min_clearance=.05, pour_max_clearance=.3, hold_steps=15),
        bowl=SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0., 0., 0.]]))),
        control_timestep=.05,
    )
    def line(goal, current, **kwargs):
        np.testing.assert_array_equal(np.delete(goal, 12), np.delete(current, 12))
        assert kwargs['qpos_step'] <= .02
        calls.append('preview')
        return {'status': line_status}
    planner = SimpleNamespace(
        planner=SimpleNamespace(update_from_simulation=lambda: None, plan_qpos_line=line),
        fixed_action_targets={9: .42},
    )
    def execute(path, *, refine):
        calls.append('execute')
        assert planner.fixed_action_targets == {**{i: 0. for i in range(6)}, 7: -1., 10: 0., 11: 0., 12: 0.}
        return -1
    planner.follow_forward_path_w_refinement = execute
    return planner, task, target, calls


def test_collision_failure_never_executes_endpoint():
    p, task, target, calls = setup_planner('finger collision')
    assert pour_with_wrist(SimpleNamespace(), p, task, target) == -1
    assert calls and set(calls) == {'preview'}
    assert p.fixed_action_targets == {9: .42}


def test_pour_locks_other_actuators_and_restores_lock_on_failure():
    p, task, target, calls = setup_planner()
    assert pour_with_wrist(SimpleNamespace(), p, task, target) == -1
    assert calls == ['preview', 'execute']
    assert p.fixed_action_targets == {9: .42}


def test_missing_grasp_refuses_before_planning():
    p, task, target, calls = setup_planner(grasped=False)
    assert pour_with_wrist(SimpleNamespace(), p, task, target) == -1
    assert calls == []
