"""Consolidated, behavior-preserving collection regression tests."""


# Source group: test_bounded_candidates.py
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import torch

from planners import season_dish_planner as search


def test_duplicate_endpoints_do_not_consume_continuation_slots():
    q = np.zeros(15)
    values = [q.copy(), q + .001, q + .2, q + .201]
    kept = search.distinct_goals(values, list(range(7, 13)))
    assert len(kept) == 2
    np.testing.assert_equal(kept[0], values[0])
    np.testing.assert_equal(kept[1], values[2])


def test_profile26_runs_eight_unrolled_ik_requests_with_32_starts():
    names = [
        'root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint',
        'torso_lift_joint', 'head_pan_joint', 'shoulder_pan_joint', 'head_tilt_joint',
        'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint',
        'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint',
        'r_gripper_finger_joint', 'l_gripper_finger_joint',
    ]
    current = np.zeros(15)
    current[9] = 1.
    current[11] = -.6
    robot = SimpleNamespace(
        get_qpos=lambda: torch.tensor(current[None]),
        get_qlimits=lambda: torch.tensor(np.tile([-3.14, 3.14], (1, 15, 1))),
        active_joints_map={n: SimpleNamespace(active_index=[i]) for i, n in enumerate(names)},
    )
    calls = []

    def ik(pose, initial, mask, **kwargs):
        calls.append(kwargs['n_init_qpos'])
        goals = []
        for i in range(20):
            q = current.copy()
            q[7] = i * .02
            q[8] = len(calls) * .15
            q[11] = -.6
            goals.append(q)
        return 'Success', goals

    engine = SimpleNamespace(
        fold_qpos=lambda q: q.copy(), IK=ik,
        _transform_goal_to_wrt_base=lambda q: q,
        joint_limits=np.tile([-3.14, 3.14], (15, 1)),
        _root_cols=lambda: [0, 1, 2],
        accepts=lambda q: bool(np.all(np.abs(q[[8, 10, 12]]) < np.pi - .05)),
    )
    planner = SimpleNamespace(
        planner=engine, _roll_indices=[8, 10, 12],
        _roll_low=np.zeros(3), _roll_high=np.zeros(3), env=None,
    )
    task = SimpleNamespace(agent=SimpleNamespace(
        robot=robot,
        controller=SimpleNamespace(controllers={'arm': SimpleNamespace(
            config=SimpleNamespace(joint_names=[names[i] for i in [5, 7, 8, 9, 10, 11, 12]])
        )}),
    ))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(search.common, 'say', lambda *args, **kwargs: None)
    try:
        result = search.candidates(
            planner, task, sapien.Pose([.7, 0, 1.]), sapien.Pose([.6, 0, 1.]), Counter(), n_init=32
        )
    finally:
        monkeypatch.undo()
    assert calls == [32] * 8
    assert 0 < len(result) <= 8
    assert all(engine.accepts(row[0]) and row[0][11] <= -.05 for row in result)


def test_dual_hover_keeps_both_start_families_within_original_budget():
    from planners.season_dish_planner import hover_goals

    calls = []
    current = np.zeros(15)

    def ik(pose, initial, mask, *, n_init_qpos):
        calls.append(n_init_qpos)
        goal = current.copy()
        goal[7] = .5 if len(calls) == 1 else -.5
        return 'Success', [goal, goal + .001]

    p = SimpleNamespace(
        move_group_joint_indices=list(range(3, 13)), fold_qpos=lambda q: q.copy(),
        unfold_qpos=lambda q, **kw: q.copy(), _transform_goal_to_wrt_base=lambda q: q,
        IK=ik, joint_limits=np.tile([-3.14, 3.14], (15, 1)),
    )
    preferred = {'approach': {'position': np.ones((1, 10))}}
    goals, records = hover_goals(p, current, sapien.Pose(), n_init=32,
                                 preferred=preferred, mode='dual')
    assert calls == [16, 16]
    assert len(goals) == 2
    assert {q[7] for q in goals} == {-.5, .5}
    assert [r['initial'] for r in records] == ['preferred', 'current']


# Source group: test_unrolled_grasp.py
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from planners.season_dish_planner import (
    ArmKinematics,
    NAMES,
    TEMPLATES,
    UNROLLED_CARRY_TARGETS,
    unrolled,
)

ROLLED_CARRY_REFERENCE = dict(torso_lift_joint=.20, shoulder_pan_joint=-.6557497327377413,
    shoulder_lift_joint=-.12904068645626654, upperarm_roll_joint=1.9212388766147683,
    elbow_flex_joint=1.6507435522454401, forearm_roll_joint=-2.8097833941893957,
    wrist_flex_joint=-1.9402379512175012, wrist_roll_joint=1.907999995746957)

ACTIVE = ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint', 'torso_lift_joint',
          'head_pan_joint', 'shoulder_pan_joint', 'head_tilt_joint', 'shoulder_lift_joint',
          'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint',
          'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint']
URDF = Path(__file__).resolve().parents[2] / 'robots/fetch/fetch.urdf'


def kinematics():
    root = SimpleNamespace(sp=SimpleNamespace(to_transformation_matrix=lambda: np.eye(4)))
    robot = SimpleNamespace(get_active_joints=lambda: [SimpleNamespace(name=n) for n in ACTIVE],
                            root_pose=[root])
    return ArmKinematics(SimpleNamespace(agent=SimpleNamespace(urdf_path=str(URDF), robot=robot)))


def qpos(values):
    q = np.zeros(len(ACTIVE))
    q[ACTIVE.index('torso_lift_joint')] = .2
    q[ACTIVE.index('shoulder_pan_joint')] = -.4
    for name, value in values.items():
        q[ACTIVE.index(name)] = value
    return q


def test_unrolled_templates_put_the_hand_in_the_same_pose():
    fk = kinematics()
    for template in TEMPLATES:
        a = fk.matrix(qpos(dict(zip(NAMES, template))))
        b = fk.matrix(qpos(dict(zip(NAMES, unrolled(template)))))
        np.testing.assert_allclose(a, b, atol=1e-9)
        assert unrolled(template)[4] == -template[4]


def test_unrolled_carry_posture_is_the_same_hand_pose_with_a_small_forearm_roll():
    fk = kinematics()
    np.testing.assert_allclose(fk.matrix(qpos(ROLLED_CARRY_REFERENCE)),
                               fk.matrix(qpos(UNROLLED_CARRY_TARGETS)), atol=1e-9)
    assert abs(UNROLLED_CARRY_TARGETS['forearm_roll_joint']) < .5
    assert UNROLLED_CARRY_TARGETS['wrist_flex_joint'] > 0 > ROLLED_CARRY_REFERENCE['wrist_flex_joint']


def test_unrolled_is_an_involution_up_to_full_turns():
    for template in TEMPLATES:
        back = np.array(unrolled(unrolled(template)))
        diff = (back - np.array(template) + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(diff, 0, atol=1e-12)


# Source group: test_season_pour_paths.py
from types import SimpleNamespace
import numpy as np
import sapien
import torch

from planners.season_dish_planner import above_via, pour_with_wrist


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


def test_above_via_returns_none_when_straight_plan_refuses(monkeypatch):
    p = SimpleNamespace(_straight_failures=['counter contact'])
    monkeypatch.setattr('planners.season_dish_planner.straight_plan', lambda *a, **kw: None)
    assert above_via(p, SimpleNamespace(), np.zeros(15), sapien.Pose(), .15) is None
    assert p._above_status == {'statuses': ['counter contact']}


from planners.season_dish_planner import rotation


# Source group: test_wrist_pour.py
import math
import numpy as np
from planners.season_dish_planner import rotation, tilt_angles


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
