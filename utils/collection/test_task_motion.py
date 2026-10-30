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


# Source group: test_season_pour_paths.py (its one-tip pour tests went with pour_with_wrist; the
# shaking pour's are in the test_season_shakes.py group)
from types import SimpleNamespace
import numpy as np
import sapien

from planners.season_dish_planner import above_via


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


# Source group: test_season_shakes.py (the shaking pour of profile 28: the wrist geometry, the
# planner's shake sequence on stubs, and the shifted hover; the counter and the scene are in
# test_task_scenes.py)
import math
from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import torch

from planners.oracle.oracle_season_dish import (SHAKE_AMPLITUDE_DEG, SHAKE_LEAN_MARGIN_DEG,
                                                SHAKE_LEAN_MARGIN_EXECUTED_DEG, lean_after, pour_and_shake,
                                                rest_angles, shake_geometry_ok, swing_angle)

SHAKE_CFG = SimpleNamespace(shake_out_deg=20., shake_return_deg=10., pour_axis_body=[0, 0, 1], pour_xy_radius=.1,
                            pour_min_clearance=.05, pour_max_clearance=.3, hold_steps=20)


def test_horizontal_wrist_rests_short_of_upside_down_and_swings_back_toward_upright():
    rest = 10. - SHAKE_LEAN_MARGIN_DEG
    deltas = rest_angles([1, 0, 0], [0, 0, 1], rest)
    assert [round(math.degrees(d)) for d in deltas[:2]] == [173, -173]
    for d in deltas[:2]:
        swing = swing_angle(d, SHAKE_AMPLITUDE_DEG)
        assert abs(swing) < abs(d)                                  # toward the start: D2 spends no more
        assert lean_after([1, 0, 0], [0, 0, 1], d) == pytest.approx(rest)
        assert lean_after([1, 0, 0], [0, 0, 1], swing) == pytest.approx(rest + SHAKE_AMPLITUDE_DEG)
        assert shake_geometry_ok([1, 0, 0], [0, 0, 1], d, SHAKE_CFG)


def test_the_whole_roll_fits_checklist_d2():
    """D2 (`paths.RollPathPlanner.accepts`): span <= pi - 0.05 rad, 177.1 deg."""
    for margin in (SHAKE_LEAN_MARGIN_DEG, SHAKE_LEAN_MARGIN_EXECUTED_DEG):
        d = rest_angles([1, 0, 0], [0, 0, 1], 10. - margin)[0]
        assert math.degrees(abs(d)) <= math.degrees(math.pi - 0.05) - 3.


@pytest.mark.parametrize("incline_deg,ok", [(3., True), (4., False), (12., False)])
def test_an_inclined_wrist_axis_cannot_come_within_twice_its_incline_of_upside_down(incline_deg, ok):
    """The upright shaker's axis sweeps a cone about the wrist axis; with that axis `i` below
    horizontal the cone passes 2i from straight down, and the rest leans 10 - 3 deg."""
    tilt = math.radians(incline_deg)
    axis = [math.cos(tilt), 0, -math.sin(tilt)]
    deltas = rest_angles(axis, [0, 0, 1], 10. - SHAKE_LEAN_MARGIN_DEG)
    assert bool(deltas) is ok
    if ok:
        assert shake_geometry_ok(axis, [0, 0, 1], deltas[0], SHAKE_CFG)


def stub(line_status="Success", grasped=True, shake_count=lambda wrist: 0):
    """A held condiment at the wrist pivot over the bowl, wrist axis along world x (as the test_season_pour_paths group)."""
    q = np.zeros(15)
    names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint',
             'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
    indices = [5, 7, 8, 9, 10, 11, 12]
    joints = {n: SimpleNamespace(active_index=[i]) for n, i in zip(names, indices)}
    joints['torso_lift_joint'] = SimpleNamespace(active_index=[3])
    robot = SimpleNamespace(
        active_joints_map=joints, get_qpos=lambda: torch.tensor(q[None]),
        links_map={'wrist_roll_link': SimpleNamespace(pose=[SimpleNamespace(sp=sapien.Pose([0, 0, .2]))])})
    target = SimpleNamespace(pose=[SimpleNamespace(sp=sapien.Pose([0, 0, .2]))])
    task = SimpleNamespace(
        agent=SimpleNamespace(robot=robot, is_grasping=lambda _: torch.tensor([grasped]),
                              controller=SimpleNamespace(controllers={'arm': SimpleNamespace(
                                  config=SimpleNamespace(joint_names=names))})),
        cfg=SHAKE_CFG, bowl=SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0., 0., 0.]]))), control_timestep=.05)
    calls, counts = [], {"n": 0}

    def line(goal, current, **kwargs):
        np.testing.assert_array_equal(np.delete(goal, 12), np.delete(current, 12))
        calls.append(('plan', math.degrees(goal[12] - current[12])))
        return {'status': line_status, 'goal': goal[12]}

    def info():
        return {'shake_count': np.array([counts["n"]]), 'shake_armed': np.array([True]), 'shake_swung': np.array([True]),
                'upright': np.array([True]), 'grasp_target': np.array([True]),
                'success': np.array([True]), 'pour_hold': np.array([SHAKE_CFG.hold_steps + 2])}

    planner = SimpleNamespace(planner=SimpleNamespace(update_from_simulation=lambda: None, plan_qpos_line=line),
                              fixed_action_targets={9: .42}, truncated=False)

    def follow(path, *, refine):
        calls.append(('move', math.degrees(path['goal'])))
        assert planner.fixed_action_targets[7] == -1. and 6 not in planner.fixed_action_targets
        counts["n"] = shake_count(path['goal'])
        return (None, 0., False, False, info())

    planner.follow_forward_path_w_refinement = follow
    planner.idle_steps = lambda t=1: (None, 0., False, False, info())
    return planner, task, target, calls


def test_every_leg_is_planned_before_the_first_moves_and_each_shake_swings_back_toward_upright():
    p, task, target, calls = stub()
    pour_and_shake(SimpleNamespace(), p, task, target, shakes=3)
    planned = [c for c in calls if c[0] == 'plan']
    assert len(planned) == 4 and calls[:4] == planned                 # in, out, back, home
    rest = 180. - (10. - SHAKE_LEAN_MARGIN_EXECUTED_DEG)
    moves = [c[1] for c in calls if c[0] == 'move']
    assert moves == pytest.approx([rest, rest - 30, rest, rest - 30, rest, rest - 30, rest, 0.], abs=1e-9)
    assert p.fixed_action_targets == {9: .42}


def test_a_leg_waits_only_until_the_counter_has_seen_it():
    """No refinement wait (D6): once the counter has seen a leg (swung on an out leg, counted on a
    back leg, upright on the way home), the next leg starts; only the final hold idles."""
    outs = []

    def count(goal):                                   # a shake counts when it comes back
        if 100. < math.degrees(goal) < 160.:
            outs.append(goal)
            return len(outs) - 1
        return len(outs)

    p, task, target, calls = stub(shake_count=count)
    idles, idle = [], p.idle_steps
    p.idle_steps = lambda t=1: (idles.append(dict(p.fixed_action_targets)), idle(t))[1]
    pour_and_shake(SimpleNamespace(), p, task, target, shakes=3)
    assert len(outs) == 3 and len(idles) == 1


def test_a_refused_leg_fails_before_any_motion():
    p, task, target, calls = stub(line_status="collision")
    assert pour_and_shake(SimpleNamespace(), p, task, target, shakes=2) == -1
    assert calls and all(c[0] == 'plan' for c in calls)
    assert p.fixed_action_targets == {9: .42}


def test_no_grasp_no_plan():
    p, task, target, calls = stub(grasped=False)
    assert pour_and_shake(SimpleNamespace(), p, task, target, shakes=1) == -1
    assert calls == []


# ---- the scene (needs ManiSkill and the RoboCasa assets) ---------------------------------------------------


def test_a_shifted_hover_puts_the_rest_where_the_hover_was():
    """Roll axis along x through the wrist; the condiment 5 cm beside it along y. Turning it nearly
    upside down carries it ~10 cm to the other side, so the hover moves 10 cm the other way."""
    pytest.importorskip("mplib")
    from planners.season_dish_planner import rest_shifts
    fk = SimpleNamespace(joints={'gripper_link': (None, 'gripper_axis', 'fixed', np.eye(4), None)})
    task = SimpleNamespace(cfg=SHAKE_CFG)
    desired = np.eye(4); desired[:3, 3] = [0., .05, 0.]
    shifts = rest_shifts(fk, task, desired, np.linalg.inv(desired))      # the wrist frame at the origin
    rest = math.radians(180. - (10. - SHAKE_LEAN_MARGIN_DEG))
    assert len(shifts) == 2
    for shift in shifts:
        assert shift == pytest.approx((0., .05 * (1 - math.cos(rest))), abs=1e-4)
