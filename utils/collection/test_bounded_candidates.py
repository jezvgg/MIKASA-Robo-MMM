"""Candidate caps and admissibility before expensive contact/continuation work."""
from types import SimpleNamespace
import numpy as np
import pytest
import sapien
import torch
from planners import season_dish_candidates as search


def test_duplicate_endpoints_do_not_consume_continuation_slots():
    q=np.zeros(15)
    values=[q.copy(),q+.001,q+.2,q+.201]
    kept=search.distinct_goals(values,list(range(7,13)))
    assert len(kept)==2
    np.testing.assert_equal(kept[0],values[0])
    np.testing.assert_equal(kept[1],values[2])


@pytest.mark.parametrize('policy,requests', [('legacy',8),('margin',8),('diverse',4)])
def test_many_ik_endpoints_cannot_expand_execution_search(monkeypatch,policy,requests):
    from collections import Counter
    names=['root_x_axis_joint','root_y_axis_joint','root_z_rotation_joint',
           'torso_lift_joint','head_pan_joint','shoulder_pan_joint','head_tilt_joint',
           'shoulder_lift_joint','upperarm_roll_joint','elbow_flex_joint',
           'forearm_roll_joint','wrist_flex_joint','wrist_roll_joint',
           'r_gripper_finger_joint','l_gripper_finger_joint']
    current=np.zeros(15);current[9]=1.;current[11]=1.
    robot=SimpleNamespace(get_qpos=lambda:torch.tensor(current[None]),
        get_qlimits=lambda:torch.tensor(np.tile([-3.14,3.14],(1,15,1))),
        active_joints_map={n:SimpleNamespace(active_index=[i]) for i,n in enumerate(names)})
    calls=[]
    def ik(pose,initial,mask,**kw):
        calls.append(kw['n_init_qpos'])
        goals=[]
        for i in range(20):
            q=current.copy();q[7]=i*.02;q[8]=len(calls)*.15
            goals.append(q)
        return 'Success',goals
    engine=SimpleNamespace(fold_qpos=lambda q:q.copy(),IK=ik,
        _transform_goal_to_wrt_base=lambda q:q, joint_limits=np.tile([-3.14,3.14],(15,1)),
        _root_cols=lambda:[0,1,2],accepts=lambda q:bool(np.all(np.abs(q[[8,10,12]])<np.pi-.05)))
    planner=SimpleNamespace(planner=engine,_roll_indices=[8,10,12],
                            _roll_low=np.zeros(3),_roll_high=np.zeros(3),env=None)
    task=SimpleNamespace(agent=SimpleNamespace(robot=robot,controller=SimpleNamespace(
        controllers={'arm':SimpleNamespace(config=SimpleNamespace(joint_names=[names[i] for i in [5,7,8,9,10,11,12]]))})),
        motion_parameters={'grasp_selection':policy})
    monkeypatch.setattr(search.common,'say',lambda *args,**kwargs:None)
    result=search.candidates(planner,task,sapien.Pose([.7,0,1.]),sapien.Pose([.6,0,1.]),Counter(),n_init=256)
    assert len(calls)==requests and set(calls)=={32}
    assert 0<len(result)<=8
    assert all(engine.accepts(row[0]) for row in result)


def test_dual_hover_keeps_both_start_families_within_original_budget():
    from planners.season_dish_transfer import hover_goals
    calls = []
    current = np.zeros(15)
    def ik(pose, initial, mask, *, n_init_qpos):
        calls.append(n_init_qpos)
        goal = current.copy(); goal[7] = .5 if len(calls) == 1 else -.5
        return 'Success', [goal, goal + .001]
    p = SimpleNamespace(move_group_joint_indices=list(range(3, 13)),
        fold_qpos=lambda q: q.copy(), unfold_qpos=lambda q, **kw: q.copy(),
        _transform_goal_to_wrt_base=lambda q: q, IK=ik,
        joint_limits=np.tile([-3.14, 3.14], (15, 1)))
    preferred = {'approach': {'position': np.ones((1, 10))}}
    goals, records = hover_goals(p, current, sapien.Pose(), n_init=256,
                                 preferred=preferred, mode='dual')
    assert calls == [16, 16]
    assert len(goals) == 2
    assert {q[7] for q in goals} == {-.5, .5}
    assert [r['initial'] for r in records] == ['preferred', 'current']
