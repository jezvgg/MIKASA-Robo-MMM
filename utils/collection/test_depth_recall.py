"""Behavioral checks for staging, remembered assignment and final completion."""
from types import SimpleNamespace

import pytest
import torch

from my_scenes.depth_recall_v1 import DepthRecallV1Config, DepthRecallV1Task


def world():
    cfg = DepthRecallV1Config()
    row = torch.tensor([[[2.5, y, cfg.shelf_top_z] for y in cfg.row_slots_y]])
    slots = torch.tensor([[[2.5+x, cfg.place_across, .92] for x in cfg.slot_x_offsets]])
    task = SimpleNamespace(cfg=cfg, num_envs=1, original_slots=torch.tensor([[0, 1, 2, 3]]),
        _row_points=row, _slot_points=slots, _counter_top_z=torch.tensor([.92]),
        elapsed_steps=torch.tensor([0]), last_eval_step=torch.tensor([-1]),
        arrangement_hold=torch.tensor([0], dtype=torch.int32))
    for key in ('left_slot', 'restored', 'was_lifted', 'staged'):
        setattr(task, key, torch.zeros((1, 4), dtype=torch.bool))
    for key in ('wrong_assign', 'target_returned', 'two_in_slot', 'succeeded'):
        setattr(task, key, torch.tensor([False]))
    task.pos = row.clone()
    task.pos[:, :, 2] += cfg.prop_half_h
    task.tilt = torch.zeros((1, 4))
    task.grasped = torch.zeros((1, 4), dtype=torch.bool)
    task.settled = torch.ones((1, 4), dtype=torch.bool)
    task._prop_state = lambda: (task.pos, task.tilt, task.grasped, task.settled)
    return task


def evaluate(task, advance=True):
    if advance:
        task.elapsed_steps += 1
    return DepthRecallV1Task.evaluate(task)


def stage_all(task):
    task.was_lifted[:] = True
    task.pos = task._slot_points.clone()
    task.pos[:, :, 2] += task.cfg.prop_half_h
    result = evaluate(task)
    assert result['staged'].all()


def restore_blockers(task):
    task.pos[:, :3] = task._row_points[:, :3]
    task.pos[:, :3, 2] += task.cfg.prop_half_h


def hold(task):
    result = None
    for _ in range(task.cfg.hold_steps):
        result = evaluate(task)
    return result


def test_reset_arrangement_is_not_a_completed_episode():
    task = world()
    result = hold(task)
    assert not result['success'].item()
    assert not result['restored'].any()
    assert not result['wrong_assign'].item()
    assert not result['target_returned'].item()


def test_target_alone_and_shelf_nudges_do_not_count_as_clearing():
    task = world()
    task.pos[:, :3, 0] += .08
    task.pos[:, 3] = task._slot_points[:, 3]
    task.pos[:, 3, 2] += task.cfg.prop_half_h
    task.was_lifted[:] = True
    evaluate(task)
    restore_blockers(task)
    result = hold(task)
    assert result['all_left'].item() and result['target_on_a_slot'].item()
    assert not result['all_restored'].item() and not result['success'].item()
    assert not result['staged'][0, :3].any()


def test_correct_counter_staging_and_restore_succeeds_after_hold():
    task = world()
    stage_all(task)
    restore_blockers(task)
    for _ in range(task.cfg.hold_steps-1):
        assert not evaluate(task)['success'].item()
    assert evaluate(task)['success'].item()


def test_repeated_evaluation_cannot_replace_physical_hold_steps():
    task = world()
    stage_all(task)
    restore_blockers(task)
    evaluate(task)
    for _ in range(100):
        result = evaluate(task, advance=False)
    assert result['arrangement_hold'].item() == 1
    assert not result['success'].item()


def test_wrong_remembered_assignment_is_an_irreversible_error():
    task = world()
    stage_all(task)
    restore_blockers(task)
    task.pos[:, [0, 1]] = task.pos[:, [1, 0]].clone()
    assert evaluate(task)['wrong_assign'].item()
    restore_blockers(task)
    assert not hold(task)['success'].item()


def test_returning_deepest_prop_is_not_the_requested_chore():
    task = world()
    stage_all(task)
    task.pos[:, 3] = task._row_points[:, 3]
    task.pos[:, 3, 2] += task.cfg.prop_half_h
    assert evaluate(task)['target_returned'].item()


def test_two_released_props_in_one_row_slot_fail():
    task = world()
    stage_all(task)
    restore_blockers(task)
    task.pos[:, 1] = task.pos[:, 0]
    assert evaluate(task)['two_in_slot'].item()
    assert not hold(task)['success'].item()


@pytest.mark.parametrize('invalid', ['held', 'moving', 'tilted', 'wrong_height'])
def test_counter_staging_requires_released_settled_upright_props(invalid):
    task = world()
    task.was_lifted[:] = True
    task.pos = task._slot_points.clone()
    task.pos[:, :, 2] += task.cfg.prop_half_h
    if invalid == 'held': task.grasped[:] = True
    if invalid == 'moving': task.settled[:] = False
    if invalid == 'tilted': task.tilt[:] = .6
    if invalid == 'wrong_height': task.pos[:, :, 2] += .2
    assert not evaluate(task)['staged'].any()


def test_knocking_a_prop_over_after_success_invalidates_final_success():
    task = world()
    stage_all(task)
    restore_blockers(task)
    assert hold(task)['success'].item()
    task.tilt[0, 0] = 1.0
    result = evaluate(task)
    assert not result['success'].item()
    assert result['success_once'].item()
    assert result['arrangement_hold'].item() == 0


@pytest.mark.parametrize('hold_pose', [False, True])
def test_loaded_drive_can_hold_targets_without_recentring_head(hold_pose):
    import numpy as np
    from robots.fetch.extand import FetchMotionPlanningSapienSolver
    position = np.zeros(3)
    arm = np.arange(7, dtype=float) / 10
    body = np.array([.4, -.3, .2])
    recorded = []
    def step(action):
        recorded.append(action.copy())
        position[0] += .01
        arm[1] -= .005  # Measured sag must not become the next absolute target.
        return {}, 0., False, False, {}
    solver = SimpleNamespace(truncated=False,
        env_agent=SimpleNamespace(controller=SimpleNamespace(controllers={
            'base':SimpleNamespace(config=SimpleNamespace(normalize_action=True,upper=[1.,3.14]))}),
            base_link=SimpleNamespace(pose=SimpleNamespace(sp=SimpleNamespace(p=position)))),
        base_env=SimpleNamespace(control_timestep=.05),_guard=SimpleNamespace(last_step=None),
        _hold_targets=lambda:(arm.copy(),body.copy()),
        _compose=lambda a,b,v:np.r_[a,-1.,b,v], _step=step,
        _stopped_by_horizon=lambda name:False,_report=lambda *args,**kwargs:None)
    FetchMotionPlanningSapienSolver.drive_straight(solver,.05,hold_pose=hold_pose)
    actions=np.asarray(recorded)
    assert len(actions)>=5
    np.testing.assert_array_equal(actions[:,8:10],np.tile([.4,-.3],(len(actions),1)))
    if hold_pose:
        np.testing.assert_array_equal(actions[:,:7],np.tile(actions[0,:7],(len(actions),1)))
    else:
        assert np.ptp(actions[:,1])>.01
