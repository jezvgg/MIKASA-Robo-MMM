"""Behavioral checks for staging on the props' own pads, remembered assignment and final completion."""
from types import SimpleNamespace

import pytest
import torch

import dataclasses

from my_scenes.depth_recall_v1 import PROPS, DepthRecallV1Config, DepthRecallV1Task


def world():
    cfg = DepthRecallV1Config()
    row = torch.tensor([[[2.5, y, cfg.shelf_top_z] for y in cfg.row_slots_y]])
    slots = torch.tensor([[[2.5+x, cfg.place_across, .92] for x in cfg.slot_x_offsets]])
    # Pads shuffled against the props, so "own pad" and "slot i" differ for every prop.
    task = SimpleNamespace(cfg=cfg, num_envs=1, original_slots=torch.tensor([[0, 1, 2, 3]]),
        pad_slots=torch.tensor([[2, 0, 3, 1]]), wrong_pad=torch.tensor([False]),
        _row_points=row, _slot_points=slots, _counter_top_z=torch.tensor([.92]),
        elapsed_steps=torch.tensor([0]), last_eval_step=torch.tensor([-1]),
        arrangement_hold=torch.tensor([0], dtype=torch.int32),
        # Origin over mesh bottom, per prop: the scene reads it off the meshes; these
        # objects' origins are their centres, so it is half the recorded height.
        _rest_lift=torch.tensor([p.height / 2 for p in cfg.props]))
    for key in ('left_slot', 'restored', 'was_lifted', 'staged'):
        setattr(task, key, torch.zeros((1, 4), dtype=torch.bool))
    for key in ('wrong_assign', 'target_returned', 'two_in_slot', 'succeeded'):
        setattr(task, key, torch.tensor([False]))
    task.pos = row.clone()
    task.pos[:, :, 2] += task._rest_lift
    task.tilt = torch.zeros((1, 4))
    task.grasped = torch.zeros((1, 4), dtype=torch.bool)
    task.settled = torch.ones((1, 4), dtype=torch.bool)
    task._prop_state = lambda: (task.pos, task.tilt, task.grasped, task.settled)
    return task


def evaluate(task, advance=True):
    if advance:
        task.elapsed_steps += 1
    return DepthRecallV1Task.evaluate(task)


def own_pads(task):
    """Where each prop's own pad is: (1, 4, 3), prop order."""
    return task._slot_points[:, task.pad_slots[0]].clone()


def stage_all(task):
    task.was_lifted[:] = True
    task.pos = own_pads(task)
    task.pos[:, :, 2] += task._rest_lift
    result = evaluate(task)
    assert result['staged'].all()


def restore_blockers(task):
    task.pos[:, :3] = task._row_points[:, :3]
    task.pos[:, :3, 2] += task._rest_lift[:3]


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
    task.pos[:, 3] = own_pads(task)[:, 3]
    task.pos[:, 3, 2] += task._rest_lift[3]
    task.was_lifted[:] = True
    evaluate(task)
    restore_blockers(task)
    result = hold(task)
    assert result['all_left'].item() and result['target_on_its_pad'].item()
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
    # Swapped in the row, each standing at its OWN height (the props differ in height).
    task.pos[:, [0, 1], :2] = task.pos[:, [1, 0], :2].clone()
    assert evaluate(task)['wrong_assign'].item()
    restore_blockers(task)
    assert not hold(task)['success'].item()


def test_returning_deepest_prop_is_not_the_requested_chore():
    task = world()
    stage_all(task)
    task.pos[:, 3] = task._row_points[:, 3]
    task.pos[:, 3, 2] += task._rest_lift[3]
    assert evaluate(task)['target_returned'].item()


def test_two_released_props_in_one_row_slot_fail():
    task = world()
    stage_all(task)
    restore_blockers(task)
    task.pos[:, 1, :2] = task.pos[:, 0, :2]          # prop 1 in prop 0's slot, standing
    assert evaluate(task)['two_in_slot'].item()
    assert not hold(task)['success'].item()


@pytest.mark.parametrize('invalid', ['held', 'moving', 'tilted', 'wrong_height'])
def test_counter_staging_requires_released_settled_upright_props(invalid):
    task = world()
    task.was_lifted[:] = True
    task.pos = own_pads(task)
    task.pos[:, :, 2] += task._rest_lift
    if invalid == 'held': task.grasped[:] = True
    if invalid == 'moving': task.settled[:] = False
    if invalid == 'tilted': task.tilt[:] = .6
    if invalid == 'wrong_height': task.pos[:, :, 2] += .2
    assert not evaluate(task)['staged'].any()


def test_a_prop_released_on_another_colours_pad_voids_the_episode():
    task = world()
    task.was_lifted[:] = True
    task.pos = task._slot_points.clone()            # slot k, not each prop's own pad
    task.pos[:, :, 2] += task._rest_lift
    result = evaluate(task)
    assert result['wrong_pad'].item()
    assert not result['staged'].any() and not result['on_own_pad'].any()
    stage_all(task)                                  # moved onto the right pads afterwards
    restore_blockers(task)
    result = hold(task)
    assert result['wrong_pad'].item() and not result['success'].item()


def test_target_on_another_pad_is_not_success():
    task = world()
    stage_all(task)
    restore_blockers(task)
    task.wrong_pad[:] = False
    task.pos[:, 3] = task._slot_points[:, 0]         # prop 3's pad is slot 1; slot 0 is prop 1's
    task.pos[:, 3, 2] += task._rest_lift[3]
    result = hold(task)
    assert result['target_on_a_slot'].item() and not result['target_on_its_pad'].item()
    assert not result['success'].item()


def test_each_prop_is_judged_standing_at_its_own_height():
    """The shakers' origins stand 4.7 cm over a surface, the bottles' 7.5-8.3 cm: one
    height for all would call either the short ones or the tall ones not standing."""
    task = world()
    stage_all(task)
    restore_blockers(task)
    assert hold(task)['success'].item()
    lifts = task._rest_lift
    assert float(lifts.max() - lifts.min()) > 2 * task.cfg.upright_tol
    task.pos[:, 3, 2] += float(lifts.min() - lifts[3])   # the mustard at a shaker's height
    result = evaluate(task)
    assert not result['target_on_its_pad'].item() and not result['success'].item()


def test_a_tipped_prop_is_not_standing_at_any_height():
    """Lying on its side a prop's origin is about its footprint half-width up: the
    tolerance must not admit that for the shortest prop (validate's rule)."""
    task = world()
    stage_all(task)
    restore_blockers(task)
    task.pos[:, 1, 2] = task._row_points[:, 1, 2] + task.cfg.props[1].half_xy   # lying
    task.tilt[:] = 0.0                                                       # even if not tilted
    assert not hold(task)['success'].item()


def test_props_are_spices_and_sauces_that_fit_the_row():
    cfg = DepthRecallV1Config()
    cfg.validate()
    assert [p.label for p in cfg.props] == ['ketchup', 'pepper', 'salt', 'mustard']
    assert len({(p.category, p.instance) for p in cfg.props}) == len(cfg.props)
    pitch = min(b - a for a, b in zip(cfg.row_slots_y, cfg.row_slots_y[1:]))
    assert pitch - cfg.max_half_xy > cfg.finger_reach + 0.005


@pytest.mark.parametrize('field,value', [('half_xy', 0.029), ('height', 0.09)])
def test_validate_refuses_a_prop_that_does_not_fit(field, value):
    bad = dataclasses.replace(PROPS[1], **{field: value})
    cfg = DepthRecallV1Config(props=(PROPS[0], bad) + PROPS[2:])
    with pytest.raises(AssertionError):
        cfg.validate()


def test_pads_are_one_per_prop_and_far_enough_apart():
    cfg = DepthRecallV1Config()
    offs = list(cfg.slot_x_offsets)
    assert min(b - a for a, b in zip(offs, offs[1:])) >= 2 * cfg.slot_radius
    assert cfg.n_props <= len(offs)


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


def test_scene_contact_offset_is_the_tasks():
    """The scene runs with a 5 mm PhysX contact offset (ManiSkill's default is 2 cm) and
    otherwise BaseEnv's own sim config."""
    from my_scenes.depth_recall_v1 import SCENE_CONTACT_OFFSET

    cfg = DepthRecallV1Task._default_sim_config.fget(None)
    assert cfg.scene_config.contact_offset == SCENE_CONTACT_OFFSET == 0.005
    assert (cfg.sim_freq, cfg.control_freq) == (100, 20)


def test_the_instruction_names_each_props_pad():
    from my_scenes.depth_recall_v1 import INSTRUCTIONS, PAD_COLOR_NAMES, PAD_MAP, PROP_COLORS
    pairs = [f"{spec.label} on the {color} pad" for spec, color in zip(PROPS, PAD_COLOR_NAMES)]
    assert PAD_MAP == ", ".join(pairs)
    assert all(PAD_MAP in text for text in INSTRUCTIONS)
    # Profile 6: a prop is recoloured only into its own pad's colour (the mustard).
    for spec, color in zip(PROPS, PROP_COLORS):
        assert all(tuple(rgba) == tuple(color) for _, rgba in spec.recolor), spec.label
    assert [spec.label for spec in PROPS if spec.recolor] == ["mustard"]
