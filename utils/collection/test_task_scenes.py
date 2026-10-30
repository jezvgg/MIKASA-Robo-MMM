"""Consolidated, behavior-preserving collection regression tests."""


# Source group: test_depth_recall.py
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
    from my_scenes.depth_recall_v1 import (
        INSTRUCTIONS, INSTRUCTIONS_V1, PAD_COLOR_NAMES, PAD_MAP, PAD_MAP_V1, PROP_COLORS)
    spoken = [getattr(spec, "spoken", "") or spec.label for spec in PROPS]
    pairs = [f"{name} on the {color} pad" for name, color in zip(spoken, PAD_COLOR_NAMES)]
    assert PAD_MAP == ", ".join(pairs)
    assert all(PAD_MAP in text for text in INSTRUCTIONS)
    # The text the dataset was recorded with names the pads without the shakers' colours.
    assert PAD_MAP_V1 == ", ".join(
        f"{spec.label} on the {color} pad" for spec, color in zip(PROPS, PAD_COLOR_NAMES))
    assert all(PAD_MAP_V1 in text for text in INSTRUCTIONS_V1)
    # Profile 6: a prop is recoloured only into its own pad's colour (the mustard).
    for spec, color in zip(PROPS, PROP_COLORS):
        assert all(tuple(rgba) == tuple(color) for _, rgba in spec.recolor), spec.label
    assert [spec.label for spec in PROPS if spec.recolor] == ["mustard"]


# Source group: test_same_drawer_rules.py
from types import SimpleNamespace

import pytest
import torch

from my_scenes.same_drawer import SameDrawerConfig, SameDrawerTask


class Drawer:
    def __init__(self):
        self.q = torch.zeros(1, 1)
        self.v = torch.zeros(1, 1)

    def get_qpos(self):
        return self.q

    def get_qvel(self):
        return self.v

    def set_qpos(self, q):
        self.q = q

    def set_qvel(self, v):
        self.v = v


def task_at(target):
    task = SimpleNamespace(
        cfg=SameDrawerConfig(),
        elapsed_steps=torch.tensor([0]),
        _last_eval_step=torch.tensor([-1], dtype=torch.int32),
        target_drawer=torch.tensor([target]),
        _drawer_arts=[Drawer() for _ in range(4)],
        held_count=torch.tensor([0]),
    )
    for name in ("closed_done", "apple_done", "wrong_drawer_touched", "sequence_violated",
                 "apple_was_grasped", "apple_placement_started", "apple_retention_violated"):
        setattr(task, name, torch.tensor([False]))
    task.apple = SimpleNamespace(
        pose=SimpleNamespace(p=torch.tensor([[0.3, 0.0, 0.03]])),
        is_static=lambda **kwargs: torch.tensor([True]),
    )
    task.plate = SimpleNamespace(pose=SimpleNamespace(p=torch.zeros(1, 3)))
    task.grasped = False
    task.agent = SimpleNamespace(is_grasping=lambda obj: torch.tensor([task.grasped]))
    task.drawer_open_amounts = lambda: -torch.cat(
        [art.get_qpos() for art in task._drawer_arts], dim=1
    )
    task._drawer_arts[target].q[:] = -0.15

    def tick(*, opened=None, apple_on_plate=None, advance=True):
        if opened is not None:
            for index, amount in opened.items():
                task._drawer_arts[index].q[:] = -amount
        if apple_on_plate is not None:
            task.apple.pose.p[0, 0] = 0.0 if apple_on_plate else 0.3
        if advance:
            task.elapsed_steps += 1
        return SameDrawerTask.evaluate(task)

    tick()
    return task, tick


def finish(task, tick):
    target = int(task.target_drawer.item())
    for _ in range(task.cfg.hold_steps):
        info = tick(opened={target: 0.15})
    return info


@pytest.mark.parametrize("target", range(4))
def test_all_four_require_close_then_released_apple_then_reopen(target):
    task, tick = task_at(target)
    assert not task.closed_done.item()
    tick(opened={target: 0.0})
    task.grasped = True
    tick(apple_on_plate=True)
    assert not task.apple_done.item()
    task.grasped = False
    tick()
    assert task.apple_done.item()
    assert finish(task, tick)["success"].item()


@pytest.mark.parametrize("violation", ["early_apple", "early_reopen", "wrong_drawer", "lost_apple"])
def test_later_correct_moves_do_not_credit_invalid_sequence(violation):
    task, tick = task_at(2)
    if violation == "early_apple":
        tick(apple_on_plate=True)
    tick(opened={2: 0.0})
    if violation == "early_reopen":
        tick(opened={2: 0.06})
        tick(opened={2: 0.0})
    if violation == "wrong_drawer":
        tick(opened={0: 0.06})
        tick(opened={0: 0.0})
    tick(apple_on_plate=violation != "lost_apple")
    assert not finish(task, tick)["success"].item()


def test_repeated_evaluate_cannot_advance_terminal_hold():
    task, tick = task_at(0)
    tick(opened={0: 0.0})
    tick(apple_on_plate=True)
    tick(opened={0: 0.15})
    held = task.held_count.clone()
    for _ in range(20):
        assert not tick(advance=False)["success"].item()
    assert torch.equal(task.held_count, held)
    assert finish(task, tick)["success"].item()


@pytest.mark.parametrize("violation", ["horizontal", "vertical", "regrasp"])
@pytest.mark.parametrize("target", range(4))
def test_apple_must_remain_released_on_plate_until_episode_end(target, violation):
    task, tick = task_at(target)
    tick(opened={target: 0.0})
    task.grasped = True
    tick(apple_on_plate=True)
    task.grasped = False
    tick()
    assert task.apple_done.item()
    if violation == "horizontal":
        task.apple.pose.p[0, 0] = 0.3
    elif violation == "vertical":
        task.apple.pose.p[0, 2] = 0.3
    else:
        task.grasped = True
    info = tick()
    assert info["failed"].item()
    assert info["apple_retention_violated"].item()
    # Replacing the apple cannot repair the demonstration.
    task.apple.pose.p[:] = torch.tensor([[0.0, 0.0, 0.03]])
    task.grasped = False
    assert not finish(task, tick)["success"].item()


def test_knockoff_before_first_settled_frame_is_not_forgotten():
    task, tick = task_at(0)
    tick(opened={0: 0.0})
    task.grasped = True
    tick(apple_on_plate=True)
    task.apple.is_static = lambda **kwargs: torch.tensor([False])
    task.grasped = False
    tick()
    assert task.apple_placement_started.item()
    assert not task.apple_done.item()
    tick(apple_on_plate=False)
    task.apple.is_static = lambda **kwargs: torch.tensor([True])
    tick(apple_on_plate=True)
    assert not finish(task, tick)["success"].item()


def test_terminal_hold_requires_apple_to_be_still_now():
    task, tick = task_at(0)
    tick(opened={0: 0.0})
    tick(apple_on_plate=True)
    assert task.apple_done.item()
    task.apple.is_static = lambda **kwargs: torch.tensor([False])
    assert not finish(task, tick)["success"].item()
    assert task.held_count.item() == 0


# Source group: test_cabinet_search_rules.py
import math

import pytest
import torch

from my_scenes.cabinet_search import (
    CabinetSearchConfig, SearchLatches, at_home_predicate,
    compartment_layout, step_search_latches,
)


def fixture(target):
    cfg = CabinetSearchConfig()
    layout = compartment_layout(cfg.compartments)
    state = SearchLatches.zeros(1, 4, 0)
    state.cube_cab[:] = target
    step = 0

    def tick(opened=None, home=False, moved=0.):
        nonlocal step
        step += 1
        theta = torch.zeros(1, 4)
        if opened is not None:
            theta[0, opened] = 1.3
        info, _ = step_search_latches(
            state, step=torch.tensor([step]), theta=theta,
            hinge_still=torch.ones(1, 4, dtype=torch.bool),
            tcp_far=torch.ones(1, 4, dtype=torch.bool),
            at_home=torch.tensor([home]), foreign_hit=torch.tensor([False]),
            layout=layout, cfg=cfg, moved_m=torch.tensor([moved]),
        )
        return info

    return tick, state


@pytest.mark.parametrize("target", range(4))
@pytest.mark.parametrize("moved", [.02, .03])
def test_nudge_ends_at_can_without_final_return(target, moved):
    tick, state = fixture(target)
    tick(home=True)
    info = tick(target, moved=moved)
    assert info["found"].item() and info["success"].item()
    assert not info["at_home"].item()


@pytest.mark.parametrize("home_between", [False, True])
def test_completed_inspection_cannot_be_repeated(home_between):
    tick, _ = fixture(3)
    tick(home=True)
    tick(0)
    for _ in range(10):
        tick()
    if home_between:
        tick(home=True)
    info = tick(0)
    assert info["reopened"].item() and info["fail"].item()


def test_next_inspection_requires_home_even_after_empty_cabinet_closed():
    tick, _ = fixture(3)
    tick(home=True)
    tick(0)
    for _ in range(10):
        tick()
    info = tick(1)
    assert info["skipped_home"].item() and info["fail"].item()


def test_random_home_uses_recorded_marker_not_config_center():
    cfg = CabinetSearchConfig()
    home = torch.tensor([[cfg.home_xy[0] + .25, cfg.home_xy[1]]])
    yaw = torch.tensor([math.radians(cfg.home_yaw_deg)])
    speed = torch.zeros(1)
    assert at_home_predicate(home, yaw, speed, cfg, home).item()
    assert not at_home_predicate(home, yaw, speed, cfg).item()


@pytest.mark.parametrize("target", range(4))
@pytest.mark.parametrize("moved", [0., .019])
def test_reveal_or_insufficient_push_does_not_finish(target, moved):
    tick, _ = fixture(target)
    tick(home=True)
    assert not tick(target, moved=moved)["success"].item()


@pytest.mark.parametrize("target", range(4))
def test_push_without_initial_home_fails(target):
    tick, _ = fixture(target)
    info = tick(target, moved=.03)
    assert info["fail"].item() and not info["success"].item()


# Source group: test_drawer_balance.py
import pytest
from utils.collection.drawer_balance import next_seeds, interleave


def manifest():
    strata={i:list(range(100+i,116,4)) for i in range(4)}
    return dict(seeds=interleave(strata),seed_to_drawer={str(s):i for i,ss in strata.items() for s in ss})


def test_failures_count_as_attempted_but_do_not_fill_quota():
    m=manifest()
    assert next_seeds(m,{100,101,102,103},{101,102,103},count=4,quota=1)==[104]


def test_resume_keeps_bank_order_and_never_overshoots_quota():
    m=manifest()
    assert next_seeds(m,set(),set(),count=8,quota=1)==[100,101,102,103]
    assert next_seeds(m,{100,101},{100},count=4,quota=1)==[102,103,105]


def test_acceptance_requires_a_recorded_attempt():
    with pytest.raises(ValueError):next_seeds(manifest(),set(),{100})


# Source group: test_physical_closure.py
import torch
from my_scenes.cabinet_search import CabinetSearchConfig, SearchLatches, compartment_layout, step_search_latches

def test_closure_requires_ten_released_steps_and_reopening_resets_dwell():
    cfg=CabinetSearchConfig(); layout=compartment_layout(cfg.compartments)
    z=SearchLatches.zeros(1,4,0);z.cube_cab[:]=3
    step=0
    def tick(angle, far=True, home=False):
        nonlocal step
        step+=1
        theta=torch.tensor([[angle, .004, 0., 0.]])
        original=theta.clone()
        info,snap=step_search_latches(z,step=torch.tensor([step]),theta=theta,
            hinge_still=torch.ones((1,4),dtype=torch.bool),
            tcp_far=torch.full((1,4),far,dtype=torch.bool),at_home=torch.tensor([home]),
            foreign_hit=torch.tensor([False]),layout=layout,cfg=cfg,moved_m=torch.tensor([0.]))
        assert torch.equal(theta,original)
        assert not snap.any() and not info['snapped_now'].any()
        assert info['door_theta_max'][0,1]==theta[0,1]
        return info
    tick(0,home=True);tick(1.3)
    for _ in range(12):tick(.004,far=False)
    assert z.is_open[0,0]
    for _ in range(9):tick(.004)
    assert z.is_open[0,0]
    tick(.02)
    assert z.closed_count[0,0]==0
    for _ in range(9):tick(.004)
    assert z.is_open[0,0]
    tick(.004)
    assert not z.is_open[0,0]
    assert tick(.10)['reopened'].item()


def test_evaluate_does_not_modify_real_hinges_or_mask_observations():
    """Counterexample to the former small-drift eraser, using actual articulations."""
    import gymnasium as gym
    import my_scenes
    from utils.collection.profile import load_profile
    profile=load_profile('cabinet_search5')
    env=gym.make(profile['env_id'],obs_mode='state',**profile['env_kwargs'])
    try:
        env.reset(seed=7800000);task=env.unwrapped
        for h,(art,j) in enumerate(task._hinges[0]):
            q=art.get_qpos().clone();v=art.get_qvel().clone()
            q.reshape(-1)[j]=.007*task.layout.open_dir[h]
            v.reshape(-1)[j]=.003
            art.set_qpos(q);art.set_qvel(v)
        task._elapsed_steps[:]=2
        before_q,before_v=task._read_hinges()
        for _ in range(3):
            info=task.evaluate();after_q,after_v=task._read_hinges()
            torch.testing.assert_close(after_q,before_q,rtol=0,atol=0)
            torch.testing.assert_close(after_v,before_v,rtol=0,atol=0)
            torch.testing.assert_close(info['door_qpos'],before_q,rtol=0,atol=0)
            torch.testing.assert_close(task._get_obs_extra(info)['door_qpos'],before_q,rtol=0,atol=0)
            assert not info['snapped_now'].any()
    finally:env.close()


@pytest.mark.parametrize('failed_stage', ['pre-touch', 'push'])
def test_nudge_arm_move_refusal_returns_failure_without_unpacking(monkeypatch, failed_stage):
    import contextlib
    from types import SimpleNamespace

    import numpy as np
    from planners import cabinet_search_planner as cabinet
    from planners.oracle import oracle_common as common

    step = (None, 0.0, False, False, {'nudged': False, 'success': False, 'moved_m': 0.0})
    calls = []
    task = SimpleNamespace(
        cfg=SimpleNamespace(cube_half=.04),
        motion_parameters={'grasp_noise_m': 0.0},
        revealed_target=lambda: np.array([[1.0, -0.8, .9]]),
        agent=SimpleNamespace(build_grasp_pose=lambda *args: object(),
                              keyframes={'rest': SimpleNamespace(qpos=np.zeros(15))}),
        _spawn_centre_x=None,
    )
    # The dock drive must land, or the ported nudge bails at "MISSED: no drive to the
    # touch dock; the cube was not nudged" before any arm_move is planned: this test is
    # about the arm_move refusal, so the base part is granted.
    planner = SimpleNamespace(
        planner=SimpleNamespace(update_from_simulation=lambda: None),
        drive_base=lambda **kwargs: step,
        drive_straight=lambda *args, **kwargs: -1,
        close_gripper=lambda **kwargs: step,
        cabinet_waypoint_noise=None,
    )

    monkeypatch.setattr(cabinet, 'door_mode', lambda _task: 'arc')
    monkeypatch.setattr(cabinet, 'NUDGE_TORSO', .1)
    monkeypatch.setattr(cabinet, 'say', lambda *args, **kwargs: None)
    monkeypatch.setattr(cabinet, 'plan_joints', lambda *args, **kwargs: -1)
    monkeypatch.setattr(cabinet, 'drive_posture_targets', lambda _task: {})
    monkeypatch.setattr(common, 'touchable', lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(common, 'say', lambda *args, **kwargs: None)

    def arm_move(*args, stage='', **kwargs):
        calls.append(stage)
        if failed_stage == 'pre-touch':
            return -1
        return -1 if stage.startswith('push') else step

    monkeypatch.setattr(common, 'arm_move', arm_move)
    result = cabinet.nudge_the_cube(SimpleNamespace(), planner, task, -1)
    # Every refused arm_move is a bare -1 the ported nudge reads as a value, never as a
    # step tuple: the loop runs NUDGE_TRIES push attempts either way and hands back a step,
    # so nothing unpacks a refusal.
    assert isinstance(result, tuple) and len(result) == 5
    # A refused pre-touch gives up on the nudge (no push is planned at all); a refused push
    # is retried, so the push stage runs NUDGE_TRIES times.
    pushes = len([c for c in calls if c.startswith('push')])
    assert pushes == (0 if failed_stage == 'pre-touch' else cabinet.NUDGE_TRIES)
