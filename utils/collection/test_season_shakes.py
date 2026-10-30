"""SeasonDish with the number card and the shakes (owner 2026-10-27): the counter, the wrist geometry,
the planner's shake sequence on stubs, and the scene's wiring (cue, count, success, fail)."""
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import sapien
import torch

from planners.oracle.wrist_pour import (SHAKE_AMPLITUDE_DEG, SHAKE_LEAN_MARGIN_DEG, SHAKE_LEAN_MARGIN_EXECUTED_DEG,
                                        lean_after, pour_and_shake, rest_angles, shake_geometry_ok, swing_angle)
from utils.mikasa.shake_counter import step_shakes

#: The scene tests read poses only; any renderer will do ("none" crashes SAPIEN 3.0.3's URDF loader on ds_fetch).
RENDER = os.environ.get("SEASON_TEST_RENDER", "sapien_cuda")

CFG = SimpleNamespace(shake_out_deg=20., shake_return_deg=10., pour_axis_body=[0, 0, 1], pour_xy_radius=.1,
                      pour_min_clearance=.05, pour_max_clearance=.3, hold_steps=20)


def run_counter(leans, held=True):
    count, armed, swung = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    over = torch.tensor([held])
    for lean in leans:
        count, armed, swung = step_shakes(count, armed, swung, torch.tensor([float(lean)]), over, 20., 10.)
    return int(count[0]), bool(armed[0])


def test_turning_the_shaker_over_is_not_a_shake():
    assert run_counter([180, 120, 60, 25, 12, 4]) == (0, True)


def test_each_excursion_from_upside_down_that_comes_back_is_one_shake():
    over = [180, 90, 5]
    assert run_counter(over + [30, 2]) == (1, True)
    assert run_counter(over + [30, 2, 30, 2]) == (2, True)
    assert run_counter(over + [30, 2, 30, 2, 30, 2]) == (3, True)


def test_a_swing_counts_only_when_it_comes_back():
    assert run_counter([180, 5, 30, 15, 30, 15, 30]) == (0, True)
    assert run_counter([180, 5, 30, 15, 30, 2]) == (1, True)


def test_turning_it_upright_again_after_the_last_shake_is_not_a_shake():
    assert run_counter([180, 90, 5, 30, 2, 30, 2, 60, 120, 180]) == (2, True)


def test_nothing_counts_off_the_bowl():
    assert run_counter([180, 5, 30, 2, 30, 2], held=False) == (0, False)


def test_horizontal_wrist_rests_short_of_upside_down_and_swings_back_toward_upright():
    rest = 10. - SHAKE_LEAN_MARGIN_DEG
    deltas = rest_angles([1, 0, 0], [0, 0, 1], rest)
    assert [round(math.degrees(d)) for d in deltas[:2]] == [173, -173]
    for d in deltas[:2]:
        swing = swing_angle(d, SHAKE_AMPLITUDE_DEG)
        assert abs(swing) < abs(d)                                  # toward the start: D2 spends no more
        assert lean_after([1, 0, 0], [0, 0, 1], d) == pytest.approx(rest)
        assert lean_after([1, 0, 0], [0, 0, 1], swing) == pytest.approx(rest + SHAKE_AMPLITUDE_DEG)
        assert shake_geometry_ok([1, 0, 0], [0, 0, 1], d, CFG)


def test_the_whole_roll_fits_checklist_d2():
    """D2 (`roll_paths.RollPathPlanner.accepts`): span <= pi - 0.05 rad, 177.1 deg."""
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
        assert shake_geometry_ok(axis, [0, 0, 1], deltas[0], CFG)


def stub(line_status="Success", grasped=True, shake_count=lambda wrist: 0):
    """A held condiment at the wrist pivot over the bowl, wrist axis along world x (as test_season_pour_paths)."""
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
        cfg=CFG, bowl=SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0., 0., 0.]]))), control_timestep=.05)
    calls, counts = [], {"n": 0}

    def line(goal, current, **kwargs):
        np.testing.assert_array_equal(np.delete(goal, 12), np.delete(current, 12))
        calls.append(('plan', math.degrees(goal[12] - current[12])))
        return {'status': line_status, 'goal': goal[12]}

    def info():
        return {'shake_count': np.array([counts["n"]]), 'shake_armed': np.array([True]), 'shake_swung': np.array([True]),
                'upright': np.array([True]), 'grasp_target': np.array([True]),
                'success': np.array([True]), 'pour_hold': np.array([CFG.hold_steps + 2])}

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

@pytest.fixture(scope="module")
def env():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    import my_scenes  # noqa: F401
    try:
        e = gym.make("MikasaSeasonDish-v0", num_envs=1, robot_uids="ds_fetch", control_mode="pd_joint_pos",
                     obs_mode="state", sim_backend="cpu", render_backend=RENDER, scene_idx=0, assisted_grasp=False,
                     sim_config={"control_freq": 20, "sim_freq": 100})
    except Exception as error:   # no RoboCasa assets / no renderer here
        pytest.skip(f"MikasaSeasonDish-v0 unavailable: {error}")
    yield e
    e.close()


def test_the_number_is_drawn_per_seed_independently_of_the_condiment_and_shown_with_the_picture(env):
    task = env.unwrapped
    seen = set()
    for seed in range(60):
        env.reset(seed=seed)
        n = int(task.num_shakes.item())
        assert n in task.cfg.shake_counts
        seen.add((bool(task.target_is_shaker.item()), n))
        shown = [bool((a.pose.p[0, 2] < 10).item()) for a in task.fridge_digits.actors]
        assert shown == [d == n for d in task.cfg.shake_counts]
        env.reset(seed=seed)
        assert int(task.num_shakes.item()) == n                       # the same seed, the same number
    assert len({n for _, n in seen}) == len(task.cfg.shake_counts) and len(seen) > 3


def test_the_card_disappears_with_the_picture(env):
    task = env.unwrapped
    env.reset(seed=5)
    hold = np.zeros(env.action_space.shape)
    hold[:] = np.r_[task.agent.controller.controllers["arm"].qpos[0].cpu().numpy(), 1.,
                    task.agent.controller.controllers["body"].qpos[0].cpu().numpy(), 0., 0.]
    for _ in range(task.cfg.cue_steps):
        env.step(hold)
    assert all(bool((a.pose.p[0, 2] > 10).item()) for a in task.fridge_digits.actors + task.fridge_pictures.actors)


def test_count_success_and_fail_in_the_scene(env, monkeypatch):
    """The held condiment is placed over the bowl at a given lean each step; the grasp is faked."""
    from mani_skill.utils.structs import Pose
    task = env.unwrapped

    def episode(leans):
        env.reset(seed=7)
        target = task.shaker if bool(task.target_is_shaker.item()) else task.condiment_bottle
        monkeypatch.setattr(task.agent, "is_grasping",
                            lambda obj, min_force=0.5: torch.tensor([obj is target], device=task.device))
        task._elapsed_steps[:] = task.cfg.cue_steps + task.cfg.delay_steps
        bowl = task.bowl.pose.p[0]
        hold = np.r_[task.agent.controller.controllers["arm"].qpos[0].cpu().numpy(), 1.,
                     task.agent.controller.controllers["body"].qpos[0].cpu().numpy(), 0., 0.]
        info = None
        for lean in leans:
            angle = math.radians(180. - lean)            # tilt about world x from upright
            q = torch.tensor([math.cos(angle / 2), math.sin(angle / 2), 0., 0.], device=task.device)
            target.set_pose(Pose.create_from_pq(p=bowl + torch.tensor([0., 0., .2], device=task.device), q=q[None]))
            target.set_linear_velocity(torch.zeros(1, 3, device=task.device))
            target.set_angular_velocity(torch.zeros(1, 3, device=task.device))
            info = env.step(hold)[-1]
        return int(task.num_shakes.item()), info

    n, _ = episode([0])
    shaken = [180, 90, 5] + [30, 2] * n
    upright = [90] + [180] * task.cfg.hold_steps          # turned upright again, then held
    n, info = episode(shaken + upright[:-1])
    assert int(info["shake_count"].item()) == n and not info["success"].item()
    n, info = episode(shaken + upright)
    assert info["success"].item() and not info["fail"].item() and int(info["shake_count"].item()) == n
    n, info = episode(shaken + [2] * 2 * task.cfg.hold_steps)       # held upside down instead
    assert int(info["shake_count"].item()) == n and not info["success"].item()
    n, info = episode([180, 90, 5] + [30, 2] * (n - 1) + upright)  # one shake short
    assert int(info["shake_count"].item()) == n - 1 and not info["success"].item() and not info["fail"].item()
    n, info = episode(shaken + [30, 2] + upright)
    assert info["fail"].item() and not info["success"].item() and int(info["shake_count"].item()) == n + 1


def test_the_counter_survives_a_state_round_trip(env):
    task = env.unwrapped
    env.reset(seed=3)
    task.shake_count[:] = 2
    task.shake_armed[:] = True
    task.shake_swung[:] = True
    state = task.get_state_dict()
    env.reset(seed=4)
    task.set_state_dict(state)
    assert int(task.shake_count.item()) == 2 and bool(task.shake_armed.item()) and bool(task.shake_swung.item())
    assert int(task.num_shakes.item()) == int(state["num_shakes"].item())


def test_a_shifted_hover_puts_the_rest_where_the_hover_was():
    """Roll axis along x through the wrist; the condiment 5 cm beside it along y. Turning it nearly
    upside down carries it ~10 cm to the other side, so the hover moves 10 cm the other way."""
    pytest.importorskip("mplib")
    from planners.season_dish_planner import rest_shifts
    fk = SimpleNamespace(joints={'gripper_link': (None, 'gripper_axis', 'fixed', np.eye(4), None)})
    task = SimpleNamespace(cfg=CFG)
    desired = np.eye(4); desired[:3, 3] = [0., .05, 0.]
    shifts = rest_shifts(fk, task, desired, np.linalg.inv(desired))      # the wrist frame at the origin
    rest = math.radians(180. - (10. - SHAKE_LEAN_MARGIN_DEG))
    assert len(shifts) == 2
    for shift in shifts:
        assert shift == pytest.approx((0., .05 * (1 - math.cos(rest))), abs=1e-4)
