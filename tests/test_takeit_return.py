"""Regression checks for the appended TakeItBack return route."""

from types import SimpleNamespace

import numpy as np
import sapien
import torch

from my_scenes.my_robocasa_takeit import MyRoboCasaSceneTakeIt
from my_scenes.my_robocasa_takeitback import MyRoboCasaSceneTakeItBack
from planners.myrobocasa_takeit_planner import (
    _needs_return,
    _counter_aligned_base_target,
    _return_cup_goal,
    _return_touch_links,
    _tray_stage_success,
)


def test_only_round_trip_task_continues_after_tray():
    back = object.__new__(MyRoboCasaSceneTakeItBack)
    takeit = object.__new__(MyRoboCasaSceneTakeIt)
    assert _needs_return(back)
    assert not _needs_return(takeit)
    assert not _needs_return(SimpleNamespace())


def test_wp16_accepts_intermediate_tray_latch_not_full_round_trip():
    back = object.__new__(MyRoboCasaSceneTakeItBack)
    back.placed_on_tray = torch.tensor([True])
    assert _tray_stage_success(back, {"success": torch.tensor([False])})
    back.placed_on_tray[:] = False
    assert not _tray_stage_success(back, {"success": torch.tensor([False])})


def test_takeit_still_requires_strict_task_success():
    takeit = object.__new__(MyRoboCasaSceneTakeIt)
    takeit.placed_on_tray = torch.tensor([True])
    assert not _tray_stage_success(takeit, {"success": torch.tensor([False])})
    assert _tray_stage_success(takeit, {"success": torch.tensor([True])})


def test_return_goal_preserves_measured_grasp_and_initial_target():
    tcp = sapien.Pose([1.2, -0.3, 1.4], [0.5, -0.5, 0.5, -0.5])
    cup = sapien.Pose([1.25, -0.27, 1.35])
    initial = np.array([0.4, 0.7, 0.95])
    before = initial.copy()
    target, cup_goal = _return_cup_goal(tcp, cup, initial)
    reached = sapien.Pose(target.p, target.q) * (tcp.inv() * cup)
    np.testing.assert_allclose(reached.p, initial + [0.0, 0.0, 0.17], atol=1e-6)
    np.testing.assert_allclose(reached.q, cup.q, atol=1e-6)
    np.testing.assert_allclose(cup_goal.p, reached.p, atol=1e-6)
    np.testing.assert_array_equal(initial, before)


def test_collision_exemptions_keep_actual_prefixed_names_and_only_grasp_links():
    names = [
        "scene-0-ds_fetch_gripper_link",
        "scene-0-ds_fetch_l_gripper_finger_link",
        "scene-0-ds_fetch_r_gripper_finger_link",
        "scene-0-ds_fetch_forearm_roll_link",
        "scene-0-ds_fetch_torso_lift_link",
        "scene-0-ds_fetch_base_link",
    ]
    assert (
        _return_touch_links([SimpleNamespace(name=name) for name in names]) == names[:3]
    )


def test_counter_route_aligns_base_before_turn_not_extended_cup():
    initial = np.array([2.5, -0.55, 1.0])
    base = np.array([1.0, -1.0, 0.0])
    for axis in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])):
        np.testing.assert_allclose(
            _counter_aligned_base_target(base, initial, axis), [2.5, -1.0, 0.0]
        )
    np.testing.assert_allclose(base, [1.0, -1.0, 0.0])
