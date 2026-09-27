"""Collision-free endpoints do not establish clearance between trajectory knots."""
from types import SimpleNamespace

import numpy as np
import pytest

from planners.oracle.path_clearance import path_clear


def planning_world(collisions):
    q = np.zeros(2)
    world = SimpleNamespace(
        set_qpos_all=lambda value: np.copyto(q, value),
        is_state_colliding=lambda: bool(collisions(q)),
        check_collision=lambda: [SimpleNamespace(link_name1=a, link_name2=b)
                                 for a, b in collisions(q)],
    )
    native = SimpleNamespace(
        move_group_joint_indices=[0, 1], fold_qpos=lambda value: value.copy(),
        robot=SimpleNamespace(set_qpos=lambda *args: None), planning_world=world)
    return SimpleNamespace(planner=native)


def test_thin_upper_handle_between_clear_endpoints_blocks_path():
    planner = planning_world(lambda q: [('forearm', 'upper_cabinet_handle')]
                             if .48 <= q[0] <= .52 else [])
    assert not path_clear(planner, np.zeros(2), [[0, 0], [1, 0]])


def test_clear_detour_passes_with_every_fixture_present():
    planner = planning_world(lambda q: [('forearm', 'upper_cabinet_handle')]
                             if .48 <= q[0] <= .52 and q[1] < .2 else [])
    assert path_clear(planner, np.zeros(2), [[0, 0], [0, 1], [1, 1], [1, 0]])


@pytest.mark.parametrize('when', ['start', 'later', 'persistent'])
def test_support_exception_is_only_for_the_starting_state(when):
    pair = ('payload', 'counter')
    def collisions(q):
        hit = q[0] == 0 if when == 'start' else q[0] > .5 if when == 'later' else True
        return [pair] if hit else []
    planner = planning_world(collisions)
    assert path_clear(planner, np.zeros(2), [[0, 0], [1, 0]],
                      initial_contacts=[pair]) == (when == 'start')


def test_support_exception_does_not_allow_robot_upper_handle_contact():
    planner = planning_world(lambda q: [('payload', 'counter'), ('wrist', 'upper_handle')]
                             if q[0] == 0 else [])
    assert not path_clear(planner, np.zeros(2), [[0, 0], [1, 0]],
                          initial_contacts=[('payload', 'counter')])
