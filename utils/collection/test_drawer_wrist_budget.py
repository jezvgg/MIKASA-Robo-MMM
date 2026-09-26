"""Temporary planning bounds must not erase the measured episode history."""
from types import SimpleNamespace

import numpy as np
import pytest

from planners.same_drawer_paths import DrawerPathPlanner


def guard():
    owner = SimpleNamespace(_roll_indices=[8, 10, 12], _roll_low=np.zeros(3),
                            _roll_high=np.zeros(3), _drawer_initial_wrist_budget=True)
    native = SimpleNamespace(joint_limits=np.tile([-6.28, 6.28], (15, 1)),
                             move_group_joint_indices=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12])
    return DrawerPathPlanner(native, owner), native, owner


def test_reserves_complete_path_then_retains_history_after_release():
    p, _, owner = guard()
    path = np.zeros((3, 11));path[1, 10] = 1.6
    assert not p.accepts(path, move_group=True)
    owner._drawer_initial_wrist_budget = False
    assert p.accepts(path, move_group=True)
    owner._roll_high[2] = 1.6
    path[1, 10] = -1.6
    assert not p.accepts(path, move_group=True)
    assert owner._roll_high[2] == 1.6


def test_restores_native_limits_even_on_refusal():
    p, native, owner = guard()
    original = native.joint_limits
    with pytest.raises(RuntimeError):
        with p._limits():
            np.testing.assert_array_equal(native.joint_limits[12], [-1.45, 1.45])
            raise RuntimeError('failed plan')
    assert native.joint_limits is original
    np.testing.assert_array_equal(owner._roll_low, np.zeros(3))
