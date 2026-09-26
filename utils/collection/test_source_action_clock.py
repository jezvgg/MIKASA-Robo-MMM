"""The 20 Hz physical recording must equal the exported 10 Hz target hold."""

import numpy as np

from utils.collection.test_gripper_timing import planner_at


def test_all_executed_channels_survive_stride_two_without_action_rewriting():
    planner, records = planner_at(0)
    planner._guard.tape = []
    for step in range(8):
        proposed = np.linspace(-.2, .2, 13) + step * .01
        proposed[7] = 1.
        planner._step(proposed)
    executed = np.stack([action for _, action in records])
    replayed = np.repeat(executed[::2], 2, axis=0)
    np.testing.assert_array_equal(executed, replayed)
    assert not np.array_equal(executed[0], executed[2])
    np.testing.assert_array_equal(np.stack(planner._guard.tape), executed)


def test_even_gripper_switch_starts_a_complete_saved_action_pair():
    planner, records = planner_at(0)
    planner.open_gripper(t=3)
    planner.close_gripper(t=2)
    executed = np.stack([action for _, action in records])
    np.testing.assert_array_equal(executed, np.repeat(executed[::2], 2, axis=0))
    assert executed[3, 7] == 1. and executed[4, 7] == -1.


def test_scoped_joint_lock_preserves_wrist_and_head_and_recording():
    planner, records = planner_at(0)
    planner._guard.tape = []
    planner.fixed_action_targets = {i: .123 for i in range(6)}
    planner.fixed_action_targets.update({10: .2, 11: 0., 12: 0.})
    for step in range(8):
        action = np.full(13, step * .01)
        action[7] = -1.
        planner._step(action)
    executed = np.stack([a for _, a in records])
    np.testing.assert_array_equal(executed[:, :6], np.full((8, 6), .123))
    assert np.ptp(executed[:, 6]) > 0 and np.ptp(executed[:, 8]) > 0
    np.testing.assert_array_equal(executed[:, 11:13], np.zeros((8, 2)))
    np.testing.assert_array_equal(executed, np.repeat(executed[::2], 2, axis=0))
    np.testing.assert_array_equal(np.stack(planner._guard.tape), executed)
