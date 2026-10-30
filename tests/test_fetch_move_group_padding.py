"""Fetch's arm move group skips interleaved head joints in full qpos."""

from types import SimpleNamespace

import numpy as np

from robots.fetch.utils import SapienPlannerV2


MOVE_GROUP = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12]
OTHER_JOINTS = [4, 6, 13, 14]  # Head pan/tilt and gripper fingers.


class Articulation:
    def __init__(self):
        self.qpos = np.zeros(15)
        self.qpos[OTHER_JOINTS] = [0.14, 0.26, 0.034, 0.031]

    def get_qpos(self):
        return self.qpos

    def get_move_group_qpos_dim(self):
        return len(MOVE_GROUP)

    def get_move_group_joint_indices(self):
        return MOVE_GROUP

    def set_qpos(self, qpos, full):
        assert full and np.asarray(qpos).shape == (15,)
        self.qpos = np.asarray(qpos).copy()


def planner():
    p = object.__new__(SapienPlannerV2)
    p.robot = Articulation()
    p.joint_limits = np.tile([-10.0, 10.0], (15, 1))
    return p


def test_move_group_padding_uses_joint_indices_and_preserves_other_joints():
    p = planner()
    before = p.robot.get_qpos().copy()
    group = np.array([1, 2, 0.3, 0.38, 0.01, -0.76, 0.002, 1.74, 0.015, -0.99, -0.012])
    full = p.pad_move_group_qpos(group)
    np.testing.assert_array_equal(full[MOVE_GROUP], group)
    np.testing.assert_array_equal(full[OTHER_JOINTS], before[OTHER_JOINTS])
    np.testing.assert_array_equal(p.robot.get_qpos(), before)
    np.testing.assert_array_equal(p.pad_move_group_qpos(full), full)


def test_prefix_move_group_keeps_previous_padding_behavior():
    p = planner()
    p.robot.get_move_group_joint_indices = lambda: list(range(11))
    before = p.robot.get_qpos().copy()
    group = np.linspace(-0.5, 0.5, 11)
    expected = before.copy()
    expected[:11] = group
    np.testing.assert_array_equal(p.pad_move_group_qpos(group), expected)
    np.testing.assert_array_equal(p.robot.get_qpos(), before)


def test_collision_check_uses_requested_fetch_pose_and_restores_model_state():
    p = planner()
    before = p.robot.get_qpos().copy()
    group = np.array([1, 2, 0.3, 0.38, 0.01, -0.76, 0.002, 1.74, 0.015, -0.99, -0.012])
    expected = before.copy()
    expected[MOVE_GROUP] = group
    checked = []

    def collision_query():
        actual = p.robot.get_qpos().copy()
        checked.append(actual)
        return [] if np.array_equal(actual, expected) else ["wrong robot pose"]

    p.planning_world = SimpleNamespace(
        get_planned_articulations=lambda: [p.robot],
        check_robot_collision=collision_query,
    )
    # This calls the inherited real MPlib collision-check path, not just pad().
    assert p.check_for_env_collision(group) == []
    np.testing.assert_array_equal(checked[0], expected)
    np.testing.assert_array_equal(p.robot.get_qpos(), before)
