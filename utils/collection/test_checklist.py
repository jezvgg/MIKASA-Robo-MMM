"""Regression cases for audit false positives and false negatives."""

import numpy as np
from utils.collection.checklist import motion_metrics

NAMES = [
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
    "torso_lift_joint",
    "head_pan_joint",
    "shoulder_pan_joint",
    "head_tilt_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "r_gripper_finger_joint",
    "l_gripper_finger_joint",
]


def test_continuous_full_turn_is_not_hidden_by_angle_wrapping():
    a = np.zeros((80, 13))
    a[:, 7] = 1
    a[:, 12] = 0.1
    q = np.zeros((81, 15))
    q[:, 2] = np.linspace(0, 2 * np.pi, 81)
    m = motion_metrics(a, q, NAMES)
    assert np.isclose(m["max_continuous_turn_rad"], 2 * np.pi)
    assert m["longest_constant_action_frames"] == 80


def test_turn_segments_split_at_zero_without_losing_endpoint():
    a = np.zeros((9, 13))
    a[:, 7] = 1
    a[:4, 12] = 0.1
    a[5:, 12] = -0.1
    q = np.zeros((10, 15))
    q[:, 2] = [0, 0.4, 0.8, 1.2, 1.6, 1.6, 1.2, 0.8, 0.4, 0]
    m = motion_metrics(a, q, NAMES)
    assert np.isclose(m["max_continuous_turn_rad"], 1.6)
    assert np.isclose(m["total_turn_rad"], 3.2)


def test_gripper_parity_and_base_roundoff_are_separate_from_real_reverse():
    a = np.zeros((8, 13))
    a[:3, 7] = 1
    a[3:, 7] = -1
    a[:, 11] = -1e-15
    q = np.zeros((9, 15))
    m = motion_metrics(a, q, NAMES)
    assert m["gripper_switch_steps"] == [3]
    assert m["base_tiny_nonzero_count"] == 8
    assert m["reverse_frames"] == 0
    a[4:, 11] = -0.1
    m = motion_metrics(a, q, NAMES)
    assert m["reverse_frames"] == 4
