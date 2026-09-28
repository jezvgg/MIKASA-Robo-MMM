"""The unrolled grasp family reaches the identical hand pose without the forearm roll."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from planners.oracle.upright_payload import ArmKinematics
from planners.season_dish_candidates import NAMES, TEMPLATES, unrolled
from planners.season_dish_paths import CARRY_TARGETS, UNROLLED_CARRY_TARGETS

ACTIVE = ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint', 'torso_lift_joint',
          'head_pan_joint', 'shoulder_pan_joint', 'head_tilt_joint', 'shoulder_lift_joint',
          'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint',
          'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint']
URDF = Path(__file__).resolve().parents[2] / 'robots/fetch/fetch.urdf'


def kinematics():
    root = SimpleNamespace(sp=SimpleNamespace(to_transformation_matrix=lambda: np.eye(4)))
    robot = SimpleNamespace(get_active_joints=lambda: [SimpleNamespace(name=n) for n in ACTIVE],
                            root_pose=[root])
    return ArmKinematics(SimpleNamespace(agent=SimpleNamespace(urdf_path=str(URDF), robot=robot)))


def qpos(values):
    q = np.zeros(len(ACTIVE))
    q[ACTIVE.index('torso_lift_joint')] = .2
    q[ACTIVE.index('shoulder_pan_joint')] = -.4
    for name, value in values.items():
        q[ACTIVE.index(name)] = value
    return q


def test_unrolled_templates_put_the_hand_in_the_same_pose():
    fk = kinematics()
    for template in TEMPLATES:
        a = fk.matrix(qpos(dict(zip(NAMES, template))))
        b = fk.matrix(qpos(dict(zip(NAMES, unrolled(template)))))
        np.testing.assert_allclose(a, b, atol=1e-9)
        assert unrolled(template)[4] == -template[4]


def test_unrolled_carry_posture_is_the_same_hand_pose_with_a_small_forearm_roll():
    fk = kinematics()
    np.testing.assert_allclose(fk.matrix(qpos(CARRY_TARGETS)), fk.matrix(qpos(UNROLLED_CARRY_TARGETS)), atol=1e-9)
    assert abs(UNROLLED_CARRY_TARGETS['forearm_roll_joint']) < .5
    assert UNROLLED_CARRY_TARGETS['wrist_flex_joint'] > 0 > CARRY_TARGETS['wrist_flex_joint']


def test_unrolled_is_an_involution_up_to_full_turns():
    for template in TEMPLATES:
        back = np.array(unrolled(unrolled(template)))
        diff = (back - np.array(template) + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(diff, 0, atol=1e-12)
