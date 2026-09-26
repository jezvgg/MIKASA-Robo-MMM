"""Do not execute an approach that leaves no admissible contact stroke."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from planners.oracle.collection_solver import CollectionMotionPlanner
from planners.oracle.roll_paths import RollPathPlanner


@pytest.mark.parametrize("case", ["no_contact", "combined_roll", "reachable"])
@pytest.mark.parametrize("torso_height", [None, 0.1])
def test_contact_lookahead_checks_both_legs_before_execution(case, torso_height):
    owner = object.__new__(CollectionMotionPlanner)
    owner._roll_indices = [8, 10, 12]
    owner._roll_low = np.zeros(3)
    owner._roll_high = np.zeros(3)
    current = np.zeros(15)
    owner.robot = SimpleNamespace(get_qpos=lambda: torch.tensor(current[None]))
    owner.base_env = SimpleNamespace(control_timestep=0.05)
    line = np.zeros((3, 11))
    contact = np.zeros((3, 11))
    line[1, 8] = 1.7
    contact[1, 8] = -1.7 if case == "combined_roll" else 1.6

    def forbidden_sync():
        raise RuntimeError("The target drawer is temporarily absent from planning")

    def inverse_kinematics(pose, start, fixed, **kwargs):
        assert start[3] == (0.0 if torso_height is None else torso_height)
        assert fixed[3] == (torso_height is not None)
        return "Success", [start.copy()]

    native = SimpleNamespace(
        joint_limits=np.tile([-6.28, 6.28], (15, 1)),
        move_group_joint_indices=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12],
        update_from_simulation=forbidden_sync,
        fold_qpos=lambda q: q.copy(),
        robot=SimpleNamespace(set_qpos=lambda *args: None),
        _transform_goal_to_wrt_base=lambda pose: pose,
        IK=inverse_kinematics,
        plan_qpos_line=lambda *args, **kwargs: {"status": "Success", "position": line},
        plan_screw=lambda *args, **kwargs: {
            "status": "No path" if case == "no_contact" else "Success",
            "position": contact,
        },
    )
    owner.planner = RollPathPlanner(native, owner)
    executed = []

    def line_executor(*args, precheck, **kwargs):
        if precheck(args[1][0]):
            executed.append(True)
            return "executed"
        return None

    owner._line_to_ik_goals = line_executor
    pose = SimpleNamespace(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    result = owner.approach_for_contact(pose, pose, torso_height=torso_height)
    np.testing.assert_array_equal(current, np.zeros(15))
    assert result == ("executed" if case == "reachable" else -1)
    assert executed == ([True] if case == "reachable" else [])
