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


@pytest.mark.parametrize("references", [False, True])
def test_reference_goal_offered_only_after_random_goals_fail_lookahead(references):
    owner = object.__new__(CollectionMotionPlanner)
    owner._roll_indices = [8, 10, 12]
    owner._roll_low = np.zeros(3)
    owner._roll_high = np.zeros(3)
    current = np.zeros(15)
    owner.robot = SimpleNamespace(get_qpos=lambda: torch.tensor(current[None]))
    owner.base_env = SimpleNamespace(control_timestep=0.05)
    random_goal = np.zeros(15)
    random_goal[5] = 1.0          # its contact stroke will wind the roll window
    reference_goal = np.zeros(15)
    reference_goal[5] = -1.0
    seen = []

    def inverse_kinematics(pose, start, fixed, *, n_init_qpos=20, **kwargs):
        seen.append((start[5], n_init_qpos))
        if n_init_qpos == 1:
            return "Success", [reference_goal.copy()]
        return "Success", [random_goal.copy()]

    def screw(pose, start, **kwargs):
        position = np.zeros((3, 11))
        position[1, 8] = 3.5 if start[5] > 0 else 0.5
        return {"status": "Success", "position": position}

    native = SimpleNamespace(
        joint_limits=np.tile([-6.28, 6.28], (15, 1)),
        move_group_joint_indices=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12],
        fold_qpos=lambda q: q.copy(),
        robot=SimpleNamespace(set_qpos=lambda *args: None),
        _transform_goal_to_wrt_base=lambda pose: pose,
        IK=inverse_kinematics,
        plan_qpos_line=lambda *args, **kwargs: {"status": "Success", "position": np.zeros((3, 11))},
        plan_screw=screw,
    )
    owner.planner = RollPathPlanner(native, owner)
    if references:
        owner._ik_reference_arms = [("r", {5: -0.9})]
        owner._ik_reference_used = []
    executed = []

    def line_executor(target, goals, *args, precheck, **kwargs):
        goal = goals[0]
        if precheck(goal):
            executed.append(goal[5])
            return "executed"
        return None

    owner._line_to_ik_goals = line_executor
    pose = SimpleNamespace(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    result = owner.approach_for_contact(pose, pose, torso_height=0.1)
    assert seen[0] == (0.0, 40)  # the random-restart request comes first
    if references:
        assert result == "executed" and executed == [-1.0]
        assert owner._ik_reference_used == ["r"]
        assert seen[1] == (-0.9, 1)
    else:
        assert result == -1 and executed == [] and len(seen) == 1
