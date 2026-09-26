"""Pour fallback must preserve root frames and reject unsafe/long paths."""

from types import SimpleNamespace

import numpy as np
import sapien

from planners.season_dish_planner import move_to_pour_pose


def setup_planner(line_status="Success", line_length=30):
    current = np.zeros(15)
    current[:3] = [.2, .3, .4]
    folded = current.copy()
    folded[:3] = [2., -1., 1.97]
    goal = folded.copy()
    goal[9] = 1.
    calls = []

    def line(candidate, start, **kwargs):
        np.testing.assert_array_equal(start, current)
        np.testing.assert_array_equal(candidate[:3], current[:3])
        np.testing.assert_array_equal(goal[:3], folded[:3])
        calls.append("line")
        return {"status": line_status, "position": np.zeros((line_length, 11))}

    native = SimpleNamespace(
        plan_screw=lambda *a, **kw: {"status": "torso stop"},
        fold_qpos=lambda q: folded.copy(),
        _transform_goal_to_wrt_base=lambda pose: pose,
        IK=lambda *a, **kw: ("Success", [goal]),
        plan_qpos_line=line,
        move_group_joint_indices=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12],
    )
    def execute(path, *, refine):
        assert refine
        calls.append("execute")
        return "executed"
    planner = SimpleNamespace(planner=native, ARM_SCREW_GOAL_TOLERANCE=(.02, .1),
                              follow_forward_path_w_refinement=execute)
    task = SimpleNamespace(agent=SimpleNamespace(robot=SimpleNamespace(
        get_qpos=lambda: current[None])), control_timestep=.05)
    return planner, task, calls


def test_reachable_joint_line_uses_simulator_root_and_keeps_ik_input_intact():
    planner, task, calls = setup_planner()
    assert move_to_pour_pose(planner, task, sapien.Pose()) == "executed"
    assert calls == ["line", "execute"]


def test_collision_failure_never_executes_the_ik_endpoint():
    planner, task, calls = setup_planner(line_status="finger collision")
    assert move_to_pour_pose(planner, task, sapien.Pose()) == -1
    assert calls == ["line"]


def test_long_pour_path_is_not_executed():
    planner, task, calls = setup_planner(line_length=101)
    assert move_to_pour_pose(planner, task, sapien.Pose()) == -1
    assert calls == ["line"]
