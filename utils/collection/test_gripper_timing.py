"""Binary command and 20 Hz / 10 Hz boundary tests through the real primitives."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from planners.oracle.collection_solver import CollectionMotionPlanner
from robots.fetch.stepping import StepGuard


def planner_at(step):
    records = []
    env = SimpleNamespace(elapsed_steps=torch.tensor([step]))

    def physical_step(action):
        records.append((int(env.elapsed_steps[0]), action.copy()))
        env.elapsed_steps += 1
        return {}, 0.0, False, False, {}

    env.step = physical_step
    planner = object.__new__(CollectionMotionPlanner)
    planner.base_env = env
    planner.control_mode = "pd_joint_pos"
    planner.env_agent = SimpleNamespace(
        controller=SimpleNamespace(
            controllers={
                "arm": SimpleNamespace(
                    qpos=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
                ),
                "body": SimpleNamespace(qpos=torch.tensor([[0.2, 0.3, 0.35]])),
            }
        )
    )
    planner._guard = StepGuard(env)
    planner.gripper_state = 1.0
    planner.print_env_info = False
    planner.vis = False
    return planner, records


@pytest.mark.parametrize("start", [0, 1, 10, 11])
def test_switch_is_even_and_preserved_by_stride_two(start):
    planner, records = planner_at(start)
    planner.close_gripper(t=2)
    switches = [step for step, action in records if action[7] == -1]
    assert switches[0] == start + start % 2
    assert switches[0] % 2 == 0
    assert len(records) == 2 + start % 2
    assert [action[7] for step, action in records if step % 2 == 0][0] == -1
    if start % 2:
        assert records[0][1][7] == 1
        np.testing.assert_array_equal(records[0][1][:7], records[1][1][:7])
    assert {action[7] for _, action in records} <= {-1, 1}


def test_repeated_command_does_not_insert_an_unneeded_hold():
    planner, records = planner_at(1)
    planner.open_gripper(t=2)
    assert [step for step, _ in records] == [1, 2]


def test_intermediate_gripper_and_ramp_are_rejected_before_any_step():
    planner, records = planner_at(0)
    with pytest.raises(ValueError, match="binary"):
        planner.change_gripper_state(gripper_state=0.4)
    with pytest.raises(ValueError, match="binary"):
        planner.open_gripper(ramp=8)
    assert records == []
