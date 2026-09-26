"""A last-step success must not bypass the collection horizon gate."""

import json

import h5py
import numpy as np
import pytest

from .client import execute_actions
from .contract import recording_info
from .pipeline import classify_collection_result


def test_success_and_truncation_on_same_step_are_rejected():
    class Env:
        def step(self, action):
            self.result = ({}, 1.0, True, True, {"success": True})
            return self.result

    env = Env()
    actions = np.zeros((1, 13), dtype=np.float32)
    actions[:, 7] = 1
    result = execute_actions(env, actions, repeat=1)
    assert result["completed"] and result["success"]
    assert result["status"] == "truncated"
    assert classify_collection_result(env.result) == "truncated"
    assert (
        classify_collection_result(({}, 1.0, True, False, {"success": True}))
        == "success"
    )


@pytest.mark.parametrize("flags", [None, [False, False], [False, True]])
def test_h5_gate_reads_actual_truncation_history(tmp_path, flags):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as h5:
        g = h5.create_group("traj_0")
        g.create_dataset("actions", data=np.zeros((2, 13)))
        g.create_dataset("success", data=[False, True])
        g.create_dataset("rewards", data=[0.0, 1.0])
        g.create_dataset("env_states/example", data=np.zeros((3, 1)))
        if flags is not None:
            g.create_dataset("truncated", data=flags)
    meta = dict(
        env_info={
            "env_kwargs": {
                "robot_uids": "ds_fetch",
                "control_mode": "pd_joint_pos",
                "sim_config": {"control_freq": 20},
            }
        },
        mikasa_data={
            "stage": "oracle",
            "version": 3,
            "policy_fps": 20,
            "action_repeat": 1,
            "instruction": "Place the apple on the plate.",
            "success": True,
            "success_once": True,
        },
        episodes=[
            {
                "episode_id": 0,
                "control_mode": "pd_joint_pos",
                "elapsed_steps": 2,
                "success": True,
                "success_once": True,
            }
        ],
    )
    path.with_suffix(".json").write_text(json.dumps(meta))
    if flags == [False, False]:
        assert recording_info(path, stage="oracle") == meta
    else:
        with pytest.raises(ValueError, match="[Tt]runcat"):
            recording_info(path, stage="oracle")
