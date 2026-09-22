"""Physical regression: arm following and base movement preserve a nonzero gaze."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .contract import write_json
from .profile import make_env, runtime_signature


def qualify(output):
    import gymnasium as gym
    from planners.oracle.oracle_common import default_planner_factory

    signature = runtime_signature()
    class Trace(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.actions = []
        def step(self, action):
            self.actions.append(np.asarray(action).reshape(-1).copy())
            return self.env.step(action)

    env = Trace(make_env({"signature": signature}))
    try:
        env.reset(seed=350)
        planner = default_planner_factory(env, False, False)
        planner.hold_head(0.4, -0.3, t=25, ramp=10)
        start = len(env.actions)
        qpos = planner.robot.get_qpos()[0].cpu().numpy().astype(float)
        indices = list(planner.planner.move_group_joint_indices)
        positions = np.tile(qpos[indices], (8, 1))
        joint = env.unwrapped.agent.robot.active_joints_map["shoulder_lift_joint"]
        index = indices.index(joint.active_index[0].item())
        positions[:, index] += np.linspace(0., 0.02, 8)
        result = {"status": "Success", "position": positions,
                  "velocity": np.zeros_like(positions)}
        planner.follow_forward_path_w_refinement(result, refine=False)
        after_arm = len(env.actions)
        planner.idle_steps(t=8)
        planner.turn_in_place(np.array([np.cos(np.pi/2+0.04), np.sin(np.pi/2+0.04), 0.]), max_steps=30)
        actions = np.asarray(env.actions[start:])
        if not len(actions) or not np.all(actions[:, 8] > 0.2) or not np.all(actions[:, 9] < -0.15):
            raise AssertionError(f"An arm/base primitive recentered the head: {actions[:, 8:10].tolist()}")
        report = dict(status="success", seed=350, source_sha256=signature["code_sha256"],
            arm_follow_steps=after_arm-start, total_checked_steps=len(actions),
            commanded_gaze=[0.4, -0.3], head_action_min=actions[:,8:10].min(0).tolist(),
            head_action_max=actions[:,8:10].max(0).tolist())
        write_json(output, report)
        print(report)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    qualify(parser.parse_args().output.resolve())
