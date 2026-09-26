"""Execution timing and command-boundary regressions for checklist C3/B2/B4."""

from types import SimpleNamespace

import numpy as np
import torch
import pytest

from utils.mikasa.execution_noise import ExecutionNoise


def test_blocks_are_seeded_and_phase_exit_restores_clean_commands():
    def sample(global_seed):
        np.random.seed(global_seed)
        noise = ExecutionNoise(
            seed=23, action_noise=0.003, noise_hold=10, log=lambda *args, **kwargs: None
        )
        original = np.arange(7) / 10
        clean = original.copy()
        with noise.phase("transfer"):
            result = np.stack([noise.apply(clean, i) for i in range(25)])
        np.testing.assert_array_equal(clean, original)
        np.testing.assert_array_equal(noise.apply(clean, 25), clean)
        np.testing.assert_array_equal(result[:10], np.broadcast_to(result[0], (10, 7)))
        np.testing.assert_array_equal(
            result[10:20], np.broadcast_to(result[10], (10, 7))
        )
        assert not np.array_equal(result[0], result[10])
        return result

    np.testing.assert_array_equal(sample(42), sample(999))


@pytest.mark.parametrize("path_length", [11, 12, 13])
def test_noise_reaches_executed_knots_but_not_refinement_or_gripper(monkeypatch, path_length):
    from planners.oracle.collection_solver import CollectionMotionPlanner
    from robots.fetch.extand import FetchMotionPlanningSapienSolver

    records, logs = [], []
    planner = object.__new__(CollectionMotionPlanner)
    planner.control_mode = "pd_joint_pos"
    planner._guard = SimpleNamespace(truncated=False)
    planner.base_env = SimpleNamespace(
        control_mode="pd_joint_pos", elapsed_steps=torch.tensor([0])
    )
    planner.env_agent = SimpleNamespace(
        controller=SimpleNamespace(
            controllers={"arm": SimpleNamespace(qpos=torch.zeros(1, 7))}
        )
    )
    planner.execution_noise = ExecutionNoise(
        seed=9,
        action_noise=0.003,
        noise_hold=10,
        log=lambda *args, **kwargs: logs.append((args, kwargs)),
    )
    clean = np.array([0.0] * 7 + [1.0, 0.2, 0.3, 0.35, 1e-15, -0.4])

    def physical_step(self, action):
        np.testing.assert_array_equal(self._last_abs, action)
        records.append(action.copy())
        self.base_env.elapsed_steps += 1
        return None

    def path_and_refinement(self, result, refine, stop_when):
        for _ in range(len(result["position"]) + 2):
            self._step(clean)

    monkeypatch.setattr(FetchMotionPlanningSapienSolver, "_step", physical_step)
    monkeypatch.setattr(
        FetchMotionPlanningSapienSolver,
        "follow_forward_path_w_refinement",
        path_and_refinement,
    )
    path = {"position": np.zeros((path_length, 15))}
    with planner.execution_noise.phase("transfer"):
        planner.follow_forward_path_w_refinement(path, refine=True)
    # A subsequent contact/gripper step stays clean.
    planner._step(clean)
    result = np.stack(records)
    noisy_ticks = 2 * ((path_length + 1) // 2)
    assert np.any(result[:noisy_ticks, :7] != 0)
    np.testing.assert_array_equal(result[noisy_ticks:, :7], 0)
    np.testing.assert_array_equal(result[:-1], np.repeat(result[:-1:2], 2, axis=0))
    np.testing.assert_array_equal(
        result[:, 7:11], np.broadcast_to(clean[7:11], (len(result), 4))
    )
    np.testing.assert_array_equal(result[:, 11], 0)
    np.testing.assert_array_equal(result[:, 12], -0.4)
    assert clean[11] == 1e-15
    steps = [
        data["control_step"] for tags, data in logs if tags[0] == "execution_noise_step"
    ]
    assert steps == list(range(0, path_length, 2))
