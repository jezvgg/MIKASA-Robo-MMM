from types import SimpleNamespace

import numpy as np

from robots.fetch.extand import FetchMotionPlanningSapienSolver


def _noise_state(seed):
    state = SimpleNamespace(
        action_noise=0.001,
        noise_hold=2,
        _execution_noise_rng=np.random.default_rng(),
        _execution_noise=None,
        _execution_noise_steps=0,
    )
    state._reset_execution_noise = lambda: (
        FetchMotionPlanningSapienSolver._reset_execution_noise(state)
    )
    FetchMotionPlanningSapienSolver.set_execution_noise_seed(state, seed)
    return state


def test_execution_noise_is_seeded_and_held_for_configured_steps():
    action = np.zeros(7)
    first = _noise_state(7)
    replay = _noise_state(7)

    draw_1 = FetchMotionPlanningSapienSolver._apply_execution_noise(first, action)
    draw_2 = FetchMotionPlanningSapienSolver._apply_execution_noise(first, action)
    draw_3 = FetchMotionPlanningSapienSolver._apply_execution_noise(first, action)
    replay_draw = FetchMotionPlanningSapienSolver._apply_execution_noise(replay, action)

    assert np.array_equal(draw_1, draw_2)
    assert not np.array_equal(draw_2, draw_3)
    assert np.array_equal(draw_1, replay_draw)
