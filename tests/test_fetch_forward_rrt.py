"""Forward surrogate decoding preserves nonholonomic base and joint bounds."""
import numpy as np
import robots.fetch.forward_rrt as forward_rrt
from robots.fetch.forward_rrt import _advance_limit, _decode

GROUP = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12]


def test_oblique_forward_decode_has_no_yaw_or_lateral_motion():
    q0 = np.linspace(-0.3, 0.3, 15)
    heading = np.array([0.6, 0.8])
    rows = np.tile(q0[GROUP], (3, 1))
    rows[:, :3] = [[0, 0, 0], [0.05, 0, 0], [0.2, 0, 0]]
    decoded = _decode(rows, q0, heading, GROUP)
    np.testing.assert_allclose(decoded[:, :2] - q0[:2], np.array([0, 0.05, 0.2])[:, None] * heading)
    np.testing.assert_array_equal(decoded[:, 2], np.full(3, q0[2]))
    np.testing.assert_array_equal(decoded[:, 3:], rows[:, 3:])
    velocities = np.zeros_like(rows); velocities[:, 0] = 0.08
    rates = _decode(velocities, q0, heading, GROUP, rates=True)
    np.testing.assert_allclose(rates[:, :2], np.tile(heading * 0.08, (3, 1)))
    assert np.all(rates[:, 2:] == 0)


def test_advance_limit_intersects_original_xy_stops_and_short_drive_cap():
    limits = np.tile([-1.0, 1.0], (15, 1))
    q0 = np.zeros(15); q0[0] = 0.94
    assert np.isclose(_advance_limit(q0, [0.6, 0.8], limits, 0.25), 0.1)
    q0[0] = -0.94
    assert np.isclose(_advance_limit(q0, [-0.6, 0.8], limits, 0.25), 0.1)
    assert _advance_limit(np.zeros(15), [1.0, 0.0], limits, 4.0) == 0.25
    assert _advance_limit(np.zeros(15), [1.0, 0.0], limits, 0.0) == 0.0


def test_ik_targets_reject_seed108_advance_limit_overshoot():
    start = np.zeros(15)
    limits = np.tile([-50., 50.], (15, 1))
    limits[0] = [0., 0.19169408272745542]
    bad = start.copy(); bad[0] = 0.19238741532977133; bad[5] = 0.1
    valid = start.copy(); valid[0] = 0.1007941243095703; valid[5] = 0.3
    goals = forward_rrt._valid_ik_goals([bad, valid], start, limits)
    assert len(goals) == 1
    np.testing.assert_array_equal(goals[0], valid)
    assert all(np.all(g >= limits[:, 0]) and np.all(g <= limits[:, 1]) for g in goals)


def test_ik_targets_keep_all_valid_solutions_ranked_by_arm_distance():
    start = np.zeros(15)
    limits = np.tile([-1., 1.], (15, 1))
    a = start.copy(); a[5] = 0.5
    b = start.copy(); b[5] = 0.1
    goals = forward_rrt._valid_ik_goals([a, b], start, limits)
    assert len(goals) == 2
    np.testing.assert_array_equal(goals[0], b)
    np.testing.assert_array_equal(goals[1], a)


def test_ik_targets_reject_nonfinite_and_all_invalid_without_mutation():
    start = np.zeros(15)
    limits = np.tile([-1., 1.], (15, 1))
    goals = np.zeros((3, 15))
    goals[0, 0] = np.nan
    goals[1, 3] = 2.0
    goals[2, 0] = np.inf
    original = goals.copy()
    assert forward_rrt._valid_ik_goals(goals, start, limits) == []
    np.testing.assert_array_equal(goals, original)


def test_ik_targets_validate_unwrapped_periodic_equivalent_without_mutation():
    start = np.zeros(15); start[8] = 3.0
    limits = np.tile([-1., 1.], (15, 1)); limits[8] = [-6.28, 6.28]
    candidate = np.zeros(15); candidate[8] = -3.0
    original = candidate.copy()
    goals = forward_rrt._valid_ik_goals([candidate], start, limits)
    assert len(goals) == 1
    assert np.isclose(goals[0][8], -3.0 + 2 * np.pi)
    np.testing.assert_array_equal(candidate, original)
