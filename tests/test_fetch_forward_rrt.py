"""Forward surrogate decoding preserves nonholonomic base and joint bounds."""
import numpy as np
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
