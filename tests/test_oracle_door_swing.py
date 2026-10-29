"""Numpy-only checks of the BIMBO door-swing geometry: python -m pytest tests/test_oracle_door_swing.py"""

import math

import numpy as np
import pytest

from planners import cabinet_door_bimbo as ds

# cab_main's right leaf (CAB_MAIN_RIGHT): hinge, closed bar, opening sense (+1, CCW)
HINGE = np.array([2.735, -0.400])
BAR = np.array([2.302, -0.430, 1.592])


def grasp_pose(bar=BAR):
    """The shipped bar grasp: approach +y into the cabinet, fingers closing across x."""
    approach, closing = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(closing, approach), closing, approach], axis=1)
    T[:3, 3] = bar
    return T


def random_rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_swing_pose_keeps_radius_height_and_composes():
    T = grasp_pose()
    a, b = 0.7, 1.05
    Ta = ds.swing_pose(T, HINGE, a)
    r0 = np.linalg.norm(T[:2, 3] - HINGE)
    assert np.linalg.norm(Ta[:2, 3] - HINGE) == pytest.approx(r0)
    assert Ta[2, 3] == pytest.approx(T[2, 3])
    assert np.allclose(ds.swing_pose(Ta, HINGE, b), ds.swing_pose(T, HINGE, a + b))
    assert np.allclose(ds.swing_pose(T, HINGE, 0.0), T)


def test_door_frames_are_bimbos_with_the_hand_fixed_to_the_bar():
    G = grasp_pose()
    _, g = ds.door_frames(G, HINGE, 1.2)
    assert np.allclose(g, ds.swing_pose(G, HINGE, 1.2))


def test_door_frames_hand_turn_sets_only_the_orientation():
    G = grasp_pose()
    _, g_fixed = ds.door_frames(G, HINGE, 1.2)
    _, g_world = ds.door_frames(G, HINGE, 1.2, hand_turn=0.0)
    _, g_half = ds.door_frames(G, HINGE, 1.2, hand_turn=0.6)
    assert np.allclose(g_world[:3, 3], g_fixed[:3, 3])
    assert np.allclose(g_world[:3, :3], G[:3, :3])
    assert np.allclose(g_half[:3, :3], ds.rot_z(0.6) @ G[:3, :3])


def test_pre_grasp_is_back_along_the_approach():
    G = grasp_pose()
    pre, g = ds.door_frames(G, HINGE, 0.9, pre_gap=0.15)
    assert np.allclose(pre[:3, :3], g[:3, :3])
    assert np.allclose(g[:3, 3] - pre[:3, 3], 0.15 * g[:3, 2])


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("theta", [0.0, 0.4, 1.2, 1.75])
def test_angle_of_reads_the_door_back_from_the_hand(sign, theta):
    G = grasp_pose()
    _, g = ds.door_frames(G, HINGE, sign * theta, hand_turn=0.0)
    assert ds.angle_of(HINGE, G[:2, 3], g[:2, 3], sign) == pytest.approx(theta, abs=1e-9)


def test_rotvec_and_pose_error_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = random_rotation(rng)
        v = ds.rotvec(R)
        angle = np.linalg.norm(v)
        assert angle <= math.pi + 1e-9
        k = v / angle
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        assert np.allclose(np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K, R)
    A, B = np.eye(4), np.eye(4)
    B[:3, :3] = ds.rot_z(-0.4)
    B[:3, 3] = [0.0, 0.2, -0.1]
    dp, dr = ds.pose_error(A, B)
    assert np.allclose(dp, [0.0, 0.2, -0.1]) and np.allclose(dr, [0.0, 0.0, -0.4])


def test_rotvec_small_and_half_turn():
    assert np.allclose(ds.rotvec(ds.rot_z(1e-7)), [0.0, 0.0, 1e-7], atol=1e-12)
    v = ds.rotvec(ds.rot_z(math.pi))
    assert np.linalg.norm(v) == pytest.approx(math.pi) and abs(v[2]) == pytest.approx(math.pi)


def planar_arm(q, links=(0.4, 0.35, 0.2)):
    """End pose of a planar 3R arm (z rotation), and its 6 x 3 Jacobian."""
    x = y = a = 0.0
    pts = [(0.0, 0.0)]
    for qi, li in zip(q, links):
        a += qi
        x += li * math.cos(a)
        y += li * math.sin(a)
        pts.append((x, y))
    T = np.eye(4)
    T[:3, :3] = ds.rot_z(a)
    T[:3, 3] = [x, y, 0.0]
    J = np.zeros((6, 3))
    for i in range(3):
        J[0, i] = -(y - pts[i][1])
        J[1, i] = x - pts[i][0]
        J[5, i] = 1.0
    return T, J


def test_dls_step_converges_and_respects_the_joint_cap():
    q = np.array([0.3, 0.5, -0.4])
    goal, _ = planar_arm([1.0, -0.3, 0.6])
    for _ in range(200):
        T, J = planar_arm(q)
        dp, dr = ds.pose_error(T, goal)
        dq = ds.dls_step(J, dp, dr, cap_q=0.05)
        assert np.max(np.abs(dq)) <= 0.05 + 1e-12
        q = q + dq
    T, _ = planar_arm(q)
    dp, dr = ds.pose_error(T, goal)
    assert np.linalg.norm(dp) < 1e-3 and np.linalg.norm(dr) < 1e-3


def test_unicycle_hold_keeps_a_bar_on_its_arc_at_the_grasp_lever():
    """The creep: a point riding the hinge's circle (the bar at BIMBO's 0.36 rad/s) stays where
    the grasp put it in the base frame, though the base can only drive and turn."""
    dt, rate = 0.05, 0.36
    base, yaw = np.array([2.30, -1.30]), math.pi / 2
    G = grasp_pose()
    f = np.array([math.cos(yaw), math.sin(yaw)])
    l = np.array([-math.sin(yaw), math.cos(yaw)])
    lever = (float((G[:2, 3] - base) @ f), float((G[:2, 3] - base) @ l))
    prev = G[:2, 3]
    worst = 0.0
    for k in range(1, int(1.75 / (rate * dt)) + 1):
        _, g = ds.door_frames(G, HINGE, rate * dt * k)
        p = g[:2, 3]
        v, w, ax, ay = ds.unicycle_hold_cmd(p, (p - prev) / dt, base, yaw, lever,
                                            v_max=1.0, w_max=3.0)
        prev = p
        base = base + v * dt * np.array([math.cos(yaw), math.sin(yaw)])
        yaw += w * dt
        worst = max(worst, abs(ax - lever[0]), abs(ay - lever[1]))
    assert worst < 0.02
    # the leaf turned +100 deg CCW; the base carrying the bar sideways turned the other way
    assert yaw < math.pi / 2


def test_unicycle_hold_floors_the_forward_lever():
    v, w, ax, ay = ds.unicycle_hold_cmd([0.0, 0.5], [0.1, 0.0], [0.0, 0.0], 0.0, (0.0, 0.5),
                                        w_max=10.0)
    assert ax == pytest.approx(0.0) and abs(w) <= 0.1 / 0.3 + 1e-9


def test_leaf_segment_and_seg_dist():
    edge0 = HINGE + (BAR[:2] - HINGE) * 1.12
    a, b = ds.leaf_segment(HINGE, edge0, math.pi / 2)
    assert np.allclose(a, HINGE)
    assert np.linalg.norm(b - HINGE) == pytest.approx(np.linalg.norm(edge0 - HINGE))
    assert ds.seg_dist(HINGE + np.array([0.1, -0.2]), a, b) == pytest.approx(
        abs(np.cross(b - a, np.array([0.1, -0.2])) / np.linalg.norm(b - a)))


def leaf_at(theta):
    """Outward normal and panel segment of cab_main's right leaf open by theta."""
    n = ds.rot2(theta) @ np.array([0.0, -1.0])
    edge0 = HINGE + (BAR[:2] - HINGE) * (1.0 + 0.05 / np.linalg.norm(BAR[:2] - HINGE))
    return n, ds.leaf_segment(HINGE, edge0, theta)


def test_clear_backs_out_along_the_approach_when_the_hand_faces_the_leaf():
    theta = 1.75
    n, (a, b) = leaf_at(theta)
    G = grasp_pose()
    _, g = ds.door_frames(G, HINGE, theta)              # BIMBO: the hand turned with the bar
    d, rule = ds.clear_direction(g[:3, 2], n)
    assert rule == "back" and np.allclose(d[:2], n, atol=1e-9)
    assert ds.path_keeps_off(g[:2, 3], g[:2, 3] + 0.15 * d[:2], a, b)


def test_clear_slides_along_the_face_when_the_hand_kept_its_heading():
    theta = 1.75
    n, (a, b) = leaf_at(theta)
    G = grasp_pose()
    _, g = ds.door_frames(G, HINGE, theta, hand_turn=0.0)   # the shipped end state
    back = -g[:3, 2]
    assert back[:2] @ n < 0          # straight back would push into the face (-0.17)
    d, rule = ds.clear_direction(g[:3, 2], n)
    assert rule == "plane" and d[:2] @ n > 0
    assert ds.path_keeps_off(g[:2, 3], g[:2, 3] + 0.15 * d[:2], a, b)
    mid = (a + b) / 2 + 0.05 * n                        # a hand 5 cm off the face, mid-leaf
    assert not ds.path_keeps_off(mid, mid - 0.15 * n, a, b)
