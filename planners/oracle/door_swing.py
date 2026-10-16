"""Door-swing geometry after BIMBO_EvoX's fridge door (task 62), for a Fetch. Numpy only.

BIMBO_EvoX opens the fridge of task 62 (`tasks/62_halve_an_egg/planners/fair/phases/door.py`,
`swing.py`, `fridgegeo.py`, branch `agent`) like this: the hand closes on the handle bar, then
EVERY control step the commanded swing angle advances by a fixed lead (`sw += sign * 0.012`), the
hand is servoed to the bar's place on the arc at that angle (`door_frames(geo, sw)`: the grasp
frame carried round the hinge), the base creeps by velocity only so that point stays in the arm's
reach (`follow_base`), the assisted-grasp weld holds the bar, and the door angle is measured from
the hand while it holds the bar (`DoorGeo.angle_of`). After the release the open hand leaves on a
straight path checked against the panel (`swing.path_clear`).

These are the pieces of that which do not need a simulator:

* `swing_pose` / `door_frames` — a pose carried round a vertical hinge (BIMBO's `door_frames`).
  The hand's own yaw may turn by less than the door (`hand_turn`): BIMBO's R1Pro strafes, a
  Fetch base cannot, see `planners/cabinet_door_bimbo.py`.
* `angle_of` — the door angle at which the handle stands where the hand is (`DoorGeo.angle_of`).
* `pose_error` / `dls_step` — one damped-least-squares step of a hand servo (BIMBO's
  `ArmServo.step` without a joint target, i.e. resolved-rate control).
* `unicycle_hold_cmd` — the base creep for a base that has no lateral velocity: the exact
  (v, omega) that keep a moving world point at a fixed place in the base frame (the K103 arc
  law's inverse, `robots.fetch.base_yaw.arc_pull_cmd`, with a feedback term).
* `leaf_segment`, `seg_dist`, `path_keeps_off`, `clear_direction` — the panel as a segment in
  the plane and the straight path the open hand leaves by (BIMBO's `seg_dist`, `path_clear`).

All poses are 4x4 homogeneous matrices in the world frame; angles in radians, lengths in metres.
"""

from __future__ import annotations

import math

import numpy as np


def rot_z(a: float) -> np.ndarray:
    """3x3 rotation about +z by `a` (counter-clockwise seen from above).

    Example:
        >>> np.round(rot_z(math.pi / 2) @ np.array([1.0, 0.0, 0.0]), 6).tolist()
        [0.0, 1.0, 0.0]
    """
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot2(a: float) -> np.ndarray:
    """2x2 rotation by `a`."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def swing_pose(T, hinge_xy, angle: float) -> np.ndarray:
    """The pose `T` carried about the vertical axis through `hinge_xy` by `angle` (CCW > 0).

    Position and orientation both turn: what a body rigidly fixed to the door does.

    Example:
        >>> T = np.eye(4); T[:3, 3] = [1.0, 0.0, 0.5]
        >>> np.round(swing_pose(T, [0.0, 0.0], math.pi / 2)[:3, 3], 6).tolist()
        [0.0, 1.0, 0.5]
    """
    T = np.asarray(T, dtype=np.float64)
    Rz = rot_z(float(angle))
    h = np.array([float(hinge_xy[0]), float(hinge_xy[1]), 0.0])
    out = T.copy()
    out[:3, :3] = Rz @ T[:3, :3]
    out[:3, 3] = h + Rz @ (T[:3, 3] - h)
    return out


def door_frames(grasp_T, hinge_xy, turn: float, *, hand_turn: float | None = None,
                pre_gap: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(pre-grasp, grasp) hand poses with the door swung by `turn` from where `grasp_T` was taken.

    BIMBO's `door_frames(geo, sw)`: the grasp point rides the hinge's circle with the door. The
    hand's orientation turns about +z by `hand_turn` (None = `turn`, BIMBO's frames, the hand
    fixed to the bar; 0 = the hand keeps its world orientation). The pre-grasp lies `pre_gap`
    back along the hand's own approach axis (+z of the TCP frame, `build_grasp_pose`).

    Args:
        grasp_T: the hand pose on the bar at the reference door angle.
        hinge_xy: the hinge axis, world xy.
        turn: signed yaw of the door from the reference angle (sense * open_dir * delta).
        hand_turn: signed yaw of the hand; None = `turn`.
        pre_gap: metres back along the approach for the pre-grasp.

    Returns:
        `(pre, grasp)`, two 4x4 poses.

    Example:
        >>> G = np.eye(4); G[:3, 3] = [-0.4, 0.0, 1.5]
        >>> pre, g = door_frames(G, [0.0, 0.0], math.pi / 2, hand_turn=0.0, pre_gap=0.1)
        >>> (np.round(g[:3, 3], 6) + 0.0).tolist(), np.allclose(g[:3, :3], np.eye(3))
        ([0.0, -0.4, 1.5], True)
        >>> (np.round(pre[:3, 3], 6) + 0.0).tolist()
        [0.0, -0.4, 1.4]
    """
    G = np.asarray(grasp_T, dtype=np.float64)
    grasp = swing_pose(G, hinge_xy, float(turn))
    yaw = float(turn) if hand_turn is None else float(hand_turn)
    grasp[:3, :3] = rot_z(yaw) @ G[:3, :3]
    pre = grasp.copy()
    pre[:3, 3] = grasp[:3, 3] - float(pre_gap) * grasp[:3, 2]
    return pre, grasp


def angle_of(hinge_xy, ref_xy, xy, sign: float = 1.0) -> float:
    """The door angle (in the opening sense) at which a handle first at `ref_xy` stands at `xy`.

    BIMBO's `DoorGeo.angle_of`: the signed angle from (ref - hinge) to (xy - hinge), times the
    opening sign (+1 when opening turns the door CCW seen from above).

    Example:
        >>> round(angle_of([0, 0], [-1, 0], [0, -1], sign=1.0), 4)
        1.5708
        >>> round(angle_of([0, 0], [1, 0], [0, -1], sign=-1.0), 4)
        1.5708
    """
    h = np.asarray(hinge_xy, dtype=np.float64).reshape(-1)[:2]
    a = np.asarray(ref_xy, dtype=np.float64).reshape(-1)[:2] - h
    b = np.asarray(xy, dtype=np.float64).reshape(-1)[:2] - h
    return float(sign) * math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))


def rotvec(R) -> np.ndarray:
    """Rotation vector (axis * angle) of a 3x3 rotation matrix.

    Example:
        >>> np.round(rotvec(rot_z(0.3)), 6).tolist()
        [0.0, 0.0, 0.3]
    """
    R = np.asarray(R, dtype=np.float64)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = 0.5 * float(np.linalg.norm(w))                 # sin(angle)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.atan2(s, c)                           # well conditioned at small angles
    if math.pi - angle < 1e-5:
        # near pi the antisymmetric part vanishes: the axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(M)))
        axis = M[:, k] / math.sqrt(max(M[k, k], 1e-12))
        return axis / np.linalg.norm(axis) * angle
    if s < 1e-12:
        return 0.5 * w
    return w / (2.0 * s) * angle


def pose_error(T_now, T_goal) -> tuple[np.ndarray, np.ndarray]:
    """(position error, rotation-vector error) taking `T_now` to `T_goal`, both in the world frame.

    Example:
        >>> A = np.eye(4); B = np.eye(4); B[:3, :3] = rot_z(0.2); B[:3, 3] = [0.1, 0, 0]
        >>> dp, dr = pose_error(A, B)
        >>> np.round(dp, 6).tolist(), np.round(dr, 6).tolist()
        ([0.1, 0.0, 0.0], [0.0, 0.0, 0.2])
    """
    A = np.asarray(T_now, dtype=np.float64)
    B = np.asarray(T_goal, dtype=np.float64)
    return B[:3, 3] - A[:3, 3], rotvec(B[:3, :3] @ A[:3, :3].T)


def _clip_norm(v, cap: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v if n <= cap or n < 1e-12 else v * (cap / n)


def dls_step(J, dp, dr, *, k_pos: float = 0.5, k_rot: float = 0.5, cap_pos: float = 0.03,
             cap_rot: float = 0.08, damping: float = 2e-3, cap_q: float = 0.06,
             w_rot: float = 0.5) -> np.ndarray:
    """One damped-least-squares joint step toward a pose error (BIMBO's `ArmServo.step`).

    The task-space step is the error scaled by its gain and capped in norm; rotation rows are
    weighted by `w_rot` (metres per radian) so a turn does not swamp a reach. The joint step is
    `J^T (J J^T + damping I)^-1 d` on the weighted rows, scaled down as a whole so no joint moves
    more than `cap_q` (a uniform scale keeps the direction of the step).

    Args:
        J: 6 x n Jacobian, rows [linear; angular], world-aligned at the controlled point.
        dp, dr: position error (m) and rotation-vector error (rad), world frame.

    Returns:
        dq, shape (n,).

    Example:
        >>> J = np.zeros((6, 6)); J[:6, :6] = np.eye(6)
        >>> np.round(dls_step(J, [0.01, 0, 0], [0, 0, 0], k_pos=1.0), 4).tolist()[:3]
        [0.01, 0.0, 0.0]
    """
    J = np.asarray(J, dtype=np.float64)
    d = np.r_[_clip_norm(k_pos * np.asarray(dp, dtype=np.float64), cap_pos),
              _clip_norm(k_rot * np.asarray(dr, dtype=np.float64), cap_rot)]
    W = np.diag([1.0, 1.0, 1.0, w_rot, w_rot, w_rot])
    Jw, dw = W @ J, W @ d
    dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + float(damping) * np.eye(6), dw)
    m = float(np.max(np.abs(dq))) if dq.size else 0.0
    if m > cap_q:
        dq = dq * (cap_q / m)
    return dq


def unicycle_hold_cmd(point_xy, point_vel_xy, base_xy, yaw: float, lever_ref, *,
                      gain: float = 1.0, v_max: float = 0.15, w_max: float = 0.6,
                      min_ax: float = 0.3) -> tuple[float, float, float, float]:
    """(v, omega, a_x, a_y) that keep a moving world point at `lever_ref` in the base frame.

    BIMBO's `follow_base` creeps a holonomic base (x and y velocities) so the hand's goal stays
    in front of the chest. A Fetch base has a forward speed and a yaw rate only. With the point at
    body coordinates r = (a_x, a_y) and moving at `point_vel`, the body-frame rate of r is
    `R^T pdot - (v - omega a_y, omega a_x)`; asking for `-gain (r - lever_ref)` gives the exact
    solution below (the K103 arc law, `arc_pull_cmd`, is its feed-forward part). `a_x` is
    floored at `min_ax` so a point that drifts beside the base cannot command a spin.

    Args:
        point_xy, point_vel_xy: the point and its world velocity (m, m/s).
        base_xy, yaw: the base pose (root x, y and yaw).
        lever_ref: where the point should stand in the base frame, (forward, left).
        gain: feedback rate, 1/s.
        v_max, w_max: caps, m/s and rad/s.

    Returns:
        `(v, omega, a_x, a_y)`: commands in physical units and the point's current lever.

    Example:
        >>> v, w, ax, ay = unicycle_hold_cmd([0, 1], [0, 0], [0, 0], math.pi / 2, (1.0, 0.0))
        >>> round(v, 6), round(w, 6), round(ax, 6), round(ay, 6)
        (0.0, 0.0, 1.0, 0.0)
        >>> v, w, *_ = unicycle_hold_cmd([0, 1], [0, 0.1], [0, 0], math.pi / 2, (1.0, 0.0))
        >>> round(v, 6), round(w, 6)
        (0.1, 0.0)
    """
    p = np.asarray(point_xy, dtype=np.float64).reshape(-1)[:2]
    u = np.asarray(point_vel_xy, dtype=np.float64).reshape(-1)[:2]
    b = np.asarray(base_xy, dtype=np.float64).reshape(-1)[:2]
    f = np.array([math.cos(yaw), math.sin(yaw)])
    l = np.array([-math.sin(yaw), math.cos(yaw)])
    r = p - b
    ax, ay = float(r @ f), float(r @ l)
    ex, ey = ax - float(lever_ref[0]), ay - float(lever_ref[1])
    ux = float(u @ f) + gain * ex
    uy = float(u @ l) + gain * ey
    w = uy / max(ax, min_ax)
    v = ux + w * ay
    return (float(np.clip(v, -v_max, v_max)), float(np.clip(w, -w_max, w_max)), ax, ay)


def leaf_segment(hinge_xy, edge0_xy, turn: float) -> tuple[np.ndarray, np.ndarray]:
    """The leaf in the plane: hinge and free edge, the edge (closed-leaf position `edge0_xy`)
    carried round the hinge by the signed `turn`.

    Example:
        >>> a, b = leaf_segment([0, 0], [-0.5, 0], math.pi / 2)
        >>> (np.round(b, 6) + 0.0).tolist()
        [0.0, -0.5]
    """
    h = np.asarray(hinge_xy, dtype=np.float64).reshape(-1)[:2]
    e = np.asarray(edge0_xy, dtype=np.float64).reshape(-1)[:2] - h
    return h.copy(), h + rot2(float(turn)) @ e


def seg_dist(p, a, b) -> float:
    """Distance from the point `p` to the segment a-b (BIMBO's `swing.seg_dist`).

    Example:
        >>> seg_dist([0, 1], [-1, 0], [1, 0]), seg_dist([2, 0], [-1, 0], [1, 0])
        (1.0, 1.0)
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)[:2]
    a = np.asarray(a, dtype=np.float64).reshape(-1)[:2]
    b = np.asarray(b, dtype=np.float64).reshape(-1)[:2]
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def path_keeps_off(a, b, seg_a, seg_b, *, n: int = 12, slack: float = 1e-3) -> bool:
    """True when no point of the straight path a -> b is nearer the panel than `a` is.

    BIMBO's `path_clear` asks for a fixed gap along the whole path; a hand that has just let
    go of a bar starts a few centimetres from the panel by construction, so the test here is
    that the path never closes on it (within `slack`).

    Example:
        >>> path_keeps_off([0, 0.05], [0, 0.20], [-1, 0], [1, 0])
        True
        >>> path_keeps_off([0, 0.05], [0.3, 0.02], [-1, 0], [1, 0])
        False
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)[:2]
    b = np.asarray(b, dtype=np.float64).reshape(-1)[:2]
    d0 = seg_dist(a, seg_a, seg_b)
    return all(seg_dist(a + (b - a) * float(t), seg_a, seg_b) >= d0 - slack
               for t in np.linspace(0.0, 1.0, int(n)))


def clear_direction(hand_z, normal_xy, *, min_off: float = 0.1,
                    lift_off: float = 0.25) -> tuple[np.ndarray, str]:
    """The horizontal direction the open hand leaves the bar by, and which rule chose it.

    `back`: straight back along the hand's own approach axis (`-hand_z`), the grasp's approach
    reversed, when that also leaves the leaf's face (`-hand_z . n >= min_off`). Otherwise
    `plane`: the hand's axis projected on the leaf's plane plus `lift_off` of the outward
    normal — the slide off the bar along the face that the shipped retreat makes
    (`RETREAT_ALONG_FACE`), tilted a little away from the face.

    Args:
        hand_z: the hand's approach axis (TCP +z), world.
        normal_xy: the leaf's outward face normal, world xy.

    Returns:
        `(d, rule)`: a horizontal unit 3-vector and "back" or "plane".

    Example:
        >>> d, rule = clear_direction([0, 1, 0], [0, -1])
        >>> rule, (np.round(d, 6) + 0.0).tolist()
        ('back', [0.0, -1.0, 0.0])
        >>> d, rule = clear_direction([0, 1, 0], [1, 0])
        >>> rule, np.round(d, 3).tolist()
        ('plane', [0.243, -0.97, 0.0])
    """
    hz = np.asarray(hand_z, dtype=np.float64).reshape(-1)[:3]
    n = np.asarray(normal_xy, dtype=np.float64).reshape(-1)[:2]
    n3 = np.array([n[0], n[1], 0.0]) / max(float(np.linalg.norm(n)), 1e-12)
    back = -np.array([hz[0], hz[1], 0.0])
    nb = float(np.linalg.norm(back))
    back = back / nb if nb > 1e-9 else n3.copy()   # a vertical hand axis: off the face
    off = float(back @ n3)
    if off >= float(min_off):
        return back, "back"
    plane = back - off * n3
    npl = float(np.linalg.norm(plane))
    plane = plane / npl if npl > 1e-9 else np.zeros(3)
    d = plane + float(lift_off) * n3
    return d / max(float(np.linalg.norm(d)), 1e-12), "plane"
