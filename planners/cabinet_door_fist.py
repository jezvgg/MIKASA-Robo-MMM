"""A left wall-cabinet leaf cracked by its bar and swung open from inside by the fist (EXPERIMENTAL).

The owner's ask for cab_1, the west wall unit (2026-10-24), after b1k_rl_guys / BIMBO_EvoX: open
the leaf a little by the handle so that the hand fits, put the hand into the gap and open the
leaf the rest of the way with the fist, the way the round already closes leaves with the fist.

Why cab_1 needs it: the shipped opening (`cabinet_door_bimbo.open_the_door`) swings every leaf on
the bar to `DOOR_OPEN_TARGET` (1.75 rad). A left leaf that open turns its outer face (the side
the closing push stands on) toward its hinge's side, and cab_1's hinge stands 0.265 m from the
room's west wall: the push would have to stand inside the wall (W24, `CAB_1.close_docks == ()`),
so W24 parked the leaf at the wall instead and the round never closed it. Opened from inside
by the fist, the leaf stops where the fist stops: `open_rad` (1.30) is past the reveal (1.2)
and short of the angle where the push side turns into the wall, so the round can push it shut.

The stage:

1. crack — `cabinet_door_bimbo.open_the_door(target_rad=crack_rad)`: the shipped dock, bar
   grasp, servoed swing on the bar, release and clear, stopped at `crack_rad` (0.45 rad, a gap
   of ~0.21 m at the free edge) with the base kept at the grasp's lever.
2. fist — the hand closes in free air (no leaf between the pads, so the assisted grasp has
   nothing to weld).
3. insert — a planned move (the doors in the planning world) to a pre-insert pose past the
   leaf's free edge, the fist pointing along the leaf toward the hinge (`approach = -u`), then
   a servoed stroke along the leaf into the wedge behind its inner face, `IN_OFFSET_M` off the
   face and `IN_ALONG_M` in from the free edge.
4. push — the fist's frame is carried round the hinge with the leaf (the hand turns with it),
   the commanded angle at most `LEAD_MAX` ahead of the measured leaf, so the fist presses the
   inner face outward; the base stands still. Stops at `open_rad`, on a stall, or after
   `PUSH_MAX_STEPS`.
5. withdraw — the fist slides back along the leaf, out past the free edge, and off.

Same return contract as the other opening stages: -1 only for a refusal before the fingers first
closed on the bar; afterwards the last 5-tuple, the caller judging the leaf by `door_rad_now`.
"""

from __future__ import annotations

import math
import os

import numpy as np
import sapien

from planners import cabinet_door_bimbo as _bimbo
from planners import cabinet_retrieval_planner as crp
from planners.oracle import door_swing as ds
from planners.oracle import oracle_common as common

WHO = "cabinet_door_fist"


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


CRACK_RAD = _f("MIKASA_FD_CRACK_RAD", 0.45)
"""How far the bar swing opens the leaf before the fist goes in: a ~0.21 m chord at the free edge."""
OPEN_RAD = _f("MIKASA_FD_OPEN_RAD", 1.30)
"""Where the fist leaves the leaf (`my_scenes.cabinet_search.CAB_1_FIST_OPEN_RAD`)."""
IN_ALONG_M = _f("MIKASA_FD_IN_ALONG_M", 0.08)
"""How far in from the free edge, along the leaf, the fist presses."""
IN_OFFSET_M = _f("MIKASA_FD_IN_OFFSET_M", 0.09)
"""The fist centre off the leaf's inner face where it goes in. `tools/cab5/reach_scan.py` (mplib IK
with the scene, the fingers shut, 24 stances x 2 torsos): at 0.06 no pose is collision-free at any
angle (the palm meets the leaf); at 0.09 the pitched fist is reachable from ~95 % of them. The push
closes the gap with the lead (LEAD_MAX)."""
PRE_PAST_M = _f("MIKASA_FD_PRE_PAST_M", 0.14)
"""The pre-insert pose stands this far past the free edge, along the leaf."""
PUSH_Z = _f("MIKASA_FD_PUSH_Z", 1.47)
"""Height of the fist: 11 cm over the leaf's bottom edge (1.391), under the bar (1.592): at
1.62 the servo could not reach the press point from the insert stance (19 cm short, two seeds)."""
LEAD_MAX = _f("MIKASA_FD_LEAD_MAX", 0.20)
"""Rad the commanded angle may run ahead of the measured leaf: ~0.12 rad closes IN_OFFSET_M's gap
at the press radius (0.40 m), the rest is the press."""
LEAD_RATE = _f("MIKASA_FD_LEAD_RATE", 0.30)
"""Rad/s the commanded angle advances."""
PUSH_MAX_STEPS = int(_f("MIKASA_FD_PUSH_MAX_STEPS", 500))
INSERT_MAX_STEPS = 80
WITHDRAW_M = _f("MIKASA_FD_WITHDRAW_M", 0.25)
STANCE = (_f("MIKASA_FD_STANCE_X", 0.96), _f("MIKASA_FD_STANCE_Y", -1.11))
"""Where the base stands for both sweeps, turning in place only: the roll scan's lowest-winding
stances for the inner press (0.95, -1.10..-1.25) and the outer push (1.0, -1.15) meet here. From the
handle dock (facing +y) the pre-insert pose has no IK (shoulder joint limit, two seeds)."""
STALL_STEPS = 40
STALL_EPS = 0.01
FIST_PITCH = math.radians(_f("MIKASA_FD_FIST_PITCH_DEG", 90.0))
"""The fist's pitch up from pointing along the leaf toward the hinge; 90 = straight up. It goes in
from under the leaf's bottom edge along its own axis. The reach scan with the roll joints
(`tools/cab5/reach_scan.py`): pitched 70 deg the press is reachable from the most stances
(0.92-0.96) but the IK winds the rolls 0.87-2.1 rad from the reset posture, and the episode's roll
window (span <= pi, D2) then had no room left for a later round's tuck (two seeds); straight up
0.63-0.87 on both faces from the same stance."""
HAND_TURN = _f("MIKASA_FD_HAND_TURN", 0.0)
"""The fraction of the leaf's turn the fist's yaw follows in the closing sweep. Pointing up, a yaw
is pure wrist roll; 0 keeps it out of the roll window (shut on 6/6)."""
HAND_TURN_OPEN = _f("MIKASA_FD_HAND_TURN_OPEN", 1.0)
"""The same for the opening push. At 0 the push stalled at 1.21-1.27 rad pressing ~450 N on six of
six seeds (the fist turned against the leaf); turned with the leaf (1) it reached 1.31 at ~2 N."""
FIST_UP = True
"""How the fist meets the leaf. "up" (default): the hand points straight up and rises into the
gap from under the leaf's bottom edge (1.391), so the wrist and forearm stay below the leaf (the
BIMBO task-62 demos drop the hand into the fridge door's gap; a wall cabinet is reached from
below instead). "along": the hand points along the leaf toward the hinge with the forearm past
the free edge — measured on four seeds: the press point is never reached (the TCP stops 6-14 cm
short, the planned insert refuses) and the leaf stalls at 0.42-0.66 rad."""
PUSH_Z_OPEN = _f("MIKASA_FD_PUSH_Z_OPEN", 1.58)
"""The opening press's fist height. At PUSH_Z (1.47) the palm's top stood ~1 cm into the leaf's
bottom edge (1.391): the push stalled at 1.19-1.25 rad pressing 420-480 N on eight of eight seeds,
the leaf even turning back (1.25 -> 1.19, gripper_link the only contact)."""
UNDER_BACK_M = _f("MIKASA_FD_UNDER_BACK_M", 0.16)
"""The pre-insert pose stands this far back along the fist's axis: 0.15 m lower (z 1.35, just
under the leaf's bottom edge at 1.391) and 0.05 m toward the free edge. At 0.28 (z 1.21) it had no
IK and its screw put the upper arm into the counter (four seeds)."""
YAW_GAIN = 2.0
YAW_W_MAX = 0.4

FIST_STEM = "cab_1_main_group"
"""The one leaf this stage serves (`my_scenes.cabinet_search.CAB_1_FIST`)."""
CLOSE_DOCKS = ((0.70, -1.30), (0.60, -1.40), (0.80, -1.20), (0.55, -1.25))
"""cab_1's closing ladder when the fist leaves it at OPEN_RAD: base xy east of the west wall and
south of the leaf, the push point on the outer face west of the free edge. GUESSES, unmeasured;
W24's mirror of the family's rungs (x 0.73/0.53/0.93 about this hinge on the WRONG side) is
not used."""


def say(env, stage: str, **extra) -> None:
    common.say(env, WHO, stage, **extra)


def _T(pose) -> np.ndarray:
    return _bimbo._T(pose)


def _fist_closing(task, approach, u, nin) -> np.ndarray:
    """The fist's closing axis: across the leaf or along it, either sign, whichever is nearest the
    hand's now (`_like_now`); orthogonal to `approach` (pointing up, all four are)."""
    cands = []
    for c in (nin, -nin, u, -u):
        c3 = np.array([c[0], c[1], 0.0])
        c3 = c3 - (c3 @ approach) * approach
        if np.linalg.norm(c3) > 1e-6:
            cands.append(c3 / np.linalg.norm(c3))
    return _like_now(task, cands)


class Leaf:
    """The leaf's frame in the world: the hinge, the opening sense, the free edge's radius."""

    def __init__(self, task, door):
        self.task, self.door = task, door
        anchor, sense = common.hinge_anchor(task, crp._art_key(task, door), door.hinge)
        self.anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)[:2]
        self.sense = int(sense)
        self.s = float(sense) * float(door.open_dir)              # yaw sense of opening
        bar0 = np.asarray(door.handle_bar, dtype=np.float64)[:2]
        r_bar = max(float(np.linalg.norm(bar0 - self.anchor)), 1e-6)
        edge0 = self.anchor + (bar0 - self.anchor) * (1.0 + _bimbo.LEAF_EDGE_PAST_BAR / r_bar)
        # The shut leaf lies along world x toward `panel_dir` with its inner face toward +y
        # (`push_frame`'s convention); the bar's 3.5 cm of depth is not the leaf's direction.
        self.u0 = np.array([float(door.panel_dir), 0.0])
        self.L = float(abs(edge0[0] - self.anchor[0]))

    def rad(self) -> float:
        return float(crp.door_rad_now(self.task, self.door))

    def u(self, rad: float) -> np.ndarray:
        """Unit vector hinge -> free edge at opening `rad`."""
        return ds.rot2(self.s * float(rad)) @ self.u0

    def inner(self, rad: float) -> np.ndarray:
        """Unit normal of the leaf's INNER face (the cabinet side when shut) at `rad`."""
        return ds.rot2(self.s * float(rad)) @ np.array([0.0, 1.0])

    def point(self, rad: float, along_from_edge: float, off_inner: float) -> np.ndarray:
        return self.anchor + self.u(rad) * (self.L - along_from_edge) + self.inner(rad) * off_inner

    def fist_pose(self, task, rad: float, along_from_edge: float, off_inner: float,
                  z: float = PUSH_Z) -> np.ndarray:
        """The fist at the leaf, fingers closing across it (so a finger's outer face meets the
        inner face): pointing up (FIST_UP) or along the leaf toward the hinge."""
        xy = self.point(rad, along_from_edge, off_inner)
        u, nin = self.u(rad), self.inner(rad)
        approach = np.array([-math.cos(FIST_PITCH) * u[0], -math.cos(FIST_PITCH) * u[1],
                             math.sin(FIST_PITCH)])
        closing = _fist_closing(task, approach, u, nin)
        pose = task.agent.build_grasp_pose(approach, closing, np.array([xy[0], xy[1], z]))
        return _T(pose)


IK_SEEDS = int(_f("MIKASA_FD_IK_SEEDS", 80))
"""mplib IK restarts for the fist's planned moves (static_manipulation's default is 20). The
collection solver keeps, among the IK goals found, the one that grows the episode's roll window the
least; with 20 restarts the up-fist's goal wound the rolls 2.1-3.0 rad from the reset posture (six
seeds), past what the later rounds' moves need, where the reach scan found 0.63-0.87 rad goals."""


def _move(env, planner, T, *, stage: str, tries: int = 3):
    """`arm_move` with IK_SEEDS restarts: a refused draw costs no steps; -1 after the last."""
    pose = sapien.Pose(p=T[:3, 3], q=_quat(T))
    res = -1
    for i in range(int(tries)):
        res = planner.static_manipulation(pose, n_init_qpos=IK_SEEDS)
        if res != -1:
            return res
        if i + 1 < tries:
            say(env, f"{stage}: plan refused, one more draw (RRT is randomized)")
    return res


LOW_ROLL_SEEDS = int(_f("MIKASA_FD_LOW_ROLL_SEEDS", 200))


def _low_roll_joints(planner, task, T):
    """The arm joints that put the TCP at `T` (base and torso held) and grow the episode's roll
    window (the collection solver's `_roll_low`/`_roll_high`, D2) the least, from LOW_ROLL_SEEDS IK
    restarts: `({joint: value}, growth_rad)`, or None without IK. Planned as a joint move, the
    goal is this one, not whichever of `static_manipulation`'s few the path planner reaches first
    (those wound the rolls 2.1-3.0 rad, two runs of six seeds)."""
    import mplib
    from robots.fetch.utils import unwrap_toward

    p = planner.planner
    robot = task.agent.robot
    cur = robot.get_qpos()[0].cpu().numpy().astype(np.float64)
    cur_f = p.fold_qpos(cur)
    goal_b = p._transform_goal_to_wrt_base(mplib.Pose(p=T[:3, 3], q=_quat(T)))
    mask = [True, True, True, True] + [False] * (len(cur) - 4)
    try:
        status, sols = p.IK(goal_b, cur_f, mask, n_init_qpos=LOW_ROLL_SEEDS)
    except Exception:
        return None
    if status != "Success" or sols is None or len(np.atleast_2d(sols)) == 0:
        return None
    idx = list(getattr(planner, "_roll_indices", []))
    low = np.asarray(getattr(planner, "_roll_low", np.zeros(len(idx))), dtype=np.float64)
    high = np.asarray(getattr(planner, "_roll_high", np.zeros(len(idx))), dtype=np.float64)
    best = None
    for g in np.atleast_2d(sols):
        g = unwrap_toward(g, cur_f, p.joint_limits)
        r = np.asarray(g, dtype=np.float64)[idx]
        if np.any(np.abs(r) > np.pi - 0.02):
            continue
        grow = float(np.max(np.maximum(high, r) - np.minimum(low, r) - (high - low))) if idx else 0.0
        if best is None or grow < best[0]:
            best = (grow, g)
    if best is None:
        return None
    jm = robot.active_joints_map
    names = list(task.agent.controller.controllers["arm"].config.joint_names)
    return {n: float(best[1][int(jm[n].active_index[0])]) for n in names}, best[0]


def _reach_low_roll(env, planner, task, T, *, stage: str):
    """A joint move to `_low_roll_joints(T)`, else `_move(T)`; the 5-tuple or -1."""
    got = _low_roll_joints(planner, task, T)
    if got is not None:
        targets, grow = got
        r = common.plan_joints(env, planner, task, targets, label=stage, who=WHO)
        say(env, f"{stage}: joint target", roll_window_growth=round(grow, 3), planned=r != -1)
        if r != -1:
            return r
    return _move(env, planner, T, stage=stage)


def _fold(env, planner, task, label: str):
    """The round's drive posture (`cabinet_search_planner.drive_posture`): the rolls back home."""
    from planners import cabinet_search_planner as csp
    return csp.drive_posture(env, planner, task, label=label)


def _like_now(task, axes) -> np.ndarray:
    """Of the candidate closing axes (world), the one nearest the hand's present closing axis
    (the TCP's y): the fist's yaw about its own axis is wrist roll, so this asks for the least."""
    y = _T(task.agent.tcp.pose.sp)[:3, 1]
    return max((np.asarray(a, dtype=np.float64) for a in axes), key=lambda a: float(a @ y))


def _servo_to(env, planner, task, drv, goal, *, steps: int, tol: float = 0.01):
    """Servo the TCP to `goal` with the base still; `(res, err)` or `(None, err)` at the horizon."""
    res, err = None, np.inf
    drv.reset_command()
    for _ in range(steps):
        res, dp, _, fresh = drv.step(goal)
        if common.stopped_by_horizon(planner):
            return None, err
        if fresh:
            err = float(np.linalg.norm(dp))
            if err < tol:
                break
    return res, err


def open_the_door(env, planner, task, *, door, crack_rad: float = CRACK_RAD,
                  open_rad: float = OPEN_RAD, record: dict | None = None):
    """Crack the leaf on the bar, then open it from inside with the fist (module docstring)."""
    res = _bimbo.open_the_door(env, planner, task, door=door, record=record, target_rad=crack_rad)
    if res == -1 or common.stopped_by_horizon(planner):
        return res
    leaf = Leaf(task, door)
    rad0 = leaf.rad()
    say(env, "cracked by the bar", door_rad=round(rad0, 3), crack_rad=crack_rad, open_rad=open_rad,
        hinge=[round(float(v), 3) for v in leaf.anchor], leaf_m=round(leaf.L, 3))
    if rad0 < 0.5 * crack_rad:
        say(env, "MISSED: the bar swing did not crack the leaf", door_rad=round(rad0, 3))
        return res

    # -- fist, in free air; the arm folded to the drive posture (its rolls) for the stance drive --
    r = planner.close_gripper(t=12)
    if r != -1:
        res = r
    r = _fold(env, planner, task, "fold the fist for the insert stance")
    if r != -1:
        res = r
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()

    # -- drive to the insert stance: facing the gap from the room's side ---------------------
    press = leaf.point(rad0, IN_ALONG_M, IN_OFFSET_M)
    stance = np.asarray(STANCE, dtype=np.float64)
    h = (press - stance) / max(float(np.linalg.norm(press - stance)), 1e-6)
    say(env, "drive to the insert stance", stance=[round(float(v), 3) for v in stance],
        heading_deg=round(float(np.degrees(np.arctan2(h[1], h[0]))), 1),
        press=[round(float(v), 3) for v in press])
    got = planner.drive_base(target_pos=np.array([stance[0], stance[1], 0.0]),
                             target_view_vec=np.array([h[0], h[1], 0.0]), freeze_arm=True,
                             arrive_tol=0.05)
    if got == -1:
        say(env, "MISSED: the insert stance drive refused", door_rad=round(leaf.rad(), 3))
        return res
    res = got
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()
    if int(getattr(planner, "gripper_state", 1)) != -1:
        r = planner.close_gripper(t=12)
        if r != -1:
            res = r

    # -- insert: planned to the pre-insert pose past the free edge, then the stroke ----------
    pre = leaf.fist_pose(task, rad0, IN_ALONG_M, IN_OFFSET_M, z=PUSH_Z_OPEN)
    pre[:3, 3] -= (UNDER_BACK_M + PUSH_Z_OPEN - PUSH_Z) * pre[:3, 2]   # back along its axis: under the leaf
    got = _reach_low_roll(env, planner, task, pre, stage="pre-insert the fist under the leaf")
    if got == -1:
        # The press pose itself is what the reach scan measured: plan there directly.
        press_T = leaf.fist_pose(task, rad0, IN_ALONG_M, IN_OFFSET_M, z=PUSH_Z_OPEN)
        got = _move(env, planner, press_T, stage="the fist straight to the press point", tries=3)
    if got == -1:
        say(env, "MISSED: no plan to the pre-insert pose", door_rad=round(leaf.rad(), 3),
            pre=[round(float(v), 3) for v in pre[:3, 3]])
        return res
    res = got
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()
    try:
        servo = _bimbo.HandServo(task, planner)
    except Exception as exc:
        say(env, "MISSED: no kinematic model for the hand servo", why=f"{type(exc).__name__}: {exc}"[:160])
        return res
    drv = _bimbo._Driver(planner, task, servo, _bimbo._torso(task))
    servo.hold_rolls_here()
    rad_in = leaf.rad()
    goal = leaf.fist_pose(task, rad_in, IN_ALONG_M, IN_OFFSET_M, z=PUSH_Z_OPEN)
    # The stroke up into the gap is the servo's alone, straight along the fist's axis: a planned move
    # here took its own IK goal and wound the upper-arm roll 0.99 -> 2.46 rad (seed 7991002).
    got = -1
    drv.reset_command()
    servo.hold_rolls_here()
    out, err = _servo_to(env, planner, task, drv, goal, steps=INSERT_MAX_STEPS)
    if out is None:
        return planner._guard.last_step
    res = out
    tcp = _T(task.agent.tcp.pose.sp)[:3, 3]
    say(env, "fist in the gap", door_rad=round(leaf.rad(), 3), err_m=round(err, 3),
        planned=got != -1, torso=round(_bimbo._torso(task), 3),
        tcp=[round(float(v), 3) for v in tcp], goal=[round(float(v), 3) for v in goal[:3, 3]],
        off_inner=round(float((tcp[:2] - leaf.anchor) @ leaf.inner(leaf.rad())), 3),
        touching=planner.touching_now())

    # -- push: the fist frame carried round the hinge with the leaf ---------------------------
    G0, ref = goal.copy(), float(rad_in)
    dt = float(planner.base_env.control_timestep)
    sw = leaf.rad()
    hist: list = []
    reason, i = "max-steps", -1
    hand = task.agent.robot.links_map["gripper_link"]
    art = task.scene.articulations[crp._art_key(task, door)]
    joints = art.get_active_joints()
    link = joints[[j.name for j in joints].index(door.hinge)].child_link
    press_max, perr_max = 0.0, 0.0
    drv.reset_command()
    q = _bimbo._qpos(task)
    d0 = G0[:2, 3] - q[:2]
    bearing0 = float(np.arctan2(d0[1], d0[0])) - float(q[2])
    for i in range(PUSH_MAX_STEPS):
        rad = leaf.rad()
        hist.append(rad)
        if rad >= open_rad:
            reason = "target"
            break
        if len(hist) > STALL_STEPS and hist[-1] - hist[-1 - STALL_STEPS] < STALL_EPS:
            reason = "stalled"
            break
        sw = min(open_rad + 0.02, sw + LEAD_RATE * dt, max(sw, rad + LEAD_MAX))
        turn = leaf.s * (sw - ref)
        goal = ds.door_frames(G0, leaf.anchor, turn, hand_turn=HAND_TURN_OPEN * turn)[1]
        # The base turns in place to keep the press point at the bearing it had at the start
        # (the fist sweeps ~25 deg left round the hinge); it never translates: a base riding
        # the hinge's circle ends in the west wall.
        q = _bimbo._qpos(task)
        d = goal[:2, 3] - q[:2]
        bearing = float(np.arctan2(d[1], d[0])) - float(q[2])
        err_yaw = (bearing - bearing0 + np.pi) % (2 * np.pi) - np.pi
        res, dp, _, fresh = drv.step(goal, (0.0, float(np.clip(YAW_GAIN * err_yaw, -YAW_W_MAX, YAW_W_MAX))))
        if fresh:
            perr_max = max(perr_max, float(np.linalg.norm(dp)))
        press = float(np.linalg.norm(_bimbo._np(task.scene.get_pairwise_contact_forces(hand, link))
                                     .reshape(-1, 3), axis=1).max())
        press_max = max(press_max, press)
        if common.stopped_by_horizon(planner):
            return res
    leaf_touching = [c for c in planner.touching_now() if "hingedoor" in c or "cab_1" in c]
    say(env, "fist push done", door_rad=round(leaf.rad(), 3), why=reason, steps=i + 1,
        cmd_rad=round(sw, 3), perr_max=round(perr_max, 3), press_max_n=round(press_max, 1),
        touching=planner.touching_now(), leaf_touching=leaf_touching,
        **({"door_tail": [round(v, 3) for v in hist[-15:]]} if reason != "target" else {}))

    # -- withdraw: back along the leaf, out past the free edge -----------------------------------
    rad1 = leaf.rad()
    T = _T(task.agent.tcp.pose.sp)
    out_goal = T.copy()
    out_goal[:3, 3] = T[:3, 3] - WITHDRAW_M * T[:3, 2]          # back down the fist's axis
    out, err = _servo_to(env, planner, task, drv, out_goal, steps=60, tol=0.02)
    if out is None:
        return planner._guard.last_step
    res = out
    planner.planner.update_from_simulation()
    say(env, "fist withdrawn", door_rad=round(leaf.rad(), 3), moved_after_push=round(leaf.rad() - rad1, 4),
        err_m=round(err, 3), touching=planner.touching_now())
    return res


def _quat(T) -> np.ndarray:
    """wxyz of a 4x4's rotation."""
    from transforms3d.quaternions import mat2quat
    return np.asarray(mat2quat(np.asarray(T)[:3, :3]), dtype=np.float64)


# -- closing: the same fist on the OUTER face, carried round the hinge back to shut -------------

CLOSE_STANCE = STANCE
CLOSE_TARGET_RAD = 0.004
"""The sweep stops once the leaf reads this shut, under the task's `theta_closed` (0.01)."""
CLOSE_CMD_FLOOR = _f("MIKASA_FD_CLOSE_CMD_FLOOR", -0.15)
"""The commanded angle's end: past shut by about the fist's gap to the face (IN_OFFSET_M less half a
fist, at the press radius), so the fist reaches the shut plane; the hinge stops the leaf at 0. At
-0.02 the leaf stopped 0.06-0.11 rad short on four seeds, the fist's gap still open."""
CLOSE_MAX_STEPS = int(_f("MIKASA_FD_CLOSE_MAX_STEPS", 700))
PRESS_IN_M = 0.04
"""A sweep that stalled short of shut (0.020 rad, seed 7991001) ends with a straight press this deep."""
OFF_FACE_M = 0.06
"""The fist leaves the shut face straight out by this much before it goes down and away."""
DWELL_STEPS_EXTRA = 2


def _outer_pose(task, leaf, rad: float) -> np.ndarray:
    """The fist IN_OFFSET_M off the leaf's OUTER face, IN_ALONG_M in from its free edge, pointing
    along the leaf toward the hinge pitched FIST_PITCH up, the fingers closing as near vertical as
    that allows (the outer-face scan: across the leaf the palm meets it at every stance)."""
    xy = leaf.point(rad, IN_ALONG_M, -IN_OFFSET_M)
    u, nin = leaf.u(rad), leaf.inner(rad)
    a = np.array([-math.cos(FIST_PITCH) * u[0], -math.cos(FIST_PITCH) * u[1], math.sin(FIST_PITCH)])
    c = _fist_closing(task, a, u, nin)
    return _T(task.agent.build_grasp_pose(a, c, np.array([xy[0], xy[1], PUSH_Z])))


def _closed_by_state(planner, task):
    """The env's own verdict after the release dwell, as `close_and_verify` reads it."""
    res = None
    for _ in range(int(task.cfg.close_dwell_steps) + DWELL_STEPS_EXTRA):
        res = planner.idle_steps(t=1)
        if common.stopped_by_horizon(planner):
            return res, False
    idx = int(_bimbo._np(task.last_opened).reshape(-1)[0])
    return res, idx >= 0 and not bool(_bimbo._np(task.is_open)[0, idx])


def close_the_leaf(env, planner, task, *, door, res):
    """Sweep the leaf shut with the fist on its outer face; `(res, closed)` judged by state."""
    leaf = Leaf(task, door)
    rad0 = leaf.rad()
    say(env, "close the leaf with the fist", door_rad=round(rad0, 3),
        stance=[round(v, 3) for v in CLOSE_STANCE])
    # The opening left the base at STANCE and the fist under the leaf: no fold and no drive
    # (a fold that blocked on the leaf went to RRT, whose joint paths are measured moving the arm
    # by nothing (K111), and the drive's screw then put the wrist into the leaf, seed 7991002).
    base = _bimbo._qpos(task)[:2]
    if float(np.linalg.norm(base - np.asarray(CLOSE_STANCE))) > 0.08:
        r = _fold(env, planner, task, "fold the fist for the closing stance")
        if r != -1:
            res = r
        if common.stopped_by_horizon(planner):
            return res, False
        planner.planner.update_from_simulation()
        first = _outer_pose(task, leaf, rad0)
        look = first[:2, 3] - np.asarray(CLOSE_STANCE)
        got = planner.drive_base(target_pos=np.array([CLOSE_STANCE[0], CLOSE_STANCE[1], 0.0]),
                                 target_view_vec=np.array([look[0], look[1], 0.0]), freeze_arm=True,
                                 arrive_tol=0.05)
        if got == -1:
            say(env, "MISSED: the closing stance drive refused", door_rad=round(leaf.rad(), 3))
            return res, False
        res = got
        if common.stopped_by_horizon(planner):
            return res, False
        planner.planner.update_from_simulation()
    if int(getattr(planner, "gripper_state", 1)) != -1:
        r = planner.close_gripper(t=12)
        if r != -1:
            res = r
    rad1 = leaf.rad()
    goal = _outer_pose(task, leaf, rad1)
    pre = goal.copy()
    pre[:3, 3] -= UNDER_BACK_M * pre[:3, 2]
    got = _reach_low_roll(env, planner, task, pre, stage="pre-contact the outer face from under the leaf")
    if got != -1:
        res = got
        if common.stopped_by_horizon(planner):
            return res, False
        planner.planner.update_from_simulation()
    try:
        servo = _bimbo.HandServo(task, planner)
    except Exception as exc:
        say(env, "MISSED: no kinematic model for the hand servo", why=f"{type(exc).__name__}: {exc}"[:160])
        return res, False
    drv = _bimbo._Driver(planner, task, servo, _bimbo._torso(task))
    servo.hold_rolls_here()
    out, err = _servo_to(env, planner, task, drv, goal, steps=INSERT_MAX_STEPS)   # the stroke up
    if out is None:
        return planner._guard.last_step, False
    res = out
    if err > 0.03:
        say(env, "MISSED: the fist did not reach the outer face", door_rad=round(leaf.rad(), 3),
            err_m=round(err, 3), goal=[round(float(v), 3) for v in goal[:3, 3]])
        return res, False
    drv.reset_command()
    G0, ref = goal.copy(), float(rad1)
    tcp = _T(task.agent.tcp.pose.sp)[:3, 3]
    say(env, "fist on the outer face", door_rad=round(leaf.rad(), 3),
        tcp=[round(float(v), 3) for v in tcp], goal=[round(float(v), 3) for v in goal[:3, 3]],
        off_outer=round(float(-(tcp[:2] - leaf.anchor) @ leaf.inner(leaf.rad())), 3),
        torso=round(_bimbo._torso(task), 3), touching=planner.touching_now())
    dt = float(planner.base_env.control_timestep)
    q = _bimbo._qpos(task)
    d0 = G0[:2, 3] - q[:2]
    bearing0 = float(np.arctan2(d0[1], d0[0])) - float(q[2])
    sw = leaf.rad()
    hist: list = []
    reason, i, perr_max = "max-steps", -1, 0.0
    for i in range(CLOSE_MAX_STEPS):
        rad = leaf.rad()
        hist.append(rad)
        if rad <= CLOSE_TARGET_RAD:
            reason = "shut"
            break
        if len(hist) > STALL_STEPS and hist[-1 - STALL_STEPS] - hist[-1] < STALL_EPS:
            reason = "stalled"
            break
        sw = max(CLOSE_CMD_FLOOR, sw - LEAD_RATE * dt, min(sw, rad - LEAD_MAX))
        turn = leaf.s * (sw - ref)
        goal = ds.door_frames(G0, leaf.anchor, turn, hand_turn=HAND_TURN * turn)[1]
        q = _bimbo._qpos(task)
        d = goal[:2, 3] - q[:2]
        err_yaw = (float(np.arctan2(d[1], d[0])) - float(q[2]) - bearing0 + np.pi) % (2 * np.pi) - np.pi
        res, dp, _, fresh = drv.step(goal, (0.0, float(np.clip(YAW_GAIN * err_yaw, -YAW_W_MAX, YAW_W_MAX))))
        if fresh:
            perr_max = max(perr_max, float(np.linalg.norm(dp)))
        if common.stopped_by_horizon(planner):
            return res, False
    say(env, "fist sweep done", door_rad=round(leaf.rad(), 4), why=reason, steps=i + 1,
        cmd_rad=round(sw, 3), perr_max=round(perr_max, 3),
        **({"door_tail": [round(v, 3) for v in hist[-15:]]} if reason != "shut" else {}))
    rad_end = leaf.rad()
    if reason != "shut" and rad_end < 0.15:
        # The last few cm: press straight into the face (toward the cabinet) by PRESS_IN_M.
        T = _T(task.agent.tcp.pose.sp)
        press_goal = T.copy()
        press_goal[:3, 3] = T[:3, 3] + PRESS_IN_M * np.array([*leaf.inner(rad_end), 0.0])
        out, _ = _servo_to(env, planner, task, drv, press_goal, steps=30, tol=0.005)
        if out is None:
            return planner._guard.last_step, False
        res = out
        say(env, "pressed the leaf in", door_rad_before=round(rad_end, 4), door_rad=round(leaf.rad(), 4))
    # Off the face first (straight out, away from the cabinet), then down and away: sliding the
    # fist along the shut face dragged the leaf back open (0.0 -> 0.03 rad, seed 7991004).
    T = _T(task.agent.tcp.pose.sp)
    off_goal = T.copy()
    off_goal[:3, 3] = T[:3, 3] - OFF_FACE_M * np.array([*leaf.inner(leaf.rad()), 0.0])
    out, _ = _servo_to(env, planner, task, drv, off_goal, steps=30, tol=0.01)
    if out is None:
        return planner._guard.last_step, False
    res = out
    T = _T(task.agent.tcp.pose.sp)
    out_goal = T.copy()
    out_goal[:3, 3] = T[:3, 3] - WITHDRAW_M * T[:3, 2]
    out, err = _servo_to(env, planner, task, drv, out_goal, steps=60, tol=0.02)
    if out is None:
        return planner._guard.last_step, False
    res = out
    planner.planner.update_from_simulation()
    res, closed = _closed_by_state(planner, task)
    say(env, "fist withdrawn from the shut leaf", door_rad=round(leaf.rad(), 4), closed=closed,
        touching=planner.touching_now())
    return res, closed
