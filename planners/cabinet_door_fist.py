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
IN_ALONG_M = _f("MIKASA_FD_IN_ALONG_M", 0.10)
"""How far in from the free edge, along the leaf, the fist presses."""
IN_OFFSET_M = _f("MIKASA_FD_IN_OFFSET_M", 0.06)
"""The fist centre off the leaf's inner face at the start of the push (half a fist plus ~2 cm)."""
PRE_PAST_M = _f("MIKASA_FD_PRE_PAST_M", 0.14)
"""The pre-insert pose stands this far past the free edge, along the leaf."""
PUSH_Z = _f("MIKASA_FD_PUSH_Z", 1.62)
"""Height of the fist: 3 cm over the bar, ~20 cm over the shelf floor (1.42)."""
LEAD_MAX = _f("MIKASA_FD_LEAD_MAX", 0.06)
"""Rad the commanded angle may run ahead of the measured leaf: the fist's press."""
LEAD_RATE = _f("MIKASA_FD_LEAD_RATE", 0.30)
"""Rad/s the commanded angle advances."""
PUSH_MAX_STEPS = int(_f("MIKASA_FD_PUSH_MAX_STEPS", 500))
INSERT_MAX_STEPS = 80
WITHDRAW_M = _f("MIKASA_FD_WITHDRAW_M", 0.25)
STALL_STEPS = 40
STALL_EPS = 0.01

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
        self.L = float(np.linalg.norm(edge0 - self.anchor))
        self.u0 = (edge0 - self.anchor) / self.L

    def rad(self) -> float:
        return float(crp.door_rad_now(self.task, self.door))

    def u(self, rad: float) -> np.ndarray:
        """Unit vector hinge -> free edge at opening `rad`."""
        return ds.rot2(self.s * float(rad)) @ self.u0

    def inner(self, rad: float) -> np.ndarray:
        """Unit normal of the leaf's INNER face (the cabinet side when shut) at `rad`."""
        _, n = crp.push_frame(self.anchor, self.sense, self.door.open_dir * float(rad), self.door)
        return -np.asarray(n, dtype=np.float64)

    def point(self, rad: float, along_from_edge: float, off_inner: float) -> np.ndarray:
        return self.anchor + self.u(rad) * (self.L - along_from_edge) + self.inner(rad) * off_inner

    def fist_pose(self, task, rad: float, along_from_edge: float, off_inner: float) -> np.ndarray:
        """The fist at the leaf: approach along the leaf toward the hinge, fingers closing
        across it (so a finger's outer face meets the inner face), at PUSH_Z."""
        xy = self.point(rad, along_from_edge, off_inner)
        u, nin = self.u(rad), self.inner(rad)
        pose = task.agent.build_grasp_pose(np.array([-u[0], -u[1], 0.0]),
                                           np.array([nin[0], nin[1], 0.0]),
                                           np.array([xy[0], xy[1], PUSH_Z]))
        return _T(pose)


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

    # -- fist, in free air ------------------------------------------------------------------
    r = planner.close_gripper(t=12)
    if r != -1:
        res = r
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()

    # -- insert: planned to the pre-insert pose past the free edge, then the stroke ----------
    pre = leaf.fist_pose(task, rad0, -PRE_PAST_M, IN_OFFSET_M)
    got = common.arm_move(env, planner, sapien.Pose(p=pre[:3, 3], q=_quat(pre)), who=WHO,
                          stage="pre-insert the fist past the free edge", tries=3)
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
    goal = leaf.fist_pose(task, rad_in, IN_ALONG_M, IN_OFFSET_M)
    out, err = _servo_to(env, planner, task, drv, goal, steps=INSERT_MAX_STEPS)
    if out is None:
        return planner._guard.last_step
    res = out if out is not None else res
    tcp = _T(task.agent.tcp.pose.sp)[:3, 3]
    say(env, "fist in the gap", door_rad=round(leaf.rad(), 3), err_m=round(err, 3),
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
        goal = ds.door_frames(G0, leaf.anchor, turn, hand_turn=turn)[1]
        res, dp, _, fresh = drv.step(goal)
        if fresh:
            perr_max = max(perr_max, float(np.linalg.norm(dp)))
        press = float(np.linalg.norm(_bimbo._np(task.scene.get_pairwise_contact_forces(hand, link))
                                     .reshape(-1, 3), axis=1).max())
        press_max = max(press_max, press)
        if common.stopped_by_horizon(planner):
            return res
    say(env, "fist push done", door_rad=round(leaf.rad(), 3), why=reason, steps=i + 1,
        cmd_rad=round(sw, 3), perr_max=round(perr_max, 3), press_max_n=round(press_max, 1),
        **({"door_tail": [round(v, 3) for v in hist[-15:]]} if reason != "target" else {}))

    # -- withdraw: back along the leaf, out past the free edge -----------------------------------
    rad1 = leaf.rad()
    T = _T(task.agent.tcp.pose.sp)
    out_goal = T.copy()
    out_goal[:3, 3] = T[:3, 3] + WITHDRAW_M * np.array([*leaf.u(rad1), 0.0])
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
