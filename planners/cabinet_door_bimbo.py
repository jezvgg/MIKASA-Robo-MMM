"""A wall-cabinet leaf opened the way BIMBO_EvoX opens the fridge of its task 62, on a Fetch.

BIMBO_EvoX (read-only clone `external/BIMBO_EvoX`, branch `agent`,
`tasks/62_halve_an_egg/planners/{oracle/planner.py, fair/phases/door.py, fair/phases/swing.py}`)
opens a fridge door that the assisted grasp (OmniGibson's weld) holds by its bar:

1. door_reach / door_grasp — the open hand goes to a pre-grasp point at the bar, then to the
   bar; it closes only with the finger midpoint on the bar and keeps the grasp only when the
   finger gap afterwards matches the bar's width (else it opens and tries again).
2. door_swing — EVERY control the commanded swing angle advances by a lead of 0.012 rad, the
   hand is servoed (resolved rate from the current joints) to the bar's place on the arc at
   that angle (`door_frames(geo, sw)`), and the base creeps by velocity so that point stays in
   front of the chest (`follow_base`); the weld holds the bar; the door angle is read from the
   hand while it holds the bar.
3. door_hold / door_wide / door_prop / door_release / door_clear — confirm the weld, widen,
   park the base in the closing sector (the fridge door swings shut on its own), let go, and
   take the open hand off the bar on a path checked against the panel.

What this module transfers, and what it does not:

* Steps 1 and 2, and the release with the checked clear of step 3. The bar is reached by the
  shipped dock/pre-grasp/contact-stroke (`cabinet_retrieval_planner.grasp_the_bar`), which is
  BIMBO's reach-then-grasp; the finger-gap test after the close is BIMBO's. The swing is a
  per-step loop (`_swing`): the lead angle (BIMBO's 0.36 rad/s), the hand servoed to the
  grasp frame carried round the hinge (`door_swing.door_frames`; a weighted damped-least-
  squares step on the arm's seven joints through the articulation's own Pinocchio model), the
  base creeping, the door angle also read from the hand (logged next to the hinge's).
* The base creep is NOT BIMBO's `follow_base`. R1Pro's base strafes; a Fetch base has a
  forward speed and a yaw rate only, so the creep is the exact unicycle law that keeps the
  servo's goal at a chosen place in the base frame (`door_swing.unicycle_hold_cmd`; its
  feed-forward part is the shipped K103 arc law). The place moves from the grasp's lever to
  the stance the shipped arc pull ENDS in (LEVER_END), and settles there: every later stage of
  the round (the push from here, the tuck, the touch-dock drive) was measured from that stance,
  and a swing that kept the grasp's lever (BIMBO's fixed point in front of the chest) lost
  them (2026-10-16: 0/10 and 1/5 with assisted grasp).
* The hand does NOT turn with the leaf. BIMBO's frames carry the hand's orientation round the
  hinge with the bar (`rz @ rot`). On a Fetch standing on the handle's side that is out of
  reach: carrying the bar ~0.5 m sideways turns a unicycle base the OTHER way (~-60 deg while
  the leaf turns +100 deg), so the arm would have to turn the hand ~160 deg against the base.
  `HAND_TURN` is the fraction of the door's turn the hand's yaw follows; 0 keeps the hand's
  world orientation, which is the end state the friction grip reaches in the shipped arc
  pull. What the servo adds over that pull is that the orientation is COMMANDED: with the weld
  (rotation free at the bar) nothing else holds it, and the held arm let the hand turn a
  further ~40 deg with the base, into the leaf's free edge (off-normal -0.74 instead of -0.16
  at the release); the servo releases at -0.21..-0.26.
* The servo stays inside the collection's roll window (`roll_window`) and keeps the non-roll
  joints near the grasp configuration in its null space, as the held arm of the shipped pull
  does; an unconstrained servo broke the window and every later plan.
* door_wide and door_prop are not transferred: the hinges of kitchen 102 hold where they are
  released (1.778 -> 1.757 rad over the release, measured), so the base need not prop the
  leaf, and the swing goes straight to `DOOR_OPEN_TARGET`.
* Added for a Fetch, not in BIMBO: `lower_the_hand` (a found can's tuck that would sweep the
  leaf starts under its bottom edge) and, opt-in and not working yet, `close_by_the_bar` (when
  the shipped push from here finds no plan, take the bar again and swing toward shut; see
  CLOSE_BY_BAR for why the re-grasp misses).

Selection lives in `planners/cabinet_search_planner.py` (`door_mode`): `MIKASA_CS_DOOR=bimbo|arc`,
default `bimbo` with assisted grasp on and `arc` (the shipped K104-K108 path, byte-identical)
with it off.
"""

from __future__ import annotations

import os

import numpy as np

from planners import cabinet_retrieval_planner as crp
from planners.oracle import door_swing as ds
from planners.oracle import oracle_common as common

WHO = "cabinet_door_bimbo"


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


LEAD_RATE = _f("MIKASA_BD_LEAD_RATE", 0.36)
"""Rad/s the commanded swing angle advances: BIMBO's 0.012 rad per 30 Hz control."""
LEAD_MAX = _f("MIKASA_BD_LEAD_MAX", 0.15)
"""The commanded angle never runs more than this ahead of the measured door (BIMBO's oracle has
no cap; its fair planner times out a stuck door instead — a cap keeps the goal within a few
centimetres of the bar however the door moves)."""
HAND_TURN = _f("MIKASA_BD_HAND_TURN", 0.0)
"""Fraction of the door's turn the hand's yaw follows (BIMBO: 1; see the module docstring)."""
SWING_MAX_STEPS = int(_f("MIKASA_BD_SWING_MAX_STEPS", 600))
STALL_STEPS = 60
STALL_EPS = 0.01
"Less door travel than STALL_EPS over STALL_STEPS controls is a stall."

#: the hand servo (`door_swing.dls_step`), per control step
K_POS, K_ROT = _f("MIKASA_BD_K_POS", 0.6), _f("MIKASA_BD_K_ROT", 0.6)
CAP_POS, CAP_ROT, CAP_Q = 0.04, 0.10, 0.08
W_ROT = 0.5
ROLL_COST = _f("MIKASA_BD_ROLL_COST", 0.5)
ROLL_HOLD = _f("MIKASA_BD_ROLL_HOLD", 0.0)
ARM_HOLD = _f("MIKASA_BD_ARM_HOLD", 0.3)
"""How the servo spends the arm's redundancy. A roll joint (upperarm, forearm, wrist) costs
ROLL_COST of another joint's motion; the null space pulls the roll joints toward where the grasp
left them at ROLL_HOLD and the other four at ARM_HOLD per control. Measured 2026-10-16 (seeds
0-4, assisted grasp): this split (the shipped pull's held arm keeps pan/lift/elbow/wrist-flex and
twists in its rolls) kept every later plan; rolls ten times dearer stalled two swings at
1.67-1.71 rad, and a free servo flipped the elbow branch (+0.15..+0.56)."""
LEVER_END = (_f("MIKASA_BD_LEVER_END_X", 0.66), _f("MIKASA_BD_LEVER_END_Y", 0.34))
"""Where the swing's goal (the bar) should stand in the base frame when the door reaches its
target: (forward, toward the side the leaf opens to — mirrored for a leaf that opens clockwise).
The creep moves it there from the grasp's lever by LEVER_LEAD_FRAC of the swing and settles.
It is the shipped arc pull's END stance, read off its pull-end base poses and the bar at the
pull-end angle (2026-10-16, collection runs without assisted grasp): cab_main L (0.64, -0.33),
cab_2 R (0.68, 0.34), cab_2 L (0.64, -0.34) — the held arm yields sideways under the pull. A
forward value <= 0 keeps the grasp's lever all the way (BIMBO's fixed point before the chest)."""
ARM_LEAD_MAX = 0.20
"""The arm's commanded joints integrate the servo's steps but stay within this of the measured
joints (rad): a pull against the door's resistance keeps its force, a blocked arm does not wind up."""

#: the base creep (`door_swing.unicycle_hold_cmd`)
BASE_GAIN = _f("MIKASA_BD_BASE_GAIN", 2.0)
LEVER_LEAD_FRAC = _f("MIKASA_BD_LEVER_LEAD_FRAC", 0.8)
"The lever reaches LEVER_END at this fraction of the swing (it lags the command otherwise)."
STANCE_TURN = _f("MIKASA_BD_STANCE_TURN", 0.0)
"""OFF (0). When > 0: after the swing, turn-drive-turn the base (hand held on the bar) onto the
shipped arc's end stance — the heading turned this far toward the side the leaf opens to (the
pull-end headings are 34.2-34.7 deg for right leaves and 147.0 deg for left ones from a 90 deg
dock: 0.977 rad) and the bar at LEVER_END. With the lever alone the swing ends right leaves 3-7
deg short of that heading and 6-9 cm off the stance, where the push from here once found no IK
(seed 9). Measured 2026-10-16, assisted grasp, seeds 0-9: off 9/10 (822cb7c), on 8/10 (565509e:
a 2.7 deg turn under the held bar slid the base 0.17 m) and, with the leg guards below, 7/10
(a860900: the turns slide the base 1-3 cm within a few steps and stop, and the stances they
leave lost two touch-dock drives). A base turning under a hand held on a free leaf is not a
free motion."""
STANCE_TOL_M, STANCE_TOL_RAD = 0.02, 0.035
STANCE_SLIDE_M = 0.03
STANCE_HAND_ERR_M = 0.03
STANCE_LEG_STEPS = 60
STANCE_AIM_MAX = 0.8
"Rad: a position leg that needs a bigger turn to aim is skipped (the arm holds the bar meanwhile)."
STANCE_MAX_STEPS = 120
SETTLE_STEPS = int(_f("MIKASA_BD_SETTLE_STEPS", 40))
SETTLE_TOL_M = 0.01
"""After the door reaches its target the goal stays on the bar and the creep runs on, at most
SETTLE_STEPS controls, until the goal stands within SETTLE_TOL_M of LEVER_END."""
BASE_V_MAX, BASE_W_MAX = 0.20, 0.6

GRASP_GAP = (common.FINGER_EMPTY_M, crp.RETRACE_GRIP_MAX)
"""Finger aperture that means the bar is between the pads (the bar is 26 mm; BIMBO's
0.022-0.05 for its 37 mm bar)."""
GRASP_RETRY_IN_M = 0.015
HOLD_STEPS = 4
"Controls the swing's last target is held with the base still before the release (BIMBO's door_hold)."
CLEAR_M = _f("MIKASA_BD_CLEAR_M", 0.15)
CLEAR_LIFT_M = _f("MIKASA_BD_CLEAR_LIFT_M", 0.0)
CLEAR_LIFT_OFF = _f("MIKASA_BD_CLEAR_LIFT_OFF", 0.0)
"""Share of the outward normal added to the in-plane slide off the bar. 0: measured 2026-10-16
(seeds 0-1, 0.25): moving outward the INNER pad (between bar and face) carries the bar out and
opens the leaf (+0.01..+0.017 rad), and it leaves the hand on the face's outer side, where the
tuck's joint line swept a finger through the leaf (seed 0, knot 4/23)."""
LOWER_M = _f("MIKASA_BD_LOWER_M", 0.30)
"""How far `lower_the_hand` drops the open hand: from bar height (1.59) to under the open
leaf's bottom edge (1.391, `stow_hand`), so no joint line from there can sweep the leaf."""
CLEAR_MAX_STEPS = 50
CLEAR_LEAF_TOL = 0.03
"Rad: a clear that moves the leaf more than this stops where it is."
LEAF_EDGE_PAST_BAR = 0.05
PATH_SLACK_M = 0.005
"""How much nearer the panel the clear's straight path may pass than where it starts: the
panel here is the hinge-axis segment, 2-3 cm off the face, and a slide parallel to it closes
by a few millimetres when the leaf turned a little since it was read."""
"The free edge stands this far past the bar along the leaf (robocasa `handle_hpad`)."


def say(env, stage: str, **extra) -> None:
    common.say(env, WHO, stage, **extra)


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _T(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64).reshape(4, 4)


def _qpos(task) -> np.ndarray:
    return _np(task.agent.robot.get_qpos()).reshape(-1).astype(np.float64)


ROLL_WINDOW_MARGIN = 0.10
"""Rad kept inside the collection's roll window (`RollPathPlanner.margin` is 0.05)."""
ROLL_EDGE_M = 0.5
"""Within this many radians of a window edge a roll joint's cost rises (x10 at the edge)."""


def roll_window(planner, indices) -> tuple[np.ndarray, np.ndarray] | None:
    """The range each roll joint may still use without breaking the collection's roll window.

    `CollectionMotionPlanner.prefer_low_roll_ik` keeps, for each of the three roll joints, the
    lowest and highest value measured over the WHOLE episode, and refuses any plan that would
    take the range beyond pi - 0.05 or a joint beyond +-(pi - 0.05) (`RollPathPlanner.accepts`).
    A motion this module makes without a plan must respect the same window, or every plan after
    it is refused (`collection joint limit: no bounded path`, seed 7, 2026-10-16: the servo had
    taken the forearm roll from -1.29 to 2.74). Returns (lo, hi) for the joints in `indices`
    (robot qpos indices), or None when the planner keeps no window.

    Example:
        >>> import types
        >>> p = types.SimpleNamespace(_roll_indices=[5, 7], _roll_low=np.array([0.0, -1.0]),
        ...                           _roll_high=np.array([1.0, 0.5]))
        >>> lo, hi = roll_window(p, [7, 5])
        >>> np.round(lo, 3).tolist(), np.round(hi, 3).tolist()
        ([-2.542, -2.042], [2.042, 3.042])
    """
    ridx = getattr(planner, "_roll_indices", None)
    low, high = getattr(planner, "_roll_low", None), getattr(planner, "_roll_high", None)
    width = np.pi - ROLL_WINDOW_MARGIN
    pos = {int(j): k for k, j in enumerate(ridx)}
    lo = np.array([max(-width, float(high[pos[int(j)]]) - width) for j in indices])
    hi = np.array([min(width, float(low[pos[int(j)]]) + width) for j in indices])
    return lo, hi


class HandServo:
    """Resolved-rate servo of `gripper_link` (the TCP) with the arm's seven joints.

    Kinematics from the simulator's own articulation (`create_pinocchio_model`: the joint order
    is the qpos order, the root is fixed at the articulation's root pose), the Jacobian by
    forward differences of that model's FK — seven extra FK calls per control, no frame
    convention to get wrong. The base and the torso are not moved by the servo.
    """

    FD_EPS = 1e-6

    def __init__(self, task, planner=None):
        robot = task.agent.robot
        self.task, self.planner = task, planner
        self.pm = robot._objs[0].create_pinocchio_model()
        self.link = int(robot.links_map["gripper_link"]._objs[0].index)
        jm = robot.active_joints_map
        names = list(task.agent.controller.controllers["arm"].config.joint_names)
        self.idx = np.array([int(jm[n].active_index[0]) for n in names])
        lim = _np(robot.get_qlimits()).reshape(-1, 2)[self.idx].astype(np.float64)
        bounded = np.isfinite(lim).all(axis=1) & ((lim[:, 1] - lim[:, 0]) < 4 * np.pi - 0.1)
        self.lo = np.where(bounded, lim[:, 0] + 0.02, -np.inf)
        self.hi = np.where(bounded, lim[:, 1] - 0.02, np.inf)
        self.T_root = _T(robot.pose.sp)
        self.roll = np.array([n.endswith("roll_joint") for n in names])
        self.cost = np.where(self.roll, ROLL_COST, 1.0)
        self.roll_ref = _qpos(task)[self.idx].copy()

    def hold_rolls_here(self) -> None:
        """The null-space pull keeps the roll joints where they stand now."""
        self.roll_ref = _qpos(self.task)[self.idx].copy()

    def pose(self, q) -> np.ndarray:
        self.pm.compute_forward_kinematics(np.asarray(q, dtype=np.float64))
        return self.T_root @ _T(self.pm.get_link_pose(self.link))

    def jacobian(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        T0 = self.pose(q)
        J = np.zeros((6, len(self.idx)))
        for k, j in enumerate(self.idx):
            q1 = q.copy()
            q1[j] += self.FD_EPS
            dp, dr = ds.pose_error(T0, self.pose(q1))
            J[:3, k], J[3:, k] = dp / self.FD_EPS, dr / self.FD_EPS
        return J

    def fk_gap(self) -> float:
        """Metres between this model's TCP and the simulator's, at the current state."""
        T = self.pose(_qpos(self.task))
        return float(np.linalg.norm(T[:3, 3] - _T(self.task.agent.tcp.pose.sp)[:3, 3]))

    def step(self, q, T_goal, q_cmd) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One control: the arm's next commanded joints, and the pose error it was taken on."""
        q = np.asarray(q, dtype=np.float64)
        dp, dr = ds.pose_error(self.pose(q), T_goal)
        meas = q[self.idx]
        bias = np.where(self.roll, ROLL_HOLD, ARM_HOLD) * (self.roll_ref - meas)
        lo, hi = self.lo.copy(), self.hi.copy()
        cost = self.cost.copy()
        win = roll_window(self.planner, self.idx[self.roll]) if self.planner is not None else None
        if win is not None:
            lo[self.roll] = np.maximum(lo[self.roll], win[0])
            hi[self.roll] = np.minimum(hi[self.roll], win[1])
            edge = np.minimum(meas[self.roll] - lo[self.roll], hi[self.roll] - meas[self.roll])
            cost[self.roll] *= 1.0 + 9.0 * np.clip(1.0 - edge / ROLL_EDGE_M, 0.0, 1.0)
        dq = ds.dls_step(self.jacobian(q), dp, dr, k_pos=K_POS, k_rot=K_ROT, cap_pos=CAP_POS,
                         cap_rot=CAP_ROT, cap_q=CAP_Q, w_rot=W_ROT, joint_cost=cost,
                         bias=bias)
        cmd = np.clip(np.asarray(q_cmd, dtype=np.float64) + dq, meas - ARM_LEAD_MAX,
                      meas + ARM_LEAD_MAX)
        return np.clip(cmd, lo, hi), dp, dr


def _base_scale(planner) -> float:
    cfg = planner.env_agent.controller.controllers["base"].config
    assert getattr(cfg, "normalize_action", False) and float(cfg.upper[0]) == 1.0, (
        f"the door servo assumes the ds_fetch base channel shape; got upper={cfg.upper}")
    return float(cfg.upper[1])


class _Driver:
    """Steps the env with the servo's command, one control at a time.

    The collection solver samples a new command on even 20 Hz steps only and repeats the last
    one on odd steps (`CollectionMotionPlanner._step`); a servo that integrated a step there
    would double its gain. So the command is recomputed only on a sampled step, and an odd
    step re-issues the last one (which the solver replaces by its own copy anyway).
    """

    def __init__(self, planner, task, servo, torso: float):
        self.planner, self.task, self.servo, self.torso = planner, task, servo, float(torso)
        self.omega_scale = _base_scale(planner)
        self.ten_hz = bool(getattr(planner, "supports_execution_noise", False))
        self.q_cmd = _qpos(task)[servo.idx].copy()
        self.last = None
        self.dp = np.zeros(3)
        self.dr = np.zeros(3)

    def reset_command(self) -> None:
        """Start the integrated arm command from the measured joints again."""
        self.q_cmd = _qpos(self.task)[self.servo.idx].copy()

    def sampled(self) -> bool:
        if not self.ten_hz or self.last is None:
            return True
        return int(_np(self.planner.base_env.elapsed_steps).reshape(-1)[0]) % 2 == 0

    def step(self, T_goal, base_vw=(0.0, 0.0)):
        """One env step toward `T_goal` with the base at `base_vw` (m/s, rad/s).

        Returns `(res, dp, dr, sampled)`: the 5-tuple, the pose error the command was taken
        on, and whether this step carried a new command."""
        p = self.planner
        if not self.sampled():
            return p._step(self.last.copy()), self.dp, self.dr, False
        q = _qpos(self.task)
        self.q_cmd, self.dp, self.dr = self.servo.step(q, T_goal, self.q_cmd)
        _, body = p._hold_targets()
        body = np.asarray(body, dtype=np.float64).copy()
        body[2] = self.torso
        v, w = base_vw
        base = np.array([np.clip(v, -1.0, 1.0),
                         np.clip(w, -self.omega_scale, self.omega_scale) / self.omega_scale])
        action = p._compose(self.q_cmd, body, base)
        self.last = np.asarray(action, dtype=np.float64).copy()
        return p._step(action), self.dp, self.dr, True


def _torso(task) -> float:
    jm = task.agent.robot.active_joints_map
    return float(_qpos(task)[int(jm["torso_lift_joint"].active_index[0])])


def _fingers(task) -> float:
    return float(_qpos(task)[-2:].sum())


def _settled_gap(planner, task, steps: int = 6) -> float:
    planner.idle_steps(t=steps)
    return _fingers(task)


def _leaf_normal(task, door, anchor_xy, sense):
    """The leaf's outward face normal at its live angle (`push_frame`'s)."""
    _, n = crp.push_frame(anchor_xy, sense, crp._hinge_qpos(task, door), door)
    return np.asarray(n, dtype=np.float64)


def _lever(point_xy, q) -> tuple:
    """Where a world point stands in the base frame: (forward, left)."""
    yaw = float(q[2])
    d = np.asarray(point_xy, dtype=np.float64)[:2] - np.asarray(q[:2], dtype=np.float64)
    return (float(d @ np.array([np.cos(yaw), np.sin(yaw)])),
            float(d @ np.array([-np.sin(yaw), np.cos(yaw)])))




def _swing(env, planner, task, door, drv, servo, *, G0, rad0, anchor_xy, s, start, target,
           lever_a, lever_b, res, label):
    """Ride the held leaf from `start` toward `target` (how open, either way), BIMBO's door_swing.

    Every control the commanded angle `sw` moves LEAD_RATE * dt toward `target`, never more
    than LEAD_MAX ahead of the measured leaf; the hand is servoed to the grasp frame carried to
    `sw` (`door_frames`, its yaw turned by HAND_TURN of the door's); the base creeps so the goal
    stands at a lever going from `lever_a` to `lever_b` by LEVER_LEAD_FRAC of the way, then
    settles there. Stops on the target, an empty hand, a stall, or SWING_MAX_STEPS.

    Returns `(res, reason, goal)`; `res` is None when the horizon struck (the caller returns
    the solver's last step).
    """
    direction = 1.0 if target >= start else -1.0
    span = max(abs(target - start), 1e-6)
    dt = float(planner.base_env.control_timestep)

    def lever_at(sw_now: float) -> tuple:
        k = float(np.clip(abs(sw_now - start) / (LEVER_LEAD_FRAC * span), 0.0, 1.0))
        return (lever_a[0] + k * (lever_b[0] - lever_a[0]),
                lever_a[1] + k * (lever_b[1] - lever_a[1]))

    def frame(sw_now: float) -> np.ndarray:
        turn = s * (sw_now - rad0)
        return ds.door_frames(G0, anchor_xy, turn, hand_turn=HAND_TURN * turn)[1]

    say(env, f"{label} the door (bimbo)", target_rad=round(target, 3), start_rad=round(start, 3),
        lead_rate=LEAD_RATE, lead_max=LEAD_MAX, hand_turn=HAND_TURN,
        lever=[round(v, 3) for v in lever_a], lever_end=[round(v, 3) for v in lever_b],
        arm_hold=ARM_HOLD, roll_cost=ROLL_COST, roll_hold=ROLL_HOLD,
        hinge=[round(float(v), 3) for v in anchor_xy], sense=int(s), ten_hz=drv.ten_hz)
    sw = float(start)
    goal = frame(sw)
    q = _qpos(task)
    q_arm0, q_base0 = q[servo.idx].copy(), q[:3].copy()
    hist: list = []
    trace: list = []
    reason = "max-steps"
    perr_max = rerr_max = 0.0
    i = -1
    for i in range(SWING_MAX_STEPS):
        rad = float(crp.door_rad_now(task, door))
        hist.append(rad)
        if direction * (rad - target) >= 0.0:
            reason = "target"
            break
        if (len(hist) > STALL_STEPS
                and direction * (hist[-1] - hist[-1 - STALL_STEPS]) < STALL_EPS):
            reason = "stalled"
            break
        if direction > 0:
            sw = min(target + 0.02, sw + LEAD_RATE * dt, max(sw, rad + LEAD_MAX))
        prev = goal
        goal = frame(sw)
        q = _qpos(task)
        pvel = (goal[:2, 3] - prev[:2, 3]) / dt
        v, w, ax, ay = ds.unicycle_hold_cmd(goal[:2, 3], pvel, q[:2], float(q[2]), lever_at(sw),
                                            gain=BASE_GAIN, v_max=BASE_V_MAX, w_max=BASE_W_MAX)
        res, dp, dr, fresh = drv.step(goal, (v, w))
        if fresh:
            perr_max = max(perr_max, float(np.linalg.norm(dp)))
            rerr_max = max(rerr_max, float(np.linalg.norm(dr)))
        if i % 10 == 0:
            hand = ds.angle_of(anchor_xy, G0[:2, 3], _T(task.agent.tcp.pose.sp)[:2, 3], s) + rad0
            trace.append([i, round(sw, 3), round(rad, 3), round(hand, 3),
                          round(float(np.linalg.norm(dp)), 3), round(float(np.linalg.norm(dr)), 3),
                          round(v, 3), round(w, 3), round(ax, 3), round(ay, 3)])
    # -- the stance settles: the goal held on the bar, the creep runs on to lever_b -----------
    settle, lever_err = 0, 0.0
    if reason == "target" and LEVER_END[0] > 0:
        for settle in range(1, SETTLE_STEPS + 1):
            q = _qpos(task)
            v, w, ax, ay = ds.unicycle_hold_cmd(goal[:2, 3], np.zeros(2), q[:2], float(q[2]),
                                                lever_b, gain=BASE_GAIN, v_max=BASE_V_MAX,
                                                w_max=BASE_W_MAX)
            lever_err = float(np.hypot(ax - lever_b[0], ay - lever_b[1]))
            if lever_err < SETTLE_TOL_M:
                break
            res, _, _, _ = drv.step(goal, (v, w))
    rad = float(crp.door_rad_now(task, door))
    hand_rad = ds.angle_of(anchor_xy, G0[:2, 3], _T(task.agent.tcp.pose.sp)[:2, 3], s) + rad0
    q_end = _qpos(task)
    say(env, f"{label} done (bimbo)", rad=round(rad, 3), door_from_hand=round(hand_rad, 3),
        why=reason, steps=i + 1, perr_max=round(perr_max, 3), rerr_max=round(rerr_max, 3),
        arm_moved=[round(float(v), 3) for v in q_end[servo.idx] - q_arm0],
        base_moved=[round(float(v), 3) for v in q_end[:3] - q_base0],
        base_end=[round(float(v), 3) for v in q_end[:3]], settle=settle,
        lever_err=round(lever_err, 3),
        trace="[i, sw, rad, hand, perr, rerr, v, w, a_x, a_y] " + str(trace[-16:]),
        **({"door_tail": [round(v, 3) for v in hist[-20:]]} if reason != "target" else {}))
    return res, reason, goal


def open_the_door(env, planner, task, *, door, park_rad: float | None = None,
                  record: dict | None = None):
    """Grasp the leaf's bar, swing it open with the servoed hand, let go and clear the hand.

    Same contract as `cabinet_retrieval_planner.open_the_door` (D6): -1 only for a refusal
    before the fingers first closed on the bar (dock, pre-grasp, stroke); afterwards the last
    gym 5-tuple, however far the leaf got — the caller judges the opening by state
    (`door_rad_now`). `record` is filled as the shipped stage fills it (the retrace reads it).

    Args:
        env, planner, task: as everywhere.
        door: the leaf (`DoorSpec`).
        park_rad: a wall-park target (cab_1); the swing then runs to the larger of it and
            `DOOR_OPEN_TARGET` on the same grip.
        record: optional dict for `retrace_the_door`.

    Example:
        >>> res = open_the_door(env, planner, task, door=CAB_MAIN_RIGHT)   # doctest: +SKIP
    """
    target = float(crp.DOOR_OPEN_TARGET if park_rad is None
                   else max(crp.DOOR_OPEN_TARGET, float(park_rad)))
    dock = np.array([door.handle_dock[0], door.handle_dock[1], 0.0])
    res, done = crp.grasp_the_bar(env, planner, task, door, bar=door.handle_bar, dock=dock)
    servo = HandServo(task, planner)
    gap_fk = servo.fk_gap()
    art_key = crp._art_key(task, door)
    anchor_xy, sense = common.hinge_anchor(task, art_key, door.hinge)
    anchor_xy = np.asarray(anchor_xy, dtype=np.float64).reshape(-1)[:2]
    s = float(sense) * float(door.open_dir)          # yaw sense of opening
    drv = _Driver(planner, task, servo, _torso(task))

    # -- door_grasp: close, then keep the grasp only if the gap is the bar's ----------------
    gap = 0.0
    for attempt in range(2):
        r = planner.close_gripper(t=12)
        if r != -1:
            res = r
        gap = _settled_gap(planner, task)
        held = GRASP_GAP[0] < gap <= GRASP_GAP[1]
        ag = getattr(task, "_ag", None)
        say(env, "bar grasp", attempt=attempt, finger_gap=round(gap, 4), held=held,
            weld=(None if ag is None else int(ag.held_index())))
        if held or attempt == 1:
            break
        # BIMBO: the fingers closed on the edge or on nothing — open and go again, here a
        # little deeper along the approach (the stroke stops at the FIRST touch of the
        # cabinet, which can be a fingertip on the bar's flank).

    # -- door_swing: the lead angle, the servoed hand, the creeping base ---------------------
    drv.torso = _torso(task)
    drv.reset_command()
    servo.hold_rolls_here()
    G0 = _T(task.agent.tcp.pose.sp)                  # the hand on the bar, as it closed
    rad0 = float(crp.door_rad_now(task, door))
    q = _qpos(task)
    lever = _lever(G0[:2, 3], q)
    lever_end = lever if LEVER_END[0] <= 0 else (LEVER_END[0], s * LEVER_END[1])
    yaw0 = float(q[2])
    res, reason, goal = _swing(env, planner, task, door, drv, servo, G0=G0, rad0=rad0,
                               anchor_xy=anchor_xy, s=s, start=rad0, target=target,
                               lever_a=lever, lever_b=lever_end, res=res, label="swing")
    if record is not None:
        record["bimbo"] = dict(G0=G0.copy(), rad0=rad0, anchor_xy=anchor_xy.copy(), s=s,
                               lever=lever, lever_end=lever_end)

    # -- door_hold: the arm relaxes to where it stands, the weld holds the door --------------
    # The servo's command runs ahead of the measured joints while it pulls (ARM_LEAD_MAX); let
    # go of that pull before the fingers open, or the arm springs back as the weld lets go.
    res = planner.idle_steps(t=HOLD_STEPS)
    planner.planner.update_from_simulation()

    if record is not None:
        _pose = task.agent.base_link.pose.sp
        _bx = _T(_pose)[:3, 0]
        _tcp = task.agent.tcp.pose.sp
        record.update(
            stem=door.stem, hinge=door.hinge,
            base_xy=_np(_pose.p).reshape(-1)[:2].astype(np.float64).copy(),
            base_yaw=float(np.arctan2(float(_bx[1]), float(_bx[0]))),
            hinge_qpos=crp._hinge_qpos(task, door), door_rad=crp.door_rad_now(task, door),
            anchor=(anchor_xy.copy(), int(sense)),
            tcp_p=np.asarray(_tcp.p, dtype=np.float64).copy(),
            tcp_q=np.asarray(_tcp.q, dtype=np.float64).copy(),
            arm_qpos=_qpos(task).copy())
        record["tape"] = planner.start_tape()

    # -- door_release ------------------------------------------------------------------------
    n = _leaf_normal(task, door, anchor_xy, sense)
    hz = _T(task.agent.tcp.pose.sp)[:3, 2]
    off_normal = float(-hz[:2] @ n)
    rad_held = float(crp.door_rad_now(task, door))
    r = planner.open_gripper()
    if r != -1:
        res = r
    rad_rel = float(crp.door_rad_now(task, door))
    say(env, "released the handle (bimbo)", door_rad_held=round(rad_held, 4),
        door_rad_now=round(rad_rel, 4), off_normal=round(off_normal, 3),
        fingers=round(_fingers(task), 4))

    # -- door_clear: off the bar on a path that never closes on the panel --------------------
    T = _T(task.agent.tcp.pose.sp)
    n = _leaf_normal(task, door, anchor_xy, sense)
    bar0 = np.asarray(door.handle_bar, dtype=np.float64)[:2]
    r_bar = max(float(np.linalg.norm(bar0 - anchor_xy)), 1e-6)
    edge0 = anchor_xy + (bar0 - anchor_xy) * (1.0 + LEAF_EDGE_PAST_BAR / r_bar)
    leaf_a, leaf_b = ds.leaf_segment(anchor_xy, edge0, s * rad_rel)   # turned from shut
    d, rule = ds.clear_direction(T[:3, 2], n, lift_off=CLEAR_LIFT_OFF)
    lift = np.array([0.0, 0.0, CLEAR_LIFT_M])
    end = T[:3, 3] + CLEAR_M * d + lift
    goal = T.copy()
    goal[:3, 3] = end
    drv.reset_command()
    rad_start = float(crp.door_rad_now(task, door))
    moved = 0.0
    k = 0
    for k in range(CLEAR_MAX_STEPS):
        res, dp, _, fresh = drv.step(goal)
        moved = float(crp.door_rad_now(task, door)) - rad_start
        if abs(moved) > CLEAR_LEAF_TOL or (fresh and float(np.linalg.norm(dp)) < 0.01):
            break
    planner.planner.update_from_simulation()
    T1 = _T(task.agent.tcp.pose.sp)
    say(env, "clear the open hand (bimbo)", rule=rule, d=[round(float(v), 3) for v in d],
        travelled=round(float(np.linalg.norm(T1[:3, 3] - T[:3, 3])), 3), steps=k + 1,
        leaf_moved=round(moved, 4), panel_gap=round(ds.seg_dist(T1[:2, 3], leaf_a, leaf_b), 3),
        door_rad=round(float(crp.door_rad_now(task, door)), 3),
        touching=planner.touching_now())
    if record is not None:
        planner.stop_tape()
        _end = task.agent.base_link.pose.sp
        _ex = _T(_end)[:3, 0]
        record["base_xy_end"] = _np(_end.p).reshape(-1)[:2].astype(np.float64).copy()
        record["base_yaw_end"] = float(np.arctan2(float(_ex[1]), float(_ex[0])))
    return res




CLOSE_BY_BAR = os.environ.get("MIKASA_BD_CLOSE_BY_BAR", "0") == "1"
"""Whether the search's closing may hand over to `close_by_the_bar`. OFF, measured 2026-10-16 on
the three rounds that needed it (seed 9 with assisted grasp, seeds 2 and 9 without): the hand
gets back to the bar (2.4 cm out along the face's normal, 1 mm along it) but the fingers close on
nothing. With the hand turned ~100 deg to the leaf the pads close along the face's NORMAL, which
is the direction the bar moves when the free leaf turns: the first pad to touch sweeps the bar
away (the leaf opens 0.03-0.05 rad) and the pads meet on air. The opening's grasp closes ALONG
the leaf, where the hinge holds the bar still."""

CLOSE_TARGET_RAD = 0.004
"""The swing back stops once the leaf reads this shut; its command runs on to 0.02 under it,
so the hand presses the leaf onto its closed stop (the hinge's limit at 0)."""
CLOSE_BACK_M = 0.30
"The withdrawal after the release at the shut leaf, as the shipped physical close's."
REGRASP_REACH_M = CLEAR_M + 0.10
"The hand must stand within this of the bar to take it again (it left by CLEAR_M)."
REGRASP_OFF_FACE_M = 0.02
"""The bar is taken again with the TCP 2 cm out from its axis along the face's normal. The open
pads stand 5 cm either side of the TCP and the bar 5 cm off the face, so this puts the inner
pad mid-way in the 3.7 cm between the bar's surface and the face (1 cm brushed the face)."""
