"""Oracle for MikasaDepthRecall-v1, built from extraction and restoration flows.

Each transfer is one of the two strokes this branch measured at 100/100:

- **extract** (shelf -> counter slot) is `cabinet_retrieval_planner`'s straight flow,
- **restore** (counter slot -> shelf) uses the shelf-placement helpers extracted
  from `cabinet_stow_planner` into `planners.oracle.cabinet_place`.

The default four-prop row requires seven transfers: four extractions and three
restorations. The separate CabinetStow environment is not needed by this oracle.

Nothing is re-derived: the docks, `GRASP_LIFT`, the depth ladder, `ready_by_line` and
its pre-checks, the loaded hold and the lift-off all come from those files. What is new
here is only the sequencing and the one lateral fact that makes it work — with the base
docked at the object's own x, the arm reaches sideways far enough that NO transfer ever
carries a prop while the base moves (measured 2026-09-10: onto the counter 36 cm toward
smaller x, into the shelf 36 cm toward larger x, both 5/5).

**Where each prop is staged** is the pad of its own colour: read off the task in simulator
mode (scenery, like the slot points), by colour from the head cameras in perception mode
(`PropBelief.identify_pads`).

**Where the answer is read, and it is one line.** The extraction order is taken from the
props' own poses — front first, which physics enforces anyway — so it needs no
privileged information. The only privileged read is the RESTORE assignment: which staged
prop belongs in which row slot. `choose_assignment` reads `task.original_slots` when
sighted and draws when blind, and `--blind` swaps exactly that. The coins come from
`default_rng(seed + 10_007)`, never from the episode RNG, or they alias the ticket
(CabinetSearch's recorded trap).

Contract: `solve(env, seed=None, debug=False, vis=False, blind=False)` returning -1 (no
plan) or the gym 5-tuple, per `template_planner.py`.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import sapien

from planners import cabinet_retrieval_planner as retr
from planners import depth_recall_v1_perception as perc
from planners.oracle import cabinet_place as sp
from planners.oracle import oracle_common as common
from utils.mikasa.seeding import seed_everything
from utils.mikasa.waypoint_noise import WaypointNoise

WHO = "depth_recall_v1_planner"

#: Set by `solve(..., perception="pointcloud")`: the planner's belief about the props,
#: built from its own cameras (`depth_recall_v1_perception`). None = read the simulator.
_BELIEF = None

#: Joint-space step (move-group norm, rad) at which every joint line this planner executes
#: is collision-checked. `plan_qpos_line`'s 0.1 is up to ~8 cm of hand travel on an
#: outstretched arm, more than a 4.5 cm prop: on seeds 1070 and 1092 (2026-10-15) a line
#: accepted at 0.1 put a finger through the counter prop it was reaching for, and the prop
#: fell over (a 0.01 check refuses the same line).
READY_LINE_STEP = 0.01


def _ready_by_line(*args, **kwargs):
    """`retr.ready_by_line` with every line checked at `READY_LINE_STEP`."""
    kwargs.setdefault("line_step", READY_LINE_STEP)
    return retr.ready_by_line(*args, **kwargs)


#: Perception mode re-docks when the measured row is further than this from the map's
#: row: the dock's own waypoint noise is 5 mm, the row shift up to 4 cm.
REDOCK_MIN_M = 0.01

#: A refused re-dock is survivable up to this shift: the episode then works from the
#: first dock. `drive_base` turns the base even for a sideways centimetre, and measured
#: 2026-10-16 (seed 3011, shift 10.4 mm) that turn's sweep hit the open right door with
#: the forearm; 2 cm aside is well inside the pads' 18 cm pitch.
REDOCK_SKIP_MAX_M = 0.02


def _grasping(task, prop) -> bool:
    """Is ``prop`` in the hand? Contact forces in simulator mode; in perception mode the
    finger encoders (`PropBelief.fingers_holding`), as a robot would tell."""
    if _BELIEF is None:
        return bool(task.agent.is_grasping(prop).any())
    return _BELIEF.fingers_holding()


def _hold(env, planner, task, prop, held: bool) -> None:
    """`common.hold_object_in_planner`; in perception mode the prop is attached at the
    transform the belief has (the grasp's estimate), not the simulator's."""
    pose = _BELIEF.link_to_prop(_index(task, prop)) if (_BELIEF is not None and held) else None
    common.hold_object_in_planner(env, planner, task, prop, held=held, who=WHO, pose=pose)


def _seen(prop):
    """What a pose is computed from: the prop itself, or its perceived stand-in."""
    return prop if _BELIEF is None else _BELIEF.proxy(prop)


def _index(task, prop) -> int:
    return list(task.props).index(prop)

#: Straight-up clearance off the shelf after a grasp, and off the counter after one.
#: Imported, not re-derived: this is the retrieval flow's own number.
UNSHELVE_DZ = retr.UNSHELVE_DZ

#: Metres the unshelve ALSO travels back along the approach axis, so the leg separates
#: the fingers from the next prop in the row instead of sliding up its face.
#:
#: Measured 2026-09-10, first run of this task: the row pitch is 5.5 cm and a prop is
#: 4.5 cm across, so 1 cm of air separates two bodies — and the finger plates reach
#: ~3 cm past the TCP along the approach. A pure +z unshelve was refused at step 1 with
#: `l_gripper_finger_link <-> prop_1`, `0.005 of the twist left`, on every draw. Going
#: up AND back moves the fingers off the neighbour's face from the first knot.
UNSHELVE_BACK = float(os.environ.get("MIKASA_DR1_UNSHELVE_BACK", "0.06"))

#: How far in front of the cabinet's face the carried prop's front stands, at least,
#: before it crosses to its pad, metres.
CARRY_CLEAR = 0.05


def carry_out(base_y: float, prop_y: float, face_y: float, half: float) -> float:
    """How far the extraction backs the base off with the prop in the hand, metres.

    Back to the dock line, the stand the restore later takes the prop off its pad from, so
    the place and that grasp are made from one stand; further only if the prop's front
    would not yet stand `CARRY_CLEAR` in front of the cabinet's face (the deepest row
    slot: 2 cm more). ``base_y`` and ``prop_y`` are where they are now, after the unshelve.

    The old rule backed off until the prop stood over the pads' line, which leaves the
    base 0.96 m behind a pad for a prop from the deepest slot. Measured 2026-10-16 with
    cuRobo's scan of the planner's own poses (`scripts/dr_pad_reach.py`, every grasp lift,
    the row's +-4 cm jitter, self-collision, hits re-checked in float64): from there the
    place reaches 44 cm west of the row; from the dock line 62 cm.
    """
    return max(base_y - retr.READY_DOCK_Y, prop_y + half - (face_y - CARRY_CLEAR))


#: Where the TCP sits above a prop's mid-height, tried in order.
#:
#: The retrieval family's single `GRASP_LIFT` (0.02) exists to keep `wrist_flex_link`,
#: which hangs 6.7 cm under the hand's axis, off the surface — and on a 12 cm prop it
#: leaves only 1.3 cm. Measured 2026-09-10, seed 9: every one of 75 IK solutions for a
#: counter grasp was refused with `gripper_link <-> counter_main`, because the props
#: already standing beside it had taken away the branches that clear. The higher rungs
#: are `depth_recall_planner`'s own measured top band (its `GRASP_Z_LADDER` is
#: 0.045/0.055/0.035), where the prop's own body is the support under the pads.
GRASP_LIFTS = tuple(
    float(v) for v in os.environ.get("MIKASA_DR1_GRASP_LIFTS", "0.02,0.035,0.045").split(",")
)

#: Half the finger pads' height in this grasp, metres: the finger links' collision boxes
#: span +-1.32 cm about the TCP along gripper_link's x (robots/fetch/fetch.urdf meshes),
#: which the side grasp turns vertical.
FINGER_HALF_H = 0.0132

#: How far `wrist_flex_link` hangs under the hand's axis in the side grasp, metres, and
#: the clearance over the support the box's first rung left it (TCP 8.0 cm up, 1.3 cm).
WRIST_HANG = 0.067
WRIST_CLEAR = float(os.environ.get("MIKASA_DR1_WRIST_CLEAR", "0.012"))


def grasp_lifts(task, prop) -> tuple:
    """`GRASP_LIFTS` for this prop, each rung clipped into the band where the grasp is
    sound, duplicates dropped. -> (lift over mid-height, ...), metres.

    The band, as TCP height over the support the prop stands on: at least
    `WRIST_HANG` + `WRIST_CLEAR` (7.9 cm), or the wrist meets the support; at most the
    prop's height less `FINGER_HALF_H`, or the finger pads' top edge stands above the
    prop. The ladder was measured on the 12 cm box, which it fits unchanged (8.0 -
    10.5 cm in a band of 7.9 - 10.7), and so do the 15 - 16.5 cm bottles. The 9.45 cm
    shakers do not: their band is 7.9 - 8.1 cm, and the 2 cm rung (TCP 6.7 cm up) is the
    one that, measured 2026-10-16 (seed 1000), grasped a shaker off the counter and then
    could not stand it in the front row slot: the wrist met the cabinet's front lip on
    the way down, the hand stopped 1.1 cm short, and the released shaker fell into the
    row and knocked over the two restored before it.

    Example:
        >>> grasp_lifts(task, task.props[1])        # a shaker  # doctest: +SKIP
        (0.0318, 0.034)
    """
    spec = task.cfg.props[_index(task, prop)]
    mid = spec.height / 2.0
    lo = WRIST_HANG + WRIST_CLEAR - mid
    hi = spec.height - FINGER_HALF_H - mid
    out = []
    for lift in GRASP_LIFTS:
        lift = round(min(max(float(lift), lo), hi), 4)
        if lift not in out:
            out.append(lift)
    return tuple(out)


#: The base's coast after `drive_straight`, metres per control step, below which it
#: counts as stopped; and the most steps to wait for that.
BASE_STILL_M = 0.0005
BASE_SETTLE_MAX = 20

#: How much further than asked the base goes on a 0.3 m drive into the cabinet at
#: 0.10 m/s, metres: the drive's own stop (+1-3 mm) and the coast after it (1.1-1.2 cm).
#: Measured 2026-10-16 (seeds 1000, 1001): 1.15, 1.44 cm. The LOADED drive asks this much
#: less: the held prop rides over its row slot with the fingertips 0.4-0.9 cm past its
#: back face, 1.8-2.3 cm short of the next prop in the row, and on seed 1001 the
#: overshoot put a finger into that prop (11 N) and knocked it and the one behind it
#: over. The descent that follows is planned after `settle_base` and corrects the
#: millimetres left either way.
LOADED_DRIVE_COAST = float(os.environ.get("MIKASA_DR1_LOADED_COAST", "0.013"))


def settle_base(planner, task):
    """Idle until the base has stopped, then re-read the planning world. -> the last
    5-tuple, or None when no step was needed.

    `drive_straight` stops commanding at the distance, and the base coasts on: measured
    2026-10-16 (seed 1000) 1.1 cm over the next ~9 steps after the 0.30 m drive into the
    cabinet. A stroke planned at the drive's end is planned from a base that is still
    moving, and the hand overshoots by the coast: the grasp stroke ended 1.7 cm past its
    goal and the fingertips pushed the next shaker in the row 3.6 cm aside (it fell
    on the unshelve), and every row placement landed 1.2-1.4 cm deeper than its slot. The
    4.5 cm boxes had 1.1 cm more room in the row, which is why this never showed.
    """
    res = None
    prev = _np(task.agent.base_link.pose.p).reshape(-1)[:2].copy()
    for _ in range(BASE_SETTLE_MAX):
        res = planner.idle_steps(t=1)
        if common.stopped_by_horizon(planner):
            return res
        now = _np(task.agent.base_link.pose.p).reshape(-1)[:2].copy()
        moved = float(np.linalg.norm(now - prev))
        prev = now
        if moved < BASE_STILL_M:
            break
    planner.planner.update_from_simulation()
    return res

#: Torso heights the lateral cross is tried at, in order. `None` means "wherever the
#: carry-out left it" — the first rung is free, the rest cost a joint line.
#:
#: The ladder stops at 0.20 on purpose: there the carried prop's base rides at 1.25,
#: which is 21 cm over the tops of the props already standing on the slots. Going lower
#: buys reach and starts sweeping them.
CROSS_TORSO_RUNGS = (None, 0.30, 0.20)

#: The cross's rungs, in order: (torso, how far the HAND goes down from the carry
#: height). A torso rung lowers the hand WITH the shoulder, so it changes what the
#: carried prop can hit but not what the arm can reach; a hand drop lowers the hand
#: UNDER the shoulder, and that is what the reach to a far pad needs. Measured
#: 2026-10-16, seed 1000 with the pads 54 cm west: every torso rung was `IK Failed`
#: for the cross, while cuRobo's scan (torso free, so it kept the torso up) reached the
#: same heights. The drops stop at 0.20, the old floor: the carried prop's base still
#: rides 21 cm over the props standing on the other pads. The torso rungs stay as the
#: last resort, for a cross that a drop makes collide.
CROSS_RUNGS = ((None, 0.0), (None, 0.10), (None, 0.20)) + tuple((t, 0.0) for t in CROSS_TORSO_RUNGS[1:])

#: A pad (or a prop on one) further than this west of the base is FAR, and both crosses
#: to it start with the rungs that lower the hand under the shoulder; nearer ones keep
#: the order that planned first there. Measured 2026-10-16 on 100 seeds (68c5f27): to the
#: pad 54 cm west the extraction's first rung (hand where it is) was refused 100/100 and
#: the 10 cm drop planned 99/100; to the pads at 36 cm and nearer the first rung planned
#: 100/100. The restore's hand-height rungs planned at 36 and 54 cm and were refused for
#: the nearer props (3.8 refused rungs per episode, each an IK search and two retries):
#: that is most of the 30 % the wall clock grew over the 12 cm version.
FAR_PAD_M = 0.45


def extraction_cross_rungs(offset: float) -> tuple:
    """`CROSS_RUNGS` ordered for a pad ``offset`` metres west of the base."""
    if offset <= FAR_PAD_M:
        return CROSS_RUNGS
    drops = tuple(r for r in CROSS_RUNGS if r[0] is None and r[1] > 0.0)
    return drops + tuple(r for r in CROSS_RUNGS if r not in drops)


def restore_cross_rungs(offset: float, z0: float) -> list:
    """(torso, hand z) rungs of the restore's cross to a prop ``offset`` metres west of the
    base, over a counter whose top is at ``z0``: hand heights first when it is far, the
    torso rungs first when it is near."""
    heights = [(None, z0 + dz) for dz in RESTORE_CROSS_Z]
    torsos = [(t, None) for t in CROSS_SAFE_TORSO]
    first, then = (heights, torsos) if offset > FAR_PAD_M else (torsos, heights)
    return first + then + [(None, None)]

#: Joint-space step the TORSO lines are planned with, metres.
#:
#: `plan_qpos_line`'s default is 0.1, which for a 0.386 m torso ride is four knots and
#: therefore four control steps — about 9.6 cm asked per step. Under
#: `pd_joint_delta_pos` the channel caps at 0.1 per step, so the command SATURATES:
#: measured 2026-09-10 on seed 0, the torso action sat at |a| = 1.000 on 1.1 % of the
#: episode's steps while the joint actually moved at most 3.35 cm. That is the "base
#: dropping sharply" in the clip, and at the 10 Hz recording clock it gets worse, not
#: better. 0.02 asks ~2 cm a step, a fifth of the channel.
#:
#: The arm never clips at either step size (0.000 % on all seven channels, same run).
TORSO_LINE_STEP = float(os.environ.get("MIKASA_DR1_TORSO_STEP", "0.02"))

#: Largest torso move issued as ONE joint line, metres; longer rides are chunked.
#:
#: The line's own knots are already fine (22-26 for a 0.386 m ride, ~1.5 cm each), so
#: the per-step speed is set by the follower, not the plan — and the follower is shared
#: with every other task here. Chunking is the part this oracle owns: a short line
#: cannot build up the same advance.
TORSO_CHUNK = float(os.environ.get("MIKASA_DR1_TORSO_CHUNK", "0.08"))

#: Torso heights the RESTORE's lateral cross is tried at, in order.
#:
#: The cross has to thread a band, and both walls are measured. Too HIGH and it hits
#: the closed LEFT door: seed 18 crossed 36 cm west with the torso at its 0.386 stop,
#: which puts the hand near z 1.5, and the door's panel occupies z 1.39-2.31 below
#: x = 2.25 — every posture was refused with
#: `l_gripper_finger_link <-> cab_main..._hingeleftdoor`. Too LOW and it sweeps the
#: props already standing on the slots, whose tops are 12 cm above the counter: seed 6
#: tipped the prop it was on its way to. The middle rungs put the hand between the two,
#: and the previous height is kept as the last resort so nothing that used to plan
#: stops planning.
CROSS_SAFE_TORSO = tuple(
    float(v) for v in os.environ.get("MIKASA_DR1_CROSS_TORSO", "0.15,0.20,0.10,0.25").split(",")
)

#: Hand heights over the counter top for the RESTORE's cross, tried first, with the
#: torso left where the previous transfer put it (up). `CROSS_SAFE_TORSO` lowers the
#: torso and the hand rides down with it, so the hand stays as high over the SHOULDER as
#: it was, and a pad 54 cm west is then out of reach: measured 2026-10-16, seed 1000,
#: `IK Failed` at all four torso rungs, after which the approach "from where we are"
#: tipped the prop over and grasped it lying (tilt 89.8). A hand 25-35 cm over the
#: counter is under the closed left door's panel (z >= 1.39) and, with the wrist hanging
#: 6.7 cm under it, still over the props standing on the other pads (tops at +12 cm).
RESTORE_CROSS_Z = tuple(
    float(v) for v in os.environ.get("MIKASA_DR1_RESTORE_CROSS_Z", "0.30,0.25,0.35").split(",")
)

#: How far in front of the prop (toward the robot) the restore's cross ends, metres:
#: the side grasp's pre-grasp stands 12 cm back along the approach, and 3 cm more.
RESTORE_CROSS_BACK = 0.15

#: Roll joints, by name. Their travel is what `roll_rank` minimises.
ROLL_JOINTS = ("upperarm_roll_joint", "forearm_roll_joint", "wrist_roll_joint")

#: Torso rungs, imported from the stow flow rather than copied: the counter reach and
#: the loaded shelf reach are the same two ladders it measured.
COUNTER_TORSO_RUNGS = sp.GRASP_TORSO_RUNGS

#: The counter GRASP gets two rungs above the stow flow's, because it has an obstacle
#: the stow flow never has: the props already standing on the other slots. Seed 6 puts
#: one in the dock's own column, and the reach to a prop 24 cm west then has no IK at
#: 0.0/0.10/0.20 — the arm has to come OVER it, not around it.
COUNTER_GRASP_TORSO_RUNGS = tuple(COUNTER_TORSO_RUNGS) + (0.30, retr.READY_TORSO)
SHELF_TORSO_RUNGS = sp.RISE_TORSO_RUNGS

#: Coin offset for the oracle's own draws. NEVER the episode RNG: `seed_everything`
#: puts numpy's global stream in the ticket's state, so an unoffset draw aliases the
#: answer (CabinetSearch, blank I).
COIN_OFFSET = 10_007


def say(env, stage: str, **extra):
    """Trace one stage to stdout and the episode log."""
    return common.say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """Print the refusal and return -1 — the contract's no-plan sentinel."""
    return common.fail(env, WHO, stage, **extra)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _b(info, key: str) -> bool:
    return bool(_np(info[key]).reshape(-1)[0])


def roll_rank(task):
    """An IK ranking that puts the least ROLL travel first, `goal_order` as the tiebreak.

    Why this exists, measured 2026-09-10 on seed 0: the planning roll window is +-3.44
    rad, which is NARROWER than a full turn, so when a solution sits near the far edge
    the joint can only get there the long way round. The wrist rolled **-5.83 rad over
    166 steps** on one lateral cross — a near-complete revolution of the gripper with
    the hand translating, which is the "spinning like a wheel" in the clip — and its
    range over the whole episode was only [-3.15, 3.07], i.e. it was bouncing between
    the window's two ends. `goal_order` cannot see this: it ranks by the largest single
    joint's travel, and a 5.8 rad roll loses to nothing when the alternatives move the
    elbow further.

    Returns a callable for `ready_by_line(rank=)`.

    Example:
        >>> res = _ready_by_line(..., rank=roll_rank(task))   # doctest: +SKIP
    """
    amap = task.agent.robot.active_joints_map
    idx = []
    for name in ROLL_JOINTS:
        joint = amap.get(name)
        if joint is not None:
            idx.append(int(np.asarray(joint.active_index).reshape(-1)[0]))

    def rank(goals, cur):
        cur = np.asarray(cur, dtype=float).reshape(-1)

        def cost(g):
            g = np.asarray(g, dtype=float).reshape(-1)
            n = min(len(g), len(cur))
            roll = sum(abs(float(g[j]) - float(cur[j])) for j in idx if j < n)
            return (round(roll, 3), float(np.abs(g[:n] - cur[:n]).max()))

        return sorted(np.atleast_2d(goals), key=cost)

    return rank


def census(env, task, stage: str):
    """Trace every prop's position and tilt. A tilt that appears between two of these
    names the leg that caused it; a tilt read only at the end names nothing.

    Example:
        >>> census(env, task, "after the lift-off")    # doctest: +SKIP
    """
    say(env, f"census: {stage}", **{
        q.name: [round(float(v), 3) for v in _np(q.pose.p).reshape(-1)[:3]]
                + [round(common._tilt_deg(q.pose.sp.q), 1)]
        for q in task.props})


def _straight(env, planner, pose, *, stage: str, tries: int = 1):
    """The screw with the torso held, never an RRT path. -1 or the 5-tuple.

    Example:
        >>> res = _straight(env, planner, up, stage="unshelve")   # doctest: +SKIP
    """
    return common.arm_move(env, planner, pose, who=WHO, stage=stage, tries=tries,
                           disable_lift_joint=True, max_knots=1, knot_refuse=True)


def row_neighbours(task, prop):
    """The other props standing in the cabinet row right now, nearest first.

    They are what the fingers have to share 1 cm of air with (see `UNSHELVE_BACK`), and
    the legs that cannot avoid them take them out of the planning world for the stroke
    — honest, because those strokes are straight lines that move AWAY from the
    neighbour, and a straight screw cannot detour through the gap it opens.

    Example:
        >>> with neighbours_touchable(planner, row_neighbours(task, prop)):  # doctest: +SKIP
        ...     res = _straight(env, planner, up, stage="unshelve")
    """
    out = []
    for i, other in enumerate(task.props):
        if other is prop:
            continue
        if _BELIEF is not None and not _BELIEF.has(i):
            # Never seen yet: still standing in the row, behind what the camera sees.
            out.append(other)
            continue
        q = _np(_seen(other).pose.p).reshape(-1)
        # In the cabinet at all: above the shelf plane and north of its front face.
        if float(q[2]) > task.cfg.shelf_top_z - 0.05 and float(q[1]) > -0.40:
            out.append(other)
    return out


@contextlib.contextmanager
def neighbours_touchable(planner, props):
    """Nest `oracle_common.touchable` over several actors for one stroke."""
    with contextlib.ExitStack() as stack:
        for prop in props:
            stack.enter_context(common.touchable(planner, prop.name))
        yield


def fold_arm(env, planner, task):
    """Fold with recorded joint targets; never rewrite physical roll angles."""
    return retr.fold_arm_to_rest(env, planner, task)


def dock_at(env, planner, task, x: float, *, stage: str, noise=None):
    """Put the base in front of `x`, empty-handed, facing the counter. -1 or the tuple.

    Example:
        >>> res = dock_at(env, planner, task, 2.41, stage="dock at slot 2")  # doctest: +SKIP
    """
    # No fold and no duck. Both were needed when this oracle re-docked before every
    # transfer; with ONE dock they are a round trip to nowhere. The robot spawns with the
    # torso already at its 0.386 stop and the arm at the rest keyframe, and the first
    # thing the first extraction asks for is 0.386 — so ducking to 0.20 and back is the
    # "torso goes down and up again" a viewer sees at the top of the clip, and the fold
    # has nothing to fold.
    dock = np.array([float(x), retr.READY_DOCK_Y, 0.0])
    if noise is not None:
        dock = noise.point(stage, dock, axes=(True, True, False))
    say(env, stage, dock=[round(float(v), 3) for v in dock])
    res = planner.drive_base(target_pos=dock, target_view_vec=np.array([0.0, 1.0, 0.0]), freeze_arm=True)
    if res != -1:
        planner.planner.update_from_simulation()
        d_dock, dyaw = common.dock_error(task, (float(x), retr.READY_DOCK_Y, np.pi / 2))
        # With ONE dock per episode this is the only place a docking defect can be seen.
        say(env, "parked", d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))
    return res


def _torso_now(task) -> float:
    """The torso lift joint's position now, metres."""
    j = task.agent.robot.active_joints_map["torso_lift_joint"]
    return float(_np(task.agent.robot.get_qpos()).reshape(-1)[int(np.asarray(j.active_index).reshape(-1)[0])])


def torso_to(env, planner, task, rungs, *, label: str, sync: bool = True):
    """First rung that plans, reached in `TORSO_CHUNK` steps. Returns (res, rung).

    ``sync=False`` skips the planning world's re-read between chunks: required inside
    `common.touchable`, where an object taken out of the world makes mplib's
    `update_from_simulation` raise (measured: every episode of a sweep crashed).
    """
    amap = task.agent.robot.active_joints_map
    j = amap.get("torso_lift_joint")
    ji = int(np.asarray(j.active_index).reshape(-1)[0]) if j is not None else None
    for torso in rungs:
        here = (float(_np(task.agent.robot.get_qpos()).reshape(-1)[ji])
                if ji is not None else None)
        if here is None or abs(torso - here) <= TORSO_CHUNK:
            waypoints = [torso]
        else:
            n = int(np.ceil(abs(torso - here) / TORSO_CHUNK))
            waypoints = [here + (torso - here) * (k + 1) / n for k in range(n)]
        res = -1
        for w in waypoints:
            res = retr.plan_joints(env, planner, task, {"torso_lift_joint": float(w)},
                                   label=f"{label} {round(float(w), 3)}", line_only=True,
                                   qpos_step=TORSO_LINE_STEP)
            if res == -1:
                break
            if sync:
                planner.planner.update_from_simulation()
            if common.stopped_by_horizon(planner):
                return res, torso
        if res != -1:
            return res, torso
    return -1, None


def grasp_ladder(env, planner, task, prop, *, where: str, lift: float = None):
    """Screw to the side grasp over `SIDE_GRASP_DEPTHS`, close, verify. (res, ok).

    The depth ladder and `GRASP_LIFT` are the retrieval flow's measured ones; the mesh
    is re-read on every rung because a refused close can have moved the prop.
    """
    res = -1
    lift = retr.GRASP_LIFT if lift is None else float(lift)
    for depth in retr.SIDE_GRASP_DEPTHS:
        if _BELIEF is not None:
            # Re-fit from the stand the grasp is made from, wrist camera first: every
            # rung, because a refused close can have moved the prop.
            i = _index(task, prop)
            fit = _BELIEF.look(where, (perc.HAND_CAM,) + perc.BASE_CAMS, only=[i]).get(i, {})
            # THIS look must accept a fit: an older estimate (a dock fit, a release) is
            # exactly what a refused close may have made wrong.
            if not fit.get("accepted"):
                return fail(env, f"{where}: {prop.name} not fitted from the grasp stand",
                            why=fit.get("reasons")), False
        grasp, _pre = retr.side_grasp_pose(task, depth, lift=lift, obj=_seen(prop))
        if grasp is None:
            return fail(env, f"{where}: {prop.name} has no collision mesh"), False
        say(env, f"{where}: grasp stroke", prop=prop.name, depth=depth, lift=lift,
            grasp=[round(float(v), 3) for v in grasp.p])
        res = _straight(env, planner, grasp, stage=f"{where} grasp (depth {depth})")
        if res == -1:
            # The same straight stroke with the TARGET out of the planning world. This
            # is `oracle_common.try_grasp`'s middle rung and it exists because the
            # fingers have to close AROUND the object: mplib sees the approach as
            # `l_gripper_finger_link <-> prop_N` and refuses a grasp that is correct.
            # Measured 2026-09-10 on the four-prop row: that refusal is the whole of the
            # counter-grasp failure class, 4 of 30 episodes.
            with common.touchable(planner, prop.name):
                res = _straight(env, planner, grasp,
                                stage=f"{where} grasp (depth {depth}, target touchable)",
                                tries=2)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            planner.planner.update_from_simulation()
            continue
        if _BELIEF is not None:
            # Close only on something: the prop's own points have to be between the
            # pads (BIMBO_EvoX's rays-between-the-pads rule, from the cameras). A
            # close on air is a wasted rung that also nudges the prop.
            chk = _BELIEF.between_pads(_index(task, prop))
            say(env, f"{where}: between the pads", prop=prop.name, depth=depth, **chk)
            if not chk["ok"]:
                planner.planner.update_from_simulation()
                continue
        res = planner.close_gripper(t=12)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if _grasping(task, prop):
            say(env, f"{where}: in the gripper", prop=prop.name, depth=depth,
                tilt=round(common._tilt_deg(prop.pose.sp.q), 1))
            if _BELIEF is not None:
                _BELIEF.attach(_index(task, prop))
            planner.planner.update_from_simulation()
            return res, True
        say(env, f"{where}: close missed", prop=prop.name, depth=depth)
        planner.open_gripper()
        planner.planner.update_from_simulation()
    return fail(env, f"{where}: grasp {prop.name}", tried=list(retr.SIDE_GRASP_DEPTHS)), False


def release_and_clear(env, planner, task, prop, place):
    """Open over `place`, detach, lift off, drive the base clear, then tidy the arm.

    Order matters and each step of it was measured. The rise before the base moves is
    the stow flow's cure for the hollow success: a 4.5 cm prop between 10 cm pads has
    2.75 cm a side, and a screw carries a wrist rotation that eats it, so the pads leave
    UPWARD and the base does the horizontal part. The base then retreats BEFORE the
    torso rises or the arm folds, because after a row placement the hand is inside the
    cabinet, where both are refused against its shell — that chain cost 5 of the first
    10 episodes, every one of them as `dock at the slot ... returned -1` two stages
    later, once the un-folded arm made the next dock's rotation sweep hit a door.

    The base always returns to the ONE dock line the episode is run from, so there is no
    per-caller distance to pass.
    """
    res = planner.open_gripper(t=retr.RELEASE_RAMP + 6, ramp=retr.RELEASE_RAMP)
    if _BELIEF is not None:
        _BELIEF.release(_index(task, prop), where="released")
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    _hold(env, planner, task, prop, held=False)
    say(env, "released", prop=prop.name,
        at=[round(float(v), 3) for v in _np(prop.pose.p).reshape(-1)[:3]],
        tilt=round(common._tilt_deg(prop.pose.sp.q), 1))
    planner.planner.update_from_simulation()

    tcp = task.agent.tcp.pose.sp
    others = row_neighbours(task, prop)
    for dz in sp.RELEASE_LIFT_RUNGS:
        with common.touchable(planner, prop.name), neighbours_touchable(planner, others):
            up = _straight(env, planner,
                           sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + dz], q=tcp.q),
                           stage=f"hand off {prop.name} ({dz} m)", tries=2)
        if up != -1:
            res = up
            say(env, "hand off", prop=prop.name, rise=dz,
                tilt=round(common._tilt_deg(prop.pose.sp.q), 1))
            break
    else:
        say(env, "no rise planned; backing the base out anyway", prop=prop.name)
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()

    # Torso, then base, then fold — and the order is the whole lesson of this stage.
    #
    # The TORSO first, because it is the only one of the three that is safe in both
    # places: after a row placement the hand is inside the cabinet, where the torso is
    # already at its stop and the raise is a no-op; after a counter placement it lifts
    # the hand clear of props whose tops stand 12 cm above the surface.
    #
    # The BASE second. Moving it before the raise drags the still-low arm across the
    # counter: measured, that knocked the TARGET prop flat and the episode came back
    # `target_on_a_slot=False tilt_max=90.0` with every transfer otherwise clean.
    #
    # The FOLD last, and only once the base is back on the dock line, because folding
    # from inside the cabinet is refused against its shell.
    #
    # The prop just released is out of this check: the torso only lifts the hand straight
    # up, away from it. The rise above leaves 7 mm between the finger pads and a 12 cm
    # prop's top, which is less than what a camera estimate of it may be off by: measured
    # 2026-10-16, 13 of 100 perception episodes had this raise refused at knot 0
    # (`gripper_link <-> prop`), and once the base no longer backs away from a pad after a
    # counter placement, nothing else separated the two before the next transfer.
    #
    # UP only. After a row placement the torso is already at its 0.386 stop, and the
    # first rung's line is then empty, which `plan_joints` books as a refusal: the ladder
    # went on to 0.34 and LOWERED the open hand 4.6 cm back onto the prop just released.
    # Measured 2026-10-16 (seed 1001, both modes): a finger or the palm caught a 15.8 g
    # shaker's top (0.15-0.22 N) and slid it 4 cm, out of its slot or into the next one.
    # The 243 g box never moved for it.
    here = _torso_now(task)
    rungs = tuple(t for t in SHELF_TORSO_RUNGS if t > here + 0.002)
    up, torso = -1, round(here, 3)
    if rungs:
        with common.touchable(planner, prop.name):
            up, torso = torso_to(env, planner, task, rungs, label="torso up to", sync=False)
    if up != -1:
        res = up
        if common.stopped_by_horizon(planner):
            return res
    planner.planner.update_from_simulation()

    base_y = float(_np(task.agent.base_link.pose.p).reshape(-1)[1])
    to_line = float(retr.READY_DOCK_Y - base_y)
    back = planner.drive_straight(to_line, v=0.10)
    if back != -1:
        res = back
        if common.stopped_by_horizon(planner):
            return res
    planner.planner.update_from_simulation()

    # No fold. It existed to make the NEXT dock's rotation sweep clear the doors, and
    # there is no next dock: the base stays on this line for the whole episode.
    # `ready_by_line` starts the following transfer from wherever the arm is and picks
    # the nearest IK branch, so a fold buys nothing and costs a visible trip to rest —
    # and it refused often enough to fold between some transfers and not others, which
    # is what made the clip look inconsistent.
    say(env, "cleared", prop=prop.name, to_line=round(to_line, 3), torso=torso)
    return res


def extract_one(env, planner, task, prop, slot_xyz, *, noise=None):
    """Shelf -> counter slot: the retrieval straight flow, aimed at a chosen slot.

    `(res, ok)`. `ok` False with -1 is a refusal; with a 5-tuple it is a miss.
    """
    res, torso = torso_to(env, planner, task, SHELF_TORSO_RUNGS, label="torso up to")
    if res == -1:
        return fail(env, "raise the torso to the shelf"), False
    if common.stopped_by_horizon(planner):
        return res, False
    say(env, "torso up", torso=torso)

    # The posture the drive-in ends at, with the grasp screw pre-checked from there.
    base_p = _np(task.agent.base_link.pose.p).reshape(-1)
    drive = float(retr.WORK_DOCK_Y - base_p[1])
    chosen_lift = None
    lifts = grasp_lifts(task, prop)
    for lift in lifts:
        grasp, pre = retr.side_grasp_pose(task, retr.SIDE_GRASP_DEPTHS[0],
                                          lift=lift, obj=_seen(prop))
        if grasp is None:
            return fail(env, f"{prop.name} has no collision mesh"), False
        ready = sapien.Pose(p=[pre.p[0], pre.p[1] - drive, pre.p[2]], q=pre.q)
        if noise is not None:
            ready = noise.pose(f"shelf approach {prop.name}", ready, (True, False, True))
        say(env, "ready posture", prop=prop.name, drive=round(drive, 3), lift=lift)
        res = _ready_by_line(env, planner, task, ready, drive=drive, grasp=grasp,
                                 label=f"ready posture ({prop.name}, lift {lift})",
                                 touch_needle=prop.name, rank=roll_rank(task))
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res != -1:
            chosen_lift = lift
            break
        planner.planner.update_from_simulation()
    if chosen_lift is None:
        return fail(env, f"ready posture for {prop.name}", lifts=list(lifts)), False
    planner.planner.update_from_simulation()

    say(env, "drive into the cabinet", distance=round(drive, 3))
    res = planner.drive_straight(drive, v=0.10)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "drive into the cabinet"), False
    settled = settle_base(planner, task)
    if settled is not None:
        res = settled
        if common.stopped_by_horizon(planner):
            return res, False

    res, ok = grasp_ladder(env, planner, task, prop, where="shelf", lift=chosen_lift)
    if not ok:
        return res, False
    _hold(env, planner, task, prop, held=True)

    tcp = task.agent.tcp.pose.sp
    others = row_neighbours(task, prop)
    # Up AND back: the fingers leave the neighbour's face from the first knot. The
    # neighbours come out of the planning world for this one straight stroke; see
    # `row_neighbours` for why that is honest and `UNSHELVE_BACK` for the measurement.
    up = sapien.Pose(p=[tcp.p[0], tcp.p[1] - UNSHELVE_BACK, tcp.p[2] + UNSHELVE_DZ], q=tcp.q)
    say(env, "unshelve", prop=prop.name, back=UNSHELVE_BACK, up=UNSHELVE_DZ,
        neighbours=[o.name for o in others])
    with neighbours_touchable(planner, others):
        res = _straight(env, planner, up, stage="unshelve", tries=2)
        if res == -1:
            plain = sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + UNSHELVE_DZ], q=tcp.q)
            res = _straight(env, planner, plain, stage="unshelve (straight up)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"unshelve {prop.name}"), False
    if not _grasping(task, prop):
        say(env, f"MISSED: {prop.name} left the gripper at the unshelve")
        return res, False
    planner.planner.update_from_simulation()

    # Carry out: the base backs off to the dock line (`carry_out`). Loaded, so the hold is
    # latched (see `_drive_kw`).
    prop_y = float(_np(_seen(prop).pose.p).reshape(-1)[1])
    cab = task.cabinets[0]
    face_y = float(np.asarray(cab.pos)[1] - np.asarray(cab.size)[1] / 2.0)
    back = carry_out(float(_np(task.agent.base_link.pose.p).reshape(-1)[1]), prop_y, face_y,
                     float(task.cfg.props[_index(task, prop)].half_xy))
    say(env, "carry out", distance=round(back, 3))
    res = planner.drive_straight(-back, v=0.10, hold_pose=True)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "carry out of the cabinet"), False
    if not _grasping(task, prop):
        say(env, f"MISSED: {prop.name} left the gripper on the way out")
        return res, False
    planner.planner.update_from_simulation()

    # ACROSS to the slot's column, before the torso comes down.
    #
    # Two constraints pull against each other and `CROSS_TORSO_RUNGS` is where they
    # meet. High is safe — a prop carried at the shelf height clears the ones already
    # standing on the slots, whose tops are 12 cm above the counter — but the far slot
    # is 36 cm sideways and 0.82 m forward, 0.90 m of radius, and at the carry-out
    # height the IK simply has no solution (measured seed 3: `IK Failed` on the line,
    # the screw and the RRT alike). Lower is reachable but eventually sweeps the staged
    # props. So the cross walks DOWN a ladder and stops at the first rung that solves,
    # and the ladder stops at 0.20, where the carried prop's base still rides 1.25 —
    # 21 cm over the tallest thing on the counter.
    #
    # Coming south of the counter first and crossing down there was tried and is worse:
    # the move back onto the slot line was refused on 3 of 10 seeds (6/10 overall).
    slot = np.asarray(slot_xyz, dtype=np.float64).reshape(3)
    crossed = False
    base_x = float(_np(task.agent.base_link.pose.p).reshape(-1)[0])
    for torso, drop in extraction_cross_rungs(base_x - float(slot[0])):
        if torso is not None:
            lowered = retr.plan_joints(env, planner, task, {"torso_lift_joint": torso},
                                       label=f"torso to {torso} for the cross",
                                       line_only=True, qpos_step=TORSO_LINE_STEP)
            if lowered == -1:
                continue
            res = lowered
            if common.stopped_by_horizon(planner):
                return res, False
            planner.planner.update_from_simulation()
        tcp = task.agent.tcp.pose.sp
        across = sapien.Pose(p=[float(slot[0]), float(slot[1]), float(tcp.p[2]) - drop], q=tcp.q)
        if noise is not None:
            across = noise.pose(f"high counter cross {prop.name}", across)
        say(env, "across to the slot's column", prop=prop.name, torso=torso, drop=drop,
            dx=round(float(slot[0]) - float(tcp.p[0]), 3))
        # A joint LINE to the nearest IK branch, not a screw: this is the longest
        # Cartesian stroke of the episode and `plan_screw` is a least-norm Jacobian walk
        # that spreads it over every joint, which is the arm rotation the owner saw.
        res = _ready_by_line(env, planner, task, across, drive=0.0, grasp=across,
                                 label=f"across to the slot ({prop.name}, torso {torso}, drop {drop})",
                                 slack=(0.0, 0.0, -0.005), touch_needle=None,
                                 rank=roll_rank(task))
        if res == -1:
            res = _straight(env, planner, across, stage="across to the slot", tries=2)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res != -1:
            crossed = True
            break
        planner.planner.update_from_simulation()
    if not crossed:
        return fail(env, f"cross to the slot column for {prop.name}",
                    rungs=list(CROSS_RUNGS)), False
    if not _grasping(task, prop):
        say(env, f"MISSED: {prop.name} left the gripper crossing to the slot")
        return res, False
    planner.planner.update_from_simulation()

    down, torso = torso_to(env, planner, task, COUNTER_TORSO_RUNGS, label="torso down to")
    if down != -1:
        res = down
        if common.stopped_by_horizon(planner):
            return res, False
        say(env, "torso down", torso=torso)
    if not _grasping(task, prop):
        say(env, f"MISSED: {prop.name} left the gripper on the way down")
        return res, False

    place = sp.place_pose_for(task, obj=_seen(prop), target=slot_xyz, compensate_xy=True)
    if place is None:
        return fail(env, f"{prop.name} has no collision mesh"), False
    say(env, "descend to the slot", prop=prop.name,
        place=[round(float(v), 3) for v in place.p])
    res = _straight(env, planner, place, stage="descend to the slot", tries=2)
    if res == -1:
        res = common.arm_move(env, planner, place, who=WHO,
                              stage="descend to the slot (any plan)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"stand {prop.name} on the slot"), False
    planner.planner.update_from_simulation()

    res = release_and_clear(env, planner, task, prop, place)
    return res, not common.stopped_by_horizon(planner)


def restore_one(env, planner, task, prop, row_xyz, *, noise=None):
    """Counter slot -> its row slot: the stow flow, aimed at a chosen row point."""
    # ACROSS FIRST, at the height the previous clearing left, and only then down. Going
    # straight to the pre-grasp posture sweeps the hand sideways across the counter AT
    # grasp height, and a line that passes a standing prop by a centimetre is one the
    # tracking error can touch: measured on seed 6, the prop the arm was on its way to
    # TIPPED OVER during the approach, and the grasp that followed aimed at a mid-height
    # of 0.942 — a lying prop — and drove the palm into it. The extraction already
    # crosses high and descends; this is the same move on the way back.
    if _BELIEF is not None:
        # Refresh the counter estimates from here; a prop this look misses keeps the
        # estimate its release left (dead reckoning), and the grasp ladder still demands
        # a fresh wrist fit before closing on it.
        i = _index(task, prop)
        fit = _BELIEF.look("counter", perc.BASE_CAMS + (perc.HAND_CAM,),
                           anchors=_BELIEF.slot_pts).get(i, {})
        say(env, "counter estimate", prop=prop.name,
            source="fit" if fit.get("accepted") else "cluster" if "coarse_xy" in fit
            else "release")
    p0 = _np(_seen(prop).pose.p).reshape(-1)[:3]
    z0 = float(_np(task._counter_top_z).reshape(-1)[0])
    moved = -1
    base_x = float(_np(task.agent.base_link.pose.p).reshape(-1)[0])
    for torso, hand_z in restore_cross_rungs(base_x - float(p0[0]), z0):
        if torso is not None:
            lowered = retr.plan_joints(env, planner, task, {"torso_lift_joint": torso},
                                       label=f"torso to {torso} for the cross",
                                       line_only=True, qpos_step=TORSO_LINE_STEP)
            if lowered == -1:
                continue
            res = lowered
            if common.stopped_by_horizon(planner):
                return res, False
            planner.planner.update_from_simulation()
        tcp0 = task.agent.tcp.pose.sp
        z = float(tcp0.p[2]) if hand_z is None else hand_z
        # Never come down BEHIND the prop: after a row placement the hand backs out to
        # ~9 cm past the pads' line, and the grasp posture from there passes over the
        # prop to its pre-grasp, low enough for the palm to catch its top (measured
        # 2026-10-16: 4 state episodes of 100 found the prop pushed 7-12 cm toward the
        # robot or lying). In front of it, the posture is a move down.
        y = min(float(tcp0.p[1]), float(p0[1]) - RESTORE_CROSS_BACK)
        over = sapien.Pose(p=[float(p0[0]), y, z], q=tcp0.q)
        if noise is not None:
            over = noise.pose(f"counter approach {prop.name}", over, (True, False, True))
        say(env, "across to the prop's column", prop=prop.name, torso=torso,
            dx=round(float(p0[0]) - float(tcp0.p[0]), 3),
            tcp_z=round(float(tcp0.p[2]), 3), to_z=round(z, 3))
        moved = _ready_by_line(env, planner, task, over, drive=0.0, grasp=over,
                                   label=f"across to {prop.name}'s column (torso {torso}, z {round(z, 3)})",
                                   slack=(0.0, 0.0, -0.005), touch_needle=None,
                                   rank=roll_rank(task))
        if moved == -1:
            moved = _straight(env, planner, over, stage="across to the prop's column",
                              tries=2)
        if moved != -1 and common.stopped_by_horizon(planner):
            return moved, False
        if moved != -1:
            res = moved
            planner.planner.update_from_simulation()
            break
        planner.planner.update_from_simulation()
    if moved == -1:
        say(env, "cross refused at every height; approaching from where we are",
            prop=prop.name, rungs=list(CROSS_SAFE_TORSO))

    # THREE ladders, outermost first, each bought by a measured failure: the TORSO,
    # because the props already staged are obstacles for the reach to the ones behind
    # them (seed 6); the LIFT, because the wrist hangs 6.7 cm under the hand and the
    # counter is right below (seed 9: 75 IK solutions, all refused against the counter);
    # the DEPTH, the retrieval flow's own. The FOLD is a recovery outside all three, not
    # a routine — folding between transfers was removed, but a posture the previous
    # release happened to leave can still refuse.
    res, ok = -1, False
    chosen = None
    for attempt in ("as we are", "after a fold"):
        if attempt != "as we are":
            say(env, "counter grasp refused; folding to rest and trying once more",
                prop=prop.name)
            if fold_arm(env, planner, task) == -1:
                break
            planner.planner.update_from_simulation()
        for torso in COUNTER_GRASP_TORSO_RUNGS:
            lowered = retr.plan_joints(env, planner, task, {"torso_lift_joint": torso},
                                       label=f"torso to {torso} for the counter grasp",
                                       line_only=True, qpos_step=TORSO_LINE_STEP)
            if lowered == -1:
                continue
            res = lowered
            if common.stopped_by_horizon(planner):
                return res, False
            planner.planner.update_from_simulation()
            for lift in grasp_lifts(task, prop):
                for depth in retr.SIDE_GRASP_DEPTHS:
                    grasp, pre = retr.side_grasp_pose(task, depth, lift=lift, obj=_seen(prop))
                    if grasp is None:
                        return fail(env, f"{prop.name} has no collision mesh"), False
                    res = _ready_by_line(
                        env, planner, task, pre, drive=0.0, grasp=grasp,
                        label=f"grasp posture ({prop.name}, torso {torso}, lift {lift}, "
                              f"depth {depth})",
                        touch_needle=prop.name, rank=roll_rank(task))
                    if res != -1 and common.stopped_by_horizon(planner):
                        return res, False
                    planner.planner.update_from_simulation()
                    if res != -1:
                        chosen = (torso, lift)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                continue
            say(env, "counter grasp posture", prop=prop.name, torso=chosen[0],
                lift=chosen[1])
            res, ok = grasp_ladder(env, planner, task, prop, where="counter",
                                   lift=chosen[1])
            if ok:
                break
            if res != -1 and common.stopped_by_horizon(planner):
                return res, False
            chosen = None
        if ok:
            break
    if not ok:
        return fail(env, f"take {prop.name} off the counter",
                    torsos=list(COUNTER_GRASP_TORSO_RUNGS),
                    lifts=list(grasp_lifts(task, prop))), False
    _hold(env, planner, task, prop, held=True)

    tcp = task.agent.tcp.pose.sp
    up = sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + UNSHELVE_DZ], q=tcp.q)
    res = _straight(env, planner, up, stage="lift off the counter")
    if res == -1:
        res = common.arm_move(env, planner, up, who=WHO,
                              stage="lift off the counter (any plan)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"lift {prop.name} off the counter"), False
    planner.planner.update_from_simulation()

    # The stow flow's rise, imported whole: it tries the same ladder and, when nothing
    # plans, retreats along the approach axis and tries it again — the difference
    # between a hard -1 and a second chance. Its body never names the object.
    res = sp.rise_with_the_cup(env, planner, task)
    if res == -1:
        return res, False
    if common.stopped_by_horizon(planner):
        return res, False
    say(env, "torso up", tilt=round(common._tilt_deg(prop.pose.sp.q), 1))
    if not _grasping(task, prop):
        say(env, f"MISSED: {prop.name} left the gripper on the way up")
        return res, False

    entry = sp.place_pose_for(task, rise=sp.ENTRY_RISE, obj=_seen(prop), target=row_xyz,
                              compensate_xy=True)
    if entry is None:
        return fail(env, f"{prop.name} has no collision mesh"), False
    base_p = _np(task.agent.base_link.pose.p).reshape(-1)
    drive = float(retr.WORK_DOCK_Y - base_p[1])
    ready = sapien.Pose(p=[entry.p[0], entry.p[1] - drive, entry.p[2]], q=entry.q)
    if noise is not None:
        ready = noise.pose(f"shelf entry {prop.name}", ready, (True, False, True))
    say(env, "entry posture", prop=prop.name, drive=round(drive, 3),
        entry=[round(float(v), 3) for v in entry.p])
    # The held prop stays in the planning world for the descent pre-check: it is exactly
    # what must not hit the shelf.
    res = _ready_by_line(env, planner, task, ready, drive=drive, grasp=entry,
                             label=f"entry posture ({prop.name})",
                             slack=sp.PLACE_PROBE_SLACK, touch_needle=None)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"entry posture for {prop.name}"), False
    planner.planner.update_from_simulation()

    # Let the arm settle under the load and correct what the joint line could not see.
    # Measured on this exact 4.5x4.5x12 cm 243 g prop (`cabinet_stow_planner`'s
    # ENTRY_SAG_TOL): the entry came in 4.4 cm low against 2 mm with an 8 g cup, the
    # object crossed the shelf's edge with 4 mm to spare, and on one seed it arrived
    # tilted 21.6 deg in the fingers and fell flat when they opened.
    settle = planner.idle_steps(t=sp.ENTRY_SETTLE_STEPS)
    if settle != -1:
        res = settle
        if common.stopped_by_horizon(planner):
            return res, False
    tcp_now = task.agent.tcp.pose.sp
    sag = float(ready.p[2]) - float(tcp_now.p[2])
    if abs(sag) > sp.ENTRY_SAG_TOL:
        say(env, "the entry sagged under the load; correcting", prop=prop.name,
            sag=round(sag, 4))
        fix = _straight(env, planner, ready, stage="correct the entry height", tries=2)
        if fix != -1:
            res = fix
            if common.stopped_by_horizon(planner):
                return res, False
            planner.planner.update_from_simulation()
        else:
            say(env, "entry correction refused; driving in as we are")

    say(env, "drive into the cabinet, loaded", distance=round(drive, 3),
        asked=round(drive - LOADED_DRIVE_COAST, 3))
    res = planner.drive_straight(drive - LOADED_DRIVE_COAST, v=0.10, hold_pose=True)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "drive into the cabinet, loaded"), False
    settled = settle_base(planner, task)
    if settled is not None:
        res = settled
        if common.stopped_by_horizon(planner):
            return res, False
    if not _grasping(task, prop):
        say(env, f"MISSED: {prop.name} left the gripper on the way in")
        return res, False

    place = sp.place_pose_for(task, obj=_seen(prop), target=row_xyz, compensate_xy=True)
    others = row_neighbours(task, prop)
    say(env, "descend onto the row slot", prop=prop.name,
        place=[round(float(v), 3) for v in place.p],
        neighbours=[o.name for o in others])
    with neighbours_touchable(planner, others):
        res = _straight(env, planner, place, stage="descend onto the row slot", tries=2)
        if res == -1:
            # The stow flow's middle rung: the last 5 mm inside a box is where a screw
            # refuses on clearance, and an RRT with a prop in the fingers inside a
            # cabinet is the worst plan in this file. Defer it by one rung.
            higher = sapien.Pose(p=[place.p[0], place.p[1], place.p[2] + 0.005], q=place.q)
            res = _straight(env, planner, higher,
                            stage="descend onto the row slot (5 mm up)", tries=2)
        if res == -1:
            res = common.arm_move(env, planner, place, who=WHO,
                                  stage="descend onto the row slot (any plan)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"stand {prop.name} back in the row"), False
    planner.planner.update_from_simulation()

    # Out of the cabinet on the drive it came in on, before anything else moves.
    res = release_and_clear(env, planner, task, prop, place)
    return res, not common.stopped_by_horizon(planner)


def choose_assignment(task, blind: bool, rng):
    """Which staged prop goes into which row slot — the ONE privileged read.

    Sighted: the episode's own permutation. Blind: a uniform assignment of the
    three blockers to their original slots (1/6 conditional guessing reference).
    The deepest target is still supplied; this is not a general memoryless baseline.

    Returns a list of `(prop_index, slot_index)`, deepest empty slot first, because
    nothing can be placed behind an occupied slot.

    Example:
        >>> plan = choose_assignment(task, blind=False, rng=rng)   # doctest: +SKIP
    """
    if _BELIEF is not None:
        # The planner's own memory: the row slot each prop was SEEN in before it moved.
        slots = np.array([_BELIEF.row_slot[i] for i in range(task.cfg.n_props)])
    else:
        slots = _np(task.original_slots).reshape(-1)
    deep = int(task.cfg.n_props - 1)
    movables = [i for i in range(len(slots)) if int(slots[i]) != deep]
    targets = sorted((int(slots[i]) for i in movables), reverse=True)
    if blind:
        order = list(rng.permutation(len(movables)))
        return [(movables[order[k]], targets[k]) for k in range(len(movables))]
    return [(i, int(slots[i])) for i in sorted(movables, key=lambda i: -int(slots[i]))]


def solve(env, seed=None, debug=False, vis=False, blind=False, *,
          waypoint_noise_seed=None, waypoint_noise_m=0.005,
          planner_factory=common.default_planner_factory, perception: str = "state"):
    """``perception="pointcloud"`` makes every prop pose the planner acts on an estimate
    from its own cameras (`depth_recall_v1_perception`); ``"state"`` reads the simulator."""
    assert perception in ("state", "pointcloud"), perception
    with common.planning_budget(4.0):
        return _solve(env, seed, debug, vis, blind, waypoint_noise_seed,
                      waypoint_noise_m, planner_factory, perception)


def _solve(env, seed, debug, vis, blind, noise_seed, noise_m, planner_factory,
           perception="state"):
    """Clear the row onto the counter, keep the deepest prop, put the rest back.

    Example:
        >>> res = solve(env, seed=0)   # doctest: +SKIP
    """
    obs, info = env.reset(seed=seed)
    if seed is not None:
        seed_everything(seed)
    assert env.unwrapped.control_mode == "pd_joint_pos", env.unwrapped.control_mode
    task = env.unwrapped
    global _BELIEF
    _BELIEF = (perc.PropBelief(task, lambda message, **data: say(env, message, **data))
               if perception == "pointcloud" else None)
    planner = planner_factory(env, debug, vis)
    if _BELIEF is not None:
        _BELIEF.bind_world(planner)
    noise = WaypointNoise((0 if seed is None else int(seed)) + 200003
                          if noise_seed is None else noise_seed, noise_m,
                          lambda message, **data: say(env, message, **data))

    def gaze(points, tilt):
        centre = np.asarray(points).reshape(-1, 3).mean(axis=0)
        local = (task.agent.base_link.pose[0].sp.inv() * sapien.Pose(centre)).p
        pan = float(np.clip(np.arctan2(local[1], local[0]), -.6, .6))
        return planner.hold_head(pan=pan, tilt=tilt, t=12, ramp=10)
    rng = np.random.default_rng((0 if seed is None else int(seed)) + COIN_OFFSET)

    props = list(task.props)
    slot_pts = _np(task._slot_points).reshape(-1, 4, 3)[0]
    row_pts = _np(task._row_points).reshape(-1, task.cfg.n_props, 3)[0]
    if _BELIEF is not None:
        # The episode's row and slot points carry a per-episode sideways shift that no
        # image shows; start from the static map, measure the shift at the dock.
        row_pts, slot_pts = _BELIEF.map_points()
        _BELIEF.slot_pts = slot_pts
    say(env, "episode", blind=bool(blind),
        slots=[round(float(v), 3) for v in slot_pts[:, 0]],
        row_x=round(float(row_pts[0][0]), 3))

    # ONE dock for the whole episode, at the row's column. Everything after it moves on
    # straight lines: the base only ever drives along its own axis, into the cabinet and
    # back out, and the arm reaches sideways to the counter slots. `drive_base` turns the
    # base, so calling it per transfer is what made the base hop and spin.
    # Retrieval ducks the torso before its dock drive; this does not, and the difference
    # is that this dock is the FIRST thing the episode does — the robot is already in the
    # posture retrieval ducks into position for. `MIKASA_DR1_DUCK=1` puts it back.
    if os.environ.get("MIKASA_DR1_DUCK", "0") == "1":
        res = retr.plan_joints(env, planner, task,
                               {"torso_lift_joint": retr.TORSO_DRIVE,
                                "wrist_flex_joint": 1.7},
                               label="duck the torso")
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            return fail(env, "duck the torso for the drive")
        planner.planner.update_from_simulation()

    res = gaze(row_pts, -.20)
    if res == -1 or common.stopped_by_horizon(planner):
        return res
    row_x = float(row_pts[0][0])
    res = dock_at(env, planner, task, row_x, stage="dock at the row (once)", noise=noise)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "dock at the row")

    # --- take the row apart, front first (read off the poses, not the answer) ---
    # Each prop goes onto its own colour's pad. Simulator mode reads which slot that is
    # off the task; perception mode reads the pads' colours after the row is calibrated.
    pad_of = (None if _BELIEF is not None else
              {i: int(k) for i, k in enumerate(_np(task.pad_slots).reshape(-1, len(props))[0])})
    staged = {}
    # Simulator mode keeps its original rule exactly: one ordering read at the start.
    # Perception mode orders by what it sees, below; no true pose enters.
    remaining = (sorted(range(len(props)), key=lambda i: float(_np(props[i].pose.p).reshape(-1)[1]))
                 if _BELIEF is None else list(range(len(props))))
    while remaining:
        if _BELIEF is None:
            prop_i = remaining[0]
        else:
            # Look at the row from the dock and take the front-most prop SEEN; the slot it
            # was seen in is what the restore will remember.
            res = gaze(row_pts, -.20)
            if res == -1 or common.stopped_by_horizon(planner):
                return res
            fits = _BELIEF.look("shelf", perc.BASE_CAMS, anchors=row_pts, trust_anchor_x=False)
            if _BELIEF.row_dx is None:
                # First look from the dock: measure the row's sideways shift, move the base
                # onto the measured row when the map's dock is off by more than
                # `REDOCK_MIN_M`, then look again so every estimate is tied to the
                # corrected points. Without the move the arm reaches the counter from up
                # to 4 cm aside of where the simulator mode stands; measured 2026-10-15 on
                # 100 seeds, that path knocked a neighbouring counter prop over on 2.
                row_pts, slot_pts = _BELIEF.calibrate_row(fits, row_pts, slot_pts)
                if abs(_BELIEF.row_dx) > REDOCK_MIN_M:
                    res = dock_at(env, planner, task, float(row_pts[0][0]),
                                  stage="re-dock at the measured row", noise=noise)
                    if res != -1 and common.stopped_by_horizon(planner):
                        return res
                    if res == -1 and abs(_BELIEF.row_dx) > REDOCK_SKIP_MAX_M:
                        return fail(env, "re-dock at the measured row")
                    if res == -1:
                        say(env, "re-dock refused; working from the first dock",
                            dx=round(float(_BELIEF.row_dx), 4))
                        planner.planner.update_from_simulation()
                    else:
                        res = gaze(row_pts, -.20)
                        if res == -1 or common.stopped_by_horizon(planner):
                            return res
                _BELIEF.look("shelf", perc.BASE_CAMS, anchors=row_pts, trust_anchor_x=False)
            if pad_of is None:
                res = gaze(slot_pts, .35)
                if res == -1 or common.stopped_by_horizon(planner):
                    return res
                pad_of = _BELIEF.identify_pads(slot_pts)
                if pad_of is None:
                    return fail(env, "read the pads' colours")
                res = gaze(row_pts, -.20)
                if res == -1 or common.stopped_by_horizon(planner):
                    return res
                _BELIEF.look("shelf", perc.BASE_CAMS, anchors=row_pts, trust_anchor_x=False)
            seen = [i for i in remaining if _BELIEF.has(i)]
            if not seen:
                return fail(env, "perceive the row", remaining=[props[i].name for i in remaining])
            prop_i = min(seen, key=lambda i: (_BELIEF.est[i]["y"], i))
            slot_seen = _BELIEF.remember_row_slot(prop_i, row_pts)
            # The row is emptied front first, so the front-most OCCUPIED slot is known; a
            # prop seen behind it means the one in front was missed, and taking the deeper
            # one would drive the hand into it.
            front = len(props) - len(remaining)
            if slot_seen != front:
                return fail(env, "the front prop was not seen", seen=props[prop_i].name,
                            seen_slot=slot_seen, front_slot=front)
            say(env, "front-most seen", prop=props[prop_i].name, row_slot=slot_seen)
        remaining.remove(prop_i)
        slot_k = int(pad_of[prop_i])
        staged[prop_i] = slot_k
        say(env, "extract", prop=props[prop_i].name, to_slot=slot_k,
            where={q.name: [round(float(v), 3) for v in _np(q.pose.p).reshape(-1)[:3]]
                   for q in props})
        res, ok = extract_one(env, planner, task, props[prop_i], slot_pts[slot_k], noise=noise)
        if not ok:
            return res
    say(env, "row cleared", staged={props[i].name: k for i, k in staged.items()})

    # Inspect the staged row independently of the remembered assignment.
    res = gaze(slot_pts, .35)
    if res == -1 or common.stopped_by_horizon(planner):
        return res

    # --- restore the three blockers, deepest empty slot first ------------------
    for prop_i, slot_idx in choose_assignment(task, blind, rng):
        say(env, "restore", prop=props[prop_i].name, to_row_slot=slot_idx)
        res, ok = restore_one(env, planner, task, props[prop_i], row_pts[slot_idx], noise=noise)
        if not ok:
            return res

    settled = planner.idle_steps(t=retr.SETTLE_STEPS)
    if settled == -1:
        return fail(env, "settle wait")
    res = settled
    info = res[-1]
    ok = _b(info, "success")
    tilt_max = float(_np(info["tilt_max_deg"]).reshape(-1)[0])
    if tilt_max > sp.TILT_WARN_DEG:
        # The line a sweep greps to tell an honest run from a hollow one.
        say(env, "KNOCKED OVER", tilt_deg=round(tilt_max, 1),
            props=[round(float(v), 1) for v in _np(info["prop_tilt_deg"]).reshape(-1)])
    say(env, "episode over", success=ok,
        restored=_np(info["restored"]).reshape(-1).tolist(),
        target_on_its_pad=_b(info, "target_on_its_pad"),
        wrong_pad=_b(info, "wrong_pad"),
        wrong_assign=_b(info, "wrong_assign"),
        target_returned=_b(info, "target_returned"),
        two_in_slot=_b(info, "two_in_slot"),
        tilt_max=round(float(_np(info["tilt_max_deg"]).reshape(-1)[0]), 1),
        steps=int(getattr(planner, "elapsed_steps", -1)))
    return res


if __name__ == "__main__":
    import argparse

    import gymnasium as gym

    import mani_skill  # noqa: F401
    import my_scenes  # noqa: F401  (registers benchmark environments)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--assisted-grasp", action="store_true",
                    help="weld props to the hand by OmniGibson's rule (changes the physics)")
    ap.add_argument("--perception", choices=("state", "pointcloud"), default="state",
                    help="where prop poses come from: the simulator, or the robot's cameras")
    ap.add_argument("--ik-starts", choices=("random", "fixed"), default=None,
                    help="IK restarts: mplib's random draws, or a fixed table (robots/fetch/ik_backend.py)")
    ap.add_argument("--ik-backend", choices=("mplib", "curobo"), default=None,
                    help="IK proposals from mplib alone, or from cuRobo (needs MIKASA_CUROBO_PY)")
    ap.add_argument("--video-dir", default="videos/depth_recall_v1",
                    help="where the render camera's clip goes (unless --no-video)")
    args = ap.parse_args()
    for flag, var in ((args.ik_starts, "MIKASA_IK_STARTS"), (args.ik_backend, "MIKASA_IK_BACKEND")):
        if flag is not None:
            os.environ[var] = flag
    e = gym.make("MikasaDepthRecall-v1", num_envs=1, robot_uids="ds_fetch",
                 control_mode="pd_joint_pos", obs_mode="state", sim_backend="cpu",
                 assisted_grasp=args.assisted_grasp,
                 render_mode=None if args.no_video else "rgb_array")
    if not args.no_video:
        from mani_skill.utils.wrappers.record import RecordEpisode

        e = RecordEpisode(e, args.video_dir, save_trajectory=False, save_video=True,
                          video_fps=30, max_steps_per_video=None)
    out = solve(e, seed=args.seed, blind=args.blind, perception=args.perception)
    print("no_plan" if isinstance(out, int) else f"success={_b(out[-1], 'success')}")
    if not args.no_video:
        e.flush_video(name=f"seed{args.seed}_{args.perception}"
                           f"{'_ag' if args.assisted_grasp else ''}{'_blind' if args.blind else ''}")
    ag = getattr(e.unwrapped, "_ag", None)
    if ag is not None:
        kinds = [k for _, k, _ in ag.events]
        print(f"assisted_grasp establish={kinds.count('establish')} "
              f"release={kinds.count('release')} events={ag.events}")
    e.close()
