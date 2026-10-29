"""WaterPlants-specific capture of swept-arc refusal text."""
from __future__ import annotations

import contextlib
import io
import sys
import types

import numpy as np
import sapien

from planners.oracle import oracle_common as common
from robots.fetch.base_yaw import SWEEP_REFUSAL


class _Tee:
    """Write stdout to its original stream and a capture buffer."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def capture_refusal(pattern: str = SWEEP_REFUSAL):
    """Capture the first swept-arc refusal while continuing to write stdout normally."""
    holder = types.SimpleNamespace(refusal=None)
    buffer = io.StringIO()
    original = sys.stdout
    sys.stdout = _Tee(original, buffer)
    try:
        yield holder
    finally:
        sys.stdout = original
        holder.refusal = next(
            (line[line.index(pattern):].strip()
             for line in buffer.getvalue().splitlines() if pattern in line),
            None,
        )

# Shared primitives used by this task-specific stage kit.
_np = common._np
say = common.say
fail = common.fail
stopped_by_horizon = common.stopped_by_horizon
_knot_kw = common._knot_kw
_tilt_deg = common._tilt_deg

CARRY_YAWS_DEG = (0.0, 90.0, -90.0, 180.0)
"""Upright reorientations for the tuck, in the order the T5 probe measured them.

Half-turns (±45, ±135) were inserted ahead of 180 on the theory that the slip on
held-out seeds 17 and 29 came from 180 being the most violent candidate, and were
**withdrawn**: seed 29 then executed a *45 deg* tuck and lost the object anyway, so the
sweep angle is not the mechanism, and the held-out rate did not move (17/20 either way)
while eval seed 3 picked up a physical miss on the drive that follows. Whatever pulls
the object out of the fingers there, it is not how far the wrist turns."""


CARRY_LIFTS_M = (0.0, 0.08)
"""Extra height for the tuck, tried only after every yaw at the height below it failed.

The tuck pulls the TCP in to `base + ahead·face` at the height the lift left it, and the
held object hangs below and swings with the yaw. When it catches something on the way in
the refusal names the object — `shaker <-> counter_main`, `shaker <-> bowl` on held-out
seeds 17 and 29 — and no further yaw at that height helps, because the height is what is
wrong. Fallback-only, so a tuck that plans at the first candidate is unchanged."""


def level_correction(q) -> np.ndarray:
    """The world-frame rotation that puts body +Z of `q` back on world +Z, minimal arc.

    Returned as a *correction* rather than a corrected pose because the thing that
    must end up level (the held object) is not the thing being commanded (the TCP).
    They are rigidly attached, so the same world-frame pre-rotation levels both:
    ``q_tcp_new = correction(q_obj) * q_tcp``. Levelling the TCP instead would leave
    the object at whatever angle it happens to sit in the fingers.

    Minimal arc, so spin about the vertical is left alone — it costs rotation and
    changes nothing about whether the object is level.

    Args:
        q: `(w, x, y, z)` of the object as it is now.

    Returns:
        float64 `(w, x, y, z)`, a world-frame rotation to pre-multiply.

    Example:
        >>> import numpy as np, sapien
        >>> half = np.radians(40.0) / 2.0
        >>> tipped = np.array([np.cos(half), np.sin(half), 0.0, 0.0])   # 40 deg about X
        >>> fixed = (sapien.Pose(q=level_correction(tipped)) * sapien.Pose(q=tipped)).q
        >>> R = sapien.Pose(q=fixed).to_transformation_matrix()[:3, :3]
        >>> float(np.round(R[2, 2], 6))
        1.0
        >>> # an already-level pose needs no correction
        >>> float(np.round(level_correction(np.array([1.0, 0, 0, 0]))[0], 6))
        1.0
    """
    q = np.asarray(q, dtype=np.float64)
    v = sapien.Pose(q=q).to_transformation_matrix()[:3, 2]
    v = v / np.linalg.norm(v)
    dot = float(np.clip(v[2], -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.cross(v, np.array([0.0, 0.0, 1.0]))
    n = float(np.linalg.norm(axis))
    axis = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else axis / n
    half = np.arccos(dot) / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def slerp(q0, q1, t: float) -> np.ndarray:
    """Shortest-arc interpolation between two `(w, x, y, z)` quaternions.

    Args:
        q0, q1: the endpoints, `(w, x, y, z)`.
        t: 0 gives `q0`, 1 gives `q1`.

    Returns:
        float64 `(w, x, y, z)`, unit length.

    Example:
        >>> import numpy as np
        >>> np.round(slerp([1.0, 0, 0, 0], [0.0, 1, 0, 0], 0.5), 4).tolist()
        [0.7071, 0.7071, 0.0, 0.0]
    """
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0.0:                      # take the short way round
        b, d = -b, -d
    if d > 0.9995:                   # nearly identical: lerp and renormalise
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    return (np.sin((1 - t) * th) / s) * a + (np.sin(t * th) / s) * b


def move_via(env, planner, task, target, *, who: str, stage: str, legs: int = 2,
             disable_lift_joint: bool = False, bow: float = 0.0,
             max_knots=None, knot_draws: int = 1):
    """Reach `target` through `legs` interpolated waypoints instead of in one plan.

    K55, and the reason is the shape of the waste rather than its size. A single
    long plan lets RRT wander: over a water-plants pair the hand travels 2557 deg
    against 858 deg of net per-stage change, a 3.0x detour, and `hover over the
    bucket` ends **1.3 deg** from where it started having travelled **269**. Broken
    into short hops the planner has far less room to wander between the ends, and —
    more importantly — `plan_screw`, which is a straight line in SE(3) and barely
    rotates at all, succeeds much more often on a short move than a long one, so
    more of the path is straight by construction rather than by luck.

    Position is interpolated linearly and orientation by shortest-arc slerp, so the
    waypoints lie on the direct path; they add no distance, only structure.

    Falls back to the single direct move the moment a leg refuses, so this can only
    change *how* a reachable target is reached, never *whether* it is. A refused leg
    executes nothing and costs no step.

    Args:
        env, planner, task: as everywhere.
        target: the final TCP pose.
        who, stage: for the trace.
        legs: waypoints including the target. 1 is the old behaviour. Each leg is a
            separately time-parameterised trajectory that decelerates to a stop, so
            more legs means more stop-start and more torso correction — 2 is the
            measured compromise between that and letting a long plan wander.
        bow: how far the curve is lifted off the straight chord, as a fraction of
            the chord length. **0, and measured.** A curved path was tried at 0.12
            on the hover and the dip and took the pair to 0/2: those two moves are
            near-vertical, so bowing them upward fights the motion instead of
            rounding it, and the arc puts the cup where the bucket rim is. The
            parameter stays because the argument for curving is sound for a long
            *horizontal* transfer — it is the wrong tool for these two stages.
        disable_lift_joint: passed through to `static_manipulation`.

    Returns:
        The 5-tuple, or -1 if both the waypointed path and the direct move refuse.

    Example:
        >>> res = move_via(env, planner, task, pose, who=WHO, stage="hover")  # doctest: +SKIP
    """
    start = task.agent.tcp.pose.sp
    p0, q0 = np.asarray(start.p, dtype=np.float64), np.asarray(start.q, dtype=np.float64)
    p1 = np.asarray(target.p, dtype=np.float64)
    q1 = np.asarray(target.q, dtype=np.float64)
    n = max(1, int(legs))

    # A quadratic Bezier through a control point lifted off the chord, rather than
    # the chord itself. Two reasons, both from watching it run: a straight chord
    # between a low pose and a high one drags the cup along the shortest line, which
    # is exactly where the counter edge and the bucket rim are; and a path that
    # curves lets the torso make one monotone move instead of correcting at every
    # waypoint. `bow` is a fraction of the chord length, so it scales with the move
    # and is nothing at all for a short one.
    chord = p1 - p0
    span = float(np.linalg.norm(chord))
    lift = np.array([0.0, 0.0, 1.0]) * (bow * span)
    ctrl = p0 + chord * 0.5 + lift

    res = -1
    for i in range(1, n + 1):
        t = i / n
        # Bezier at t; at t=1 this is exactly p1, so the target is never approximated.
        pos = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
        leg = sapien.Pose(p=pos, q=slerp(q0, q1, t))
        res = planner.static_manipulation(leg, disable_lift_joint=disable_lift_joint,
                                          **_knot_kw(max_knots, knot_draws))
        if res != -1 and stopped_by_horizon(planner):
            return res
        if res == -1:
            if i == 1:
                break            # the first hop refused: nothing executed, fall back
            say(env, who, f"{stage}: leg {i}/{n} refused, finishing direct")
            return planner.static_manipulation(target, disable_lift_joint=disable_lift_joint,
                                               **_knot_kw(max_knots, knot_draws))
    if res == -1:
        return planner.static_manipulation(target, disable_lift_joint=disable_lift_joint,
                                           **_knot_kw(max_knots, knot_draws))
    return res


def level_in_place(env, planner, task, obj, who: str = "oracle", *, tol_deg: float = 3.0,
                   max_step_deg: float = 30.0, max_bites: int = 3) -> bool:
    """Rotate the held object back to level without moving the TCP's position.

    The cheapest possible correction: the hand stays where it is and only turns, by
    the minimal arc that puts the object's +Z back on world +Z. Used before a tuck
    or a drive so the object is not carried at an angle, and after a pour so the
    tip does not persist into the next leg.

    Best effort by design. A refused plan executes nothing and costs no steps, so
    the caller carries on with the object still tilted rather than losing the
    episode to a cosmetic stage — the tilt is a quality measure here, not a success
    condition.

    Args:
        env: the (possibly wrapped) env, for `say`.
        planner: the solver (or the stub).
        task: `env.unwrapped`.
        obj: the held Actor.
        who: the oracle's tag.
        tol_deg: skip the move when the object is already this close to level.
        max_step_deg: largest single correction attempted, in degrees. 30 is under
            the 33 deg that was observed to plan and over half the 65 deg that was
            observed never to.

            **This defaulted to 180.0 until K58, which is the number the sentence
            above says not to use.** At 180 every real tilt gives
            `frac = min(1.0, 180/tilt) = 1.0`, so the first bite asks for the *entire*
            correction — the 65 deg case documented as never planning — and the
            `if res == -1 and frac >= 1.0: break` guard below then exits before the
            halving path or `max_bites` can be reached. The adaptive machinery was
            unreachable from the default. Measured over 60 seeds (412 attempts):
            **216 of 412 levellings, 52.4 %, came back `level fell short` with zero
            progress, at a median tilt of 65.1 deg.** So the object was carried at ~65
            deg through most return legs, and — because a 65 deg object makes the
            `levelled` carry candidates a large reorientation *and* translation — it
            also pushed the tuck onto its harder `as held` fallbacks (seed 56 lost the
            episode there).
        max_bites: cap on corrections, so a pose that cannot be levelled costs a
            bounded number of refused plans rather than the episode's horizon.

    Returns:
        True if the object ended level to within `tol_deg`.

    Example:
        >>> level_in_place(env, planner, task, task.cup, who="water_plants_planner")  # doctest: +SKIP
        True
    """
    before = _tilt_deg(_np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64))
    if before <= tol_deg:
        return True
    say(env, who, "level the cup in place", tilt_deg=round(float(before), 1))

    # One big correction is refused where several small ones are not. Measured over
    # a water-plants pair: every attempt from 21-33 deg planned and landed the cup
    # at 0.3-0.5 deg, and **every attempt from 65-70 deg was refused** — the wrist
    # simply does not have that turn available in one move at the pose the pour
    # leaves it in. Walking the correction in `max_step_deg` bites turns the same
    # total into moves the arm will accept, and each bite is re-derived from the
    # cup's *measured* orientation rather than from the plan, so a bite that lands
    # short or slips in the fingers is corrected by the next one instead of
    # accumulating (that accumulation is the ratchet this whole change removes).
    for _ in range(max_bites):
        now_q = _np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64)
        now = _tilt_deg(now_q)
        if now <= tol_deg:
            break
        frac = min(1.0, max_step_deg / now)
        corr = level_correction(now_q)
        if frac < 1.0:
            # Same axis, a fraction of the angle: slerp the correction toward identity.
            w = float(np.clip(corr[0], -1.0, 1.0))
            ang = 2.0 * np.arccos(w)
            axis = corr[1:]
            n = float(np.linalg.norm(axis))
            if n > 1e-9:
                axis = axis / n
                half = (ang * frac) / 2.0
                corr = np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)
        tcp = task.agent.tcp.pose
        tcp_p = _np(tcp.p)[0].astype(np.float64)
        tcp_q = _np(tcp.q)[0].astype(np.float64)
        q = np.asarray((sapien.Pose(q=corr) * sapien.Pose(q=tcp_q)).q, dtype=np.float64)
        # Torso frozen, and that is measured, not inherited. Freeing it — the
        # obvious move, since a prismatic joint lifts the cup at zero rotation cost
        # where the arm can only do it by turning — was tried on seeds 0-1 and is
        # worse on every axis: hand rotation 2557 -> 2616 deg, **1/2 instead of
        # 2/2**, and the cup past the spill line for 20.9% of the carry against
        # 8.9%. K52's "the tuck must not spend the torso" holds, and
        # `test_carry_pose_tucks_the_tcp_inside_the_base_footprint_with_the_torso_frozen`
        # is guarding a real property rather than pinning a preference.
        # **Do not execute the part of a refused correction that did plan.** K59 tried
        # exactly that, on the reasoning that a partial screw *is* the smaller bite this
        # loop's halving spends its time hunting for — 219 of the v3 sweep's levellings
        # were refused, `shoulder_lift` 122 of them with a median 0.385 of the twist
        # left, so a third of each correction was already in hand and was being thrown
        # away. Measured over a full 30-seed sweep it converted **8 of 91** attempts
        # here, and cost exactly what the contract above exists to protect: the arm ends
        # the partial standing *on* the joint limit that stopped it, so the RRT that
        # follows has to climb out of that corner and comes back **longer** (median 100
        # knots against 79 for the same stage without it), and two episodes lost the cup
        # outright — `object slipped out on the way in`, both previously green — to
        # partial rotations executed at the pour pose on a 2.75 cm rim grasp.
        #
        # The reasoning that was wrong is worth naming: "re-deriving from the new
        # configuration gives the solver a fresh Jacobian" ignores that the binding
        # constraint travels with the arm. A refused bite costs no steps. Keep it so.
        res = planner.static_manipulation(sapien.Pose(p=tcp_p, q=q), disable_lift_joint=True)
        if res != -1 and stopped_by_horizon(planner):
            break
        after_bite = _tilt_deg(_np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64))
        # No progress on this bite and nothing left to try smaller: give up rather
        # than spend the horizon on refusals.
        if res == -1 and frac >= 1.0:
            break
        if res == -1 and abs(after_bite - now) < 0.5:
            max_step_deg = max_step_deg / 2.0
            if max_step_deg < 5.0:
                break

    after_deg = _tilt_deg(_np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64))
    say(env, who, "levelled" if after_deg <= tol_deg else "level fell short",
        tilt_deg=round(float(after_deg), 1), was=round(float(before), 1))
    return bool(after_deg <= tol_deg)


def carry_pose(
    env, planner, task, obj, who: str = "oracle", *, ahead: float = 0.25, after: str | None = None,
    yaws_deg=CARRY_YAWS_DEG, lifts_m=CARRY_LIFTS_M,
    upright: bool = False, legs: int = 1, max_knots=None, knot_draws: int = 1,
):
    """Bring the held object into a carry pose over the base before driving (K52/D12).

    Why: `rotate_base_z` sweeps the turn's arc for collisions **with the attached
    object** (K51), and with the arm stretched out after a lift the held object
    can genuinely sweep through furniture. The object is pulled in: TCP at
    `base + ahead·face` (inside the base's footprint), at the TCP's current height,
    the object kept upright — its current orientation first, then the TCP yawed
    about the world's z axis by each entry of `yaws_deg` (the object stays vertical
    under a yaw); one `static_manipulation` per candidate with
    `disable_lift_joint=True` so the torso is not spent on the tuck; the first
    candidate that plans is executed. A candidate that does not plan costs no
    step (the solver executes nothing on a refused plan).

    History: until K53 (T5, 2026-08-18) this was a *recovery* only (D13) — under
    mplib 0.2.1's attached-body frame error (the held cup drawn 3.296 m off at
    shoulder_pan+90°) the tuck itself was unplannable on both calibration seeds:
    the held orientation on a real `shoulder_lift` limit, every upright
    reorientation on phantom collisions. With the frame fixed the reorientations
    plan (measured for the shipped set at 0.25 m ahead, `tools/probes/t5_carry_yaws.py`:
    RRT Success at yaw +90 / −90 / 180, the held orientation still on the real
    limit; `t3_carry_reach.py`'s +90/+135/+180 at 0.3–0.4 m agree), so the carry is
    a full stage again in the burner and the season-dish oracles: between the lift
    and the drive.

    After the move the grasp is verified with `task.agent.is_grasping(obj)`; no
    candidate planning, or a slipped object, is an honest `-1` ("carry pose
    unreachable") — a grasp/plan failure before any decision, per D6. A horizon cut
    is returned as the truncated tuple.

    Args:
        env: the (possibly wrapped) env, for `say`.
        planner: the solver (or the stub).
        task: `env.unwrapped`.
        obj: the held Actor.
        who: the oracle's tag.
        ahead: metres ahead of the base centre along its facing (default 0.25 —
            inside the Fetch base's ~0.3 m footprint).
        after: a refusal this tuck is recovering from, e.g. the `rotation sweep
            hits …` text; when given, the *one* `FAILED:` line a refused tuck emits
            is the chained `carry pose unreachable after: …`.
        yaws_deg: the upright candidates, in order (`CARRY_YAWS_DEG`).
        lifts_m: TCP heights above the current one to try, in order
            (`CARRY_LIFTS_M`); each height gets the full yaw ladder.
        upright: try a levelled orientation basis before the as-held one (K55,
            water-plants); False keeps the single-basis behaviour byte for byte.
        legs: above 1 the tuck is broken into short hops via `move_via`.
        max_knots: the K58 redraw knife — forwarded to the solver.
        knot_draws: the K58 redraw knife — forwarded to the solver.

    Example:
        >>> res = carry_pose(env, planner, task, task.cup, who="burner_planner")  # doctest: +SKIP
        >>> if res != -1 and stopped_by_horizon(planner): return res               # doctest: +SKIP
        >>> if res == -1: return res  # carry_pose already said why               # doctest: +SKIP
    """
    if upright:
        # Level the object **where it stands** before pulling it in. Asking for the
        # correction and the translation in one move is what made every candidate
        # refuse: after a pour the cup is 80 deg over, and "level it and tuck it"
        # is a much harder request than either half (measured, water-plants seeds
        # 0 and 1 — all four yaws refused, `carry pose unreachable`). A rotation
        # about the object in place keeps the wrist near where it already is.
        # Best effort: a refusal here costs no steps and the tuck below still runs.
        level_in_place(env, planner, task, obj, who=who)

    base = task.agent.base_link.pose
    base_p = _np(base.p)[0].astype(np.float64)
    face = _np(base.to_transformation_matrix())[0][:3, 0].astype(np.float64)
    face[2] = 0.0
    face = face / np.linalg.norm(face)
    tcp = task.agent.tcp.pose
    tcp_z = float(_np(tcp.p)[0][2])
    tcp_q = _np(tcp.q)[0].astype(np.float64)

    # K55. The docstring has always said "the object kept upright", and what the
    # code did was keep the object's *current* orientation — only upright if it
    # already was. Measured on water-plants seed 0 it was not: the tilt ratcheted
    # 0.0 -> 10.9 -> 31.7 -> 49.0 deg across the stages, each one preserving the
    # last one's error, and after a pour the cup was carried on at 108 deg.
    #
    # With `upright` the levelled orientation is tried **first** and the preserved
    # one is kept as a fallback rather than dropped. That ordering is the whole
    # design: asking for a large correction and the tuck translation in one move
    # is much harder than either half, and demanding it cost both seeds their
    # episode (`carry pose unreachable`, all four yaws, after an 80 deg pour). The
    # in-place levelling above removes most of the correction; this fallback means
    # that even when it does not, the tuck still plans and the episode survives
    # with a tilted cup — a worse number, not a lost run.
    #
    # `upright=False` keeps the old single-basis behaviour byte for byte: the
    # burner and season-dish oracles have published numbers measured against it.
    bases = [tcp_q]
    if upright:
        obj_q = _np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64)
        corr = level_correction(obj_q)
        levelled = np.asarray(
            (sapien.Pose(q=corr) * sapien.Pose(q=tcp_q)).q, dtype=np.float64
        )
        bases = [levelled, tcp_q]

    base_target = np.array([base_p[0], base_p[1], 0.0]) + ahead * face + np.array([0.0, 0.0, tcp_z])
    chain = f" after: {after}" if after else ""
    tried = []
    # **Every levelled candidate before any as-held one, and the order is load-bearing.**
    #
    # K59 tried interleaving them, on the measurement that over the v3 sweep the
    # levelled basis planned 86/139 at yaw 0 and then **0 of 100** at yaw -90 and 180,
    # while as-held planned 36/49 at yaw 0 — so three of the four levelled rungs looked
    # like pure waste standing in front of a rung that works two thirds of the time.
    # That measurement was of the wrong thing. Plan rate is not the outcome; keeping the
    # cup is. Interleaved, the sweep produced **2 `object slipped out on the way in` in
    # 16 episodes where the ordered ladder produced 0 in 30** — seeds 0 and 25, both
    # previously successful, both lost at a tuck that planned.
    #
    # The mechanism: a refused rung costs no steps, so reordering cannot change the pose
    # the surviving rung starts from — what it changes is *which* rung survives. The
    # rare levelled winner (yaw 90 planned 3 times in 30 episodes) is the one that tucks
    # a **levelled** cup; promoting as-held substitutes a rung that carries the cup's
    # pour tilt in through the base, and a 2.75 cm rim grasp does not survive that. The
    # dead rungs are the price of reaching the safe winner, not waste to be optimised
    # out.
    #
    # Inside a basis, each `lifts_m` height gets the full yaw ladder — a refused
    # rung is free, and the raised tuck is the season-dish held-out fallback that
    # runs only after every yaw at the current height has refused. The failure
    # form `tried_yaw_lift` counts lifts × yaws per basis.
    for basis_i, basis in enumerate(bases):
        for lift in lifts_m:
            target_p = base_target + np.array([0.0, 0.0, float(lift)])
            for yaw in yaws_deg:
                half = np.radians(float(yaw)) / 2.0
                q = (sapien.Pose(q=np.array([np.cos(half), 0.0, 0.0, np.sin(half)])) * sapien.Pose(q=basis)).q
                target = sapien.Pose(p=target_p, q=q)
                say(env, who, "carry pose", target=[round(float(v), 3) for v in target.p],
                    yaw_deg=float(yaw), lift=round(float(lift), 3),
                    basis="levelled" if (upright and basis_i == 0) else "as held")
                # K55: hold the cup level for the whole tuck, not just at its ends.
                # Only on the levelled basis — constraining the fallback would ask the
                # planner to preserve a tilt it is meant to be escaping.
                # K55: waypointed, for the same reason the other long arm moves are —
                # the tuck is a large reorientation and a single plan wanders through it.
                # K55: `legs > 1` breaks the tuck into short hops (`move_via`), for the
                # same reason the other long arm moves are broken up. Default 1 keeps
                # the burner and season-dish oracles on the single plan their published
                # numbers and their tests were measured against.
                res = (
                    move_via(env, planner, task, target, who=who, stage="carry pose",
                             legs=legs, disable_lift_joint=True,
                             max_knots=max_knots, knot_draws=knot_draws)
                    if legs > 1
                    else planner.static_manipulation(
                        target, disable_lift_joint=True, **_knot_kw(max_knots, knot_draws)
                    )
                )
                if res != -1 and stopped_by_horizon(planner):
                    return res
                if res == -1:
                    tried.append((float(yaw), round(float(lift), 3)))
                    continue
                if not bool(_np(task.agent.is_grasping(obj)).any()):
                    return fail(env, who, f"carry pose unreachable: object slipped out on the way in{chain}")
                if upright:
                    # Second attempt at levelling, now that the arm is tucked. The
                    # first one runs before the tuck, where a large correction is
                    # cheapest to plan *if* it plans at all — and after a pour it does
                    # not: the arm is at full extension over the pot and an 80 deg
                    # wrist correction is refused there (measured, both seeds). Pulled
                    # in over the base the same correction is an ordinary move, and
                    # leaving it undone is what carries a tipped cup through the whole
                    # drive back. Best effort, exactly as above.
                    level_in_place(env, planner, task, obj, who=who)
                return res
    return fail(env, who, f"carry pose unreachable{chain}", tried_yaw_lift=tried)




def arm_move(env, planner, pose, *, who: str, stage: str, disable_lift_joint: bool = False,
             tries: int = 2, legs: int = 1, max_knots: int | None = None,
             knot_draws: int = 1, knot_refuse: bool = False):
    """WaterPlants arm motion, including its measured waypointed leg mode."""
    if legs <= 1:
        return common.arm_move(
            env, planner, pose, who=who, stage=stage,
            disable_lift_joint=disable_lift_joint, tries=tries,
            max_knots=max_knots, knot_draws=knot_draws, knot_refuse=knot_refuse,
        )
    result = -1
    for i in range(int(tries)):
        result = move_via(
            env, planner, env.unwrapped, pose, who=who, stage=stage, legs=legs,
            disable_lift_joint=disable_lift_joint, max_knots=max_knots,
            knot_draws=knot_draws,
        )
        if result != -1:
            return result
        if i + 1 < int(tries):
            common.say(env, who, f"{stage}: plan refused, one more draw (RRT is randomized)")
    return result

