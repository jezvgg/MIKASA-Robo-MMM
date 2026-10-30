"""Cross-task motion primitives shared by distinct oracle planners.

Task-specific motion and calibration helpers live in ``oracle_<task>.py``. This
module owns the shared planner factories, status contract, contact handling, planning
budget, and motion primitives. Return ``-1`` only for a refusal before a physical
decision; preserve the last Gym tuple for physical misses and horizon truncations.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import sapien
import torch

# Refinement cap the oracles run the solver with: ~3 s of nudging at the control
# rate. The solver's own default is 200 (the inherited planners rely on it); an
# oracle would rather fail a stage inside the horizon than spend 10 s converging.
ORACLE_MAX_REFINE_STEPS = 60


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def say(env, who: str, stage: str, **extra) -> None:
    """Announce a stage on stdout and, when `env` is a PlannerLogger, in events.jsonl.

    The oracle's stdout is what the next debugging turn reads
    (docs/coding-agent-primer.md §2-3). One line per stage, `[{who}] {stage} ...`,
    so it can be grepped out of the solver's IK/RRT prints; and the same line as a
    `phase` event when the sweep wrapped the env with
    utils.logging_utils.PlannerLogger, so `--log-dir` traces name the stage
    next to the step.

    Args:
        env: the (possibly wrapped) env passed to `solve`.
        who: the oracle's tag, e.g. `"burner_planner"`.
        stage: short stage name, e.g. `"reach cup"`.
        **extra: JSON-serialisable fields for the event line (target burner, ...).

    Example:
        >>> say(env, "burner_planner", "drive to stove", target=2)  # doctest: +SKIP
        [burner_planner] drive to stove target=2
    """
    tail = " ".join(f"{k}={v}" for k, v in extra.items())
    print(f"[{who}] {stage}" + (f" {tail}" if tail else ""), flush=True)
    log = getattr(env, "log_event", None)
    if callable(log):
        log("phase", stage, **extra)


def fail(env, who: str, stage: str, **extra):
    """Return -1 for a failed stage, saying which one — never a silent -1.

    Example:
        >>> if res == -1: return fail(env, "burner_planner", "reach cup")  # doctest: +SKIP
    """
    say(env, who, f"FAILED: {stage} (planner returned -1)", **extra)
    return -1


def stopped_by_horizon(planner) -> bool:
    """True once the episode was truncated under this planner's feet.

    Reads `planner.truncated` — the latch `StepGuard` keeps on the real solver and
    `tools/stub_planner.py` mirrors. Check it after **every** primitive, before
    interpreting the primitive's result (K23): once the horizon has struck, a
    still-open gripper or an unread `-1` is the horizon's doing, and the honest
    return value is the truncated 5-tuple the primitive handed back — the sweep
    then books the episode as `truncated`, not as `no_plan`.

    Example:
        >>> res = planner.idle_steps(t=1)                           # doctest: +SKIP
        >>> if stopped_by_horizon(planner):                          # doctest: +SKIP
        ...     return res  # truncated: preserve the Gym tuple
    """
    return bool(getattr(planner, "truncated", False))


#: TOPP limits the solver plans the ARM under when the env is in `pd_joint_delta_pos`
#: (`default_planner_factory`), for a probe: 0 (the default) keeps the solver's own.
DELTA_JOINT_VEL_LIMIT = float(os.environ.get("MIKASA_DELTA_VEL", "0"))
DELTA_JOINT_ACC_LIMIT = float(os.environ.get("MIKASA_DELTA_ACC", "0.9"))


def default_planner_factory(env, debug: bool, vis: bool, *, max_refine_steps: int = ORACLE_MAX_REFINE_STEPS,
                            joint_vel_limits=None, joint_acc_limits=None, solver_class=None):
    """The real Fetch solver, capped at the oracle refinement budget. Needs mplib.

    Imported here, not at the top, so this module (and the oracles that defer to
    it) import on a Mac.

    Args:
        env: the (possibly wrapped) env.
        debug, vis: passed through to the solver.
        max_refine_steps: per-instance cap on the post-path refinement loop
            (`ORACLE_MAX_REFINE_STEPS` = 60 ≈ 3 s); the solver's class default of
            200 is for the inherited planners.
        joint_vel_limits, joint_acc_limits: per-instance TOPP limits (solver default
            0.9 each). `None` passes nothing, so every existing caller builds a
            byte-identical solver — the only form in which touching shared solver
            configuration is defensible (K65). Lowering them trades episode steps for
            tracking accuracy, which is the quantity the K79c post-mortems measured as
            the root cause: 2-7 deg of joint lag, 7-9 cm of Cartesian error at 0.8 m
            extension, against collision margins of 3-6 cm.

    Example:
        >>> planner = default_planner_factory(env, debug=False, vis=False)  # doctest: +SKIP
    """
    from robots.fetch.extand import FetchMotionPlanningSapienSolver

    if joint_vel_limits is None and DELTA_JOINT_VEL_LIMIT > 0 \
            and getattr(env.unwrapped, "control_mode", None) == "pd_joint_delta_pos":
        # Optional, measured OFF (2026-09-09): planning the arm slower under the delta
        # controller (whose 0.1 rad step caps the PD torque and so the speed) moved the
        # end configurations of Retrieval's stages onto joint limits (1101, 1102 refused
        # at 0.6 and 0.75 where 0.9 passed). The lag is handled where it arises instead:
        # `follow_forward_path_w_refinement` gates the knot on the arm's lag.
        joint_vel_limits = DELTA_JOINT_VEL_LIMIT
        if joint_acc_limits is None:
            joint_acc_limits = DELTA_JOINT_ACC_LIMIT
    solver_class = FetchMotionPlanningSapienSolver if solver_class is None else solver_class
    return solver_class(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        max_refine_steps=max_refine_steps,
        **({} if joint_vel_limits is None else {"joint_vel_limits": joint_vel_limits}),
        **({} if joint_acc_limits is None else {"joint_acc_limits": joint_acc_limits}),
    )


def collection_planner_factory(*args, **kwargs):
    """Collection motion commands without changing DSFetch or its controllers."""
    from planners.oracle.collection_solver import CollectionMotionPlanner
    return default_planner_factory(*args, solver_class=CollectionMotionPlanner, **kwargs)


@contextlib.contextmanager
def touchable(planner, needle: str):
    """Take every planning-world object whose name contains `needle` OUT of the world for
    the block, and put it back after — for a stroke whose GOAL is contact.

    mplib plans collision-free by construction; a pose inside the cube, which is what a
    push commands, is refused (`finger_link <-> cube` in the screw's report, 2026-09-08).
    The sim still collides — that is the push — only the planner stops objecting.
    Re-added under the same name after the stroke so later plans see the object where
    the simulator left it after the next `update_from_simulation`.

    Example:
        >>> with touchable(planner, "cube"):                     # doctest: +SKIP
        ...     res = planner.static_manipulation(push)          # the pose inside the cube plans
    """
    world = planner.planner.planning_world
    taken: list = []
    try:
        for name in list(world.get_object_names()):
            if needle in name:
                obj = world.get_object(name)
                planner.planner.remove_object(name)
                taken.append((name, obj))
        yield
    finally:
        for _name, obj in taken:
            world.add_object(obj)  # an FCLObject carries its own name


@contextlib.contextmanager
def contact_stroke(planner, articulation_names):
    """Remove named articulations from the planning world for the block (K100).

    A drawer or a door can only be moved by TOUCHING it, and every planning leg in
    this solver is collision-checked — `SapienPlannerV2.plan_screw` walks the stroke
    against the world and RRT goals inside the handle geometry have no valid IK — so
    a push or a pull aimed AT an articulated fixture refuses by construction (the
    upstream Panda pushes only because its vanilla `plan_screw` never checks). For a
    contact stroke the fixture being moved is the goal, not an obstacle: take exactly
    it out of the world for the leg, and leave every OTHER fixture in, so a stroke at
    drawer 3 still plans around drawers 1, 2 and 4.

    Restored on every exit path, then re-synced by the caller's next
    `update_from_simulation` (re-adding uses the model as it stood; the sync puts its
    links back where the simulator has them).

    Args:
        planner: the solver (needs `.planner.planning_world`).
        articulation_names: names to remove; matched by substring against
            `get_articulation_names()`, because the world prefixes scene names.

    Example:
        >>> with contact_stroke(planner, ["stack_1_main_group_3_0"]):   # doctest: +SKIP
        ...     res = planner.static_manipulation(push_pose)            # may touch it
    """
    world = planner.planner.planning_world
    removed = []
    try:
        for full in list(world.get_articulation_names()):
            if any(want in full for want in articulation_names):
                model = world.get_articulation(full)
                world.remove_articulation(full)
                removed.append(model)
        yield
    finally:
        for model in removed:
            world.add_articulation(model)


#: `fingers.sum()` at or below this is an empty hand — the W12 threshold: a closed
#: mimic gripper bottoms out at 0.000 on nothing, and 8 mm between the pads is less
#: than any handle bar or edge in kitchen 102 (the bar alone is 26 mm).
FINGER_EMPTY_M = 0.008


def hinge_anchor(task, articulation_name, joint_name):
    """World axis of a vertical hinge and the sense of +qpos, from the raw frames.

    Reads the joint's own frame — `parent_link.entity_pose * pose_in_parent`, the
    parent being the fixture's body so the anchor does not move with the door — and
    takes the axis from the frame's x column (a SAPIEN revolute rotates about its
    frame's x). W14 verified this read against an independent three-point circle fit
    of the scanned handle bar: 0.0 cm apart, same sense.

    Args:
        task: `env.unwrapped`.
        articulation_name: key into `task.scene.articulations`.
        joint_name: the hinge's active-joint name, e.g. `"rightdoorhinge"`.

    Returns:
        `(anchor_xy, sense)`: the axis' world xy as float64, and +1 when +qpos is
        CCW about world +z (-1 otherwise).

    Raises:
        ValueError: the axis is not vertical — the arc-pull law is planar and has
            no meaning for such a hinge.

    Example:
        >>> H, s = hinge_anchor(task, "cab_main_main_group_0", "rightdoorhinge")  # doctest: +SKIP
        >>> H.round(3), s                                                         # doctest: +SKIP
        (array([ 2.735, -0.4  ]), 1)
    """
    art = task.scene.articulations[articulation_name]
    joint = [j for j in art.get_active_joints() if j.name == joint_name][0]
    raw = joint._objs[0]
    anchor = raw.get_parent_link().entity_pose * raw.get_pose_in_parent()
    T = anchor.to_transformation_matrix()
    axis_world = T[:3, 0]
    if abs(float(axis_world[2])) < 0.99:
        raise ValueError(
            f"{articulation_name}/{joint_name}: hinge axis {np.round(axis_world, 3)} "
            "is not vertical; the arc-pull law is planar")
    sense = 1 if float(axis_world[2]) > 0 else -1
    return T[:2, 3].astype(np.float64), sense


@contextlib.contextmanager
def planning_budget(seconds: float):
    """Temporarily widen the unified solver's `PLANNING_TIME` for one oracle's `solve()`.

    `PLANNING_TIME` is module state shared by all eight oracles, so raising the
    module default would silently re-plan the burner and water-plants runs too —
    the thing K65 says may not be done to shared solver code. A context manager
    scopes it to one caller and restores it on every exit path.

    RRTConnect stops the moment its two trees meet and never improves the path
    afterwards, so a wider budget costs nothing on a plan that succeeds; it is
    paid only by plans that would otherwise have been refused. That is why the
    wall-clock cost of widening it is small (K79: +4.5% median episode) while the
    refusal count halves.

    The collection profiles provide an explicit finite budget.

    Example:
        >>> with planning_budget(8.0):  # doctest: +SKIP
        ...     search_one_stage()
    """
    import robots.fetch.extand as solver_module
    old = solver_module.PLANNING_TIME
    solver_module.PLANNING_TIME = float(seconds)
    try:
        yield
    finally:
        solver_module.PLANNING_TIME = old


def hold_object_in_planner(env, planner, task, obj, held: bool, who: str = "oracle",
                           extra_touch=(), pose=None) -> None:
    """Tell the planning world that `obj` is (or no longer is) part of the hand.

    Without this, the first emulated burner run died one step into the drive:
    `plan_screw` reported `collision scene-0-ds_fetch_gripper_link<->scene-0_cup_110`
    — the cup it had just grasped was still an obstacle in the planning world, so
    every state with the fingers closed on it "collides". `attach_object`
    (robots/fetch/utils.py) moves the object into the
    gripper's frame and allows its contact with the touching links; `detach_object`
    puts it back as an obstacle once released, so the retract plan sees it where it
    landed. The planning world is synced right after the attach, so the stored
    link->object transform is taken at the configuration the hand is actually in
    (K52); and since K53 the planning world holds the robot with its base pose
    folded into the root joints, so that transform is the rigid `T_link_obj` and
    the held object is drawn where the hand is in every configuration — a hover,
    a tuck (`carry_pose`) and a pour tilt are collision-checked against the object
    itself, not against the 3 m phantom of before.

    Args:
        env: the (possibly wrapped) env, for `say`.
        planner: the FetchMotionPlanningSapienSolver.
        task: `env.unwrapped`.
        obj: the ManiSkill Actor being held (e.g. `task.cup`).
        held: True after a verified grasp, False after the release.
        who: the oracle's tag, for the trace.
        extra_touch: link-name stems (e.g. "shoulder_lift") whose contact with the held
            object is allowed in the model, on top of the whole hand.
        pose: gripper_link -> object to attach at (an mplib.Pose); None reads it from the
            simulator, as before. A perception-only planner passes its own estimate.

    Example:
        >>> hold_object_in_planner(env, planner, task, task.cup, held=True, who="burner_planner")  # doctest: +SKIP
        >>> # ... drive, hover, lower, open_gripper
        >>> hold_object_in_planner(env, planner, task, task.cup, held=False, who="burner_planner")  # doctest: +SKIP
    """
    world = planner.planner.planning_world
    from robots.fetch.utils import attach_object, convert_object_name

    entity = obj._objs[0]
    robot = task.agent.robot._objs[0]
    if held:
        link = next(l for l in robot.links if l.name.endswith("gripper_link"))
        # touch_links spelled out: the planning world holds the object as a convex
        # hull conversion, which fills a cup's
        # opening, so the palm above the rim is "inside" it and mplib's
        # auto-detected touch set (links colliding at attach time) still left
        # gripper_link<->object as a collision on the very next plan. Allow the
        # whole hand.
        touch = [l for l in robot.links if any(k in l.name for k in ("gripper", "wrist"))]
        # `extra_touch`: link-name stems a caller has measured the held object to touch
        # harmlessly (SeasonDish 2025, 2026-09-06: the carry tuck rests a 16 cm bottle
        # against `shoulder_lift_link`, and the drive's start state is then a collision
        # in the model though the object is held). Allowed for this attachment only.
        touch += [l for l in robot.links
                  if any(k in l.name for k in extra_touch) and l not in touch]
        attach_object(world, entity, robot, link, pose=pose, touch_links=touch)
        planner.planner.update_from_simulation()  # the transform at the hand's true configuration
        say(env, who, "object attached to gripper_link in the planning world",
            touch_links=[l.name.split("_", 1)[-1] for l in touch])
    else:
        world.detach_object(convert_object_name(entity))
        say(env, who, "object detached in the planning world")


#: The tucks `carry_pose` tries, in order: the TCP yawed about the world's z axis by
#: this much (degrees) from its current orientation — the object stays upright under
#: each. 0 = orientation held (least motion). Measured for exactly this set at the
#: shipped 0.25 m ahead (`tools/probes/t5_carry_yaws.py`, burner seed 3 post-lift,
#: journal 2026-08-18): 0 → screw `joint limit at index [7]` (a real shoulder_lift
#: limit) + `IK Failed`; +90, −90 and 180 → RRT `Success` (the screw refuses all
#: three on the same limit).
def _knot_kw(max_knots, knot_draws, knot_refuse: bool = False) -> dict:
    """The knot-cap kwargs, or nothing at all.

    Passed only when a caller asked for a cap: the oracles' test doubles implement
    `static_manipulation(pose, disable_lift_joint=...)` and nothing more, and widening
    the call unconditionally breaks them for a feature they never use. `knot_refuse`
    rides along only when asked for (SeasonDish's pre-hover waypoint, 1507 @0.15).
    """
    if max_knots is None:
        return {}
    kw = dict(max_knots=max_knots, knot_draws=knot_draws)
    if knot_refuse:
        kw["knot_refuse"] = True
    return kw


def _tilt_deg(q) -> float:
    """Angle in degrees between the pose's body +Z and world +Z."""
    R = sapien.Pose(q=np.asarray(q, dtype=np.float64)).to_transformation_matrix()[:3, :3]
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def arm_move(env, planner, pose, *, who: str, stage: str, disable_lift_joint: bool = False,
             tries: int = 2, max_knots: int | None = None,
             knot_draws: int = 1, knot_refuse: bool = False):
    """`static_manipulation(pose)` with up to `tries` draws of the plan.

    A refused plan returns `-1` without stepping the env, and the solver's RRT
    fallback (`RRTConnect`, 4 s under emulation) is randomized: on the season dish's
    seed 12 the same reachable pose (IK found it) came back `RRTConnect Failed.
    Approximate solution` on one draw and as a 94-knot path on the next. So a stage
    that is RRT-bound — a hover from the tucked arm, a reach from the rest posture —
    gets one more draw before it is booked as a failed plan. Nothing is executed
    twice: the first draw that plans is the one that runs. Check the horizon after
    the call as usual (K23).

    Args:
        env, planner, who: as `say`.
        pose: the TCP target (sapien.Pose).
        stage: the stage's name for the retry line.
        disable_lift_joint: passed to `static_manipulation`.
        tries: draws in total (default 2).
        max_knots: if set, prefer an RRT path no longer than this many knots.
        knot_draws: how many RRT draws to take when `max_knots` is set; the shortest
            is kept. Draws cost no simulation steps, only planner time.

    Returns:
        the primitive's result: the 5-tuple, or -1 after the last refused draw.

    Example:
        >>> res = arm_move(env, planner, hover, who=WHO, stage="hover over bowl")  # doctest: +SKIP
        >>> if res != -1 and stopped_by_horizon(planner): return res              # doctest: +SKIP
        >>> if res == -1: return fail(env, WHO, "hover over bowl")                # doctest: +SKIP
    """
    res = -1
    for i in range(int(tries)):
        res = planner.static_manipulation(
            pose, disable_lift_joint=disable_lift_joint,
            **_knot_kw(max_knots, knot_draws, knot_refuse),
        )
        if res != -1:
            return res
        if i + 1 < int(tries):
            say(env, who, f"{stage}: plan refused, one more draw (RRT is randomized)")
    return res


def dock_error(task, dock_xyyaw) -> tuple[float, float]:
    """`(d_dock, dyaw_deg)`: how far the base parked from a dock, in xy metres and yaw degrees.

    The one parking measurement the memory tasks share (K26): the burner logs it
    after every drive, and it is the number that decides the station task's zone
    mode (D5: `radius` if the parking error stays well under 0.10 m, `nearest`
    otherwise). Reads `task.agent.base_link.pose`; nothing else.

    Args:
        task: `env.unwrapped`.
        dock_xyyaw: `(x, y, yaw)` of the dock the drive aimed at.

    Returns:
        `(d_dock, dyaw_deg)`: xy distance in metres, absolute heading error in
        degrees (wrapped to [0, 180]).

    Example:
        >>> d_dock, dyaw = dock_error(task, (3.134, -1.21, math.pi / 2))  # doctest: +SKIP
        >>> say(env, "burner_planner", "parked at the dock", d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))  # doctest: +SKIP
    """
    d = np.asarray(dock_xyyaw, dtype=np.float64).reshape(-1)
    pose = task.agent.base_link.pose
    base_p = _np(pose.p).reshape(-1, 3)[0]
    base_x = _np(pose.to_transformation_matrix()).reshape(-1, 4, 4)[0][:3, 0]
    d_dock = float(np.linalg.norm(base_p[:2] - d[:2]))
    yaw = float(np.arctan2(base_x[1], base_x[0]))
    dyaw = float(abs((yaw - float(d[2]) + np.pi) % (2 * np.pi) - np.pi))
    return d_dock, float(np.degrees(dyaw))


# ------------------------------------- the joint-space channel (K111, DepthRecall) --



def plan_joints(env, planner, task, targets: dict, *, label: str, tries: int = 2,
                who: str = "oracle", line_only: bool = False,
                trace_rrt: bool = True, qpos_step: float | None = None):
    """Plan and execute a joint-space move: straight line first, RRT as the fallback.

    The line is the verified channel; the RRT draw is what the validation seed sets rely
    on when a straight joint line is blocked or degenerate (already at the goal). A
    planner that cannot time a path has refused the leg, not crashed the episode.
    """
    p = planner.planner
    robot = task.agent.robot
    for i in range(int(tries)):
        cur = _np(robot.get_qpos()).reshape(-1).astype(np.float64)
        goal = cur.copy()
        for name, value in targets.items():
            goal[robot.active_joints_map[name].active_index[0].item()] = float(value)
        try:
            line = p.plan_qpos_line(
                goal, cur, time_step=task.control_timestep,
                **({} if qpos_step is None else {"qpos_step": float(qpos_step)}),
                ref_yaw=float(cur[2]),
            )
        except RuntimeError as exc:
            line = {"status": f"mplib raised: {exc}", "position": []}
        if line["status"] == "Success" and len(np.asarray(line["position"])) > 1:
            say(env, who, f"{label}: joint line",
                knots=int(np.asarray(line["position"]).shape[0]))
            return planner.follow_forward_path_w_refinement(line, refine=True)
        if line_only:
            # The caller wants the one channel that measurably executes or nothing.
            say(env, who, f"{label}: line only; no RRT draw", status=line["status"])
            return -1
        try:
            result = p.plan_qpos(
                [p.fold_qpos(p.pad_move_group_qpos(goal))],
                p.fold_qpos(p.pad_move_group_qpos(cur)),
                time_step=task.control_timestep, planning_time=8.0, rrt_range=0.1,
                simplify=True, fixed_joint_indices=[0, 1, 2], ref_yaw=float(cur[2]),
            )
        except RuntimeError as exc:
            result = {"status": f"mplib raised: {exc}", "position": []}
        if result["status"] == "Success":
            if trace_rrt:
                say(env, who, f"{label}: rrt joints",
                    knots=int(np.asarray(result["position"]).shape[0]))
            return planner.follow_forward_path_w_refinement(result, refine=True)
        say(env, who, f"{label}: plan refused", status=result["status"], draw=i + 1)
    return -1


REST_TORSO = 0.386


def fold_via_tcp(env, planner, task, rest_tcp, *, stage: str, who: str = "oracle",
                 rest_torso: float = REST_TORSO):
    """Return to the rest CARRIAGE via the channels that measurably execute.

    A torso LINE first (the duck's own form, reversed), then a Cartesian move
    (`static_manipulation`, torso frozen) to the episode-start rest TCP point
    re-based onto the current base pose, then a branch LINE from the actual
    measured joint coordinates to the rest configuration. No qpos wrapping:
    every physical change must be represented by recorded actions. Solo 5/6 (K111) measured
    `plan_joints`' RRT branch executing 185 knots and moving the arm by NOTHING
    (arm_vs_rest 0.562 before and after, bit-for-bit) — the line branch and
    `static_manipulation` are the two channels this repo has verified end to end,
    so the fold uses only those. Stage 2 exists because the screw lands the TCP on
    the rest POINT while the arm can sit on another IK branch (solo 11:
    arm_vs_rest ~4 rad with the fold "ok"), and a frozen weird-branch arm honestly
    blocks the next drive's rotate; a branch line that flakes (arm_vs_rest > 1.0
    after it) gets one full retry round from a re-screwed start. Non-fatal, like
    the fold it replaces: the return value is the screw's, and the caller reads
    `arm_vs_rest` from qpos if it needs the truth.

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        rest_tcp: the rest TCP pose IN THE BASE FRAME, captured at solve start
            (`base_link.pose.sp.inv() * tcp.pose.sp` with the arm still on the
            reset keyframe), so the re-basing is exact under any dock yaw.
        stage: prefix for the trace lines.
        who: the oracle's tag for `say`.
        rest_torso: the un-duck target (`REST_TORSO`, the rest keyframe's lift).

    Returns:
        The screw's result: the gym 5-tuple, or -1 when both draws refused.

    Example:
        >>> rest_tcp = task.agent.base_link.pose.sp.inv() * task.agent.tcp.pose.sp  # doctest: +SKIP
        >>> ...                                                                   # doctest: +SKIP
        >>> fold_via_tcp(env, planner, task, rest_tcp, stage="fold", who=WHO)     # doctest: +SKIP
    """
    plan_joints(env, planner, task, {"torso_lift_joint": float(rest_torso)},
                label=f"{stage}: un-duck", who=who)
    goal = task.agent.base_link.pose.sp * rest_tcp
    r = planner.static_manipulation(goal, disable_lift_joint=True)
    if r == -1:
        r = planner.static_manipulation(goal, disable_lift_joint=True)
    say(env, who, f"{stage}: fold via tcp {'ok' if r != -1 else 'REFUSED'}")
    planner.planner.update_from_simulation()
    rest_q = np.asarray(task.agent.keyframes["rest"].qpos,
                        dtype=np.float64).reshape(-1)
    jm = task.agent.robot.active_joints_map
    arm_names = list(task.agent.controller.controllers["arm"].config.joint_names)
    targets = {n: float(rest_q[jm[n].active_index[0].item()])
               for n in arm_names}
    plan_joints(env, planner, task, targets, label=f"{stage}: branch line",
                tries=1, who=who)
    q_now = _np(task.agent.robot.get_qpos()).reshape(-1)
    d_rest = float(np.linalg.norm(q_now[3:11] - rest_q[3:11]))
    say(env, who, f"{stage}: post-branch arm_vs_rest", d=round(d_rest, 3))
    if d_rest > 1.0:
        # the tf-sweep S2/S6 mechanism: a flaked branch line leaves the arm
        # on a weird branch and every following fetch pre refuses. One full
        # retry round — the TCP screw moves the arm first, so the line
        # starts from a different config.
        say(env, who, f"{stage}: branch line flaked — one retry round")
        goal2 = task.agent.base_link.pose.sp * rest_tcp
        r2 = planner.static_manipulation(goal2, disable_lift_joint=True)
        if r2 == -1:
            planner.static_manipulation(goal2, disable_lift_joint=True)
        planner.planner.update_from_simulation()
        plan_joints(env, planner, task, targets,
                    label=f"{stage}: branch line retry", tries=1, who=who)
        q_now = _np(task.agent.robot.get_qpos()).reshape(-1)
        say(env, who, f"{stage}: post-retry arm_vs_rest",
            d=round(float(np.linalg.norm(q_now[3:11] - rest_q[3:11])), 3))
    return r
