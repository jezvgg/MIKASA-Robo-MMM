"""Plan a one-joint pour; reject it rather than recruit other arm joints.

Two pours. The one-tip pour turns the wrist until the condiment is tilted `target_degrees` (165) and
holds. The shaking pour (SeasonDish since 2026-10-27) turns it to just short of upside down, then per
shake turns it `SHAKE_AMPLITUDE_DEG` back toward upright and returns, and holds there; only wrist_roll
moves in either.

The shakes go to one side only. Checklist D2 lets each roll joint span at most pi - 0.05 over an
episode (`roll_paths.RollPathPlanner.accepts`), and turning upside down already takes ~180 deg, so
a swing past upside down is never accepted: the +-30 deg version was refused on 8 of 8 seeds
(2026-10-28), and the owner chose the one-sided swing.
"""
import math
from contextlib import contextmanager

import numpy as np

from planners.oracle import oracle_common as common

#: Wrist roll each shake swings back toward upright and returns (the owner, 2026-10-27: 30 deg;
#: one side only since 2026-10-28, see above).
SHAKE_AMPLITUDE_DEG = 30.0
#: Lean kept clear of the task's shake thresholds by the planned rest and shake poses, degrees: the
#: route preview (a model of the held condiment) keeps 3, the pour itself (the measured one) 1.5, so a
#: degree of model-to-physics drift does not refuse at the bowl a pour the preview accepted. The rest
#: leans exactly `shake_return_deg` less the margin: the least roll that still re-arms the counter.
SHAKE_LEAN_MARGIN_DEG = 3.0
SHAKE_LEAN_MARGIN_EXECUTED_DEG = 1.5
#: Speed and acceleration limits the shake legs are timed with, rad/s and rad/s^2. The planner's own,
#: 0.9 and 0.9 (robots/fetch/extand.py's default, no rule in BIBLE or the checklist), took ~2.8 s per
#: shake; the owner asked for sharper shakes (2026-10-28 «можно более резко трясти?»). 2.0 stays under
#: the URDF's 2.268 for the wrist joints.
SHAKE_SPEED, SHAKE_ACCELERATION = 2.0, 8.0
#: Control steps the pour may wait after a leg for the shake counter to see it (the arm lags its
#: command). The legs used to end in the solver's refinement, which waits for the joint itself: 1-3
#: identical 10 Hz frames a leg, and with nine shakes that alone broke D6's 5 % (2026-10-28).
SETTLE_STEPS = 20


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def tilt_angles(axis, object_axis, target_degrees=165.):
    """All signed rotations within one turn that reach the requested tilt."""
    n, v = np.asarray(axis, dtype=float), np.asarray(object_axis, dtype=float)
    n, v = n / np.linalg.norm(n), v / np.linalg.norm(v)
    c0 = float(n[2] * (n @ v))
    c1, c2 = float(v[2] - c0), float(np.cross(n, v)[2])
    radius = math.hypot(c1, c2)
    if radius < 1e-9:
        return []
    value = (math.cos(math.radians(target_degrees)) - c0) / radius
    if abs(value) > 1 + 1e-9:
        return []
    phase, offset = math.atan2(c2, c1), math.acos(float(np.clip(value, -1, 1)))
    values = {phase + sign * offset + turn * 2 * math.pi
              for sign in (-1, 1) for turn in (-1, 0, 1)}
    return sorted((x for x in values if abs(x) <= 2 * math.pi), key=abs)


def lean_after(axis, object_axis, delta):
    """Lean of `object_axis` from straight down, degrees, after turning it `delta` about `axis`.

    Example:
        >>> round(lean_after([1, 0, 0], [0, 0, 1], math.pi), 1), round(lean_after([1, 0, 0], [0, 0, 1], math.pi + math.radians(30)), 1)
        (0.0, 30.0)
    """
    v = rotation(axis, delta) @ (np.asarray(object_axis, dtype=float) / np.linalg.norm(object_axis))
    return math.degrees(math.acos(float(np.clip(-v[2], -1.0, 1.0))))


def rest_angles(axis, object_axis, lean_deg):
    """The turns about `axis` that leave `object_axis` `lean_deg` short of straight down, smallest
    first: the near side of upside down, where the roll joint spends least of its D2 span.

    Example:
        >>> [round(math.degrees(d)) for d in rest_angles([1, 0, 0], [0, 0, 1], 7.)][:2]
        [173, -173]
        >>> rest_angles([0, 0, 1], [0, 0, 1], 7.)
        []
    """
    return sorted(tilt_angles(axis, object_axis, 180. - lean_deg), key=lambda x: (round(abs(x), 9), x < 0))


def swing_angle(delta, amplitude_deg):
    """The shake's turn: `amplitude_deg` from the rest turn `delta` back toward the start."""
    return delta - math.copysign(math.radians(amplitude_deg), delta)


def shake_geometry_ok(axis, object_axis, delta, cfg, amplitude_deg=SHAKE_AMPLITUDE_DEG,
                      margin_deg=SHAKE_LEAN_MARGIN_DEG):
    """Does turning to `delta` rest the condiment inside the counter's rest band, and does the swing
    `amplitude_deg` back toward upright from there lean it past the shake threshold, with the margin?

    Example:
        >>> from types import SimpleNamespace
        >>> cfg = SimpleNamespace(shake_out_deg=20., shake_return_deg=10.)
        >>> shake_geometry_ok([1, 0, 0], [0, 0, 1], math.radians(173), cfg)
        True
        >>> shake_geometry_ok([1, 0, 0], [0, 0, 1], math.radians(173), cfg, amplitude_deg=15.)
        False
    """
    rest = lean_after(axis, object_axis, delta)
    swing = lean_after(axis, object_axis, swing_angle(delta, amplitude_deg))
    return rest <= cfg.shake_return_deg - margin_deg + 1e-6 and swing >= cfg.shake_out_deg + margin_deg


@contextmanager
def sharp(planner):
    """Time the lines planned inside with the shake legs' limits (mplib's TOPP reads them off the planner)."""
    p = planner.planner
    speed, acceleration = p.joint_vel_limits, p.joint_acc_limits
    p.joint_vel_limits = np.full(len(speed), SHAKE_SPEED)
    p.joint_acc_limits = np.full(len(acceleration), SHAKE_ACCELERATION)
    try:
        yield
    finally:
        p.joint_vel_limits, p.joint_acc_limits = speed, acceleration


def _over_bowl(task, end, bowl):
    clearance = end[2] - bowl[2]
    return (np.linalg.norm(end[:2] - bowl[:2]) <= task.cfg.pour_xy_radius
            and task.cfg.pour_min_clearance <= clearance <= task.cfg.pour_max_clearance)


def _plan_shakes(planner, task, q, wrist_index, axis, pivot, obj, bowl, amplitude_deg):
    """The first near-upside-down turn whose rest pose and swing stay over the bowl and plan as
    wrist-only lines: (rest, {leg name: path}) or None. Legs: in (q -> rest), out (rest -> swing), back."""
    object_axis = obj[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    margin = SHAKE_LEAN_MARGIN_EXECUTED_DEG
    line = lambda goal, start: planner.planner.plan_qpos_line(
        goal, start, time_step=task.control_timestep, ref_yaw=float(start[2]), qpos_step=0.02)
    for delta in rest_angles(axis, object_axis, task.cfg.shake_return_deg - margin):
        if not shake_geometry_ok(axis, object_axis, delta, task.cfg, amplitude_deg, margin_deg=margin):
            continue
        swing = swing_angle(delta, amplitude_deg)
        ends = [pivot + rotation(axis, d) @ (obj[:3, 3] - pivot) for d in (delta, swing)]
        if not all(_over_bowl(task, end, bowl) for end in ends):
            continue
        rest = q.copy(); rest[wrist_index] += delta
        out = q.copy(); out[wrist_index] += swing
        try:
            legs = {"in": line(rest, q)}
            with sharp(planner):
                legs.update(out=line(out, rest), back=line(rest, out))
        except RuntimeError:
            continue
        if all(leg.get('status') == 'Success' for leg in legs.values()):
            return rest, legs, out
    return None


def pour_and_shake(env, planner, task, target, shakes, amplitude_deg=SHAKE_AMPLITUDE_DEG):
    """Turn the held condiment just short of upside down over the bowl with the wrist alone, shake it
    `shakes` times (back toward upright and again), and hold it there until the task's hold succeeds.

    Every leg is planned before the first one moves; all other actuator targets stay fixed.
    """
    log = lambda text, **kw: common.say(env, "season_dish_planner", text, **kw)
    robot = task.agent.robot
    joints = robot.active_joints_map
    q = robot.get_qpos()[0].cpu().numpy().astype(float)
    wrist_index = int(joints['wrist_roll_joint'].active_index[0])
    wrist = robot.links_map['wrist_roll_link'].pose[0].sp.to_transformation_matrix()
    obj = target.pose[0].sp.to_transformation_matrix()
    axis, pivot = wrist[:3, 0], wrist[:3, 3]
    bowl = task.bowl.pose.p[0].cpu().numpy()
    if not bool(task.agent.is_grasping(target).any()):
        return common.fail(env, "season_dish_planner", "wrist pour requires a held object")
    planner.planner.update_from_simulation()
    plan = _plan_shakes(planner, task, q, wrist_index, axis, pivot, obj, bowl, amplitude_deg)
    if plan is None:
        return common.fail(env, "season_dish_planner", "no feasible wrist-only shake")
    rest, legs, out = plan
    object_axis = obj[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    rest_tilt = 180.0 - lean_after(axis, object_axis, rest[wrist_index] - q[wrist_index])
    names = task.agent.controller.controllers['arm'].config.joint_names
    fixed = {i: float(q[int(joints[name].active_index[0])])
             for i, name in enumerate(names) if name != 'wrist_roll_joint'}
    fixed.update({7: -1., 10: float(q[int(joints['torso_lift_joint'].active_index[0])]),
                  11: 0., 12: 0.})
    previous = getattr(planner, 'fixed_action_targets', {})
    planner.fixed_action_targets = fixed
    stopped = lambda r: (isinstance(r, (int, np.integer)) and r == -1) or common.stopped_by_horizon(planner)
    try:
        log('pour', motion='wrist_roll_only', target_degrees=round(rest_tilt, 2), shakes=int(shakes),
            amplitude_deg=float(amplitude_deg), sides=1, wrist_start=float(q[wrist_index]),
            wrist_goal=float(rest[wrist_index]), shake_speed=SHAKE_SPEED, shake_acceleration=SHAKE_ACCELERATION)
        armed = lambda info: bool(np.asarray(common._np(info['shake_armed'])).any())
        counted = lambda info: int(common._np(info['shake_count']).item())

        def move(leg, goal, done):
            """Follow one leg without the refinement's wait, then idle at its goal only until `done`."""
            result = planner.follow_forward_path_w_refinement(legs[leg], refine=False)
            planner.fixed_action_targets = dict(fixed)
            planner.fixed_action_targets[6] = float(goal)   # idle must not chase measured joint lag
            for _ in range(SETTLE_STEPS):
                if stopped(result) or done(result[-1]):
                    break
                result = planner.idle_steps(t=1)
            planner.fixed_action_targets = fixed
            return result

        result = move("in", rest[wrist_index], armed)
        if stopped(result):
            return result
        for k in range(int(shakes)):
            result = move("out", out[wrist_index], lambda info, k=k: counted(info) > k)
            if stopped(result):
                return result
            result = move("back", rest[wrist_index], armed)
            if stopped(result):
                return result
            info = result[-1]
            log('shake', index=k + 1, shake_count=int(common._np(info['shake_count']).item()))
            if not bool(np.asarray(common._np(info['grasp_target'])).any()):
                return result
        # Keep the final wrist target too; idle must not chase measured joint lag.
        planner.fixed_action_targets = dict(fixed)
        planner.fixed_action_targets[6] = float(rest[wrist_index])
        for _ in range(60):
            result = planner.idle_steps(t=1)
            if common.stopped_by_horizon(planner):
                return result
            info = result[-1]
            if not bool(np.asarray(common._np(info['grasp_target'])).any()):
                return result
            if bool(np.asarray(common._np(info['success'])).any()) and int(common._np(info['pour_hold']).item()) >= task.cfg.hold_steps + 2:
                break
        log('held', success=bool(common._np(result[-1]['success']).item()),
            pour_hold=int(common._np(result[-1]['pour_hold']).item()),
            shake_count=int(common._np(result[-1]['shake_count']).item()))
        return result
    finally:
        planner.fixed_action_targets = previous


def pour_with_wrist(env, planner, task, target, target_degrees=165.):
    """Collision-check the full wrist arc, then hold all other actuator targets."""
    log = lambda text, **kw: common.say(env, "season_dish_planner", text, **kw)
    robot = task.agent.robot
    joints = robot.active_joints_map
    q = robot.get_qpos()[0].cpu().numpy().astype(float)
    wrist_index = int(joints['wrist_roll_joint'].active_index[0])
    wrist = robot.links_map['wrist_roll_link'].pose[0].sp.to_transformation_matrix()
    obj = target.pose[0].sp.to_transformation_matrix()
    axis, pivot = wrist[:3, 0], wrist[:3, 3]
    object_axis = obj[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    bowl = task.bowl.pose.p[0].cpu().numpy()
    if not bool(task.agent.is_grasping(target).any()):
        return common.fail(env, "season_dish_planner", "wrist pour requires a held object")
    planner.planner.update_from_simulation()
    selected = None
    for delta in tilt_angles(axis, object_axis, target_degrees):
        goal = q.copy()
        goal[wrist_index] += delta
        end = pivot + rotation(axis, delta) @ (obj[:3, 3] - pivot)
        clearance = end[2] - bowl[2]
        if (np.linalg.norm(end[:2] - bowl[:2]) > task.cfg.pour_xy_radius
                or not task.cfg.pour_min_clearance <= clearance <= task.cfg.pour_max_clearance):
            continue
        try:
            path = planner.planner.plan_qpos_line(goal, q,
                time_step=task.control_timestep, ref_yaw=float(q[2]), qpos_step=0.02)
        except RuntimeError as error:
            path = {"status": f"path timing refused: {error}"}
        if path.get('status') != 'Success':
            log('wrist pour path refused', delta_rad=float(delta), reason=path.get('status'))
            continue
        selected = (goal, path)
        break
    if selected is None:
        return common.fail(env, "season_dish_planner", "no feasible wrist-only pour")
    goal, path = selected
    names = task.agent.controller.controllers['arm'].config.joint_names
    fixed = {i: float(q[int(joints[name].active_index[0])])
             for i, name in enumerate(names) if name != 'wrist_roll_joint'}
    fixed.update({7: -1., 10: float(q[int(joints['torso_lift_joint'].active_index[0])]),
                  11: 0., 12: 0.})
    previous = getattr(planner, 'fixed_action_targets', {})
    planner.fixed_action_targets = fixed
    try:
        log('pour', motion='wrist_roll_only', target_degrees=target_degrees,
            wrist_start=float(q[wrist_index]), wrist_goal=float(goal[wrist_index]))
        result = planner.follow_forward_path_w_refinement(path, refine=True)
        if isinstance(result, (int, np.integer)) and result == -1 or common.stopped_by_horizon(planner):
            return result
        # Keep the final wrist target too; idle must not chase measured joint lag.
        planner.fixed_action_targets = dict(fixed)
        planner.fixed_action_targets[6] = float(goal[wrist_index])
        for _ in range(60):
            result = planner.idle_steps(t=1)
            if common.stopped_by_horizon(planner):
                return result
            info = result[-1]
            if not bool(np.asarray(common._np(info['grasp_target'])).any()):
                return result
            if bool(np.asarray(common._np(info['success'])).any()) and int(common._np(info['pour_hold']).item()) >= task.cfg.hold_steps + 2:
                break
        log('held', success=bool(common._np(result[-1]['success']).item()),
            pour_hold=int(common._np(result[-1]['pour_hold']).item()))
        return result
    finally:
        planner.fixed_action_targets = previous
