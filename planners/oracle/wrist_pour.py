"""Plan a one-joint pour; reject it rather than recruit other arm joints.

Two pours. The one-tip pour turns the wrist until the condiment is tilted `target_degrees` (165) and
holds. The shaking pour (SeasonDish since 2026-10-27) turns it as close to upside down as the wrist
axis allows, then swings the wrist `SHAKE_AMPLITUDE_DEG` to one side and back, to the other side and
back, once per shake, and holds upside down; only wrist_roll moves in either.
"""
import math
import numpy as np

from planners.oracle import oracle_common as common

#: Wrist roll each shake swings away from upside down and back (the owner, 2026-10-27: +-30 deg).
SHAKE_AMPLITUDE_DEG = 30.0
#: Lean kept clear of the task's shake thresholds by the planned rest and shake poses, degrees: the
#: route preview (a model of the held condiment) keeps 3, the pour itself (the measured one) 1.5, so a
#: degree of model-to-physics drift does not refuse at the bowl a pour the preview accepted.
SHAKE_LEAN_MARGIN_DEG = 3.0
SHAKE_LEAN_MARGIN_EXECUTED_DEG = 1.5


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


def inversion_angles(axis, object_axis):
    """The turns about `axis` within one turn either way that bring `object_axis` closest to straight
    down, smallest first. The tilt is cos = c0 + R cos(delta - phase) (see `tilt_angles`), lowest at
    delta = phase + pi, and turning +-a from there tilts it back by the same amount on both sides,
    so the shakes are symmetric only around this pose.

    Example:
        >>> [round(math.degrees(d)) for d in inversion_angles([1, 0, 0], [0, 0, 1])]
        [180, -180]
        >>> inversion_angles([0, 0, 1], [0, 0, 1])
        []
    """
    n, v = np.asarray(axis, dtype=float), np.asarray(object_axis, dtype=float)
    n, v = n / np.linalg.norm(n), v / np.linalg.norm(v)
    c0 = float(n[2] * (n @ v))
    c1, c2 = float(v[2] - c0), float(np.cross(n, v)[2])
    if math.hypot(c1, c2) < 1e-9:
        return []
    best = math.atan2(c2, c1) + math.pi
    values = {round(best + turn * 2 * math.pi, 12) for turn in (-2, -1, 0, 1)}
    return sorted((x for x in values if abs(x) <= 2 * math.pi + 1e-9), key=lambda x: (abs(x), x < 0))


def shake_geometry_ok(axis, object_axis, delta, cfg, amplitude_deg=SHAKE_AMPLITUDE_DEG,
                      margin_deg=SHAKE_LEAN_MARGIN_DEG):
    """Does turning to `delta` rest the condiment inside the counter's rest band, and does each
    +-`amplitude_deg` swing from there lean it past the shake threshold, with the margin?

    Example:
        >>> from types import SimpleNamespace
        >>> cfg = SimpleNamespace(shake_out_deg=20., shake_return_deg=10.)
        >>> shake_geometry_ok([1, 0, 0], [0, 0, 1], math.pi, cfg)
        True
        >>> shake_geometry_ok([1, 0, 0], [0, 0, 1], math.pi, cfg, amplitude_deg=21.)
        False
    """
    a = math.radians(amplitude_deg)
    rest = lean_after(axis, object_axis, delta)
    swing = min(lean_after(axis, object_axis, delta + a), lean_after(axis, object_axis, delta - a))
    return rest <= cfg.shake_return_deg - margin_deg and swing >= cfg.shake_out_deg + margin_deg


def _over_bowl(task, end, bowl):
    clearance = end[2] - bowl[2]
    return (np.linalg.norm(end[:2] - bowl[:2]) <= task.cfg.pour_xy_radius
            and task.cfg.pour_min_clearance <= clearance <= task.cfg.pour_max_clearance)


def _plan_shakes(planner, task, q, wrist_index, axis, pivot, obj, bowl, amplitude_deg):
    """The first upside-down turn whose rest pose and both swings stay over the bowl and plan as
    wrist-only lines: (goal, {leg name: path}) or None. Legs: in (q -> rest), out+/back+, out-/back-."""
    object_axis = obj[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    a = math.radians(amplitude_deg)
    line = lambda goal, start: planner.planner.plan_qpos_line(
        goal, start, time_step=task.control_timestep, ref_yaw=float(start[2]), qpos_step=0.02)
    for delta in inversion_angles(axis, object_axis):
        if not shake_geometry_ok(axis, object_axis, delta, task.cfg, amplitude_deg,
                                 margin_deg=SHAKE_LEAN_MARGIN_EXECUTED_DEG):
            continue
        ends = [pivot + rotation(axis, d) @ (obj[:3, 3] - pivot) for d in (delta, delta + a, delta - a)]
        if not all(_over_bowl(task, end, bowl) for end in ends):
            continue
        rest = q.copy(); rest[wrist_index] += delta
        plus = rest.copy(); plus[wrist_index] += a
        minus = rest.copy(); minus[wrist_index] -= a
        try:
            legs = {"in": line(rest, q), "out+": line(plus, rest), "back+": line(rest, plus),
                    "out-": line(minus, rest), "back-": line(rest, minus)}
        except RuntimeError:
            continue
        if all(leg.get('status') == 'Success' for leg in legs.values()):
            return rest, legs
    return None


def pour_and_shake(env, planner, task, target, shakes, amplitude_deg=SHAKE_AMPLITUDE_DEG):
    """Turn the held condiment upside down over the bowl with the wrist alone, shake it `shakes`
    times (+, back, -, back, ...), and hold it upside down until the task's hold succeeds.

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
    rest, legs = plan
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
            amplitude_deg=float(amplitude_deg), wrist_start=float(q[wrist_index]),
            wrist_goal=float(rest[wrist_index]))
        result = planner.follow_forward_path_w_refinement(legs["in"], refine=True)
        if stopped(result):
            return result
        for k in range(int(shakes)):
            side = "+" if k % 2 == 0 else "-"
            for leg in ("out" + side, "back" + side):
                result = planner.follow_forward_path_w_refinement(legs[leg], refine=True)
                if stopped(result):
                    return result
            info = result[-1]
            log('shake', index=k + 1, side=side, shake_count=int(common._np(info['shake_count']).item()))
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
