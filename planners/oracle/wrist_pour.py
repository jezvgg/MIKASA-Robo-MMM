"""Plan a one-joint pour; reject it rather than recruit other arm joints."""
import math
import numpy as np

from planners.oracle import oracle_common as common


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
