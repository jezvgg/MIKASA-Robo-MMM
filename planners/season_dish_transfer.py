"""Plan the complete fixed-base upright transfer and wrist-only pour."""
import math
import mplib
import numpy as np
import sapien

from planners.oracle.upright_payload import kinematics, upright_path, elbow_only
from planners.oracle.path_clearance import path_clear
from planners.oracle.wrist_pour import rotation, tilt_angles
from robots.fetch.utils import unwrap_toward


def held_transform(task, target):
    return (task.agent.robot.links_map['gripper_link'].pose[0].sp.inv()
            * target.pose[0].sp).to_transformation_matrix().astype(float)


def preview_pour(planner, task, q, hand_object):
    p = planner.planner
    fk = kinematics(planner, task)
    wrist = fk.matrix(q, 'wrist_roll_link')
    obj = fk.matrix(q) @ hand_object
    axis, pivot = wrist[:3, 0], wrist[:3, 3]
    bowl = task.bowl.pose[0].sp.p
    index = fk.indices['wrist_roll_joint']
    for delta in tilt_angles(axis, obj[:3, :3] @ np.asarray(task.cfg.pour_axis_body), 165.):
        end = pivot + rotation(axis, delta) @ (obj[:3, 3] - pivot)
        if (np.linalg.norm(end[:2] - bowl[:2]) > task.cfg.pour_xy_radius
                or not task.cfg.pour_min_clearance <= end[2] - bowl[2] <= task.cfg.pour_max_clearance):
            continue
        goal = q.copy(); goal[index] += delta
        try:
            path = p.plan_qpos_line(goal, q, time_step=task.control_timestep,
                                    ref_yaw=float(q[2]), qpos_step=.02)
        except RuntimeError:
            continue
        if path.get('status') == 'Success' and path_clear(planner, q, path['position']):
            return path
    return None


def plan_loaded_hover(planner, task, current, hand_object, *, n_init=80):
    """Preview only: neither unsuccessful candidates nor lookahead execute steps."""
    p = planner.planner
    fk = kinematics(planner, task)
    bowl = task.bowl.pose[0].sp.p.astype(float)
    hand = fk.matrix(current)
    obj = hand @ hand_object
    # A generous geometric rejection avoids repeated IK for the opposite counter.
    # Use the supplied planning state for hypothetical loaded dock previews too.
    base = np.r_[p.fold_qpos(current)[:2], task.agent.base_link.pose[0].sp.p[2]]
    if np.linalg.norm(bowl[:2] - base[:2]) > 1.25:
        return None
    local_tcp = (task.agent.robot.links_map['gripper_link'].pose[0].sp.inv()
                 * task.agent.tcp.pose[0].sp).to_transformation_matrix()
    obj_inverse = np.linalg.inv(hand_object)
    moves = list(p.move_group_joint_indices)
    away = bowl - base; away[2] = 0.; away /= max(np.linalg.norm(away), 1e-9)
    with elbow_only(planner, task):
        for extra, back, spin in ((0., 0., 0.), (.06, 0., 0.), (0., .06, 0.),
                                  (-.05, 0., 0.), (0., 0., 30.), (0., 0., -30.),
                                  (.06, 0., 30.), (.06, 0., -30.)):
            desired = obj.copy()
            desired[:3, 3] = bowl - back * away + np.array([0., 0., .20 + extra])
            desired[:3, :3] = rotation([0., 0., 1.], math.radians(spin)) @ obj[:3, :3]
            tcp = sapien.Pose(desired @ obj_inverse @ local_tcp)
            status, goals = p.IK(p._transform_goal_to_wrt_base(mplib.Pose(tcp.p, tcp.q)),
                                 p.fold_qpos(current), [True] * 3 + [False] * 12,
                                 n_init_qpos=n_init)
            if status != 'Success':
                continue
            goals = [unwrap_toward(g, p.fold_qpos(current), p.joint_limits) for g in np.atleast_2d(goals)]
            goals = [p.unfold_qpos(g, ref_yaw=float(current[2])) for g in goals]
            goals.sort(key=lambda g: float(np.linalg.norm(g[3:13] - current[3:13])))
            for goal in goals:
                goal[:3] = current[:3]
                path = upright_path(planner, task, current, goal, hand_object)
                if path is None:
                    continue
                reached = current.copy(); reached[moves] = path['position'][-1]
                arc = preview_pour(planner, task, reached, hand_object)
                if arc is not None and p.accepts(np.vstack([path['position'], arc['position']]), move_group=True):
                    return dict(approach=path, pour=arc, hover_extra=extra, hover_back=back, hover_spin=spin)
    return None


def preview_base_legs(planner, current, destinations, final_yaw):
    """Sweep the complete held robot through turn/drive/turn legs, model only."""
    p = planner.planner
    poses = [p.fold_qpos(current)]

    def turn(yaw):
        start = poses[-1]
        delta = (yaw - start[2] + math.pi) % (2 * math.pi) - math.pi
        for alpha in np.linspace(0., 1., max(2, int(abs(delta) / .02) + 1))[1:]:
            q = start.copy(); q[2] += alpha * delta; poses.append(q)

    for destination in destinations:
        xy = np.asarray(destination)[:2]
        delta = xy - poses[-1][:2]
        if np.linalg.norm(delta) > 1e-6:
            turn(math.atan2(delta[1], delta[0]))
            start = poses[-1]
            for alpha in np.linspace(0., 1., max(2, int(np.linalg.norm(delta) / .01) + 1))[1:]:
                q = start.copy(); q[:2] = start[:2] + alpha * delta; poses.append(q)
    turn(final_yaw)
    full = np.asarray([p.unfold_qpos(q, ref_yaw=float(current[2])) for q in poses])
    if not path_clear(planner, current, full[:, p.move_group_joint_indices]):
        return None
    return full[-1]


def plan_bowl_drive(planner, task, current, hand_object, dock, face):
    """Keep upright fingers clear of the left wall and preview the final pour."""
    offsets = (0., .15, .25, .35, .45) if float(dock[0]) < 1. else (0.,)
    for offset in offsets:
        candidate = np.asarray(dock).copy(); candidate[0] += offset
        destinations = ([candidate + np.array([.40, -.60, 0.]), candidate]
                        if candidate[0] < 1. else [candidate])
        arrival = preview_base_legs(planner, current, destinations, math.atan2(face[1], face[0]))
        if arrival is None:
            continue
        hover = plan_loaded_hover(planner, task, arrival, hand_object)
        if hover is not None:
            return dict(dock=candidate, offset_x_m=offset)
    return None
