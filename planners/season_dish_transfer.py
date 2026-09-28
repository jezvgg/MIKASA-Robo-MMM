"""Plan the complete fixed-base upright transfer and wrist-only pour."""
from planners.oracle.search_budget import measured
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


def pour_targets(planner, task, q, hand_object):
    """Cheap endpoint geometry and joint-budget checks; no path or IK calls."""
    p = planner.planner
    fk = kinematics(planner, task)
    wrist = fk.matrix(q, 'wrist_roll_link')
    obj = fk.matrix(q) @ hand_object
    axis, pivot = wrist[:3, 0], wrist[:3, 3]
    bowl = task.bowl.pose[0].sp.p
    index = fk.indices['wrist_roll_joint']
    # Preview needs room for the measured model-to-physics deviation. These
    # reserves tighten candidate selection; execution and checker stay at 165/155.
    parameters=getattr(task,'motion_parameters',{})
    clearance_reserve=float(parameters.get('pour_preview_clearance_reserve_m',0.))
    wrist_reserve=float(parameters.get('pour_preview_wrist_reserve_rad',0.))
    for delta in tilt_angles(axis, obj[:3, :3] @ np.asarray(task.cfg.pour_axis_body), 165.):
        end = pivot + rotation(axis, delta) @ (obj[:3, 3] - pivot)
        if (np.linalg.norm(end[:2] - bowl[:2]) > task.cfg.pour_xy_radius
                or not task.cfg.pour_min_clearance+clearance_reserve <= end[2] - bowl[2] <= task.cfg.pour_max_clearance-clearance_reserve):
            continue
        goal = q.copy(); goal[index] += delta
        reserve=goal.copy();reserve[index]+=np.sign(delta)*wrist_reserve
        if not p.accepts(np.vstack([q,goal,reserve])):
            continue
        yield goal, reserve


@measured
def preview_pour(planner, task, q, hand_object):
    p = planner.planner
    for goal, reserve in pour_targets(planner, task, q, hand_object):
        try:
            path = p.plan_qpos_line(goal, q, time_step=task.control_timestep,
                                    ref_yaw=float(q[2]), qpos_step=.02)
        except RuntimeError:
            continue
        if path.get('status') == 'Success' and path_clear(planner, q, path['position']):
            path['preview_reserve_qpos'] = reserve
            return path
    return None


def hover_goals(p, current, tcp, *, n_init=32, preferred=None, mode='dual'):
    """At most two distinct endpoints and 32 total initial guesses per geometry.

    A dual request reserves one endpoint for each initial-state family. It does
    not add another endpoint/path budget or retry after a failed continuation.
    """
    if mode not in ('current', 'preferred', 'dual'):
        raise ValueError(f'Unknown hover IK start mode: {mode}')
    n_init = min(32, max(1, int(n_init)))
    starts = [('current', current.copy(), n_init)]
    moves = list(p.move_group_joint_indices)
    if preferred is not None and mode != 'current':
        predicted = current.copy()
        predicted[moves] = preferred['approach']['position'][-1]
        predicted[:3] = current[:3]
        starts = [('preferred', predicted, n_init)]
        if mode == 'dual' and n_init >= 2:
            first = n_init // 2
            starts = [('preferred', predicted, first), ('current', current.copy(), n_init-first)]
    groups, records = [], []
    for label, initial, budget in starts:
        status, found = p.IK(p._transform_goal_to_wrt_base(mplib.Pose(tcp.p, tcp.q)),
                            p.fold_qpos(initial), [True]*3 + [False]*12,
                            n_init_qpos=budget)
        records.append(dict(initial=label, initial_guesses=budget, status=status))
        goals = []
        if status == 'Success':
            for raw in np.atleast_2d(found):
                q = unwrap_toward(raw, p.fold_qpos(current), p.joint_limits)
                q = p.unfold_qpos(q, ref_yaw=float(current[2])); q[:3] = current[:3]
                if not any(np.linalg.norm(q[3:13]-v[3:13]) < .12 for v in goals):
                    goals.append(q)
            goals.sort(key=lambda q: float(np.linalg.norm(q[3:13]-current[3:13])))
        groups.append(goals)
    chosen = []
    # Take one from each family first, then fill unused slots from those same
    # already returned endpoints. Nearly identical solutions share one slot.
    for rank in range(max([len(g) for g in groups] or [0])):
        for group in groups:
            if rank < len(group) and not any(np.linalg.norm(group[rank][3:13]-v[3:13]) < .12 for v in chosen):
                chosen.append(group[rank])
                if len(chosen) == 2:
                    return chosen, records
    return chosen, records


@measured
def plan_loaded_hover(planner, task, current, hand_object, *, n_init=32, preferred=None):
    """Preview only: neither unsuccessful candidates nor lookahead execute steps."""
    p = planner.planner
    failures=[];planner._hover_failures=failures
    fk = kinematics(planner, task)
    bowl = task.bowl.pose[0].sp.p.astype(float)
    hand = fk.matrix(current)
    obj = hand @ hand_object
    # A generous geometric rejection avoids repeated IK for the opposite counter.
    # Use the supplied planning state for hypothetical loaded dock previews too.
    base = np.r_[p.fold_qpos(current)[:2], task.agent.base_link.pose[0].sp.p[2]]
    if np.linalg.norm(bowl[:2] - base[:2]) > 1.25:
        failures.append(dict(stage='reach',reason='bowl_beyond_geometric_bound'))
        return None
    local_tcp = (task.agent.robot.links_map['gripper_link'].pose[0].sp.inv()
                 * task.agent.tcp.pose[0].sp).to_transformation_matrix()
    obj_inverse = np.linalg.inv(hand_object)
    moves = list(p.move_group_joint_indices)
    away = bowl - base; away[2] = 0.; away /= max(np.linalg.norm(away), 1e-9)
    # Compact carry points the fingers back over the base. The hover must
    # face the bowl, otherwise all eight IK requests repeat an unreachable
    # backward-facing reach. Rotate around world up; keep the condiment upright.
    tcp_now = hand @ local_tcp
    approach_now = tcp_now[:3,2]
    yaw_anchor = math.atan2(away[1],away[0]) - math.atan2(approach_now[1],approach_now[0])
    options=[(0.,0.,0.),(.06,0.,0.),(0.,.06,0.),(-.05,0.,0.),
             (0.,0.,30.),(0.,0.,-30.),(.06,0.,30.),(.06,0.,-30.)]
    if preferred is not None:
        chosen=tuple(preferred[k] for k in ('hover_extra','hover_back','hover_spin'))
        options=[chosen]+[v for v in options if v!=chosen]
    with elbow_only(planner, task):
        for extra, back, spin in options:
            desired = obj.copy()
            desired[:3, 3] = bowl - back * away + np.array([0., 0., .20 + extra])
            desired[:3, :3] = rotation([0., 0., 1.], yaw_anchor + math.radians(spin)) @ obj[:3, :3]
            tcp = sapien.Pose(desired @ obj_inverse @ local_tcp)
            mode = getattr(task, 'motion_parameters', {}).get('hover_ik_start', 'dual')
            goals, ik_records = hover_goals(p, current, tcp, n_init=n_init,
                                           preferred=preferred, mode=mode)
            failures.extend(dict(stage='ik', extra=extra, back=back, spin=spin, **v)
                            for v in ik_records if v['status'] != 'Success')
            for goal in goals[:2]:
                goal[:3] = current[:3]
                compensation = getattr(task, 'motion_parameters', {}).get('upright_compensation', 'exact')
                if compensation in ('cone12', 'cone_hover12') and next(pour_targets(planner, task, goal, hand_object), None) is None:
                    failures.append(dict(stage='endpoint_pour', reason='geometry_or_joint_span',
                                         extra=extra, back=back, spin=spin))
                    continue
                path = upright_path(planner, task, current, goal, hand_object)
                if path is None:
                    failures.append(dict(stage='upright',reasons=getattr(planner,'_upright_failures',[]),
                                         extra=extra,back=back,spin=spin))
                    continue
                reached = current.copy(); reached[moves] = path['position'][-1]
                arc = preview_pour(planner, task, reached, hand_object)
                if arc is not None and p.accepts(np.vstack([path['position'], arc['position'],
                                                           arc['preview_reserve_qpos'][moves]]), move_group=True):
                    return dict(approach=path, pour=arc, hover_extra=extra, hover_back=back, hover_spin=spin, hover_yaw_anchor=math.degrees(yaw_anchor), hover_ik_records=ik_records, hover_goal=goal.tolist())
                failures.append(dict(stage='wrist_pour',reason='pour_unavailable_or_joint_span',
                                     extra=extra,back=back,spin=spin))
    return None


@measured
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


@measured
def plan_bowl_drive(planner, task, current, hand_object, dock, face):
    """Keep upright fingers clear of the left wall and preview the final pour."""
    offsets = (0., .15, .25, .35, .45) if float(dock[0]) < 1. else (0.,)
    for offset in offsets:
        candidate = np.asarray(dock).copy(); candidate[0] += offset
        destinations = [candidate]  # Loaded compact carry goes directly to the bowl.
        arrival = preview_base_legs(planner, current, destinations, math.atan2(face[1], face[0]))
        if arrival is None:
            continue
        hover = plan_loaded_hover(planner, task, arrival, hand_object)
        if hover is not None:
            return dict(dock=candidate, offset_x_m=offset, hover=hover)
    return None
