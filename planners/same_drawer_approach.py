"""Scene-specific continuous bottom approach and moving-drawer push previews."""
from contextlib import contextmanager
import mplib
import numpy as np
from robots.fetch.utils import unwrap_toward
from planners.oracle.straight_paths import straight_plan


@contextmanager
def hand_contact(planner, task, drawer, *, allowed=True):
    """Permit intentional hand contact only; keep the forearm and all fixtures."""
    from mplib.pymp.collision_detection import AllowedCollision
    acm=planner.planner.planning_world.get_allowed_collision_matrix()
    from my_scenes.same_drawer import DRAWER_FIXTURES, DRAWER_ART_SUFFIX
    world=planner.planner.planning_world
    hand=[n for n in planner.planner.robot.get_user_link_names() if 'gripper' in n]
    selected=DRAWER_FIXTURES[drawer]+DRAWER_ART_SUFFIX
    links=[n for name in world.get_articulation_names() if selected in name
           for n in world.get_articulation(name).get_user_link_names()]
    previous=[]
    for a in hand:
        for b in links:
            previous.append((a,b,acm.get_entry(a,b)))
            acm.set_entry(a,b,allowed)
    try:yield
    finally:
        for a,b,value in previous:
            if value is None:acm.remove_entry(a,b)
            else:acm.set_entry(a,b,value==AllowedCollision.ALWAYS)


def moving_drawer_push(planner, task, current, goal, drawer):
    """Preview the slider at its proposed position, without changing physics."""
    from planners.oracle import oracle_common as common
    from planners.oracle.path_clearance import dense_samples
    from my_scenes.same_drawer import DRAWER_FIXTURES, DRAWER_ART_SUFFIX
    p = planner.planner
    selected = DRAWER_FIXTURES[drawer] + DRAWER_ART_SUFFIX
    with common.contact_stroke(planner, [selected]):
        path = straight_plan(planner, goal, current=current)
    if path is None:
        return None
    world = p.planning_world
    model = next(world.get_articulation(n) for n in world.get_articulation_names() if selected in n)
    original = model.get_qpos().copy()
    pin = p.pinocchio_model
    ee = p.link_name_2_idx[p.move_group]
    pin.compute_forward_kinematics(p.fold_qpos(current))
    y0 = float(pin.get_link_pose(ee).p[1])
    try:
        # The drawer moves with the pushing hand, not as a frozen open obstacle.
        with hand_contact(planner, task, drawer):
            for knot in dense_samples(path['position']):
                q = current.copy(); q[p.move_group_joint_indices] = knot
                folded = p.fold_qpos(q)
                pin.compute_forward_kinematics(folded)
                travel = max(0.,float(pin.get_link_pose(ee).p[1])-y0)
                drawer_q = original.copy(); drawer_q[0] = min(0.,float(original[0])+travel)
                model.set_qpos(drawer_q, True)
                p.robot.set_qpos(folded, True)
                if world.is_state_colliding():
                    planner._straight_failures.append('moving drawer collision: '+str(
                        [(c.link_name1,c.link_name2) for c in world.check_collision()]))
                    return None
    finally:
        model.set_qpos(original, True)
    return path


def direct_bottom_contact(planner, task, grasp, push, *, drawer):
    """One continuous arm/torso curve to contact, with no standoff fallback."""
    from planners.same_drawer_curve import continuous_lower_curve
    p = planner.planner
    current = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
    torso = int(task.agent.robot.active_joints_map['torso_lift_joint'].active_index[0])
    reasons = []; planner._bottom_contact_diagnostics = reasons
    planner._bottom_path_refusals = []
    with hand_contact(planner, task, drawer):
        for height in (.01, .10, None):
            initial = p.fold_qpos(current).copy()
            if height is not None:
                initial[torso] = height
            status, found = p.IK(
                p._transform_goal_to_wrt_base(mplib.Pose(grasp.p, grasp.q)), initial,
                [True]*3 + [height is not None] + [False]*11, n_init_qpos=40)
            reasons.append(dict(stage='contact_ik', torso=height, status=status))
            if status != 'Success':
                continue
            goals = [p.unfold_qpos(unwrap_toward(q, initial, p.joint_limits),
                                  ref_yaw=float(current[2])) for q in np.atleast_2d(found)]
            goals.sort(key=lambda q: float(np.linalg.norm(q[3:13] - current[3:13])))
            for goal in goals[:8]:
                goal[:3] = current[:3]
                reasons.append(dict(stage='endpoint', qpos=goal.tolist()))
                stroke = moving_drawer_push(planner, task, goal, push, drawer)
                if stroke is None:
                    reasons.append(dict(stage='push', torso=height,
                                        reason=list(planner._straight_failures)))
                    continue
                # Every contact is inspected geometrically by the curve helper;
                # disable the broad endpoint-only hand allowance for its sweep.
                with hand_contact(planner, task, drawer, allowed=False):
                    path = continuous_lower_curve(planner, task, current, goal, grasp, drawer)
                if path is None:
                    planner._bottom_path_refusals.extend(planner._continuous_lower_curve_failures)
                    continue
                if not p.accepts(np.vstack([path['position'], stroke['position']]), move_group=True):
                    reasons.append(dict(stage='complete_path',reason='joint_window'))
                    continue
                return path
    return None
