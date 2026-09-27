"""Scene-checked continuous lower-handle approach, no intermediate stop.

Fixed progress coefficients were fitted once to the successful control free-space
approach (drawer-final-001 native seed 8000000, steps 7..104). Torso and shoulder
lift are projected to monotone cubic progress; elbow is linear. Runtime inputs
are only measured current joints, actual-handle IK endpoint, and scene geometry.
No lookup of seed-specific endpoint positions is performed.
"""
import numpy as np
from mplib.pymp.collision_detection import fcl
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

# Bezier progress control values. q(s)=q0+(q1-q0)*B(0,c1,c2,1;s).
# Pan's necessary excursion provides clearance around the counter; it is reported,
# not described as monotone. The other overshoots are bounded by URDF screening.
PROGRESS = {
    'torso_lift_joint': (.24356265113691844, 1.0),
    'shoulder_pan_joint': (.7235411583118544, 3.2484893124565657),
    'shoulder_lift_joint': (.3711874490030574, 1.0),
    'upperarm_roll_joint': (.28753258941446025, .47784108658036134),
    'elbow_flex_joint': (1/3, 2/3),
    'forearm_roll_joint': (.4352533305624711, 1.5414777951332241),
    'wrist_flex_joint': (.4142209566172384, 1.3592145964309263),
    'wrist_roll_joint': (.38528795568215, 1.1441944997639484),
}


def continuous_lower_curve(planner, task, current, goal, grasp, drawer):
    """Return one checked, retimed curve or None; does not execute physics.

    Check the subsequent moving-drawer push before executing this returned path.
    Use the existing scene/noise/roll-window policy; this helper relaxes none of
    its limits. It permits only measured low hand contact with the selected
    moving drawer wall/handle. No global allowed-collision entry is changed.
    """
    from my_scenes.same_drawer import DRAWER_FIXTURES, DRAWER_ART_SUFFIX
    p = planner.planner
    moves = list(p.move_group_joint_indices)
    idx = lambda name: int(task.agent.robot.active_joints_map[name].active_index[0])
    alpha = np.linspace(0., 1., 241)
    full = np.repeat(current[None], len(alpha), axis=0)
    for name, (c1, c2) in PROGRESS.items():
        i = idx(name)
        progress = (3*(1-alpha)**2*alpha*c1
                    + 3*(1-alpha)*alpha**2*c2 + alpha**3)
        full[:, i] = current[i] + progress*(goal[i]-current[i])

    selected = DRAWER_FIXTURES[drawer] + DRAWER_ART_SUFFIX
    model = next(p.planning_world.get_articulation(name)
                 for name in p.planning_world.get_articulation_names()
                 if selected in name)
    selected_links = set(model.get_user_link_names())
    inner = next(link for link in task._drawer_arts[drawer].get_links()
                 if link.name == 'inner_box')
    upper_edge = float(get_component_mesh(inner._objs[0]).bounds[1, 2])
    request = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    limits = np.asarray(p.joint_limits)
    failures = []
    planner._continuous_lower_curve_failures = failures

    def checked(poses):
        # RollPathPlanner.accepts checks the collection windows, not every URDF
        # bound. Explicitly check every folded joint against the unchanged URDF.
        if not p.accepts(poses):
            failures.append(dict(reason='collection_joint_window'))
            return False
        previous_distance = None
        for knot, q in enumerate(poses):
            folded = p.fold_qpos(q)
            if (np.any(folded < limits[:, 0]-1e-7)
                    or np.any(folded > limits[:, 1]+1e-7)):
                failures.append(dict(reason='URDF_joint_limit', knot=knot))
                return False
            p.pinocchio_model.compute_forward_kinematics(folded)
            tcp = np.asarray(p.pinocchio_model.get_link_pose(
                p.link_name_2_idx[p.move_group]).p)
            distance = float(np.linalg.norm(tcp-grasp.p))
            p.robot.set_qpos(folded, True)
            for collision in p.planning_world.check_collision(request):
                pair = (collision.link_name1, collision.link_name2)
                contacts = collision.res.get_contacts()
                hand = any('ds_fetch' in name and 'gripper' in name for name in pair)
                wall = any(name in selected_links and
                           ('inner_box' in name or 'handle' in name) for name in pair)
                # Contact is beneath the top edge, approached toward the actual
                # handle, with bounded geometric overlap. This rejects landing
                # on the drawer from above while allowing its low front surface.
                allowed = (hand and wall and bool(contacts)
                    and all(float(c.pos[2]) < upper_edge-.04
                            and float(c.penetration_depth) < .025 for c in contacts)
                    and (previous_distance is None
                         or distance <= previous_distance+1e-5))
                if not allowed:
                    failures.append(dict(reason='collision', knot=knot, pair=pair,
                                         tcp=tcp.tolist()))
                    return False
            previous_distance = distance
        return True

    if not checked(full):
        return None
    old_velocity = p.joint_vel_limits.copy()
    old_acceleration = p.joint_acc_limits.copy()
    torso_column = moves.index(idx('torso_lift_joint'))
    try:
        speed = old_velocity.copy()
        acceleration = old_acceleration.copy()
        speed[torso_column] = min(speed[torso_column], .06)
        acceleration[torso_column] = min(acceleration[torso_column], .12)
        p.joint_vel_limits = speed
        p.joint_acc_limits = acceleration
        times, pos, vel, acc, duration = p.TOPP(full[:, moves], task.control_timestep)
    except RuntimeError as error:
        failures.append(dict(reason='TOPP', error=str(error)))
        return None
    finally:
        p.joint_vel_limits = old_velocity
        p.joint_acc_limits = old_acceleration
    timed = np.repeat(current[None], len(pos), axis=0)
    timed[:, moves] = pos
    if not checked(timed):
        return None
    return dict(status='Success', time=times, position=pos, velocity=vel,
                acceleration=acc, duration=duration,
                curve_family='lower_handle_cubic_v1', named_progress=PROGRESS,
                contact_torso_m=float(goal[idx('torso_lift_joint')]),
                no_intermediate_pose=True)
