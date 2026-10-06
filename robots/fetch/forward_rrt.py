"""Skip-only RRT: bounded scalar advance plus arm, not a holonomic base.

World pose is baked into an in-memory URDF's fixed root. The real robot,
original planning world, and source URDF are never modified.
"""
from xml.etree import ElementTree as ET

import mplib
import numpy as np
import sapien.physx as physx
from mplib.collision_detection.fcl import CollisionObject, FCLObject
from mplib.sapien_utils.conversion import convert_object_name
from mplib.sapien_utils.srdf_exporter import export_srdf
from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf
from transforms3d.euler import quat2euler

from robots.fetch.utils import SapienPlannerV2, SapienPlanningWorldV2, unwrap_toward, pose_error

ARM = [5, 7, 8, 9, 10, 11, 12]


def _clone(obj):
    return FCLObject(obj.name, obj.pose,
                     [CollisionObject(s.get_collision_geometry()) for s in obj.shapes],
                     list(obj.shape_poses))


def _advance_limit(q0, direction, limits, maximum):
    limit = min(float(maximum), 0.25)
    for i, d in enumerate(direction):
        if abs(d) > 1e-12:
            stop = limits[i, 1] if d > 0 else limits[i, 0]
            limit = min(limit, (stop - q0[i]) / d)
    return max(0.0, limit)


def _decode(rows, q0, direction, indices, rates=False):
    out = np.asarray(rows).copy()
    s = out[:, indices.index(0)].copy()
    for axis in (0, 1):
        out[:, indices.index(axis)] = s * direction[axis]
        if not rates:
            out[:, indices.index(axis)] += q0[axis]
    out[:, indices.index(2)] = 0.0 if rates else q0[2]
    return out


def plan_forward_rrt(solver, target_pose, max_advance, speed=0.08):
    """One RRT over advance+arm; return a decoded simulator-space trajectory."""
    original = solver.planner
    original.update_from_simulation()
    agent = solver.base_env.agent
    sim = solver.robot._objs[0]
    q0 = solver.robot.get_qpos()[0].cpu().numpy().astype(float)
    heading = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
    root_rotation = solver.robot.pose.sp.to_transformation_matrix()[:3, :3]
    direction = (root_rotation.T @ heading)[:2]
    maximum = _advance_limit(q0, direction, original.joint_limits, max_advance)
    if maximum < 1e-4:
        return {"status": "forward RRT: no safe advance available"}

    xml = ET.fromstring(export_kinematic_chain_urdf(sim))
    root_joint = xml.find("joint[@name='__root_joint__']")
    if root_joint is None:
        return {"status": "forward RRT: exported fixed root missing"}
    anchor = ET.SubElement(root_joint, 'origin')
    base = agent.base_link.pose.sp
    anchor.set("xyz", " ".join(str(float(x)) for x in base.p))
    anchor.set("rpy", " ".join(str(float(x)) for x in quat2euler(base.q)))
    names = original.user_joint_names
    # Keep exporter dummy-joint rotations: together they define the root axes.
    limit = xml.find(f"joint[@name='{names[0]}']/limit")
    limit.set('lower', '0'); limit.set('upper', str(maximum))
    model = mplib.ArticulatedModel.create_from_urdf_string(
        ET.tostring(xml, encoding='unicode'), export_srdf(sim),
        [_clone(o) for o in original.robot.get_fcl_model().get_collision_objects()],
        name=original.robot.name, link_names=original.user_link_names,
        joint_names=names,
    )
    model.set_base_pose(mplib.Pose())
    start = q0.copy(); start[:3] = 0
    model.set_qpos(start, True)
    world = mplib.PlanningWorld([model])
    source_world = original.planning_world
    for name in source_world.get_articulation_names():
        if name != original.robot.name:
            world.add_articulation(source_world.get_articulation(name))
    for name in source_world.get_object_names():
        obj = source_world.get_object(name)
        if obj is not None:
            world.add_object(_clone(obj))
    source_acm = source_world.get_allowed_collision_matrix()
    acm = world.get_allowed_collision_matrix()
    entries = source_acm.get_all_entry_names()
    for i, a in enumerate(entries):
        default = source_acm.get_default_entry(a)
        if default is not None:
            acm.set_default_entry(a, int(default) == 1)
        for b in entries[i:]:
            entry = source_acm.get_entry(a, b)
            if entry is not None:
                acm.set_entry(a, b, int(entry) == 1)
    component = solver.base_env.cup._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
    cup_name = convert_object_name(component.entity)
    if not world.has_object(cup_name):
        obj = SapienPlanningWorldV2.convert_physx_component(component)
        if obj is None:
            return {"status": "forward RRT: held cup missing"}
        world.add_object(obj)
    touch = [n for n in original.user_link_names if 'gripper' in n or 'finger' in n]
    for name in original.user_link_names:
        acm.set_entry(name, cup_name, name in touch)
    world.attach_object(cup_name, model.name, original.link_name_2_idx[original.move_group], touch_links=touch)
    velocity = np.asarray(original.joint_vel_limits).copy(); velocity[0] = speed
    acceleration = np.asarray(original.joint_acc_limits).copy(); acceleration[0] = min(acceleration[0], 0.2)
    planner = SapienPlannerV2(world, original.move_group, joint_vel_limits=velocity, joint_acc_limits=acceleration)
    mask = [True] * len(q0)
    for index in [0] + ARM: mask[index] = False
    planner.pinocchio_model.compute_forward_kinematics(start)
    initial = planner.pinocchio_model.get_link_pose(planner.move_group_link_id)
    measured = agent.tcp.pose.sp
    if max(pose_error(initial.p, initial.q, measured.p, measured.q)) > 1e-5:
        return {"status": "forward RRT: surrogate FK differs from simulator"}
    status, goals = planner.IK(target_pose, start, mask=mask, n_init_qpos=100)
    if status != 'Success': return {"status": status}
    goals = [unwrap_toward(g, start, planner.joint_limits) for g in np.atleast_2d(goals)]
    # ponytail: one nearest IK goal; more RRT goals only if measured refusals need it.
    nearest = min(goals, key=lambda goal: np.linalg.norm((goal - start)[ARM]))
    result = planner.plan_qpos([nearest], start, time_step=solver.base_env.control_timestep,
                              fixed_joint_indices=[1, 2, 3], planning_time=2.0,
                              rrt_range=0.1, simplify=True)
    if result.get('status') != 'Success': return result
    indices = list(planner.move_group_joint_indices)
    positions = result['position']
    dense = [positions[0]]
    for a, b in zip(positions[:-1], positions[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.01)))
        dense.extend(np.linspace(a, b, count + 1)[1:])
    frozen = [1, 2, 3, 4, 6, 13, 14]
    for sample_index, row in enumerate(dense):
        full = start.copy(); full[indices] = row
        if np.any(full < planner.joint_limits[:, 0] - 1e-6) or np.any(full > planner.joint_limits[:, 1] + 1e-6):
            return {"status": "forward RRT: timed path exceeds limits"}
        if np.max(np.abs(full[frozen] - start[frozen])) > 1e-6:
            return {"status": "forward RRT: frozen joints moved"}
        model.set_qpos(full, True)
        if world.is_state_colliding():
            pairs = [f'{c.link_name1}<->{c.link_name2}' for c in world.check_collision()]
            return {"status": "forward RRT: timed path collides", "contacts": pairs,
                    "sample": sample_index, "knots": len(positions), "qpos": full.tolist()}
    planner.pinocchio_model.compute_forward_kinematics(full)
    endpoint = planner.pinocchio_model.get_link_pose(planner.move_group_link_id)
    error = pose_error(target_pose.p, target_pose.q, endpoint.p, endpoint.q)
    if error[0] > 0.02 or error[1] > np.deg2rad(5):
        return {"status": "forward RRT: endpoint misses target"}
    for key in ('position', 'velocity', 'acceleration'):
        result[key] = _decode(result[key], q0, direction, indices, rates=key != 'position')
    decoded = np.tile(q0, (len(positions), 1)); decoded[:, indices] = result['position']
    if np.any(decoded < original.joint_limits[:, 0] - 1e-6) or np.any(decoded > original.joint_limits[:, 1] + 1e-6):
        return {"status": "forward RRT: decoded path exceeds original limits"}
    result.update(forward_advance=float(positions[-1, indices.index(0)]), forward_limit=maximum, goal_error=error)
    return result
