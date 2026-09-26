"""Choose a hover whose complete wrist-only pour fits the recorded roll range."""
import mplib
import numpy as np

from planners.oracle.wrist_pour import rotation, tilt_angles


def prepare_for_wrist_pour(env, planner, task, target, hover):
    """Check both the hover approach and the attached object's entire pour arc."""
    p = planner.planner
    q = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
    folded = p.fold_qpos(q)
    tcp_now = task.agent.tcp.pose[0].sp
    wrist_now = task.agent.robot.links_map['wrist_roll_link'].pose[0].sp
    # Wrist/TCP and object/TCP transforms come from the physical held state.
    wrist_goal = hover * (tcp_now.inv() * wrist_now)
    obj_goal = hover * (tcp_now.inv() * target.pose[0].sp)
    wrist_matrix = wrist_goal.to_transformation_matrix()
    object_matrix = obj_goal.to_transformation_matrix()
    axis, pivot = wrist_matrix[:3, 0], wrist_matrix[:3, 3]
    object_axis = object_matrix[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    wrist_index = int(task.agent.robot.active_joints_map['wrist_roll_joint'].active_index[0])
    bowl = task.bowl.pose[0].sp.p
    deltas = []
    for delta in tilt_angles(axis, object_axis):
        end = pivot + rotation(axis, delta) @ (object_matrix[:3, 3] - pivot)
        if (np.linalg.norm(end[:2] - bowl[:2]) <= task.cfg.pour_xy_radius
                and task.cfg.pour_min_clearance <= end[2] - bowl[2] <= task.cfg.pour_max_clearance):
            deltas.append(delta)
    if not deltas:
        return -1
    target_pose = mplib.Pose(hover.p, hover.q)
    status, goals = p.IK(
        p._transform_goal_to_wrt_base(target_pose), folded,
        [True, True, True] + [False] * 12, n_init_qpos=160)
    if status != 'Success':
        return -1

    def precheck(goal):
        line = p.plan_qpos_line(goal, q, time_step=task.control_timestep,
                                ref_yaw=float(q[2]))
        if line.get('status') != 'Success':
            return False
        for delta in deltas:
            end = goal.copy()
            end[wrist_index] += delta
            if not p.accepts(end):
                continue
            try:
                arc = p.plan_qpos_line(end, goal, time_step=task.control_timestep,
                                      ref_yaw=float(q[2]), qpos_step=.02)
            except RuntimeError:
                continue
            if (arc.get('status') == 'Success'
                    and p.accepts(np.vstack([line['position'], arc['position']]),
                                  move_group=True)):
                return True
        return False

    result = planner._line_to_ik_goals(
        target_pose, goals, q, folded, float(q[2]), 1, None, None,
        tag='wrist_pour_hover', precheck=precheck)
    return -1 if result is None else result
