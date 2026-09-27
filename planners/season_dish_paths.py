"""Connected horizontal grasp, vertical lift and one compact carry posture."""
from contextlib import contextmanager
from collections import Counter
from itertools import product
import mplib
import numpy as np
import sapien
from planners.oracle import oracle_common as common
from planners.oracle.straight_paths import straight_plan, execute_straight
from planners.oracle.path_clearance import path_clear
from robots.fetch.utils import attach_object, convert_object_name, unwrap_toward
from planners.oracle.upright_payload import upright_path, elbow_only, preview_roll_history
from planners.season_dish_transfer import plan_loaded_hover

CARRY_TARGETS = dict(torso_lift_joint=.38, shoulder_pan_joint=-.37,
    shoulder_lift_joint=-.8, upperarm_roll_joint=1.7, elbow_flex_joint=2.1,
    forearm_roll_joint=-.6, wrist_flex_joint=.5, wrist_roll_joint=-.4)


def carry_goal(task, current):
    goal = current.copy()
    for name, value in CARRY_TARGETS.items():
        goal[int(task.agent.robot.active_joints_map[name].active_index[0])] = value
    return goal


def initial_support_contacts(planner, obj, current):
    """Exact payload/counter pairs at lift start; never robot or upper fixtures."""
    p = planner.planner
    p.robot.set_qpos(p.fold_qpos(current), True)
    p.planning_world.set_qpos_all(p.fold_qpos(current)[p.move_group_joint_indices])
    name = convert_object_name(obj._objs[0])
    pairs = []
    for collision in p.planning_world.check_collision():
        pair = (collision.link_name1, collision.link_name2)
        if name in pair and any("counter_main" in link for link in pair if link != name):
            pairs.append(pair)
    return pairs


@contextmanager
def preview_payload(planner, task, obj, grasp):
    """Attach only the planning model at a hypothetical grasp; no physics writes."""
    world = planner.planner.planning_world
    robot = task.agent.robot._objs[0]
    hand = next(link for link in robot.links if link.name.endswith('gripper_link'))
    hand_tcp = task.agent.robot.links_map['gripper_link'].pose[0].sp.inv() * task.agent.tcp.pose[0].sp
    local = hand_tcp * grasp.inv() * obj.pose[0].sp
    touch = [link for link in robot.links if 'gripper' in link.name or 'wrist' in link.name]
    attach_object(world, obj._objs[0], robot, hand,
                  pose=mplib.Pose(local.p, local.q), touch_links=touch)
    try:
        yield
    finally:
        world.detach_object(convert_object_name(obj._objs[0]))
        planner.planner.update_from_simulation()


def monotone_approach(planner, task, start, goal):
    """Curve joint progress without adding poses, reversing joints or changing IK branch."""
    p=planner.planner;move=list(p.move_group_joint_indices)
    groups=[[3],[5,7],[8,9],[10,11,12]]
    def clear(full):
        return path_clear(planner, start, full[:, move])
    count=max(81,int(np.ceil(np.linalg.norm(goal-start)/.025))+1)
    alpha=np.linspace(0.,1.,count)[:,None]
    for exponents in product((1.,2.,.5,4.),repeat=4):
        if all(e==1 for e in exponents):continue
        powers=np.ones(len(start))
        for idx,e in zip(groups,exponents):powers[idx]=e
        full=start[None]+(goal-start)[None]*alpha**powers[None]
        if not p.accepts(full) or not clear(full):continue
        try:times,pos,vel,acc,duration=p.TOPP(full[:,move],task.control_timestep)
        except RuntimeError:continue
        timed=np.broadcast_to(start,(len(pos),len(start))).copy();timed[:,move]=pos
        if not p.accepts(timed) or not clear(timed):continue
        return dict(status='Success',time=times,position=pos,velocity=vel,
                    acceleration=acc,duration=duration,progress_powers=exponents)
    return None


def grasp_chain(planner, task, obj, grasp, reach, *, n_init=160):
    """Reject grasps whose lift or carry would require changing elbow branch."""
    p = planner.planner
    current = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
    folded = p.fold_qpos(current)
    reasons=Counter(); planner._chain_reasons=reasons
    goals=[]
    for torso in [None,.20,.25,.30]:
        initial=folded.copy()
        if torso is not None:initial[3]=torso
        status, found = p.IK(p._transform_goal_to_wrt_base(mplib.Pose(reach.p, reach.q)),
                            initial, [True]*3+[torso is not None]+[False]*11,
                            n_init_qpos=n_init)
        reasons['ik '+str(torso)+' '+status]+=1
        if status=='Success':goals.extend(np.atleast_2d(found))
    if not goals:return None
    reasons['candidates']=len(goals)
    mesh = obj.get_first_collision_mesh(to_world_frame=True)
    tallest = max(float(a.get_first_collision_mesh(to_world_frame=True).bounds[1,2])
                  for a in (task.shaker,task.condiment_bottle))
    hang = max(0.,float(grasp.p[2])-float(mesh.bounds[0,2]))
    floor = max(float(grasp.p[2])+.04,tallest+hang+.025)
    height = max(float(grasp.p[2])+.15,floor)
    heights = [height]+[height-d for d in [.02,.04,.06] if height-d>=floor]
    moves = list(p.move_group_joint_indices)
    candidates=[]
    for q in np.atleast_2d(goals):
        q=unwrap_toward(q,folded,p.joint_limits)
        for idx in p._root_cols(): q[idx]=current[idx]
        candidates.append(q)
    candidates.sort(key=lambda q: float(np.abs(q[3:13]-current[3:13]).sum()))
    other=task.condiment_bottle if obj is task.shaker else task.shaker
    for q in candidates:
        with common.keepout(planner,[obj,other],pad=[.025,.03]):
            line=p.plan_qpos_line(q,current,time_step=task.control_timestep,
                                 ref_yaw=float(current[2]),qpos_step=.02)
        reasons['approach '+line.get('status','?')]+=1
        line_clear = (line.get('status')=='Success'
                      and path_clear(planner,current,line['position']))
        contact=straight_plan(planner,grasp,current=q)
        reasons['contact '+str(contact is not None)]+=1
        if contact is None:continue
        if not line_clear:
            with common.keepout(planner,[obj,other],pad=[.025,.03]):
                line=monotone_approach(planner,task,current,q)
            reasons['curved approach '+str(line is not None)]+=1
            if line is None:continue
        closed=q.copy();closed[moves]=contact['position'][-1]
        continuation=None
        with preview_payload(planner,task,obj,grasp), common.keepout(planner,[other],pad=.03):
            hand_tcp = task.agent.robot.links_map['gripper_link'].pose[0].sp.inv() * task.agent.tcp.pose[0].sp
            attachment = (hand_tcp * grasp.inv() * obj.pose[0].sp).to_transformation_matrix()
            for z in heights:
                lift=sapien.Pose([grasp.p[0],grasp.p[1],z],grasp.q)
                support = initial_support_contacts(planner, obj, closed)
                up=straight_plan(planner,lift,current=closed,initial_contacts=support)
                reasons['lift '+str(up is not None)]+=1
                if up is None:
                    for failure in planner._straight_failures:reasons['lift reason '+failure]+=1
                    continue
                raised=closed.copy();raised[moves]=up['position'][-1]
                prefix_knots=np.vstack([line['position'],contact['position'],up['position']])
                prefix=np.broadcast_to(current,(len(prefix_knots),len(current))).copy()
                prefix[:,moves]=prefix_knots
                # The preview must spend the roll range used by the proposed
                # grasp/lift, not just the already executed approach to the table.
                with preview_roll_history(planner,prefix), elbow_only(planner,task):
                    # Either continuation makes this grasp usable. The physical
                    # post-lift decision still always checks direct pouring first.
                    fold=upright_path(planner,task,raised,carry_goal(task,raised),attachment)
                    direct=None if fold is not None else plan_loaded_hover(planner,task,raised,attachment)
                    if direct is not None:fold=direct['approach']
                reasons['direct pour '+str(direct is not None)]+=1
                reasons['upright continuation '+str(fold is not None)]+=1
                if fold is None:continue
                continuation=(up,fold,z,direct)
                break
        if continuation is None:continue
        up,fold,z,direct=continuation
        all_knots=np.vstack([line['position'],contact['position'],up['position'],fold['position']])
        if direct is not None:all_knots=np.vstack([all_knots,direct['pour']['position']])
        with elbow_only(planner,task):
            if not p.accepts(all_knots,move_group=True):continue
        return dict(approach=line,contact=contact,lift_z=z,
                    grasp_qpos=closed,carry=fold,direct_pour=direct is not None,
                    total_joint_travel=float(np.abs(np.diff(all_knots,axis=0)).sum()))
    return None


def grasp_with_continuation(env,planner,task,obj,grasp,reach):
    planner.planner.update_from_simulation()
    chain=grasp_chain(planner,task,obj,grasp,reach)
    if chain is None:
        common.say(env,'season_dish_planner','connected grasp/lift/carry refused')
        return -1,False
    common.say(env,'season_dish_planner','connected grasp/lift/carry selected',
               lift_z=chain['lift_z'],joint_travel=chain['total_joint_travel'],
               grasp_qpos=chain['grasp_qpos'].tolist(),direct_pour=chain['direct_pour'])
    result=planner.follow_forward_path_w_refinement(chain['approach'],refine=True)
    if common.stopped_by_horizon(planner):return result,False
    planner.planner.update_from_simulation()
    path=straight_plan(planner,grasp)
    if path is None:return -1,False
    result=execute_straight(planner,path)
    if common.stopped_by_horizon(planner):return result,False
    result=planner.close_gripper(t=8)
    planner._season_lift_z=chain['lift_z']
    return result,bool(task.agent.is_grasping(obj).any())
