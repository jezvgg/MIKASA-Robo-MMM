"""Connected horizontal grasp, vertical lift and one compact carry posture."""
from contextlib import contextmanager
from collections import Counter
import time
from planners.season_dish_candidates import candidates as grasp_candidates, journal, failure_kind
from planners.oracle.search_budget import measured
import mplib
import numpy as np
import sapien
from transforms3d.quaternions import mat2quat
from planners.oracle.wrist_pour import rotation
from planners.oracle import oracle_common as common
from planners.oracle.straight_paths import straight_plan, execute_straight
from planners.oracle.path_clearance import path_clear
from robots.fetch.utils import attach_object, convert_object_name, unwrap_toward
from planners.oracle.upright_payload import upright_path, elbow_only, preview_roll_history, kinematics
from planners.season_dish_transfer import plan_loaded_hover, plan_bowl_drive

# Fitted once from pa40l/ManiSkill:dair@6168333 carry_pose geometry on the
# unchanged DSFetch. TCP is 25 cm in front of the base. Wrists are solved from
# the measured payload transform, not copied from this reference grasp.
CARRY_TARGETS = dict(torso_lift_joint=.20, shoulder_pan_joint=-.6557497327377413,
    shoulder_lift_joint=-.12904068645626654, upperarm_roll_joint=1.9212388766147683,
    elbow_flex_joint=1.6507435522454401, forearm_roll_joint=-2.8097833941893957,
    wrist_flex_joint=-1.9402379512175012, wrist_roll_joint=1.907999995746957)


def carry_targets(task):
    return CARRY_TARGETS


def carry_goal(task, current):
    goal = current.copy()
    for name, value in carry_targets(task).items():
        goal[int(task.agent.robot.active_joints_map[name].active_index[0])] = value
    return goal


@measured
def carry_path(planner, task, current, attachment):
    path = upright_path(planner, task, current, carry_goal(task,current), attachment, terminal="carry")
    if path is None:
        return None
    reached = current.copy()
    reached[planner.planner.move_group_joint_indices] = path['position'][-1]
    hand_tcp = (task.agent.robot.links_map['gripper_link'].pose[0].sp.inv()
                * task.agent.tcp.pose[0].sp).to_transformation_matrix()
    base = task.agent.base_link.pose[0].sp.to_transformation_matrix()
    hand = kinematics(planner,task).matrix(reached)
    tcp = (np.linalg.inv(base) @ hand @ hand_tcp)[:3,3]
    payload = (np.linalg.inv(base) @ hand @ attachment)[:3,3]
    # A second exact-upright wrist branch can put the TCP 85 cm ahead even at
    # identical shoulder/elbow angles. Reject that branch as noncompact.
    if not (.10 <= tcp[0] <= .40 and abs(tcp[1]) <= .20
            and np.linalg.norm(payload[:2]) <= .40):
        planner._upright_failures.append(dict(reason='noncompact_wrist_branch',
                                              tcp_in_base=tcp.tolist()))
        return None
    path['tcp_in_base'] = tcp.tolist()
    return path


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
def preview_payload(planner, task, obj, grasp, current=None):
    """Attach only the planning model at a hypothetical grasp; no physics writes."""
    world = planner.planner.planning_world
    robot = task.agent.robot._objs[0]
    hand = next(link for link in robot.links if link.name.endswith('gripper_link'))
    hand_tcp = task.agent.robot.links_map['gripper_link'].pose[0].sp.inv() * task.agent.tcp.pose[0].sp
    local = (hand_tcp * grasp.inv() * obj.pose[0].sp if current is None else
             sapien.Pose(np.linalg.inv(kinematics(planner, task).matrix(current))
                         @ obj.pose[0].sp.to_transformation_matrix()))
    touch = [link for link in robot.links if 'gripper' in link.name or 'wrist' in link.name]
    attach_object(world, obj._objs[0], robot, hand,
                  pose=mplib.Pose(local.p, local.q), touch_links=touch)
    try:
        yield local.to_transformation_matrix()
    finally:
        world.detach_object(convert_object_name(obj._objs[0]))
        planner.planner.update_from_simulation()


@measured
def monotone_approach(planner, task, start, goal):
    """Curve joint progress without adding poses, reversing joints or changing IK branch."""
    p=planner.planner;move=list(p.move_group_joint_indices)
    groups=[[3],[5,7],[8,9],[10,11,12]]
    def clear(full):
        return path_clear(planner, start, full[:, move])
    count=max(81,int(np.ceil(np.linalg.norm(goal-start)/.025))+1)
    alpha=np.linspace(0.,1.,count)[:,None]
    # Unfold the elbow before advancing/lowering the shoulder. The two
    # schedules are fixed for each endpoint family, not an episode search grid.
    schedules = (((2.,4.,.5,1.), (1.,2.,.5,1.)) if goal[7] < 0. else
                 ((1.,4.,.5,4.), (1.,2.,.5,1.)))
    for exponents in schedules:
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


@measured
def grasp_chain(planner, task, obj, grasp, reach, *, n_init=32):
    """Reject grasps whose lift or carry would require changing elbow branch."""
    p = planner.planner
    current = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
    folded = p.fold_qpos(current)
    reasons=Counter(); planner._chain_reasons=reasons
    candidates = grasp_candidates(planner,task,grasp,reach,reasons,n_init=n_init)
    if not candidates:
        return None
    mesh = obj.get_first_collision_mesh(to_world_frame=True)
    tallest = max(float(a.get_first_collision_mesh(to_world_frame=True).bounds[1,2])
                  for a in (task.shaker,task.condiment_bottle))
    hang = max(0.,float(grasp.p[2])-float(mesh.bounds[0,2]))
    floor = max(float(grasp.p[2])+.04,tallest+hang+.025)
    height = max(float(grasp.p[2])+.15,floor)
    heights = list(dict.fromkeys((height, max(floor, height-.04))))
    moves = list(p.move_group_joint_indices)
    other=task.condiment_bottle if obj is task.shaker else task.shaker
    for q,grasp,reach,record in candidates:
        candidate_start = time.perf_counter()
        contact=straight_plan(planner,grasp,current=q)
        reasons['contact '+str(contact is not None)]+=1
        if contact is None:
            journal(planner,record,'contact',failure_kind(planner._straight_failures,
                    getattr(planner,'_last_path_collision',None)),
                    statuses=planner._straight_failures,
                    collision=getattr(planner,'_last_path_collision',None))
            continue
        with common.keepout(planner,[obj,other],pad=[.025,.03]):
            line=p.plan_qpos_line(q,current,time_step=task.control_timestep,
                                 ref_yaw=float(current[2]),qpos_step=.02)
            line_clear = (line.get('status')=='Success'
                          and path_clear(planner,current,line['position']))
        reasons['approach '+line.get('status','?')]+=1
        if not line_clear:
            with common.keepout(planner,[obj,other],pad=[.025,.03]):
                line=monotone_approach(planner,task,current,q)
            if line is None:
                journal(planner,record,'approach',failure_kind([],getattr(planner,'_last_path_collision',None)),
                        collision=getattr(planner,'_last_path_collision',None))
                continue
        closed=q.copy();closed[moves]=contact['position'][-1]
        continuation=None
        with preview_payload(planner,task,obj,grasp,closed) as attachment, common.keepout(planner,[other],pad=.03):
            hand_tcp = task.agent.robot.links_map['gripper_link'].pose[0].sp.inv() * task.agent.tcp.pose[0].sp
            actual_tcp = sapien.Pose(kinematics(planner,task).matrix(closed)) * hand_tcp
            for z in heights:
                lift=sapien.Pose([actual_tcp.p[0],actual_tcp.p[1],z],actual_tcp.q)
                support = initial_support_contacts(planner, obj, closed)
                up=straight_plan(planner,lift,current=closed,initial_contacts=support)
                reasons['lift '+str(up is not None)]+=1
                if up is None:
                    for failure in planner._straight_failures:reasons['lift reason '+failure]+=1
                    journal(planner,record,'lift',failure_kind(planner._straight_failures,getattr(planner,'_last_path_collision',None)), height_m=z,statuses=planner._straight_failures)
                    continue
                raised=closed.copy();raised[moves]=up['position'][-1]
                # Both approach families move each joint monotonically. Their
                # endpoint extrema suffice here; build an expensive curved
                # approach only after this grasp has a usable continuation.
                prefix_knots=np.vstack([current[moves],q[moves],contact['position'],up['position']])
                prefix=np.broadcast_to(current,(len(prefix_knots),len(current))).copy()
                prefix[:,moves]=prefix_knots
                # The preview must spend the roll range used by the proposed
                # grasp/lift, not just the already executed approach to the table.
                with preview_roll_history(planner,prefix), elbow_only(planner,task):
                    # A reachable carry endpoint is not a usable continuation
                    # unless its loaded drive, hover and wrist-only pour also
                    # pass. These are model previews, never physical retries.
                    # The physical post-lift choice still checks direct first.
                    fold=carry_path(planner,task,raised,attachment)
                    route=None
                    if fold is not None:
                        carried=raised.copy();carried[moves]=fold['position'][-1]
                        fold_prefix=np.broadcast_to(raised,(len(fold['position']),len(raised))).copy()
                        fold_prefix[:,moves]=fold['position']
                        dock=np.asarray(task._bowl_dock_np)[0]
                        face=np.array([np.cos(dock[2]),np.sin(dock[2]),0.])
                        with preview_roll_history(planner,fold_prefix):
                            route=plan_bowl_drive(planner,task,carried,attachment,
                                                  np.r_[dock[:2],0.],face)
                        if route is None:
                            journal(planner,record,'loaded_continuation','pour_or_loaded_route_unavailable',
                                    height_m=z,details=getattr(planner,'_hover_failures',[]))
                    direct=None if route is not None else plan_loaded_hover(planner,task,raised,attachment)
                    if route is None:fold=None if direct is None else direct['approach']
                reasons['direct pour '+str(direct is not None)]+=1
                reasons['upright continuation '+str(fold is not None)]+=1
                if fold is None:
                    journal(planner,record,'continuation','carry_and_direct_pour_unavailable',
                            height_m=z, upright_reasons=getattr(planner,'_upright_failures',[]))
                    continue
                continuation=(up,fold,z,direct,route)
                break
        if continuation is None:
            journal(planner,record,'rejected','no_continuation',seconds=time.perf_counter()-candidate_start)
            continue
        up,fold,z,direct,route=continuation
        all_knots=np.vstack([line['position'],contact['position'],up['position'],fold['position']])
        if direct is not None:all_knots=np.vstack([all_knots,direct['pour']['position']])
        if route is not None:
            all_knots=np.vstack([all_knots,route['hover']['approach']['position'],route['hover']['pour']['position']])
        with elbow_only(planner,task):
            if not p.accepts(all_knots,move_group=True):
                journal(planner,record,'complete_path','joint_limit',seconds=time.perf_counter()-candidate_start)
                continue
        journal(planner,record,'accepted',seconds=time.perf_counter()-candidate_start,
                direct_pour=direct is not None,complete_wrist_pour=True)
        return dict(approach=line,contact=contact,lift_z=z,grasp_pose=grasp,
                    grasp_qpos=closed,carry=fold,direct_pour=direct is not None,
                    progress_powers=line.get('progress_powers'),
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
               grasp_qpos=chain['grasp_qpos'].tolist(),direct_pour=chain['direct_pour'],
               progress_powers=chain['progress_powers'])
    result=planner.follow_forward_path_w_refinement(chain['approach'],refine=True)
    if common.stopped_by_horizon(planner):return result,False
    planner.planner.update_from_simulation()
    path=straight_plan(planner,chain['grasp_pose'])
    if path is None:return -1,False
    result=execute_straight(planner,path)
    if common.stopped_by_horizon(planner):return result,False
    result=planner.close_gripper(t=8)
    planner._season_lift_z=chain['lift_z']
    return result,bool(task.agent.is_grasping(obj).any())
