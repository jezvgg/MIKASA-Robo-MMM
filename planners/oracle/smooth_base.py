"""A checked, forward-only rounded route through a navigation waypoint."""
import math
import numpy as np

from planners.oracle.path_clearance import path_clear
from planners.oracle import oracle_common as common


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rounded_route(start, corner, finish, cut):
    """Quintic corner with zero curvature at both straight-line joins."""
    start, corner, finish = (np.asarray(x, float)[:2] for x in (start, corner, finish))
    incoming, outgoing = corner - start, finish - corner
    lengths = np.array([np.linalg.norm(incoming), np.linalg.norm(outgoing)])
    if min(lengths) < .05:
        return None
    u, v = incoming / lengths[0], outgoing / lengths[1]
    distance = min(float(cut), .45 * min(lengths))
    a, b = corner - distance * u, corner + distance * v
    control = np.array([a, a + .25 * distance * u, a + .5 * distance * u,
                        b - .5 * distance * v, b - .25 * distance * v, b])
    t = np.linspace(0, 1, max(81, int(distance / .002)))
    curve = sum(math.comb(5, i) * (1-t[:, None])**(5-i) * t[:, None]**i * control[i] for i in range(6))
    line = lambda x,y: np.linspace(x,y,max(2,int(np.ceil(np.linalg.norm(y-x)/.005))+1))
    points = np.vstack([line(start,a)[:-1], curve[:-1], line(b,finish)])
    ds = np.linalg.norm(np.diff(points,axis=0),axis=1)
    points = points[np.r_[True,ds>1e-7]]
    s = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    tangent = np.gradient(points,s,axis=0)
    yaw = np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
    curvature = np.gradient(yaw,s)
    return dict(points=points, distance=s, yaw=yaw, curvature=curvature, cut=distance)


def route_qpos(planner, current, points, yaw):
    p = planner.planner
    full = np.broadcast_to(p.fold_qpos(current),(len(points),len(current))).copy()
    full[:,:2] = points
    full[:,2] = yaw
    return np.asarray([p.unfold_qpos(q,ref_yaw=float(current[2])) for q in full])


def drive_rounded_route(env, planner, task, corner, finish, final_yaw):
    """Only the base moves; one physical stop at the terminal working dock."""
    log = lambda text,**kw: common.say(env,'same_drawer_planner',text,**kw)
    robot, p = task.agent.robot, planner.planner
    start = np.asarray(task.agent.base_link.pose[0].sp.p)[:2]
    current = robot.get_qpos()[0].cpu().numpy().astype(float)
    moves = list(p.move_group_joint_indices)
    selected = None
    p.update_from_simulation()
    for cut in (.40,.32,.25,.18,.12):
        route = rounded_route(start,corner,finish,cut)
        if route is None:continue
        poses = route_qpos(planner,current,route['points'],route['yaw'])
        if path_clear(planner,current,poses[:,moves]):
            selected = route;break
    if selected is None:
        return common.fail(env,'same_drawer_planner','no collision-free rounded aisle route')
    names = task.agent.controller.controllers['arm'].config.joint_names
    previous = getattr(planner,'fixed_action_targets',{})
    fixed = {i:float(current[int(robot.active_joints_map[n].active_index[0])]) for i,n in enumerate(names)}
    fixed[10] = float(current[int(robot.active_joints_map['torso_lift_joint'].active_index[0])])
    planner.fixed_action_targets = dict(previous);planner.fixed_action_targets.update(fixed)
    try:
        heading = selected['yaw'][0]
        result = planner.rotate_base_z([math.cos(heading),math.sin(heading),0.])
        if result == -1 or planner.truncated:return result
        log('smooth return through open aisle',corner=np.asarray(corner).tolist(),finish=np.asarray(finish).tolist(),
            cut_m=selected['cut'],path_length_m=float(selected['distance'][-1]))
        points,s,yaw,kappa = (selected[k] for k in ('points','distance','yaw','curvature'))
        v,omega,index = 0.,0.,0
        dt=2*float(task.control_timestep)
        travelled=[]
        for tick in range(int(s[-1]/(.08*dt))+100):
            base = task.agent.base_link.pose[0].sp.to_transformation_matrix()
            here = base[:2,3]; heading=math.atan2(base[1,0],base[0,0])
            distance = float(np.linalg.norm(here-np.asarray(finish)[:2]))
            arrived = distance <= .025 and index >= len(points)-15
            if arrived and v <= .35*dt and abs(omega) <= 1.5*dt:break
            end=min(len(points),index+100)
            index += int(np.argmin(np.linalg.norm(points[index:end]-here,axis=1)))
            remaining = max(distance,float(s[-1]-s[index]))
            preview=min(len(points)-1,int(np.searchsorted(s,s[index]+.20)))
            bend=max(float(np.max(np.abs(kappa[index:preview+1]))),1e-6)
            desired=min(.42,math.sqrt(.25/bend),.85/bend,math.sqrt(2*.35*remaining),1.2*remaining)
            if arrived:desired=0.
            v=float(np.clip(desired,v-.35*dt,v+.35*dt))
            aim_index=min(len(points)-1,int(np.searchsorted(s,s[index]+max(.08,.45*v))))
            delta=points[aim_index]-here
            angle=wrap(math.atan2(delta[1],delta[0])-heading)
            desired_omega=2*v*math.sin(angle)/max(float(np.linalg.norm(delta)),.04)
            if arrived:desired_omega=0.
            omega=float(np.clip(np.clip(desired_omega,-.85,.85),omega-1.5*dt,omega+1.5*dt))
            # Check the commanded 10 Hz hold, including intermediate poses.
            times=np.linspace(0,dt,6)
            headings=heading+omega*times
            if abs(omega)<1e-8:
                predicted=here+times[:,None]*v*np.array([math.cos(heading),math.sin(heading)])
            else:
                predicted=here+v/omega*np.column_stack([np.sin(headings)-math.sin(heading),-np.cos(headings)+math.cos(heading)])
            q=robot.get_qpos()[0].cpu().numpy().astype(float)
            p.update_from_simulation()
            full=route_qpos(planner,q,predicted,headings)
            if not path_clear(planner,q,full[:,moves]):
                return common.fail(env,'same_drawer_planner','rounded-route command intersects fixture',tick=tick)
            arm,body=planner._hold_targets()
            command=planner._compose(arm,body,[v,omega/float(task.agent.controller.controllers['base'].config.upper[1])])
            for _ in range(2):
                result=planner._step(command)
                if planner.truncated:return result
            travelled.append((float(v),float(omega)))
        else:
            return common.fail(env,'same_drawer_planner','rounded-route follower did not arrive')
        log('smooth travel finished',distance_m=distance,ticks=len(travelled))
        result=planner.idle_steps(t=2)
        p.update_from_simulation()
        result=planner.rotate_base_z([math.cos(final_yaw),math.sin(final_yaw),0.])
        log('smooth return arrived',distance_m=distance,ticks=len(travelled),
            minimum_moving_speed_m_s=min(x[0] for x in travelled) if travelled else 0.)
        return result
    finally:
        planner.fixed_action_targets=previous
