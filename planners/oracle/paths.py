"""Shared path-planning helpers used by the active task planners."""

from __future__ import annotations



# Collision and path geometry

import numpy as np


def dense_samples(positions, max_joint_step=0.025):
    """Include endpoints and bound the Euclidean joint displacement per sample."""
    positions = np.asarray(positions)
    if not len(positions):
        return
    yield positions[0]
    for start, end in zip(positions[:-1], positions[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / max_joint_step)))
        for index in range(1, count + 1):
            yield start + (end - start) * (index / count)


def path_clear(planner, current, positions, *, initial_contacts=()):
    """Planning-model updates only; preserve all other joints and all fixtures."""
    p = planner.planner
    planner._last_path_collision = None
    move = list(p.move_group_joint_indices)
    current = np.asarray(current, dtype=float)
    p.robot.set_qpos(p.fold_qpos(current), True)
    allowed = {frozenset(pair) for pair in initial_contacts}
    for index, knot in enumerate(dense_samples(positions)):
        full = current.copy()
        full[move] = knot
        p.planning_world.set_qpos_all(p.fold_qpos(full)[move])
        if p.planning_world.is_state_colliding():
            # A held object can still touch its support at the starting state.
            # Permit only explicitly named pairs there; the very next sample
            # and every later sample must be entirely collision-free.
            if index == 0 and allowed and np.allclose(full, current, atol=1e-6):
                pairs = {frozenset((c.link_name1, c.link_name2))
                         for c in p.planning_world.check_collision()}
                if pairs and pairs <= allowed:
                    continue
            planner._last_path_collision = [(c.link_name1,c.link_name2)
                for c in p.planning_world.check_collision()]
            return False
    return True


# Rest and recovery paths

from itertools import product
import numpy as np
from planners.oracle import oracle_common as common


def monotone_knots(start, goal, powers, count=81):
    """Vary joint progress without an extra pose or a joint reversal."""
    alpha = np.linspace(0., 1., count)[:, None]
    return start[None] + (goal - start)[None] * alpha ** powers[None]


def curved_rest(env, planner, task, targets, *, who):
    """Try bounded progress curves when the direct rest line hits furniture."""
    p = planner.planner
    q = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
    goal = q.copy()
    joints = task.agent.robot.active_joints_map
    idx = lambda name: int(joints[name].active_index[0])
    for name, value in targets.items():
        goal[idx(name)] = value
    groups = [
        [idx('torso_lift_joint')],
        [idx('shoulder_pan_joint'), idx('shoulder_lift_joint')],
        [idx('elbow_flex_joint')],
        [idx('forearm_roll_joint'), idx('wrist_flex_joint'), idx('wrist_roll_joint')],
    ]
    move = p.move_group_joint_indices

    def clear(knots):
        p.robot.set_qpos(p.fold_qpos(q), True)
        for knot in knots:
            p.planning_world.set_qpos_all(p.fold_qpos(knot)[move])
            if p.planning_world.is_state_colliding():
                return False
        return True

    for exponents in product((1., 2., 4., .5), repeat=len(groups)):
        if all(x == 1 for x in exponents):
            continue  # The caller already checked the direct joint line.
        powers = np.ones(len(q))
        for indices, exponent in zip(groups, exponents):
            powers[indices] = exponent
        knots = monotone_knots(q, goal, powers)
        if not p.accepts(knots) or not clear(knots):
            continue
        try:
            times, pos, vel, acc, duration = p.TOPP(knots[:, move], task.control_timestep)
        except RuntimeError:
            continue
        full = np.broadcast_to(q, (len(pos), len(q))).copy()
        full[:, move] = pos
        if not p.accepts(full) or not clear(full):
            continue
        common.say(env, who, 'fold to canonical rest: curved joint path',
                   progress_powers=list(exponents), knots=len(pos))
        return planner.follow_forward_path_w_refinement(
            dict(status='Success', time=times, position=pos, velocity=vel,
                 acceleration=acc, duration=duration), refine=True)
    common.say(env, who, 'canonical rest curves refused')
    return -1


# Wrist-roll-constrained planning proxy

from contextlib import contextmanager
from functools import partial

import numpy as np


class RollPathPlanner:
    """Reject winding paths before execution; never rewrite simulator states.

    Python IK/screw limits narrow candidate generation. OMPL retains its original
    sampling bounds, so every returned path also needs the explicit check below.
    The check includes intermediate knots and the measured episode history.
    """

    margin = 0.05
    methods = {"IK", "plan_qpos_line", "plan_screw", "plan_qpos", "plan_pose"}

    def __init__(self, planner, owner):
        object.__setattr__(self, "_planner", planner)
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        if name in self.methods:
            return partial(self._plan, name)
        return getattr(self._planner, name)

    def __setattr__(self, name, value):
        setattr(self._planner, name, value)

    def accepts(self, qpos, *, move_group=False):
        full = np.atleast_2d(qpos)
        for index, sign in getattr(self._owner, "_grasp_branch", {}).items():
            column = list(self.move_group_joint_indices).index(index) if move_group else index
            if not np.all(sign * full[:, column] >= self.margin):
                return False
        indices = self._owner._roll_indices
        if move_group:
            indices = [list(self.move_group_joint_indices).index(i) for i in indices]
        roll = np.atleast_2d(qpos)[:, indices]
        if not np.isfinite(roll).all():
            return False
        # A joint line can report an empty success when the goal is already
        # reached. It adds no motion; the caller decides whether a step is needed.
        low, high = self._owner._roll_low, self._owner._roll_high
        if len(roll):
            low = np.minimum(low, roll.min(axis=0))
            high = np.maximum(high, roll.max(axis=0))
        return bool(
            np.all(low >= -np.pi + self.margin)
            and np.all(high <= np.pi - self.margin)
            and np.all(high - low <= np.pi - self.margin)
        )

    def ends_on_goal_branch(self, qpos, *, move_group=False):
        """Only the endpoint must carry the owner's `_goal_branch` signs (D3).

        Unlike `_grasp_branch`, the path may pass through the other branch: the
        transit posture itself has a positive elbow.
        """
        full = np.atleast_2d(qpos)
        if not len(full):
            return True
        for index, sign in getattr(self._owner, "_goal_branch", {}).items():
            column = list(self.move_group_joint_indices).index(index) if move_group else index
            if sign * full[-1, column] < self.margin:
                return False
        return True

    def _screw_on_goal_branch(self, goal, current, *args, **kwargs):
        """Retry a screw that ended on the wrong branch with shoulder_lift held.

        This is the solver's own `joint limit` unjam variant, which the published
        door grasps already take whenever their screw jams at shoulder_lift.
        """
        if kwargs.get("masked_joints") is None:
            return None
        joints = self._owner.env_agent.robot.active_joints_map
        held = np.array(kwargs["masked_joints"], dtype=bool).copy()
        held[int(joints["shoulder_lift_joint"].active_index[0])] = False
        result = self._planner.plan_screw(goal, current, *args, **{**kwargs, "masked_joints": held})
        if result.get("status") != "Success":
            return None
        if not (self.accepts(result["position"], move_group=True)
                and self.ends_on_goal_branch(result["position"], move_group=True)):
            return None
        report = getattr(self._owner, "_report", None)
        if callable(report):
            report("goal_branch", held="shoulder_lift_joint", knots=int(len(result["position"])))
        return result

    @contextmanager
    def _limits(self):
        previous = self._planner.joint_limits
        limits = previous.copy()
        idx = self._owner._roll_indices
        limits[idx, 0] = np.maximum(
            -np.pi + self.margin, self._owner._roll_high - np.pi + self.margin
        )
        limits[idx, 1] = np.minimum(
            np.pi - self.margin, self._owner._roll_low + np.pi - self.margin
        )
        for index, sign in getattr(self._owner, "_grasp_branch", {}).items():
            if sign > 0:
                limits[index, 0] = max(limits[index, 0], self.margin)
            else:
                limits[index, 1] = min(limits[index, 1], -self.margin)
        self._planner.joint_limits = limits
        try:
            yield
        finally:
            self._planner.joint_limits = previous

    def _ik_mask(self, current, supplied):
        mask = np.zeros(len(current), dtype=bool)
        if supplied is not None and len(supplied):
            mask[:] = supplied
        movable = np.zeros(len(current), dtype=bool)
        movable[self.move_group_joint_indices] = True
        return mask | ~movable

    def reference_goals(self, goal, current, mask):
        """Accepted IK solutions from the owner's reference starts alone.

        For callers whose random-restart solutions were valid IK but failed a
        later check (for example the next contact stroke).
        """
        if not self.accepts(current):
            return []
        with self._limits():
            return self._reference_ik(goal, current, self._ik_mask(current, mask))

    def _reference_ik(self, goal, current, *args, **kwargs):
        """Deterministic IK from fixed arm configurations, after random restarts.

        The owner sets `_ik_reference_arms` ((label, {joint index: value}), ...)
        only around a chosen motion. Each start is tried once; every solution
        still passes the grasp branch and roll-window check of `accepts`.
        """
        references = getattr(self._owner, "_ik_reference_arms", None)
        if not references:
            return []
        found = []
        for label, arm in references:
            start = np.array(current, dtype=float)
            start[list(arm)] = list(arm.values())
            status, candidates = self._planner.IK(
                goal, start, *args, **dict(kwargs, n_init_qpos=1))
            if status != "Success":
                continue
            for q in np.atleast_2d(candidates):
                if self.accepts(q) and not any(np.linalg.norm(q - v) < 0.1 for v in found):
                    found.append(q)
                    self._owner._ik_reference_used.append(label)
        return found

    def _plan(self, method, goal, current, *args, **kwargs):
        failure = "collection joint limit: no bounded path"
        # Prevent inherited fix_joint_limits from silently moving the start of a
        # proposed path to a different roll angle than the simulator's actual one.
        if not self.accepts(current):
            return (failure, None) if method == "IK" else {"status": failure}
        if method == "plan_qpos_line" and not (self.accepts(goal) and self.ends_on_goal_branch(goal)):
            return {"status": failure}
        if method == "plan_qpos":
            goal = [q for q in goal if self.accepts(q) and self.ends_on_goal_branch(q)]
            if not goal:
                return {"status": failure}
        if method == "IK":
            # IK random restarts sample the full robot, although the returned
            # path moves only the arm/base group. Keep the head and fingers at
            # their real values, including in subsequent contact previews.
            mask = self._ik_mask(current, args[0] if args else kwargs.get("mask"))
            if args:
                args = (mask, *args[1:])
            else:
                kwargs["mask"] = mask
        elif method == "plan_qpos_line":
            # Native joint-line collision checks update only the move group.
            # A previous hypothetical screw preview can leave its head/fingers
            # in the planning model. Restore the whole MODEL from the measured
            # start; do not change the simulator or restore omitted obstacles.
            self._planner.robot.set_qpos(self._planner.fold_qpos(current), True)
        attempts = 3 if method in {"plan_pose", "plan_qpos"} else 1
        with self._limits():
            for _ in range(attempts):
                result = getattr(self._planner, method)(goal, current, *args, **kwargs)
                if method == "IK":
                    status, candidates = result
                    valid = [] if status != "Success" else [
                        q for q in np.atleast_2d(candidates)
                        if self.accepts(q) and self.ends_on_goal_branch(q)]
                    if not valid:
                        valid = self._reference_ik(goal, current, *args, **kwargs)
                    if not valid:
                        return result if status != "Success" else (failure, None)
                    return "Success", valid[0] if kwargs.get("return_closest") else valid
                if result.get("status") != "Success":
                    return result
                if self.accepts(result["position"], move_group=True):
                    if self.ends_on_goal_branch(result["position"], move_group=True):
                        return result
                    if method == "plan_screw":
                        held = self._screw_on_goal_branch(goal, current, *args, **kwargs)
                        return held if held is not None else {
                            "status": "collection goal branch: wrong elbow/wrist signs at the end"}
        return {"status": failure}


# Smooth base routes

import math
import numpy as np

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


# Straight arm paths

import mplib
import numpy as np


def straight_plan(planner, pose, *, current=None, freeze_lift=None, initial_contacts=()):
    """Return the actual checked path, so execution never replans into an RRT."""
    if current is None:
        current = planner.robot.get_qpos()[0].cpu().numpy()
    planner._straight_failures = []
    for freeze in (True, False) if freeze_lift is None else (freeze_lift,):
        result = planner.planner.plan_screw(
            mplib.Pose(p=pose.p, q=pose.q), current,
            time_step=planner.base_env.control_timestep,
            masked_joints=~np.array([True, True, True, freeze] + [False] * 11),
            goal_tolerance=planner.ARM_SCREW_GOAL_TOLERANCE,
        )
        planner._straight_failures.append(result.get("status"))
        if (result.get("status") == "Success"
                and path_clear(planner, current, result["position"],
                               initial_contacts=initial_contacts)):
            return result
    return None


def execute_straight(planner, path, *, stop_when=None):
    """Finish contact changes on an action-pair boundary; never refine past a stop."""
    if path is None:
        return -1
    def paired_stop():
        return (int(planner.base_env.elapsed_steps[0]) % 2 == 0 and stop_when())
    return planner.follow_forward_path_w_refinement(
        path, refine=False, stop_when=paired_stop if stop_when else None)


# Release settling

import numpy as np


class ReleaseSettling:
    """Require four quiet intervals; a moving sample resets the entire window.

    Contact-constrained bodies can have nonzero solver velocities while their
    actual poses remain stationary. Measure displacement and rotation over each
    control interval instead, and independently require a quiet arm. No state is
    changed. The task success checker still uses its original velocity checks.
    """

    def __init__(self, pose, dt, *, required=4):
        self.previous = np.asarray(pose, dtype=float).copy()
        self.dt = float(dt)
        self.required = required
        self.stable_steps = 0
        self.linear_speed = float("inf")
        self.angular_speed = float("inf")
        self.arm_speed = float("inf")

    @property
    def ready(self):
        return self.stable_steps >= self.required

    def update(self, pose, arm_velocity):
        current = np.asarray(pose, dtype=float)
        self.linear_speed = float(np.linalg.norm(current[:3] - self.previous[:3]) / self.dt)
        qa = self.previous[3:] / np.linalg.norm(self.previous[3:])
        qb = current[3:] / np.linalg.norm(current[3:])
        self.angular_speed = float(2 * np.arccos(np.clip(abs(qa @ qb), 0., 1.)) / self.dt)
        self.arm_speed = float(np.max(np.abs(arm_velocity)))
        stable = (self.linear_speed < .01 and self.angular_speed < .5
                  and self.arm_speed < .05)
        self.stable_steps = self.stable_steps + 1 if stable else 0
        self.previous = current.copy()
        return self.ready
