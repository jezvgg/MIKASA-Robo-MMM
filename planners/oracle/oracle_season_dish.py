"""SeasonDish-specific planning, upright-payload, and wrist-pour helpers."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import signal
import time
import math
import numpy as np
from planners.oracle import oracle_common as common
import contextlib
import mplib
import sapien
from mplib.collision_detection import fcl
import xml.etree.ElementTree as ET
from transforms3d.euler import euler2mat
from planners.oracle.paths import dense_samples, path_clear


class PlanningBudgetExceeded(RuntimeError):
    pass

class SearchBudget:
    def __init__(self, seconds=120., clock=time.perf_counter):
        self.limit=float(seconds); self.used=0.; self.depth=0; self.clock=clock
        self.exhausted=False; self.calls={}

    @contextmanager
    def measure(self, name):
        if self.exhausted or self.used >= self.limit:
            raise PlanningBudgetExceeded(f'planning budget exhausted: {self.used:.3f}/{self.limit:.0f}s')
        self.calls[name]=self.calls.get(name,0)+1
        outer=self.depth==0
        if outer:
            start=self.clock()
            previous=signal.getsignal(signal.SIGALRM)
            def timeout(*_):
                self.exhausted=True
                raise PlanningBudgetExceeded(f'planning budget exhausted: {self.limit:.0f}s')
            signal.signal(signal.SIGALRM,timeout)
            signal.setitimer(signal.ITIMER_REAL,max(.001,self.limit-self.used))
        self.depth+=1
        try:
            yield
        finally:
            self.depth-=1
            if outer:
                signal.setitimer(signal.ITIMER_REAL,0)
                signal.signal(signal.SIGALRM,previous)
                self.used+=self.clock()-start
                self.exhausted |= self.used >= self.limit
        if self.exhausted:
            raise PlanningBudgetExceeded(f'planning budget exhausted: {self.used:.3f}/{self.limit:.0f}s')

    def wrap(self, function, name, *, ik=False):
        @wraps(function)
        def call(*args, **kwargs):
            if ik:
                kwargs['n_init_qpos']=min(32,int(kwargs.get('n_init_qpos',32)))
            with self.measure(name):
                return function(*args,**kwargs)
        return call

def install(planner, task, seconds=120.):
    budget=SearchBudget(seconds); planner.search_budget=budget; task.search_budget=budget
    for name in ('IK','plan_qpos_line','plan_screw','plan_pose','plan_qpos','TOPP'):
        method=getattr(planner.planner,name,None)
        if callable(method):
            setattr(planner.planner,name,budget.wrap(method,name,ik=name=='IK'))
    return budget

def measured(function):
    @wraps(function)
    def call(planner,*args,**kwargs):
        budget=getattr(planner,'search_budget',None)
        if budget is None:return function(planner,*args,**kwargs)
        with budget.measure(function.__name__):
            return function(planner,*args,**kwargs)
    return call


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


FINGER_LENGTH = 0.025


def wait_cue(env, planner, info, who: str = "season_dish_planner"):
    """Wait out the visual cue; preserve planner refusals and horizon truncations."""
    task = env.unwrapped
    cue_steps = int(task.cfg.cue_steps)
    remaining = cue_steps - int(common._np(task.elapsed_steps).reshape(-1)[0])
    common.say(env, who, "wait out the cue", steps=remaining)
    while remaining > 0:
        chunk = min(remaining, 20)
        result = planner.idle_steps(t=chunk)
        if result == -1:
            return common.fail(env, who, "wait out the cue")
        if common.stopped_by_horizon(planner):
            return result
        remaining -= chunk
        info = result[-1]
    return info


def grasp_geometry(task, obb, ee_direction, target_closing, grasp_info):
    """Build the measured SeasonDish grasp and its 10 cm standoff pose."""
    info = grasp_info(obb, ee_direction, target_closing)
    grasp = task.agent.build_grasp_pose(info["approaching"], info["closing"], info["center"])
    reach = grasp * sapien.Pose([0, 0, -0.1])
    base_pos = common._np(task.agent.base_link.pose.p)[0]
    obj_center = np.asarray(obb.center_mass)
    if np.dot(reach.p - obj_center, base_pos - obj_center) < 0:
        # The standoff is on the far side of the object from the base: flip it.
        info = grasp_info(obb, -ee_direction, target_closing)
        grasp = task.agent.build_grasp_pose(info["approaching"], info["closing"], info["center"])
        reach = grasp * sapien.Pose([0, 0, -0.1])
    return grasp, reach


@contextlib.contextmanager
def keepout(planner, actors, pad=0.03, prefix: str = "keepout"):
    """Inflate loose objects in the planning world while planning the approach."""
    actors = list(actors)
    pads = [float(pad)] * len(actors) if np.isscalar(pad) else [float(x) for x in pad]
    world = planner.planner.planning_world
    added: list[str] = []
    try:
        for i, actor in enumerate(actors):
            mesh = actor.get_first_collision_mesh(to_world_frame=True)
            if mesh is None:
                continue
            obb = mesh.bounding_box_oriented
            extents = np.asarray(obb.primitive.extents, dtype=np.float64)
            centre = np.asarray(obb.centroid, dtype=np.float64)
            shape = fcl.Cylinder(
                float(max(extents[0], extents[1]) / 2.0 + pads[i]),
                float(extents[2] + 2.0 * pads[i]),
            )
            name = f"{prefix}_{i}"
            world.add_object(
                name,
                fcl.CollisionObject(
                    shape, mplib.Pose(p=centre, q=[1.0, 0.0, 0.0, 0.0])
                ),
            )
            added.append(name)
        yield
    finally:
        for name in added:
            planner.planner.remove_object(name)


class PayloadTiltError(RuntimeError):
    pass


class ArmKinematics:
    """Read the pinned URDF; evaluate named links without touching physics."""

    def __init__(self, task):
        self.task = task
        robot = task.agent.robot
        self.indices = {j.name: i for i, j in enumerate(robot.get_active_joints())}
        self.joints = {}
        for joint in ET.parse(task.agent.urdf_path).getroot().findall('joint'):
            origin = joint.find('origin')
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                transform[:3, :3] = euler2mat(*np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
            axis = joint.find('axis')
            self.joints[joint.find('child').get('link')] = (
                joint.find('parent').get('link'), joint.get('name'), joint.get('type'),
                transform, np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else None)
        self.chains = {}
        for link in ('gripper_link', 'wrist_roll_link'):
            chain, current = [], link
            while current in self.joints:
                item = self.joints[current]
                chain.append(item)
                current = item[0]
            self.chains[link] = chain[::-1]

    def matrix(self, q, link='gripper_link'):
        result = self.task.agent.robot.root_pose[0].sp.to_transformation_matrix().astype(float)
        for _, name, kind, origin, axis in self.chains[link]:
            result = result @ origin
            if kind == 'fixed':
                continue
            value = q[self.indices[name]]
            if kind == 'prismatic':
                result[:3, 3] += result[:3, :3] @ (axis * value)
            else:
                result[:3, :3] = result[:3, :3] @ rotation(axis, value)
        return result


def kinematics(planner, task):
    if not hasattr(planner, '_payload_kinematics'):
        planner._payload_kinematics = ArmKinematics(task)
    return planner._payload_kinematics


@contextmanager
def elbow_only(planner, task):
    previous = getattr(planner, '_grasp_branch', {})
    planner._grasp_branch = {int(task.agent.robot.active_joints_map['elbow_flex_joint'].active_index[0]): 1}
    try:
        yield
    finally:
        planner._grasp_branch = previous


@contextmanager
def preview_roll_history(planner, positions):
    """Include a hypothetical prefix in the later path's episode roll budget."""
    low, high = planner._roll_low, planner._roll_high
    rolls = np.atleast_2d(positions)[:, planner._roll_indices]
    planner._roll_low = np.minimum(low, rolls.min(axis=0))
    planner._roll_high = np.maximum(high, rolls.max(axis=0))
    try:
        yield
    finally:
        planner._roll_low, planner._roll_high = low, high


def upright_wrist_candidates(axis_in_wrist, up_in_parent, previous, limits):
    """Solve Ry(flex) Rx(roll) axis = up, choosing continuous nearby angles."""
    a = np.asarray(axis_in_wrist, float)
    v = np.asarray(up_in_parent, float)
    radius = math.hypot(a[1], a[2])
    if radius < 1e-9 or abs(v[1]) > radius + 1e-6:
        return []
    offset = math.acos(float(np.clip(v[1] / radius, -1., 1.)))
    phase = math.atan2(-a[2], a[1])
    candidates = []
    for sign in (-1, 1):
        for turn in (-1, 0, 1):
            roll = phase + sign * offset + turn * 2 * math.pi
            b = rotation([1, 0, 0], roll) @ a
            flex = math.atan2(v[0], v[2]) - math.atan2(b[0], b[2])
            flex = (flex + math.pi) % (2 * math.pi) - math.pi
            value = np.array([flex, roll])
            if np.all(value >= limits[:, 0]) and np.all(value <= limits[:, 1]):
                candidates.append(value)
    return sorted(candidates, key=lambda x: float(np.linalg.norm(x - previous)))


@measured
def upright_path(planner, task, current, goal, hand_object, *, max_tilt_degrees=5., terminal="hover"):
    """Compensate the wrist throughout a monotone fold, then collision-check TOPP."""
    mode = getattr(task, 'motion_parameters', {}).get('upright_compensation', 'exact')
    if mode in ('cone12', 'cone_hover12'):
        raise ValueError(
            "upright_compensation modes 'cone12' and 'cone_hover12' were removed"
        )
    p = planner.planner
    failures = []; planner._upright_failures = failures
    fk = kinematics(planner, task)
    wrist = [fk.indices[n] for n in ('wrist_flex_joint', 'wrist_roll_joint')]
    local_axis = hand_object[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    limits = task.agent.robot.get_qlimits()[0].cpu().numpy()[wrist].copy()
    limits[1] = [max(limits[1, 0], -math.pi + .05), min(limits[1, 1], math.pi - .05)]
    moves = list(p.move_group_joint_indices)
    start_axis = fk.matrix(current)[:3, :3] @ local_axis
    if start_axis[2] <= math.cos(math.radians(15.)):
        failures.append("initial_tilt_limit")
        return None
    count = max(81, int(np.ceil(np.linalg.norm(goal - current) / .02)) + 1)
    # Time progress can curve around fixtures without introducing another pose
    # or reversing the elbow. All candidates preserve the same compact endpoint.
    for shoulder_power, elbow_power in ((1., 1.), (2., 1.)):
        full = [current.copy()]
        for alpha in np.linspace(0., 1., count)[1:]:
            progress = np.full(len(current), alpha)
            progress[[fk.indices['shoulder_pan_joint'], fk.indices['shoulder_lift_joint']]] = alpha ** shoulder_power
            progress[[fk.indices['upperarm_roll_joint'], fk.indices['elbow_flex_joint']]] = alpha ** elbow_power
            q = current + progress * (goal - current)
            q[wrist] = 0.
            wr = fk.matrix(q, 'wrist_roll_link')[:3, :3]
            hand = fk.matrix(q)[:3, :3]
            up = (1 - min(1., alpha * 8)) * start_axis + min(1., alpha * 8) * np.array([0., 0., 1.])
            up /= np.linalg.norm(up)
            candidates = upright_wrist_candidates(wr.T @ hand @ local_axis, wr.T @ up, full[-1][wrist], limits)
            if not candidates:
                failures.append("wrist_limit_or_unreachable_axis")
                break
            q[wrist] = candidates[0]
            if np.max(np.abs(q[wrist] - full[-1][wrist])) > .20:
                failures.append("configuration_jump")
                break
            full.append(q)
        if len(full) != count:
            continue
        full = np.asarray(full)
        with elbow_only(planner, task):
            if not p.accepts(full):
                failures.append("joint_limit")
                continue
            if not path_clear(planner, current, full[:, moves]):
                failures.append({"collision": getattr(planner,"_last_path_collision",None)})
                continue
            try:
                times, pos, vel, acc, duration = p.TOPP(full[:, moves], task.control_timestep)
            except RuntimeError:
                continue
            if not p.accepts(pos, move_group=True) or not path_clear(planner, current, pos):
                continue
        max_tilt = 0.
        for knot in dense_samples(pos):
            q = current.copy(); q[moves] = knot
            axis = fk.matrix(q)[:3, :3] @ local_axis
            max_tilt = max(max_tilt, math.degrees(math.acos(float(np.clip(axis[2], -1, 1)))))
        if max_tilt >= 15. or max_tilt > max(max_tilt_degrees, math.degrees(math.acos(float(np.clip(start_axis[2], -1, 1)))) + .1):
            continue
        return dict(status='Success', time=times, position=pos, velocity=vel,
                    acceleration=acc, duration=duration, predicted_max_tilt_deg=max_tilt)
    return None


@contextmanager
def enforce_upright(planner, task, target, maximum_degrees=15.):
    """Observe every executed control step; do not alter any recorded action."""
    original = planner._step
    def check():
        axis = target.pose[0].sp.to_transformation_matrix()[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
        tilt = math.degrees(math.acos(float(np.clip(axis[2], -1., 1.))))
        if tilt >= maximum_degrees or not bool(task.agent.is_grasping(target).any()):
            raise PayloadTiltError(f'payload transfer refused: tilt={tilt:.3f} degrees or lost grasp')
    def checked(action):
        result = original(action)
        check()
        return result
    check()
    planner._step = checked
    try:
        yield
    finally:
        planner._step = original
