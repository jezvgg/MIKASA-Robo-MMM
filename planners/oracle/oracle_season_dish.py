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


# The wrist pour. Only wrist_roll moves. The shaking pour (profile 28, since 2026-10-27) turns the
# held condiment to just short of upside down, then per shake turns it `SHAKE_AMPLITUDE_DEG` back
# toward upright and returns, then turns it upright again (the wrist back where it was before the
# pour, owner 2026-10-28) and holds. The shakes go to one side only: checklist D2 lets each roll
# joint span at most pi - 0.05 over an episode (`paths.RollPathPlanner.accepts`), and turning
# upside down already takes ~180 deg, so a swing past upside down is never accepted (the +-30 deg
# version was refused on 8 of 8 seeds, 2026-10-28, and the owner chose the one-sided swing).

#: Wrist roll each shake swings back toward upright and returns (the owner, 2026-10-27: 30 deg;
#: one side only since 2026-10-28, see above).
SHAKE_AMPLITUDE_DEG = 30.0
#: Lean kept clear of the task's shake thresholds by the planned rest and shake poses, degrees: the
#: route preview (a model of the held condiment) keeps 3, the pour itself (the measured one) 1.5, so a
#: degree of model-to-physics drift does not refuse at the bowl a pour the preview accepted. The rest
#: leans exactly `shake_return_deg` less the margin: the least roll that still re-arms the counter.
SHAKE_LEAN_MARGIN_DEG = 3.0
SHAKE_LEAN_MARGIN_EXECUTED_DEG = 1.5
#: Control steps the pour may wait after a leg for the shake counter to see it (the arm lags its
#: command). The legs used to end in the solver's refinement, which waits for the joint itself: 1-3
#: identical 10 Hz frames a leg, and with nine shakes that alone broke D6's 5 % (2026-10-28).
SETTLE_STEPS = 20


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def tilt_angles(axis, object_axis, target_degrees=165.):
    """All signed rotations within one turn that reach the requested tilt."""
    n, v = np.asarray(axis, dtype=float), np.asarray(object_axis, dtype=float)
    n, v = n / np.linalg.norm(n), v / np.linalg.norm(v)
    c0 = float(n[2] * (n @ v))
    c1, c2 = float(v[2] - c0), float(np.cross(n, v)[2])
    radius = math.hypot(c1, c2)
    if radius < 1e-9:
        return []
    value = (math.cos(math.radians(target_degrees)) - c0) / radius
    if abs(value) > 1 + 1e-9:
        return []
    phase, offset = math.atan2(c2, c1), math.acos(float(np.clip(value, -1, 1)))
    values = {phase + sign * offset + turn * 2 * math.pi
              for sign in (-1, 1) for turn in (-1, 0, 1)}
    return sorted((x for x in values if abs(x) <= 2 * math.pi), key=abs)


def lean_after(axis, object_axis, delta):
    """Lean of `object_axis` from straight down, degrees, after turning it `delta` about `axis`.

    Example:
        >>> round(lean_after([1, 0, 0], [0, 0, 1], math.pi), 1), round(lean_after([1, 0, 0], [0, 0, 1], math.pi + math.radians(30)), 1)
        (0.0, 30.0)
    """
    v = rotation(axis, delta) @ (np.asarray(object_axis, dtype=float) / np.linalg.norm(object_axis))
    return math.degrees(math.acos(float(np.clip(-v[2], -1.0, 1.0))))


def rest_angles(axis, object_axis, lean_deg):
    """The turns about `axis` that leave `object_axis` `lean_deg` short of straight down, smallest
    first: the near side of upside down, where the roll joint spends least of its D2 span.

    Example:
        >>> [round(math.degrees(d)) for d in rest_angles([1, 0, 0], [0, 0, 1], 7.)][:2]
        [173, -173]
        >>> rest_angles([0, 0, 1], [0, 0, 1], 7.)
        []
    """
    return sorted(tilt_angles(axis, object_axis, 180. - lean_deg), key=lambda x: (round(abs(x), 9), x < 0))


def swing_angle(delta, amplitude_deg):
    """The shake's turn: `amplitude_deg` from the rest turn `delta` back toward the start."""
    return delta - math.copysign(math.radians(amplitude_deg), delta)


def shake_geometry_ok(axis, object_axis, delta, cfg, amplitude_deg=SHAKE_AMPLITUDE_DEG,
                      margin_deg=SHAKE_LEAN_MARGIN_DEG):
    """Does turning to `delta` rest the condiment inside the counter's rest band, and does the swing
    `amplitude_deg` back toward upright from there lean it past the shake threshold, with the margin?

    Example:
        >>> from types import SimpleNamespace
        >>> cfg = SimpleNamespace(shake_out_deg=20., shake_return_deg=10.)
        >>> shake_geometry_ok([1, 0, 0], [0, 0, 1], math.radians(173), cfg)
        True
        >>> shake_geometry_ok([1, 0, 0], [0, 0, 1], math.radians(173), cfg, amplitude_deg=15.)
        False
    """
    rest = lean_after(axis, object_axis, delta)
    swing = lean_after(axis, object_axis, swing_angle(delta, amplitude_deg))
    return rest <= cfg.shake_return_deg - margin_deg + 1e-6 and swing >= cfg.shake_out_deg + margin_deg


def _over_bowl(task, end, bowl):
    clearance = end[2] - bowl[2]
    return (np.linalg.norm(end[:2] - bowl[:2]) <= task.cfg.pour_xy_radius
            and task.cfg.pour_min_clearance <= clearance <= task.cfg.pour_max_clearance)


def _plan_shakes(planner, task, q, wrist_index, axis, pivot, obj, bowl, amplitude_deg):
    """The first near-upside-down turn whose rest pose and swing stay over the bowl and plan as
    wrist-only lines: (rest, {leg name: path}, swing qpos) or None. Legs: in (q -> rest), out (rest ->
    swing), back, and home (rest -> q: upright again, over the same roll range the turn over used)."""
    object_axis = obj[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    margin = SHAKE_LEAN_MARGIN_EXECUTED_DEG
    line = lambda goal, start: planner.planner.plan_qpos_line(
        goal, start, time_step=task.control_timestep, ref_yaw=float(start[2]), qpos_step=0.02)
    for delta in rest_angles(axis, object_axis, task.cfg.shake_return_deg - margin):
        if not shake_geometry_ok(axis, object_axis, delta, task.cfg, amplitude_deg, margin_deg=margin):
            continue
        swing = swing_angle(delta, amplitude_deg)
        ends = [pivot + rotation(axis, d) @ (obj[:3, 3] - pivot) for d in (delta, swing)]
        if not all(_over_bowl(task, end, bowl) for end in ends):
            continue
        rest = q.copy(); rest[wrist_index] += delta
        out = q.copy(); out[wrist_index] += swing
        try:
            legs = {"in": line(rest, q), "out": line(out, rest), "back": line(rest, out), "home": line(q, rest)}
        except RuntimeError:
            continue
        if all(leg.get('status') == 'Success' for leg in legs.values()):
            return rest, legs, out
    return None


def pour_and_shake(env, planner, task, target, shakes, amplitude_deg=SHAKE_AMPLITUDE_DEG):
    """Turn the held condiment just short of upside down over the bowl with the wrist alone, shake it
    `shakes` times (back toward upright and again), turn it upright again and hold it there until the
    task's hold succeeds.

    Every leg is planned before the first one moves; all other actuator targets stay fixed.
    """
    log = lambda text, **kw: common.say(env, "season_dish_planner", text, **kw)
    robot = task.agent.robot
    joints = robot.active_joints_map
    q = robot.get_qpos()[0].cpu().numpy().astype(float)
    wrist_index = int(joints['wrist_roll_joint'].active_index[0])
    wrist = robot.links_map['wrist_roll_link'].pose[0].sp.to_transformation_matrix()
    obj = target.pose[0].sp.to_transformation_matrix()
    axis, pivot = wrist[:3, 0], wrist[:3, 3]
    bowl = task.bowl.pose.p[0].cpu().numpy()
    if not bool(task.agent.is_grasping(target).any()):
        return common.fail(env, "season_dish_planner", "wrist pour requires a held object")
    planner.planner.update_from_simulation()
    plan = _plan_shakes(planner, task, q, wrist_index, axis, pivot, obj, bowl, amplitude_deg)
    if plan is None:
        return common.fail(env, "season_dish_planner", "no feasible wrist-only shake")
    rest, legs, out = plan
    object_axis = obj[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    rest_tilt = 180.0 - lean_after(axis, object_axis, rest[wrist_index] - q[wrist_index])
    names = task.agent.controller.controllers['arm'].config.joint_names
    fixed = {i: float(q[int(joints[name].active_index[0])])
             for i, name in enumerate(names) if name != 'wrist_roll_joint'}
    fixed.update({7: -1., 10: float(q[int(joints['torso_lift_joint'].active_index[0])]),
                  11: 0., 12: 0.})
    previous = getattr(planner, 'fixed_action_targets', {})
    planner.fixed_action_targets = fixed
    stopped = lambda r: (isinstance(r, (int, np.integer)) and r == -1) or common.stopped_by_horizon(planner)
    try:
        log('pour', motion='wrist_roll_only', target_degrees=round(rest_tilt, 2), shakes=int(shakes),
            amplitude_deg=float(amplitude_deg), sides=1, wrist_start=float(q[wrist_index]),
            wrist_goal=float(rest[wrist_index]))
        flag = lambda key: (lambda info: bool(np.asarray(common._np(info[key])).any()))
        counted = lambda info: int(common._np(info['shake_count']).item())

        def move(leg, goal, done):
            """Follow one leg without the refinement's wait, then idle at its goal only until `done`."""
            result = planner.follow_forward_path_w_refinement(legs[leg], refine=False)
            planner.fixed_action_targets = dict(fixed)
            planner.fixed_action_targets[6] = float(goal)   # idle must not chase measured joint lag
            for _ in range(SETTLE_STEPS):
                if stopped(result) or done(result[-1]):
                    break
                result = planner.idle_steps(t=1)
            planner.fixed_action_targets = fixed
            return result

        result = move("in", rest[wrist_index], flag('shake_armed'))
        if stopped(result):
            return result
        for k in range(int(shakes)):
            result = move("out", out[wrist_index], flag('shake_swung'))
            if stopped(result):
                return result
            result = move("back", rest[wrist_index], lambda info, k=k: counted(info) > k)
            if stopped(result):
                return result
            info = result[-1]
            log('shake', index=k + 1, shake_count=int(common._np(info['shake_count']).item()))
            if not bool(np.asarray(common._np(info['grasp_target'])).any()):
                return result
        result = move("home", q[wrist_index], flag('upright'))
        if stopped(result):
            return result
        # Keep the final wrist target too; idle must not chase measured joint lag.
        planner.fixed_action_targets = dict(fixed)
        planner.fixed_action_targets[6] = float(q[wrist_index])
        for _ in range(60):
            result = planner.idle_steps(t=1)
            if common.stopped_by_horizon(planner):
                return result
            info = result[-1]
            if not bool(np.asarray(common._np(info['grasp_target'])).any()):
                return result
            if bool(np.asarray(common._np(info['success'])).any()) and int(common._np(info['pour_hold']).item()) >= task.cfg.hold_steps + 2:
                break
        log('held', success=bool(common._np(result[-1]['success']).item()),
            pour_hold=int(common._np(result[-1]['pour_hold']).item()),
            shake_count=int(common._np(result[-1]['shake_count']).item()))
        return result
    finally:
        planner.fixed_action_targets = previous


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
