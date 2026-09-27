"""Keep a held condiment upright without changing the robot or its controller."""
from contextlib import contextmanager
import math
import xml.etree.ElementTree as ET

import numpy as np
from transforms3d.euler import euler2mat

from planners.oracle.path_clearance import dense_samples, path_clear
from planners.oracle.wrist_pour import rotation


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


def upright_path(planner, task, current, goal, hand_object, *, max_tilt_degrees=5.):
    """Compensate the wrist throughout a monotone fold, then collision-check TOPP."""
    p = planner.planner
    fk = kinematics(planner, task)
    wrist = [fk.indices[n] for n in ('wrist_flex_joint', 'wrist_roll_joint')]
    local_axis = hand_object[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    limits = task.agent.robot.get_qlimits()[0].cpu().numpy()[wrist].copy()
    limits[1] = [max(limits[1, 0], -math.pi + .05), min(limits[1, 1], math.pi - .05)]
    moves = list(p.move_group_joint_indices)
    start_axis = fk.matrix(current)[:3, :3] @ local_axis
    if start_axis[2] < math.cos(math.radians(10.)):
        return None
    count = max(81, int(np.ceil(np.linalg.norm(goal - current) / .02)) + 1)
    # Time progress can curve around fixtures without introducing another pose
    # or reversing the elbow. All candidates preserve the same compact endpoint.
    for shoulder_power, elbow_power in ((1., 1.), (2., 1.), (.5, 1.), (1., 2.), (2., 2.), (.5, 2.)):
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
                break
            q[wrist] = candidates[0]
            if np.max(np.abs(q[wrist] - full[-1][wrist])) > .20:
                break
            full.append(q)
        if len(full) != count:
            continue
        full = np.asarray(full)
        with elbow_only(planner, task):
            if not p.accepts(full) or not path_clear(planner, current, full[:, moves]):
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
        if max_tilt > max(max_tilt_degrees, math.degrees(math.acos(float(np.clip(start_axis[2], -1, 1)))) + .1):
            continue
        return dict(status='Success', time=times, position=pos, velocity=vel,
                    acceleration=acc, duration=duration, predicted_max_tilt_deg=max_tilt)
    return None


@contextmanager
def enforce_upright(planner, task, target, maximum_degrees=10.):
    """Observe every executed control step; do not alter any recorded action."""
    original = planner._step
    def checked(action):
        result = original(action)
        axis = target.pose[0].sp.to_transformation_matrix()[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
        tilt = math.degrees(math.acos(float(np.clip(axis[2], -1., 1.))))
        if tilt > maximum_degrees or not bool(task.agent.is_grasping(target).any()):
            raise PayloadTiltError(f'payload transfer refused: tilt={tilt:.3f} degrees or lost grasp')
        return result
    planner._step = checked
    try:
        yield
    finally:
        planner._step = original
