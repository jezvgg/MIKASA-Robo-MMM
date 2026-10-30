"""Head commands for the collection oracle, using the unchanged DSFetch cameras."""

import math

import numpy as np


def rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    out = np.eye(4)
    if axis == "z":
        out[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    elif axis == "y":
        out[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        raise ValueError(axis)
    return out


def aim_angles(target, tilt_origin, eye_offset, optical_pitch, limits):
    """Aim the midpoint of the two external cameras at a point in the pan frame.

    Both cameras sit behind the head and already point down. Account for their
    offset rotating with head tilt, instead of treating them as eyes at the joint.
    """
    pan = np.clip(math.atan2(target[1], target[0]), *limits[0])
    local = rotation("z", -pan)[:3, :3] @ target - tilt_origin
    x, z = float(local[0]), float(local[2])
    radius = math.hypot(x, z)
    offset = (
        math.sin(optical_pitch) * eye_offset[0]
        + math.cos(optical_pitch) * eye_offset[2]
    )
    tilt = (
        math.asin(np.clip(offset / max(radius, 1e-9), -1, 1))
        - math.atan2(z, x)
        - optical_pitch
    )
    tilt = (tilt + math.pi) % (2 * math.pi) - math.pi
    return np.array([pan, np.clip(tilt, *limits[1])])


def aim_one_eye(target, tilt_origin, eye, limits, start, iterations=40):
    """Aim ONE camera's optical axis (+x of its frame) at a point in the pan frame.

    `eye` is the camera's pose in the tilt link's frame. The two cameras are toed in, so the
    midpoint aim leaves a near target off-centre in each; this centres it in the one named.
    Fixed-point steps on the target's yaw and pitch in the camera frame, from `start`.
    """
    pan, tilt = float(start[0]), float(start[1])
    lift = np.eye(4)
    lift[:3, 3] = tilt_origin
    for _ in range(iterations):
        cam = rotation("z", pan) @ lift @ rotation("y", tilt) @ eye
        d = cam[:3, :3].T @ (np.asarray(target, dtype=float) - cam[:3, 3])
        yaw = math.atan2(d[1], d[0])
        pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
        if abs(yaw) < 1e-4 and abs(pitch) < 1e-4:
            break
        pan = float(np.clip(pan + yaw, *limits[0]))
        tilt = float(np.clip(tilt - pitch, *limits[1]))
    return np.array([pan, tilt])


class HeadTracker:
    """Recompute absolute head targets as the base, torso and target move.

    This helper belongs only to the scripted oracle. Policy evaluation sends the
    policy's recorded 13D actions directly; no tracker or task answer is added.
    """

    deadband = 0.005  # About 0.3 degrees; ignore tiny contact/pose fluctuations.

    def __init__(self, agent, timestep, rate=1.0):
        self.agent = agent
        self.max_delta = float(timestep) * float(rate)
        joints = agent.robot.active_joints_map
        self.indices = [
            int(joints[name].active_index[0])
            for name in ("head_pan_joint", "head_tilt_joint")
        ]
        self.limits = agent.robot.get_qlimits()[0].cpu().numpy()[self.indices].copy()
        self.limits[:, 0] += 0.01
        self.limits[:, 1] -= 0.01
        angles = self.measured()
        pan, tilt, head = (
            self.pose(name)
            for name in ("head_pan_link", "head_tilt_link", "head_camera_link")
        )
        pan_to_tilt = np.linalg.inv(pan) @ tilt @ rotation("y", -angles[1])
        tilt_to_head = np.linalg.inv(tilt) @ head
        if not np.allclose(pan_to_tilt[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("Head tracker expects DSFetch pan/tilt joint axes")
        self.tilt_origin = pan_to_tilt[:3, 3]
        named = {
            cfg.uid: cfg.pose.sp.to_transformation_matrix()
            for cfg in agent._sensor_configs
            if cfg.uid in ("left_base_camera_link", "right_base_camera_link")
        }
        eyes = list(named.values())
        if len(eyes) != 2:
            raise ValueError("Expected the two standard DSFetch external cameras")
        #: Each camera in the tilt link's frame, for `eye`: None aims the midpoint of the two
        #: (the default), a camera uid aims that one camera (`aim_one_eye`).
        self.eyes = {uid: tilt_to_head @ e for uid, e in named.items()}
        self.eye = None
        self.eye_offset = tilt_to_head[:3, 3] + tilt_to_head[:3, :3] @ np.mean(
            [e[:3, 3] for e in eyes], axis=0
        )
        axis = tilt_to_head[:3, :3] @ np.mean([e[:3, 0] for e in eyes], axis=0)
        self.optical_pitch = math.atan2(-axis[2], axis[0])
        self.last_command = angles.copy()
        self.target = None

    def pose(self, name):
        return self.agent.robot.links_map[name].pose[0].sp.to_transformation_matrix()

    def measured(self):
        return self.agent.robot.get_qpos()[0].cpu().numpy()[self.indices].astype(float)

    def desired(self):
        target = self.target() if callable(self.target) else self.target
        target = np.asarray(target, dtype=float).reshape(3)
        if not np.isfinite(target).all():
            raise ValueError("Nonfinite head tracking target")
        neutral = self.pose("head_pan_link") @ rotation("z", -self.measured()[0])
        local = neutral[:3, :3].T @ (target - neutral[:3, 3])
        both = aim_angles(
            local, self.tilt_origin, self.eye_offset, self.optical_pitch, self.limits
        )
        if self.eye is None:
            return both
        return aim_one_eye(local, self.tilt_origin, self.eyes[self.eye], self.limits, both)

    def command(self):
        goal = self.desired()
        delta = goal - self.last_command
        delta[np.abs(delta) < self.deadband] = 0.0
        self.last_command += np.clip(delta, -self.max_delta, self.max_delta)
        return self.last_command.copy()

    def aligned(self, tolerance=0.02):
        return bool(np.max(np.abs(self.measured() - self.desired())) <= tolerance)
