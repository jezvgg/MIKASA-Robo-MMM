"""Per-episode wrist, torso-yaw, and hand-impact checks for tray demos."""

import numpy as np

CHECKER_VERSION = 4  # bump when thresholds or interpretation changes
WRIST_180_RAD = np.deg2rad(179.0)
TORSO_180_RAD = np.deg2rad(179.0)
MIN_IMPACT_IMPULSE_NS = 1e-3
HAND_LINKS = {
    "wrist_flex_link",
    "wrist_roll_link",
    "gripper_link",
    "l_gripper_finger_link",
    "r_gripper_finger_link",
}


def _labels(body):
    entity = getattr(body, "entity", None)
    names = {getattr(body, "name", ""), getattr(entity, "name", "")}
    return {
        name.removeprefix("scene-0_").removeprefix("scene-0-")
        for name in names
        if isinstance(name, str) and name
    }


class TrayEpisodeCheckers:
    """Track wrist roll and physical hand/object impacts on executed steps."""

    def __init__(self, task):
        self.task = task
        self.callback = self.observe_step
        joints = task.agent.robot.active_joints
        self.wrist_index = next(
            i for i, joint in enumerate(joints) if joint.name == "wrist_roll_joint"
        )
        qpos = task.agent.robot.get_qpos()[0].detach().cpu().numpy()
        self._last_wrist = float(qpos[self.wrist_index])
        self._last_torso_yaw = self._torso_yaw()
        self._torso_yaw_delta = 0.0
        self._max_torso_yaw_excursion = 0.0
        self.torso_180_triggered = False
        self._wrist_delta = 0.0
        self._max_wrist_excursion = 0.0
        self.wrist_180_triggered = False
        self.steps = 0
        self.robot_links = {}
        for link in task.agent.robot.get_links():
            for label in _labels(link._objs[0]) | {link.name}:
                self.robot_links[label] = link.name
        self.cup_labels = {"cup"}
        self.impacts = {}

    def _torso_yaw(self):
        # Fetch torso_lift is prismatic; base_link yaw is robot body heading.
        w, x, y, z = self.task.agent.base_link.pose.q[0].detach().cpu().numpy()
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def observe_step(self):
        self.steps += 1
        qpos = self.task.agent.robot.get_qpos()[0].detach().cpu().numpy()
        wrist = float(qpos[self.wrist_index])
        delta = (wrist - self._last_wrist + np.pi) % (2 * np.pi) - np.pi
        self._wrist_delta += delta
        self._last_wrist = wrist
        self._max_wrist_excursion = max(
            self._max_wrist_excursion, abs(self._wrist_delta)
        )
        self.wrist_180_triggered |= self._max_wrist_excursion >= WRIST_180_RAD

        torso_yaw = self._torso_yaw()
        yaw_delta = (torso_yaw - self._last_torso_yaw + np.pi) % (2 * np.pi) - np.pi
        self._torso_yaw_delta += yaw_delta
        self._last_torso_yaw = torso_yaw
        self._max_torso_yaw_excursion = max(
            self._max_torso_yaw_excursion, abs(self._torso_yaw_delta)
        )
        self.torso_180_triggered |= self._max_torso_yaw_excursion >= TORSO_180_RAD

        step_impacts = {}
        for contact in self.task.scene.get_contacts():
            bodies = contact.bodies
            if len(bodies) != 2:
                continue
            a, b = (_labels(body) for body in bodies)
            robot_a = a & self.robot_links.keys()
            robot_b = b & self.robot_links.keys()
            if bool(robot_a) == bool(robot_b):
                continue
            robot = robot_a or robot_b
            hand_aliases = [
                name for name in robot if self.robot_links[name] in HAND_LINKS
            ]
            other = b if robot_a else a
            if (
                not hand_aliases
                or not other
                or other & self.cup_labels
                or any("ground" in n or "floor" in n for n in other)
            ):
                continue
            robot_link = self.robot_links[hand_aliases[0]]
            obj = sorted(other)[0]

            impulse = sum(
                float(np.linalg.norm(point.impulse)) for point in contact.points
            )
            if impulse < MIN_IMPACT_IMPULSE_NS:
                continue
            key = (robot_link, obj)
            step_impacts[key] = max(step_impacts.get(key, 0.0), impulse)
        for key, impulse in step_impacts.items():
            impact = self.impacts.setdefault(
                key, {"frames": 0, "max_impulse_ns": 0.0, "first_step": self.steps}
            )
            impact["frames"] += 1
            impact["max_impulse_ns"] = max(impact["max_impulse_ns"], impulse)

    def summary(self):
        return {
            "checker_version": CHECKER_VERSION,
            "wrist_180_triggered": bool(self.wrist_180_triggered),
            "wrist_roll_max_excursion_deg": round(
                float(np.rad2deg(self._max_wrist_excursion)), 3
            ),
            "torso_180_triggered": bool(self.torso_180_triggered),
            "torso_yaw_max_excursion_deg": round(
                float(np.rad2deg(self._max_torso_yaw_excursion)), 3
            ),
            "collision_triggered": bool(self.impacts),
            "collision_contact_frames": sum(x["frames"] for x in self.impacts.values()),
            "collision_impulse_threshold_ns": MIN_IMPACT_IMPULSE_NS,
            "collision_pairs": [
                {"robot_link": link, "object": obj, **details}
                for (link, obj), details in sorted(self.impacts.items())
            ],
            "checker_steps": self.steps,
        }


def checker_rejection(trace_dir, seed):
    """Return rejection reason, or None when both checks passed."""
    result = read_episode_checkers(trace_dir, seed)
    if (
        result.get("checker_version") != CHECKER_VERSION
        or result.get("checker_error")
        or result.get("checker_steps", 0) < 1
    ):
        return "checker_failed"
    if not all(
        key in result
        for key in (
            "wrist_180_triggered",
            "torso_180_triggered",
            "collision_triggered",
        )
    ):
        return "checker_failed"
    reasons = []
    if result["wrist_180_triggered"]:
        reasons.append("wrist_180")
    if result["collision_triggered"]:
        reasons.append("collision")
    if result["torso_180_triggered"]:
        reasons.append("torso_180")
    return "filtered_" + "_".join(reasons) if reasons else None


def read_episode_checkers(trace_dir, seed):
    """Read final checker event for one logged seed; fail closed if it is absent."""
    import json
    from pathlib import Path

    logs = sorted(Path(trace_dir).glob(f"seed_{seed}/*_events.jsonl"))
    if len(logs) != 1:
        raise ValueError(f"expected one event log for seed {seed}, found {len(logs)}")
    results = [
        json.loads(line)
        for line in logs[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [event for event in results if event.get("event") == "safety_checkers"]
    verdicts = [event for event in results if event.get("event") == "verdict"]
    if not matches:
        raise ValueError(f"safety-checker result missing for seed {seed}")
    if not verdicts or verdicts[-1].get("success") is not True:
        raise ValueError(f"successful planner verdict missing for seed {seed}")
    return matches[-1]
