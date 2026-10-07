import json
from types import SimpleNamespace

import numpy as np
import torch

from utils.tray_episode_checkers import (
    CHECKER_VERSION,
    TrayEpisodeCheckers,
    checker_rejection,
)


class _Body:
    def __init__(self, name, entity_name):
        self.name = name
        self.entity = SimpleNamespace(name=entity_name)


class _Contact:
    def __init__(self, *bodies, impulse):
        self.bodies = bodies
        self.points = [SimpleNamespace(impulse=np.asarray(impulse, dtype=float))]


class _Task:
    def __init__(self):
        self.agent = SimpleNamespace()
        names = [f"joint_{i}" for i in range(13)]
        names[12] = "wrist_roll_joint"
        self.agent.base_link = SimpleNamespace(
            pose=SimpleNamespace(q=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        )
        self.agent.robot = SimpleNamespace(
            active_joints=[SimpleNamespace(name=name) for name in names],
            qpos=torch.zeros((1, 13)),
            get_qpos=lambda: self.agent.robot.qpos,
            get_links=lambda: [
                SimpleNamespace(
                    name="wrist_flex_link",
                    _objs=[
                        _Body("scene-0-ds_fetch_wrist_flex_link", "wrist_flex_link")
                    ],
                ),
                SimpleNamespace(
                    name="l_gripper_finger_link",
                    _objs=[
                        _Body(
                            "scene-0-ds_fetch_l_gripper_finger_link",
                            "l_gripper_finger_link",
                        )
                    ],
                ),
                SimpleNamespace(
                    name="forearm_roll_link",
                    _objs=[
                        _Body(
                            "scene-0-ds_fetch_forearm_roll_link",
                            "forearm_roll_link",
                        )
                    ],
                ),
            ],
        )
        self.agent.grasped = False
        self.agent.is_grasping = lambda _cup: torch.tensor([self.agent.grasped])
        self.cup = SimpleNamespace(
            name="cup",
            _objs=[_Body("cup", "scene-0_cup")],
            pose=SimpleNamespace(p=torch.zeros((1, 3))),
            linear_velocity=torch.zeros((1, 3)),
        )
        self.tray = SimpleNamespace(
            name="tray", _objs=[_Body("baking_tray", "scene-0_baking_tray")]
        )
        self.scene = SimpleNamespace(
            contacts=[], get_contacts=lambda: self.scene.contacts
        )


def test_wrist_checker_tracks_wrapped_roll_excursion():
    task = _Task()
    checks = TrayEpisodeCheckers(task)
    task.agent.robot.qpos[0, 12] = np.deg2rad(178)
    checks.observe_step()
    assert not checks.summary()["wrist_180_triggered"]
    task.agent.robot.qpos[0, 12] = np.deg2rad(179.1)
    checks.observe_step()
    assert checks.summary()["wrist_180_triggered"]


def _set_torso_yaw(task, degrees):
    yaw = np.deg2rad(degrees)
    task.agent.base_link.pose.q[0] = torch.tensor(
        [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    )


def test_torso_yaw_checker_unwraps_angle_crossing():
    task = _Task()
    checks = TrayEpisodeCheckers(task)
    for yaw in (178, -178):
        _set_torso_yaw(task, yaw)
        checks.observe_step()
    result = checks.summary()
    assert result["torso_180_triggered"]
    assert result["torso_yaw_max_turn_deg"] == 182.0


def test_torso_checker_resets_between_separate_turns():
    task = _Task()
    checks = TrayEpisodeCheckers(task)
    _set_torso_yaw(task, 100)
    checks.observe_step()
    for _ in range(3):
        checks.observe_step()
    _set_torso_yaw(task, -160)
    checks.observe_step()
    assert not checks.summary()["torso_180_triggered"]
    assert checks.summary()["torso_yaw_max_turn_deg"] == 100.0
    _set_torso_yaw(task, -80)
    checks.observe_step()
    assert checks.summary()["torso_180_triggered"]
    assert checks.summary()["torso_yaw_max_turn_deg"] == 180.0


def test_torso_checker_resets_on_direction_reversal():
    task = _Task()
    checks = TrayEpisodeCheckers(task)
    for yaw in (100, 0):
        _set_torso_yaw(task, yaw)
        checks.observe_step()
    assert not checks.summary()["torso_180_triggered"]
    assert checks.summary()["torso_yaw_max_turn_deg"] == 100.0
    _set_torso_yaw(task, -80)
    checks.observe_step()
    assert checks.summary()["torso_180_triggered"]
    assert checks.summary()["torso_yaw_max_turn_deg"] == 180.0


def test_hand_counter_hit_flags_but_grasp_contact_does_not():
    task = _Task()
    checks = TrayEpisodeCheckers(task)
    finger = _Body("scene-0-ds_fetch_l_gripper_finger_link", "l_gripper_finger_link")
    cup = _Body("cup", "scene-0_cup")
    counter = _Body("counter_main", "scene-0_counter_main")
    wrist = _Body("scene-0-ds_fetch_wrist_flex_link", "wrist_flex_link")
    task.scene.contacts = [_Contact(finger, cup, impulse=[0.2, 0, 0])]
    checks.observe_step()
    assert not checks.summary()["collision_triggered"]
    task.scene.contacts = [_Contact(wrist, counter, impulse=[0.01, 0, 0])]
    checks.observe_step()
    result = checks.summary()
    assert result["collision_triggered"]
    assert result["collision_pairs"][0]["object"] == "counter_main"


def test_cup_contacts_do_not_count_as_hand_impacts():
    task = _Task()
    checks = TrayEpisodeCheckers(task)
    counter = _Body("counter_main", "scene-0_counter_main")
    task.scene.contacts = [
        _Contact(task.cup._objs[0], counter, impulse=[0.2, 0, 0]),
        _Contact(task.cup._objs[0], task.tray._objs[0], impulse=[0.2, 0, 0]),
    ]
    checks.observe_step()
    assert not checks.summary()["collision_triggered"]


def test_forearm_impact_is_not_a_hand_impact():
    task = _Task()
    checks = TrayEpisodeCheckers(task)
    forearm = _Body("scene-0-ds_fetch_forearm_roll_link", "forearm_roll_link")
    counter = _Body("counter_main", "scene-0_counter_main")
    task.scene.contacts = [_Contact(forearm, counter, impulse=[0.2, 0, 0])]
    checks.observe_step()
    assert not checks.summary()["collision_triggered"]


def test_pipeline_rejects_either_trigger_and_fails_closed(tmp_path):
    seed_dir = tmp_path / "seed_12"
    seed_dir.mkdir()
    log = seed_dir / "tray_events.jsonl"
    base = {
        "event": "safety_checkers",
        "checker_version": CHECKER_VERSION,
        "checker_steps": 1,
        "wrist_180_triggered": False,
        "torso_180_triggered": False,
        "collision_triggered": False,
    }

    def write_log():
        verdict = json.dumps({"event": "verdict", "success": True})
        log.write_text(json.dumps(base) + "\n" + verdict + "\n")

    write_log()
    assert checker_rejection(tmp_path, 12) is None
    base["wrist_180_triggered"] = True
    write_log()
    assert checker_rejection(tmp_path, 12) == "filtered_wrist_180"
    base["collision_triggered"] = True
    write_log()
    assert checker_rejection(tmp_path, 12) == "filtered_wrist_180_collision"
    base["torso_180_triggered"] = True
    write_log()
    assert checker_rejection(tmp_path, 12) == "filtered_wrist_180_collision_torso_180"
