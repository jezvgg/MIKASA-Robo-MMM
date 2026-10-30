"""Consolidated, behavior-preserving collection regression tests."""


# Source group: test_contract.py
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from .client import execute_actions, policy_metadata, policy_observation
from .contract import CAMERAS, DATASET_REPO_ID, DATASET_TASK_PLANNERS, check_actions, held_actions
from .pipeline import NonReplayableMotion, action_only_oracle, initial_state_matches
from .profile import class_methods_sha, json_value


def test_hf_root_contract_lists_all_six_dataset_planners():
    from pathlib import Path

    assert DATASET_REPO_ID == "MIKASA-Robo-MMM/MIKASA-Robo-MMM"
    assert tuple(DATASET_TASK_PLANNERS) == (
        "takeitback",
        "veggie",
        "samedrawer",
        "seasondish",
        "cabinetsearch",
        "depthrecall",
    )
    repo = Path(__file__).resolve().parents[2]
    for module in DATASET_TASK_PLANNERS.values():
        assert (repo / Path(*module.split(".")).with_suffix(".py")).is_file()


def raw_obs():
    return {"agent": {"qpos": np.arange(15, dtype=np.float32)[None]},
            "sensor_data": {name: {"rgb": np.zeros((1, *shape), dtype=np.uint8)}
                            for name, shape in CAMERAS.items()},
            "extra": {"cube_cab": 3, "opened_count": [1, 0, 0, 1],
                      "base_pose": [20, 21, 22], "tcp_pose": [1, 2, 3]}}


def test_global_coordinates_and_task_answers_cannot_cross_policy_boundary():
    original = raw_obs()
    altered = copy.deepcopy(original)
    altered["agent"]["qpos"][0, :3] = [500, 600, 700]
    altered["extra"] = {"cube_cab": 0, "opened_count": [0, 1, 1, 0]}
    first, second = [policy_observation(obs, "Find and nudge the cube.")
                     for obs in (original, altered)]
    assert set(first) == {"observation.state", "prompt"} | {
        f"observation.images.{name}" for name in CAMERAS}
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(first["observation.state"], np.arange(3, 15))
    assert policy_metadata()["state_dim"] == 12


def test_input_rejects_a_fourth_camera_or_wrong_rgb():
    obs = raw_obs()
    obs["sensor_data"]["render_camera"] = obs["sensor_data"]["fetch_hand"]
    with pytest.raises(ValueError, match="cameras"):
        policy_observation(obs, "Find the cube.")
    obs = raw_obs()
    obs["sensor_data"]["fetch_hand"]["rgb"] = np.zeros((1, 128, 128, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="RGB"):
        policy_observation(obs, "Find the cube.")


def test_ten_hz_exec_preserves_head_targets_and_pad_last_interval():
    source = np.zeros((3, 13), dtype=np.float32)
    source[:, 8:10] = [[0.2, -0.3], [0.8, -0.9], [-0.2, 0.4]]
    actual = []
    class Env:
        def step(self, action):
            actual.append(action.copy())
            return {}, 0., False, False, {"success": len(actual) == 4}
    result = execute_actions(Env(), source[::2])
    np.testing.assert_array_equal(actual, source[[0, 0, 2, 2]])
    np.testing.assert_array_equal(actual, held_actions(source))
    assert result["success"] and result["completed"] and result["control_steps"] == 4


def test_controller_checks_do_not_normalize_absolute_arm_positions():
    action = np.zeros((1, 13), dtype=np.float32)
    action[0, 3] = 2.1
    check_actions(action)
    action[0, 11] = 1.1
    with pytest.raises(ValueError, match="gripper/base"):
        check_actions(action)


def test_robot_mutation_guard_allows_reset_and_rejects_motion_writes():
    class Robot:
        def set_qpos(self, value):
            self.value = value
    class Env:
        def __init__(self):
            self.unwrapped = self
            self.agent = SimpleNamespace(robot=Robot())
        def reset(self):
            self.agent.robot = Robot()
            self.agent.robot.set_qpos(3)
            return {}, {}
    env = Env()
    with action_only_oracle(env):
        with pytest.raises(NonReplayableMotion):
            env.agent.robot.set_qpos(1)
        env.reset()
        assert env.agent.robot.value == 3
        with pytest.raises(NonReplayableMotion):
            env.agent.robot.set_qpos(2)
    env.agent.robot.set_qpos(4)
    assert env.agent.robot.value == 4


def test_task_configuration_sets_are_stably_serialized():
    assert json_value({"untangle": frozenset(("b", "a"))}) == {"untangle": ["a", "b"]}


@pytest.mark.parametrize("flags, final_success, ever_success", [
    ([False, True], True, True),
    ([True, False], False, True),
    ([False, False], False, False),
])
def test_h5_time_and_proprio_are_derived_from_recorded_articulation(
    tmp_path, flags, final_success, ever_success
):
    import h5py
    import json
    from .pipeline import add_contract
    path = tmp_path / "trajectory.h5"
    raw = np.arange(3*43, dtype=np.float32).reshape(3, 43)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("traj_0/actions", data=np.zeros((2, 13)))
        h5.create_dataset("traj_0/env_states/articulations/ds_fetch", data=raw)
        h5.create_dataset("traj_0/rewards", data=np.array([0., 1.]))
        h5.create_dataset("traj_0/success", data=np.array(flags))
    path.with_suffix(".json").write_text(json.dumps({"episodes": [{"episode_id": 0}]}))
    add_contract(path, {"stage": "oracle"})
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5["traj_0/qpos"], raw[:, 13:28])
        np.testing.assert_array_equal(h5["traj_0/proprio"], raw[:, 16:28])
        np.testing.assert_array_equal(h5["traj_0/global_state"], raw[:, 13:16])
        np.testing.assert_array_equal(h5["traj_0/timestamp"], [0., 0.05, 0.1])
    metadata = json.loads(path.with_suffix(".json").read_text())["mikasa_data"]
    assert metadata["reward_sum"] == 1 and metadata["success"] is final_success
    assert metadata["success_once"] is ever_success
    episode = json.loads(path.with_suffix(".json").read_text())["episodes"][0]
    assert episode["success"] is final_success and episode["success_once"] is ever_success


def test_failed_attempt_is_retained_only_as_diagnostic(tmp_path):
    from .pipeline import preserve_diagnostic
    path = tmp_path / "trajectory.h5"
    path.write_bytes(b"diagnostic payload")
    path.with_suffix(".json").write_text('{"success": false}')
    assert preserve_diagnostic(path) == "failed-trajectory.h5"
    assert not path.exists()
    assert (tmp_path / "failed-trajectory.h5").read_bytes() == b"diagnostic payload"
    assert (tmp_path / "failed-trajectory.json").exists()


def test_validation_selection_ignores_replay_and_worker_order():
    from .validation import select_validation
    candidates = [{"seed": seed, "planner_success": seed != 1,
                   "replay_success": seed != 2} for seed in (3, 1, 2)]
    assert [c["seed"] for c in select_validation(candidates, set(), count=2)] == [2, 3]
    with pytest.raises(ValueError, match="overlaps"):
        select_validation(candidates, {1}, count=2)  # Failed train attempts still exclude a seed.
    with pytest.raises(ValueError, match="Duplicate"):
        select_validation(candidates+candidates, set(), count=2)
    with pytest.raises(ValueError, match="need 100"):
        select_validation(candidates, set())


def test_campaign_stops_queued_workers_after_storage_error(tmp_path, monkeypatch):
    from . import campaign as module

    monkeypatch.setattr(module, "runtime_signature", lambda: {
        "profile": {"env_id": "test", "env_kwargs": {}}})
    calls = []

    def disk_full(root, phase, seed, timeout):
        calls.append((phase, seed))
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(module, "run_worker", disk_full)
    with pytest.raises(OSError, match="Disk quota exceeded"):
        module.campaign(tmp_path / "run", start_seed=0, num_seeds=100,
                        purpose="development", jobs=1, through="validated", timeout=10)
    assert calls == [("oracle", 0)]


@pytest.mark.parametrize("task_name", ["cabinet_search", "season_dish", "same_drawer"])
def test_robot_adapter_keeps_finger_counter_contact(task_name):
    """The old all-link ignore mask makes the penetrating finger fall through."""
    import sapien
    from my_scenes.cabinet_search import CabinetSearchTask
    from my_scenes.season_dish import SeasonDishTask
    from my_scenes.same_drawer import SameDrawerTask
    task_class = {"cabinet_search": CabinetSearchTask, "season_dish": SeasonDishTask,
                  "same_drawer": SameDrawerTask}[task_name]

    system = sapien.physx.PhysxCpuSystem()
    scene = sapien.Scene([system])
    scene.set_timestep(0.01)

    def body(name, half_size, position, *, static=False, ignore=0):
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.set_initial_pose(sapien.Pose(position))
        entity = builder.build_static(name) if static else builder.build(name)
        component = entity.find_component_by_type(
            sapien.physx.PhysxRigidStaticComponent if static
            else sapien.physx.PhysxRigidDynamicComponent
        )
        for shape in component.get_collision_shapes():
            shape.set_collision_groups([1, 1, ignore, 0])
        return component

    class Link:
        def __init__(self, component):
            self._bodies = [component]

        def set_collision_group_bit(self, group, bit_idx, bit):
            for shape in self._bodies[0].get_collision_shapes():
                groups = shape.get_collision_groups()
                groups[group] = (groups[group] & ~(1 << bit_idx)) | (int(bit) << bit_idx)
                shape.set_collision_groups(groups)

    counter = body("counter", [.1, .1, .1], [0, 0, 0], static=True, ignore=1 << 26)
    finger = body("finger", [.02, .02, .02], [0, 0, .11], ignore=1 << 7)
    left = Link(body("left_wheel", [.02] * 3, [3, 0, 1], ignore=1 << 30))
    right = Link(body("right_wheel", [.02] * 3, [4, 0, 1], ignore=1 << 30))
    base = Link(body("base", [.02] * 3, [5, 0, 1], ignore=1 << 31))
    env = SimpleNamespace(robot_uids="ds_fetch", agent=SimpleNamespace(
        l_wheel_link=left, r_wheel_link=right, base_link=base,
        robot=SimpleNamespace(links=[left, right, base, Link(finger)]),
    ))
    task_class._fix_ds_fetch_collision_bits(env)
    scene.step()
    contacts = [contact for contact in system.get_contacts()
                if finger in contact.bodies and counter in contact.bodies]
    assert contacts, "The scene adapter disabled finger contact with the counter"
    assert any(np.linalg.norm(point.impulse) > 0 for c in contacts for point in c.points)


def test_waypoint_noise_repeats_by_seed_without_mutating_goals():
    import sapien
    from utils.mikasa.waypoint_noise import WaypointNoise
    logs = []
    noise = WaypointNoise(200007, .005, lambda msg, **data: logs.append((msg, data)))
    original = sapien.Pose([.4, .6, .8], [1, 0, 0, 0])
    first = noise.pose("approach", original)
    second = noise.point("dock", [1, 2, 0], axes=(True, True, False))
    repeated = WaypointNoise(200007, .005, lambda *args, **kwargs: None)
    np.testing.assert_array_equal(first.p, repeated.pose("approach", original).p)
    np.testing.assert_array_equal(second, repeated.point("dock", [1, 2, 0], axes=(True, True, False)))
    np.testing.assert_allclose(original.p, [.4, .6, .8])
    np.testing.assert_array_equal(first.q, original.q)
    assert second[2] == 0
    assert all(abs(x) <= .005 for _, e in logs[1:] for x in e["offset_m"])
    assert all(any(x != 0 for x in e["offset_m"]) for _, e in logs[1:])
    with pytest.raises(ValueError, match="0.01"):
        WaypointNoise(0, .02, lambda *a, **kw: None)


@pytest.mark.parametrize("env_id", ["MikasaSeasonDish-v0", "MikasaSameDrawer-v0"])
def test_collection_requires_instruction_preflight(env_id):
    from .profile import validate_instructions
    with pytest.raises(ValueError, match="PaliGemma"):
        validate_instructions({"env_id": env_id}, None)


def test_export_retains_failed_source_summary_and_unknown_incomplete_worker(tmp_path):
    import h5py
    import json
    from .export_lerobot import source_episode_outcomes
    for seed in (7, 8):
        directory = tmp_path / "oracle" / str(seed)
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({"status": "missed" if seed == 7 else "error"}))
    directory = tmp_path / "oracle" / "7"
    with h5py.File(directory / "failed-trajectory.h5", "w") as h5:
        h5.create_dataset("traj_0/actions", data=np.zeros((2, 13)))
        h5.create_dataset("traj_0/rewards", data=[0., 1.])
        h5.create_dataset("traj_0/success", data=[True, False])
        h5.create_dataset("traj_0/terminated", data=[False, False])
        h5.create_dataset("traj_0/truncated", data=[False, True])
    (directory / "failed-trajectory.json").write_text('{"mikasa_data": {"stage": "oracle"}}')
    directory = tmp_path / "oracle" / "8"
    (directory / "trajectory.h5").write_bytes(b"interrupted, not a complete H5")
    (directory / "trajectory.json").write_text('{"episodes": []}')
    completed, incomplete = source_episode_outcomes(tmp_path, {
        "seeds": [7, 8], "signature": {"code_sha256": "source-version"}})
    assert completed["success"] is False and completed["success_once"] is True
    assert completed["terminated"] is False and completed["truncated"] is True
    assert completed["reward_sum"] == 1 and completed["control_steps"] == 2
    assert incomplete["status"] == "error" and "source_h5" not in incomplete
    for key in ("success", "success_once", "reward_sum", "terminated", "truncated"):
        assert incomplete[key] is None


@pytest.mark.parametrize("reopen_m", [None, .007, .13])
def test_same_drawer_rejects_apple_out_of_order(reopen_m):
    """Neither preplacement nor a re-exposed cue may bypass the memory interval.

    Diagnostic state edits create both orders cheaply; contact/settling and task
    updates still use the real environment. These are not demonstrations.
    """
    import gymnasium as gym
    import torch
    import my_scenes  # noqa: F401
    from mani_skill.utils.structs import Pose

    env = gym.make("MikasaSameDrawer-v0", scene_idx=0, sim_backend="cpu",
                   obs_mode="state", control_mode="pd_joint_pos",
                   sim_config={"control_freq": 20, "sim_freq": 100})
    task = env.unwrapped

    def step(count=1):
        arm = task.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body = task.agent.controller.controllers["body"].qpos[0].cpu().numpy()
        for _ in range(count):
            info = env.step(np.r_[arm, 1., body, 0., 0.])[-1]
        return info

    def drawer(index, amount):
        art = task._drawer_arts[index]
        art.set_qpos(torch.full_like(art.get_qpos(), -amount))
        art.set_qvel(torch.zeros_like(art.get_qvel()))

    def place_apple():
        task.apple.set_pose(Pose.create_from_pq(
            p=task.plate.pose.p + torch.tensor([0., 0., .045], device=task.device)))
        task.apple.set_linear_velocity(torch.zeros(1, 3, device=task.device))
        task.apple.set_angular_velocity(torch.zeros(1, 3, device=task.device))
        return step(35)

    try:
        env.reset(seed=371 if reopen_m is None else 374)
        target = int(task.target_drawer.item())
        if reopen_m is not None:
            drawer(target, 0.)
            assert step()["closed_done"].item()
            drawer(target, reopen_m)
            assert step()["sequence_violated"].item()
        before = place_apple()
        assert before["closed_done"].item() == (reopen_m is not None)
        assert not before["apple_done"].item()
        assert before["sequence_violated"].item()
        drawer(int(task.target_drawer.item()), 0.)
        after = step(20)
        assert after["closed_done"].item()
        assert not after["apple_done"].item() and after["failed"].item()
        drawer(target, .13)
        after = step(task.cfg.hold_steps + 5)
        assert after["failed"].item() and not after["success"].item()

        env.reset(seed=372)
        target = int(task.target_drawer.item())
        drawer(target, 0.)
        assert step()["closed_done"].item()
        assert place_apple()["apple_done"].item()
        drawer(target, .13)
        assert not step(task.cfg.hold_steps - 1)["success"].item()
        assert step()["success"].item()
        count = task.held_count.clone()
        for _ in range(10):
            assert task.get_info()["success"].item()
        assert torch.equal(task.held_count, count)

        # Enumerating another drawer invalidates even a previously successful state.
        other = next(i for i in task.cfg.drawer_choices if i != target)
        drawer(other, .03)
        info = step()
        assert info["failed"].item() and not info["success"].item()
        drawer(other, 0.)
        info = step()
        assert info["failed"].item() and not info["success"].item()
    finally:
        env.close()


def test_same_drawer_detent_allows_slow_opening():
    """The first millimetres of an opening stroke must not be reset every tick."""
    import gymnasium as gym
    import torch
    import my_scenes  # noqa: F401

    env = gym.make("MikasaSameDrawer-v0", scene_idx=0, sim_backend="cpu",
                   obs_mode="state", control_mode="pd_joint_pos")
    try:
        env.reset(seed=373)
        task = env.unwrapped
        art = task._drawer_arts[int(task.target_drawer.item())]
        for velocity, expected in ((-.01, -.0025), (.01, 0.), (0., 0.)):
            art.set_qpos(torch.full_like(art.get_qpos(), -.0025))
            art.set_qvel(torch.full_like(art.get_qvel(), velocity))
            task._last_eval_step[:] = -1
            task.get_info()
            assert float(art.get_qpos()[0, 0]) == pytest.approx(expected)
    finally:
        env.close()


# Source group: test_checklist.py
import numpy as np
from utils.collection.checklist import motion_metrics

NAMES = [
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
    "torso_lift_joint",
    "head_pan_joint",
    "shoulder_pan_joint",
    "head_tilt_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "r_gripper_finger_joint",
    "l_gripper_finger_joint",
]


def test_continuous_full_turn_is_not_hidden_by_angle_wrapping():
    a = np.zeros((80, 13))
    a[:, 7] = 1
    a[:, 12] = 0.1
    q = np.zeros((81, 15))
    q[:, 2] = np.linspace(0, 2 * np.pi, 81)
    m = motion_metrics(a, q, NAMES)
    assert np.isclose(m["max_continuous_turn_rad"], 2 * np.pi)
    assert m["longest_constant_action_frames"] == 80


def test_turn_segments_split_at_zero_without_losing_endpoint():
    a = np.zeros((9, 13))
    a[:, 7] = 1
    a[:4, 12] = 0.1
    a[5:, 12] = -0.1
    q = np.zeros((10, 15))
    q[:, 2] = [0, 0.4, 0.8, 1.2, 1.6, 1.6, 1.2, 0.8, 0.4, 0]
    m = motion_metrics(a, q, NAMES)
    assert np.isclose(m["max_continuous_turn_rad"], 1.6)
    assert np.isclose(m["total_turn_rad"], 3.2)


def test_gripper_parity_and_base_roundoff_are_separate_from_real_reverse():
    a = np.zeros((8, 13))
    a[:3, 7] = 1
    a[3:, 7] = -1
    a[:, 11] = -1e-15
    q = np.zeros((9, 15))
    m = motion_metrics(a, q, NAMES)
    assert m["gripper_switch_steps"] == [3]
    assert m["base_tiny_nonzero_count"] == 8
    assert m["reverse_frames"] == 0
    a[4:, 11] = -0.1
    m = motion_metrics(a, q, NAMES)
    assert m["reverse_frames"] == 4


def test_cue_exception_does_not_hide_later_pauses_or_other_tasks():
    from utils.collection.checklist import pause_check
    a = np.zeros((150, 13)); a[:, 7] = 1
    a[50:, 0] = np.arange(100) * .01
    profile = {"env_id": "MikasaSeasonDish-v0", "profile_version": 4,
               "quality_exceptions": {"D6": {"initial_observation_control_steps": 100,
               "phase": "fridge_cue_observation", "preserve_all_frames": True}}}
    check = pause_check(a, profile)
    assert check["passed"] and not check["raw"]["passed"]
    assert check["excluded_initial_frames"] == 50
    other = dict(profile, env_id="MikasaSameDrawer-v0")
    assert not pause_check(a, other)["passed"]
    a[70:110] = a[70]
    assert not pause_check(a, profile)["passed"]
    a[:5, 11] = .1
    assert not pause_check(a, profile)["passed"]


def test_the_shaking_task_excludes_its_final_upright_hold_and_only_that():
    """Profile 27 (owner 2026-10-28): the last final_hold_control_steps / 2 frames are the hold."""
    from utils.collection.checklist import pause_check
    a = np.zeros((250, 13)); a[:, 7] = 1
    a[50:, 0] = np.arange(200) * .01
    a[-12:] = a[-13]                                       # the hold: 12 still frames
    d6 = {"initial_observation_control_steps": 100, "phase": "fridge_cue_observation",
          "preserve_all_frames": True, "final_hold_control_steps": 24}
    profile = {"env_id": "MikasaSeasonDish-v0", "profile_version": 27, "quality_exceptions": {"D6": d6}}
    check = pause_check(a, profile)
    assert check["passed"] and check["excluded_final_frames"] == 12
    assert not pause_check(a, dict(profile, profile_version=26))["passed"]   # 12/199 > 5 % without it
    a[100:110] = a[100]                                    # a pause before the hold still counts
    a[150:160] = a[150]
    assert not pause_check(a, profile)["passed"]


def test_profile_28_excludes_its_shorter_cue_and_only_profile_28_may():
    from utils.collection.checklist import pause_check
    a = np.zeros((150, 13)); a[:, 7] = 1
    a[25:, 0] = np.arange(125) * .01
    d6 = {"initial_observation_control_steps": 50, "phase": "fridge_cue_observation", "preserve_all_frames": True}
    profile = {"env_id": "MikasaSeasonDish-v0", "profile_version": 28, "quality_exceptions": {"D6": d6}}
    check = pause_check(a, profile)
    assert check["passed"] and check["excluded_initial_frames"] == 25
    assert not pause_check(a, dict(profile, profile_version=27))["passed"]


# Source group: test_profile_release.py
import pytest

from utils.collection import profile


@pytest.mark.parametrize("installed,expected,ok", [
    ("2.14.0", "2.14.0", True),
    ("2.14.0+cu126", "2.14.0", True),
    ("2.14.1+cu126", "2.14.0", False),
    ("3.0.0b22", "3.0.0b22", True),
    ("3.0.0b21", "3.0.0b22", False),
])
def test_release_ignores_only_the_local_label(installed, expected, ok):
    assert (profile.release(installed) == expected) is ok


def test_load_profile_accepts_a_local_build_of_the_pinned_release(monkeypatch):
    pins = profile.read_json(profile.PROFILE.with_name("same_drawer_profile.json"))["runtime_packages"]
    installed = {name: version + ("+cu126" if name == "torch" else "") for name, version in pins.items()}
    monkeypatch.setattr(profile.importlib.metadata, "version", lambda name: installed[name])
    for key in [k for k in profile.os.environ if k.startswith("MIKASA_")]:
        monkeypatch.delenv(key)
    assert profile.load_profile("same_drawer")["runtime_packages"] == pins
    installed["torch"] = "2.15.0"
    with pytest.raises(ValueError, match="Runtime requires torch"):
        profile.load_profile("same_drawer")


# Source group: test_retention.py
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from utils.collection import retention
from utils.collection.source_storage import file_sha256, numerical_source


def test_compaction_preserves_raw_states_actions_and_source(tmp_path):
    source = tmp_path / "render.h5"
    target = tmp_path / "numeric.h5"
    actions = np.arange(52, dtype=np.float32).reshape(4, 13)
    with h5py.File(source, "w") as f:
        f.attrs["origin"] = "physical replay"
        f["traj_0/actions"] = actions
        f["traj_0/qpos"] = np.arange(75, dtype=np.float32).reshape(5, 15)
        f["traj_0/env_states/drawer"] = np.linspace(0, .2, 5)
        f["traj_0/obs/agent/qpos"] = f["traj_0/qpos"][:]
        f["traj_0/obs/sensor_data/wrist/rgb"] = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        f["schema_scalar"] = 7
    source.with_suffix(".json").write_text(json.dumps({"seed": 17}))
    digest = file_sha256(source)
    retention.compact_recording(source, target)
    assert file_sha256(source) == digest
    with h5py.File(target) as f:
        np.testing.assert_array_equal(f["traj_0/actions"], actions)
        assert "traj_0/env_states/drawer" in f
        assert "traj_0/obs/agent/qpos" in f
        assert "traj_0/obs/sensor_data" not in f
        assert f["schema_scalar"][()] == 7
        assert f.attrs["origin"] == "physical replay"


def test_modified_numeric_recording_is_rejected(tmp_path):
    p = tmp_path / "numeric.h5"
    p.write_bytes(b"verified numerical recording")
    p.with_suffix(".json").write_text('{}')
    episode = dict(source_h5="unused", numeric_source_h5=str(p),
                   numeric_source_sha256=file_sha256(p),
                   numeric_metadata_sha256=file_sha256(p.with_suffix(".json")))
    assert numerical_source(episode) == p
    p.write_bytes(b"changed numerical recording")
    with pytest.raises(ValueError, match="numerical recording changed"):
        numerical_source(episode)


def test_pruning_rejects_physical_source_even_if_other_checks_pass(tmp_path, monkeypatch):
    p = tmp_path / "oracle/9/trajectory.h5"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"physical source must survive")
    metadata = dict(source_root=str(tmp_path), episodes=[dict(scene_seed=9, source_h5=str(p))])
    (tmp_path / "source_h5_metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "retention.json").write_text('{}')
    monkeypatch.setattr(retention, "verify", lambda root: None)
    monkeypatch.setattr(retention, "numerical_source", lambda episode: Path("verified"))
    with pytest.raises(ValueError, match="restricted"):
        retention.prune(tmp_path)
    assert p.read_bytes() == b"physical source must survive"


# Source group: test_provenance.py
import subprocess

from utils.collection.provenance import committed_snapshot
from utils.collection.source_storage import file_sha256


def test_revision_binding_checks_contents_and_untracked_sources(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    p = tmp_path / "scene.py"
    p.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "scene.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    original = {"scene.py": file_sha256(p)}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    assert committed_snapshot(tmp_path, original) == head
    p.write_text("VALUE = 2\n")
    assert committed_snapshot(tmp_path, {"scene.py": file_sha256(p)}) is None
    # A past recording can truthfully resolve to its committed old contents.
    assert committed_snapshot(tmp_path, original) == head
    assert committed_snapshot(tmp_path, {"untracked.py": file_sha256(p)}) is None


# Source group: test_linked_aggregate.py
import errno, os
import pytest

from utils.collection import linked_aggregate as m


def test_shared_episode_file_and_chunk_rotation():
    frame = {
        "videos/hand/chunk_index": [0, 0, 1, 1],
        "videos/hand/file_index": [0, 0, 0, 1],
        "from": [0.0, 1.0, 0.0, 2.0],
    }
    m.remap_pairs(
        frame, "videos/hand", {(0, 0): (3, 1), (1, 0): (4, 0), (1, 1): (4, 1)}
    )
    assert frame["videos/hand/chunk_index"] == [3, 3, 4, 4]
    assert frame["videos/hand/file_index"] == [1, 1, 0, 1]
    assert frame["from"] == [0.0, 1.0, 0.0, 2.0]
    assert [m.file_index(n, 2) for n in range(3)] == [(0, 0), (0, 1), (1, 0)]


def test_hardlink_survives_removing_original_and_preserves_bytes(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    video = src / "clip.mp4"
    video.write_bytes(b"original encoded bytes")
    dest = tmp_path / "out/clip.mp4"
    row = m.link_video(video, dest, src)
    m.verify_links([row])
    assert os.path.samefile(video, dest)
    video.unlink()
    assert dest.read_bytes() == b"original encoded bytes"


def test_existing_output_and_symlinks_are_never_overwritten(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    v = src / "clip.mp4"
    v.write_bytes(b"original")
    out = tmp_path / "dest.mp4"
    out.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        m.link_video(v, out, src)
    link = src / "symlink.mp4"
    link.symlink_to(v)
    with pytest.raises(ValueError):
        m.link_video(link, tmp_path / "new.mp4", src)
    assert out.read_bytes() == b"preserve" and v.read_bytes() == b"original"


def test_cross_filesystem_has_no_copy_fallback(tmp_path, monkeypatch):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"original")
    out = tmp_path / "out.mp4"

    def cross(*args, **kwargs):
        raise OSError(errno.EXDEV, "different filesystem")

    monkeypatch.setattr(m.os, "link", cross)
    with pytest.raises(OSError) as error:
        m.link_video(v, out, tmp_path)
    assert error.value.errno == errno.EXDEV and not out.exists()


def test_detects_source_modification(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"original")
    out = tmp_path / "out.mp4"
    row = m.link_video(v, out, tmp_path)
    v.write_bytes(b"changed bytes")
    with pytest.raises(ValueError):
        m.verify_links([row])


# Source group: test_merge_lerobot.py
import copy
import hashlib
import json

import pytest

from utils.collection.merge_lerobot import merge_provenance, select_episodes, validate_release
from utils.collection.provenance import dataset_provenance_complete


def batch(root, candidates, selected):
    return dict(
        control_hz=20,
        policy_hz=10,
        action_repeat=2,
        tokenizer_sha256="tokenizer",
        source_root=root,
        source_run=dict(seeds=candidates, signature=dict(code_sha256="same-code")),
        source_episode_outcomes=[
            dict(
                scene_seed=s,
                success=s in selected,
                status="success" if s in selected else "missed",
            )
            for s in candidates
        ],
        episodes=[
            dict(scene_seed=s, episode_index=i, success=True, truncated=False)
            for i, s in enumerate(selected)
        ],
    )


def test_merge_keeps_real_campaigns_and_failed_attempts():
    a, b = batch("/first", [1, 2], [1]), batch("/second", [3, 4, 5], [3, 5])
    merged = merge_provenance([a, b])
    assert merged["source_run"]["seeds"] == [1, 2]
    assert [s["run"]["seeds"] for s in merged["source_runs"]] == [[1, 2], [3, 4, 5]]
    assert [e["scene_seed"] for e in merged["source_episode_outcomes"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert sum(e["success"] for e in merged["source_episode_outcomes"]) == 3
    assert [e["episode_index"] for e in merged["episodes"]] == [0, 1, 2]
    assert [e["source_root"] for e in merged["episodes"]] == [
        "/first",
        "/second",
        "/second",
    ]
    assert "source_runs" not in a


def test_two_shards_count_one_campaign_only_once():
    a = batch("/same", [1, 2], [1, 2])
    b = copy.deepcopy(a)
    a["episodes"] = a["episodes"][:1]
    b["episodes"] = b["episodes"][1:]
    merged = merge_provenance([a, b])
    assert len(merged["source_runs"]) == 1
    assert len(merged["source_episode_outcomes"]) == 2


@pytest.mark.parametrize(
    "kind", ["duplicate", "runtime", "attempt_overlap", "outcome", "truncated"]
)
def test_invalid_inputs_are_rejected(kind):
    a, b = batch("/one", [1, 2], [1]), batch("/two", [3, 4], [3])
    if kind == "duplicate":
        b["episodes"][0]["scene_seed"] = 1
    elif kind == "runtime":
        b["source_run"]["signature"]["code_sha256"] = "different-code"
    elif kind == "attempt_overlap":
        b["source_run"]["seeds"] = [2, 3, 4]
    elif kind == "outcome":
        b = copy.deepcopy(a)
        b["episodes"] = []
        b["source_episode_outcomes"][0]["success"] = False
    else:
        b["episodes"][0]["truncated"] = True
    with pytest.raises(ValueError):
        merge_provenance([a, b])


def qualified_batch(root, start, stop, converter):
    metadata = batch(root, list(range(start, stop)), list(range(start, stop)))
    files = {"scene.py": "f" * 64}
    signature = dict(
        source_files=files,
        code_sha256=hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        profile=dict(robot_reference=dict(commit="a" * 40)),
    )
    metadata["source_run"]["signature"] = signature
    metadata["provenance"] = dict(
        source_signature_sha256=signature["code_sha256"],
        source_file_count=1,
        component_commits=dict(
            robot="a" * 40,
            environment="b" * 40,
            planner="b" * 40,
            converter=converter * 40,
        ),
        exporter_source_files={"converter.py": converter * 64},
    )
    return metadata


def test_nested_merge_keeps_each_original_converter():
    a = qualified_batch("/one", 1, 3, "c")
    b = qualified_batch("/two", 3, 5, "d")
    c = qualified_batch("/three", 5, 7, "e")
    ab = merge_provenance([a, b])
    merged = merge_provenance([ab, c])
    by_id = {c["id"]: c for c in merged["conversion_runs"]}
    original = [
        by_id[e["conversion_id"]]["provenance"]["component_commits"]["converter"]
        for e in merged["episodes"]
    ]
    assert original == ["c" * 40] * 2 + ["d" * 40] * 2 + ["e" * 40] * 2
    assert dataset_provenance_complete(merged)
    assert "conversion_id" not in a["episodes"][0]


@pytest.mark.parametrize(
    "missing",
    ["original_converter", "final_converter", "episode_binding", "source_manifest"],
)
def test_production_merge_cannot_bypass_commit_gate_with_small_batches(missing):
    a = qualified_batch("/one", 0, 500, "c")
    b = qualified_batch("/two", 500, 1000, "d")
    merged = merge_provenance([a, b])
    validate_release(merged, "release-1000ep")
    if missing == "original_converter":
        merged["conversion_runs"][1]["provenance"]["component_commits"][
            "converter"
        ] = None
    elif missing == "final_converter":
        merged["provenance"]["component_commits"]["converter"] = None
    elif missing == "episode_binding":
        merged["episodes"][-1]["conversion_id"] = "unknown"
    else:
        merged["source_run"]["signature"]["source_files"]["scene.py"] = "0" * 64
    assert not dataset_provenance_complete(merged)
    with pytest.raises(ValueError, match="A2"):
        validate_release(merged, "release-1000ep")


def test_legacy_pilot_is_not_a_production_release():
    merged = merge_provenance([batch("/one", [1, 2], [1])])
    validate_release(merged, "dataset-test-1ep")
    assert not dataset_provenance_complete(merged)
    with pytest.raises(ValueError, match="sub-1000"):
        validate_release(merged, "release")


def test_nested_merge_keeps_distinct_video_settings_for_same_converter():
    a, b = batch("/one", [1], [1]), batch("/two", [2], [2])
    a["video"] = {"profile": "reference", "gop": 2}
    b["video"] = {"profile": "compact", "gop": 12}
    a["episodes"][0]["video_encoding"] = {"fetch_hand": {"crf": 12}}
    b["episodes"][0]["video_encoding"] = {"fetch_hand": {"crf": 16}}
    merged = merge_provenance([merge_provenance([a]), b])
    by_id = {row["id"]: row for row in merged["conversion_runs"]}
    assert len(by_id) == 2
    for episode, expected in zip(merged["episodes"], [a, b]):
        assert by_id[episode["conversion_id"]]["video"] == expected["video"]
        assert episode["video_encoding"] == expected["episodes"][0]["video_encoding"]


def test_selection_keeps_every_attempt_and_renumbers_selected_episodes():
    merged = merge_provenance([batch("/first", [1, 2, 3], [1, 3]), batch("/second", [4, 5, 6], [4, 5, 6])])
    selected = select_episodes(merged, dict(rule="first three accepted", episode_seeds=[1, 3, 4]))
    assert [e["scene_seed"] for e in selected["episodes"]] == [1, 3, 4]
    assert [e["episode_index"] for e in selected["episodes"]] == [0, 1, 2]
    assert len(selected["source_episode_outcomes"]) == 6
    assert sum(e["success"] for e in selected["source_episode_outcomes"]) == 5
    assert selected["episode_selection"]["accepted_not_selected_seeds"] == [5, 6]
    assert len(merged["episodes"]) == 5


@pytest.mark.parametrize("seeds", [[1, 1], [2], [7]])
def test_selection_rejects_duplicates_and_unmerged_seeds(seeds):
    merged = merge_provenance([batch("/first", [1, 2, 3], [1, 3])])
    with pytest.raises(ValueError):
        select_episodes(merged, dict(rule="x", episode_seeds=seeds))


# Source group: test_normalization_stats.py
import json

import numpy as np
import pytest

from utils.collection.normalization_stats import (
    QUANTILES,
    read_episode_metadata,
    verify_normalization,
)


def metadata_fixture(root):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    shapes = {"action": [13], "observation.state": [12]}
    stats = {
        feature: {
            **{
                name: [0.0] * shape[0]
                for name in (*QUANTILES, "mean", "std", "min", "max")
            },
            "count": [8],
        }
        for feature, shape in shapes.items()
    }
    paths = root / "meta/episodes/chunk-000"
    paths.mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "features": {k: {"shape": v} for k, v in shapes.items()},
            }
        )
    )
    (root / "meta/stats.json").write_text(json.dumps(stats))
    records = [
        {
            "episode_index": episode,
            **{
                f"stats/{feature}/{q}": [float(episode)] * shape[0]
                for feature, shape in shapes.items()
                for q in QUANTILES
            },
        }
        for episode in [1, 0]
    ]
    file = paths / "file-000.parquet"
    pq.write_table(pa.Table.from_pylist(records), file)
    return file


def test_original_parquet_retains_quantiles_filtered_by_lerobot_runtime(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.utils import load_episodes

    metadata_fixture(tmp_path)
    runtime = load_episodes(tmp_path)
    assert not any(name.startswith("stats/") for name in runtime.column_names)
    full = read_episode_metadata(tmp_path)
    assert full["episode_index"].tolist() == [0, 1]
    np.testing.assert_array_equal(full["stats/action/q01"].iloc[1], np.ones(13))
    assert verify_normalization(tmp_path)["episode_quantile_columns"] == 10


@pytest.mark.parametrize("damage", ["missing_column", "null_value"])
def test_missing_episode_quantiles_fail_before_video_loading(tmp_path, damage):
    file = metadata_fixture(tmp_path)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(file)
    if damage == "missing_column":
        table = table.drop(["stats/action/q50"])
    else:
        records = table.to_pylist()
        records[0]["stats/action/q50"] = None
        table = pa.Table.from_pylist(records, schema=table.schema)
    pq.write_table(table, file)
    with pytest.raises(ValueError, match="G4"):
        verify_normalization(tmp_path)


# Source group: test_video_encoding.py
import numpy as np
import pytest

from utils.collection.video_encoding import encode_checked_video, measure_encoded_video


def write_frames(root, count):
    from PIL import Image

    root.mkdir()
    rng = np.random.default_rng(18)
    for index in range(count):
        # High-frequency, non-square images exercise quality and shape handling.
        pixels = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / f"frame-{index:06d}.png")


def test_real_codec_falls_back_without_losing_source_images(tmp_path):
    pytest.importorskip("lerobot")
    images, target = tmp_path / "images", tmp_path / "video.mp4"
    write_frames(images, 4)
    result = encode_checked_video(images, target, 10, crfs=(40, 0), gop=12)
    assert result["crf"] == 0
    assert result["trials"][0]["mean_absolute_error_255"] > 2
    assert result["mean_absolute_error_255"] <= 2
    assert result["frames"] == 4 and len(list(images.glob("*.png"))) == 4
    assert (
        measure_encoded_video(images, target)["decoded_rgb_sha256"]
        == result["decoded_rgb_sha256"]
    )
    previous = target.read_bytes()
    with pytest.raises(FileExistsError):
        encode_checked_video(images, target, 10)
    assert target.read_bytes() == previous


def test_real_decoder_detects_missing_frames(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.video_utils import encode_video_frames

    images, target = tmp_path / "images", tmp_path / "video.mp4"
    write_frames(images, 3)
    last = images / "frame-000002.png"
    hidden = images / "saved.png"
    last.rename(hidden)
    encode_video_frames(images, target, 10, vcodec="h264", pix_fmt="yuv444p", crf=0)
    hidden.rename(last)
    with pytest.raises(ValueError, match="missing source frames"):
        measure_encoded_video(images, target)
    assert len(list(images.glob("frame-*.png"))) == 3


def test_lerobot_reads_concatenated_episodes_with_different_fallback_crfs(
    tmp_path, monkeypatch
):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from utils.collection.contract import CAMERAS
    from utils.collection.export_lerobot import dataset_class
    from utils.collection.video_encoding import PROFILES

    # Force both encoding branches using real pixels, then test actual LeRobot
    # seeking across the boundary of two concatenated episode bitstreams.
    monkeypatch.setitem(PROFILES, "compact", {"crfs": (40, 0), "gop": 12})
    features = {
        "action": {"dtype": "float32", "shape": (13,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (12,), "names": None},
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
            for camera, shape in CAMERAS.items()
        },
    }
    root = tmp_path / "dataset"
    writer = dataset_class("compact").create(
        repo_id="local/codec-test",
        root=root,
        fps=10,
        robot_type="ds_fetch",
        features=features,
        use_videos=True,
        image_writer_threads=1,
        video_backend="pyav",
        vcodec="h264",
    )
    originals = []
    rng = np.random.default_rng(35)
    try:
        for episode in range(2):
            for _ in range(4):
                images = {
                    camera: (
                        np.zeros(shape, dtype=np.uint8)
                        if episode == 0
                        else rng.integers(0, 256, shape, dtype=np.uint8)
                    )
                    for camera, shape in CAMERAS.items()
                }
                originals.append(images)
                writer.add_frame(
                    {
                        "action": np.zeros(13, dtype=np.float32),
                        "observation.state": np.zeros(12, dtype=np.float32),
                        "task": "Test video episode boundaries.",
                        **{f"observation.images.{k}": v for k, v in images.items()},
                    }
                )
            writer.save_episode(parallel_encoding=False)
        encodings = writer._mikasa_video_encoding
    finally:
        writer.finalize()
        writer.stop_image_writer()
    assert {v["crf"] for v in encodings[0].values()} == {40}
    assert {v["crf"] for v in encodings[1].values()} == {0}
    reader = LeRobotDataset("local/codec-test", root=root, video_backend="pyav")
    assert len(reader) == 8
    for index in reversed(range(8)):
        sample = reader[index]
        for camera in CAMERAS:
            actual = np.rint(
                sample[f"observation.images.{camera}"].numpy().transpose(1, 2, 0) * 255
            ).astype(np.int16)
            assert (
                np.abs(actual - originals[index][camera].astype(np.int16)).mean() <= 2
            )
