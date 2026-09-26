"""Guard the policy boundary and the timing semantics that affect learning."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from .client import execute_actions, policy_metadata, policy_observation
from .contract import CAMERAS, check_actions, held_actions
from .pipeline import NonReplayableMotion, action_only_oracle, initial_state_matches
from .profile import class_methods_sha, json_value


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
