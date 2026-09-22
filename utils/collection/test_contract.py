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


def test_cabinet_robot_adapter_keeps_finger_counter_contact():
    """The old all-link ignore mask makes the penetrating finger fall through."""
    import sapien
    from my_scenes.cabinet_search import CabinetSearchTask

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
    CabinetSearchTask._fix_ds_fetch_collision_bits(env)
    scene.step()
    contacts = [contact for contact in system.get_contacts()
                if finger in contact.bodies and counter in contact.bodies]
    assert contacts, "The scene adapter disabled finger contact with the counter"
    assert any(np.linalg.norm(point.impulse) > 0 for c in contacts for point in c.points)
