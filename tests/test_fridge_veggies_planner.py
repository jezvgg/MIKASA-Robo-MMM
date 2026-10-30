"""CPU regression tests; run with `uv run python -m pytest tests/test_fridge_veggies_planner.py`."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import sapien
import torch

from planners import myrobocasa_fridge_veggies_planner as staged
from robots.fetch.ds_fetch import DSFetch


def test_scene_ready_pose_preserves_other_joints_and_unreset_rows():
    """Scene reset contract without depending on the optional collector CLI."""
    from my_scenes.my_robocasa_fridge_veggies import MyRoboCasaFridgeVeggies

    original = torch.arange(30, dtype=torch.float32).reshape(2, 15)
    arm_names = [f"arm_{i}" for i in range(7)]
    mapping = dict(zip(arm_names, [5, 7, 8, 9, 10, 11, 12]))
    mapping["torso_lift_joint"] = 3
    robot = SimpleNamespace(
        get_qpos=lambda: original,
        active_joints_map={
            name: SimpleNamespace(active_index=torch.tensor([index]))
            for name, index in mapping.items()
        },
        set_qpos=Mock(),
    )
    rest = np.arange(15, dtype=np.float32) + 100
    agent = SimpleNamespace(
        robot=robot,
        controller=SimpleNamespace(controllers={
            "arm": SimpleNamespace(config=SimpleNamespace(joint_names=arm_names))}),
        keyframes={"rest": SimpleNamespace(qpos=rest)},
    )
    MyRoboCasaFridgeVeggies._initialize_ready_pose(
        SimpleNamespace(agent=agent), torch.tensor([1]))
    expected = original.clone()
    for index in mapping.values():
        expected[1, index] = float(rest[index])
    torch.testing.assert_close(robot.set_qpos.call_args.args[0], expected)
    torch.testing.assert_close(original, torch.arange(30, dtype=torch.float32).reshape(2, 15))


def _box_vertices(extents):
    return (np.array([
        [x, y, z] for x in (-0.5, 0.5)
        for y in (-0.5, 0.5) for z in (-0.5, 0.5)
    ]) * np.asarray(extents))


def _navigation_env(position, yaw=np.pi / 2):
    pose = SimpleNamespace()

    def place(position, yaw):
        pose.p = torch.as_tensor(np.asarray(position), dtype=torch.float32).reshape(1, 3)
        pose.sp = sapien.Pose(
            p=position, q=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        )

    place(position, yaw)
    agent = Mock(base_link=SimpleNamespace(pose=pose))
    agent.build_grasp_pose.side_effect = DSFetch.build_grasp_pose
    agent.controller.controllers = {
        "arm": SimpleNamespace(qpos=torch.zeros((1, 7))),
        "body": SimpleNamespace(
            qpos=torch.tensor([[0.0, 0.0, 0.4]]),
            active_joint_indices=torch.tensor([11, 12, 3]),
        ),
    }
    agent.robot.get_qlimits.return_value = torch.tensor([[[0.0, 1.0]] * 15])
    agent.tcp = SimpleNamespace(
        pose=SimpleNamespace(
            p=torch.tensor([[2.416, -0.509, 1.02]]),
            q=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            sp=sapien.Pose(p=[2.416, -0.509, 1.02]),
        )
    )
    veggie = Mock(pose=SimpleNamespace(p=torch.tensor([[2.416, -0.509, 0.945]])))
    vertices = _box_vertices([0.08, 0.03, 0.06]) + [2.416, -0.509, 0.95]
    veggie.get_first_collision_mesh.return_value = SimpleNamespace(
        vertices=vertices,
        bounds=np.array([vertices.min(axis=0), vertices.max(axis=0)]),
    )
    scene = SimpleNamespace(
        agent=agent, veggies=[veggie], plate=Mock(),
        _picture_target=0, control_timestep=0.05,
    )
    env = Mock(unwrapped=scene)
    env.log_motion.side_effect = lambda _, fn, *args, **kwargs: fn(*args, **kwargs)
    planner = Mock(gripper_state=1.0, ARM_SCREW_GOAL_TOLERANCE=(0.02, 0.1))
    return env, planner, place


def _check_route(
    position, stalled=False, failed_final_turn=False, failed_hover=False,
    failed_axis=False, failed_grasp=False, failed_retreat=False,
    waypoint_offsets=None,
):
    env, planner, place = _navigation_env(position)
    if waypoint_offsets is None:
        waypoint_offsets = {
            "wp3_standoff_m": 0.0,
            "wp12_pot_radius_m": 0.0,
            "wp14_hover_clearance_m": 0.0,
        }
    veggie = env.unwrapped.veggies[0].pose.p[0].numpy()
    standoff = staged.VEGETABLE_APPROACH_STANDOFF_M + waypoint_offsets[
        "wp3_standoff_m"
    ]
    expected = np.array([veggie[0], veggie[1] - standoff, 0.0])
    turns = []
    drives = []
    lowering_actions = []

    def step(action):
        lowering_actions.append(action.copy())
        env.unwrapped.agent.controller.controllers["body"].qpos = torch.tensor(
            np.asarray(action[8:11]).reshape(1, 3), dtype=torch.float32
        )

    env.step.side_effect = step

    def turn(env, planner, target, **kwargs):
        current = env.unwrapped.agent.base_link.pose.sp.p.copy()
        np.testing.assert_allclose(target, expected, atol=1e-6)
        delta = target[:2] - current[:2]
        place(current, np.arctan2(delta[1], delta[0]))
        turns.append(np.array(target))

    def perpendicular_turn(direction):
        assert len(drives) == 1, "WP4 must follow the completed approach"
        assert len(lowering_actions) == 60, "lower torso before turning"
        np.testing.assert_array_equal(direction, [0.0, 1.0, 0.0])
        current = env.unwrapped.agent.base_link.pose.sp.p.copy()
        turns.append(np.array(direction))
        if failed_final_turn:
            return -1
        place(current, np.pi / 2)
        return 0

    planner.rotate_base_z.side_effect = perpendicular_turn
    hover_calls = []

    def hover(pose, **kwargs):
        assert len(turns) == 2 and len(drives) == 1
        assert kwargs["disable_lift_joint"] is True
        np.testing.assert_allclose(pose.p, [veggie[0], veggie[1], 1.08], atol=1e-6)
        np.testing.assert_allclose(
            pose.to_transformation_matrix()[:3, 2], [0.0, 0.0, -1.0], atol=1e-6
        )
        closing = pose.to_transformation_matrix()[:3, 1]
        if hover_calls:
            np.testing.assert_allclose(abs(closing), [0.0, 1.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(pose.p, hover_calls[0].p, atol=1e-6)
        else:
            np.testing.assert_allclose(closing, [1.0, 0.0, 0.0], atol=1e-6)
        hover_calls.append(pose)
        if failed_hover or (failed_axis and len(hover_calls) == 2):
            return -1
        env.unwrapped.agent.tcp.pose = SimpleNamespace(
            p=torch.tensor(np.asarray(pose.p).reshape(1, 3)),
            q=torch.tensor(np.asarray(pose.q).reshape(1, 4)),
            sp=pose,
        )
        return 0

    planner.static_manipulation.side_effect = hover

    def drive(env, planner, target):
        np.testing.assert_allclose(target, expected, atol=1e-6)
        np.testing.assert_allclose(target, turns[0], atol=1e-6)
        delta = target[:2] - np.asarray(position)[:2]
        place(target, np.arctan2(delta[1], delta[0]))
        drives.append(np.array(target))
        return {
            "initial_target_error_m": np.linalg.norm(delta),
            "final_target_error_m": 0.0,
            "final_remaining_m": 0.0,
            "cross_track_m": 0.0,
            "heading_drift_deg": 0.0,
            "steps": 100,
            "stalled": stalled,
        }

    def grasp_gate(*args):
        env.log_event("grasp_validation", success=not failed_grasp)
        return not failed_grasp

    def retreat_stage(*args):
        env.log_event("result", success=not failed_retreat)
        return not failed_retreat

    def pot_stage(*args):
        env.log_event("result", success=True)
        return True

    with (
        patch.object(staged, "_install_render_cameras"),
        patch.object(staged, "_pose_render_cameras"),
        patch.object(staged, "_ready_qpos", return_value=(None, np.zeros(15))),
        patch.object(staged, "FetchMotionPlanningSapienSolver", return_value=planner),
        patch.object(
            staged, "TrayEpisodeCheckers",
            return_value=Mock(summary=Mock(return_value={"checker_steps": 0})),
        ) as checker_factory,
        patch.object(
            staged, "_sample_waypoint_offsets", return_value=waypoint_offsets
        ),
        patch.object(staged, "_yaw_base_to", side_effect=turn),
        patch.object(staged, "_drive_straight_to_point", side_effect=drive),
        patch.object(staged, "_grasp_and_lift", side_effect=grasp_gate) as grasp,
        patch.object(staged, "_retreat_with_vegetable", side_effect=retreat_stage) as retreat,
        patch.object(staged, "_approach_pot", side_effect=pot_stage) as pot,
        patch.object(staged, "_face_pot", side_effect=pot_stage) as face,
        patch.object(staged, "_hand_over_pot", side_effect=pot_stage) as hand,
        patch.object(staged, "_finish_placement", side_effect=pot_stage) as finish,
    ):
        result = staged.planning(env, seed=3)
    checker_factory.assert_called_once_with(
        env.unwrapped, grasp_contact_objects=(env.unwrapped.veggies[0],)
    )
    checker = checker_factory.return_value
    env.add_step_observer.assert_called_once_with(checker.callback)
    env.remove_step_observer.assert_called_once_with(checker.callback)
    assert not hasattr(env.unwrapped, "_veggie_episode_checkers")
    assert planner.action_noise == staged.EXECUTION_ACTION_NOISE
    assert planner.noise_hold == staged.EXECUTION_NOISE_HOLD
    planner.set_execution_noise_seed.assert_called_once_with(3)
    assert planner.execution_noise_enabled is False
    assert len(drives) == 1
    assert len(turns) == (1 if stalled else 2)
    if not stalled:
        actions = np.asarray(lowering_actions)
        np.testing.assert_array_equal(actions[:, :7], 0.0)
        np.testing.assert_array_equal(actions[:, -2:], 0.0)
        assert np.all(np.diff(actions[:, 10]) < 0.0)
        np.testing.assert_allclose(actions[-1, 10], 0.18, atol=1e-6)
    else:
        assert not lowering_actions
    env.reset.assert_called_once_with(seed=3, options={"reconfigure": True})
    expected_motions = 0 if stalled or failed_final_turn else (1 if failed_hover else 2)
    assert len(hover_calls) == expected_motions
    reached_grasp = not (stalled or failed_final_turn or failed_hover or failed_axis)
    reached_retreat = reached_grasp and not failed_grasp
    expected_success = reached_retreat and not failed_retreat
    assert result == expected_success
    assert grasp.call_count == int(reached_grasp)
    assert retreat.call_count == int(reached_retreat)
    assert pot.call_count == int(expected_success)
    assert face.call_count == int(expected_success)
    assert hand.call_count == int(expected_success)
    assert finish.call_count == int(expected_success)
    result_events = [call for call in env.log_event.call_args_list if call.args[0] == "result"]
    assert result_events[-1].kwargs["success"] == expected_success
    assert env.log_event.call_args.args[0] == "safety_checkers"
    assert env.log_event.call_args.kwargs["checker_steps"] == 0
    if expected_success:
        final = env.unwrapped.agent.base_link.pose.sp
        np.testing.assert_allclose(final.p, expected, atol=1e-6)
        np.testing.assert_allclose(
            final.to_transformation_matrix()[:3, 0], [0.0, 1.0, 0.0], atol=1e-6
        )


def test_turn_and_drive_target_counter_front_not_robot_ray():
    for position in ((4.672, -1.869, 0.0), (3.8, -2.4, 0.0)):
        _check_route(position)


def test_wp3_uses_seeded_waypoint_standoff_offset():
    _check_route(
        (4.672, -1.869, 0.0),
        waypoint_offsets={
            "wp3_standoff_m": 0.02,
            "wp12_pot_radius_m": -0.01,
            "wp14_hover_clearance_m": 0.001,
        },
    )


def test_stall_cannot_be_reported_as_success():
    _check_route((4.672, -1.869, 0.0), stalled=True)


def test_wp4_failed_turn_cannot_be_reported_as_success():
    _check_route((4.672, -1.869, 0.0), failed_final_turn=True)


def test_wp5_failed_hover_cannot_be_reported_as_success():
    _check_route((4.672, -1.869, 0.0), failed_hover=True)


def test_wp6_failed_alignment_cannot_be_reported_as_success():
    _check_route((4.672, -1.869, 0.0), failed_axis=True)


def test_narrow_axis_uses_rotated_mesh_vertices_not_world_aabb():
    angle = np.deg2rad(40)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    vertices = _box_vertices([0.08, 0.03, 0.01]) @ rotation.T + [2.4, -0.5, 0.95]
    mesh = SimpleNamespace(vertices=vertices)
    # Neither world AABB axis is the true narrow closing direction.
    assert np.ptp(vertices[:, 0]) > 0.03
    assert np.ptp(vertices[:, 1]) > 0.03
    narrow = rotation[:, 1]
    np.testing.assert_allclose(staged._narrow_closing_axis(mesh, narrow), narrow)
    np.testing.assert_allclose(staged._narrow_closing_axis(mesh, -narrow), -narrow)


def test_narrow_axis_rejects_invalid_or_degenerate_vertices():
    for vertices in (
        np.empty((0, 3)),
        np.zeros((8, 2)),
        np.full((8, 3), np.nan),
        np.full((8, 3), np.inf),
        np.ones((8, 3)),
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
    ):
        mesh = SimpleNamespace(vertices=vertices)
        try:
            staged._narrow_closing_axis(mesh, [1, 0, 0])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid or degenerate mesh should not choose a grasp axis")


def test_narrow_axis_rolled_corn_uses_horizontal_transverse_width():
    yaw = np.deg2rad(27.0)
    roll = np.deg2rad(45.0)
    yaw_rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    roll_rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(roll), -np.sin(roll)],
        [0.0, np.sin(roll), np.cos(roll)],
    ])
    rotation = yaw_rotation @ roll_rotation
    # These are the actual box's 3D OBB short axes, both too tilted for
    # the former 30-degree filter, which left only the 137 mm long axis.
    assert np.all(np.abs(rotation[2, 1:]) > np.sin(np.deg2rad(30.0)))
    vertices = _box_vertices([0.137, 0.033, 0.032]) @ rotation.T + [2.4, -0.5, 0.95]
    mesh = SimpleNamespace(vertices=vertices)
    transverse = yaw_rotation[:, 1]
    for reference in (transverse, -transverse):
        axis = staged._narrow_closing_axis(mesh, reference)
        np.testing.assert_allclose(axis, reference, atol=1e-8)
        width = np.ptp(vertices @ axis)
        np.testing.assert_allclose(width, (0.033 + 0.032) / np.sqrt(2), atol=1e-8)
        assert width < 0.05
        assert width < np.ptp(vertices @ rotation[:, 0])


def _grasp_fixture(no_contact=False, lost_lift=False):
    env, planner, _ = _navigation_env((2.416, -1.009, 0.0))
    agent = env.unwrapped.agent
    veggie = env.unwrapped.veggies[0]
    env.unwrapped.counter_pos = np.array([1.5, -0.325, 0.46])
    env.unwrapped.counter_size = np.array([2.5, 0.65, 0.92])
    body = agent.controller.controllers["body"]
    body.qpos = torch.tensor([[0.0, 0.0, 0.2]])
    body.active_joint_indices = torch.tensor([11, 12, 3])
    agent.robot.get_qlimits.return_value = torch.tensor([[[0.0, 1.0]] * 15])
    agent.tcp.pose.p = torch.tensor([[2.416, -0.509, 1.1]])
    state = SimpleNamespace(closed=False)
    commands = []

    def bounds():
        z = float(agent.tcp.pose.p[0, 2])
        return np.array([[0, 0, z - 0.03], [0, 0, z + 0.03]])

    for name in ("finger1_link", "finger2_link"):
        setattr(agent, name, SimpleNamespace(_objs=[
            SimpleNamespace(get_global_aabb_fast=bounds)
        ]))

    def step(action):
        commands.append(action.copy())
        body.qpos = torch.tensor(np.asarray(action[8:11]).reshape(1, 3))
        torso = float(body.qpos[0, 2])
        agent.tcp.pose.p[0, 2] = 1.1 + torso - 0.2
        if state.closed and not (no_contact or lost_lift):
            veggie.pose.p[0, 2] = 0.945 + torso - state.close_torso

    def close(t):
        state.closed = True
        state.close_torso = float(body.qpos[0, 2])
        planner.gripper_state = -1.0
        return 0

    env.step.side_effect = step
    planner.close_gripper.side_effect = close
    agent.is_grasping.side_effect = lambda target: (
        target is veggie and state.closed and not no_contact
        and not (lost_lift and float(body.qpos[0, 2]) > state.close_torso + 0.001)
    )
    return env, planner, veggie, commands


def test_torso_grasp_and_lift_with_fixed_arm_and_base():
    env, planner, veggie, commands = _grasp_fixture()
    assert staged._grasp_and_lift(env, planner, veggie)
    actions = np.asarray(commands)
    assert len(actions) == 120
    np.testing.assert_array_equal(actions[:, :7], 0.0)
    np.testing.assert_array_equal(actions[:, -2:], 0.0)
    assert np.all(np.diff(actions[:60, 10]) < 0)
    assert np.all(np.diff(actions[60:, 10]) > 0)
    np.testing.assert_array_equal(actions[:60, 7], 1.0)
    np.testing.assert_array_equal(actions[60:, 7], -1.0)
    planner.close_gripper.assert_called_once_with(t=20)
    assert env.log_event.call_args.kwargs["vegetable_lift_m"] > 0.10


def test_no_lift_without_gripper_contact():
    env, planner, veggie, commands = _grasp_fixture(no_contact=True)
    assert not staged._grasp_and_lift(env, planner, veggie)
    assert len(commands) == 60
    assert env.log_event.call_args.kwargs["failed_waypoint"] == 8


def test_dropped_object_cannot_be_reported_as_lifted():
    env, planner, veggie, _ = _grasp_fixture(lost_lift=True)
    assert not staged._grasp_and_lift(env, planner, veggie)
    assert env.log_event.call_args.kwargs["vegetable_lift_m"] == 0.0


def test_torso_descent_respects_joint_limit():
    env, planner, veggie, commands = _grasp_fixture()
    env.unwrapped.agent.controller.controllers["body"].qpos[0, 2] = 0.05
    assert not staged._grasp_and_lift(env, planner, veggie)
    assert not commands
    planner.close_gripper.assert_not_called()


def test_grasp_gate_rejects_slip_even_with_contact_and_lift():
    env, planner, veggie, commands = _grasp_fixture()
    original = env.step.side_effect

    def step(action):
        original(action)
        if len(commands) == 61:
            veggie.pose.p[0, 0] += 0.05

    env.step.side_effect = step
    assert not staged._grasp_and_lift(env, planner, veggie)
    final = env.log_event.call_args.kwargs
    assert final["grasped"] and final["vegetable_lift_m"] > 0.10
    assert "vegetable_slipped" in final["failure_reasons"]
    assert final["failed_waypoint"] == 9


def test_failed_grasp_gate_prevents_retreat():
    _check_route((4.672, -1.869, 0.0), failed_grasp=True)


def _retreat_fixture(lose_contact=False):
    env, planner, place = _navigation_env((0.0, 0.0, 0.0), yaw=0.0)
    agent = env.unwrapped.agent
    veggie = env.unwrapped.veggies[0]
    agent.is_grasping.return_value = True
    planner.gripper_state = -1.0
    commands = []

    def step(action):
        commands.append(action.copy())
        delta = float(action[-2]) * env.unwrapped.control_timestep
        position = agent.base_link.pose.sp.p.copy()
        position[0] += delta
        place(position, 0.0)
        agent.tcp.pose.p[0, 0] += delta
        veggie.pose.p[0, 0] += delta
        if lose_contact and len(commands) >= 5:
            agent.is_grasping.return_value = False

    env.step.side_effect = step
    return env, planner, veggie, commands


def test_retreat_is_smooth_reverse_without_yaw():
    env, planner, veggie, commands = _retreat_fixture()
    assert staged._retreat_with_vegetable(env, planner, veggie)
    actions = np.asarray(commands)
    assert np.all(actions[:, -2] <= 0.0)
    np.testing.assert_array_equal(actions[:, -1], 0.0)
    speeds = np.r_[0.0, actions[:, -2], 0.0]
    assert np.max(np.abs(np.diff(speeds))) <= 0.20 * 0.05 + 1e-8
    np.testing.assert_array_equal(actions[:, 7], -1.0)
    assert abs(env.log_event.call_args.kwargs["retreat_distance_m"] - 0.20) <= 0.003 + 1e-8


def test_retreat_stops_when_target_contact_is_lost():
    env, planner, veggie, commands = _retreat_fixture(lose_contact=True)
    assert not staged._retreat_with_vegetable(env, planner, veggie)
    final = env.log_event.call_args.kwargs
    assert final["stopped_on_grasp_loss"]
    assert final["retreat_distance_m"] < 0.03
    assert len(commands) < 20


def test_retreat_never_starts_without_target_contact():
    env, planner, veggie, commands = _retreat_fixture()
    env.unwrapped.agent.is_grasping.return_value = False
    assert not staged._retreat_with_vegetable(env, planner, veggie)
    assert not commands


def test_failed_retreat_prevents_pot_approach():
    _check_route((4.672, -1.869, 0.0), failed_retreat=True)


def _pot_fixture(failed_turn=False, lose_on_turn=False, lose_on_drive=False):
    env, planner, place = _navigation_env((0.0, 0.0, 0.0), yaw=0.0)
    agent = env.unwrapped.agent
    veggie = env.unwrapped.veggies[0]
    agent.is_grasping.return_value = True
    planner.gripper_state = -1.0
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    xy = np.c_[-0.8 + 0.1 * np.cos(angles), 0.2 + 0.1 * np.sin(angles)]
    vertices = np.r_[np.c_[xy, np.full(48, 0.92)], np.c_[xy, np.full(48, 1.0)]]
    plate = Mock(pose=SimpleNamespace(
        p=torch.tensor([[-0.65, 0.2, 0.96]]),
        q=torch.tensor([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]]),
    ))
    plate.get_first_collision_mesh.return_value = SimpleNamespace(
        vertices=vertices, bounds=np.array([vertices.min(axis=0), vertices.max(axis=0)])
    )
    env.unwrapped.plate = plate
    commands = []

    def turn(direction):
        offsets = getattr(planner, "waypoint_offsets", None)
        if not isinstance(offsets, dict):
            offsets = {}
        approach_radius = staged.POT_APPROACH_RADIUS_M + offsets.get(
            "wp12_pot_radius_m", 0.0
        )
        approach_direction = np.array([
            0.25,
            -np.sqrt(staged.POT_APPROACH_RADIUS_M**2 - 0.25**2),
        ])
        approach_direction /= np.linalg.norm(approach_direction)
        expected = np.r_[
            np.array([-0.8, 0.2]) + approach_direction * approach_radius,
            0.0,
        ]
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(direction, expected, atol=1e-6)
        if failed_turn:
            return -1
        position = agent.base_link.pose.sp.p.copy()
        yaw = np.arctan2(direction[1], direction[0])
        place(position, yaw)
        agent.tcp.pose.sp = sapien.Pose(
            p=agent.tcp.pose.p[0].numpy(),
            q=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
        )
        if lose_on_turn:
            agent.is_grasping.return_value = False
        return 0

    def step(action):
        commands.append(action.copy())
        agent.controller.controllers["body"].qpos = torch.tensor(
            np.asarray(action[8:11]).reshape(1, 3)
        )
        base = agent.base_link.pose.sp
        heading = base.to_transformation_matrix()[:3, 0]
        delta = action[-2] * env.unwrapped.control_timestep * heading
        yaw = np.arctan2(heading[1], heading[0])
        place(base.p + delta, yaw)
        agent.tcp.pose.p[0] += torch.tensor(delta)
        veggie.pose.p[0] += torch.tensor(delta)
        if lose_on_drive and len(commands) >= 5:
            agent.is_grasping.return_value = False

    planner.rotate_base_z.side_effect = turn
    env.step.side_effect = step
    return env, planner, veggie, commands


def test_pot_bowl_center_ignores_handle_offset_and_rotation():
    env, _, _, _ = _pot_fixture()
    np.testing.assert_allclose(
        staged._pot_bowl_center_world(env.unwrapped.plate), [-0.8, 0.2, 0.96], atol=1e-6
    )


def test_pot_approach_one_turn_then_straight_drive_with_payload():
    env, planner, veggie, commands = _pot_fixture()
    with patch.object(staged, "_attach_object") as attach:
        assert staged._approach_pot(env, planner, veggie)
    attach.assert_called_once_with(planner, veggie)
    planner.rotate_base_z.assert_called_once()
    actions = np.asarray(commands)
    assert np.all(actions[:, -2] >= 0.0)
    np.testing.assert_array_equal(actions[:, -1], 0.0)
    np.testing.assert_array_equal(actions[:, 7], -1.0)
    assert env.log_event.call_args.kwargs["approach_point_error_m"] <= 0.004


def test_pot_approach_uses_seeded_radial_waypoint_offset():
    env, planner, veggie, _ = _pot_fixture()
    planner.waypoint_offsets = {"wp12_pot_radius_m": 0.02}
    with patch.object(staged, "_attach_object"):
        assert staged._approach_pot(env, planner, veggie)

    start_event = next(
        call for call in env.log_event.call_args_list
        if call.args[1] == "Waypoint 12: straight drive to pot approach point"
    )
    assert start_event.kwargs["target_standoff_m"] == 0.52


def test_pot_approach_never_drives_after_refused_turn():
    env, planner, veggie, commands = _pot_fixture(failed_turn=True)
    with patch.object(staged, "_attach_object"):
        assert not staged._approach_pot(env, planner, veggie)
    assert not commands
    assert env.log_event.call_args.kwargs["failed_waypoint"] == 11


def test_pot_approach_never_drives_after_grasp_lost_on_turn():
    env, planner, veggie, commands = _pot_fixture(lose_on_turn=True)
    with patch.object(staged, "_attach_object"):
        assert not staged._approach_pot(env, planner, veggie)
    assert not commands
    assert env.log_event.call_args.kwargs["failed_waypoint"] == 11


def test_pot_approach_stops_on_grasp_loss_during_drive():
    env, planner, veggie, commands = _pot_fixture(lose_on_drive=True)
    with patch.object(staged, "_attach_object"):
        assert not staged._approach_pot(env, planner, veggie)
    assert len(commands) < 20
    assert env.log_event.call_args.kwargs["stopped_on_grasp_loss"]
    assert env.log_event.call_args.kwargs["failed_waypoint"] == 12


def _face_pot_fixture(failed_turn=False, lost_grasp=False):
    env, planner, veggie, commands = _pot_fixture()

    def turn(direction):
        expected = np.array([-0.8, 0.2, 0.0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(direction, expected, atol=1e-6)
        if failed_turn:
            return -1
        base = env.unwrapped.agent.base_link.pose
        yaw = np.arctan2(direction[1], direction[0])
        base.sp = sapien.Pose(
            p=base.p[0].numpy(), q=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        )
        env.unwrapped.agent.tcp.pose.sp = base.sp
        if lost_grasp:
            env.unwrapped.agent.is_grasping.return_value = False
        return 0

    planner.rotate_base_z.side_effect = turn
    return env, planner, veggie, commands


def test_face_pot_bowl_centre_without_translation():
    env, planner, veggie, commands = _face_pot_fixture()
    assert staged._face_pot(env, planner, veggie)
    assert len(commands) == 60  # smooth 4 cm torso pre-lift before turning
    actions = np.asarray(commands)
    np.testing.assert_array_equal(actions[:, :7], 0.0)
    np.testing.assert_array_equal(actions[:, -2:], 0.0)
    np.testing.assert_allclose(actions[-1, 10], 0.44, atol=1e-6)
    assert env.log_event.call_args.kwargs["base_drift_m"] == 0.0
    planner.rotate_base_z.assert_called_once()


def test_face_pot_refused_turn_cannot_succeed():
    env, planner, veggie, commands = _face_pot_fixture(failed_turn=True)
    assert not staged._face_pot(env, planner, veggie)
    assert len(commands) == 60
    assert env.log_event.call_args.kwargs["failed_waypoint"] == 13


def test_face_pot_rejects_lost_grasp():
    env, planner, veggie, _ = _face_pot_fixture(lost_grasp=True)
    assert not staged._face_pot(env, planner, veggie)
    assert not env.log_event.call_args.kwargs["grasped"]


def _payload_hover_fixture(failed_move=False, lost_grasp=False, hover_offset_m=0.0):
    env, planner, veggie, commands = _pot_fixture()
    planner.execution_noise_enabled = False
    planner.waypoint_offsets = {"wp14_hover_clearance_m": hover_offset_m}
    env.execution_noise_states = []
    env.hover_targets = []
    agent = env.unwrapped.agent
    initial = veggie.pose.p[0].numpy().copy()
    bounds_before = np.array([[2.4, -0.53, 0.92], [2.43, -0.48, 0.98]])
    veggie.get_first_collision_mesh.side_effect = lambda **kwargs: SimpleNamespace(
        bounds=bounds_before + veggie.pose.p[0].numpy() - initial
    )

    def finger_bounds():
        z = float(agent.tcp.pose.p[0, 2])
        return np.array([[0, 0, z - 0.03], [0, 0, z + 0.03]])

    for name in ("finger1_link", "finger2_link"):
        setattr(agent, name, SimpleNamespace(_objs=[
            SimpleNamespace(get_global_aabb_fast=finger_bounds)
        ]))

    def hover(pose, **kwargs):
        env.execution_noise_states.append(planner.execution_noise_enabled)
        env.hover_targets.append(np.array(pose.p, copy=True))
        assert kwargs["disable_lift_joint"] is True
        np.testing.assert_array_equal(pose.q, agent.tcp.pose.sp.q)
        if failed_move:
            return -1
        delta = pose.p - agent.tcp.pose.p[0].numpy()
        veggie.pose.p[0] += torch.tensor(delta)
        agent.tcp.pose.p = torch.tensor(np.asarray(pose.p).reshape(1, 3))
        agent.tcp.pose.sp = pose
        if lost_grasp:
            agent.is_grasping.return_value = False
        return 0

    planner.static_manipulation.side_effect = hover
    return env, planner, veggie, commands


def test_payload_hover_centres_vegetable_with_rim_clearance():
    env, planner, veggie, commands = _payload_hover_fixture(hover_offset_m=0.002)
    with patch.object(staged, "_detach_object") as detach, \
         patch.object(staged, "_attach_object") as attach:
        assert staged._hand_over_pot(env, planner, veggie)
    assert detach.call_count == 2
    assert attach.call_count == 2
    detach.assert_called_with(planner)
    attach.assert_called_with(planner, veggie)
    np.testing.assert_allclose(veggie.pose.p[0, :2], [-0.8, 0.2], atol=1e-6)
    assert not commands
    planner.open_gripper.assert_not_called()
    assert planner.static_manipulation.call_count == 2
    assert env.execution_noise_states == [True, False]
    assert planner.execution_noise_enabled is False
    np.testing.assert_allclose(
        env.hover_targets[0][:2], env.hover_targets[1][:2], atol=1e-6
    )
    np.testing.assert_allclose(
        env.hover_targets[0][2] - env.hover_targets[1][2], 0.002, atol=1e-6
    )
    assert env.log_event.call_args.kwargs["rim_clearance_m"] >= 0.05
    recovery_event = next(
        call for call in env.log_event.call_args_list
        if call.args[1] == "Waypoint 14b: replan from measured state to nominal hover"
    )
    assert recovery_event.kwargs["previous_motion_success"] is True
    assert "previous_motion_result" not in recovery_event.kwargs


def test_payload_hover_failed_motion_cannot_succeed():
    env, planner, veggie, commands = _payload_hover_fixture(failed_move=True)
    with patch.object(staged, "_detach_object"), patch.object(staged, "_attach_object"):
        assert not staged._hand_over_pot(env, planner, veggie)
    assert not commands
    assert env.log_event.call_args.kwargs["failed_waypoint"] == 14


def test_payload_hover_rejects_lost_grasp():
    env, planner, veggie, _ = _payload_hover_fixture(lost_grasp=True)
    with patch.object(staged, "_detach_object"), patch.object(staged, "_attach_object"):
        assert not staged._hand_over_pot(env, planner, veggie)
    assert not env.log_event.call_args.kwargs["grasped"]


def test_payload_hover_never_starts_without_grasp():
    env, planner, veggie, _ = _payload_hover_fixture()
    env.unwrapped.agent.is_grasping.return_value = False
    assert not staged._hand_over_pot(env, planner, veggie)
    planner.static_manipulation.assert_not_called()


def test_veggie_waypoint_offsets_are_seeded_and_bounded():
    first = staged._sample_waypoint_offsets(23)
    replay = staged._sample_waypoint_offsets(23)
    other_seed = staged._sample_waypoint_offsets(24)
    scales = {
        "wp3_standoff_m": staged.VEGETABLE_APPROACH_STANDOFF_M,
        "wp12_pot_radius_m": staged.POT_APPROACH_RADIUS_M,
        "wp14_hover_clearance_m": staged.WP14_HOVER_CLEARANCE_M,
    }

    assert first == replay
    assert first != other_seed
    for name, scale in scales.items():
        assert abs(first[name]) <= staged.WAYPOINT_NOISE * scale


def _placement_fixture(failed_release=False, failed_evaluation=False):
    env, planner, veggie, commands = _payload_hover_fixture()
    agent = env.unwrapped.agent
    with patch.object(staged, "_detach_object"), patch.object(staged, "_attach_object"):
        assert staged._hand_over_pot(env, planner, veggie)
    body = agent.controller.controllers["body"]
    body.qpos = torch.tensor([[0.0, 0.0, 0.2]])
    body.active_joint_indices = torch.tensor([11, 12, 3])
    agent.robot.get_qlimits.return_value = torch.tensor([[[0.0, 1.0]] * 15])
    state = SimpleNamespace(closed=True)

    def step(action):
        commands.append(action.copy())
        change = float(action[10]) - float(body.qpos[0, 2])
        body.qpos = torch.tensor(np.asarray(action[8:11]).reshape(1, 3))
        agent.tcp.pose.p[0, 2] += change
        if state.closed:
            veggie.pose.p[0, 2] += change
        else:
            veggie.pose.p[0, 2] = 0.947

    def release(t, ramp):
        if not failed_release:
            state.closed = False
            planner.gripper_state = 1.0
        return 0

    env.step.side_effect = step
    planner.open_gripper.side_effect = release
    agent.is_grasping.side_effect = lambda target: state.closed
    env.unwrapped.evaluate = Mock(side_effect=lambda: {
        "success": torch.tensor([not failed_evaluation and not state.closed])
    })
    return env, planner, veggie, commands


def test_placement_lowers_releases_raises_and_checks_environment():
    env, planner, veggie, commands = _placement_fixture()
    with patch.object(staged, "_detach_object") as detach:
        assert staged._finish_placement(env, planner, veggie)
    detach.assert_called_once_with(planner)
    planner.open_gripper.assert_called_once_with(t=20, ramp=15)
    actions = np.asarray(commands)
    assert len(actions) == 124  # 60-step raise plus 4 steps after the initial stable sample
    np.testing.assert_array_equal(actions[:, -2:], 0.0)
    np.testing.assert_array_equal(actions[:60, 7], -1.0)
    np.testing.assert_array_equal(actions[60:, 7], 1.0)
    assert np.all(np.diff(actions[:60, 10]) < 0)
    assert np.all(np.diff(actions[60:120, 10]) > 0)
    np.testing.assert_allclose(actions[120:, 10], 0.2, atol=1e-6)
    assert env.log_event.call_args.kwargs["task_success"]
    assert env.log_event.call_args.kwargs["stable_success_steps"] >= 20


def test_placement_cannot_succeed_when_environment_rejects_it():
    env, planner, veggie, _ = _placement_fixture(failed_evaluation=True)
    with patch.object(staged, "_detach_object"):
        assert not staged._finish_placement(env, planner, veggie)
    assert not env.log_event.call_args.kwargs["task_success"]


def test_placement_never_raises_before_release():
    env, planner, veggie, commands = _placement_fixture(failed_release=True)
    with patch.object(staged, "_detach_object"):
        assert not staged._finish_placement(env, planner, veggie)
    assert len(commands) == 60
    assert env.log_event.call_args.kwargs["failed_waypoint"] == 16


def test_placement_respects_torso_lower_limit():
    env, planner, veggie, commands = _placement_fixture()
    env.unwrapped.agent.controller.controllers["body"].qpos[0, 2] = 0.005
    assert not staged._finish_placement(env, planner, veggie)
    assert not commands
    planner.open_gripper.assert_not_called()


def test_straight_drive_smooth_and_no_yaw():
    env, planner, place = _navigation_env((0.0, 0.0, 0.0), yaw=0.0)
    commands = []

    def step(action):
        commands.append(action.copy())
        position = env.unwrapped.agent.base_link.pose.sp.p.copy()
        position[0] += action[-2] * env.unwrapped.control_timestep
        place(position, 0.0)

    env.step.side_effect = step
    result = staged._drive_straight_to_point(env, planner, [1.0, 0.0, 0.0])
    actions = np.asarray(commands)
    speeds = np.r_[0.0, actions[:, -2], 0.0]
    np.testing.assert_array_equal(actions[:, -1], 0.0)
    assert np.all(speeds >= 0.0)
    assert np.max(np.abs(np.diff(speeds))) <= 0.35 * 0.05 + 1e-8
    assert np.all(actions[:-1, -2] > 0.0), "intermediate stops in straight drive"
    assert result["final_target_error_m"] <= 0.02
    assert result["cross_track_m"] <= 1e-6
    assert not result["stalled"]


def test_small_velocity_deadband_does_not_cause_false_stall():
    env, planner, place = _navigation_env((0.0, 0.0, 0.0), yaw=0.0)

    def step(action):
        position = env.unwrapped.agent.base_link.pose.sp.p.copy()
        if action[-2] >= 0.001:
            position[0] += action[-2] * env.unwrapped.control_timestep
        place(position, 0.0)

    env.step.side_effect = step
    result = staged._drive_straight_to_point(env, planner, [1.0, 0.0, 0.0])
    assert result["final_target_error_m"] <= 0.02
    assert not result["stalled"]


def test_zero_distance_needs_no_drive():
    env, planner, _ = _navigation_env((0.0, 0.0, 0.0), yaw=0.0)
    result = staged._drive_straight_to_point(env, planner, [0.0, 0.0, 0.0])
    env.step.assert_not_called()
    assert result["final_target_error_m"] == 0.0
    assert result["cross_track_m"] == 0.0
    assert not result["stalled"]


def _torso_hold_fixture(move=True):
    body = SimpleNamespace(qpos=torch.tensor([[0.12, -0.05, 0.198]], dtype=torch.float64))
    arm = SimpleNamespace(qpos=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=torch.float64))
    agent = SimpleNamespace(controller=SimpleNamespace(controllers={"arm": arm, "body": body}))
    actions = []
    events = []

    def step(action):
        actions.append(action.copy())
        if move:
            body.qpos[0, 2] += 0.5 * (action[10] - body.qpos[0, 2])
        arm.qpos.add_(0.01)
        body.qpos[0, 0] += 0.01

    env = SimpleNamespace(
        unwrapped=SimpleNamespace(agent=agent),
        step=step,
        log_event=lambda *args, **kwargs: events.append((args, kwargs)),
    )
    planner = SimpleNamespace(
        gripper_state=1,
        planner=SimpleNamespace(update_from_simulation=lambda: None),
    )
    return env, planner, actions, events


def test_fixed_goal_torso_hold_survives_measured_state_changes():
    env, planner, actions, events = _torso_hold_fixture()
    result = staged._hold_torso_absolute(env, planner, 0.21)
    assert result["converged"] and result["steps"] < 80
    assert len(actions) == result["steps"]
    for action in actions:
        assert action.shape == (13,)
        np.testing.assert_allclose(action[:7], np.arange(1, 8) * 0.1)
        np.testing.assert_allclose(action[8:11], [0.12, -0.05, 0.21])
        assert action[7] == 1 and np.array_equal(action[11:], [0, 0])
    assert events[-1][1]["error_m"] <= 0.002


def test_torso_hold_is_bounded_when_controller_does_not_move():
    env, planner, actions, _ = _torso_hold_fixture(move=False)
    result = staged._hold_torso_absolute(env, planner, 0.21, max_steps=7)
    assert len(actions) == result["steps"] == 7
    assert not result["converged"] and result["error_m"] > 0.01


def test_canonical_entrypoint_replaces_legacy_algorithm():
    assert staged.__name__ == "planners.myrobocasa_fridge_veggies_planner"
    assert not hasattr(staged, "_reach_and_grasp")
    assert not hasattr(staged, "_transport_veg")


@pytest.mark.parametrize("outcome", [True, False, RuntimeError("episode failed")])
def test_safety_diagnostics_finalize_on_every_planner_exit(outcome):
    task = SimpleNamespace()
    env = Mock(unwrapped=task)
    checks = Mock(summary=Mock(return_value={"checker_steps": 17}))

    def episode(*args, **kwargs):
        task._veggie_episode_checkers = checks
        env.add_step_observer(checks.callback)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with patch.object(staged, "_planning_episode", side_effect=episode):
        if isinstance(outcome, Exception):
            with pytest.raises(RuntimeError, match="episode failed"):
                staged.planning(env, 201)
        else:
            assert staged.planning(env, 201) is outcome
    env.remove_step_observer.assert_called_once_with(checks.callback)
    assert env.log_event.call_count == 2
    result_call, safety_call = env.log_event.call_args_list
    assert result_call.args == ("result", "Veggie planner final result")
    assert result_call.kwargs["success"] is (outcome if isinstance(outcome, bool) else False)
    assert result_call.kwargs["seed"] == 201
    expected_error = "RuntimeError: episode failed" if isinstance(outcome, Exception) else None
    assert result_call.kwargs["error"] == expected_error
    assert safety_call.args == ("safety_checkers", "Episode safety-checker results")
    assert safety_call.kwargs == {"checker_steps": 17}
    assert not hasattr(task, "_veggie_episode_checkers")


def test_episode_checkers_are_finalized_after_exception():
    env = Mock(unwrapped=SimpleNamespace())
    checker = Mock(summary=Mock(return_value={"checker_steps": 12}))

    def failing_episode(env, seed, debug=False):
        env.unwrapped._veggie_episode_checkers = checker
        raise RuntimeError("planning failed")

    with patch.object(staged, "_planning_episode", side_effect=failing_episode):
        with pytest.raises(RuntimeError, match="planning failed"):
            staged.planning(env, seed=201)
    env.remove_step_observer.assert_called_once_with(checker.callback)
    assert env.log_event.call_args.args[0] == "safety_checkers"
    assert env.log_event.call_args.kwargs["checker_steps"] == 12
    assert not hasattr(env.unwrapped, "_veggie_episode_checkers")
