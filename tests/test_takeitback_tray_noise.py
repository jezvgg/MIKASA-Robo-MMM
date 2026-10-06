from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import sapien

from planners.myrobocasa_takeitback_tray_planner import (
    READY_ARM_POSTURE,
    WAYPOINT_NOISE,
    _choose_shortest_safe_plan,
    _execution_metrics,
    _plan_straight_arm_translation,
    _pregrasp_drive_distance,
    _torso_height_target,
    _waypoint_jitter,
)
from robots.fetch.extand import FetchMotionPlanningSapienSolver


def _noise_state(seed):
    state = SimpleNamespace(
        action_noise=0.001,
        noise_hold=2,
        _execution_noise_rng=np.random.default_rng(),
        _execution_noise=None,
        _execution_noise_steps=0,
    )
    state._reset_execution_noise = lambda: (
        FetchMotionPlanningSapienSolver._reset_execution_noise(state)
    )
    FetchMotionPlanningSapienSolver.set_execution_noise_seed(state, seed)
    return state


def _straight_planner(collisions=0):
    tcp = sapien.Pose([2.49, -0.67, 0.98], [0.5, -0.5, 0.5, -0.5])
    qpos = np.linspace(0.0, 0.2, 15)[None, :]
    calls = []
    result = {"status": "Success", "position": np.zeros((2, 15))}

    def screw(target, current_qpos, **kwargs):
        calls.append((target, current_qpos.copy(), kwargs))
        return result

    def rrt(*args, **kwargs):
        raise AssertionError("Straight cup approach must not use RRT")

    agent = SimpleNamespace(
        tcp=SimpleNamespace(pose=SimpleNamespace(sp=tcp)),
        robot=SimpleNamespace(get_qpos=lambda: SimpleNamespace(
            cpu=lambda: SimpleNamespace(numpy=lambda: qpos)
        )),
    )
    planner = SimpleNamespace(
        base_env=SimpleNamespace(agent=agent, control_timestep=0.05),
        planner=SimpleNamespace(plan_screw=screw, plan_pose=rrt),
        path_env_collisions=lambda position: collisions,
        ARM_SCREW_GOAL_TOLERANCE=(0.02, 0.10),
    )
    return planner, calls, result


def test_front_cup_approach_preserves_measured_orientation_and_fixed_body():
    planner, calls, result = _straight_planner()
    # The actual front pre-grasp and grasp share one wrist orientation.
    for target in ([2.49, -0.67, 0.98], [2.49, -0.55, 0.98]):
        selected, details = _plan_straight_arm_translation(planner, target)
        assert selected is result
        pose, qpos, kwargs = calls[-1]
        np.testing.assert_allclose(pose.p, target)
        np.testing.assert_allclose(pose.q, planner.base_env.agent.tcp.pose.sp.q)
        np.testing.assert_allclose(qpos, np.linspace(0.0, 0.2, 15))
        assert kwargs["masked_joints"] == [False] * 4 + [True] * 11
        assert kwargs["time_step"] == 0.05
        assert kwargs["goal_tolerance"] == (0.02, 0.10)
        assert details["ik_screw"]["collisions"] == 0
    assert len(calls) == 2


def test_unsafe_straight_approach_is_rejected_without_rrt_rescue():
    for collisions in (1, -1):
        planner, calls, _ = _straight_planner(collisions)
        selected, details = _plan_straight_arm_translation(planner, [2.49, -0.55, 0.98])
        assert selected is None
        assert details["ik_screw"]["collisions"] == collisions
        assert len(calls) == 1


def test_ready_pose_raises_hand_without_tilting_it_and_keeps_elbow_up():
    old = np.array([0.0, 1.31, 0.0, -2.09, 0.0, 0.79, 0.0])
    urdf = ElementTree.parse(Path(__file__).parents[1] / "robots/fetch/fetch.urdf")

    def offset(joint):
        origin = urdf.find(f".//joint[@name='{joint}']/origin")
        return float(origin.attrib["xyz"].split()[0])

    lengths = np.array(
        [
            offset("upperarm_roll_joint") + offset("elbow_flex_joint"),
            offset("forearm_roll_joint") + offset("wrist_flex_joint"),
            offset("wrist_roll_joint") + offset("gripper_axis"),
        ]
    )

    def hand_position(arm):
        angles = np.cumsum(arm[[1, 3, 5]])
        return np.array([lengths @ np.cos(angles), -lengths @ np.sin(angles)])

    # Unchanged pan/roll and total pitch preserve the full gripper orientation.
    np.testing.assert_array_equal(READY_ARM_POSTURE[[0, 2, 4, 6]], old[[0, 2, 4, 6]])
    np.testing.assert_allclose(
        READY_ARM_POSTURE[[1, 3, 5]].sum(), old[[1, 3, 5]].sum(), atol=1e-6
    )
    np.testing.assert_allclose(
        hand_position(READY_ARM_POSTURE), hand_position(old) + [0.0, 0.03], atol=1e-6
    )
    assert -lengths[0] * np.sin(old[1]) < 0  # Old elbow below shoulder.
    assert -lengths[0] * np.sin(READY_ARM_POSTURE[1]) > 0.15
    assert READY_ARM_POSTURE[3] > 0
    assert abs(READY_ARM_POSTURE[[1, 3, 5]].sum()) < np.deg2rad(1.0)


def test_pregrasp_parking_moves_both_ways_along_current_heading():
    tcp = np.array([0.77, 0.0, 1.08])
    forward = np.array([1.0, 0.0, 0.0])
    for cup_x, expected in ((0.67, -0.22), (1.09, 0.20), (0.89, 0.0)):
        cup = np.array([cup_x, 0.0, 0.98])
        distance = _pregrasp_drive_distance(cup, tcp, forward, 0.12)
        assert np.isclose(distance, expected)
        np.testing.assert_allclose(
            (cup - (tcp + distance * forward)) @ forward, 0.12
        )
    assert np.isclose(
        _pregrasp_drive_distance([0, 0.67, 0.98], [0, 0.77, 1.08], [0, 1, 0], 0.12),
        -0.22,
    )


def test_torso_target_matches_pregrasp_height_and_clamps_to_joint_limits():
    limits = (0.0, 0.38615)
    target = _torso_height_target(0.38615, 1.09, 0.98, limits)
    assert np.isclose(target, 0.27615)
    assert np.isclose(1.09 + target - 0.38615, 0.98)
    assert np.isclose(_torso_height_target(0.2, 1.0, 1.0, limits), 0.2)
    assert _torso_height_target(0.2, 1.0, 0.1, limits) == limits[0]
    assert _torso_height_target(0.2, 1.0, 2.0, limits) == limits[1]
    # A higher pre-grasp torso may leave less than 15 cm for a torso-only lift.
    assert _torso_height_target(target, 0.98, 1.15, limits) == limits[1]


def test_waypoint_jitter_is_seeded_and_bounded():
    scale = np.array([0.28, 0.25, 0.12, 0.15])
    first_rng = np.random.default_rng(7)
    replay_rng = np.random.default_rng(7)
    first = np.array([_waypoint_jitter(s, first_rng) for s in scale])
    replay = np.array([_waypoint_jitter(s, replay_rng) for s in scale])

    assert np.array_equal(first, replay)
    assert np.all(np.abs(first) <= scale * WAYPOINT_NOISE)


def test_shortest_successful_safe_nonempty_candidate_is_selected():
    class CollisionChecker:
        def path_env_collisions(self, position):
            return {1: 2, 2: 0}[int(position[0, 0])]

    candidates = [
        ("ik_screw", {"status": "Success", "position": np.ones((2, 1))}),
        ("rrt", {"status": "Success", "position": np.full((4, 1), 2)}),
        ("empty", {"status": "Success", "position": np.empty((0, 1))}),
        ("failed", {"status": "Failure", "position": np.ones((1, 1))}),
    ]
    selected, method, details = _choose_shortest_safe_plan(CollisionChecker(), candidates)

    assert method == "rrt"
    assert selected is candidates[1][1]
    assert details["ik_screw"]["collisions"] == 2
    assert details["empty"]["knots"] == 0
    assert details["failed"]["collisions"] is None


def test_episode_execution_metrics_include_sim_time_and_rate():
    metrics = _execution_metrics(4.0, 20, 0.05)

    assert metrics == {
        "wall_seconds": 4.0,
        "control_steps": 20,
        "simulated_seconds": 1.0,
        "steps_per_second": 5.0,
    }


def test_execution_noise_is_seeded_and_held_for_configured_steps():
    action = np.zeros(7)
    first = _noise_state(7)
    replay = _noise_state(7)

    draw_1 = FetchMotionPlanningSapienSolver._apply_execution_noise(first, action)
    draw_2 = FetchMotionPlanningSapienSolver._apply_execution_noise(first, action)
    draw_3 = FetchMotionPlanningSapienSolver._apply_execution_noise(first, action)
    replay_draw = FetchMotionPlanningSapienSolver._apply_execution_noise(replay, action)

    assert np.array_equal(draw_1, draw_2)
    assert not np.array_equal(draw_2, draw_3)
    assert np.array_equal(draw_1, replay_draw)
