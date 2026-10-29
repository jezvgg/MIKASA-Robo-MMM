"""Consolidated, behavior-preserving collection regression tests."""


# Source group: test_contact_approach.py
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from planners.oracle.collection_solver import CollectionMotionPlanner
from planners.oracle.paths import RollPathPlanner


@pytest.mark.parametrize("case", ["no_contact", "combined_roll", "reachable"])
@pytest.mark.parametrize("torso_height", [None, 0.1])
def test_contact_lookahead_checks_both_legs_before_execution(case, torso_height):
    owner = object.__new__(CollectionMotionPlanner)
    owner._roll_indices = [8, 10, 12]
    owner._roll_low = np.zeros(3)
    owner._roll_high = np.zeros(3)
    current = np.zeros(15)
    owner.robot = SimpleNamespace(get_qpos=lambda: torch.tensor(current[None]))
    owner.base_env = SimpleNamespace(control_timestep=0.05)
    line = np.zeros((3, 11))
    contact = np.zeros((3, 11))
    line[1, 8] = 1.7
    contact[1, 8] = -1.7 if case == "combined_roll" else 1.6

    def forbidden_sync():
        raise RuntimeError("The target drawer is temporarily absent from planning")

    def inverse_kinematics(pose, start, fixed, **kwargs):
        assert start[3] == (0.0 if torso_height is None else torso_height)
        assert fixed[3] == (torso_height is not None)
        return "Success", [start.copy()]

    native = SimpleNamespace(
        joint_limits=np.tile([-6.28, 6.28], (15, 1)),
        move_group_joint_indices=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12],
        update_from_simulation=forbidden_sync,
        fold_qpos=lambda q: q.copy(),
        robot=SimpleNamespace(set_qpos=lambda *args: None),
        _transform_goal_to_wrt_base=lambda pose: pose,
        IK=inverse_kinematics,
        plan_qpos_line=lambda *args, **kwargs: {"status": "Success", "position": line},
        plan_screw=lambda *args, **kwargs: {
            "status": "No path" if case == "no_contact" else "Success",
            "position": contact,
        },
    )
    owner.planner = RollPathPlanner(native, owner)
    executed = []

    def line_executor(*args, precheck, **kwargs):
        if precheck(args[1][0]):
            executed.append(True)
            return "executed"
        return None

    owner._line_to_ik_goals = line_executor
    pose = SimpleNamespace(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    result = owner.approach_for_contact(pose, pose, torso_height=torso_height)
    np.testing.assert_array_equal(current, np.zeros(15))
    assert result == ("executed" if case == "reachable" else -1)
    assert executed == ([True] if case == "reachable" else [])


@pytest.mark.parametrize("references", [False, True])
def test_reference_goal_offered_only_after_random_goals_fail_lookahead(references):
    owner = object.__new__(CollectionMotionPlanner)
    owner._roll_indices = [8, 10, 12]
    owner._roll_low = np.zeros(3)
    owner._roll_high = np.zeros(3)
    current = np.zeros(15)
    owner.robot = SimpleNamespace(get_qpos=lambda: torch.tensor(current[None]))
    owner.base_env = SimpleNamespace(control_timestep=0.05)
    random_goal = np.zeros(15)
    random_goal[5] = 1.0          # its contact stroke will wind the roll window
    reference_goal = np.zeros(15)
    reference_goal[5] = -1.0
    seen = []

    def inverse_kinematics(pose, start, fixed, *, n_init_qpos=20, **kwargs):
        seen.append((start[5], n_init_qpos))
        if n_init_qpos == 1:
            return "Success", [reference_goal.copy()]
        return "Success", [random_goal.copy()]

    def screw(pose, start, **kwargs):
        position = np.zeros((3, 11))
        position[1, 8] = 3.5 if start[5] > 0 else 0.5
        return {"status": "Success", "position": position}

    native = SimpleNamespace(
        joint_limits=np.tile([-6.28, 6.28], (15, 1)),
        move_group_joint_indices=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12],
        fold_qpos=lambda q: q.copy(),
        robot=SimpleNamespace(set_qpos=lambda *args: None),
        _transform_goal_to_wrt_base=lambda pose: pose,
        IK=inverse_kinematics,
        plan_qpos_line=lambda *args, **kwargs: {"status": "Success", "position": np.zeros((3, 11))},
        plan_screw=screw,
    )
    owner.planner = RollPathPlanner(native, owner)
    if references:
        owner._ik_reference_arms = [("r", {5: -0.9})]
        owner._ik_reference_used = []
    executed = []

    def line_executor(target, goals, *args, precheck, **kwargs):
        goal = goals[0]
        if precheck(goal):
            executed.append(goal[5])
            return "executed"
        return None

    owner._line_to_ik_goals = line_executor
    pose = SimpleNamespace(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    result = owner.approach_for_contact(pose, pose, torso_height=0.1)
    assert seen[0] == (0.0, 40)  # the random-restart request comes first
    if references:
        assert result == "executed" and executed == [-1.0]
        assert owner._ik_reference_used == ["r"]
        assert seen[1] == (-0.9, 1)
    else:
        assert result == -1 and executed == [] and len(seen) == 1


# Source group: test_navigation_arrival.py
from types import SimpleNamespace
import numpy as np
from planners.oracle.collection_solver import CollectionMotionPlanner
from robots.fetch.extand import FetchMotionPlanningSapienSolver


def make_planner(distance=.2,tolerance=.1):
    p=object.__new__(CollectionMotionPlanner)
    p.navigation_arrival_tolerance=tolerance
    p.env_agent=SimpleNamespace(base_link=SimpleNamespace(pose=SimpleNamespace(sp=SimpleNamespace(p=np.array([distance,0.,0.])))))
    p._guard=SimpleNamespace(truncated=False)
    p._navigation_leg=[np.zeros(2),0]
    p.idle_steps=lambda t:("settled",t)
    return p


def test_stop_before_another_action_pair_after_measured_arrival(monkeypatch):
    p=make_planner();executed=[]
    path={"position":np.arange(6)[:,None],"velocity":np.array([1,1,1,1,-1,-1])[:,None]}
    before=path["velocity"].copy()
    def follow(self,pair,refine_steps=0):
        executed.extend(pair["velocity"][:,0].tolist())
        self.env_agent.base_link.pose.sp.p[0]-=.075
        return "moving"
    monkeypatch.setattr(FetchMotionPlanningSapienSolver,"follow_moving_forward",follow)
    assert p.follow_moving_forward(path)==("settled",2)
    assert executed==[1,1,1,1]
    np.testing.assert_array_equal(path["velocity"],before)


def test_reverse_path_outside_arrival_region_is_not_filtered(monkeypatch):
    p=make_planner(distance=1.);executed=[]
    path={"position":np.zeros((4,1)),"velocity":np.array([1,1,-1,-1])[:,None]}
    def follow(self,pair,refine_steps=0):
        executed.extend(pair["velocity"][:,0].tolist());return "moving"
    monkeypatch.setattr(FetchMotionPlanningSapienSolver,"follow_moving_forward",follow)
    assert p.follow_moving_forward(path)=="moving"
    assert executed==[1,1,-1,-1]


def test_default_preserves_canonical_path(monkeypatch):
    p=make_planner(tolerance=0.)
    monkeypatch.setattr(FetchMotionPlanningSapienSolver,"follow_moving_forward",lambda self,result,refine_steps:result)
    path={"unchanged":True}
    assert p.follow_moving_forward(path) is path


def test_already_at_goal_needs_no_movement(monkeypatch):
    p=make_planner(distance=.01)
    def unexpected(*args,**kwargs):raise AssertionError("Already inside goal region")
    monkeypatch.setattr(FetchMotionPlanningSapienSolver,"follow_moving_forward",unexpected)
    assert p.follow_moving_forward({"position":np.zeros((2,1)),"velocity":np.ones((2,1))})==("settled",2)


def test_arrived_drive_skips_opening_turn_but_keeps_checked_final_view(monkeypatch):
    p = make_planner(distance=.03)
    p.forward_navigation = True
    calls = []
    def canonical(self, **kwargs):
        calls.append(kwargs)
        return "checked view"
    monkeypatch.setattr(FetchMotionPlanningSapienSolver, "drive_base", canonical)
    assert p.drive_base(np.zeros(3), [1, 0, 0], freeze_arm=True) == "checked view"
    assert calls == [dict(target_pos=None, target_view_vec=[1, 0, 0],
                          freeze_arm=True, arrive_tol=None, reverse_ok=False)]
    assert p.drive_base(np.zeros(3)) == ("settled", 2)
    assert len(calls) == 1


def test_drive_outside_region_preserves_target_and_explicit_reverse_override(monkeypatch):
    p = make_planner(distance=.2)
    p.forward_navigation = True
    monkeypatch.setattr(FetchMotionPlanningSapienSolver, "drive_base", lambda self, **kw: kw)
    target = np.zeros(3)
    result = p.drive_base(target, reverse_ok=True)
    assert result["target_pos"] is target
    assert result["reverse_ok"] is True


# Source group: test_collection_contact.py
from types import SimpleNamespace

import torch

from planners.oracle.collection_solver import CollectionMotionPlanner


def test_articulation_contact_uses_the_selected_link_instance():
    selected = SimpleNamespace(name="inner_box")
    other = SimpleNamespace(name="inner_box")
    finger = object()
    queries = []

    def impulses(hand, target):
        queries.append(target)
        return torch.tensor([[0.1 if target is other else 0.0, 0.0, 0.0]])

    planner = object.__new__(CollectionMotionPlanner)
    planner.env_agent = SimpleNamespace(
        finger1_link=finger,
        finger2_link=object(),
        robot=SimpleNamespace(links_map={"gripper_link": object()}),
    )
    planner.base_env = SimpleNamespace(
        scene=SimpleNamespace(get_pairwise_contact_impulses=impulses)
    )
    assert not planner.gripper_touching(SimpleNamespace(get_links=lambda: [selected]))
    assert all(target is selected for target in queries)
    assert planner.gripper_touching(SimpleNamespace(get_links=lambda: [other]))


# Source group: test_drawer_hand_contact.py
import gymnasium as gym
import numpy as np
import sapien


def test_moving_push_keeps_enclosing_hand_allowance():
    import my_scenes  # noqa: F401
    from mplib.pymp.collision_detection import AllowedCollision
    from my_scenes.same_drawer import DRAWER_ART_SUFFIX, DRAWER_FIXTURES
    from planners.same_drawer_planner import (
        _hand_pairs,
        hand_contact,
        moving_drawer_push,
    )
    from planners.oracle import oracle_common as common

    env = gym.make("MikasaSameDrawer-v0", scene_idx=0, sim_backend="cpu",
                   obs_mode="state", control_mode="pd_joint_pos",
                   sim_config={"control_freq": 20, "sim_freq": 100})
    try:
        env.reset(seed=8000000)
        task = env.unwrapped
        planner = common.collection_planner_factory(env, False, False)
        planner.planner.update_from_simulation()

        def entries():
            acm, pairs = _hand_pairs(planner, 0)
            assert pairs
            return {acm.get_entry(a, b) for a, b in pairs}

        before = entries()
        with hand_contact(planner, task, 0):
            with common.contact_stroke(planner, [DRAWER_FIXTURES[0] + DRAWER_ART_SUFFIX]):
                pass
            # Documents the mplib behaviour the preview must compensate for.
            assert entries() == {None}
        assert entries() == before

        current = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
        tcp = task.agent.tcp.pose[0].sp
        push = sapien.Pose(tcp.p + [0, .02, 0], tcp.q)
        with hand_contact(planner, task, 0):
            moving_drawer_push(planner, task, current, push, 0)
            assert entries() == {AllowedCollision.ALWAYS}
        assert entries() == before
        assert np.allclose(task.agent.robot.get_qpos()[0].cpu().numpy(), current)
    finally:
        env.close()


# Source group: test_drawer_wrist_budget.py
from types import SimpleNamespace

import numpy as np
import pytest

from planners.same_drawer_planner import DrawerPathPlanner


def guard():
    owner = SimpleNamespace(_roll_indices=[8, 10, 12], _roll_low=np.zeros(3),
                            _roll_high=np.zeros(3), _drawer_initial_wrist_budget=True)
    native = SimpleNamespace(joint_limits=np.tile([-6.28, 6.28], (15, 1)),
                             move_group_joint_indices=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12])
    return DrawerPathPlanner(native, owner), native, owner


def test_reserves_complete_path_then_retains_history_after_release():
    p, _, owner = guard()
    path = np.zeros((3, 11));path[1, 10] = 1.6
    assert not p.accepts(path, move_group=True)
    owner._drawer_initial_wrist_budget = False
    assert p.accepts(path, move_group=True)
    owner._roll_high[2] = 1.6
    path[1, 10] = -1.6
    assert not p.accepts(path, move_group=True)
    assert owner._roll_high[2] == 1.6


def test_restores_native_limits_even_on_refusal():
    p, native, owner = guard()
    original = native.joint_limits
    with pytest.raises(RuntimeError):
        with p._limits():
            np.testing.assert_array_equal(native.joint_limits[12], [-1.45, 1.45])
            raise RuntimeError('failed plan')
    assert native.joint_limits is original
    np.testing.assert_array_equal(owner._roll_low, np.zeros(3))


# Source group: test_gripper_timing.py
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from planners.oracle.collection_solver import CollectionMotionPlanner
from robots.fetch.stepping import StepGuard


def planner_at(step):
    records = []
    env = SimpleNamespace(elapsed_steps=torch.tensor([step]))

    def physical_step(action):
        records.append((int(env.elapsed_steps[0]), action.copy()))
        env.elapsed_steps += 1
        return {}, 0.0, False, False, {}

    env.step = physical_step
    planner = object.__new__(CollectionMotionPlanner)
    planner.base_env = env
    planner.control_mode = "pd_joint_pos"
    planner.env_agent = SimpleNamespace(
        controller=SimpleNamespace(
            controllers={
                "arm": SimpleNamespace(
                    qpos=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
                ),
                "body": SimpleNamespace(qpos=torch.tensor([[0.2, 0.3, 0.35]])),
            }
        )
    )
    planner._guard = StepGuard(env)
    planner.gripper_state = 1.0
    planner.print_env_info = False
    planner.vis = False
    return planner, records


@pytest.mark.parametrize("start", [0, 1, 10, 11])
def test_switch_is_even_and_preserved_by_stride_two(start):
    planner, records = planner_at(start)
    planner.close_gripper(t=2)
    switches = [step for step, action in records if action[7] == -1]
    assert switches[0] == start + start % 2
    assert switches[0] % 2 == 0
    assert len(records) == 2 + start % 2
    assert [action[7] for step, action in records if step % 2 == 0][0] == -1
    if start % 2:
        assert records[0][1][7] == 1
        np.testing.assert_array_equal(records[0][1][:7], records[1][1][:7])
    assert {action[7] for _, action in records} <= {-1, 1}


def test_repeated_command_does_not_insert_an_unneeded_hold():
    planner, records = planner_at(1)
    planner.open_gripper(t=2)
    assert [step for step, _ in records] == [1, 2]


def test_intermediate_gripper_and_ramp_are_rejected_before_any_step():
    planner, records = planner_at(0)
    with pytest.raises(ValueError, match="binary"):
        planner.change_gripper_state(gripper_state=0.4)
    with pytest.raises(ValueError, match="binary"):
        planner.open_gripper(ramp=8)
    assert records == []


# Source group: test_source_action_clock.py
import numpy as np



def test_all_executed_channels_survive_stride_two_without_action_rewriting():
    planner, records = planner_at(0)
    planner._guard.tape = []
    for step in range(8):
        proposed = np.linspace(-.2, .2, 13) + step * .01
        proposed[7] = 1.
        planner._step(proposed)
    executed = np.stack([action for _, action in records])
    replayed = np.repeat(executed[::2], 2, axis=0)
    np.testing.assert_array_equal(executed, replayed)
    assert not np.array_equal(executed[0], executed[2])
    np.testing.assert_array_equal(np.stack(planner._guard.tape), executed)


def test_even_gripper_switch_starts_a_complete_saved_action_pair():
    planner, records = planner_at(0)
    planner.open_gripper(t=3)
    planner.close_gripper(t=2)
    executed = np.stack([action for _, action in records])
    np.testing.assert_array_equal(executed, np.repeat(executed[::2], 2, axis=0))
    assert executed[3, 7] == 1. and executed[4, 7] == -1.


def test_scoped_joint_lock_preserves_wrist_and_head_and_recording():
    planner, records = planner_at(0)
    planner._guard.tape = []
    planner.fixed_action_targets = {i: .123 for i in range(6)}
    planner.fixed_action_targets.update({10: .2, 11: 0., 12: 0.})
    for step in range(8):
        action = np.full(13, step * .01)
        action[7] = -1.
        planner._step(action)
    executed = np.stack([a for _, a in records])
    np.testing.assert_array_equal(executed[:, :6], np.full((8, 6), .123))
    assert np.ptp(executed[:, 6]) > 0 and np.ptp(executed[:, 8]) > 0
    np.testing.assert_array_equal(executed[:, 11:13], np.zeros((8, 2)))
    np.testing.assert_array_equal(executed, np.repeat(executed[::2], 2, axis=0))
    np.testing.assert_array_equal(np.stack(planner._guard.tape), executed)


# Source group: test_execution_noise.py
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from utils.mikasa.execution_noise import ExecutionNoise


def test_blocks_are_seeded_and_phase_exit_restores_clean_commands():
    def sample(global_seed):
        np.random.seed(global_seed)
        noise = ExecutionNoise(
            seed=23, action_noise=0.003, noise_hold=10, log=lambda *args, **kwargs: None
        )
        original = np.arange(7) / 10
        clean = original.copy()
        with noise.phase("transfer"):
            result = np.stack([noise.apply(clean, i) for i in range(25)])
        np.testing.assert_array_equal(clean, original)
        np.testing.assert_array_equal(noise.apply(clean, 25), clean)
        np.testing.assert_array_equal(result[:10], np.broadcast_to(result[0], (10, 7)))
        np.testing.assert_array_equal(
            result[10:20], np.broadcast_to(result[10], (10, 7))
        )
        assert not np.array_equal(result[0], result[10])
        return result

    np.testing.assert_array_equal(sample(42), sample(999))


@pytest.mark.parametrize("path_length", [11, 12, 13])
def test_noise_reaches_executed_knots_but_not_refinement_or_gripper(monkeypatch, path_length):
    from planners.oracle.collection_solver import CollectionMotionPlanner
    from robots.fetch.extand import FetchMotionPlanningSapienSolver

    records, logs = [], []
    planner = object.__new__(CollectionMotionPlanner)
    planner.control_mode = "pd_joint_pos"
    planner._guard = SimpleNamespace(truncated=False)
    planner.base_env = SimpleNamespace(
        control_mode="pd_joint_pos", elapsed_steps=torch.tensor([0])
    )
    planner.env_agent = SimpleNamespace(
        controller=SimpleNamespace(
            controllers={"arm": SimpleNamespace(qpos=torch.zeros(1, 7))}
        )
    )
    planner.execution_noise = ExecutionNoise(
        seed=9,
        action_noise=0.003,
        noise_hold=10,
        log=lambda *args, **kwargs: logs.append((args, kwargs)),
    )
    clean = np.array([0.0] * 7 + [1.0, 0.2, 0.3, 0.35, 1e-15, -0.4])

    def physical_step(self, action):
        np.testing.assert_array_equal(self._last_abs, action)
        records.append(action.copy())
        self.base_env.elapsed_steps += 1
        return None

    def path_and_refinement(self, result, refine, stop_when):
        for _ in range(len(result["position"]) + 2):
            self._step(clean)

    monkeypatch.setattr(FetchMotionPlanningSapienSolver, "_step", physical_step)
    monkeypatch.setattr(
        FetchMotionPlanningSapienSolver,
        "follow_forward_path_w_refinement",
        path_and_refinement,
    )
    path = {"position": np.zeros((path_length, 15))}
    with planner.execution_noise.phase("transfer"):
        planner.follow_forward_path_w_refinement(path, refine=True)
    # A subsequent contact/gripper step stays clean.
    planner._step(clean)
    result = np.stack(records)
    noisy_ticks = 2 * ((path_length + 1) // 2)
    assert np.any(result[:noisy_ticks, :7] != 0)
    np.testing.assert_array_equal(result[noisy_ticks:, :7], 0)
    np.testing.assert_array_equal(result[:-1], np.repeat(result[:-1:2], 2, axis=0))
    np.testing.assert_array_equal(
        result[:, 7:11], np.broadcast_to(clean[7:11], (len(result), 4))
    )
    np.testing.assert_array_equal(result[:, 11], 0)
    np.testing.assert_array_equal(result[:, 12], -0.4)
    assert clean[11] == 1e-15
    steps = [
        data["control_step"] for tags, data in logs if tags[0] == "execution_noise_step"
    ]
    assert steps == list(range(0, path_length, 2))


# Source group: test_horizon_gate.py
import json

import h5py
import numpy as np
import pytest

from .client import execute_actions
from .contract import recording_info
from .pipeline import classify_collection_result


def test_success_and_truncation_on_same_step_are_rejected():
    class Env:
        def step(self, action):
            self.result = ({}, 1.0, True, True, {"success": True})
            return self.result

    env = Env()
    actions = np.zeros((1, 13), dtype=np.float32)
    actions[:, 7] = 1
    result = execute_actions(env, actions, repeat=1)
    assert result["completed"] and result["success"]
    assert result["status"] == "truncated"
    assert classify_collection_result(env.result) == "truncated"
    assert (
        classify_collection_result(({}, 1.0, True, False, {"success": True}))
        == "success"
    )


@pytest.mark.parametrize("flags", [None, [False, False], [False, True]])
def test_h5_gate_reads_actual_truncation_history(tmp_path, flags):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as h5:
        g = h5.create_group("traj_0")
        g.create_dataset("actions", data=np.zeros((2, 13)))
        g.create_dataset("success", data=[False, True])
        g.create_dataset("rewards", data=[0.0, 1.0])
        g.create_dataset("env_states/example", data=np.zeros((3, 1)))
        if flags is not None:
            g.create_dataset("truncated", data=flags)
    meta = dict(
        env_info={
            "env_kwargs": {
                "robot_uids": "ds_fetch",
                "control_mode": "pd_joint_pos",
                "sim_config": {"control_freq": 20},
            }
        },
        mikasa_data={
            "stage": "oracle",
            "version": 3,
            "policy_fps": 20,
            "action_repeat": 1,
            "instruction": "Place the apple on the plate.",
            "success": True,
            "success_once": True,
        },
        episodes=[
            {
                "episode_id": 0,
                "control_mode": "pd_joint_pos",
                "elapsed_steps": 2,
                "success": True,
                "success_once": True,
            }
        ],
    )
    path.with_suffix(".json").write_text(json.dumps(meta))
    if flags == [False, False]:
        assert recording_info(path, stage="oracle") == meta
    else:
        with pytest.raises(ValueError, match="[Tt]runcat"):
            recording_info(path, stage="oracle")


# Source group: test_head_tracking.py
import numpy as np
import pytest
from planners.oracle.head_tracking import aim_angles, rotation, HeadTracker


@pytest.mark.parametrize(
    "pan,tilt,distance", [(-1.0, -0.2, 3.0), (0.6, 0.4, 1.0), (-0.3, 0.9, 0.5)]
)
def test_camera_ray_recovers_head_angles(pan, tilt, distance):
    # Neutral DSFetch head geometry and the midpoint of its offset cameras.
    origin = np.array([0.14253, 0, 0.057999])
    eye = np.array([-0.445, 0, 0.0225])
    pitch = 0.306
    ray = np.array([np.cos(pitch), 0, -np.sin(pitch)])
    target = rotation("z", pan)[:3, :3] @ (
        origin + rotation("y", tilt)[:3, :3] @ (eye + distance * ray)
    )
    actual = aim_angles(target, origin, eye, pitch, [[-1.56, 1.56], [-0.75, 1.44]])
    np.testing.assert_allclose(actual, [pan, tilt], atol=1e-9)


def test_slew_limit_and_joint_bounds_for_unreachable_target():
    angles = aim_angles(
        np.array([-2.0, 1.0, 3.0]),
        np.array([0.14, 0, 0.06]),
        np.array([-0.445, 0, 0.0225]),
        0.306,
        [[-1.56, 1.56], [-0.75, 1.44]],
    )
    assert -1.56 <= angles[0] <= 1.56
    assert -0.75 <= angles[1] <= 1.44
    tracker = object.__new__(HeadTracker)
    tracker.last_command = np.zeros(2)
    tracker.max_delta = 0.05
    tracker.desired = lambda: angles
    before = tracker.last_command.copy()
    command = tracker.command()
    assert np.max(np.abs(command - before)) <= 0.05 + 1e-12
    command[:] = 99
    assert np.max(np.abs(tracker.last_command)) <= 0.05 + 1e-12


def test_small_pose_fluctuations_do_not_make_head_dither():
    tracker = object.__new__(HeadTracker)
    tracker.last_command = np.array([0.2, 0.4])
    tracker.max_delta = 0.05
    start = tracker.last_command.copy()
    for offset in [0.001, -0.001, 0.004, -0.004]:
        tracker.desired = lambda: start + offset
        np.testing.assert_array_equal(tracker.command(), start)
    tracker.desired = lambda: start + 0.01
    np.testing.assert_allclose(tracker.command(), start + 0.01)
