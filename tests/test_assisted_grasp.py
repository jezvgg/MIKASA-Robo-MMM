"""Contact-window regressions ported from integration/four-tasks; local wiring tests."""
import numpy as np
import pytest

from utils.mikasa.assisted_grasp import GraspWindow, pick_candidate

DT = 0.01  # 100 Hz physics, the ManiSkill default


def run(window, steps, applying=True, candidate=0):
    events = []
    for _ in range(steps):
        ev = window.tick(applying, candidate)
        if ev is not None:
            events.append(ev)
    return events


def test_attaches_after_the_window_and_not_before():
    w = GraspWindow(DT)
    assert run(w, 30) == []                  # tick 1 arms the counter, 29 more: 0.29 s
    assert run(w, 1) == [("establish", 0)]   # 0.30 s
    assert w.attached and w.held == 0


def test_losing_the_candidate_restarts_the_window():
    w = GraspWindow(DT)
    run(w, 20)
    run(w, 1, candidate=None)
    assert run(w, 30) == []
    assert run(w, 1) == [("establish", 0)]


def test_switching_bodies_restarts_the_window():
    w = GraspWindow(DT)
    run(w, 20, candidate=0)
    # The switch tick resets, the next one arms, then 30 more are needed.
    assert run(w, 31, candidate=1) == []
    assert run(w, 1, candidate=1) == [("establish", 1)]


def test_an_open_command_never_attaches():
    w = GraspWindow(DT)
    assert run(w, 100, applying=False) == []
    assert w.held is None


def test_release_is_immediate_and_regrasp_waits_a_thirtieth():
    w = GraspWindow(DT)
    run(w, 31)
    assert run(w, 1, applying=False) == [("release", 0)]
    assert not w.attached
    # 1/30 s = 3.33 steps: still blocked for three more ticks, free on the fourth.
    run(w, 3, applying=True)
    assert w.held is not None
    run(w, 1, applying=True)
    assert w.held is None
    assert run(w, 31) == [("establish", 0)]


def test_checkpoint_state_round_trips():
    w = GraspWindow(DT)
    run(w, 12)
    copy = GraspWindow(DT)
    copy.load(w.state())
    assert run(copy, 19) == run(w, 19) == [("establish", 0)]


def test_push_from_outside_is_not_a_candidate():
    # Both fingers touch the prop, no pad ray passes through it: a closed hand pushing.
    assert pick_candidate(touch=[2], ray_hit=[False], dist=[0.05]) is None


def test_one_finger_is_not_enough():
    assert pick_candidate(touch=[1], ray_hit=[True], dist=[0.05]) is None
    assert pick_candidate(touch=[2], ray_hit=[True], dist=[0.05]) == 0


def test_nearest_one_finger_body_blocks_a_farther_grasp():
    # OmniGibson's order, kept on purpose: nearest eligible first, fingers counted after.
    assert pick_candidate(touch=[1, 2], ray_hit=[True, True], dist=[0.02, 0.06]) is None
    assert pick_candidate(touch=[0, 2], ray_hit=[True, True], dist=[0.02, 0.06]) == 1




from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils.mikasa.assisted_grasp import AssistedGrasp, AssistedGraspTask


class FakeBase:
    def __init__(self, **kwargs):
        self.num_envs = 1
        self.device = "cpu"
        self.calls = []

    def _after_reconfigure(self, options):
        self.calls.append("configure")

    def reset(self, **kwargs):
        self.calls.append("reset")

    def _after_simulation_step(self):
        self.calls.append("step")

    def get_info(self):
        return {"success": False}

    def get_state_dict(self):
        return {"actors": {}}

    def set_state_dict(self, state, env_idx=None):
        self.calls.append("restore")


class FakeTask(AssistedGraspTask, FakeBase):
    def _assisted_grasp_objects(self):
        return ["cup"]


def test_disabled_mixin_preserves_physics_info_and_state():
    task = FakeTask()
    with patch("utils.mikasa.assisted_grasp.AssistedGrasp") as constructor:
        task._after_reconfigure({})
    constructor.assert_not_called()
    task.reset()
    task._after_simulation_step()
    assert task._ag is None
    assert task.get_info() == {"success": False}
    assert task.get_state_dict() == {"actors": {}}


def test_enabled_mixin_hooks_reset_substeps_info_and_checkpoint():
    task = FakeTask(assisted_grasp=True)
    ag = Mock()
    ag.held_index.return_value = 0
    ag.state_dict.return_value = {"ag_window": "saved"}
    with patch("utils.mikasa.assisted_grasp.AssistedGrasp", return_value=ag) as constructor:
        task._after_reconfigure({})
    constructor.assert_called_once_with(task, ["cup"])
    ag.reset.side_effect = lambda: task.calls.append("release")
    task.reset()
    assert task.calls[-2:] == ["release", "reset"]
    task._after_simulation_step()
    ag.after_substep.assert_called_once_with()
    assert task.get_info()["ag_held"].tolist() == [0]
    state = task.get_state_dict()
    assert state == {"actors": {}, "ag_window": "saved"}
    task.set_state_dict(state)
    ag.load_state_dict.assert_called_once_with(state)


@pytest.mark.parametrize("gpu,num_envs", [(True, 1), (False, 2)])
def test_unsupported_backends_fail_explicitly(gpu, num_envs):
    with pytest.raises(NotImplementedError, match="CPU simulation with one env only"):
        AssistedGrasp(SimpleNamespace(gpu_sim_enabled=gpu, num_envs=num_envs), [])


def test_takeitback_and_takeit_only_allow_the_cup():
    from my_scenes.my_robocasa_takeitback import MyRoboCasaSceneTakeItBack
    from my_scenes.my_robocasa_takeit import MyRoboCasaSceneTakeIt

    for cls in (MyRoboCasaSceneTakeItBack, MyRoboCasaSceneTakeIt):
        assert issubclass(cls, AssistedGraspTask)
        task = object.__new__(cls)
        task.cup = object()
        assert task._assisted_grasp_objects() == [task.cup]


def test_veggies_allow_all_vegetables_but_not_the_pot():
    from my_scenes.my_robocasa_fridge_veggies import MyRoboCasaFridgeVeggies

    assert issubclass(MyRoboCasaFridgeVeggies, AssistedGraspTask)
    task = object.__new__(MyRoboCasaFridgeVeggies)
    task.veggies = [object(), object(), object()]
    task.plate = object()
    assert task._assisted_grasp_objects() == task.veggies
    assert task.plate not in task._assisted_grasp_objects()


def test_sweep_assistance_is_explicit_opt_in():
    from utils.test_planner import parse_args

    assert not parse_args([]).assisted_grasp
    assert parse_args(["--assisted-grasp"]).assisted_grasp
