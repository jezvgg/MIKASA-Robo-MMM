from types import SimpleNamespace

import torch

from my_scenes.my_robocasa_takeit import MyRoboCasaSceneTakeIt


class _Actor:
    def __init__(self, position):
        self.pose = SimpleNamespace(p=torch.tensor([position], dtype=torch.float32))
        self.linear_velocity = torch.zeros((1, 3))
        self.angular_velocity = torch.zeros((1, 3))


class _Agent:
    def is_grasping(self, _object):
        return torch.tensor([False])


class _Scene:
    def __init__(self, force):
        self.force = torch.tensor([force], dtype=torch.float32)

    def get_pairwise_contact_forces(self, _first, _second):
        return self.force


def _checker(cup, tray, force):
    env = object.__new__(MyRoboCasaSceneTakeIt)
    env.cup = _Actor(cup)
    env.tray = _Actor(tray)
    env.cup_half = [0.036, 0.036, 0.058]
    env.tray_half = [0.141, 0.141, 0.010]
    env.agent = _Agent()
    env.scene = _Scene(force)
    env.device = torch.device("cpu")
    return bool(env.evaluate()["success"][0])


def test_tray_checker_requires_centered_supported_cup():
    tray = [0.0, 0.0, 0.93]
    tray_top = tray[2] + 0.010
    cup_on_tray = [0.0, 0.0, tray_top + 0.058]
    cup_on_edge = [0.09, 0.0, tray_top + 0.058]
    cup_on_counter = [0.15, 0.0, tray_top + 0.058 - 0.019]

    assert _checker(cup_on_tray, tray, [0.0, 0.0, 0.4])
    assert not _checker(cup_on_edge, tray, [0.0, 0.0, 0.4])
    assert not _checker(cup_on_counter, tray, [0.0, 0.0, 0.0])
