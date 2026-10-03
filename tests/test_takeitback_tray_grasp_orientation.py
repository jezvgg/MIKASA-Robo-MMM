from types import SimpleNamespace

import numpy as np
import sapien
from mani_skill.agents.robots.fetch.fetch import Fetch

from planners.myrobocasa_takeitback_tray_planner import (
    WAYPOINT_NOISE,
    _build_grasp_pose,
    _waypoint_jitter,
)


def test_waypoint_jitter_is_seeded_and_bounded():
    scale = np.array([0.25, 0.40, 0.12])
    first = _waypoint_jitter(scale, np.random.default_rng(7))
    second = _waypoint_jitter(scale, np.random.default_rng(7))

    assert np.array_equal(first, second)
    assert np.all(np.abs(first) <= scale * WAYPOINT_NOISE)


def test_far_cup_grasp_keeps_upright_wrist_orientation():
    base = np.array([2.205, -1.22, 0.0])
    cup = np.array([2.488, -0.55, 0.0])
    approach = cup - base
    approach[2] = 0.0
    approach /= np.linalg.norm(approach)
    agent = SimpleNamespace(build_grasp_pose=Fetch.build_grasp_pose)

    pose = _build_grasp_pose(agent, approach, np.array([2.488, -0.55, 0.9975]))
    rotation = sapien.Pose(q=pose.q).to_transformation_matrix()[:3, :3]

    assert np.allclose(rotation[:, 2], approach, atol=1e-4)
    assert np.allclose(rotation[:, 0], [0.0, 0.0, 1.0], atol=1e-4)
