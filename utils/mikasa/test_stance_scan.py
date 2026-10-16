"""Stance choice by a robust reach scan (planners/oracle/stance_scan.py), on stub scans."""
import numpy as np
import pytest

from planners.oracle import stance_scan


def test_jitter_moves_along_and_across_the_heading():
    j = stance_scan.jitter((0.0, 0.0, 0.0), along=(0.0, -0.1), lateral=(0.0, 0.03), yaw=(0.0, 0.1))
    assert j.shape == (8, 3)
    assert np.allclose(j[0], [0, 0, 0]) and np.allclose(j[1], [0, 0, 0.1])
    assert np.allclose(j[2], [0, 0.03, 0])          # lateral = to the left of the heading
    assert np.allclose(j[4], [-0.1, 0, 0])          # along = backwards when negative
    turned = stance_scan.jitter((1.0, 1.0, np.pi / 2), along=(-0.1,))
    assert np.allclose(turned[0], [1.0, 0.9, np.pi / 2])


def test_coverage_is_all_requirements_any_alternative_all_goals():
    masks = {None: np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], bool),
             0.2: np.array([[0], [0], [1]], bool)}
    need = [[(None, [0, 1]), (0.2, [0])], [(None, [1]), (0.2, [0])]]
    assert stance_scan.coverage(masks, need).tolist() == [True, False, True]


def _reach_scan(limit):
    """A stub scan: a goal is reachable when it is within `limit` metres of the stand."""
    calls = []

    def scan(goals, stands, torso=None, coll=False):
        calls.append((len(goals), len(stands), torso))
        d = np.linalg.norm(goals[None, :, :2, 3] - np.asarray(stands)[:, None, :2], axis=2)
        return d <= limit, None

    scan.calls = calls
    return scan


def test_robust_coverage_scores_the_share_of_perturbations_that_still_reach():
    goal = np.eye(4)
    goal[:3, 3] = [1.0, 0.0, 0.9]
    scan = _reach_scan(0.95)
    stands = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)]
    scores, cov = stance_scan.robust_coverage({None: goal[None]}, stands, [[(None, [0])]],
                                              along=(0.0, -0.1), scan=scan)
    # Stand 0 never reaches the goal 1.0 m away; stand 1 does from 0.9 m, not when 10 cm short.
    assert scores.tolist() == [0.0, 0.5]
    assert cov.shape == (2, 2)
    assert scan.calls == [(1, 4, None)]                 # ONE scan per torso setting


def test_robust_coverage_scans_each_torso_setting_once():
    goal = np.eye(4)
    scan = _reach_scan(10.0)
    scores, _ = stance_scan.robust_coverage({None: goal[None], 0.2: np.stack([goal, goal])},
                                            [(0, 0, 0)], [[(None, [0])], [(0.2, [0, 1])]], scan=scan)
    assert scores.tolist() == [1.0]
    assert sorted(c[2] for c in scan.calls if c[2] is not None) == [0.2] and len(scan.calls) == 2


def test_rank_holds_fragile_stances_in_reserve():
    order = stance_scan.rank([0.5, 1.0, 0.95, 0.59], [0.0, 0.2, 0.1, 0.0])
    assert order == [2, 1, 3, 0]


def test_choose_takes_the_first_verified_of_the_top_k_and_skips_empty_scores():
    seen = []

    def verify(i):
        seen.append(i)
        return i == 3

    chosen, tried = stance_scan.choose([0, 1, 2, 3], [1.0, 0.0, 0.8, 0.9], verify, k=3)
    assert chosen is None and seen == [0, 2]
    chosen, tried = stance_scan.choose([3, 0], [1.0, 0.0, 0.8, 0.9], verify, k=3)
    assert chosen == 3 and tried == [(3, True)]


def test_require_server_refuses_without_a_server(monkeypatch):
    monkeypatch.delenv("MIKASA_CUROBO_SOCKET", raising=False)
    monkeypatch.delenv("MIKASA_CUROBO_PY", raising=False)
    with pytest.raises(RuntimeError, match="cuRobo server"):
        stance_scan.require_server()
    monkeypatch.setenv("MIKASA_CUROBO_SOCKET", "/tmp/x.sock")
    stance_scan.require_server()


class _T:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class _Pose:
    def __init__(self, T):
        self._T = T

    @property
    def sp(self):
        return self

    def to_transformation_matrix(self):
        return self._T


class _Task:
    """Root frame at world (2, 1) turned 90 deg; joints q = (0.3, -0.2, 0.4)."""

    def __init__(self):
        from robots.fetch.curobo_client import stand_matrix

        self.q = np.array([0.3, -0.2, 0.4, 0.1])
        root = stand_matrix(2.0, 1.0, np.pi / 2)
        base = root @ stand_matrix(*self.q[:3])
        base[2, 3] = 0.12
        robot = type("R", (), {"get_qpos": lambda _s: [_T(self.q)]})()
        self.agent = type("A", (), {"robot": robot, "base_link": type("B", (), {"pose": [_Pose(base)]})()})()
        self.root = root


def test_root_joints_for_inverts_the_root_frame():
    from robots.fetch.curobo_client import stand_matrix

    task = _Task()
    now = stance_scan.base_stance(task)
    assert np.allclose(stance_scan.root_joints_for(task, *now), task.q[:3])
    target = (2.5, 0.7, -2.9)
    q = stance_scan.root_joints_for(task, *target)
    world = task.root @ stand_matrix(*q)
    assert np.allclose(world[:2, 3], target[:2])
    assert np.isclose(np.arctan2(world[1, 0], world[0, 0]), target[2])
