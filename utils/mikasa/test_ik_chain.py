"""The hardest-pose-first IK chain (planners/oracle/ik_chain.py) on stub solvers."""
import numpy as np
import pytest
import sapien

from planners.oracle import ik_chain


def test_joint_jump_is_the_largest_change_on_the_given_columns():
    assert ik_chain.joint_jump([0.0, 5.0, 1.0], [0.2, -5.0, 0.7], [0, 2]) == pytest.approx(0.3)


def test_distinct_keeps_one_configuration_per_branch():
    qs = [np.array([0.0, 0.0]), np.array([0.2, 0.1]), np.array([0.0, 0.3]), np.array([1.0, 1.0])]
    kept = ik_chain.distinct(qs, [0, 1])
    assert [q.tolist() for q in kept] == [[0.0, 0.0], [0.0, 0.3], [1.0, 1.0]]


def test_line_poses_end_on_the_easy_pose_in_short_steps():
    q_hard = [1.0, 0.0, 0.0, 0.0]
    q_easy = [np.cos(0.25), 0.0, 0.0, np.sin(0.25)]
    poses = ik_chain.line_poses(sapien.Pose([0, 0, 0], q_hard), sapien.Pose([0, 0.07, 0], q_easy))
    assert len(poses) == 4
    gaps = np.diff([0.0] + [float(p.p[1]) for p in poses])
    assert np.all(gaps <= 0.02 + 1e-9)
    assert np.allclose(poses[-1].p, [0, 0.07, 0])
    assert abs(abs(float(np.dot(poses[-1].q, q_easy))) - 1.0) < 1e-6  # sapien stores float32


def _arm(branches, *, gap_at=None, jump_at=None):
    """A one-joint arm whose angle is the pose's x plus a branch offset.

    `gap_at`: x where branch 1 has no solution; `jump_at`: x from which branch 2 flips by
    1 rad (a branch change the walk must refuse).
    """
    calls = []

    def solve(pose, seed, n):
        calls.append(n)
        x = float(pose.p[0])
        out = []
        for b in branches:
            if gap_at is not None and b == 1.0 and abs(x - gap_at) < 1e-6:
                continue
            q = x + b
            if jump_at is not None and b == 2.0 and x >= jump_at - 1e-6:
                q += 1.0
            out.append(np.array([q]))
        if n == 1:  # one start: the branch nearest the seed only, as CLIK from it would
            out = [min(out, key=lambda q: abs(q[0] - seed[0]))] if out else []
        return out

    solve.calls = calls
    return solve


def test_chain_walks_each_branch_and_ranks_by_the_easy_end():
    solve = _arm([0.0, 3.0])
    report = ik_chain.ChainReport()
    chains = ik_chain.chain_back(solve, sapien.Pose([0.0, 0, 0]), sapien.Pose([0.1, 0, 0]),
                                 np.array([2.9]), cols=[0], report=report)
    assert [round(float(c.easy[0]), 2) for c in chains] == [3.1, 0.1]
    assert all(len(c.knots) == 6 for c in chains)
    assert report.hard_found == 2 and report.chains == 2
    assert solve.calls[0] == 32


def test_a_step_with_no_solution_breaks_only_its_branch():
    solve = _arm([0.0, 1.0], gap_at=0.04)
    report = ik_chain.ChainReport()
    chains = ik_chain.chain_back(solve, sapien.Pose([0.0, 0, 0]), sapien.Pose([0.1, 0, 0]),
                                 np.array([1.0]), cols=[0], report=report)
    # Branch 1's one-start solve fails at x=0.04; the 8-start retry finds only branch 0,
    # a 0.96 rad jump: refused. Branch 0 walks through.
    assert [round(float(c.easy[0]), 2) for c in chains] == [0.1]
    assert report.broken_jump == 1


def test_a_branch_change_mid_walk_is_refused():
    solve = _arm([2.0], jump_at=0.06)
    report = ik_chain.ChainReport()
    chains = ik_chain.chain_back(solve, sapien.Pose([0.0, 0, 0]), sapien.Pose([0.1, 0, 0]),
                                 np.array([2.0]), cols=[0], report=report)
    assert chains == [] and report.broken_jump == 1


def test_no_hard_solution_means_no_chain():
    report = ik_chain.ChainReport()
    chains = ik_chain.chain_back(lambda pose, seed, n: [], sapien.Pose([0, 0, 0]),
                                 sapien.Pose([0.1, 0, 0]), np.array([0.0]), cols=[0], report=report)
    assert chains == [] and report.as_dict()["hard_found"] == 0


def test_keep_bounds_the_branches_walked():
    solve = _arm([0.0, 1.0, 2.0, 3.0])
    report = ik_chain.ChainReport()
    ik_chain.chain_back(solve, sapien.Pose([0, 0, 0]), sapien.Pose([0.02, 0, 0]),
                        np.array([0.0]), cols=[0], keep=2, report=report)
    assert report.walked == 2


def test_nearest_rank_orders_by_anchor_and_keeps_fallback_order_on_ties():
    goals = [np.array([0.0]), np.array([2.0]), np.array([1.004])]
    rank = ik_chain.nearest_rank([np.array([1.0])], [0], fallback=lambda gs, cur: list(reversed(gs)))
    assert [float(g[0]) for g in rank(goals, np.array([0.0]))] == [1.004, 2.0, 0.0]


class _FakeIK:
    joint_limits = np.array([[-6.0, 6.0]])

    def __init__(self, raw):
        self.raw = raw

    def _transform_goal_to_wrt_base(self, pose):
        return pose

    def IK(self, goal, start, mask, n_init_qpos=20):
        return "Success", [np.array([self.raw])]


class _Planner:
    def __init__(self, raw):
        self.planner = _FakeIK(raw)


def test_mplib_solver_unwraps_toward_the_seed_and_falls_back_to_the_raw_value():
    solve = ik_chain.mplib_solver(_Planner(-3.0), None)
    assert float(solve(sapien.Pose(), np.array([3.0]), 1)[0][0]) == pytest.approx(-3.0 + 2 * np.pi)
    window = ik_chain.mplib_solver(_Planner(-3.0), None, accept=lambda q: abs(q[0]) <= 3.2)
    assert float(window(sapien.Pose(), np.array([3.0]), 1)[0][0]) == pytest.approx(-3.0)
    none = ik_chain.mplib_solver(_Planner(-3.0), None, accept=lambda q: False)
    assert none(sapien.Pose(), np.array([3.0]), 1) == []


def test_the_chain_is_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv("MIKASA_IK_CHAIN", raising=False)
    assert not ik_chain.enabled()
    monkeypatch.setenv("MIKASA_IK_CHAIN", "1")
    assert ik_chain.enabled()
