"""Solve the hardest pose first, then walk back to the easy one on the same IK branch.

Our planners solve the EASY pose first — a standoff or a pre-grasp, from a template or the
current posture — and only then try the stroke to the contact. When the contact sits by
the counter the stroke is refused right there: SeasonDish seed 1005 (2026-10-16) had three
of its seven standoff candidates hit ``wrist_flex_link`` against the counter in the last
centimetre of the contact screw, on a branch that was chosen without ever asking IK about
the contact.

The pattern here is BIMBO_EvoX task 40's ``chain()`` (solve the hardest target, then each
earlier target seeded with the later solution; a link with a joint jump over the bound
breaks the chain) and b1k_rl_guys ``plan_descent`` (contact pose first, then the straight
line back in 2 cm steps, each step seeded from the previous one, a step refused above
0.6 rad). mplib's CLIK takes the seed as its FIRST start, so a one-start solve from the
previous knot stays on that knot's branch when one exists there.

The chain only proposes configurations. Every knot is mplib's own IK answer (CLIK plus
the planning world's collision check), and the caller's existing screw and line checks
still decide what is executed.

Joint jumps are measured on unwrapped solutions with no angle wrapping: `unwrap_toward`
already picks the nearest equivalent angle inside the joint limits, so a roll that stays
far after unwrapping is a real turn of the joint (the episode's roll window is narrower
than a full revolution; `depth_recall_v1_planner.roll_rank` documents a 5.8 rad wrist
spin of exactly that kind).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import sapien

#: b1k ``plan_descent`` refuses a 2 cm step whose joints move more than this.
JUMP_RAD = 0.6
#: Cartesian step of the walk back, metres (b1k ``plan_descent``).
STEP_M = 0.02
#: Hard-pose solutions closer than this on the compared joints are one branch (b1k c01).
DEDUP_RAD = 0.25
#: Hard-pose solutions walked back at most (b1k: up to 6 contact solutions).
KEEP = 6

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
              "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint")


def enabled() -> bool:
    """``MIKASA_IK_CHAIN=1`` turns the chain on; off by default (the pinned profiles)."""
    return os.environ.get("MIKASA_IK_CHAIN", "0") == "1"


def arm_columns(task) -> list[int]:
    """qpos columns of the seven arm joints, in `ARM_JOINTS` order."""
    amap = task.agent.robot.active_joints_map
    return [int(np.asarray(amap[n].active_index).reshape(-1)[0]) for n in ARM_JOINTS]


def joint_jump(qa, qb, cols) -> float:
    """Largest single-joint change between two configurations over `cols`.

    Example:
        >>> joint_jump([0.0, 1.0, 2.0], [0.1, 1.5, 2.0], [0, 1, 2])
        0.5
    """
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    return float(np.abs(qb[cols] - qa[cols]).max())


def distinct(qs, cols, threshold: float = DEDUP_RAD) -> list:
    """Keep the first of every group of configurations within `threshold` (max-norm).

    Example:
        >>> [q.tolist() for q in distinct([np.array([0.0]), np.array([0.1]), np.array([0.5])], [0])]
        [[0.0], [0.5]]
    """
    kept: list = []
    for q in qs:
        if all(joint_jump(q, k, cols) >= threshold for k in kept):
            kept.append(q)
    return kept


def _slerp(qa, qb, t: float) -> np.ndarray:
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = qa + t * (qb - qa)
        return q / np.linalg.norm(q)
    theta = math.acos(min(1.0, dot))
    return (math.sin((1 - t) * theta) * qa + math.sin(t * theta) * qb) / math.sin(theta)


def line_poses(hard: sapien.Pose, easy: sapien.Pose, step: float = STEP_M) -> list[sapien.Pose]:
    """Poses from `hard` (excluded) to `easy` (included), consecutive ones at most `step` apart.

    Example:
        >>> ps = line_poses(sapien.Pose([0, 0, 0]), sapien.Pose([0.1, 0, 0]))
        >>> [round(float(p.p[0]), 3) for p in ps]
        [0.02, 0.04, 0.06, 0.08, 0.1]
    """
    a = np.asarray(hard.p, dtype=float)
    b = np.asarray(easy.p, dtype=float)
    n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step - 1e-6)))  # float32 poses
    return [sapien.Pose(p=a + (b - a) * k / n, q=_slerp(hard.q, easy.q, k / n)) for k in range(1, n + 1)]


@dataclass
class Chain:
    """Knots from the hard pose (first) to the easy one (last), full qpos each."""

    knots: list
    jump: float

    @property
    def hard(self):
        return self.knots[0]

    @property
    def easy(self):
        return self.knots[-1]


@dataclass
class ChainReport:
    """What one `chain_back` call found, for the planners' logs."""

    hard_found: int = 0
    walked: int = 0
    chains: int = 0
    broken_ik: int = 0
    broken_jump: int = 0
    worst_kept_jump: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = dict(hard_found=self.hard_found, walked=self.walked, chains=self.chains,
                   broken_ik=self.broken_ik, broken_jump=self.broken_jump,
                   worst_kept_jump=(None if self.worst_kept_jump is None
                                    else round(self.worst_kept_jump, 3)))
        out.update(self.extra)
        return out


def chain_back(solve, hard: sapien.Pose, easy: sapien.Pose, start, *, cols, keep: int = KEEP,
               n_hard: int = 32, step: float = STEP_M, jump: float = JUMP_RAD,
               report: ChainReport | None = None) -> list[Chain]:
    """Chains from the hard pose back to the easy one, nearest-to-`start` easy end first.

    Args:
        solve: ``solve(pose, seed, n) -> list of qpos`` — IK at a world pose with `n`
            starts, the first being `seed`; solutions already unwrapped toward `seed` and
            filtered by the caller's acceptance rules (see `mplib_solver`).
        hard: the pose the manipulation cannot do without (the contact, the place).
        easy: the pose the planner reaches first (the standoff, the pre-grasp, the entry).
        start: the configuration the hard pose is seeded from (and the ranking reference).
        cols: the qpos columns the jump bound and the ranking look at (the arm).
        keep: at most this many distinct hard-pose branches are walked.
        n_hard: IK starts for the hard pose.
        step: Cartesian step of the walk, metres.
        jump: a step that moves any of `cols` further than this breaks its chain.
        report: filled in place when given.

    Example:
        >>> def solve(pose, seed, n):                  # a 1-joint arm whose angle is x
        ...     return [np.array([float(pose.p[0])]), np.array([float(pose.p[0]) + 3.0])]
        >>> cs = chain_back(solve, sapien.Pose([0.0, 0, 0]), sapien.Pose([0.1, 0, 0]),
        ...                 np.array([0.0]), cols=[0])
        >>> [round(float(c.easy[0]), 2) for c in cs]   # both branches walk, nearest first
        [0.1, 3.1]
    """
    report = report if report is not None else ChainReport()
    start = np.asarray(start, dtype=float)
    found = list(solve(hard, start, n_hard))
    report.hard_found = len(found)
    found.sort(key=lambda q: joint_jump(start, q, cols))
    roots = distinct(found, cols)[:keep]
    report.walked = len(roots)
    poses = line_poses(hard, easy, step)
    chains: list[Chain] = []
    for q0 in roots:
        knots = [np.asarray(q0, dtype=float)]
        worst = 0.0
        for pose in poses:
            prev = knots[-1]
            sols = list(solve(pose, prev, 1)) or list(solve(pose, prev, 8))
            if not sols:
                report.broken_ik += 1
                break
            q = min(sols, key=lambda s: joint_jump(prev, s, cols))
            j = joint_jump(prev, q, cols)
            if j > jump:
                report.broken_jump += 1
                break
            knots.append(np.asarray(q, dtype=float))
            worst = max(worst, j)
        else:
            chains.append(Chain(knots, worst))
    chains.sort(key=lambda c: joint_jump(start, c.easy, cols))
    report.chains = len(chains)
    if chains:
        report.worst_kept_jump = max(c.jump for c in chains)
    return chains


def mplib_solver(planner, mask, *, accept=None):
    """A `solve` for `chain_back` on the planner's own IK.

    `planner.planner.IK` is mplib's (or the opt-in backend's) collision-checked IK; each
    solution is unwrapped toward the seed and, like `season_dish_candidates`, kept in its
    raw representative when the unwrapped one leaves the episode's roll window.
    """
    import mplib

    from robots.fetch.utils import unwrap_toward

    p = planner.planner

    def solve(pose, seed, n):
        goal = p._transform_goal_to_wrt_base(mplib.Pose(p=np.asarray(pose.p), q=np.asarray(pose.q)))
        try:
            status, found = p.IK(goal, np.asarray(seed, dtype=float), mask, n_init_qpos=int(n))
        except Exception:  # a solver refusal, not an episode error
            return []
        if status != "Success" or found is None:
            return []
        out = []
        for raw in np.atleast_2d(found):
            q = unwrap_toward(np.asarray(raw, dtype=float), seed, p.joint_limits)
            if accept is not None and not accept(q):
                q = np.asarray(raw, dtype=float).copy()
                if not accept(q):
                    continue
            out.append(q)
        return out

    return solve


def nearest_rank(anchors, cols, fallback=None):
    """A ``ready_by_line(rank=)`` callable: goals nearest any anchor first.

    Ties (equal to 0.01 rad) keep `fallback`'s order, or the given order without one.

    Example:
        >>> rank = nearest_rank([np.array([1.0])], [0])
        >>> [float(g[0]) for g in rank([np.array([0.0]), np.array([0.9])], np.array([0.0]))]
        [0.9, 0.0]
    """
    anchors = [np.asarray(a, dtype=float) for a in anchors]

    def rank(goals, cur):
        goals = list(np.atleast_2d(goals))
        if fallback is not None:
            goals = list(fallback(goals, cur))
        return sorted(goals, key=lambda g: round(min(joint_jump(a, g, cols) for a in anchors), 2))

    return rank
