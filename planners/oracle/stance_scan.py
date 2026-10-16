"""Where the base should stand: a batched reach scan that survives base error, then the scene.

Our planners park where a constant says (SeasonDish: the station dock backed off 10 cm;
Depth Recall: the row's column at y = -1.35) and take whatever the base delivers — the
SeasonDish drive stops up to 10 cm short and accepts 0.25 rad of turn residual. The
pattern here is BIMBO_EvoX's (task 98's stance rings ranked by one batched scan per arm,
``fair_skills/stance.py``'s jitter with ``robust = mask.mean`` and a 1.5 * (1 - robust)
cost, stances under 0.6 held in reserve) and b1k_rl_guys c01 ``stand_for`` (a stance is
valid only if the WHOLE manipulation builds from it; failed stances are excluded).

Two tiers, because the scan cannot see the scene:

1. `robust_coverage`: one cuRobo scan (`robots/fetch/curobo_client.scan`) per torso
   setting over every candidate stance and every perturbation of it, goals held fixed —
   the fraction of perturbations from which every requirement still has a reachable
   alternative. Kinematics only.
2. the caller's `verify` (mplib, with the scene) on the best few, in rank order — a
   stance slide that only the scan approved gained nothing on SeasonDish before
   (60/75 vs 59/75, refusals were scene collisions; robots/fetch/IK_BACKENDS.md).

``MIKASA_STANCE_SCAN=1`` turns it on in the planners; it needs the cuRobo server
(``MIKASA_CUROBO_SOCKET`` or ``MIKASA_CUROBO_PY``) and refuses loudly without one.
"""
from __future__ import annotations

import os

import numpy as np

#: Stances whose robust coverage is below this go behind every other (BIMBO stance.py).
RESERVE = 0.6
#: Cost weight of lost coverage (BIMBO stance.py: cost + 1.5 * (1 - robust)).
ROBUST_WEIGHT = 1.5
#: Stances handed to the scene check at most (b1k c01 allows 40; ours pay mplib IK each).
VERIFY_K = 3


def enabled() -> bool:
    """``MIKASA_STANCE_SCAN=1`` turns the scan on; off by default (the pinned profiles)."""
    return os.environ.get("MIKASA_STANCE_SCAN", "0") == "1"


def require_server() -> None:
    """Raise unless a cuRobo server is configured — no silent fallback to the old stance."""
    if not (os.environ.get("MIKASA_CUROBO_SOCKET") or os.environ.get("MIKASA_CUROBO_PY")):
        raise RuntimeError("MIKASA_STANCE_SCAN=1 needs the cuRobo server: set MIKASA_CUROBO_SOCKET "
                           "or MIKASA_CUROBO_PY (robots/fetch/IK_BACKENDS.md)")


def jitter(stand, along=(0.0,), lateral=(0.0,), yaw=(0.0,)) -> np.ndarray:
    """Perturbations of ``stand = (x, y, yaw)`` in its own heading frame, as (J, 3).

    `along` moves along the heading, `lateral` to its left, `yaw` turns it.

    Example:
        >>> np.round(jitter((1.0, 2.0, np.pi / 2), along=(0.0, 0.1), lateral=(0.05,)), 3).tolist()
        [[0.95, 2.0, 1.571], [0.95, 2.1, 1.571]]
    """
    x, y, th = (float(v) for v in stand)
    c, s = np.cos(th), np.sin(th)
    out = [(x + a * c - l * s, y + a * s + l * c, th + w)
           for a in along for l in lateral for w in yaw]
    return np.asarray(out, dtype=np.float64)


def coverage(masks: dict, need) -> np.ndarray:
    """Per row of the masks: does every requirement have a fully reachable alternative?

    Args:
        masks: ``{torso: (K, N_torso) bool}`` — row k is one perturbed stance, column n
            the n-th goal scanned at that torso setting (None = torso free).
        need: requirements, ALL of which must hold; each a list of alternatives, ANY of
            which suffices; each alternative ``(torso, [goal indices])``, ALL reachable.

    Example:
        >>> m = {None: np.array([[1, 1, 0], [1, 0, 1]], bool), 0.2: np.array([[0], [1]], bool)}
        >>> coverage(m, [[(None, [0, 1]), (0.2, [0])], [(None, [0])]]).tolist()
        [True, True]
        >>> coverage(m, [[(None, [1, 2])]]).tolist()
        [False, False]
    """
    rows = len(next(iter(masks.values())))
    ok = np.ones(rows, dtype=bool)
    for requirement in need:
        any_alt = np.zeros(rows, dtype=bool)
        for torso, idx in requirement:
            any_alt |= np.asarray(masks[torso], dtype=bool)[:, list(idx)].all(axis=1)
        ok &= any_alt
    return ok


def robust_coverage(goals: dict, stands, need, *, along=(0.0,), lateral=(0.0,), yaw=(0.0,),
                    scan=None, coll: bool = False):
    """Robust coverage of each candidate stance: (scores (M,), per-perturbation (M, J)).

    Args:
        goals: ``{torso: (N, 4, 4)}`` world gripper_link poses, scanned with the torso
            held at that height (None: free).
        stands: (M, 3) candidate stances (x, y, yaw).
        need: see `coverage`.
        along, lateral, yaw: the perturbation grid every stance is scanned under.
        scan: ``scan(goals, stands, torso=..., coll=...) -> (ok, q)``;
            `robots.fetch.curobo_client.scan` by default.
    """
    if scan is None:
        from robots.fetch.curobo_client import scan as scan_
        scan = scan_
    stands = np.asarray(stands, dtype=np.float64).reshape(-1, 3)
    perturbed = np.concatenate([jitter(s, along, lateral, yaw) for s in stands])
    J = len(perturbed) // len(stands)
    masks = {}
    for torso, G in goals.items():
        G = np.asarray(G, dtype=np.float64).reshape(-1, 4, 4)
        masks[torso] = scan(G, perturbed, torso=torso, coll=coll)[0] if len(G) else \
            np.zeros((len(perturbed), 0), dtype=bool)
    cov = coverage(masks, need).reshape(len(stands), J)
    return cov.mean(axis=1), cov


def rank(scores, cost, *, reserve: float = RESERVE, weight: float = ROBUST_WEIGHT) -> list[int]:
    """Candidate order: robust ones first, by ``cost + weight * (1 - robust)``.

    Example:
        >>> rank([1.0, 0.5, 0.9], [0.3, 0.0, 0.0])
        [2, 0, 1]
    """
    scores = np.asarray(scores, dtype=float)
    cost = np.asarray(cost, dtype=float)
    key = [(bool(scores[i] < reserve), float(cost[i] + weight * (1.0 - scores[i])), i)
           for i in range(len(scores))]
    return [i for *_, i in sorted(key)]


def choose(order, scores, verify, *, k: int = VERIFY_K, floor: float = 0.0):
    """The first of the top `k` stances (with coverage above `floor`) that `verify` accepts.

    Returns ``(index or None, [(index, verdict), ...])``; a stance that fails is not tried
    again in this call (b1k c01 excludes failed stances).

    Example:
        >>> choose([2, 0, 1], [0.9, 0.5, 1.0], lambda i: i == 0)
        (0, [(2, False), (0, True)])
    """
    tried = []
    for i in list(order)[:k]:
        if float(scores[i]) <= floor:
            continue
        verdict = bool(verify(i))
        tried.append((i, verdict))
        if verdict:
            return i, tried
    return None, tried


def root_joints_for(task, x: float, y: float, yaw: float) -> np.ndarray:
    """The simulator's three root-joint values that put base_link at world (x, y, yaw).

    The root joints are x, y prismatic then z revolute under the articulation's root
    frame; that frame is recovered from the current base_link pose and joints, so this
    holds whatever pose the scene spawned the robot with.
    """
    from robots.fetch.curobo_client import stand_matrix

    q = np.asarray(task.agent.robot.get_qpos()[0].cpu().numpy(), dtype=np.float64)
    T_now = task.agent.base_link.pose[0].sp.to_transformation_matrix()
    J_now = stand_matrix(q[0], q[1], q[2])
    T_root = T_now @ np.linalg.inv(J_now)
    goal = stand_matrix(x, y, yaw)
    goal[2, 3] = T_now[2, 3]
    J = np.linalg.inv(T_root) @ goal
    th = float(np.arctan2(J[1, 0], J[0, 0]))
    th = q[2] + (th - q[2] + np.pi) % (2 * np.pi) - np.pi
    return np.array([J[0, 3], J[1, 3], th])


def base_stance(task) -> np.ndarray:
    """base_link's world (x, y, yaw) now."""
    T = task.agent.base_link.pose[0].sp.to_transformation_matrix()
    return np.array([T[0, 3], T[1, 3], float(np.arctan2(T[1, 0], T[0, 0]))])
