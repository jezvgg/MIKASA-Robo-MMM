"""Where `SapienPlannerV2.IK` gets its solutions. Off by default: mplib, random restarts.

Two opt-in changes, chosen by environment variables so every planner and every call
site picks them up without new arguments:

``MIKASA_IK_STARTS=fixed``
    mplib's IK runs CLIK from the current joints, then from ``n_init_qpos - 1``
    configurations drawn from mplib's GLOBAL generator. The set of solutions — and so
    the branch a caller's nearest-first ordering picks — depends on how many draws
    happened before, and a refused call shifts every later plan (jezvgg/ManiSkill#6).
    ``fixed`` runs the same loop (same CLIK, joint-limit wrap, FCL check, threshold and
    0.1 rad de-duplication) from the current joints and then from a table of starts
    fixed per planner (half placed around the current posture, half over the whole joint
    range), and mplib's generator is never touched.
    The first time a request is asked it gets block 0 of the table, so the same
    situation gets the same answer; asking the SAME request again in the episode (the
    planners' retries, which counted on randomness) gets the next fixed block, so a
    retry explores instead of repeating a refused answer (measured on SeasonDish: one
    table for every retry made episodes 4x longer). b1k_rl_guys' controllers and
    BIMBO_EvoX's ArmServo use fixed starts the same way, current joints first.

``MIKASA_IK_BACKEND=curobo``
    cuRobo (GPU) proposes solutions seeded with the current joints, in a separate
    process under its own venv (``MIKASA_CUROBO_PY``, `curobo_client.py`); each one is
    refined by mplib's CLIK from there (same threshold as mplib) and kept only if
    mplib's FCL check passes against the whole planning world, as mplib's own IK does.
    cuRobo sees the robot (self-collision spheres), not the scene.

The defaults (``random``, ``mplib``) call mplib's ``Planner.IK`` unchanged.
"""

from __future__ import annotations

import os

import numpy as np

STARTS_ENV, BACKEND_ENV = "MIKASA_IK_STARTS", "MIKASA_IK_BACKEND"
#: The fixed-start table's generator seed; any constant works, it only has to never change.
FIXED_STARTS_SEED = 20261015


def starts_mode() -> str:
    mode = os.environ.get(STARTS_ENV, "random")
    if mode not in ("random", "fixed"):
        raise ValueError(f"{STARTS_ENV}={mode!r}: expected 'random' or 'fixed'")
    return mode


def backend() -> str:
    name = os.environ.get(BACKEND_ENV, "mplib")
    if name not in ("mplib", "curobo"):
        raise ValueError(f"{BACKEND_ENV}={name!r}: expected 'mplib' or 'curobo'")
    return name


def is_default() -> bool:
    return starts_mode() == "random" and backend() == "mplib"


def fixed_starts(planner, n: int, block: int = 0) -> np.ndarray:
    """``n`` configurations inside the planner's joint limits; block ``k`` is always the same.

    Cached on the planner. Revolute limits wider than one turn are sampled over one
    turn (the wrap maps the rest onto it); masked joints are overwritten by the caller.
    """
    cache = planner.__dict__.setdefault("_mikasa_fixed_starts", {})
    if (n, block) not in cache:
        lim = np.asarray(planner.joint_limits, dtype=np.float64).reshape(-1, 2)
        lo, hi = lim[:, 0].copy(), lim[:, 1].copy()
        wide = (hi - lo) > 2 * np.pi
        hi[wide] = lo[wide] + 2 * np.pi
        rng = np.random.default_rng([FIXED_STARTS_SEED, int(block)])
        cache[(n, block)] = rng.uniform(lo, hi, size=(n, len(lo)))
    return cache[(n, block)]


#: Fractions of each joint's range (one turn at most) for the starts placed around the
#: current posture; the rest of the table spans the whole range.
LOCAL_SPREADS = (0.05, 0.1, 0.2)


def starts_around(planner, start_qpos, n: int, block: int = 0) -> list:
    """``n`` fixed starts: half around ``start_qpos`` (deterministic offsets, a few
    spreads), half from the whole-range table. Solutions near the current posture are
    the ones a nearest-first caller takes and a joint line reaches; the whole-range half
    keeps the other branches reachable. b1k_rl_guys starts from the current joints and
    from postures near rest for the same reason."""
    n_local = n // 2
    table = fixed_starts(planner, n, block)
    lim = np.asarray(planner.joint_limits, dtype=np.float64).reshape(-1, 2)
    lo, hi = lim[:, 0], lim[:, 1]
    span = np.minimum(hi - lo, 2 * np.pi)
    unit = (table[:n_local] - lo) / np.maximum(hi - lo, 1e-9) * 2.0 - 1.0     # in [-1, 1], fixed
    out = []
    for i in range(n_local):
        frac = LOCAL_SPREADS[i % len(LOCAL_SPREADS)]
        out.append(np.clip(np.asarray(start_qpos, dtype=np.float64) + unit[i] * frac * span, lo, hi))
    return out + list(table[n_local:])


def retry_index(planner, goal_pose, start_qpos, mask) -> int:
    """How many times this exact request (goal to 1 mm / 0.01, start to 1e-3, mask) was
    asked before on this planner; counts it. Retries of a refused request get the next
    block of fixed starts instead of the same answer."""
    q = np.asarray(goal_pose.q, dtype=np.float64).reshape(-1)
    q = q if q[0] >= 0 else -q
    key = (tuple(np.round(np.asarray(goal_pose.p, dtype=np.float64).reshape(-1), 3)), tuple(np.round(q, 2)),
           tuple(np.round(np.asarray(start_qpos, dtype=np.float64), 3)),
           tuple(bool(m) for m in (mask if mask is not None else ())))
    seen = planner.__dict__.setdefault("_mikasa_ik_requests", {})
    k = seen.get(key, 0)
    seen[key] = k + 1
    return k


def solve_from_starts(planner, goal_pose, start_qpos, mask, starts, threshold: float,
                      verbose: bool = False):
    """mplib's ``Planner.IK`` loop over the given starts instead of random ones.

    Returns ``(status, q_goals)`` with mplib's status strings; ``q_goals`` is None on
    failure. Every check is mplib's own: CLIK, ``wrap_joint_limit``, the planning
    world's collision check, the pose distance threshold and the 0.1 rad uniqueness.
    """
    mask = np.zeros(len(start_qpos), dtype=bool) if mask is None or len(mask) == 0 \
        else np.asarray(mask, dtype=bool)
    start_qpos = np.asarray(start_qpos, dtype=np.float64)
    link = planner.link_name_2_idx[planner.move_group]
    move = planner.move_group_joint_indices
    planner.robot.set_qpos(start_qpos, True)
    min_dist, q_goals = 1e9, []
    for start in starts:
        qpos = np.asarray(start, dtype=np.float64).copy()
        qpos[mask] = start_qpos[mask]
        ik_qpos, ok, _ = planner.pinocchio_model.compute_IK_CLIK(link, goal_pose, qpos, mask)
        ok = ok and planner.wrap_joint_limit(ik_qpos)
        if ok:
            planner.planning_world.set_qpos_all(ik_qpos[move])
            collisions = planner.planning_world.check_collision()
            if collisions:
                ok = False
                if verbose:
                    for c in collisions:
                        print(f"Collision between {c.link_name1} of entity {c.object_name1} with "
                              f"{c.link_name2} of entity {c.object_name2}")
        if ok:
            planner.pinocchio_model.compute_forward_kinematics(ik_qpos)
            dist = goal_pose.distance(planner.pinocchio_model.get_link_pose(link))
            min_dist = min(min_dist, dist)
            if dist < threshold and all(np.linalg.norm(g[move] - ik_qpos[move]) >= 0.1 for g in q_goals):
                q_goals.append(ik_qpos)
    if q_goals:
        return "Success", q_goals
    if min_dist != 1e9:
        return f"IK Failed! Distance {min_dist} is greater than threshold={threshold}.", None
    return "IK Failed! Cannot find valid solution.", None


def ik(planner, goal_pose, start_qpos, mask=None, *, n_init_qpos: int = 20, threshold: float = 1e-3,
       return_closest: bool = False, verbose: bool = False):
    """The non-default IK paths; `SapienPlannerV2.IK` calls this only when not `is_default()`."""
    start_qpos = np.asarray(start_qpos, dtype=np.float64)
    if backend() == "curobo":
        from robots.fetch import curobo_client

        seeds = curobo_client.propose(planner, goal_pose, start_qpos, mask)
        starts = list(seeds)
        if starts_mode() == "fixed":       # cuRobo first, then the fixed table as before
            block = retry_index(planner, goal_pose, start_qpos, mask)
            starts += starts_around(planner, start_qpos, max(n_init_qpos - 1, 0), block)
        status, goals = solve_from_starts(planner, goal_pose, start_qpos, mask, [start_qpos] + starts,
                                          threshold, verbose)
    else:
        block = retry_index(planner, goal_pose, start_qpos, mask)
        starts = [start_qpos] + starts_around(planner, start_qpos, max(n_init_qpos - 1, 0), block)
        status, goals = solve_from_starts(planner, goal_pose, start_qpos, mask, starts, threshold, verbose)
    if goals is not None and return_closest:
        move = planner.move_group_joint_indices
        goals = min(goals, key=lambda g: float(np.linalg.norm(g[move] - start_qpos[move])))
    return status, goals
