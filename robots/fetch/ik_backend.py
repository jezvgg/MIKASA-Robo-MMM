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
    fixed once per planner from its own joint limits: the same request always gets the
    same solutions, and mplib's generator is never touched. b1k_rl_guys' controllers
    and BIMBO_EvoX's ArmServo do the same (current joints first, fixed starts after).

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


def fixed_starts(planner, n: int) -> np.ndarray:
    """``n`` configurations inside the planner's joint limits, the same for every call.

    Cached on the planner. Revolute limits wider than one turn are sampled over one
    turn (the wrap maps the rest onto it); masked joints are overwritten by the caller.
    """
    cache = planner.__dict__.setdefault("_mikasa_fixed_starts", {})
    if n not in cache:
        lim = np.asarray(planner.joint_limits, dtype=np.float64).reshape(-1, 2)
        lo, hi = lim[:, 0].copy(), lim[:, 1].copy()
        wide = (hi - lo) > 2 * np.pi
        hi[wide] = lo[wide] + 2 * np.pi
        cache[n] = np.random.default_rng(FIXED_STARTS_SEED).uniform(lo, hi, size=(n, len(lo)))
    return cache[n]


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
            starts += list(fixed_starts(planner, max(n_init_qpos - 1, 0)))
        status, goals = solve_from_starts(planner, goal_pose, start_qpos, mask, [start_qpos] + starts,
                                          threshold, verbose)
    else:
        starts = [start_qpos] + list(fixed_starts(planner, max(n_init_qpos - 1, 0)))
        status, goals = solve_from_starts(planner, goal_pose, start_qpos, mask, starts, threshold, verbose)
    if goals is not None and return_closest:
        move = planner.move_group_joint_indices
        goals = min(goals, key=lambda g: float(np.linalg.norm(g[move] - start_qpos[move])))
    return status, goals
