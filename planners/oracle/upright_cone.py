"""Bounded alternative to exact-upright compensation, never a fallback ladder.

Call inside the existing measured upright_path. This helper performs no IK,
changes no physics, and tries only the two existing monotone arm schedules.
The caller owns the attached-payload collision context and final pour preview.
"""
import math
import time

import numpy as np

from planners.oracle.path_clearance import dense_samples, path_clear
from planners.oracle.upright_payload import (
    elbow_only, kinematics, upright_wrist_candidates,
)

CONE_DEGREES = 12.0
MAX_WRIST_KNOT_DELTA = 0.20
ARM_SCHEDULES = ((1.0, 1.0), (2.0, 1.0))
DIRECTIONS = np.asarray([[0.0, 0.0, 1.0]] + [
    [math.sin(math.radians(tilt)) * math.cos(azimuth),
     math.sin(math.radians(tilt)) * math.sin(azimuth),
     math.cos(math.radians(tilt))]
    for tilt in (6.0, 12.0)
    for azimuth in np.arange(8) * math.pi / 4
])
DIRECTION_TILTS = np.asarray([0.0] + [6.0] * 8 + [12.0] * 8)


def _tilt(axis):
    return math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0))))


def _inside_limits(positions, limits):
    """Physical URDF bounds; epsilon is float32 representation tolerance only."""
    return bool(np.isfinite(positions).all()
                and np.all(positions >= limits[:, 0] - 1e-7)
                and np.all(positions <= limits[:, 1] + 1e-7))


def _compact_terminal(task, fk, hand_object):
    hand_tcp = (task.agent.robot.links_map['gripper_link'].pose[0].sp.inv()
                * task.agent.tcp.pose[0].sp).to_transformation_matrix()
    world_to_base = np.linalg.inv(task.agent.base_link.pose[0].sp.to_transformation_matrix())

    def accepted(q):
        hand = world_to_base @ fk.matrix(q)
        tcp = (hand @ hand_tcp)[:3, 3]
        payload = (hand @ hand_object)[:3, 3]
        return bool(.10 <= tcp[0] <= .40 and abs(tcp[1]) <= .20
                    and np.linalg.norm(payload[:2]) <= .40)
    return accepted


def _geometric_chain(planner, task, current, goal, hand_object, *,
                     shoulder_power, elbow_power, terminal):
    """One best chain per analytic candidate; no retry of alternate chains."""
    started = time.monotonic()
    fk = kinematics(planner, task)
    wrist = [fk.indices[n] for n in ('wrist_flex_joint', 'wrist_roll_joint')]
    shoulder = [fk.indices[n] for n in ('shoulder_pan_joint', 'shoulder_lift_joint')]
    elbow = [fk.indices[n] for n in ('upperarm_roll_joint', 'elbow_flex_joint')]
    roll_column = list(planner._roll_indices).index(wrist[1])
    span_limit = math.pi - planner.planner.margin
    history_low = float(planner._roll_low[roll_column])
    history_high = float(planner._roll_high[roll_column])
    wrist_limits = task.agent.robot.get_qlimits()[0].cpu().numpy()[wrist].astype(float)
    wrist_limits[1] = [max(wrist_limits[1, 0], -span_limit, history_high - span_limit),
                       min(wrist_limits[1, 1], span_limit, history_low + span_limit)]
    local_axis = hand_object[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    compact = _compact_terminal(task, fk, hand_object) if terminal == 'carry' else None
    count = max(81, int(np.ceil(np.linalg.norm(goal - current) / .02)) + 1)
    full_without_wrist = []
    nodes = [current[None, wrist].copy()]
    parents = []
    cost = np.asarray([0.0])
    low, high = np.asarray([history_low]), np.asarray([history_high])

    for knot_index, alpha in enumerate(np.linspace(0.0, 1.0, count)[1:], 1):
        progress = np.full(len(current), alpha)
        progress[shoulder] = alpha ** shoulder_power
        progress[elbow] = alpha ** elbow_power
        q = current + progress * (goal - current)
        q[wrist] = 0.0
        full_without_wrist.append(q)
        parent_rotation = fk.matrix(q, 'wrist_roll_link')[:3, :3]
        axis_in_wrist = parent_rotation.T @ fk.matrix(q)[:3, :3] @ local_axis
        candidates, tilts = [], []
        final = knot_index == count - 1
        if final and terminal == 'hover':
            # The caller's IK endpoint includes the full desired TCP pose.
            # A different upright wrist branch is not an equivalent endpoint.
            final_tilt = _tilt(fk.matrix(goal)[:3, :3] @ local_axis)
            if (final_tilt <= CONE_DEGREES + 1e-7
                    and _inside_limits(goal[wrist], wrist_limits)):
                candidates.append(goal[wrist].copy())
                tilts.append(final_tilt)
        else:
            for direction, tilt in zip(DIRECTIONS, DIRECTION_TILTS):
                solutions = upright_wrist_candidates(
                    axis_in_wrist, parent_rotation.T @ direction,
                    current[wrist], wrist_limits)
                for values in solutions:
                    if any(np.linalg.norm(values - prior) < 1e-8 for prior in candidates):
                        continue
                    if final and compact is not None:
                        endpoint = q.copy(); endpoint[wrist] = values
                        if not compact(endpoint):
                            continue
                    candidates.append(values)
                    tilts.append(float(tilt))
        if not candidates:
            return None, dict(reason='cone_no_terminal_or_axis_solution', alpha=float(alpha),
                              seconds=time.monotonic() - started)
        candidates = np.asarray(candidates)
        delta = candidates[:, None, :] - nodes[-1][None, :, :]
        next_low = np.minimum(candidates[:, None, 1], low[None, :])
        next_high = np.maximum(candidates[:, None, 1], high[None, :])
        valid = ((np.max(np.abs(delta), axis=2) <= MAX_WRIST_KNOT_DELTA + 1e-9)
                 & (next_high - next_low <= span_limit + 1e-9))
        scores = (cost[None, :] + np.sum(delta * delta, axis=2)
                  + 1e-5 * (np.asarray(tilts)[:, None] / CONE_DEGREES) ** 2)
        scores[~valid] = np.inf
        parent = scores.argmin(axis=1)
        best = scores[np.arange(len(candidates)), parent]
        keep = np.flatnonzero(np.isfinite(best))
        if not len(keep):
            return None, dict(reason='cone_disconnected', alpha=float(alpha),
                              seconds=time.monotonic() - started)
        parents.append(parent[keep])
        nodes.append(candidates[keep])
        cost = best[keep]
        low = next_low[keep, parent[keep]]
        high = next_high[keep, parent[keep]]

    selected = int(np.argmin(cost))
    reverse = []
    for knot_index in range(count - 1, 0, -1):
        q = full_without_wrist[knot_index - 1].copy()
        q[wrist] = nodes[knot_index][selected]
        reverse.append(q)
        selected = int(parents[knot_index - 1][selected])
    full = np.vstack([current] + reverse[::-1])
    return full, dict(reason='cone_chain', seconds=time.monotonic() - started,
                      knots=count, max_wrist_knot_delta=float(
                          np.max(np.abs(np.diff(full[:, wrist], axis=0)))))


def upright_cone_path(planner, task, current, goal, hand_object, *, terminal='hover'):
    """Return one checked <=12 degree path or None; no exact-up fallback.

    terminal='hover': preserve the IK goal, including both wrist values.
    terminal='carry': choose compensated wrists subject to the existing compact
    TCP/payload footprint. The shoulder/elbow/core goal remains unchanged.
    """
    if terminal not in ('hover', 'carry'):
        raise ValueError('terminal must be hover or carry')
    current = np.asarray(current, dtype=float)
    goal = np.asarray(goal, dtype=float)
    p = planner.planner
    failures = []; planner._upright_failures = failures
    fk = kinematics(planner, task)
    local_axis = hand_object[:3, :3] @ np.asarray(task.cfg.pour_axis_body)
    limits = task.agent.robot.get_qlimits()[0].cpu().numpy().astype(float)
    moves = list(p.move_group_joint_indices)
    if current.shape != goal.shape or not np.isfinite(current).all() or not np.isfinite(goal).all():
        failures.append('cone_nonfinite_or_mismatched_input'); return None
    # Base stays fixed throughout transfer. No adjustment of recorded state.
    base = [fk.indices[n] for n in ('root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint')]
    if not np.allclose(current[base], goal[base], atol=1e-8, rtol=0.0):
        failures.append('cone_base_motion_forbidden'); return None
    if not _inside_limits(current, limits):
        failures.append('cone_initial_physical_limit'); return None
    initial_tilt = _tilt(fk.matrix(current)[:3, :3] @ local_axis)
    if initial_tilt > CONE_DEGREES + 1e-7:
        failures.append('cone_initial_tilt_limit'); return None
    with elbow_only(planner, task):
        if not p.accepts(current):
            failures.append('cone_initial_collection_limit'); return None
        for shoulder_power, elbow_power in ARM_SCHEDULES:
            full, diagnostic = _geometric_chain(
                planner, task, current, goal, hand_object, shoulder_power=shoulder_power,
                elbow_power=elbow_power, terminal=terminal)
            diagnostic.update(shoulder_power=shoulder_power, elbow_power=elbow_power)
            if full is None:
                failures.append(diagnostic); continue
            if not _inside_limits(full, limits) or not p.accepts(full):
                failures.append(dict(diagnostic, reason='cone_geometric_joint_limit')); continue
            if not path_clear(planner, current, full[:, moves]):
                failures.append(dict(diagnostic, reason='cone_collision',
                                     pairs=getattr(planner, '_last_path_collision', None))); continue
            try:
                times, pos, vel, acc, duration = p.TOPP(full[:, moves], task.control_timestep)
            except RuntimeError as error:
                failures.append(dict(diagnostic, reason='cone_TOPP', error=str(error))); continue
            timed = np.repeat(current[None, :], len(pos), axis=0); timed[:, moves] = pos
            if not len(pos) or not _inside_limits(timed, limits) or not p.accepts(pos, move_group=True):
                failures.append(dict(diagnostic, reason='cone_timed_joint_limit')); continue
            if not np.allclose(pos[-1], full[-1, moves], atol=1e-6, rtol=0.0):
                failures.append(dict(diagnostic, reason='cone_timed_terminal_changed')); continue
            if not path_clear(planner, current, pos):
                failures.append(dict(diagnostic, reason='cone_timed_collision',
                                     pairs=getattr(planner, '_last_path_collision', None))); continue
            max_tilt = 0.0
            for knot in dense_samples(pos):
                q = current.copy(); q[moves] = knot
                max_tilt = max(max_tilt, _tilt(fk.matrix(q)[:3, :3] @ local_axis))
            if max_tilt > CONE_DEGREES + 1e-7:
                failures.append(dict(diagnostic, reason='cone_timed_tilt_limit',
                                     max_tilt_deg=max_tilt)); continue
            return dict(status='Success', time=times, position=pos, velocity=vel,
                        acceleration=acc, duration=duration,
                        predicted_max_tilt_deg=max_tilt, cone_terminal=terminal,
                        cone_diagnostic=diagnostic)
    return None
