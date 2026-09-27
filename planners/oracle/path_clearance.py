"""Check the swept path between trajectory knots against the full planning world."""

import numpy as np


def dense_samples(positions, max_joint_step=0.025):
    """Include endpoints and bound the Euclidean joint displacement per sample."""
    positions = np.asarray(positions)
    if not len(positions):
        return
    yield positions[0]
    for start, end in zip(positions[:-1], positions[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / max_joint_step)))
        for index in range(1, count + 1):
            yield start + (end - start) * (index / count)


def path_clear(planner, current, positions):
    """Planning-model updates only; preserve all other joints and all fixtures."""
    p = planner.planner
    move = list(p.move_group_joint_indices)
    current = np.asarray(current, dtype=float)
    p.robot.set_qpos(p.fold_qpos(current), True)
    for knot in dense_samples(positions):
        full = current.copy()
        full[move] = knot
        p.planning_world.set_qpos_all(p.fold_qpos(full)[move])
        if p.planning_world.is_state_colliding():
            return False
    return True
