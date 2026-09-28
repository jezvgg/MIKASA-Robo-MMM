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


def path_clear(planner, current, positions, *, initial_contacts=()):
    """Planning-model updates only; preserve all other joints and all fixtures."""
    p = planner.planner
    planner._last_path_collision = None
    move = list(p.move_group_joint_indices)
    current = np.asarray(current, dtype=float)
    p.robot.set_qpos(p.fold_qpos(current), True)
    allowed = {frozenset(pair) for pair in initial_contacts}
    for index, knot in enumerate(dense_samples(positions)):
        full = current.copy()
        full[move] = knot
        p.planning_world.set_qpos_all(p.fold_qpos(full)[move])
        if p.planning_world.is_state_colliding():
            # A held object can still touch its support at the starting state.
            # Permit only explicitly named pairs there; the very next sample
            # and every later sample must be entirely collision-free.
            if index == 0 and allowed and np.allclose(full, current, atol=1e-6):
                pairs = {frozenset((c.link_name1, c.link_name2))
                         for c in p.planning_world.check_collision()}
                if pairs and pairs <= allowed:
                    continue
            planner._last_path_collision = [(c.link_name1,c.link_name2)
                for c in p.planning_world.check_collision()]
            return False
    return True
