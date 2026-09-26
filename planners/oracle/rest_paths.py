"""Collision-checked, monotone joint paths to the unmodified robot rest pose."""
from itertools import product
import numpy as np
from planners.oracle import oracle_common as common


def monotone_knots(start, goal, powers, count=81):
    """Vary joint progress without an extra pose or a joint reversal."""
    alpha = np.linspace(0., 1., count)[:, None]
    return start[None] + (goal - start)[None] * alpha ** powers[None]


def curved_rest(env, planner, task, targets, *, who):
    """Try bounded progress curves when the direct rest line hits furniture."""
    p = planner.planner
    q = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
    goal = q.copy()
    joints = task.agent.robot.active_joints_map
    idx = lambda name: int(joints[name].active_index[0])
    for name, value in targets.items():
        goal[idx(name)] = value
    groups = [
        [idx('torso_lift_joint')],
        [idx('shoulder_pan_joint'), idx('shoulder_lift_joint')],
        [idx('elbow_flex_joint')],
        [idx('forearm_roll_joint'), idx('wrist_flex_joint'), idx('wrist_roll_joint')],
    ]
    move = p.move_group_joint_indices

    def clear(knots):
        p.robot.set_qpos(p.fold_qpos(q), True)
        for knot in knots:
            p.planning_world.set_qpos_all(p.fold_qpos(knot)[move])
            if p.planning_world.is_state_colliding():
                return False
        return True

    for exponents in product((1., 2., 4., .5), repeat=len(groups)):
        if all(x == 1 for x in exponents):
            continue  # The caller already checked the direct joint line.
        powers = np.ones(len(q))
        for indices, exponent in zip(groups, exponents):
            powers[indices] = exponent
        knots = monotone_knots(q, goal, powers)
        if not p.accepts(knots) or not clear(knots):
            continue
        try:
            times, pos, vel, acc, duration = p.TOPP(knots[:, move], task.control_timestep)
        except RuntimeError:
            continue
        full = np.broadcast_to(q, (len(pos), len(q))).copy()
        full[:, move] = pos
        if not p.accepts(full) or not clear(full):
            continue
        common.say(env, who, 'fold to canonical rest: curved joint path',
                   progress_powers=list(exponents), knots=len(pos))
        return planner.follow_forward_path_w_refinement(
            dict(status='Success', time=times, position=pos, velocity=vel,
                 acceleration=acc, duration=duration), refine=True)
    common.say(env, who, 'canonical rest curves refused')
    return -1
