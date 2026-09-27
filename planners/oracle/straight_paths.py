"""Plan and execute a Cartesian stroke without a joint-space fallback."""

import mplib
import numpy as np
from planners.oracle.path_clearance import path_clear


def straight_plan(planner, pose, *, current=None, freeze_lift=None):
    """Return the actual checked path, so execution never replans into an RRT."""
    if current is None:
        current = planner.robot.get_qpos()[0].cpu().numpy()
    planner._straight_failures = []
    for freeze in (True, False) if freeze_lift is None else (freeze_lift,):
        result = planner.planner.plan_screw(
            mplib.Pose(p=pose.p, q=pose.q), current,
            time_step=planner.base_env.control_timestep,
            masked_joints=~np.array([True, True, True, freeze] + [False] * 11),
            goal_tolerance=planner.ARM_SCREW_GOAL_TOLERANCE,
        )
        planner._straight_failures.append(result.get("status"))
        if (result.get("status") == "Success"
                and path_clear(planner, current, result["position"])):
            return result
    return None


def execute_straight(planner, path, *, stop_when=None):
    """Finish contact changes on an action-pair boundary; never refine past a stop."""
    if path is None:
        return -1
    def paired_stop():
        return (int(planner.base_env.elapsed_steps[0]) % 2 == 0 and stop_when())
    return planner.follow_forward_path_w_refinement(
        path, refine=False, stop_when=paired_stop if stop_when else None)
