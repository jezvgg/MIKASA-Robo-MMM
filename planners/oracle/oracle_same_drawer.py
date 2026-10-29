"""SameDrawer-specific grasp stage using the profile's collection solver."""
from __future__ import annotations

import os

import mplib
import numpy as np

from planners.oracle import oracle_common as common

GRASP_ARRIVE_M = float(os.environ.get("MIKASA_GRASP_ARRIVE_M", "0.03"))


def try_grasp(env, planner, task, obj, grasp, reach, *, n_init_qpos: int,
              approach_max_knots: int, grasp_max_knots: int, grasp_knot_draws: int):
    """Execute the measured SameDrawer reach, sync, grasp, and contact-stop close."""
    reach_result = planner.static_manipulation(
        reach,
        n_init_qpos=int(n_init_qpos),
        by_line=True,
        skim_max_knots=int(approach_max_knots),
    )
    if reach_result == -1 or common.stopped_by_horizon(planner):
        return reach_result, False

    # The executed reach moves the apple; plan the grasp against the resulting world.
    planner.planner.update_from_simulation()
    grasp_result = planner.static_manipulation(
        grasp,
        n_init_qpos=int(n_init_qpos),
        max_knots=int(grasp_max_knots),
        knot_draws=int(grasp_knot_draws),
        knot_refuse=True,
    )
    if grasp_result == -1 or common.stopped_by_horizon(planner):
        return grasp_result, False

    tcp_error = float(planner.last_tcp_err)
    if tcp_error > GRASP_ARRIVE_M:
        common.say(env, "oracle", "grasp stroke fell short; not closing on air",
                   tcp_err=round(tcp_error, 3), tol=GRASP_ARRIVE_M)
        return grasp_result, False

    close_result = planner.close_gripper(
        stop_when=lambda: bool(common._np(task.agent.is_grasping(obj)).any())
    )
    if close_result == -1 or common.stopped_by_horizon(planner):
        return close_result, False
    return close_result, bool(common._np(task.agent.is_grasping(obj)).any())


def screw_plans(planner, target_tcp_pose, *, disable_lift_joint: bool = False) -> bool:
    """Return whether the collection solver's straight screw reaches ``target``."""
    only_manipulate = [True, True, True, bool(disable_lift_joint)] + [False] * 11
    result = planner.planner.plan_screw(
        mplib.Pose(p=np.asarray(target_tcp_pose.p), q=np.asarray(target_tcp_pose.q)),
        planner.robot.get_qpos().cpu().numpy()[0],
        time_step=planner.base_env.control_timestep,
        masked_joints=~np.array(only_manipulate),
        goal_tolerance=planner.ARM_SCREW_GOAL_TOLERANCE,
    )
    return str(result["status"]) == "Success"
