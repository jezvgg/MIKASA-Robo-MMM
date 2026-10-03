"""Incremental planner for ``MyRoboCasa_TakeItBackTray-v1``.

Current checkpoint: waypoints 1-16.  The planner completes the carry to the
tray, centers the held cup over it, lowers it, releases it, and verifies placement.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import cast

import gymnasium as gym
import mplib
import numpy as np
import sapien
import torch
from transforms3d.quaternions import axangle2quat, qmult
from mani_skill.agents.robots import Fetch
from mani_skill.utils.wrappers import RecordEpisode

from my_scenes.my_robocasa_takeitback_tray import MyRoboCasaSceneTakeItBackTray
from planners.takeitback_common import _grasp_pose, _tcp_to
from robots.fetch.extand import (
    PLANNING_TIME,
    RRT_RANGE,
    FetchMotionPlanningSapienSolver,
)
from utils.logging_utils import PlannerLogger, StreamingVideoRecorder, capture_stdout


def _repair_trajectory_metadata(run_dir: Path) -> None:
    """Keep RoboCasa reconfigure seeds replayable by ManiSkill's checker."""
    path = run_dir / "trajectory.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for episode in data.get("episodes", []):
        seed = episode.get("reset_kwargs", {}).get("seed")
        if isinstance(seed, list) and len(seed) == 1:
            seed = seed[0]
        if seed is not None and episode.get("episode_seed") != seed:
            episode["episode_seed"] = int(seed)
            changed = True
    if changed:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _counter_direction_toward_cup(
    task: MyRoboCasaSceneTakeItBackTray, agent: Fetch
) -> np.ndarray:
    """Choose countertop's long axis with its sign pointing toward the cup."""
    size = np.asarray(task.counter_size[:2], dtype=float)
    axis = np.array([1.0, 0.0]) if size[0] >= size[1] else np.array([0.0, 1.0])
    base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
    cup_xy = task.cup.pose.p[0].cpu().numpy()[:2]
    if np.dot(cup_xy - base_xy, axis) < 0:
        axis = -axis
    return np.r_[axis, 0.0]


def _heading(agent: Fetch) -> float:
    matrix = agent.base_link.pose.sp.to_transformation_matrix()
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _heading_error(agent: Fetch, direction: np.ndarray) -> float:
    current = np.array([np.cos(_heading(agent)), np.sin(_heading(agent))])
    target = direction[:2] / np.linalg.norm(direction[:2])
    cross = current[0] * target[1] - current[1] * target[0]
    return float(abs(np.arctan2(cross, np.dot(current, target))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Waypoint 1-16 planner for MyRoboCasa_TakeItBackTray-v1"
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--render-mode",
        choices=["rgb_array", "human", "sensors"],
        default="rgb_array",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def planning(
    env,
    seed: int,
    debug: bool = False,
    vis: bool | None = None,
    info: bool = False,
) -> bool:
    if vis is None:
        vis = env.unwrapped.render_mode == "human"

    task: MyRoboCasaSceneTakeItBackTray = env.unwrapped
    env.reset(seed=seed, options={"reconfigure": True})
    agent: Fetch = cast(Fetch, task.agent)

    planner = FetchMotionPlanningSapienSolver(
        env,
        base_pose=agent.robot.pose.sp,
        vis=vis,
        print_env_info=info,
        debug=debug,
    )

    # Keep cup as a real obstacle throughout planning; only grasp stages may
    # later opt into contact explicitly.
    import sapien.physx as physx
    from mplib.sapien_utils.conversion import convert_object_name

    cup_component = task.cup._objs[0].find_component_by_type(
        physx.PhysxRigidBaseComponent
    )
    cup_name = convert_object_name(cup_component.entity)
    acm = planner.planner.planning_world.get_allowed_collision_matrix()
    for link in agent.robot._objs[0].get_links():
        acm.set_entry(link.name, cup_name, False)

    env.track_object(task.cup, "cup")
    env.track_object(task.tray, "tray")
    env.track_object(agent.tcp, "robot_tcp")
    env.track_object(agent.base_link, "robot_base")
    env.log_event("start", "Waypoints 1-16 planning started")
    initial_tray_position = task.tray.pose.p[0].cpu().numpy()[:3].copy()
    initial_robot_position = agent.base_link.pose.p[0].cpu().numpy()[:3].copy()
    initial_tray_robot_delta = initial_tray_position - initial_robot_position
    env.log_event(
        "scene",
        "Initial tray and robot-base coordinates",
        tray_position=initial_tray_position,
        robot_base_position=initial_robot_position,
        tray_minus_robot_base=initial_tray_robot_delta,
        tray_robot_distance=float(np.linalg.norm(initial_tray_robot_delta[:2])),
    )
    print(
        "[SCENE] tray:",
        np.round(initial_tray_position, 4),
        "robot_base:",
        np.round(initial_robot_position, 4),
        "tray-robot:",
        np.round(initial_tray_robot_delta, 4),
    )

    # WAYPOINT 1: face along the countertop, with forward direction toward cup.
    target_direction = _counter_direction_toward_cup(task, agent)
    before = _heading(agent)
    env.log_event(
        "waypoint",
        "Waypoint 1: turn parallel to countertop toward cup",
        target_direction=target_direction,
        heading_before=before,
    )
    print(
        "[WAYPOINT 1] target direction:",
        np.round(target_direction, 3),
        "initial heading:",
        round(before, 4),
    )

    result = env.log_motion(
        "Waypoint 1 rotate",
        planner.rotate_base_z,
        target_direction,
    )
    if result == -1:
        env.log_event("error", "Waypoint 1 rotation failed")
        env.log_event("result", "Waypoint 1 failed", waypoint_success=False)
        return False

    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    after = _heading(agent)
    heading_error = _heading_error(agent, target_direction)
    if heading_error > np.deg2rad(5.0):
        env.log_event(
            "error",
            "Waypoint 1 did not reach parallel heading",
            heading_after=after,
            heading_error_deg=float(np.rad2deg(heading_error)),
        )
        env.log_event("result", "Waypoint 1 failed", waypoint_success=False)
        return False
    print(
        "[WAYPOINT 1] final heading:",
        round(after, 4),
        "error_deg:",
        round(float(np.rad2deg(heading_error)), 3),
    )
    env.log_event(
        "waypoint_complete",
        "Waypoint 1 complete",
        heading_after=after,
        heading_error_deg=float(np.rad2deg(heading_error)),
    )

    # WAYPOINT 2: move only along the countertop until the cup is beside the
    # base, i.e. the cup-base vector is perpendicular to the drive direction.
    axis = target_direction[:2]

    def along_gap() -> float:
        base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
        cup_xy = task.cup.pose.p[0].cpu().numpy()[:2]
        return float(np.dot(cup_xy - base_xy, axis))

    gap_before = along_gap()
    env.log_event(
        "waypoint",
        "Waypoint 2: drive parallel to countertop to cup perpendicular",
        along_gap_before=gap_before,
    )
    print("[WAYPOINT 2] along-gap before:", round(gap_before, 4))

    if gap_before > 0.03:
        result = env.log_motion(
            "Waypoint 2 drive",
            planner.drive_straight,
            gap_before,
            v=0.10,
            stop_when=lambda: along_gap() <= 0.02,
        )
        if result == -1:
            env.log_event("error", "Waypoint 2 drive failed")
            env.log_event("result", "Waypoint 2 failed", waypoint_success=False)
            return False
    else:
        planner.idle_steps(t=1)

    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    gap_after = abs(along_gap())
    success = gap_after <= 0.06
    base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
    cup_xy = task.cup.pose.p[0].cpu().numpy()[:2]
    lateral_gap = float(np.linalg.norm(cup_xy - base_xy))
    print(
        "[WAYPOINT 2] along-gap after:",
        round(gap_after, 4),
        "cup-base distance:",
        round(lateral_gap, 4),
        "success:",
        success,
    )
    env.log_event(
        "waypoint_complete" if success else "error",
        "Waypoint 2 complete" if success else "Waypoint 2 did not reach perpendicular position",
        along_gap_after=gap_after,
        cup_base_distance=lateral_gap,
    )
    if not success:
        env.log_event("result", "Waypoint 2 failed", waypoint_success=False)
        return False

    # WAYPOINT 3: turn in place toward the cup.
    base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
    cup_xy = task.cup.pose.p[0].cpu().numpy()[:2]
    to_cup = cup_xy - base_xy
    cup_distance = float(np.linalg.norm(to_cup))
    if cup_distance < 1e-6:
        env.log_event("error", "Waypoint 3 has no direction to cup")
        env.log_event("result", "Waypoint 3 failed", waypoint_success=False)
        return False
    cup_direction = np.r_[to_cup / cup_distance, 0.0]

    # Raise the torso to its transport limit before the collision-checked turn.
    body = agent.controller.controllers["body"]
    torso_before = float(body.qpos[0].cpu().numpy()[2])
    torso_target = max(torso_before, 0.386)
    if torso_target > torso_before + 1e-4:
        for i in range(100):
            arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            body_target = body.qpos[0].cpu().numpy().copy()
            body_target[2] = torso_before + (torso_target - torso_before) * (i + 1) / 100
            env.step(np.hstack([arm, planner.gripper_state, body_target, np.zeros(2)]))
        planner.planner.update_from_simulation()
        env.log_event(
            "safety",
            "Raise torso before waypoint 3 rotation for fixture clearance",
            torso_before=torso_before,
            torso_target=torso_target,
        )

    heading_before = _heading(agent)
    env.log_event(
        "waypoint",
        "Waypoint 3: turn toward cup",
        target_direction=cup_direction,
        heading_before=heading_before,
        cup_base_distance=cup_distance,
    )
    print(
        "[WAYPOINT 3] target direction:",
        np.round(cup_direction, 3),
        "cup-base distance:",
        round(cup_distance, 4),
    )

    result = env.log_motion(
        "Waypoint 3 rotate toward cup",
        planner.rotate_base_z,
        cup_direction,
    )
    if result == -1:
        env.log_event("error", "Waypoint 3 rotation failed")
        env.log_event("result", "Waypoint 3 failed", waypoint_success=False)
        return False

    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    heading_after = _heading(agent)
    heading_error = _heading_error(agent, cup_direction)
    success = heading_error <= np.deg2rad(5.0)
    print(
        "[WAYPOINT 3] final heading:",
        round(heading_after, 4),
        "error_deg:",
        round(float(np.rad2deg(heading_error)), 3),
        "success:",
        success,
    )
    env.log_event(
        "waypoint_complete" if success else "error",
        "Waypoint 3 complete" if success else "Waypoint 3 did not reach cup heading",
        heading_after=heading_after,
        heading_error_deg=float(np.rad2deg(heading_error)),
    )
    if not success:
        env.log_event("result", "Waypoint 3 failed", waypoint_success=False)
        return False
    # WAYPOINT 4: approach the countertop, leaving 20-30 cm between its front
    # edge and the Fetch base. The torso stays high so the arm clears fixtures.
    cup_before = task.cup.pose.p[0].cpu().numpy()[:3].copy()
    base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
    counter_center = np.asarray(task.counter_pos[:2], dtype=float)
    normal = np.array([-axis[1], axis[0]], dtype=float)
    if np.dot(base_xy - counter_center, normal) < 0:
        normal = -normal
    half_normal = 0.5 * float(
        abs(normal[0]) * task.counter_size[0]
        + abs(normal[1]) * task.counter_size[1]
    )
    counter_front = counter_center + normal * half_normal
    base_radius = float(getattr(task, "ROBOT_RADIUS", 0.35))
    desired_gap = 0.25
    target_base = counter_front + normal * (base_radius + desired_gap)
    approach_distance = float(np.dot(target_base - base_xy, cup_direction[:2]))
    env.log_event(
        "waypoint",
        "Waypoint 4: approach countertop with 20-30 cm clearance",
        approach_distance=approach_distance,
        desired_gap=desired_gap,
    )
    print(
        "[WAYPOINT 4] approach distance:",
        round(approach_distance, 4),
        "desired gap:",
        desired_gap,
    )

    if approach_distance > 0.03:
        result = env.log_motion(
            "Waypoint 4 drive toward countertop",
            planner.drive_straight,
            approach_distance,
            v=0.12,
            max_steps=400,
            stop_when=lambda: float(
                np.dot(target_base - agent.base_link.pose.p[0].cpu().numpy()[:2], cup_direction[:2])
            ) <= 0.02,
        )
        if result == -1:
            env.log_event("error", "Waypoint 4 drive failed")
            env.log_event("result", "Waypoint 4 failed", waypoint_success=False)
            return False
    else:
        planner.idle_steps(t=1)

    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
    actual_gap = float(np.dot(base_xy - counter_front, normal) - base_radius)
    cup_after = task.cup.pose.p[0].cpu().numpy()[:3]
    cup_shift = float(np.linalg.norm(cup_after - cup_before))
    success = 0.20 <= actual_gap <= 0.32 and cup_shift <= 0.02
    print(
        "[WAYPOINT 4] actual gap:",
        round(actual_gap, 4),
        "cup shift:",
        round(cup_shift, 4),
        "success:",
        success,
    )
    env.log_event(
        "waypoint_complete" if success else "error",
        "Waypoint 4 complete" if success else "Waypoint 4 clearance or cup check failed",
        actual_gap=actual_gap,
        cup_shift=cup_shift,
    )
    if not success:
        env.log_event("result", "Waypoint 4 failed", waypoint_success=False)
        return False

    # WAYPOINT 5: lower torso first, then use RRT for an open-hand pre-grasp
    # 10-15 cm in front of the cup. No base motion is mixed into this reach.
    cup_center = task.cup.pose.p[0].cpu().numpy()[:3].copy()
    body = agent.controller.controllers["body"]
    torso_before = float(body.qpos[0].cpu().numpy()[2])
    tcp_z_before = float(agent.tcp.pose.p[0][2])
    torso_target = 0.0
    env.log_event(
        "waypoint",
        "Waypoint 5: lower torso to cup height",
        torso_before=torso_before,
        torso_target=torso_target,
        tcp_z_before=tcp_z_before,
        cup_z=float(cup_center[2]),
    )
    print(
        "[WAYPOINT 5] torso:",
        round(torso_before, 4),
        "->",
        round(torso_target, 4),
        "cup z:",
        round(float(cup_center[2]), 4),
    )

    if torso_target < torso_before - 1e-4:
        for i in range(120):
            arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            body_target = body.qpos[0].cpu().numpy().copy()
            body_target[2] = torso_before + (torso_target - torso_before) * (i + 1) / 120
            env.step(np.hstack([arm, planner.gripper_state, body_target, np.zeros(2)]))
        planner.planner.update_from_simulation()

    torso_after = float(body.qpos[0].cpu().numpy()[2])
    if torso_after > torso_target + 0.05:
        env.log_event(
            "error",
            "Waypoint 5 torso did not lower",
            torso_after=torso_after,
            torso_target=torso_target,
        )
        env.log_event("result", "Waypoint 5 failed", waypoint_success=False)
        return False

    planner.open_gripper()
    mesh = task.cup.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        env.log_event("error", "Waypoint 5 cup collision mesh unavailable")
        env.log_event("result", "Waypoint 5 failed", waypoint_success=False)
        return False
    obb = mesh.bounding_box_oriented
    cup_center = np.asarray(obb.center_mass, dtype=float)
    approach = cup_direction.copy()
    approach[2] = 0.0
    approach /= np.linalg.norm(approach)
    closing = np.array([1.0, 0.0, 0.0])
    grasp_pose, pregrasp_pose = _grasp_pose(
        agent,
        obb,
        cup_center,
        approach,
        closing,
        raise_z=0.10,
        back_off=0.12,
        force_front=True,
    )
    # Rest arm collides with counter during torso descent. Low-torso starts can
    # reach directly; raised starts use high standoff plus deterministic wrist roll.
    tcp_q = agent.tcp.pose.q[0].cpu().numpy()
    tray_x = float(task.tray.pose.p[0].cpu().numpy()[0])
    x_delta = float(cup_center[0] - tray_x)
    low_torso = torso_before < 0.10
    if low_torso:
        selected_angle = 0.0
        result = planner.static_manipulation(
            pregrasp_pose,
            n_init_qpos=100,
            disable_lift_joint=True,
            draws=8,
        )
    else:
        if cup_center[0] > 2.4 and x_delta > 0.0:
            selected_angle = np.pi
        elif cup_center[0] >= 2.44:
            selected_angle = -np.pi / 2
        else:
            selected_angle = np.pi / 2
        candidate_q = qmult(axangle2quat(approach, selected_angle), tcp_q)
        pregrasp_pose = sapien.Pose(p=pregrasp_pose.p, q=candidate_q)
        if selected_angle == np.pi:
            grasp_pose = sapien.Pose(p=grasp_pose.p, q=candidate_q)
        above_pose = sapien.Pose(
            p=np.asarray(pregrasp_pose.p) + np.array([0.0, 0.0, 0.25]),
            q=pregrasp_pose.q,
        )
        env.log_event(
            "waypoint",
            "Waypoint 5: collision-checked high pre-grasp",
            pregrasp_distance=0.14,
            target_position=pregrasp_pose.p,
            wrist_roll_angle=float(selected_angle),
            x_delta=x_delta,
        )
        result = planner.move_to_pose_with_RRTConnect(
            above_pose,
            n_init_qpos=100,
            mask=[True, True, True, True] + [False] * 11,
        )
        if result != -1:
            result = planner.static_manipulation(
                pregrasp_pose,
                n_init_qpos=100,
                disable_lift_joint=True,
                draws=8,
            )
    if result == -1:
        for fallback_angle in (np.pi / 2, -np.pi / 2, np.pi):
            if fallback_angle == selected_angle:
                continue
            fallback_q = qmult(axangle2quat(approach, fallback_angle), tcp_q)
            fallback_pre = sapien.Pose(p=pregrasp_pose.p, q=fallback_q)
            fallback_above = sapien.Pose(
                p=np.asarray(fallback_pre.p) + np.array([0.0, 0.0, 0.25]),
                q=fallback_pre.q,
            )
            high_retry = planner.move_to_pose_with_RRTConnect(
                fallback_above,
                n_init_qpos=100,
                mask=[True, True, True, True] + [False] * 11,
            )
            if high_retry == -1:
                continue
            retry = planner.static_manipulation(
                fallback_pre,
                n_init_qpos=100,
                disable_lift_joint=True,
                draws=8,
            )
            if retry != -1:
                selected_angle = fallback_angle
                pregrasp_pose = fallback_pre
                if fallback_angle == np.pi:
                    grasp_pose = sapien.Pose(p=grasp_pose.p, q=fallback_q)
                result = retry
                env.log_event(
                    "waypoint_plan",
                    "Waypoint 5 used alternate wrist roll",
                    wrist_roll_angle=float(fallback_angle),
                )
                break
    if result == -1:
        env.log_event("error", "Waypoint 5 pre-grasp execution failed")
        env.log_event("result", "Waypoint 5 failed", waypoint_success=False)
        return False
    planner.planner.update_from_simulation()
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    pregrasp_distance = _tcp_to(agent, cup_center)
    tcp_level_error = abs(float(tcp[2] - cup_center[2]))
    cup_shift = float(
        np.linalg.norm(task.cup.pose.p[0].cpu().numpy()[:3] - cup_before)
    )
    success = (
        0.10 <= pregrasp_distance <= 0.16
        and tcp_level_error <= 0.07
        and cup_shift <= 0.04
    )
    print(
        "[WAYPOINT 5] TCP-cup distance:",
        round(pregrasp_distance, 4),
        "level error:",
        round(tcp_level_error, 4),
        "cup shift:",
        round(cup_shift, 4),
        "success:",
        success,
    )
    env.log_event(
        "waypoint_complete" if success else "error",
        "Waypoint 5 complete" if success else "Waypoint 5 pre-grasp check failed",
        pregrasp_distance=pregrasp_distance,
        tcp_level_error=tcp_level_error,
        torso_after=torso_after,
        cup_shift=cup_shift,
    )
    if not success:
        env.log_event("result", "Stopped after waypoint 5", waypoint_success=False)
        return False

    # WAYPOINT 6: move the open hand from pre-grasp to the actual grasp pose.
    # Compare straight IK/screw against RRT, but keep base and torso fixed.
    env.log_event(
        "waypoint",
        "Waypoint 6: move open hand to grasp pose",
        target_position=grasp_pose.p,
    )
    grasp_acm_links = []
    for link in agent.robot._objs[0].get_links():
        name = link.name.lower()
        if "gripper" in name or "finger" in name:
            acm.set_entry(link.name, cup_name, True)
            grasp_acm_links.append(link.name)

    arm_mask = [True, True, True, True] + [False] * 11
    current_qpos = agent.robot.get_qpos().cpu().numpy()[0]
    planner.planner.update_from_simulation()
    grasp_target = mplib.Pose(p=grasp_pose.p, q=grasp_pose.q)
    screw_result = planner.planner.plan_screw(
        grasp_target,
        current_qpos,
        time_step=env.unwrapped.control_timestep,
        masked_joints=~np.asarray(arm_mask),
        goal_tolerance=planner.ARM_SCREW_GOAL_TOLERANCE,
    )
    rrt_result = planner.planner.plan_pose(
        grasp_target,
        current_qpos,
        time_step=env.unwrapped.control_timestep,
        wrt_world=True,
        verbose=True,
        planning_time=PLANNING_TIME,
        rrt_range=RRT_RANGE,
        simplify=True,
        mask=arm_mask,
        fixed_joint_indices=[0, 1, 2, 3],
        n_init_qpos=100,
    )
    screw_ok = screw_result["status"] == "Success"
    rrt_ok = rrt_result["status"] == "Success"
    screw_knots = len(screw_result.get("position", [])) if screw_ok else None
    rrt_knots = len(rrt_result.get("position", [])) if rrt_ok else None
    selected = None
    method = None
    if screw_ok and (not rrt_ok or screw_knots <= rrt_knots):
        selected, method = screw_result, "ik_screw"
    elif rrt_ok:
        selected, method = rrt_result, "rrt"
    env.log_event(
        "waypoint_plan",
        "Compared IK screw and RRT grasp approaches",
        ik_status=screw_result["status"],
        ik_knots=screw_knots,
        rrt_status=rrt_result["status"],
        rrt_knots=rrt_knots,
        selected=method,
        allowed_grasp_links=len(grasp_acm_links),
    )
    if selected is None:
        env.log_event("error", "Waypoint 6 grasp approach planning failed")
        env.log_event("result", "Stopped after waypoint 6", waypoint_success=False)
        return False

    planner.follow_path(selected)
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    grasp_error = float(np.linalg.norm(tcp - np.asarray(grasp_pose.p)))
    tcp_cup_distance = _tcp_to(agent, cup_center)
    cup_shift = float(
        np.linalg.norm(task.cup.pose.p[0].cpu().numpy()[:3] - cup_before)
    )
    torso_grasp_error = abs(
        float(body.qpos[0].cpu().numpy()[2] - torso_after)
    )
    waypoint6_success = (
        grasp_error <= 0.05
        and tcp_cup_distance <= 0.10
        and torso_grasp_error <= 0.05
        and cup_shift <= 0.04
    )
    print(
        "[WAYPOINT 6] grasp error:",
        round(grasp_error, 4),
        "TCP-cup distance:",
        round(tcp_cup_distance, 4),
        "cup shift:",
        round(cup_shift, 4),
        "method:",
        method,
        "success:",
        waypoint6_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint6_success else "error",
        "Waypoint 6 complete" if waypoint6_success else "Waypoint 6 grasp approach check failed",
        method=method,
        grasp_error=grasp_error,
        tcp_cup_distance=tcp_cup_distance,
        torso_grasp_error=torso_grasp_error,
        cup_shift=cup_shift,
    )
    if not waypoint6_success:
        env.log_event("result", "Stopped after waypoint 6", waypoint_success=False)
        return False

    # WAYPOINT 7: close the gripper without moving base, torso, or arm.
    env.log_event("waypoint", "Waypoint 7: close gripper on cup")
    planner.close_gripper(t=12)
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    grasped = bool(task.agent.is_grasping(task.cup).item())
    tcp_cup_distance = _tcp_to(agent, cup_center)
    cup_shift = float(
        np.linalg.norm(task.cup.pose.p[0].cpu().numpy()[:3] - cup_before)
    )
    waypoint7_success = (
        grasped
        and tcp_cup_distance <= 0.10
        and cup_shift <= 0.08
    )
    print(
        "[WAYPOINT 7] grasped:",
        grasped,
        "TCP-cup distance:",
        round(tcp_cup_distance, 4),
        "cup shift:",
        round(cup_shift, 4),
        "success:",
        waypoint7_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint7_success else "error",
        "Waypoint 7 complete" if waypoint7_success else "Waypoint 7 grasp check failed",
        grasped=grasped,
        tcp_cup_distance=tcp_cup_distance,
        cup_shift=cup_shift,
    )
    if not waypoint7_success:
        env.log_event("result", "Stopped after waypoint 7", waypoint_success=False)
        return False

    # WAYPOINT 8: raise the torso with the closed gripper; base and arm stay
    # fixed. Verify cup lift and load-bearing grasp before tray transport.
    env.log_event("waypoint", "Waypoint 8: raise torso with grasped cup")
    torso_before_lift = float(body.qpos[0].cpu().numpy()[2])
    torso_target_lift = max(torso_before_lift, 0.30)
    cup_z_before_lift = float(task.cup.pose.p[0][2])
    cup_xy_before_lift = task.cup.pose.p[0].cpu().numpy()[:2].copy()
    tcp_z_before_lift = float(agent.tcp.pose.p[0][2])
    for i in range(150):
        arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_target = body.qpos[0].cpu().numpy().copy()
        body_target[2] = torso_before_lift + (
            torso_target_lift - torso_before_lift
        ) * (i + 1) / 150
        env.step(np.hstack([arm, planner.gripper_state, body_target, np.zeros(2)]))
    planner.planner.update_from_simulation()
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    torso_after_lift = float(body.qpos[0].cpu().numpy()[2])
    cup_z_after_lift = float(task.cup.pose.p[0][2])
    cup_xy_shift = float(
        np.linalg.norm(task.cup.pose.p[0].cpu().numpy()[:2] - cup_xy_before_lift)
    )
    grasped_after_lift = bool(task.agent.is_grasping(task.cup).item())
    tcp_cup_distance = _tcp_to(agent, task.cup.pose.p[0].cpu().numpy()[:3])
    cup_rise = cup_z_after_lift - cup_z_before_lift
    tcp_rise = float(agent.tcp.pose.p[0][2]) - tcp_z_before_lift
    waypoint8_success = (
        grasped_after_lift
        and torso_after_lift >= torso_target_lift - 0.05
        and cup_rise >= 0.15
        and tcp_rise >= 0.15
        and tcp_cup_distance <= 0.12
        and cup_xy_shift <= 0.02
    )
    print(
        "[WAYPOINT 8] grasped:",
        grasped_after_lift,
        "torso:",
        round(torso_after_lift, 4),
        "cup rise:",
        round(cup_rise, 4),
        "TCP-cup distance:",
        round(tcp_cup_distance, 4),
        "success:",
        waypoint8_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint8_success else "error",
        "Waypoint 8 complete" if waypoint8_success else "Waypoint 8 lift check failed",
        grasped=grasped_after_lift,
        torso_before=torso_before_lift,
        torso_after=torso_after_lift,
        torso_target=torso_target_lift,
        cup_z_before=cup_z_before_lift,
        cup_z_after=cup_z_after_lift,
        cup_rise=cup_rise,
        tcp_rise=tcp_rise,
        tcp_cup_distance=tcp_cup_distance,
        cup_xy_shift=cup_xy_shift,
    )
    env.log_event("result", "Stopped after waypoint 8", waypoint_success=waypoint8_success)
    if not waypoint8_success:
        return False

    # WAYPOINT 9: back away from the countertop before any transport turn.
    env.log_event("waypoint", "Waypoint 9: retreat from countertop")
    base_before_retreat = agent.base_link.pose.p[0].cpu().numpy()[:2].copy()
    cup_xy_before_retreat = task.cup.pose.p[0].cpu().numpy()[:2].copy()
    planner.drive_straight(-0.15, v=0.10, max_steps=350)
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    base_after_retreat = agent.base_link.pose.p[0].cpu().numpy()[:2]
    cup_xy_after_retreat = task.cup.pose.p[0].cpu().numpy()[:2]
    base_delta = base_after_retreat - base_before_retreat
    retreat_along_cup = float(np.dot(base_delta, cup_direction[:2]))
    base_travel = float(np.linalg.norm(base_delta))
    cup_xy_shift = float(np.linalg.norm(cup_xy_after_retreat - cup_xy_before_retreat))
    grasped_after_retreat = bool(task.agent.is_grasping(task.cup).item())
    tcp_cup_distance = _tcp_to(agent, task.cup.pose.p[0].cpu().numpy()[:3])
    cup_motion_error = abs(cup_xy_shift - base_travel)
    waypoint9_success = (
        retreat_along_cup <= -0.08
        and base_travel >= 0.10
        and grasped_after_retreat
        and tcp_cup_distance <= 0.12
        and cup_motion_error <= 0.08
    )
    print(
        "[WAYPOINT 9] retreat:",
        round(retreat_along_cup, 4),
        "base travel:",
        round(base_travel, 4),
        "grasped:",
        grasped_after_retreat,
        "TCP-cup distance:",
        round(tcp_cup_distance, 4),
        "cup xy shift:",
        round(cup_xy_shift, 4),
        "cup motion error:",
        round(cup_motion_error, 4),
        "success:",
        waypoint9_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint9_success else "error",
        "Waypoint 9 complete" if waypoint9_success else "Waypoint 9 retreat check failed",
        retreat_along_cup=retreat_along_cup,
        base_travel=base_travel,
        grasped=grasped_after_retreat,
        tcp_cup_distance=tcp_cup_distance,
        cup_xy_shift=cup_xy_shift,
        cup_motion_error=cup_motion_error,
    )
    env.log_event("result", "Stopped after waypoint 9", waypoint_success=waypoint9_success)
    if not waypoint9_success:
        return False

    # WAYPOINT 10: turn in place to face tray's countertop direction.
    env.log_event("waypoint", "Waypoint 10: turn toward tray")
    for link in agent.robot._objs[0].get_links():
        acm.set_entry(link.name, cup_name, True)
    base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
    tray_xy = task.tray.pose.p[0].cpu().numpy()[:2]
    tray_direction = np.r_[axis, 0.0]
    if np.dot(tray_xy - base_xy, tray_direction[:2]) < 0:
        tray_direction = -tray_direction
    heading_before = _heading(agent)
    result = env.log_motion(
        "Waypoint 10 rotate toward tray",
        planner.rotate_base_z,
        tray_direction,
    )
    if result == -1:
        env.log_event("error", "Waypoint 10 rotation failed")
        env.log_event("result", "Stopped after waypoint 10", waypoint_success=False)
        return False
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    heading_after = _heading(agent)
    heading_error = _heading_error(agent, tray_direction)
    grasped_after_turn = bool(task.agent.is_grasping(task.cup).item())
    tcp_cup_distance = _tcp_to(agent, task.cup.pose.p[0].cpu().numpy()[:3])
    waypoint10_success = (
        heading_error <= np.deg2rad(5.0)
        and grasped_after_turn
        and tcp_cup_distance <= 0.12
    )
    print(
        "[WAYPOINT 10] heading error:",
        round(float(np.rad2deg(heading_error)), 3),
        "grasped:",
        grasped_after_turn,
        "TCP-cup distance:",
        round(tcp_cup_distance, 4),
        "success:",
        waypoint10_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint10_success else "error",
        "Waypoint 10 complete" if waypoint10_success else "Waypoint 10 turn check failed",
        target_direction=tray_direction,
        heading_before=heading_before,
        heading_after=heading_after,
        heading_error_deg=float(np.rad2deg(heading_error)),
        grasped=grasped_after_turn,
        tcp_cup_distance=tcp_cup_distance,
    )
    env.log_event("result", "Stopped after waypoint 10", waypoint_success=waypoint10_success)
    if not waypoint10_success:
        return False

    # WAYPOINT 11: repeat the simple countertop drive, reversing the previous
    # command so the base moves toward the tray instead of away from it.
    env.log_event("waypoint", "Waypoint 11: drive toward tray")
    cup_before_align = task.cup.pose.p[0].cpu().numpy()[:3].copy()
    tray_before_align = task.tray.pose.p[0].cpu().numpy()[:3].copy()
    base_before_align = agent.base_link.pose.p[0].cpu().numpy()[:3].copy()
    arm_before_align = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    torso_before_align = float(body.qpos[0].cpu().numpy()[2])
    tray_robot_delta_before = tray_before_align[:2] - base_before_align[:2]
    robot_axis_error_before = abs(
        float(np.dot(tray_robot_delta_before, tray_direction[:2]))
    )
    drive_distance = float(
        np.dot(tray_robot_delta_before, tray_direction[:2])
    )
    target_base = base_before_align.copy()
    target_base[:2] += tray_direction[:2] * drive_distance
    env.log_event(
        "waypoint",
        "Waypoint 11: align robot base with tray",
        tray_position=tray_before_align,
        robot_base_position=base_before_align,
        tray_minus_robot_base=tray_robot_delta_before,
        robot_axis_error_before=robot_axis_error_before,
        drive_distance=drive_distance,
        target_base=target_base,
    )
    if abs(drive_distance) > 0.03:
        result = env.log_motion(
            "Waypoint 11 drive toward tray",
            planner.move_base_forward,
            target_base,
            freeze_arm=True,
        )
        if result == -1:
            env.log_event("error", "Waypoint 11 base translation failed")
            env.log_event("result", "Stopped after waypoint 11", waypoint_success=False)
            return False
    else:
        planner.idle_steps(t=1)
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    cup_after_align = task.cup.pose.p[0].cpu().numpy()[:3]
    tray_after_align = task.tray.pose.p[0].cpu().numpy()[:3]
    base_after_align = agent.base_link.pose.p[0].cpu().numpy()[:3]
    arm_after_align = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    torso_after_align = float(body.qpos[0].cpu().numpy()[2])
    base_delta = base_after_align[:2] - base_before_align[:2]
    base_toward_tray = float(np.dot(base_delta, tray_direction[:2]))
    tray_robot_delta_after = tray_after_align[:2] - base_after_align[:2]
    robot_axis_error_after = abs(
        float(np.dot(tray_robot_delta_after, tray_direction[:2]))
    )
    grasped_after_align = bool(task.agent.is_grasping(task.cup).item())
    tcp_cup_distance = _tcp_to(agent, cup_after_align)
    base_heading_error = _heading_error(agent, tray_direction)
    short_already_aligned = abs(drive_distance) <= 0.08 and robot_axis_error_after <= 0.03
    waypoint11_success = (
        robot_axis_error_after <= 0.08
        and (
            short_already_aligned
            or (
                base_toward_tray >= 0.08
                and np.linalg.norm(base_delta) >= min(abs(drive_distance) * 0.5, 0.10)
            )
        )
        and grasped_after_align
        and tcp_cup_distance <= 0.12
        and np.max(np.abs(arm_after_align - arm_before_align)) <= 0.08
        and abs(torso_after_align - torso_before_align) <= 0.05
        and base_heading_error <= np.deg2rad(5.0)
    )
    print(
        "[WAYPOINT 11] robot-tray axis error:",
        round(robot_axis_error_after, 4),
        "base toward tray:",
        round(base_toward_tray, 4),
        "base travel:",
        round(float(np.linalg.norm(base_delta)), 4),
        "grasped:",
        grasped_after_align,
        "TCP-cup distance:",
        round(tcp_cup_distance, 4),
        "success:",
        waypoint11_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint11_success else "error",
        "Waypoint 11 complete" if waypoint11_success else "Waypoint 11 robot-tray alignment failed",
        tray_position=tray_after_align,
        robot_base_position=base_after_align,
        tray_minus_robot_base=tray_robot_delta_after,
        robot_axis_error_before=robot_axis_error_before,
        robot_axis_error_after=robot_axis_error_after,
        drive_distance=drive_distance,
        base_toward_tray=base_toward_tray,
        base_travel=float(np.linalg.norm(base_delta)),
        grasped=grasped_after_align,
        tcp_cup_distance=tcp_cup_distance,
        arm_motion=float(np.max(np.abs(arm_after_align - arm_before_align))),
        torso_motion=abs(torso_after_align - torso_before_align),
        heading_error_deg=float(np.rad2deg(base_heading_error)),
    )
    env.log_event("result", "Stopped after waypoint 11", waypoint_success=waypoint11_success)
    if not waypoint11_success:
        return False

    # WAYPOINT 12: turn the robot to face the tray across the counter. The base
    # is already aligned with the tray along the countertop axis; no translation
    # or arm motion is mixed into this turn.
    env.log_event("waypoint", "Waypoint 12: turn perpendicular toward tray")
    tray_before_turn = task.tray.pose.p[0].cpu().numpy()[:3].copy()
    base_before_turn = agent.base_link.pose.p[0].cpu().numpy()[:3].copy()
    arm_before_turn = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    torso_before_turn = float(body.qpos[0].cpu().numpy()[2])
    tray_approach_axis = np.array([-axis[1], axis[0]], dtype=float)
    if np.dot(tray_before_turn[:2] - base_before_turn[:2], tray_approach_axis) < 0:
        tray_approach_axis = -tray_approach_axis
    tray_approach_direction = np.r_[tray_approach_axis, 0.0]
    heading_before = _heading(agent)
    env.log_event(
        "waypoint",
        "Waypoint 12: collision-checked turn toward tray",
        tray_position=tray_before_turn,
        robot_base_position=base_before_turn,
        tray_minus_robot_base=tray_before_turn - base_before_turn,
        target_direction=tray_approach_direction,
        heading_before=heading_before,
    )
    result = env.log_motion(
        "Waypoint 12 rotate toward tray",
        planner.rotate_base_z,
        tray_approach_direction,
    )
    if result == -1:
        env.log_event("error", "Waypoint 12 rotation failed")
        env.log_event("result", "Stopped after waypoint 12", waypoint_success=False)
        return False
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    tray_after_turn = task.tray.pose.p[0].cpu().numpy()[:3].copy()
    base_after_turn = agent.base_link.pose.p[0].cpu().numpy()[:3].copy()
    arm_after_turn = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    torso_after_turn = float(body.qpos[0].cpu().numpy()[2])
    heading_after = _heading(agent)
    heading_error = _heading_error(agent, tray_approach_direction)
    base_drift = float(np.linalg.norm(base_after_turn[:2] - base_before_turn[:2]))
    robot_axis_error = abs(float(np.dot(
        tray_after_turn[:2] - base_after_turn[:2], axis
    )))
    grasped_after_turn = bool(task.agent.is_grasping(task.cup).item())
    tcp_cup_distance = _tcp_to(agent, task.cup.pose.p[0].cpu().numpy()[:3])
    waypoint12_success = (
        heading_error <= np.deg2rad(5.0)
        and robot_axis_error <= 0.08
        and base_drift <= 0.08
        and grasped_after_turn
        and tcp_cup_distance <= 0.12
        and np.max(np.abs(arm_after_turn - arm_before_turn)) <= 0.08
        and abs(torso_after_turn - torso_before_turn) <= 0.05
    )
    print(
        "[WAYPOINT 12] heading error:",
        round(float(np.rad2deg(heading_error)), 3),
        "robot-tray axis error:",
        round(robot_axis_error, 4),
        "base drift:",
        round(base_drift, 4),
        "grasped:",
        grasped_after_turn,
        "TCP-cup distance:",
        round(tcp_cup_distance, 4),
        "success:",
        waypoint12_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint12_success else "error",
        "Waypoint 12 complete" if waypoint12_success else "Waypoint 12 turn check failed",
        tray_position=tray_after_turn,
        robot_base_position=base_after_turn,
        tray_minus_robot_base=tray_after_turn - base_after_turn,
        target_direction=tray_approach_direction,
        heading_before=heading_before,
        heading_after=heading_after,
        heading_error_deg=float(np.rad2deg(heading_error)),
        robot_axis_error=robot_axis_error,
        base_drift=base_drift,
        grasped=grasped_after_turn,
        tcp_cup_distance=tcp_cup_distance,
        arm_motion=float(np.max(np.abs(arm_after_turn - arm_before_turn))),
        torso_motion=abs(torso_after_turn - torso_before_turn),
    )
    env.log_event("result", "Stopped after waypoint 12", waypoint_success=waypoint12_success)
    if not waypoint12_success:
        return False

    # WAYPOINT 13: advance straight toward the tray with the robot now facing
    # it. Compute the target from the live cup/base offset, not a fixed arm
    # model; the collision-checked base-only plan keeps arm and torso fixed.
    env.log_event("waypoint", "Waypoint 13: advance cup toward tray")
    cup_before_advance = task.cup.pose.p[0].cpu().numpy()[:3].copy()
    tray_before_advance = task.tray.pose.p[0].cpu().numpy()[:3].copy()
    base_before_advance = agent.base_link.pose.p[0].cpu().numpy()[:3].copy()
    arm_before_advance = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    torso_before_advance = float(body.qpos[0].cpu().numpy()[2])
    cup_tray_distance_before = float(
        np.linalg.norm(cup_before_advance[:2] - tray_before_advance[:2])
    )
    drive_distance = float(np.dot(
        tray_before_advance[:2] - cup_before_advance[:2],
        tray_approach_direction[:2],
    ))
    target_base = base_before_advance.copy()
    target_base[:2] += tray_approach_direction[:2] * drive_distance
    env.log_event(
        "waypoint",
        "Waypoint 13: collision-checked advance toward tray",
        tray_position=tray_before_advance,
        robot_base_position=base_before_advance,
        tray_minus_cup=tray_before_advance - cup_before_advance,
        cup_tray_distance_before=cup_tray_distance_before,
        drive_distance=drive_distance,
        target_base=target_base,
    )
    if abs(drive_distance) > 0.03:
        result = env.log_motion(
            "Waypoint 13 advance toward tray",
            planner.move_base_forward,
            target_base,
            freeze_arm=True,
        )
        if result == -1:
            env.log_event(
                "waypoint_plan",
                "Waypoint 13 retrying collision-checked base drive in chunks",
            )
            result = 0
            for fraction in (0.25, 0.50, 0.75, 1.0):
                chunk_target = base_before_advance.copy()
                chunk_target[:2] += tray_approach_direction[:2] * drive_distance * fraction
                result = env.log_motion(
                    "Waypoint 13 chunked advance",
                    planner.move_base_forward,
                    chunk_target,
                    freeze_arm=True,
                )
                if result == -1:
                    break
        if result == -1:
            # Retry collision-checked translation with extra torso clearance.  The
            # arm remains fixed; torso and base are still separate motions.
            torso_retry_before = float(body.qpos[0].cpu().numpy()[2])
            torso_retry_target = max(torso_retry_before, 0.386)
            if torso_retry_target > torso_retry_before + 1e-4:
                for i in range(80):
                    arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                    body_target = body.qpos[0].cpu().numpy().copy()
                    body_target[2] = torso_retry_before + (
                        torso_retry_target - torso_retry_before
                    ) * (i + 1) / 80
                    env.step(np.hstack([arm, planner.gripper_state, body_target, np.zeros(2)]))
                planner.planner.update_from_simulation()
                env.log_event(
                    "waypoint_plan",
                    "Waypoint 13 retry with raised torso clearance",
                    torso_before=torso_retry_before,
                    torso_target=torso_retry_target,
                )
                result = env.log_motion(
                    "Waypoint 13 raised-torso advance",
                    planner.move_base_forward,
                    target_base,
                    freeze_arm=True,
                )
                if result != -1 and torso_retry_target > torso_retry_before + 1e-4:
                    for i in range(80):
                        arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                        body_target = body.qpos[0].cpu().numpy().copy()
                        body_target[2] = torso_retry_target + (
                            torso_retry_before - torso_retry_target
                        ) * (i + 1) / 80
                        env.step(np.hstack([arm, planner.gripper_state, body_target, np.zeros(2)]))
                    for _ in range(80):
                        arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                        body_target = body.qpos[0].cpu().numpy().copy()
                        body_target[2] = torso_retry_before
                        env.step(np.hstack([arm, planner.gripper_state, body_target, np.zeros(2)]))
                        if abs(float(body.qpos[0].cpu().numpy()[2]) - torso_retry_before) <= 0.01:
                            break
                    planner.planner.update_from_simulation()
        if result == -1:
            if torso_retry_target > torso_retry_before + 1e-4:
                for i in range(80):
                    arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                    body_target = body.qpos[0].cpu().numpy().copy()
                    body_target[2] = torso_retry_target + (
                        torso_retry_before - torso_retry_target
                    ) * (i + 1) / 80
                    env.step(np.hstack([arm, planner.gripper_state, body_target, np.zeros(2)]))
                for _ in range(80):
                    arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                    body_target = body.qpos[0].cpu().numpy().copy()
                    body_target[2] = torso_retry_before
                    env.step(np.hstack([arm, planner.gripper_state, body_target, np.zeros(2)]))
                    if abs(float(body.qpos[0].cpu().numpy()[2]) - torso_retry_before) <= 0.01:
                        break
                planner.planner.update_from_simulation()

            # The last resort is still monitored for new robot/environment
            # contacts; abort rather than driving through a physical collision.
            baseline_contacts = set(planner.touching_now())
            sign = -1.0 if drive_distance < 0 else 1.0
            guarded_result = None
            for _ in range(500):
                arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                body_target = body.qpos[0].cpu().numpy().copy()
                action = planner._compose(
                    arm,
                    body_target,
                    np.array([sign * 0.08, 0.0]),
                )
                guarded_result = planner._step(action)
                if planner.truncated:
                    break
                new_contacts = set(planner.touching_now()) - baseline_contacts
                if new_contacts:
                    env.log_event(
                        "error",
                        "Waypoint 13 guarded base drive contacted obstacle",
                        contacts=sorted(new_contacts),
                    )
                    guarded_result = -1
                    break
                if float(
                    np.dot(
                        tray_before_advance[:2]
                        - task.cup.pose.p[0].cpu().numpy()[:2],
                        tray_approach_direction[:2],
                    )
                ) <= 0.04:
                    break
            result = guarded_result if guarded_result is not None else -1
        if result == -1:
            env.log_event("error", "Waypoint 13 base translation failed")
            env.log_event("result", "Stopped after waypoint 13", waypoint_success=False)
            return False
    else:
        planner.idle_steps(t=1)
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    cup_after_advance = task.cup.pose.p[0].cpu().numpy()[:3]
    tray_after_advance = task.tray.pose.p[0].cpu().numpy()[:3]
    base_after_advance = agent.base_link.pose.p[0].cpu().numpy()[:3]
    arm_after_advance = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    torso_after_advance = float(body.qpos[0].cpu().numpy()[2])
    base_delta = base_after_advance[:2] - base_before_advance[:2]
    cup_tray_distance_after = float(
        np.linalg.norm(cup_after_advance[:2] - tray_after_advance[:2])
    )
    cup_depth_error = abs(float(np.dot(
        cup_after_advance[:2] - tray_after_advance[:2],
        tray_approach_direction[:2],
    )))
    robot_axis_error = abs(float(np.dot(
        tray_after_advance[:2] - base_after_advance[:2], axis
    )))
    grasped_after_advance = bool(task.agent.is_grasping(task.cup).item())
    tcp_cup_distance = _tcp_to(agent, cup_after_advance)
    heading_error = _heading_error(agent, tray_approach_direction)
    waypoint13_success = (
        cup_depth_error <= 0.08
        and cup_tray_distance_after <= 0.22
        and cup_tray_distance_after < cup_tray_distance_before
        and robot_axis_error <= 0.08
        and np.linalg.norm(base_delta) >= min(abs(drive_distance) * 0.5, 0.10)
        and grasped_after_advance
        and tcp_cup_distance <= 0.12
        and np.max(np.abs(arm_after_advance - arm_before_advance)) <= 0.08
        and abs(torso_after_advance - torso_before_advance) <= 0.05
        and heading_error <= np.deg2rad(5.0)
    )
    print(
        "[WAYPOINT 13] cup-tray distance:",
        round(cup_tray_distance_after, 4),
        "before:",
        round(cup_tray_distance_before, 4),
        "depth error:",
        round(cup_depth_error, 4),
        "base travel:",
        round(float(np.linalg.norm(base_delta)), 4),
        "grasped:",
        grasped_after_advance,
        "TCP-cup distance:",
        round(tcp_cup_distance, 4),
        "success:",
        waypoint13_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint13_success else "error",
        "Waypoint 13 complete" if waypoint13_success else "Waypoint 13 tray-approach check failed",
        tray_position=tray_after_advance,
        robot_base_position=base_after_advance,
        tray_minus_cup=tray_after_advance - cup_after_advance,
        drive_distance=drive_distance,
        cup_tray_distance_before=cup_tray_distance_before,
        cup_tray_distance_after=cup_tray_distance_after,
        cup_depth_error=cup_depth_error,
        robot_axis_error=robot_axis_error,
        base_travel=float(np.linalg.norm(base_delta)),
        grasped=grasped_after_advance,
        tcp_cup_distance=tcp_cup_distance,
        arm_motion=float(np.max(np.abs(arm_after_advance - arm_before_advance))),
        torso_motion=abs(torso_after_advance - torso_before_advance),
        heading_error_deg=float(np.rad2deg(heading_error)),
    )
    env.log_event("result", "Completed waypoint 13", waypoint_success=waypoint13_success)
    if not waypoint13_success:
        return False

    # WAYPOINT 14: move the grasped cup horizontally over the tray center while
    # still high in the air. Base and torso remain fixed; the arm-only screw
    # plan checks the whole path before execution.
    env.log_event("waypoint", "Waypoint 14: center cup over tray")
    cup_before_center = task.cup.pose.p[0].cpu().numpy()[:3].copy()
    tray_before_center = task.tray.pose.p[0].cpu().numpy()[:3].copy()
    base_before_center = agent.base_link.pose.p[0].cpu().numpy()[:3].copy()
    torso_before_center = float(body.qpos[0].cpu().numpy()[2])
    tcp_before_center = agent.tcp.pose.p[0].cpu().numpy()[:3].copy()
    tcp_orientation = agent.tcp.pose.q[0].cpu().numpy().copy()
    target_tcp_position = tcp_before_center.copy()
    target_tcp_position[:2] += tray_before_center[:2] - cup_before_center[:2]
    arm_only_mask = [False] * 4 + [True] * 11
    current_qpos = agent.robot.get_qpos().cpu().numpy()[0]
    target_tcp_pose = mplib.Pose(p=target_tcp_position, q=tcp_orientation)
    screw_result = planner.planner.plan_screw(
        target_tcp_pose,
        current_qpos,
        time_step=env.unwrapped.control_timestep,
        masked_joints=arm_only_mask,
        goal_tolerance=(0.03, np.deg2rad(5.0)),
    )
    selected_center_plan = screw_result if screw_result["status"] == "Success" else None
    center_method = "ik_screw" if selected_center_plan is not None else None
    if selected_center_plan is None:
        rrt_result = planner.planner.plan_pose(
            target_tcp_pose,
            current_qpos,
            time_step=env.unwrapped.control_timestep,
            wrt_world=True,
            verbose=True,
            planning_time=PLANNING_TIME,
            rrt_range=RRT_RANGE,
            simplify=True,
            mask=[True] * 4 + [False] * 11,
            fixed_joint_indices=[0, 1, 2, 3],
            n_init_qpos=100,
        )
        if rrt_result["status"] == "Success":
            selected_center_plan = rrt_result
            center_method = "rrt"
    env.log_event(
        "waypoint_plan",
        "Planned arm-only cup centering over tray",
        method=center_method,
        screw_status=screw_result["status"],
        target_tcp_position=target_tcp_position,
        cup_position=cup_before_center,
        tray_position=tray_before_center,
    )
    if selected_center_plan is None:
        env.log_event("error", "Waypoint 14 cup-centering plan failed")
        env.log_event("result", "Stopped after waypoint 14", waypoint_success=False)
        return False
    planner.follow_path(selected_center_plan)
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    cup_after_center = task.cup.pose.p[0].cpu().numpy()[:3]
    base_after_center = agent.base_link.pose.p[0].cpu().numpy()[:3]
    arm_after_center = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    torso_after_center = float(body.qpos[0].cpu().numpy()[2])
    cup_tray_xy_error = float(np.linalg.norm(
        cup_after_center[:2] - tray_before_center[:2]
    ))
    cup_z_shift = abs(float(cup_after_center[2] - cup_before_center[2]))
    base_drift = float(np.linalg.norm(base_after_center[:2] - base_before_center[:2]))
    torso_drift = abs(torso_after_center - torso_before_center)
    grasped_after_center = bool(task.agent.is_grasping(task.cup).item())
    tcp_cup_distance = _tcp_to(agent, cup_after_center)
    waypoint14_success = (
        cup_tray_xy_error <= 0.08
        and cup_z_shift <= 0.06
        and base_drift <= 0.06
        and torso_drift <= 0.05
        and grasped_after_center
        and tcp_cup_distance <= 0.12
    )
    print(
        "[WAYPOINT 14] cup-tray XY error:",
        round(cup_tray_xy_error, 4),
        "cup z shift:",
        round(cup_z_shift, 4),
        "base drift:",
        round(base_drift, 4),
        "grasped:",
        grasped_after_center,
        "TCP-cup distance:",
        round(tcp_cup_distance, 4),
        "method:",
        center_method,
        "success:",
        waypoint14_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint14_success else "error",
        "Waypoint 14 complete" if waypoint14_success else "Waypoint 14 cup-centering check failed",
        method=center_method,
        tray_position=tray_before_center,
        cup_position_before=cup_before_center,
        cup_position_after=cup_after_center,
        robot_base_position=base_after_center,
        cup_tray_xy_error=cup_tray_xy_error,
        cup_z_shift=cup_z_shift,
        base_drift=base_drift,
        torso_drift=torso_drift,
        grasped=grasped_after_center,
        tcp_cup_distance=tcp_cup_distance,
    )
    env.log_event("result", "Completed waypoint 14", waypoint_success=waypoint14_success)
    if not waypoint14_success:
        return False

    # WAYPOINT 15: lower the closed gripper and cup to the tray surface.
    env.log_event("waypoint", "Waypoint 15: lower cup onto tray")
    tray_before_lower = task.tray.pose.p[0].cpu().numpy()[:3].copy()
    cup_before_lower = task.cup.pose.p[0].cpu().numpy()[:3].copy()
    base_before_lower = agent.base_link.pose.p[0].cpu().numpy()[:3].copy()
    arm_before_lower = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    torso_before_lower = float(body.qpos[0].cpu().numpy()[2])
    tray_top = float(tray_before_lower[2] + task.tray_half[2])
    cup_rest_z = tray_top + float(task.cup_half[2])
    lower_steps = 0
    for _ in range(800):
        cup_z = float(task.cup.pose.p[0][2])
        torso_z = float(body.qpos[0].cpu().numpy()[2])
        if cup_z <= cup_rest_z + 0.015 or torso_z <= 0.02:
            break
        arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_target = body.qpos[0].cpu().numpy().copy()
        body_target[2] = max(0.02, torso_z - 0.005)
        env.step(np.hstack([arm, planner.gripper_state, body_target, np.zeros(2)]))
        lower_steps += 1
    planner.planner.update_from_simulation()
    planner.idle_steps(t=30)
    planner.planner.update_from_simulation()
    cup_after_lower = task.cup.pose.p[0].cpu().numpy()[:3]
    base_after_lower = agent.base_link.pose.p[0].cpu().numpy()[:3]
    arm_after_lower = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    torso_after_lower = float(body.qpos[0].cpu().numpy()[2])
    grasped_before_release = bool(task.agent.is_grasping(task.cup).item())
    tcp_cup_distance = _tcp_to(agent, cup_after_lower)
    cup_xy_shift = float(np.linalg.norm(cup_after_lower[:2] - cup_before_lower[:2]))
    base_drift = float(np.linalg.norm(base_after_lower[:2] - base_before_lower[:2]))
    cup_rest_error = abs(float(cup_after_lower[2] - cup_rest_z))
    waypoint15_success = (
        cup_after_lower[2] <= cup_rest_z + 0.12
        and torso_after_lower < torso_before_lower - 0.10
        and grasped_before_release
        and tcp_cup_distance <= 0.12
        and cup_xy_shift <= 0.10
        and base_drift <= 0.10
    )
    print(
        "[WAYPOINT 15] cup z:",
        round(float(cup_after_lower[2]), 4),
        "target:",
        round(cup_rest_z, 4),
        "rest error:",
        round(cup_rest_error, 4),
        "torso:",
        round(torso_before_lower, 4),
        "->",
        round(torso_after_lower, 4),
        "grasped:",
        grasped_before_release,
        "success:",
        waypoint15_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint15_success else "error",
        "Waypoint 15 complete" if waypoint15_success else "Waypoint 15 lowering check failed",
        tray_position=tray_before_lower,
        cup_position_before=cup_before_lower,
        cup_position_after=cup_after_lower,
        robot_base_position=base_after_lower,
        tray_top=tray_top,
        cup_rest_z=cup_rest_z,
        cup_rest_error=cup_rest_error,
        lower_steps=lower_steps,
        torso_before=torso_before_lower,
        torso_after=torso_after_lower,
        grasped=grasped_before_release,
        tcp_cup_distance=tcp_cup_distance,
        cup_xy_shift=cup_xy_shift,
        base_drift=base_drift,
    )
    env.log_event("result", "Completed waypoint 15", waypoint_success=waypoint15_success)
    if not waypoint15_success:
        return False

    # WAYPOINT 16: release without moving the base, arm, or torso, then let the
    # cup settle before using the task's real tray-success predicate.
    env.log_event("waypoint", "Waypoint 16: release cup on tray")
    cup_before_release = task.cup.pose.p[0].cpu().numpy()[:3].copy()
    planner.open_gripper(t=12, ramp=12)
    planner.idle_steps(t=60)
    planner.planner.update_from_simulation()
    cup_after_release = task.cup.pose.p[0].cpu().numpy()[:3]
    tray_after_release = task.tray.pose.p[0].cpu().numpy()[:3]
    grasped_after_release = bool(task.agent.is_grasping(task.cup).item())
    cup_xy_error = np.abs(cup_after_release[:2] - tray_after_release[:2])
    tray_xy_tol = task.tray_half[:2] + 0.05
    tray_top_after_release = float(tray_after_release[2] + task.tray_half[2])
    cup_z_error = abs(float(
        cup_after_release[2] - tray_top_after_release - task.cup_half[2]
    ))
    cup_speed = float(torch.linalg.norm(task.cup.linear_velocity[0]).item())
    cup_angular_speed = float(torch.linalg.norm(task.cup.angular_velocity[0]).item())
    evaluation = task.evaluate()
    task_success = bool(evaluation["success"].item())
    waypoint16_success = (
        task_success
        and not grasped_after_release
        and bool(np.all(cup_xy_error <= tray_xy_tol))
        and cup_z_error <= 0.10
        and cup_speed <= 0.1
        and cup_angular_speed <= 0.2
    )
    print(
        "[WAYPOINT 16] task success:",
        task_success,
        "grasped:",
        grasped_after_release,
        "cup-tray xy error:",
        np.round(cup_xy_error, 4),
        "z error:",
        round(cup_z_error, 4),
        "speed:",
        round(cup_speed, 4),
        "success:",
        waypoint16_success,
    )
    env.log_event(
        "waypoint_complete" if waypoint16_success else "error",
        "Waypoint 16 complete" if waypoint16_success else "Waypoint 16 placement check failed",
        cup_position_before=cup_before_release,
        cup_position_after=cup_after_release,
        tray_position=tray_after_release,
        grasped=grasped_after_release,
        cup_xy_error=cup_xy_error,
        tray_xy_tolerance=tray_xy_tol,
        cup_z_error=cup_z_error,
        cup_speed=cup_speed,
        cup_angular_speed=cup_angular_speed,
        task_success=task_success,
    )
    env.log_event("result", "Planner completed after waypoint 16", success=waypoint16_success)
    return waypoint16_success


if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    from mplib.pymp import set_global_seed

    set_global_seed(seed)
    run_id = f"takeitback_tray_wp16_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "MyRoboCasa_TakeItBackTray-v1",
        num_envs=1,
        render_mode=None if args.no_video else args.render_mode,
        obs_mode="state" if args.no_video else "rgb",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
        sim_config=dict(scene_config=dict(cpu_workers=1, enable_enhanced_determinism=True)),
    )
    if not args.no_video:
        env = StreamingVideoRecorder(env, output_dir=str(run_dir), video_fps=30)
    env = RecordEpisode(
        env,
        output_dir=str(run_dir),
        trajectory_name="trajectory",
        save_trajectory=True,
        save_video=False,
        source_type="motionplanning",
        source_desc="TakeItBack tray waypoint 16 complete",
    )
    env = PlannerLogger(
        env,
        log_dir=run_dir,
        name=f"takeitback_tray_wp16_seed{seed}",
        log_freq=args.log_freq,
        run_dir=run_dir,
    )
    env.action_space.seed(seed)

    waypoint_success = False
    try:
        with capture_stdout(env.dir / "console.log"):
            waypoint_success = planning(
                env,
                seed,
                debug=args.debug,
                info=args.info,
            )
    finally:
        env.close()
    _repair_trajectory_metadata(run_dir)
    print(f"[INFO] waypoint_16_success={waypoint_success}")
    if not args.no_video:
        print(f"[INFO] Video recording saved in '{run_dir}/'")
