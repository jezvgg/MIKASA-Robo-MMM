import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch
from mani_skill.utils.wrappers import RecordEpisode
from my_scenes.my_robocasa_takeitback_tray import MyRoboCasaSceneTakeItBackTray
from robots.fetch.extand import FetchMotionPlanningSapienSolver
from utils.logging_utils import PlannerLogger, StreamingVideoRecorder, capture_stdout


COUNTER_DIRECTION = np.array([-1.0, 0.0, 0.0])
ALIGN_TOL = np.deg2rad(5.0)
SAFE_TORSO = 0.30
BASE_STANDOFF = 0.40
CUP_X_GAP = 0.25
CUP_BACK_GAP = 0.10
HEAD_PAN_LIMITS = (-1.57, 1.57)
HEAD_TILT_LIMITS = (-0.76, 1.45)
HEAD_LOOK_TOL = np.deg2rad(12.0)
PREGRASP_GAP = 0.12
GRIPPER_OPEN = 1
GRIPPER_CLOSED = -1
TORSO_LIFT_DELTA = 0.08
TRAY_DROP_GAP = 0.01
TRAY_DROP_TOL = 0.03


def _repair_trajectory_metadata(run_dir):
    """Make RoboCasa reconfigure seeds replayable by ManiSkill's checker."""
    path = Path(run_dir) / "trajectory.json"
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incremental planner for MyRoboCasa_TakeItBackTray-v1"
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        choices=["rgb_array", "human", "sensors"],
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def _heading(agent: Fetch) -> np.ndarray:
    direction = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0].copy()
    direction[2] = 0.0
    return direction / np.linalg.norm(direction)


def _head_look_at(agent: Fetch, target_pos: np.ndarray) -> tuple[float, float]:
    head = next(
        link for link in agent.robot.get_links()
        if link.get_name() == "head_camera_link"
    )
    head_pos = np.asarray(head.pose.sp.p, dtype=float)
    target = np.asarray(target_pos, dtype=float).reshape(3)
    delta = target - head_pos
    base_forward = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
    base_yaw = float(np.arctan2(base_forward[1], base_forward[0]))
    pan = np.arctan2(delta[1], delta[0]) - base_yaw
    pan = (pan + np.pi) % (2 * np.pi) - np.pi
    horizontal = float(np.hypot(delta[0], delta[1]))
    tilt = float(np.arctan2(-delta[2], horizontal))
    return (
        float(np.clip(pan, *HEAD_PAN_LIMITS)),
        float(np.clip(tilt, *HEAD_TILT_LIMITS)),
    )


def planning(env, seed, debug=False, vis=None, info=False) -> bool:
    """Execute waypoints 1-16; later waypoints are added after review."""
    unwenv: MyRoboCasaSceneTakeItBackTray = env.unwrapped
    env.reset(seed=seed, options={"reconfigure": True})
    agent: Fetch = cast(Fetch, unwenv.agent)

    planner = FetchMotionPlanningSapienSolver(
        env,
        base_pose=agent.robot.pose.sp,
        vis=bool(vis),
        print_env_info=info,
        debug=debug,
    )

    # Solver and environment both use absolute joint targets.
    planner.control_mode = "pd_joint_pos"
    env.track_object(unwenv.cup, "cup")
    env.track_object(unwenv.tray, "tray")
    env.track_object(agent.base_link, "robot_base")
    env.track_object(agent.tcp, "robot_tcp")
    env.log_event("start", "Incremental planner started")

    def drive_fixed_arm(distance, v=0.10, max_steps=350):
        """Drive base while keeping arm/body joint targets fixed."""
        arm_target = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
        body_target = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
        start_xy = agent.base_link.pose.sp.p[:2].copy()
        heading = _heading(agent)[:2]
        sign = -1.0 if distance < 0.0 else 1.0
        out = None
        for _ in range(max_steps):
            action = planner._compose(
                arm_target,
                body_target,
                np.array([sign * abs(v), 0.0]),
            )
            out = planner._step(action)
            if planner.truncated:
                break
            progress = float(
                np.dot(agent.base_link.pose.sp.p[:2] - start_xy, heading)
            )
            if (distance >= 0.0 and progress >= distance - 0.02) or (
                distance < 0.0 and progress <= distance + 0.02
            ):
                break
        return out

    def rotate_direct(direction, max_steps=240):
        """Fallback yaw-only turn when conservative sweep rejects a quarter turn."""
        arm_target = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
        body_target = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
        out = None
        for _ in range(max_steps):
            angle = planner._yaw_to(direction)
            if abs(angle) <= ALIGN_TOL:
                return out
            action = planner._compose(
                arm_target,
                body_target,
                np.array([0.0, np.clip(np.sign(angle) * 0.25, -1.0, 1.0)]),
            )
            out = planner._step(action)
            if planner.truncated:
                return out
        return out if abs(planner._yaw_to(direction)) <= ALIGN_TOL else -1

    direction = COUNTER_DIRECTION.copy()

    env.log_event(
        "waypoint",
        "1: align base parallel to countertop",
        target_direction=direction.tolist(),
    )
    result = env.log_motion(
        "Waypoint 1 rotate",
        planner.rotate_base_z,
        direction,
    )
    if result == -1:
        env.log_event("error", "Waypoint 1 failed")
        return False

    planner.idle_steps(t=30)
    actual = _heading(agent)
    aligned = float(np.dot(actual, direction)) >= np.cos(ALIGN_TOL)
    env.log_event(
        "waypoint_complete" if aligned else "error",
        "Waypoint 1 complete" if aligned else "Waypoint 1 heading outside tolerance",
        heading=actual.tolist(),
        error_rad=float(np.arccos(np.clip(np.dot(actual, direction), -1.0, 1.0))),
    )
    print(
        "[WAYPOINT 1]",
        "ok" if aligned else "failed",
        "target=", np.round(direction, 3),
        "actual=", np.round(actual, 3),
    )
    if not aligned:
        return False

    env.log_event(
        "waypoint",
        "2: drive parallel, stopping 25 cm before the cup",
    )
    cup_x = float(unwenv.cup.pose.p[0].cpu().numpy()[0])
    target_x = cup_x + direction[0] * CUP_X_GAP
    for attempt in range(3):
        base = agent.base_link.pose.sp.p.copy()
        error_x = target_x - float(base[0])
        if abs(error_x) <= 0.04:
            break
        heading = _heading(agent)
        distance = float(np.dot(np.array([error_x, 0.0]), heading[:2]))
        result = env.log_motion(
            f"Waypoint 2 drive {attempt + 1}",
            drive_fixed_arm,
            distance,
            v=0.10,
        )
        if result == -1:
            env.log_event("error", "Waypoint 2 drive failed", attempt=attempt + 1)
            return False
        planner.idle_steps(t=10)

    base = agent.base_link.pose.sp.p.copy()
    error_x = target_x - float(base[0])
    cup_gap = abs(float(np.dot(np.array([base[0] - cup_x, 0.0]), -direction[:2])))
    staged = abs(error_x) <= 0.04 and abs(abs(cup_gap) - CUP_X_GAP) <= 0.04
    env.log_event(
        "waypoint_complete" if staged else "error",
        "Waypoint 2 complete" if staged else "Waypoint 2 standoff outside tolerance",
        base_xy=base[:2].tolist(),
        cup_x=cup_x,
        target_base_x=target_x,
        cup_gap=cup_gap,
    )
    print(
        "[WAYPOINT 2]",
        "ok" if staged else "failed",
        "base_x=", round(float(base[0]), 3),
        "cup_x=", round(cup_x, 3),
        "cup_gap=", round(cup_gap, 3),
    )
    if not staged:
        return False

    # Raise torso only enough for the straight hand to clear the counter.
    # With the arm folded, the base turn becomes much larger than 90 degrees.
    body = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    start_torso = float(body[2])
    arm_hold = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    for i in range(60):
        body = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
        body[2] = start_torso + (SAFE_TORSO - start_torso) * ((i + 1) / 60)
        env.step(np.hstack([arm_hold, planner.gripper_state, body, [0.0, 0.0]]))
    planner.planner.update_from_simulation()
    env.log_event("support", f"Raise torso to {SAFE_TORSO:.3f} for straight-arm turn")

    env.log_event("waypoint", "3: turn right toward cup")
    # Counter normal is +y; from the fixed -x parallel heading this is exactly
    # the minimal right-hand quarter turn.
    target = np.array([direction[1], -direction[0], 0.0])
    result = env.log_motion(
        "Waypoint 3 rotate right toward cup",
        planner.rotate_base_z,
        target,
    )
    if result == -1:
        result = env.log_motion(
            "Waypoint 3 direct yaw fallback",
            rotate_direct,
            target,
        )
    if result == -1:
        env.log_event("error", "Waypoint 3 failed")
        return False

    planner.idle_steps(t=30)
    actual = _heading(agent)
    facing = float(np.dot(actual, target)) >= np.cos(ALIGN_TOL)
    env.log_event(
        "waypoint_complete" if facing else "error",
        "Waypoint 3 complete" if facing else "Waypoint 3 heading outside tolerance",
        target_direction=target.tolist(),
        heading=actual.tolist(),
        error_rad=float(np.arccos(np.clip(np.dot(actual, target), -1.0, 1.0))),
    )
    print(
        "[WAYPOINT 3]",
        "ok" if facing else "failed",
        "target=", np.round(target, 3),
        "actual=", np.round(actual, 3),
    )
    if not facing:
        return False

    env.log_event("waypoint", "4: approach counter with a 40 cm base standoff")
    counter_front_y = float(
        unwenv.counter_pos[1] - unwenv.counter_size[1] / 2
    )
    base_radius = float(getattr(unwenv, "ROBOT_RADIUS", 0.35))
    target_y = counter_front_y - base_radius - BASE_STANDOFF
    base_before = agent.base_link.pose.sp.p.copy()
    cup_before = unwenv.cup.pose.p[0].cpu().numpy().copy()
    result = env.log_motion(
        "Waypoint 4 drive to counter standoff",
        drive_fixed_arm,
        target_y - float(base_before[1]),
        v=0.10,
        max_steps=500,
    )
    if result == -1:
        env.log_event("error", "Waypoint 4 drive failed")
        return False
    planner.idle_steps(t=20)

    base_after = agent.base_link.pose.sp.p.copy()
    cup_after = unwenv.cup.pose.p[0].cpu().numpy().copy()
    actual_gap = counter_front_y - (float(base_after[1]) + base_radius)
    cup_shift = float(np.linalg.norm(cup_after[:2] - cup_before[:2]))
    cup_reach = float(np.linalg.norm(cup_after[:2] - base_after[:2]))
    tray_reach = float(
        np.linalg.norm(unwenv.tray.pose.p[0].cpu().numpy()[:2] - base_after[:2])
    )
    positioned = (
        abs(float(base_after[1]) - target_y) <= 0.05
        and cup_shift <= 0.02
        and cup_reach <= 1.10
    )
    env.log_event(
        "waypoint_complete" if positioned else "error",
        "Waypoint 4 complete" if positioned else "Waypoint 4 position/cup check failed",
        target_base_y=target_y,
        base_y=float(base_after[1]),
        counter_gap=actual_gap,
        cup_shift=cup_shift,
        cup_reach=cup_reach,
        tray_reach=tray_reach,
    )
    print(
        "[WAYPOINT 4]",
        "ok" if positioned else "failed",
        "base_y=", round(float(base_after[1]), 3),
        "counter_gap=", round(actual_gap, 3),
        "cup_reach=", round(cup_reach, 3),
        "tray_reach=", round(tray_reach, 3),
        "cup_shift=", round(cup_shift, 3),
    )
    if not positioned:
        return False

    env.log_event("support", "Keep straight arm during turn")
    env.log_event("waypoint", "5: turn parallel toward tray")
    tray_x = float(unwenv.tray.pose.p[0].cpu().numpy()[0])
    cup_x = float(unwenv.cup.pose.p[0].cpu().numpy()[0])
    target = np.array([1.0 if tray_x >= cup_x else -1.0, 0.0, 0.0])
    turn_direction = target.copy()
    result = env.log_motion(
        "Waypoint 5 rotate toward tray",
        planner.rotate_base_z,
        turn_direction,
    )
    if result == -1:
        result = env.log_motion(
            "Waypoint 5 direct yaw fallback",
            rotate_direct,
            turn_direction,
        )
    if result == -1:
        env.log_event("error", "Waypoint 5 failed")
        return False

    planner.idle_steps(t=30)
    actual = _heading(agent)
    aligned = float(np.dot(actual, target)) >= np.cos(ALIGN_TOL)
    tcp_z = float(agent.tcp.pose.p[0].cpu().numpy()[2])
    cup_z = float(unwenv.cup.pose.p[0].cpu().numpy()[2])
    level_error = abs(tcp_z - cup_z)
    env.log_event(
        "waypoint_complete" if aligned else "error",
        "Waypoint 5 complete" if aligned else "Waypoint 5 heading/height outside tolerance",
        target_direction=target.tolist(),
        heading=actual.tolist(),
        error_rad=float(np.arccos(np.clip(np.dot(actual, target), -1.0, 1.0))),
        tcp_z=tcp_z,
        cup_z=cup_z,
        level_error=level_error,
        tray_x=tray_x,
    )
    print(
        "[WAYPOINT 5]",
        "ok" if aligned else "failed",
        "target=", np.round(target, 3),
        "actual=", np.round(actual, 3),
    )
    if not aligned:
        return False

    env.log_event(
        "waypoint",
        "6: reverse to 10 cm behind the cup",
    )
    base_before = agent.base_link.pose.sp.p.copy()
    cup_x = float(unwenv.cup.pose.p[0].cpu().numpy()[0])
    target_x = cup_x - turn_direction[0] * CUP_BACK_GAP
    distance = float(
        np.dot(
            np.array([target_x - float(base_before[0]), 0.0]),
            _heading(agent)[:2],
        )
    )
    result = env.log_motion(
        "Waypoint 6 reverse past cup",
        drive_fixed_arm,
        distance,
        v=0.10,
        max_steps=500,
    )
    if result == -1:
        env.log_event("error", "Waypoint 6 reverse failed")
        return False
    planner.idle_steps(t=20)

    base_after = agent.base_link.pose.sp.p.copy()
    cup_after = unwenv.cup.pose.p[0].cpu().numpy().copy()
    tray_after = unwenv.tray.pose.p[0].cpu().numpy().copy()
    cup_shift = float(np.linalg.norm(cup_after[:2] - cup_before[:2]))
    cup_reach = float(np.linalg.norm(cup_after[:2] - base_after[:2]))
    tray_reach = float(np.linalg.norm(tray_after[:2] - base_after[:2]))
    reversed_past = (
        abs(float(base_after[0]) - target_x) <= 0.05
        and cup_shift <= 0.02
        and cup_reach <= 1.10
    )
    env.log_event(
        "waypoint_complete" if reversed_past else "error",
        "Waypoint 6 complete" if reversed_past else "Waypoint 6 reverse/check failed",
        target_base_x=target_x,
        base_x=float(base_after[0]),
        cup_x=cup_x,
        cup_shift=cup_shift,
        cup_reach=cup_reach,
        tray_reach=tray_reach,
    )
    print(
        "[WAYPOINT 6]",
        "ok" if reversed_past else "failed",
        "base_x=", round(float(base_after[0]), 3),
        "target_x=", round(target_x, 3),
        "cup_reach=", round(cup_reach, 3),
        "tray_reach=", round(tray_reach, 3),
    )
    if not reversed_past:
        return False

    env.log_event("waypoint", "7: turn head toward cup")
    arm_before = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    base_before = agent.base_link.pose.sp.p.copy()
    cup_pos = unwenv.cup.pose.p[0].cpu().numpy().copy()
    pan, tilt = _head_look_at(agent, cup_pos)
    planner.hold_head(pan, tilt, t=40, ramp=20)

    head = next(
        link for link in agent.robot.get_links()
        if link.get_name() == "head_camera_link"
    )
    look_direction = head.pose.sp.to_transformation_matrix()[:3, 0]
    target_direction = cup_pos - np.asarray(head.pose.sp.p, dtype=float)
    look_error = float(
        np.arccos(
            np.clip(
                np.dot(look_direction, target_direction)
                / (np.linalg.norm(look_direction) * np.linalg.norm(target_direction)),
                -1.0,
                1.0,
            )
        )
    )
    arm_after = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    base_after = agent.base_link.pose.sp.p.copy()
    arm_error = float(np.max(np.abs(arm_after - arm_before)))
    base_shift = float(np.linalg.norm(base_after[:2] - base_before[:2]))
    looked = look_error <= HEAD_LOOK_TOL and arm_error <= 0.03 and base_shift <= 0.01
    env.log_event(
        "waypoint_complete" if looked else "error",
        "Waypoint 7 complete" if looked else "Waypoint 7 head/look check failed",
        pan=pan,
        tilt=tilt,
        look_error_rad=look_error,
        arm_error=arm_error,
        base_shift=base_shift,
    )
    print(
        "[WAYPOINT 7]",
        "ok" if looked else "failed",
        "pan=", round(pan, 3),
        "tilt=", round(tilt, 3),
        "look_error_deg=", round(float(np.degrees(look_error)), 2),
        "arm_error=", round(arm_error, 4),
    )
    if not looked:
        return False

    env.log_event("waypoint", "8: RRT open hand to 12 cm pregrasp")
    planner.planner.update_from_simulation()
    arm_before = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    base_before = agent.base_link.pose.sp.p.copy()
    cup_before = unwenv.cup.pose.p[0].cpu().numpy().copy()
    approach = cup_before - base_before
    approach[2] = 0.0
    approach /= np.linalg.norm(approach)
    closing = np.cross(np.array([0.0, 0.0, 1.0]), approach)
    closing /= np.linalg.norm(closing)
    grasp_pose = agent.build_grasp_pose(approach, closing, cup_before)
    pregrasp_pose = grasp_pose * sapien.Pose([0.0, 0.0, -PREGRASP_GAP])

    # Keep gripper open while RRT moves only the arm/body; head is restored after
    # follow_path because the solver's arm path parks head joints at zero.
    planner.gripper_state = GRIPPER_OPEN
    result = env.log_motion(
        "Waypoint 8 RRT pregrasp",
        planner.move_to_pose_with_RRTConnect,
        pregrasp_pose,
        n_init_qpos=100,
        disable_lift_joint=False,
    )
    rrt_ok = result != -1
    if rrt_ok:
        planner.hold_head(pan, tilt, t=20, ramp=10)

    tcp = agent.tcp.pose.p[0].cpu().numpy()
    cup_after = unwenv.cup.pose.p[0].cpu().numpy().copy()
    base_after = agent.base_link.pose.sp.p.copy()
    tcp_error = float(np.linalg.norm(tcp - pregrasp_pose.p))
    tcp_to_cup = cup_after - tcp
    standoff = float(np.dot(tcp_to_cup, approach))
    lateral_error = float(np.linalg.norm(tcp_to_cup - approach * standoff))
    arm_error = float(
        np.max(
            np.abs(
                agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                - arm_before
            )
        )
    )
    base_shift = float(np.linalg.norm(base_after[:2] - base_before[:2]))
    cup_shift = float(np.linalg.norm(cup_after[:2] - cup_before[:2]))
    pregrasped = (
        rrt_ok
        and 0.08 <= standoff <= 0.16
        and lateral_error <= 0.05
        and tcp_error <= 0.06
        and base_shift <= 0.03
        and cup_shift <= 0.02
        and not bool(unwenv.agent.is_grasping(unwenv.cup).item())
    )
    env.log_event(
        "waypoint_complete" if pregrasped else "error",
        "Waypoint 8 complete" if pregrasped else "Waypoint 8 RRT pregrasp check failed",
        rrt_ok=rrt_ok,
        target_gap=PREGRASP_GAP,
        standoff=standoff,
        lateral_error=lateral_error,
        tcp_error=tcp_error,
        arm_error=arm_error,
        base_shift=base_shift,
        cup_shift=cup_shift,
    )
    print(
        "[WAYPOINT 8]",
        "ok" if pregrasped else "failed",
        "standoff=", round(standoff, 3),
        "lateral=", round(lateral_error, 3),
        "tcp_error=", round(tcp_error, 3),
        "arm_error=", round(arm_error, 4),
    )
    if not pregrasped:
        return False

    env.log_event("waypoint", "9: IK/RRT move open hand to grasp pose")
    planner.planner.update_from_simulation()
    arm_before = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    base_before = agent.base_link.pose.sp.p.copy()
    cup_before = unwenv.cup.pose.p[0].cpu().numpy().copy()
    planner.gripper_state = GRIPPER_OPEN

    rrt_check = env.log_motion(
        "Waypoint 9 RRT dry-run",
        planner.move_to_pose_with_RRTConnect,
        grasp_pose,
        dry_run=True,
        n_init_qpos=100,
        disable_lift_joint=False,
    )
    rrt_ok = rrt_check != -1
    motion = env.log_motion(
        "Waypoint 9 IK grasp approach",
        planner.static_manipulation,
        grasp_pose,
        n_init_qpos=100,
        disable_lift_joint=False,
    )
    motion_ok = motion != -1
    # Keep head target active after the arm solver and do not close the gripper yet.
    planner.hold_head(pan, tilt, t=20, ramp=10)
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    cup_after = unwenv.cup.pose.p[0].cpu().numpy().copy()
    base_after = agent.base_link.pose.sp.p.copy()
    body_now = agent.controller.controllers["body"].qpos[0].cpu().numpy()
    tcp_error = float(np.linalg.norm(tcp - grasp_pose.p))
    head_error = float(np.max(np.abs(body_now[:2] - np.array([pan, tilt]))))
    arm_error = float(
        np.max(
            np.abs(
                agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                - arm_before
            )
        )
    )
    base_shift = float(np.linalg.norm(base_after[:2] - base_before[:2]))
    cup_shift = float(np.linalg.norm(cup_after[:2] - cup_before[:2]))
    at_grasp = (
        motion_ok
        and tcp_error <= 0.06
        and head_error <= 0.03
        and base_shift <= 0.03
        and cup_shift <= 0.02
        and not bool(unwenv.agent.is_grasping(unwenv.cup).item())
    )
    env.log_event(
        "waypoint_complete" if at_grasp else "error",
        (
            "Waypoint 9 complete via IK fallback"
            if at_grasp and not rrt_ok
            else "Waypoint 9 complete"
            if at_grasp
            else "Waypoint 9 grasp-pose check failed"
        ),
        rrt_ok=rrt_ok,
        motion_ok=motion_ok,
        tcp_error=tcp_error,
        head_error=head_error,
        arm_error=arm_error,
        base_shift=base_shift,
        cup_shift=cup_shift,
        gripper_open=not bool(unwenv.agent.is_grasping(unwenv.cup).item()),
    )
    print(
        "[WAYPOINT 9]",
        "ok" if at_grasp else "failed",
        "rrt=", rrt_ok,
        "motion=", motion_ok,
        "mode=", "IK fallback" if not rrt_ok else "RRT/IK",
        "tcp_error=", round(tcp_error, 3),
        "head_error=", round(head_error, 4),
    )
    if not at_grasp:
        return False

    env.log_event("waypoint", "10: close gripper on cup")
    planner.gripper_state = GRIPPER_CLOSED
    arm_hold = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    body_hold = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_hold[:2] = np.array([pan, tilt])
    base_before = agent.base_link.pose.sp.p.copy()
    cup_before = unwenv.cup.pose.p[0].cpu().numpy().copy()
    close_arm_drift = 0.0
    for _ in range(20):
        planner._compose(arm_hold, body_hold, np.array([0.0, 0.0]))
        planner._step(planner._from_abs(planner._last_abs))
        close_arm_drift = max(
            close_arm_drift,
            float(
                np.max(
                    np.abs(
                        agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                        - arm_hold
                    )
                )
            ),
        )
        if planner.truncated:
            break
    planner.planner.update_from_simulation()
    grasped = bool(unwenv.agent.is_grasping(unwenv.cup).item())
    env.log_event(
        "waypoint_complete" if grasped else "error",
        "Waypoint 10 complete" if grasped else "Waypoint 10 grasp failed",
        grasped=grasped,
        arm_drift=close_arm_drift,
        cup_shift=float(
            np.linalg.norm(unwenv.cup.pose.p[0].cpu().numpy()[:2] - cup_before[:2])
        ),
    )
    print(
        "[WAYPOINT 10]",
        "ok" if grasped else "failed",
        "arm_drift=", round(close_arm_drift, 4),
    )
    if not grasped:
        return False

    env.log_event("waypoint", "11: lift torso with cup")
    cup_z_before = float(unwenv.cup.pose.p[0].cpu().numpy()[2])
    torso_before = float(body_hold[2])
    torso_target = min(torso_before + TORSO_LIFT_DELTA, 0.386)
    lift_arm_drift = 0.0
    lift_steps = 80
    for i in range(lift_steps):
        body = body_hold.copy()
        body[2] = torso_before + (torso_target - torso_before) * ((i + 1) / lift_steps)
        planner._compose(arm_hold, body, np.array([0.0, 0.0]))
        planner._step(planner._from_abs(planner._last_abs))
        lift_arm_drift = max(
            lift_arm_drift,
            float(
                np.max(
                    np.abs(
                        agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                        - arm_hold
                    )
                )
            ),
        )
        if planner.truncated:
            break
    planner.planner.update_from_simulation()
    cup_after = unwenv.cup.pose.p[0].cpu().numpy().copy()
    base_after = agent.base_link.pose.sp.p.copy()
    torso_after = float(agent.controller.controllers["body"].qpos[0].cpu().numpy()[2])
    cup_rise = float(cup_after[2] - cup_z_before)
    cup_xy_shift = float(np.linalg.norm(cup_after[:2] - cup_before[:2]))
    base_shift = float(np.linalg.norm(base_after[:2] - base_before[:2]))
    head_error = float(
        np.max(
            np.abs(
                agent.controller.controllers["body"].qpos[0].cpu().numpy()[:2]
                - np.array([pan, tilt])
            )
        )
    )
    lifted = (
        cup_rise >= 0.04
        and bool(unwenv.agent.is_grasping(unwenv.cup).item())
        and lift_arm_drift <= 0.03
        and cup_xy_shift <= 0.03
        and base_shift <= 0.03
        and head_error <= 0.03
    )
    env.log_event(
        "waypoint_complete" if lifted else "error",
        "Waypoint 11 complete" if lifted else "Waypoint 11 lift check failed",
        torso_before=torso_before,
        torso_target=torso_target,
        torso_after=torso_after,
        cup_z_before=cup_z_before,
        cup_z_after=float(cup_after[2]),
        cup_rise=cup_rise,
        arm_drift=lift_arm_drift,
        cup_xy_shift=cup_xy_shift,
        base_shift=base_shift,
        head_error=head_error,
    )
    print(
        "[WAYPOINT 11]",
        "ok" if lifted else "failed",
        "cup_rise=", round(cup_rise, 3),
        "arm_drift=", round(lift_arm_drift, 4),
        "torso=", round(torso_after, 3),
    )
    if not lifted:
        return False

    env.log_event("waypoint", "12: drive cup to tray x alignment")
    arm_hold = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    base_before = agent.base_link.pose.sp.p.copy()
    cup_before = unwenv.cup.pose.p[0].cpu().numpy().copy()
    tray_before = unwenv.tray.pose.p[0].cpu().numpy().copy()
    distance = float(
        np.dot(tray_before[:2] - cup_before[:2], _heading(agent)[:2])
    )
    result = env.log_motion(
        "Waypoint 12 drive toward tray",
        drive_fixed_arm,
        distance,
        v=0.10,
        max_steps=500,
    )
    if result == -1:
        env.log_event("error", "Waypoint 12 drive failed")
        return False
    planner.idle_steps(t=20)
    planner.planner.update_from_simulation()

    cup_after = unwenv.cup.pose.p[0].cpu().numpy().copy()
    tray_after = unwenv.tray.pose.p[0].cpu().numpy().copy()
    base_after = agent.base_link.pose.sp.p.copy()
    arm_after = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    heading_after = _heading(agent)
    cup_tray_x_error = abs(float(cup_after[0] - tray_after[0]))
    cup_tray_y_error = abs(float(cup_after[1] - tray_after[1]))
    tray_shift = float(np.linalg.norm(tray_after[:2] - tray_before[:2]))
    base_shift = float(np.linalg.norm(base_after[:2] - base_before[:2]))
    arm_drift = float(np.max(np.abs(arm_after - arm_hold)))
    aligned = (
        cup_tray_x_error <= 0.05
        and cup_tray_y_error <= 0.35
        and tray_shift <= 0.02
        and arm_drift <= 0.05
        and bool(unwenv.agent.is_grasping(unwenv.cup).item())
        and float(np.dot(heading_after[:2], turn_direction[:2])) >= np.cos(ALIGN_TOL)
    )
    env.log_event(
        "waypoint_complete" if aligned else "error",
        "Waypoint 12 complete" if aligned else "Waypoint 12 tray alignment failed",
        cup_before=cup_before[:2].tolist(),
        cup_after=cup_after[:2].tolist(),
        tray_xy=tray_after[:2].tolist(),
        cup_tray_x_error=cup_tray_x_error,
        cup_tray_y_error=cup_tray_y_error,
        tray_shift=tray_shift,
        base_shift=base_shift,
        arm_drift=arm_drift,
        grasped=bool(unwenv.agent.is_grasping(unwenv.cup).item()),
        heading=heading_after.tolist(),
    )
    print(
        "[WAYPOINT 12]",
        "ok" if aligned else "failed",
        "cup_tray_x_error=", round(cup_tray_x_error, 3),
        "cup_tray_y_error=", round(cup_tray_y_error, 3),
        "arm_drift=", round(arm_drift, 4),
    )
    if not aligned:
        return False

    env.log_event("waypoint", "13: RRT move cup over tray center")
    from planners.oracle.oracle_common import hold_object_in_planner

    hold_object_in_planner(
        env,
        planner,
        unwenv,
        unwenv.cup,
        held=True,
        who="takeitback_tray",
    )
    tcp_before = agent.tcp.pose.sp
    cup_before = unwenv.cup.pose.sp
    tray_before = unwenv.tray.pose.sp
    tcp_to_cup = tcp_before.inv() * cup_before
    target_cup = sapien.Pose(
        p=[float(tray_before.p[0]), float(tray_before.p[1]), float(cup_before.p[2])],
        q=cup_before.q,
    )
    target_tcp = target_cup * tcp_to_cup.inv()
    base_before = agent.base_link.pose.sp.p.copy()
    torso_before = float(agent.controller.controllers["body"].qpos[0].cpu().numpy()[2])
    planner.gripper_state = GRIPPER_CLOSED
    result = env.log_motion(
        "Waypoint 13 RRT cup over tray",
        planner.move_to_pose_with_RRTConnect,
        target_tcp,
        n_init_qpos=100,
        disable_lift_joint=True,
    )
    rrt_ok = result != -1
    planner.hold_head(pan, tilt, t=20, ramp=10)
    cup_after = unwenv.cup.pose.p[0].cpu().numpy().copy()
    tray_after = unwenv.tray.pose.p[0].cpu().numpy().copy()
    base_after = agent.base_link.pose.sp.p.copy()
    torso_after = float(agent.controller.controllers["body"].qpos[0].cpu().numpy()[2])
    head_error = float(
        np.max(
            np.abs(
                agent.controller.controllers["body"].qpos[0].cpu().numpy()[:2]
                - np.array([pan, tilt])
            )
        )
    )
    cup_tray_xy_error = float(np.linalg.norm(cup_after[:2] - tray_after[:2]))
    cup_z_error = abs(float(cup_after[2] - cup_before.p[2]))
    tray_shift = float(np.linalg.norm(tray_after[:2] - tray_before.p[:2]))
    base_shift = float(np.linalg.norm(base_after[:2] - base_before[:2]))
    aligned_over_tray = (
        rrt_ok
        and cup_tray_xy_error <= 0.06
        and cup_z_error <= 0.03
        and tray_shift <= 0.02
        and base_shift <= 0.03
        and abs(torso_after - torso_before) <= 0.03
        and head_error <= 0.03
        and bool(unwenv.agent.is_grasping(unwenv.cup).item())
    )
    env.log_event(
        "waypoint_complete" if aligned_over_tray else "error",
        "Waypoint 13 complete" if aligned_over_tray else "Waypoint 13 RRT tray placement failed",
        rrt_ok=rrt_ok,
        cup_tray_xy_error=cup_tray_xy_error,
        cup_z_error=cup_z_error,
        tray_shift=tray_shift,
        base_shift=base_shift,
        torso_shift=abs(torso_after - torso_before),
        head_error=head_error,
        grasped=bool(unwenv.agent.is_grasping(unwenv.cup).item()),
    )
    print(
        "[WAYPOINT 13]",
        "ok" if aligned_over_tray else "failed",
        "rrt=", rrt_ok,
        "cup_tray_error=", round(cup_tray_xy_error, 3),
        "cup_z_error=", round(cup_z_error, 3),
    )
    if not aligned_over_tray:
        return False

    env.log_event("waypoint", "14: lower cup to 1 cm above tray")
    arm_hold = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    body_hold = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_hold[:2] = np.array([pan, tilt])
    torso_before_lower = float(body_hold[2])
    cup_before_lower = unwenv.cup.pose.p[0].cpu().numpy().copy()
    tray_pose = unwenv.tray.pose.p[0].cpu().numpy().copy()
    tray_top = float(tray_pose[2] + unwenv.tray_half[2])
    target_cup_z = float(tray_top + unwenv.cup_half[2] + TRAY_DROP_GAP)
    torso_target = max(0.0, torso_before_lower - (cup_before_lower[2] - target_cup_z))
    lower_arm_drift = 0.0
    lower_steps = 100
    for i in range(lower_steps):
        body = body_hold.copy()
        body[2] = torso_before_lower + (torso_target - torso_before_lower) * ((i + 1) / lower_steps)
        planner._compose(arm_hold, body, np.array([0.0, 0.0]))
        planner._step(planner._from_abs(planner._last_abs))
        lower_arm_drift = max(
            lower_arm_drift,
            float(
                np.max(
                    np.abs(
                        agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                        - arm_hold
                    )
                )
            ),
        )
        if planner.truncated:
            break
    planner.planner.update_from_simulation()
    cup_after_lower = unwenv.cup.pose.p[0].cpu().numpy().copy()
    gap_after_lower = float(cup_after_lower[2] - unwenv.cup_half[2] - tray_top)
    lowered = (
        abs(gap_after_lower - TRAY_DROP_GAP) <= TRAY_DROP_TOL
        and bool(unwenv.agent.is_grasping(unwenv.cup).item())
        and lower_arm_drift <= 0.05
    )
    env.log_event(
        "waypoint_complete" if lowered else "error",
        "Waypoint 14 complete" if lowered else "Waypoint 14 lowering failed",
        tray_top=tray_top,
        target_cup_z=target_cup_z,
        cup_z=float(cup_after_lower[2]),
        gap=gap_after_lower,
        torso_before=torso_before_lower,
        torso_target=torso_target,
        arm_drift=lower_arm_drift,
    )
    print(
        "[WAYPOINT 14]",
        "ok" if lowered else "failed",
        "gap=", round(gap_after_lower, 3),
        "arm_drift=", round(lower_arm_drift, 4),
    )
    if not lowered:
        return False

    env.log_event("waypoint", "15: release cup")
    planner.gripper_state = GRIPPER_OPEN
    release_body = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    release_body[:2] = np.array([pan, tilt])
    release_arm_drift = 0.0
    for _ in range(30):
        planner._compose(arm_hold, release_body, np.array([0.0, 0.0]))
        planner._step(planner._from_abs(planner._last_abs))
        release_arm_drift = max(
            release_arm_drift,
            float(
                np.max(
                    np.abs(
                        agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                        - arm_hold
                    )
                )
            ),
        )
        if planner.truncated:
            break
    planner.planner.update_from_simulation()
    hold_object_in_planner(
        env,
        planner,
        unwenv,
        unwenv.cup,
        held=False,
        who="takeitback_tray",
    )
    cup_after_release = unwenv.cup.pose.p[0].cpu().numpy().copy()
    tray_after_release = unwenv.tray.pose.p[0].cpu().numpy().copy()
    released = (
        not bool(unwenv.agent.is_grasping(unwenv.cup).item())
        and float(np.linalg.norm(cup_after_release[:2] - tray_after_release[:2])) <= 0.12
        and release_arm_drift <= 0.05
    )
    env.log_event(
        "waypoint_complete" if released else "error",
        "Waypoint 15 complete" if released else "Waypoint 15 release failed",
        grasped=bool(unwenv.agent.is_grasping(unwenv.cup).item()),
        cup_xy_error=float(np.linalg.norm(cup_after_release[:2] - tray_after_release[:2])),
        arm_drift=release_arm_drift,
    )
    print(
        "[WAYPOINT 15]",
        "ok" if released else "failed",
        "cup_xy_error=", round(float(np.linalg.norm(cup_after_release[:2] - tray_after_release[:2])), 3),
    )
    if not released:
        return False

    env.log_event("waypoint", "16: raise torso back")
    restore_body = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    restore_body[:2] = np.array([pan, tilt])
    torso_before_restore = float(restore_body[2])
    restore_arm_drift = 0.0
    restore_steps = 80
    for i in range(restore_steps):
        body = restore_body.copy()
        body[2] = torso_before_restore + (torso_before_lower - torso_before_restore) * ((i + 1) / restore_steps)
        planner._compose(arm_hold, body, np.array([0.0, 0.0]))
        planner._step(planner._from_abs(planner._last_abs))
        restore_arm_drift = max(
            restore_arm_drift,
            float(
                np.max(
                    np.abs(
                        agent.controller.controllers["arm"].qpos[0].cpu().numpy()
                        - arm_hold
                    )
                )
            ),
        )
        if planner.truncated:
            break
    planner.planner.update_from_simulation()
    final_body = agent.controller.controllers["body"].qpos[0].cpu().numpy()
    final_cup = unwenv.cup.pose.p[0].cpu().numpy().copy()
    final_tray = unwenv.tray.pose.p[0].cpu().numpy().copy()
    final_head_error = float(np.max(np.abs(final_body[:2] - np.array([pan, tilt]))))
    final_torso_error = abs(float(final_body[2]) - torso_before_lower)
    final_xy_error = float(np.linalg.norm(final_cup[:2] - final_tray[:2]))
    success = (
        final_torso_error <= 0.03
        and final_head_error <= 0.03
        and restore_arm_drift <= 0.05
        and final_xy_error <= 0.12
        and not bool(unwenv.agent.is_grasping(unwenv.cup).item())
    )
    env.log_event(
        "waypoint_complete" if success else "error",
        "Waypoint 16 complete" if success else "Waypoint 16 torso restore failed",
        torso_error=final_torso_error,
        head_error=final_head_error,
        arm_drift=restore_arm_drift,
        cup_xy_error=final_xy_error,
        grasped=bool(unwenv.agent.is_grasping(unwenv.cup).item()),
        task_success=bool(unwenv.evaluate()["success"].item()),
    )
    print(
        "[WAYPOINT 16]",
        "ok" if success else "failed",
        "torso_error=", round(final_torso_error, 3),
        "cup_xy_error=", round(final_xy_error, 3),
    )
    return success


if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    from mplib.pymp import set_global_seed

    set_global_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        source_desc="TakeItBack tray waypoints 1-16 review run",
    )
    env = PlannerLogger(
        env,
        log_dir=str(run_dir),
        name=f"takeitback_tray_wp16_seed{seed}",
        log_freq=args.log_freq,
        run_dir=run_dir,
    )

    env.action_space.seed(seed)
    with capture_stdout(env.dir / "console.log"):
        ok = planning(env, seed, debug=args.debug, info=args.info)
    env.close()
    _repair_trajectory_metadata(run_dir)

    print(f"[INFO] waypoints_1_16={'ok' if ok else 'failed'}")
    print(f"[INFO] run_dir={run_dir}")
    if not args.no_video:
        for video in sorted(run_dir.glob("video_*.mp4")):
            print(f"[INFO] video={video}")
