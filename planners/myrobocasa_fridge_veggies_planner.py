"""Canonical staged vegetable planner: sequential waypoints 2–17, fail-fast."""

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien
import torch

import my_scenes  # noqa: F401 — register the benchmark environment
from robots.fetch.actions import base_cmd as _base_cmd
from robots.fetch.extand import FetchMotionPlanningSapienSolver
from utils.logging_utils import PlannerLogger, StreamingVideoRecorder, capture_stdout
from utils.planners_utils import lower_torso_smooth


CAMERA_NAMES = ("counter_side", "overhead_rear", "fridge_robot_start")
ROBOT_FRONT_DISTANCE_M = 1.15
WP4_TORSO_DROP_M = 0.22
WP13_TORSO_RAISE_M = 0.04
WP14_HOVER_CLEARANCE_M = 0.05


def _fit_rim_circle(xy):
    """RANSAC circle fit of the pot's rim wall vertices (xy). Returns
    (center_xy, R) or None. Deterministic (fixed RNG) so the measured bowl
    center is stable across runs with the same geometry."""
    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        idx = rng.choice(len(xy), 3, replace=False)
        a, b, c = xy[idx]
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-9:
            continue
        ux = (
            (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
            + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
            + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
        ) / d
        uy = (
            (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
            + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
            + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])
        ) / d
        R = np.hypot(ux - a[0], uy - a[1])
        if not (0.03 < R < 0.35):
            continue
        n = int(np.sum(np.abs(np.hypot(xy[:, 0] - ux, xy[:, 1] - uy) - R) < 0.015))
        if best is None or n > best[0]:
            best = (n, np.array([ux, uy]), R)
    return best



def _attach_object(planner, obj):
    """Register the grasped object as attached to the gripper in the mplib
    planning world so lift/transport plans stop treating it as an obstacle."""
    from robots.fetch.utils import attach_object

    gripper_link = next(
        l for l in planner.robot._objs[0].get_links() if l.name.endswith("gripper_link")
    )
    attach_object(
        planner.planner.planning_world,
        obj._objs[0],
        planner.robot._objs[0],
        gripper_link,
    )



def _detach_object(planner):
    """Release the object from the planning world (after the gripper is opened)."""
    planner.planner.detach_object()



def _yaw_base_to(env, planner, target_pos, max_rot=500, align_deg=3.0):
    """Rotate the base in place (velocity yaw) until its x-axis points at
    target_pos. Only the yaw joint is commanded; the arm is expected to be in
    a folded/safe pose. Rotation drifts the base slightly, which is fine for
    stance positioning."""
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[:2] = 0.0
    target = np.asarray(target_pos, dtype=float).copy()
    target[2] = 0.0
    for _ in range(max_rot):
        sp = agent.base_link.pose.sp
        p = sp.p.copy()
        p[2] = 0.0
        delta = target - p
        dist = np.linalg.norm(delta)
        if dist < 1e-4:
            break
        xa = sp.to_transformation_matrix()[:3, 0]
        xa[2] = 0.0
        nx = np.linalg.norm(xa)
        if nx < 1e-6:
            break
        xa /= nx
        he = float(np.arctan2(np.cross(xa, delta / dist)[2], np.dot(xa, delta / dist)))
        if abs(he) < np.deg2rad(align_deg):
            break
        base_action = _base_cmd(yaw=float(np.clip(1.5 * he, -0.6, 0.6)))
        env.step(np.hstack([arm_action, 1, body_action, base_action]))
    planner.planner.update_from_simulation()



def _render_camera_configs():
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils

    placeholder = sapien_utils.look_at(
        np.array([0.0, -1.5, 1.5]), np.array([0.0, 0.0, 0.5])
    )
    return [
        CameraConfig(name, placeholder, 1280, 720, 70 * np.pi / 180, 0.01, 100)
        for name in CAMERA_NAMES
    ]


def _install_render_cameras(env):
    type(env.unwrapped)._default_human_render_camera_configs = property(
        lambda self: _render_camera_configs()
    )


def _pose_render_cameras(env):
    from mani_skill.utils import sapien_utils

    scene = env.unwrapped
    cameras = scene._human_render_cameras
    veggie = scene.veggies[scene._picture_target].pose.p[0].cpu().numpy()
    base = scene.agent.base_link.pose.p[0].cpu().numpy()
    counter = np.asarray(scene.counter_pos, dtype=float)

    table_z = counter[2] + scene.counter_size[2] / 2
    side_position = veggie + np.array([0.65, -0.85, 0.0])
    side_position[2] = table_z + 0.08
    side_target = veggie.copy()
    side_target[2] = table_z + 0.04
    side = sapien_utils.look_at(side_position, side_target)
    overhead_target = 0.55 * counter + 0.45 * base
    overhead_target[2] = 0.65
    overhead = sapien_utils.look_at(
        overhead_target + np.array([0.4, -2.8, 3.0]), overhead_target
    )

    picture = np.asarray(scene.picture_center, dtype=float)
    front = np.asarray(scene.robot_spawn_pos, dtype=float)[:2] - picture[:2]
    front /= np.linalg.norm(front)
    front = np.array([front[0], front[1], 0.0])
    width = np.cross(np.array([0.0, 0.0, 1.0]), front)
    fridge_target = 0.5 * (picture + base)
    fridge_target[2] = 0.9
    fridge = sapien_utils.look_at(
        fridge_target + front * 2.5 + width * 0.45 + np.array([0.0, 0.0, 0.9]),
        fridge_target,
    )

    for name, pose in zip(CAMERA_NAMES, (side, overhead, fridge)):
        cameras[name].camera.set_local_pose(pose.sp)


def _ready_qpos(agent):
    robot = agent.robot
    current = robot.get_qpos().cpu().numpy()[0].astype(np.float64)
    rest = np.asarray(agent.keyframes["rest"].qpos, dtype=np.float64)
    goal = current.copy()
    names = list(agent.controller.controllers["arm"].config.joint_names)
    names.append("torso_lift_joint")
    for name in names:
        index = robot.active_joints_map[name].active_index[0].item()
        goal[index] = rest[index]
    return current, goal


def _heading_error_degrees(agent, target):
    base = agent.base_link.pose.sp
    direction = np.asarray(target, dtype=float)[:2] - base.p[:2]
    direction = np.array([direction[0], direction[1], 0.0])
    distance = np.linalg.norm(direction)
    if distance < 1e-8:
        return 0.0
    direction /= distance
    heading = base.to_transformation_matrix()[:3, 0].copy()
    heading[2] = 0.0
    heading /= np.linalg.norm(heading)
    error = np.arctan2(np.cross(heading, direction)[2], np.dot(heading, direction))
    return float(np.degrees(error))


def _turn_toward_point(env, planner, target):
    _yaw_base_to(env, planner, target, align_deg=0.05)
    planner.idle_steps(t=10)
    return abs(_heading_error_degrees(env.unwrapped.agent, target))


def _drive_straight_to_point(
    env, planner, target_position, max_speed=0.18, acceleration=0.35,
    reverse=False, stop_when=None, tolerance=0.01,
):
    agent = env.unwrapped.agent
    arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    grip = planner.gripper_state
    body = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    target = np.asarray(target_position, dtype=float)[:2]
    dt = float(env.unwrapped.control_timestep)
    speed = 0.0
    stalled_steps = 0
    max_steps = 1500

    start = agent.base_link.pose.p[0].cpu().numpy()[:2].copy()
    drive_direction = target - start
    length = float(np.linalg.norm(drive_direction))
    if length > 0.0:
        drive_direction /= length

    def remaining_distance():
        base = agent.base_link.pose.p[0].cpu().numpy()
        return float(np.dot(target - base[:2], drive_direction))

    def target_error():
        base = agent.base_link.pose.p[0].cpu().numpy()
        return float(np.linalg.norm(base[:2] - target))

    def heading():
        value = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0].copy()
        value[2] = 0.0
        return value / np.linalg.norm(value)

    initial_heading = heading()
    initial_error = target_error()
    initial_remaining = remaining_distance()
    previous_remaining = initial_remaining
    steps = 0
    stalled = False
    interrupted = False
    for steps in range(1, max_steps + 1):
        if stop_when is not None and stop_when():
            interrupted = True
            break
        remaining = remaining_distance()
        if remaining <= tolerance:
            break
        desired_speed = min(max_speed, 0.9 * remaining)
        speed += float(
            np.clip(desired_speed - speed, -acceleration * dt, acceleration * dt)
        )
        # Slew-limit speed; hold yaw at zero for the entire straight segment.
        forward = -speed if reverse else speed
        env.step(np.hstack([arm, grip, body, _base_cmd(forward=forward, yaw=0.0)]))
        remaining = remaining_distance()
        stalled_steps = (
            stalled_steps + 1 if remaining >= previous_remaining - 1e-5 else 0
        )
        previous_remaining = remaining
        if steps % 100 == 0:
            env.log_event(
                "drive_progress",
                "Straight approach to point",
                remaining_m=round(remaining, 3),
                target_error_m=round(target_error(), 3),
                forward_speed=round(forward, 3),
                yaw_command=0.0,
            )
        if stalled_steps >= 50:
            stalled = True
            break

    while speed > 1e-4:
        speed = max(0.0, speed - acceleration * dt)
        forward = -speed if reverse else speed
        env.step(np.hstack([arm, grip, body, _base_cmd(forward=forward, yaw=0.0)]))
    planner.planner.update_from_simulation()
    final_error = target_error()
    final_remaining = remaining_distance()
    final_position = agent.base_link.pose.p[0].cpu().numpy()[:2]
    displacement = final_position - start
    cross_track = abs(
        drive_direction[0] * displacement[1]
        - drive_direction[1] * displacement[0]
    )
    final_heading = heading()
    heading_drift = float(
        np.degrees(
            np.arctan2(
                np.cross(initial_heading, final_heading)[2],
                np.dot(initial_heading, final_heading),
            )
        )
    )
    return {
        "initial_target_error_m": initial_error,
        "final_target_error_m": final_error,
        "final_remaining_m": final_remaining,
        "cross_track_m": cross_track,
        "steps": steps,
        "heading_drift_deg": heading_drift,
        "stalled": stalled,
        "interrupted": interrupted,
    }


def _narrow_closing_axis(mesh, reference):
    """Choose the minimum-width horizontal grasp from real collision vertices.

    A 3D OBB's short axes can roll out of the horizontal plane (e.g. corn),
    leaving only its long axis after a tilt filter. Project the mesh into XY
    instead. The parallel jaws are symmetric, so search modulo 180 degrees.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3
            or len(vertices) < 3 or not np.isfinite(vertices).all()):
        raise ValueError("vegetable collision mesh needs finite 3D vertices")
    xy = vertices[:, :2] - vertices[:, :2].mean(axis=0)
    if np.linalg.matrix_rank(xy, tol=1e-10) < 2:
        raise ValueError("vegetable has degenerate horizontal collision geometry")
    angles = np.arange(1800, dtype=np.float64) * (np.pi / 1800)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    widths = np.array([np.ptp(xy @ direction) for direction in directions])
    closing = np.r_[directions[int(np.argmin(widths))], 0.0]
    return -closing if np.dot(closing, reference) < 0.0 else closing


def _grasp_and_lift(env, planner, target_veg):
    agent = env.unwrapped.agent
    body = agent.controller.controllers["body"]
    torso_index = int(body.active_joint_indices[2])
    torso_start = float(body.qpos[0, 2])
    torso_min = float(agent.robot.get_qlimits()[0, torso_index, 0])
    base_start = agent.base_link.pose.p[0].cpu().numpy().copy()
    tcp_start = agent.tcp.pose.p[0].cpu().numpy().copy()
    counter = env.unwrapped.counter_pos
    table_z = float(counter[2] + env.unwrapped.counter_size[2] / 2)

    def finger_bottom():
        return min(
            float(obj.get_global_aabb_fast()[0][2])
            for link in (agent.finger1_link, agent.finger2_link)
            for obj in link._objs
        )

    # Put the real fingertips just above the table, not the TCP below it.
    drop = finger_bottom() - (table_z + 0.005)
    if drop <= 0.0 or torso_start - drop < torso_min:
        env.log_event(
            "result", "Waypoints 2-9", success=False, failed_waypoint=7,
            reason="safe grasp height is outside torso travel",
            requested_drop_m=round(drop, 4),
        )
        return False
    env.log_event(
        "waypoint_start", "Waypoint 7: lower torso to grasp height",
        target_index=env.unwrapped._picture_target,
        target_name=target_veg.name,
        requested_drop_m=round(drop, 4),
        target_tcp_height_m=round(float(tcp_start[2]) - drop, 4),
        fingertip_table_clearance_m=0.005,
    )
    env.log_motion(
        "Waypoint 7: smooth torso descent", lower_torso_smooth,
        env, planner, target_drop=drop, total_steps=60,
    )
    planner.idle_steps(t=10)
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    vegetable = target_veg.pose.p[0].cpu().numpy()
    actual_drop = torso_start - float(body.qpos[0, 2])
    height_error = abs(float(tcp[2]) - (float(tcp_start[2]) - drop))
    xy_error = float(np.linalg.norm(tcp[:2] - vegetable[:2]))
    floor_clearance = finger_bottom() - table_z
    lowered = (
        actual_drop > 0.0 and height_error <= 0.01 and xy_error <= 0.01
        and floor_clearance >= -0.001
        and np.linalg.norm(agent.base_link.pose.p[0].cpu().numpy()[:2] - base_start[:2]) <= 0.01
    )
    env.log_event(
        "waypoint_result", "Waypoint 7: torso lowered", success=bool(lowered),
        actual_drop_m=round(actual_drop, 4), tcp_xy_error_m=round(xy_error, 4),
        fingertip_table_clearance_m=round(floor_clearance, 4),
    )
    if not lowered:
        env.log_event("result", "Waypoints 2-9", success=False, failed_waypoint=7)
        return False

    env.log_event("waypoint_start", "Waypoint 8: grasp target vegetable")
    env.log_motion("Waypoint 8: close gripper", planner.close_gripper, t=20)
    planner.idle_steps(t=10)
    grasped = bool(agent.is_grasping(target_veg))
    env.log_event("waypoint_result", "Waypoint 8: gripper contact", success=grasped)
    if not grasped:
        env.log_event(
            "result", "Waypoints 2-10", success=False, failed_waypoint=8,
            reason="target vegetable not gripped by both fingers",
            target_index=env.unwrapped._picture_target,
            target_name=target_veg.name,
        )
        return False

    veg_before_lift = target_veg.pose.p[0].cpu().numpy().copy()
    relative_before = veg_before_lift - agent.tcp.pose.p[0].cpu().numpy()
    env.log_event(
        "waypoint_start", "Waypoint 9: raise torso with vegetable",
        lift_amount_m=round(actual_drop, 4),
    )
    # Negative drop uses the same smooth torso interpolation to raise it.
    env.log_motion(
        "Waypoint 9: smooth torso lift", lower_torso_smooth,
        env, planner, target_drop=-actual_drop, total_steps=60,
    )
    planner.idle_steps(t=20)
    vegetable = target_veg.pose.p[0].cpu().numpy()
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    lifted = float(vegetable[2] - veg_before_lift[2])
    relative_error = float(np.linalg.norm(vegetable - tcp - relative_before))
    base_drift = float(np.linalg.norm(
        agent.base_link.pose.p[0].cpu().numpy()[:2] - base_start[:2]
    ))
    grasped = bool(agent.is_grasping(target_veg))
    torso_restore_error = abs(float(body.qpos[0, 2]) - torso_start)
    failure_reasons = [
        reason for passed, reason in (
            (grasped, "target_contact_lost"),
            (lifted >= 0.03, "vegetable_not_lifted"),
            (relative_error <= 0.02, "vegetable_slipped"),
            (base_drift <= 0.01, "base_moved_during_lift"),
            (torso_restore_error <= 0.01, "torso_not_restored"),
        ) if not passed
    ]
    success = not failure_reasons
    env.log_event(
        "grasp_validation" if success else "result",
        "WP7-9 mandatory grasp gate", success=success,
        failed_waypoint=None if success else 9,
        failure_reasons=failure_reasons,
        target_index=env.unwrapped._picture_target,
        target_name=target_veg.name,
        grasped=grasped, vegetable_lift_m=round(lifted, 4),
        grasp_relative_error_m=round(relative_error, 4),
        torso_restore_error_m=round(torso_restore_error, 4),
        base_drift_m=round(base_drift, 4),
    )
    print(
        f"Waypoints 7-9: success={success}, grasped={grasped}, "
        f"vegetable_lift={lifted:.4f} m, relative_error={relative_error:.4f} m"
    )
    return success


def _retreat_with_vegetable(env, planner, target_veg):
    agent = env.unwrapped.agent
    if not bool(agent.is_grasping(target_veg)):
        env.log_event(
            "result", "Waypoints 2-10", success=False, failed_waypoint=10,
            reason="target not held before retreat",
        )
        return False
    start = agent.base_link.pose.p[0].cpu().numpy().copy()
    heading = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0].copy()
    heading[2] = 0.0
    heading /= np.linalg.norm(heading)
    target = start - heading * 0.20
    relative = (
        target_veg.pose.p[0].cpu().numpy() - agent.tcp.pose.p[0].cpu().numpy()
    ).copy()

    def lost_grasp():
        current = target_veg.pose.p[0].cpu().numpy() - agent.tcp.pose.p[0].cpu().numpy()
        return not bool(agent.is_grasping(target_veg)) or np.linalg.norm(current - relative) > 0.02

    env.log_event(
        "waypoint_start", "Waypoint 10: retreat 0.20 m without turning",
        target_position_m=np.round(target, 4).tolist(),
        distance_m=0.20, yaw_command=0.0,
    )
    drive = env.log_motion(
        "Waypoint 10: retreat while holding vegetable",
        _drive_straight_to_point, env, planner, target,
        max_speed=0.10, acceleration=0.20, reverse=True,
        stop_when=lost_grasp, tolerance=0.003,
    )
    planner.idle_steps(t=20)
    base = agent.base_link.pose.p[0].cpu().numpy()
    target_error = float(np.linalg.norm(base[:2] - target[:2]))
    retreated = float(np.dot(start[:2] - base[:2], heading[:2]))
    held = not lost_grasp()
    success = bool(
        held and not drive["interrupted"] and not drive["stalled"]
        and target_error <= 0.015 and drive["cross_track_m"] <= 0.01
        and abs(drive["heading_drift_deg"]) <= 0.3
    )
    env.log_event(
        "waypoint_result" if success else "result",
        "Waypoint 10: retreat validated", success=success,
        failed_waypoint=None if success else 10,
        grasped=held, retreat_distance_m=round(retreated, 4),
        target_error_m=round(target_error, 4),
        heading_drift_deg=round(drive["heading_drift_deg"], 3),
        cross_track_m=round(float(drive["cross_track_m"]), 4),
        stopped_on_grasp_loss=drive["interrupted"],
    )
    print(
        f"Waypoint 10: success={success}, retreat={retreated:.4f} m, "
        f"target_error={target_error:.4f} m, held={held}"
    )
    return success


def _pot_bowl_center_world(plate):
    mesh = plate.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        raise ValueError("pot collision mesh unavailable")
    vertices = np.asarray(mesh.vertices)
    middle_z = (vertices[:, 2].min() + vertices[:, 2].max()) / 2
    # Fit in world XY; the handle must not bias the bowl centre at rotated poses.
    rim = vertices[vertices[:, 2] > middle_z, :2]
    fit = _fit_rim_circle(rim) if len(rim) >= 10 else None
    if fit is None:
        raise ValueError("pot bowl rim could not be identified")
    return np.r_[fit[1], float(plate.pose.p[0, 2])]


def _approach_pot(env, planner, target_veg):
    agent = env.unwrapped.agent

    def grasp_offset():
        rotation = agent.tcp.pose.sp.to_transformation_matrix()[:3, :3]
        delta = target_veg.pose.p[0].cpu().numpy() - agent.tcp.pose.p[0].cpu().numpy()
        return rotation.T @ delta

    if not bool(agent.is_grasping(target_veg)):
        env.log_event(
            "result", "Waypoints 2-12", success=False, failed_waypoint=11,
            reason="target not held before pot approach",
        )
        return False
    relative = grasp_offset().copy()

    def lost_grasp():
        return not bool(agent.is_grasping(target_veg)) or np.linalg.norm(grasp_offset() - relative) > 0.02

    try:
        pot_center = _pot_bowl_center_world(env.unwrapped.plate)
    except ValueError as exc:
        env.log_event(
            "result", "Waypoints 2-12", success=False, failed_waypoint=11,
            reason=str(exc),
        )
        return False
    # Southeast dock keeps the west-facing hand clear of the left kitchen wall.
    approach_point = np.array([
        pot_center[0] + 0.25,
        pot_center[1] - np.sqrt(0.50**2 - 0.25**2),
        0.0,
    ])
    # Include the real payload in collision sweeps; attachment is planning-only.
    _attach_object(planner, target_veg)
    planner.planner.update_from_simulation()
    base_before = agent.base_link.pose.p[0].cpu().numpy().copy()
    direction = approach_point - base_before
    direction[2] = 0.0
    direction /= np.linalg.norm(direction)
    env.log_event(
        "waypoint_start", "Waypoint 11: turn toward pot approach point",
        pot_bowl_center_m=np.round(pot_center, 4).tolist(),
        target_position_m=np.round(approach_point, 4).tolist(),
        requested_turn_deg=round(_heading_error_degrees(agent, approach_point), 3),
    )
    motion = env.log_motion(
        "Waypoint 11: collision-aware turn with vegetable",
        planner.rotate_base_z, direction,
    )
    planner.idle_steps(t=20)
    turn_error = abs(_heading_error_degrees(agent, approach_point))
    base_drift = float(np.linalg.norm(
        agent.base_link.pose.p[0].cpu().numpy()[:2] - base_before[:2]
    ))
    turned = bool(
        motion != -1 and turn_error <= 0.15 and base_drift <= 0.02
        and not lost_grasp()
    )
    env.log_event(
        "waypoint_result", "Waypoint 11: pot approach heading", success=turned,
        turn_error_deg=round(turn_error, 3), base_drift_m=round(base_drift, 4),
    )
    if not turned:
        env.log_event(
            "result", "Waypoints 2-12", success=False, failed_waypoint=11,
            turn_error_deg=round(turn_error, 3), grasped=not lost_grasp(),
        )
        return False

    env.log_event(
        "waypoint_start", "Waypoint 12: straight drive to pot approach point",
        target_position_m=np.round(approach_point, 4).tolist(),
        target_standoff_m=0.50, yaw_command=0.0,
    )
    drive = env.log_motion(
        "Waypoint 12: approach pot while holding vegetable",
        _drive_straight_to_point, env, planner, approach_point,
        max_speed=0.10, acceleration=0.20, tolerance=0.003,
        stop_when=lost_grasp,
    )
    planner.idle_steps(t=20)
    base = agent.base_link.pose.p[0].cpu().numpy()
    target_error = float(np.linalg.norm(base[:2] - approach_point[:2]))
    held = not lost_grasp()
    success = bool(
        held and not drive["interrupted"] and not drive["stalled"]
        and target_error <= 0.015 and drive["cross_track_m"] <= 0.01
        and abs(drive["heading_drift_deg"]) <= 0.3
    )
    env.log_event(
        "waypoint_result" if success else "result",
        "Waypoint 12: pot approach validated", success=success,
        failed_waypoint=None if success else 12,
        grasped=held, approach_point_error_m=round(target_error, 4),
        pot_standoff_m=round(float(np.linalg.norm(base[:2] - pot_center[:2])), 4),
        heading_drift_deg=round(drive["heading_drift_deg"], 3),
        cross_track_m=round(float(drive["cross_track_m"]), 4),
        stopped_on_grasp_loss=drive["interrupted"],
    )
    print(
        f"Waypoints 11-12: success={success}, turn_error={turn_error:.3f} deg, "
        f"approach_error={target_error:.4f} m, held={held}"
    )
    return success


def _face_pot(env, planner, target_veg):
    agent = env.unwrapped.agent
    if not bool(agent.is_grasping(target_veg)):
        env.log_event(
            "result", "Waypoints 2-13", success=False, failed_waypoint=13,
            reason="target not held before pot-facing turn",
        )
        return False
    if WP13_TORSO_RAISE_M > 0.0:
        body = agent.controller.controllers["body"]
        torso_before = float(body.qpos[0, 2])
        torso_index = int(body.active_joint_indices[2])
        upper = float(agent.robot.get_qlimits()[0, torso_index, 1])
        if torso_before + WP13_TORSO_RAISE_M > upper - 0.001:
            env.log_event(
                "result", "Waypoint 13: pre-turn lift outside torso limits",
                success=False, failed_waypoint=13,
            )
            return False
        env.log_event(
            "waypoint_start", "Waypoint 13: raise torso before pot-facing turn",
            requested_raise_m=WP13_TORSO_RAISE_M,
        )
        env.log_motion(
            "Waypoint 13: pre-turn torso lift", lower_torso_smooth,
            env, planner, target_drop=-WP13_TORSO_RAISE_M, total_steps=60,
        )
        planner.idle_steps(t=15)
        env.log_event(
            "diagnostic", "Waypoint 13: measured pre-turn lift",
            requested_raise_m=WP13_TORSO_RAISE_M,
            actual_raise_m=float(body.qpos[0, 2]) - torso_before,
            grasped=bool(agent.is_grasping(target_veg)),
        )
        if not bool(agent.is_grasping(target_veg)):
            env.log_event(
                "result", "Waypoint 13: grasp lost during pre-turn lift",
                success=False, failed_waypoint=13,
            )
            return False
    base_before = agent.base_link.pose.p[0].cpu().numpy().copy()

    def grasp_offset():
        rotation = agent.tcp.pose.sp.to_transformation_matrix()[:3, :3]
        delta = target_veg.pose.p[0].cpu().numpy() - agent.tcp.pose.p[0].cpu().numpy()
        return rotation.T @ delta

    relative = grasp_offset().copy()
    try:
        pot_center = _pot_bowl_center_world(env.unwrapped.plate)
    except ValueError as exc:
        env.log_event(
            "result", "Waypoints 2-13", success=False, failed_waypoint=13,
            reason=str(exc),
        )
        return False
    direction = pot_center - base_before
    direction[2] = 0.0
    direction /= np.linalg.norm(direction)
    env.log_event(
        "waypoint_start", "Waypoint 13: face pot bowl centre",
        pot_bowl_center_m=np.round(pot_center, 4).tolist(),
        target_heading_deg=round(float(np.degrees(np.arctan2(direction[1], direction[0]))), 3),
        requested_turn_deg=round(_heading_error_degrees(agent, pot_center), 3),
    )
    motion = env.log_motion(
        "Waypoint 13: collision-aware turn toward pot",
        planner.rotate_base_z, direction,
    )
    planner.idle_steps(t=20)
    base = agent.base_link.pose.p[0].cpu().numpy()
    error = abs(_heading_error_degrees(agent, pot_center))
    drift = float(np.linalg.norm(base[:2] - base_before[:2]))
    slip = float(np.linalg.norm(grasp_offset() - relative))
    held = bool(agent.is_grasping(target_veg))
    success = bool(
        motion != -1 and error <= 1.0 and drift <= 0.02 and held and slip <= 0.02
    )
    env.log_event(
        "waypoint_result" if success else "result",
        "Waypoint 13: pot-facing validated", success=success,
        failed_waypoint=None if success else 13,
        pot_facing_error_deg=round(error, 3), base_drift_m=round(drift, 4),
        grasped=held, grasp_relative_error_m=round(slip, 4),
    )
    print(
        f"Waypoint 13: success={success}, pot_facing_error={error:.3f} deg, "
        f"base_drift={drift:.4f} m, held={held}"
    )
    return success


def _hand_over_pot(env, planner, target_veg):
    agent = env.unwrapped.agent
    if not bool(agent.is_grasping(target_veg)):
        env.log_event(
            "result", "Waypoints 2-14", success=False, failed_waypoint=14,
            reason="target not held before pot hover",
        )
        return False
    pot_center = _pot_bowl_center_world(env.unwrapped.plate)
    pot_mesh = env.unwrapped.plate.get_first_collision_mesh(to_world_frame=True)
    veg_mesh = target_veg.get_first_collision_mesh(to_world_frame=True)
    if pot_mesh is None or veg_mesh is None:
        env.log_event(
            "result", "Waypoints 2-14", success=False, failed_waypoint=14,
            reason="payload or pot collision mesh unavailable",
        )
        return False
    rim_z = float(pot_mesh.bounds[1, 2])
    base_before = agent.base_link.pose.p[0].cpu().numpy().copy()
    torso_before = float(agent.controller.controllers["body"].qpos[0, 2])
    tcp_before = agent.tcp.pose.p[0].cpu().numpy().copy()
    veggie_before = target_veg.pose.p[0].cpu().numpy().copy()
    relative = veggie_before - tcp_before

    def lowest_point():
        mesh = target_veg.get_first_collision_mesh(to_world_frame=True)
        return min(
            float(mesh.bounds[0, 2]),
            *(float(obj.get_global_aabb_fast()[0][2])
              for link in (agent.finger1_link, agent.finger2_link)
              for obj in link._objs),
        )

    height = float(tcp_before[2]) + max(0.0, rim_z + WP14_HOVER_CLEARANCE_M - lowest_point())
    hover = sapien.Pose(
        p=[pot_center[0] - relative[0], pot_center[1] - relative[1], height],
        q=agent.tcp.pose.sp.q,
    )
    # Refresh the planning-only attachment after any small slip during transport.
    _detach_object(planner)
    planner.planner.update_from_simulation()
    _attach_object(planner, target_veg)
    planner.planner.update_from_simulation()
    env.log_event(
        "waypoint_start", "Waypoint 14: held vegetable above pot bowl centre",
        pot_bowl_center_m=np.round(pot_center, 4).tolist(),
        target_tcp_position_m=np.round(hover.p, 4).tolist(),
        required_rim_clearance_m=WP14_HOVER_CLEARANCE_M,
    )
    old_tolerance = planner.ARM_SCREW_GOAL_TOLERANCE
    planner.ARM_SCREW_GOAL_TOLERANCE = (0.005, old_tolerance[1])
    try:
        motion = env.log_motion(
            "Waypoint 14: fixed-base and fixed-torso pot hover",
            planner.static_manipulation, hover,
            disable_lift_joint=True, n_init_qpos=100,
        )
    finally:
        planner.ARM_SCREW_GOAL_TOLERANCE = old_tolerance
    planner.idle_steps(t=20)
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    vegetable = target_veg.pose.p[0].cpu().numpy()
    xy_error = float(np.linalg.norm(vegetable[:2] - pot_center[:2]))
    clearance = lowest_point() - rim_z
    slip = float(np.linalg.norm(vegetable - tcp - relative))
    drift = float(np.linalg.norm(
        agent.base_link.pose.p[0].cpu().numpy()[:2] - base_before[:2]
    ))
    torso_drift = abs(float(agent.controller.controllers["body"].qpos[0, 2]) - torso_before)
    held = bool(agent.is_grasping(target_veg))
    success = bool(
        motion != -1 and held and xy_error <= 0.01 and clearance >= 0.04
        and slip <= 0.02 and drift <= 0.01 and torso_drift <= 0.005
    )
    env.log_event(
        "waypoint_result" if success else "result",
        "Waypoint 14: payload hover validated", success=success,
        failed_waypoint=None if success else 14,
        grasped=held, vegetable_pot_xy_error_m=round(xy_error, 4),
        rim_clearance_m=round(clearance, 4), grasp_relative_error_m=round(slip, 4),
        base_drift_m=round(drift, 4), torso_drift_m=round(torso_drift, 4),
    )
    print(
        f"Waypoint 14: success={success}, xy_error={xy_error:.4f} m, "
        f"rim_clearance={clearance:.4f} m, held={held}"
    )
    return success


def _hold_torso_absolute(env, planner, target_height, *, max_steps=80, tolerance=0.002):
    """Finish the existing ramp; retain its absolute goal, never recapture its lag."""
    agent = env.unwrapped.agent
    body_ctrl = agent.controller.controllers["body"]
    arm = agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
    body = body_ctrl.qpos[0].cpu().numpy().copy()
    body[2] = target_height
    action = np.hstack([arm, planner.gripper_state, body, _base_cmd()])
    stable = 0
    steps = 0
    for tick in range(max_steps):
        error = abs(float(body_ctrl.qpos[0, 2]) - target_height)
        stable = stable + 1 if error <= tolerance else 0
        if stable >= 5:
            break
        env.step(action.copy())
        steps += 1
    error = abs(float(body_ctrl.qpos[0, 2]) - target_height)
    planner.planner.update_from_simulation()
    env.log_event("diagnostic", "Fixed-goal torso settling", target_height_m=float(target_height),
                  extra_steps=steps, error_m=error, converged=bool(error <= tolerance))
    return {"steps": steps, "error_m": error, "converged": bool(error <= tolerance)}


def _finish_placement(env, planner, target_veg):
    scene = env.unwrapped
    agent = scene.agent
    body = agent.controller.controllers["body"]
    if not bool(agent.is_grasping(target_veg)):
        env.log_event(
            "result", "Waypoints 2-17", success=False, failed_waypoint=15,
            reason="target not held before pot descent",
        )
        return False
    pot_center = _pot_bowl_center_world(scene.plate)
    pot_mesh = scene.plate.get_first_collision_mesh(to_world_frame=True)
    mesh = target_veg.get_first_collision_mesh(to_world_frame=True)
    floor_z = float(pot_mesh.bounds[0, 2])
    bottom_z = float(mesh.bounds[0, 2])
    torso_start = float(body.qpos[0, 2])
    torso_min = float(agent.robot.get_qlimits()[0, int(body.active_joint_indices[2]), 0])
    base_start = agent.base_link.pose.p[0].cpu().numpy().copy()
    requested_drop = max(0.0, bottom_z - (floor_z + 0.01))
    available = max(0.0, torso_start - torso_min - 0.005)
    drop = min(requested_drop, available)
    if drop <= 0.0 or bottom_z - drop > floor_z + 0.03:
        env.log_event(
            "result", "Waypoints 2-17", success=False, failed_waypoint=15,
            reason="insufficient torso travel for a low release",
        )
        return False
    env.log_event(
        "waypoint_start", "Waypoint 15: lower vegetable into pot",
        torso_drop_m=round(drop, 4), pot_floor_z_m=round(floor_z, 4),
    )
    env.log_motion(
        "Waypoint 15: smooth torso descent", lower_torso_smooth,
        env, planner, target_drop=drop, total_steps=60,
    )
    planner.idle_steps(t=10)
    vegetable = target_veg.pose.p[0].cpu().numpy()
    bottom_z = float(target_veg.get_first_collision_mesh(to_world_frame=True).bounds[0, 2])
    xy_error = float(np.linalg.norm(vegetable[:2] - pot_center[:2]))
    held = bool(agent.is_grasping(target_veg))
    lowered = held and xy_error <= 0.02 and bottom_z <= floor_z + 0.03
    env.log_event(
        "waypoint_result", "Waypoint 15: low release height", success=bool(lowered),
        vegetable_pot_xy_error_m=round(xy_error, 4),
        vegetable_bottom_above_floor_m=round(bottom_z - floor_z, 4),
    )
    if not lowered:
        env.log_event("result", "Waypoints 2-17", success=False, failed_waypoint=15)
        return False

    env.log_event("waypoint_start", "Waypoint 16: release vegetable in pot")
    env.log_motion("Waypoint 16: open gripper", planner.open_gripper, t=20, ramp=15)
    _detach_object(planner)
    planner.planner.update_from_simulation()
    planner.idle_steps(t=20)
    released = not bool(agent.is_grasping(target_veg))
    env.log_event("waypoint_result", "Waypoint 16: gripper released", success=released)
    if not released:
        env.log_event("result", "Waypoints 2-17", success=False, failed_waypoint=16)
        return False

    raise_amount = torso_start - float(body.qpos[0, 2])
    env.log_event(
        "waypoint_start", "Waypoint 17: raise torso after release",
        torso_raise_m=round(raise_amount, 4),
    )
    env.log_motion(
        "Waypoint 17: smooth torso raise", lower_torso_smooth,
        env, planner, target_drop=-raise_amount, total_steps=60,
    )
    _hold_torso_absolute(env, planner, torso_start)
    planner.idle_steps(t=20)
    stable = 0
    evaluation = scene.evaluate()
    for _ in range(300):
        planner.idle_steps(t=1)
        evaluation = scene.evaluate()
        stable = stable + 1 if bool(evaluation["success"]) else 0
        if stable >= 20:
            break
    torso_error = abs(float(body.qpos[0, 2]) - torso_start)
    base_drift = float(np.linalg.norm(
        agent.base_link.pose.p[0].cpu().numpy()[:2] - base_start[:2]
    ))
    task_success = bool(evaluation["success"])
    success = bool(task_success and stable >= 20 and torso_error <= 0.01 and base_drift <= 0.01)
    env.log_event(
        "result", "Task complete: waypoints 2-17", success=success,
        task_success=task_success, stable_success_steps=stable,
        torso_restore_error_m=round(torso_error, 4), base_drift_m=round(base_drift, 4),
        evaluation=evaluation,
    )
    print(
        f"Task complete: success={success}, env_success={task_success}, "
        f"stable_steps={stable}, torso_error={torso_error:.4f} m"
    )
    return success


def planning(env, seed, debug=False):
    _install_render_cameras(env)
    env.unwrapped.ROBOT_FRONT_DISTANCE = ROBOT_FRONT_DISTANCE_M
    env.reset(seed=seed, options={"reconfigure": True})
    scene = env.unwrapped
    agent = scene.agent
    _pose_render_cameras(env)
    # Preserve the approved ready pose before first physics step; reset starts
    # with the extended arm intersecting the fridge and can shove the base.
    _, ready_qpos = _ready_qpos(agent)
    agent.robot.set_qpos(ready_qpos)
    planner = FetchMotionPlanningSapienSolver(
        env, base_pose=agent.robot.pose, vis=False, debug=debug
    )
    planner.MAX_REFINE_STEPS = 100

    target_veg = scene.veggies[scene._picture_target]
    for index, veggie in enumerate(scene.veggies):
        env.track_object(veggie, f"veg_{index}")
    env.track_object(agent.tcp, "robot_tcp")
    env.track_object(agent.base_link, "robot_base")
    env.track_object(scene.plate, "pot")
    env.log_event("start", "Waypoints 2-12: grasp and approach pot")

    planner.open_gripper()
    planner.planner.update_from_simulation()

    requested_standoff = 0.5
    veg_center = target_veg.pose.p[0].cpu().numpy().copy()
    # This scene's counter front faces -Y, not toward the robot's spawn.
    approach_point = np.array([veg_center[0], veg_center[1] - requested_standoff, 0.0])
    requested_turn = _heading_error_degrees(agent, approach_point)
    env.log_event(
        "waypoint_start",
        "Waypoint 2: shortest-arc turn toward approach point",
        requested_turn_deg=round(requested_turn, 2),
        vegetable_position_m=np.round(veg_center, 4).tolist(),
        target_position_m=np.round(approach_point, 4).tolist(),
    )
    turn_error = env.log_motion(
        "Waypoint 2: turn toward approach point",
        _turn_toward_point,
        env,
        planner,
        approach_point,
    )
    if turn_error > 0.15:
        planner.idle_steps(t=30)
        env.log_event(
            "result",
            "Waypoints 2-4",
            success=False,
            failed_waypoint=2,
            turn_error_deg=round(turn_error, 2),
        )
        return False

    env.log_event(
        "waypoint_start",
        "Waypoint 3: straight to point 0.5 m in front of vegetable",
        target_position_m=np.round(approach_point, 4).tolist(),
        target_standoff_m=requested_standoff,
        yaw_command=0.0,
    )
    drive = env.log_motion(
        "Waypoint 3: straight drive to approach point",
        _drive_straight_to_point,
        env,
        planner,
        approach_point,
    )
    planner.idle_steps(t=10)

    base = agent.base_link.pose.p[0].cpu().numpy()
    veggie = target_veg.pose.p[0].cpu().numpy()
    target_error = float(np.linalg.norm(base[:2] - approach_point[:2]))
    standoff = float(np.linalg.norm(base[:2] - veggie[:2]))
    standoff_error = abs(standoff - requested_standoff)
    success = (
        target_error <= 0.03
        and standoff_error <= 0.05
        and drive["cross_track_m"] <= 0.05
        and abs(drive["heading_drift_deg"]) <= 3.0
        and not drive["stalled"]
    )
    env.log_event(
        "waypoint_result",
        "Waypoint 3: approach point",
        success=success,
        turn_error_deg=round(turn_error, 2),
        initial_target_error_m=round(drive["initial_target_error_m"], 3),
        approach_point_error_m=round(target_error, 3),
        remaining_to_point_m=round(drive["final_remaining_m"], 3),
        straight_cross_track_m=round(drive["cross_track_m"], 3),
        vegetable_standoff_m=round(standoff, 3),
        standoff_error_m=round(standoff_error, 3),
        straight_drive_steps=drive["steps"],
        heading_drift_deg=round(drive["heading_drift_deg"], 2),
        stalled=drive["stalled"],
    )
    print(
        f"Waypoints 2-3: success={success}, turn_error={turn_error:.1f} deg, "
        f"approach_point_error={target_error:.3f} m, standoff={standoff:.3f} m, "
        f"cross_track={drive['cross_track_m']:.3f} m, "
        f"heading_drift={drive['heading_drift_deg']:.2f} deg, "
        f"drive_steps={drive['steps']}"
    )
    if not success:
        env.log_event(
            "result", "Waypoints 2-4", success=False, failed_waypoint=3
        )
        return False

    base_before = agent.base_link.pose.p[0].cpu().numpy().copy()
    veggie_before = target_veg.pose.p[0].cpu().numpy().copy()
    # Face the counter normal (+Y), without correcting position by driving.
    counter_normal = np.array([0.0, 1.0, 0.0])
    env.log_event(
        "waypoint_start",
        "Waypoint 4: face vegetable perpendicular to counter",
        requested_turn_deg=round(
            _heading_error_degrees(agent, base_before + counter_normal), 2
        ),
        target_heading_deg=90.0,
        base_position_m=np.round(base_before, 4).tolist(),
    )
    torso_before = float(agent.controller.controllers["body"].qpos[0, 2])
    env.log_motion(
        "Waypoint 4: lower torso for clearance",
        lower_torso_smooth,
        env,
        planner,
        target_drop=WP4_TORSO_DROP_M,
        total_steps=60,
    )
    planner.idle_steps(t=10)
    torso_drop = torso_before - float(agent.controller.controllers["body"].qpos[0, 2])
    if abs(torso_drop - WP4_TORSO_DROP_M) > 0.01:
        env.log_event(
            "result",
            "Waypoints 2-4",
            success=False,
            failed_waypoint=4,
            reason="torso did not reach clearance height",
            torso_drop_m=round(torso_drop, 4),
        )
        return False
    rotation = env.log_motion(
        "Waypoint 4: perpendicular turn in place",
        planner.rotate_base_z,
        counter_normal,
    )
    planner.idle_steps(t=20)
    base = agent.base_link.pose.p[0].cpu().numpy()
    veggie = target_veg.pose.p[0].cpu().numpy()
    perpendicular_error = abs(
        _heading_error_degrees(agent, base + np.array([0.0, 1.0, 0.0]))
    )
    facing_error = abs(_heading_error_degrees(agent, veggie))
    base_drift = float(np.linalg.norm(base[:2] - base_before[:2]))
    veggie_drift = float(np.linalg.norm(veggie - veggie_before))
    standoff = float(np.linalg.norm(base[:2] - veggie[:2]))
    success = (
        rotation != -1
        and perpendicular_error <= 1.0
        and facing_error <= 2.0
        and base_drift <= 0.02
        and veggie_drift <= 0.01
        and abs(standoff - requested_standoff) <= 0.03
    )
    env.log_event(
        "waypoint_result",
        "Waypoint 4: perpendicular turn",
        success=success,
        torso_drop_m=round(torso_drop, 4),
        perpendicular_error_deg=round(perpendicular_error, 3),
        vegetable_facing_error_deg=round(facing_error, 3),
        base_drift_m=round(base_drift, 4),
        vegetable_drift_m=round(veggie_drift, 4),
        vegetable_standoff_m=round(standoff, 3),
    )
    print(
        f"Waypoint 4: success={success}, "
        f"perpendicular_error={perpendicular_error:.3f} deg, "
        f"facing_error={facing_error:.3f} deg, base_drift={base_drift:.4f} m, "
        f"standoff={standoff:.3f} m"
    )
    if not success:
        env.log_event(
            "result", "Waypoints 2-5", success=False, failed_waypoint=4
        )
        return False

    base_before = agent.base_link.pose.p[0].cpu().numpy().copy()
    torso_before = float(agent.controller.controllers["body"].qpos[0, 2])
    veggie_before = target_veg.pose.p[0].cpu().numpy().copy()
    mesh = target_veg.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        env.log_event(
            "result", "Waypoints 2-5", success=False, failed_waypoint=5,
            reason="target collision mesh unavailable",
        )
        return False
    vegetable_top = float(mesh.bounds[1, 2])
    hover = agent.build_grasp_pose(
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([veggie_before[0], veggie_before[1], vegetable_top + 0.10]),
    )
    env.log_event(
        "waypoint_start",
        "Waypoint 5: top-down hand above target vegetable",
        target_index=scene._picture_target,
        target_name=target_veg.name,
        target_tcp_position_m=np.round(hover.p, 4).tolist(),
        tcp_clearance_above_vegetable_m=round(float(hover.p[2]) - vegetable_top, 4),
    )
    motion = env.log_motion(
        "Waypoint 5: fixed-base and fixed-torso hover",
        planner.static_manipulation,
        hover,
        disable_lift_joint=True,
        n_init_qpos=100,
    )
    planner.idle_steps(t=20)
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    actual_q = agent.tcp.pose.q[0].cpu().numpy()
    position_error = float(np.linalg.norm(tcp - hover.p))
    xy_error = float(np.linalg.norm(tcp[:2] - target_veg.pose.p[0].cpu().numpy()[:2]))
    orientation_error = float(np.degrees(
        2 * np.arccos(np.clip(abs(float(np.dot(actual_q, hover.q))), 0.0, 1.0))
    ))
    base_drift = float(np.linalg.norm(
        agent.base_link.pose.p[0].cpu().numpy()[:2] - base_before[:2]
    ))
    torso_drift = abs(float(agent.controller.controllers["body"].qpos[0, 2]) - torso_before)
    veggie_drift = float(np.linalg.norm(
        target_veg.pose.p[0].cpu().numpy() - veggie_before
    ))
    success = (
        motion != -1
        and position_error <= 0.02
        and xy_error <= 0.01
        and float(tcp[2]) > vegetable_top
        and orientation_error <= 3.0
        and base_drift <= 0.01
        and torso_drift <= 0.005
        and veggie_drift <= 0.01
    )
    env.log_event(
        "waypoint_result",
        "Waypoint 5: top-down hover",
        success=success,
        tcp_position_error_m=round(position_error, 4),
        tcp_xy_error_m=round(xy_error, 4),
        tcp_orientation_error_deg=round(orientation_error, 3),
        tcp_approach_axis=np.round(
            agent.tcp.pose.sp.to_transformation_matrix()[:3, 2], 4
        ).tolist(),
        tcp_clearance_above_vegetable_m=round(float(tcp[2]) - vegetable_top, 4),
        base_drift_m=round(base_drift, 4),
        torso_drift_m=round(torso_drift, 4),
        vegetable_drift_m=round(veggie_drift, 4),
    )
    print(
        f"Waypoint 5: success={success}, tcp_error={position_error:.4f} m, "
        f"xy_error={xy_error:.4f} m, orientation_error={orientation_error:.3f} deg"
    )
    if not success:
        env.log_event("result", "Waypoints 2-6", success=False, failed_waypoint=5)
        return False

    base_before = agent.base_link.pose.p[0].cpu().numpy().copy()
    torso_before = float(agent.controller.controllers["body"].qpos[0, 2])
    veggie_before = target_veg.pose.p[0].cpu().numpy().copy()
    tcp_before = agent.tcp.pose.p[0].cpu().numpy().copy()
    mesh = target_veg.get_first_collision_mesh(to_world_frame=True)
    try:
        if mesh is None:
            raise ValueError("target collision mesh unavailable")
        closing = _narrow_closing_axis(
            mesh, agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
        )
    except ValueError as exc:
        env.log_event(
            "result", "Waypoints 2-6", success=False, failed_waypoint=6,
            reason=str(exc),
        )
        return False
    grasp_orientation = agent.build_grasp_pose(
        np.array([0.0, 0.0, -1.0]), closing, tcp_before
    )
    env.log_event(
        "waypoint_start",
        "Waypoint 6: align fingers across vegetable narrow axis",
        target_index=scene._picture_target,
        target_name=target_veg.name,
        closing_axis_world=np.round(closing, 4).tolist(),
        tcp_position_m=np.round(tcp_before, 4).tolist(),
    )
    old_tolerance = planner.ARM_SCREW_GOAL_TOLERANCE
    planner.ARM_SCREW_GOAL_TOLERANCE = (old_tolerance[0], 0.025)
    try:
        motion = env.log_motion(
            "Waypoint 6: fixed-position grasp-axis rotation",
            planner.static_manipulation,
            grasp_orientation,
            disable_lift_joint=True,
            n_init_qpos=100,
        )
    finally:
        planner.ARM_SCREW_GOAL_TOLERANCE = old_tolerance
    planner.idle_steps(t=20)
    frame = agent.tcp.pose.sp.to_transformation_matrix()
    closing_error = float(np.degrees(np.arccos(np.clip(
        abs(float(np.dot(frame[:3, 1], closing))), 0.0, 1.0
    ))))
    downward_error = float(np.degrees(np.arccos(np.clip(-frame[2, 2], -1.0, 1.0))))
    position_drift = float(np.linalg.norm(
        agent.tcp.pose.p[0].cpu().numpy() - tcp_before
    ))
    base_drift = float(np.linalg.norm(
        agent.base_link.pose.p[0].cpu().numpy()[:2] - base_before[:2]
    ))
    torso_drift = abs(float(agent.controller.controllers["body"].qpos[0, 2]) - torso_before)
    veggie_drift = float(np.linalg.norm(
        target_veg.pose.p[0].cpu().numpy() - veggie_before
    ))
    success = (
        motion != -1
        and closing_error <= 3.0
        and downward_error <= 3.0
        and position_drift <= 0.01
        and base_drift <= 0.01
        and torso_drift <= 0.005
        and veggie_drift <= 0.01
    )
    env.log_event(
        "waypoint_result", "Waypoint 6: grasp axis aligned", success=success,
        closing_axis_error_deg=round(closing_error, 3),
        downward_error_deg=round(downward_error, 3),
        tcp_position_drift_m=round(position_drift, 4),
        base_drift_m=round(base_drift, 4),
        torso_drift_m=round(torso_drift, 4),
        vegetable_drift_m=round(veggie_drift, 4),
    )
    print(
        f"Waypoint 6: success={success}, closing_error={closing_error:.3f} deg, "
        f"downward_error={downward_error:.3f} deg, tcp_drift={position_drift:.4f} m"
    )
    if not success:
        env.log_event("result", "Waypoints 2-9", success=False, failed_waypoint=6)
        return False
    if not _grasp_and_lift(env, planner, target_veg):
        return False
    if not _retreat_with_vegetable(env, planner, target_veg):
        return False
    if not _approach_pot(env, planner, target_veg):
        return False
    if not _face_pot(env, planner, target_veg):
        return False
    if not _hand_over_pot(env, planner, target_veg):
        return False
    return _finish_placement(env, planner, target_veg)


def main():
    parser = argparse.ArgumentParser(description="Veggie planner waypoints 2 through 17")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    import mplib.pymp as mplib_pymp

    mplib_pymp.set_global_seed(args.seed)

    run_dir = args.log_dir / (
        f"veggies_wp02-17_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "MyRoboCasa_FridgeVeggies-v1",
        num_envs=1,
        render_mode="rgb_array",
        obs_mode="rgb",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
        sim_config=dict(
            scene_config=dict(cpu_workers=1, enable_enhanced_determinism=True)
        ),
    )
    env = StreamingVideoRecorder(env, output_dir=str(run_dir), video_fps=30)
    env = PlannerLogger(
        env,
        log_dir=str(run_dir),
        name=f"veggies_wp02-17_seed{args.seed}",
        log_freq=1,
        run_dir=str(run_dir),
    )
    try:
        env.action_space.seed(args.seed)
        with capture_stdout(env.dir / "console.log"):
            success = planning(env, args.seed, debug=args.debug)
    finally:
        env.close()
    print(f"Waypoints 2-17 logs/video: {run_dir.resolve()}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
