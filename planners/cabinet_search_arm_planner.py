"""Search four compartments by opening and closing each bar with the arm.

The base stays still during the hinge stroke. An empty door is closed on the
same grip; every inspected compartment is followed by a visit to the marker.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import sapien

from planners import cabinet_retrieval_planner as retrieval
from planners import cabinet_search_planner as search
from planners.oracle import oracle_common as common
from utils.mikasa.execution_noise import configure_execution_noise
from utils.mikasa.seeding import seed_everything


def refused(result):
    return isinstance(result, (int, np.integer)) and result == -1


def arc_pose(tcp, anchor, angle):
    """Rotate the contact point around the vertical hinge, preserving jaw yaw.

    The vertical round bar can turn inside the fingers. Rotating the entire
    wrist with the leaf instead exhausts the arm's reach before the reveal.
    """
    centre = np.r_[np.asarray(anchor), tcp.p[2]]
    rotate = (sapien.Pose(centre, [math.cos(angle / 2), 0, 0, math.sin(angle / 2)])
              * sapien.Pose(-centre))
    return sapien.Pose((rotate * tcp).p, tcp.q)


def solve(env, seed=None, debug=False, vis=False, blind=False,
          planner_factory=common.collection_planner_factory,
          waypoint_noise_seed=None, waypoint_noise_m=0.005,
          execution_noise_seed=None, action_noise=0.003, noise_hold=10):
    env.reset(seed=seed)
    seed_everything(seed)
    task = env.unwrapped
    if task.control_mode != "pd_joint_pos":
        raise ValueError("Collection requires the canonical pd_joint_pos controller")
    cfg = task.cfg
    if len(cfg.compartments) != 4:
        raise ValueError("This oracle covers the four upper compartments")
    planner = planner_factory(env, debug, vis)
    planner.forward_navigation = True
    planner.navigation_arrival_tolerance = 0.08
    planner.touch_dock_aisle_y = -1.75
    planner.touch_dock_y = -1.22
    # Keep the elbow folded on the handle's side of its straight configuration.
    # Crossing from a negative elbow to +2.1 beside an open leaf sweeps the arm
    # through that leaf. This fixed pose has a bent elbow and the hand near the
    # body, and is also reachable after the can contact.
    planner.navigation_posture = {
        "torso_lift_joint": .3, "shoulder_pan_joint": 0.,
        "shoulder_lift_joint": 0., "upperarm_roll_joint": 0.,
        "elbow_flex_joint": -2.1, "forearm_roll_joint": 0.,
        "wrist_flex_joint": -.4, "wrist_roll_joint": 0.,
    }
    planner.prefer_low_roll_ik()
    configure_execution_noise(
        planner, env, int(seed or 0) + 300007 if execution_noise_seed is None else execution_noise_seed,
        action_noise=action_noise, noise_hold=noise_hold)
    if not 0 <= waypoint_noise_m <= 0.01:
        raise ValueError("Free-floor waypoint noise must be in [0, 0.01] m")
    planner.cabinet_waypoint_noise = np.random.default_rng(
        int(seed or 0) + 200003 if waypoint_noise_seed is None else waypoint_noise_seed)
    planner.cabinet_waypoint_noise_m = float(waypoint_noise_m)
    log = lambda message, **kw: search.say(env, message, **kw)

    def stopped(result):
        return refused(result) or common.stopped_by_horizon(planner)

    def gaze(cab=None):
        centres = search._np(task._spawn_centre_x).reshape(-1, 4)[0]
        return np.array([float(centres.mean() if cab is None else centres[cab]),
                         cfg.spawn_depth, cfg.shelf_top_z + cfg.cube_half])

    def withdraw(art_name):
        result = planner.open_gripper(t=10)
        if stopped(result):
            return result
        planner._grasp_branch = {}
        planner.planner.update_from_simulation()
        tcp = task.agent.tcp.pose[0].sp
        goal = sapien.Pose(tcp.p - .20 * tcp.to_transformation_matrix()[:3, 2], tcp.q)
        with common.contact_stroke(planner, [art_name]):
            result = planner.static_manipulation(
                goal, disable_lift_joint=True, by_line=False,
                max_knots=100, knot_refuse=True, stretch=1)
        if stopped(result):
            return result
        planner.planner.update_from_simulation()
        return search.drive_posture(env, planner, task, label="fold after the handle")

    planner.track_target(lambda: task.home_marker.pose[0].sp.p)
    result = search.drive_posture(env, planner, task, label="fold for the first drive")
    if stopped(result):
        return result
    result = search.drive_home(env, planner, task, via_south=False)
    if stopped(result) or not search._b(result[-1], "passed_home"):
        return result
    rng, order = search.search_plan(seed, 4)
    log("episode", n=4, blind=bool(blind), route="coin" if blind else order.tolist())
    for k in range(4):
        cab = search.pick_compartment(rng, order, k, blind=blind)
        comp = cfg.compartments[cab]
        door = search.door_spec_for(comp, closed_rad=cfg.theta_closed)
        planner.track_target(lambda cab=cab: gaze(cab))
        log("round", k=k, pick=cab, compartment=comp.name)
        result = planner.drive_base(
            target_pos=np.array([*door.handle_dock[:2], 0.]),
            target_view_vec=np.array([0., 1., 0.]), freeze_arm=True)
        if stopped(result):
            return result
        planner.planner.update_from_simulation()
        # The four handles share height and dock-relative geometry. Enter the
        # same elbow-down pregrasp before solving the short contact stroke.
        ready = dict(zip(
            task.agent.controller.controllers["arm"].config.joint_names,
            [.153, -1.014, .553, -.401, -.624, 1.334, -.151]))
        ready["torso_lift_joint"] = .205
        result = common.plan_joints(env, planner, task, ready,
                                   label="handle pregrasp branch", who=search.WHO)
        if stopped(result):
            return result
        planner.set_grasp_branch(elbow=-1, wrist=1)
        grasp = task.agent.build_grasp_pose(
            np.array([0., 1., 0.]), np.array([1., 0., 0.]), np.array(door.handle_bar))
        planner.planner.update_from_simulation()
        with common.contact_stroke(planner, [door.stem]):
            result = planner.static_manipulation(
                grasp, stop_on_touch=SimpleNamespace(name=door.stem),
                by_line=False, stretch=1)
        if stopped(result):
            return result
        result = planner.close_gripper(t=12)
        if stopped(result):
            return result
        art_name = retrieval._art_key(task, door)
        planner.planner.update_from_simulation()
        anchor, sense = common.hinge_anchor(task, art_name, door.hinge)
        start_tcp = task.agent.tcp.pose[0].sp
        direction = sense * door.open_dir

        def move_arc(delta, phase):
            goal = arc_pose(start_tcp, anchor, direction * delta)
            with common.contact_stroke(planner, [art_name]):
                r = planner.static_manipulation(
                    goal, disable_lift_joint=(phase == "open"), by_line=False, n_init_qpos=40,
                    max_knots=100, knot_draws=1, knot_refuse=True, stretch=1)
            log("arm hinge arc", phase=phase, delta=float(delta),
                door_rad=float(retrieval.door_rad_now(task, door)))
            return r

        for delta in np.linspace(.08, 1.44, 18):
            if float(task.agent.robot.get_qpos()[0, -2:].sum()) <= common.FINGER_EMPTY_M:
                log("MISSED: empty handle grasp")
                return result
            result = move_arc(delta, "open")
            if stopped(result):
                return result
        if retrieval.door_rad_now(task, door) < cfg.theta_reveal:
            log("MISSED: door below reveal angle")
            return result
        aligned = 0
        for _ in range(34):
            result = planner.idle_steps(t=1)
            if stopped(result):
                return result
            aligned = aligned + 1 if planner._head_tracker.aligned() else 0
            if aligned >= max(int(cfg.reveal_dwell_steps), 1):
                break
        if search._b(result[-1], "fail"):
            return result
        found = search._b(result[-1], "revealed")
        if not found:
            for delta in np.linspace(1.36, 0., 18):
                result = move_arc(delta, "close")
                if stopped(result):
                    return result
                if retrieval.door_rad_now(task, door) <= .08:
                    break
            if retrieval.door_rad_now(task, door) > cfg.theta_closed:
                log("MISSED: door did not close on the held bar")
                return result
        # Aim at the next visible task goal while the arm withdraws and folds.
        # A return trip must show the floor marker, not the cabinet just inspected.
        planner.track_target((lambda: task.cube.pose[0].sp.p) if found
                             else (lambda: task.home_marker.pose[0].sp.p))
        result = withdraw(art_name)
        if stopped(result):
            return result
        if found and cfg.terminal == "nudge":
            planner.track_target(lambda: task.cube.pose[0].sp.p)
            result = search.nudge_the_cube(env, planner, task, result, cab=cab)
            if stopped(result) or not search._b(result[-1], "found"):
                return result
            planner.track_target(lambda: task.home_marker.pose[0].sp.p)
            result = search.drive_posture(env, planner, task, label="fold after the can")
            if stopped(result):
                return result
        result = search.drive_home(env, planner, task, via_south=True)
        if stopped(result):
            return result
        log("home after inspection", k=k, found=found, **search._latches(result[-1]))
        if search._b(result[-1], "success"):
            return result
        if found or not search._b(result[-1], "passed_home"):
            return result
    return result
