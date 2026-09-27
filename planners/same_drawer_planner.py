"""Action-only SameDrawer oracle: close, apple transfer, return, remembered pull.

The drawer answer is read once at reset. Blind evaluation replaces only the final
choice with an independent uniform guess. Every motion is an IK/RRT waypoint or a
base velocity command; fixture joints and robot state are never written here.
"""
from __future__ import annotations

import argparse
import math

import gymnasium as gym
import numpy as np
import sapien

import my_scenes  # noqa: F401
from my_scenes.same_drawer import DRAWER_ART_SUFFIX, DRAWER_FIXTURES
from planners.oracle import oracle_common as common
from planners.oracle.rest_paths import curved_rest
from planners.oracle.settling import ReleaseSettling
from planners.oracle.straight_paths import straight_plan, execute_straight
from planners.same_drawer_paths import DrawerPathPlanner
from planners.same_drawer_approach import direct_bottom_contact
from utils.mikasa.seeding import seed_everything
from utils.mikasa.waypoint_noise import WaypointNoise
from utils.mikasa.execution_noise import configure_execution_noise, transfer_phase

WHO = "same_drawer_planner"
REACH_N_INIT = 40


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def bar_poses(task, drawer, amount, *, flip=False, opening=False):
    home = array(task.handle_home)[0, drawer].astype(float)
    centre = home + [0, -float(amount), 0]
    # Approach diagonally from the right. A horizontal closing axis keeps the
    # wrist camera above the hand while the pads close across the handle bar.
    # Below waist height, approach from above as well as from the side.
    # A level approach to the low handle nearly straightens the elbow; the
    # following contact screw can then oscillate through a very slow joint path.
    bottom_pull = opening and centre[2] < 0.25
    middle_handle = 0.5 <= centre[2] < 0.65
    pitch = math.radians(40 if bottom_pull else 30) if centre[2] < 0.5 or middle_handle else 0.0
    upper_push = not opening and centre[2] >= 0.5
    yaw = math.radians(155 if bottom_pull else 145 if upper_push or middle_handle else 135)
    approaching = np.array([math.cos(yaw) * math.cos(pitch),
                            math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
    closing = np.cross(approaching, np.array([0., 0., 1.]))
    closing /= np.linalg.norm(closing)
    if bottom_pull:
        # A horizontal jaw enters the drawer front before enclosing this low
        # bar. Tilt the closing axis diagonally around the approach direction
        # so one finger passes above the bar and the other below it.
        jaw_angle = math.radians(45)
        closing = (closing * math.cos(jaw_angle)
                   + np.cross(approaching, closing) * math.sin(jaw_angle))
    if flip:
        closing = -closing
    # The same centred handle grasp is used in both directions. Leave space
    # for the drawer front behind the pads, and enter along the finger axis.
    centre -= 0.01 * approaching
    grasp = task.agent.build_grasp_pose(approaching, closing, centre)
    # Enter and leave along the same finger axis. The closing dock is farther
    # back by the drawer extension, so an open drawer does not crowd the arm.
    reach = centre - 0.14 * approaching
    return grasp, sapien.Pose(reach, grasp.q)


def solve(env, seed=None, debug=False, vis=False, blind=False, *,
          waypoint_noise_seed=None, waypoint_noise_m=0.005,
          planner_factory=common.collection_planner_factory,
          execution_noise_seed=None, action_noise=0.003, noise_hold=10):
    with common.planning_budget(4.0):
        return _solve(env, seed, debug, vis, blind, waypoint_noise_seed,
                      waypoint_noise_m, planner_factory, execution_noise_seed, action_noise, noise_hold)


def _solve(env, seed, debug, vis, blind, noise_seed, noise_m, planner_factory,
           execution_noise_seed, action_noise, noise_hold):
    env.reset(seed=seed)
    seed_everything(seed)
    task = env.unwrapped
    assert task.control_mode == "pd_joint_pos", task.control_mode
    cfg = task.cfg
    # The new bottom approach is explicit; withdrawn distance variants stay off.
    selected = dict(closing_approach="bottom_continuous", drawer_backoff_m=0.,
                    apple_backoff_m=0., plate_backoff_m=0.,
                    skip_plate_step_if_reachable=False)
    for name, value in getattr(task, "motion_parameters", {}).items():
        if name not in selected or value != selected[name]:
            raise ValueError(f"SameDrawer motion option was not selected: {name}={value}")
    planner = planner_factory(env, debug, vis)
    planner.prefer_low_roll_ik()
    reserved = np.asarray(task.agent.keyframes["rest"].qpos)[planner._roll_indices]
    planner._roll_low = np.minimum(planner._roll_low, reserved)
    planner._roll_high = np.maximum(planner._roll_high, reserved)
    planner.planner = DrawerPathPlanner(planner.planner._planner, planner)
    planner.forward_navigation = True
    planner.navigation_arrival_tolerance = 0.10
    configure_execution_noise(
        planner, env, int(seed or 0) + 300007 if execution_noise_seed is None else execution_noise_seed,
        action_noise=action_noise, noise_hold=noise_hold)
    log = lambda stage, **kw: common.say(env, WHO, stage, **kw)
    noise = WaypointNoise((0 if seed is None else seed) + 200003 if noise_seed is None
                          else noise_seed, noise_m, log)
    target = int(array(task.target_drawer).item())
    reopen = int(np.random.default_rng((0 if seed is None else seed) + 900019).choice(cfg.drawer_choices)) if blind else target
    log("remember open drawer", target=target, reopen=reopen, blind=bool(blind))

    def stopped(result):
        return isinstance(result, (int, np.integer)) and result == -1 or common.stopped_by_horizon(planner)

    def wait_for_state(label, condition, max_steps, result):
        waited = 0
        while not condition() and waited < max_steps:
            result = planner.idle_steps(t=1)
            waited += 1
            if stopped(result):
                break
        log(label, waited_steps=waited, reached=bool(condition()))
        return result

    def move(label, pose, *, noisy=True, axes=(True, True, True), sync=True, contact=False, freeze_lift=False, stop_on_touch=None, contact_stretch=2):
        if noisy:
            pose = noise.pose(label, pose, axes)
        log(label, goal=pose.p.tolist())
        if sync:
            planner.planner.update_from_simulation()
        fixed_lift = freeze_lift
        if contact:
            # Contact must follow a Cartesian line, not a joint-space/RRT detour.
            # Freeze the lift first: a lowered torso otherwise jams the Jacobian
            # solver at its lower limit despite a feasible arm-only stroke.
            fixed_lift = common.screw_plans(planner, pose, disable_lift_joint=True)
            if not fixed_lift and not common.screw_plans(planner, pose):
                return common.fail(env, WHO, "no straight contact path", waypoint=label)
        return planner.static_manipulation(pose, disable_lift_joint=fixed_lift,
            n_init_qpos=REACH_N_INIT, max_knots=160, knot_draws=1, knot_refuse=True,
            by_line=not contact, stretch=contact_stretch if contact else 1, stop_on_touch=stop_on_touch)

    def drive(label, point, yaw=math.pi/2, *, already_at_station=False):
        p = noise.point(label, [float(point[0]), float(point[1]), 0.], (True, True, False))
        log(label, position=p.tolist())
        if already_at_station:
            forward = task.agent.base_link.pose[0].sp.to_transformation_matrix()[:3, 0]
            heading = math.atan2(float(forward[1]), float(forward[0]))
            error = (yaw - heading + math.pi) % (2 * math.pi) - math.pi
            if abs(error) <= math.radians(3):
                # The visible open handle is already reachable from this pose.
                return planner._guard.last_step
        return planner.drive_base(target_pos=None if already_at_station else p, target_view_vec=[math.cos(yaw), math.sin(yaw), 0.],
                                  freeze_arm=True)

    def gaze(point):
        # The following drive/approach provides settling time while tracking.
        planner.track_target(point)
        return planner._guard.last_step

    def approach_for_contact(*args, **kwargs):
        planner.set_grasp_branch(elbow=1, wrist=1)
        return planner.approach_for_contact(*args, **kwargs)

    def open_gripper(t=6):
        aperture = float(array(task.agent.robot.get_qpos())[0, -2:].sum())
        if planner.gripper_state > 0 and aperture >= 0.095:
            return planner._guard.last_step
        return planner.open_gripper(t=t)

    def close_gripper(*args, **kwargs):
        result = planner.close_gripper(*args, **kwargs)
        planner._grasp_branch = {}
        return result

    def stow(*, from_handle=False, from_apple=False):
        planner._grasp_branch = {}
        rest = np.asarray(task.agent.keyframes["rest"].qpos)
        joints = task.agent.robot.active_joints_map
        names = [*task.agent.controller.controllers["arm"].config.joint_names,
                 "torso_lift_joint"]
        targets = {name: float(rest[int(joints[name].active_index[0])]) for name in names}
        planner.planner.update_from_simulation()
        # The selected hand withdrawal must clear the entire rest trajectory,
        # not just the handle. Extend that same straight retreat in bounded
        # increments if the countertop still obstructs the fold.
        attempts = 4 if from_handle else 3 if from_apple else 1
        for attempt in range(attempts):
            result = common.plan_joints(env, planner, task, targets,
                                       label="fold to canonical rest", tries=1,
                                       who=WHO, line_only=True)
            if not stopped(result):
                return result
            result = curved_rest(env, planner, task, targets, who=WHO)
            if not stopped(result) or attempt == attempts - 1:
                return result
            tcp = task.agent.tcp.pose[0].sp
            clearance = sapien.Pose(
                tcp.p - (0.08 if from_apple else 0.04) * tcp.to_transformation_matrix()[:3, 2], tcp.q)
            if from_apple:
                log("extend open-hand apple clearance before rest", goal=clearance.p.tolist())
                result = execute_straight(planner, straight_plan(planner, clearance))
            else:
                result = move("extend handle withdrawal for rest clearance", clearance,
                              noisy=False, contact=True, contact_stretch=1)
            if stopped(result):
                return result
            planner.planner.update_from_simulation()

    def drawer_stroke(drawer, opening):
        amount = float(array(task.drawer_open_amounts())[0, drawer])
        # Preserve the camera-up jaw assignment for both closing and opening.
        flip = False
        grasp, reach = bar_poses(task, drawer, amount, flip=flip, opening=opening)
        bottom_pull = opening and grasp.p[2] < 0.25
        # Physically close the empty fingers for the free approach past adjacent
        # fronts, then open at the standoff before entering the selected handle.
        result = (close_gripper(t=6) if bottom_pull
                  else open_gripper(t=6))
        if stopped(result):
            return result, False
        planner.planner.update_from_simulation()
        # Keep the open drawer as a planning obstacle during the free approach;
        # otherwise a feasible arm path can sweep the bar before the grasp.
        low_handle = grasp.p[2] < 0.5
        middle_handle = 0.5 <= grasp.p[2] < 0.65
        if (not opening and not low_handle) or middle_handle:
            # Start at the already reachable upper handle, and reserve enough
            # wrist roll for the later apple grasp before committing the arm.
            planner._drawer_initial_wrist_budget = not opening
            if middle_handle:
                approach = grasp.to_transformation_matrix()[:3, 2]
                reach = sapien.Pose(grasp.p - 0.25 * approach, grasp.q)
            reach = noise.pose("approach drawer", reach)
            log("approach drawer", goal=reach.p.tolist())
            result = approach_for_contact(
                reach, grasp, 120, torso_height=(0.10 if middle_handle else
                                                0.20 if not opening else None),
                contact_context=lambda: common.contact_stroke(
                    planner, [DRAWER_FIXTURES[drawer] + DRAWER_ART_SUFFIX]))
            if stopped(result):
                return result, False
        direct_contact = not opening and drawer == 0
        if direct_contact:
            planner.set_grasp_branch(elbow=1, wrist=1)
            planner._drawer_initial_wrist_budget = False
            # Consume the former free-waypoint noise draw to preserve all later
            # paired noise draws. Contact itself remains at the measured handle.
            noise.pose("retired standoff noise draw (not executed)", reach)
            push = sapien.Pose(grasp.p + [0, max(0.,amount), 0], grasp.q)
            path = direct_bottom_contact(planner, task, grasp, push, drawer=drawer)
            if path is None:
                log("continuous bottom contact refused",
                    diagnostics=planner._bottom_contact_diagnostics,
                    path_refusals=planner._bottom_path_refusals)
                return -1, False
            log("continuous bottom contact", goal=grasp.p.tolist(),
                curve_family=path['curve_family'], named_progress=path['named_progress'],
                torso_height_m=path['contact_torso_m'], no_intermediate_pose=True)
            result = execute_straight(planner, path)
            if stopped(result):
                return result, False
            planner.planner.update_from_simulation()
        # Deliberate contact is allowed only with the selected drawer in the
        # planning model. All physical contacts remain enabled in the simulator.
        with common.contact_stroke(planner, [DRAWER_FIXTURES[drawer] + DRAWER_ART_SUFFIX]):
            if (opening or low_handle) and not middle_handle and not direct_contact:
                if low_handle:
                    # Lower outside the worktop before reaching under it. Check
                    # both legs before leaving the compact travel posture.
                    approach = grasp.to_transformation_matrix()[:3, 2]
                    reach = sapien.Pose(grasp.p - 0.25 * approach, grasp.q)
                    reach = noise.pose("approach drawer", reach)
                    torso = 0.01 if grasp.p[2] < 0.25 else 0.10
                    log("approach drawer", goal=reach.p.tolist(), standoff_m=0.25, torso_height_m=torso)
                    result = approach_for_contact(
                        reach, grasp, REACH_N_INIT, torso_height=torso)
                    if opening and isinstance(result, (int, np.integer)) and result == -1:
                        # Unfold the upper arm before lowering through the
                        # countertop edge. The direct diagonal joint line from
                        # the travel posture can sweep the sink at the bottom
                        # handle even when both endpoint poses are reachable.
                        result = common.plan_joints(
                            env, planner, task,
                            {"shoulder_lift_joint": 0.0, "elbow_flex_joint": 1.8,
                             "wrist_flex_joint": 1.7},
                            label="clear counter for low handle", who=WHO,
                            line_only=True, tries=1)
                        if stopped(result):
                            return result, False
                        result = approach_for_contact(
                            reach, grasp, 120, torso_height=torso)
                else:
                    # Upper handles also need a reachable straight final stroke;
                    # a free-space approach alone can exhaust the remaining reach.
                    result = approach_for_contact(reach, grasp, 80)
                if stopped(result):
                    return result, False
            if bottom_pull:
                result = open_gripper(t=6)
                if stopped(result):
                    return result, False
            grasp, _ = bar_poses(task, drawer, float(array(task.drawer_open_amounts())[0, drawer]),
                                 flip=flip, opening=opening)
            # Execute the full approach: first contact can be the edge of a pad,
            # before the handle is between the fingers. Once at a closed drawer,
            # bound the settling hold instead of pressing into it for seconds.
            previous_refine_limit = planner.max_refine_steps
            try:
                if opening:
                    planner.max_refine_steps = min(previous_refine_limit, 8)
                if not direct_contact:
                    result = move("grasp drawer handle", grasp, noisy=False, sync=False, contact=True,
                                  contact_stretch=1 if opening else 2,
                                  freeze_lift=bottom_pull)
            finally:
                planner.max_refine_steps = previous_refine_limit
            if stopped(result):
                return result, False
            planner._drawer_initial_wrist_budget = False
            result = close_gripper(t=6) if opening else open_gripper(t=6)
            if stopped(result):
                return result, False
            aperture = float(array(task.agent.robot.get_qpos())[0, -2:].sum())
            touching = planner.gripper_touching(task._drawer_arts[drawer])
            if not opening and not touching:
                # Re-aim from the measured state if compliant contact was lost
                # during settling. Stop at hand contact and keep fingers open.
                result = move("establish open-hand push contact", grasp,
                              noisy=False, sync=False, contact=True,
                              contact_stretch=1, stop_on_touch=task._drawer_arts[drawer])
                if stopped(result):
                    return result, False
                touching = planner.gripper_touching(task._drawer_arts[drawer])
            reliable_grip = opening and any(
                bool(array(task.agent.is_grasping(link)).any())
                for link in task._drawer_arts[drawer].get_links())
            log("handle grip" if opening else "drawer push contact",
                aperture_m=aperture, drawer_contact=touching,
                reliable_grip=bool(reliable_grip))
            # A palm contact can push; pulling requires opposing finger forces
            # on the selected drawer, not merely a nonzero finger aperture.
            if (opening and (aperture <= common.FINGER_EMPTY_M or not reliable_grip)
                    or not opening and not touching):
                return result, False
            # Contact can move the drawer. Use its remaining travel now, and
            # stop at closed rather than loading the grasp past the hard stop.
            amount = float(array(task.drawer_open_amounts())[0, drawer])
            if opening:
                # Pull along the drawer's slider with the arm. This brings the
                # hand nearer the body without reversing the whole mobile base.
                tcp = task.agent.tcp.pose[0].sp
                travel = max(0.0, cfg.open_success + 0.08 - amount)
                goal = sapien.Pose(tcp.p + [0, -travel, 0], tcp.q)
                # Avoid exhausting the wrist solely to pull beyond the success
                # threshold. Precheck shorter straight pulls before moving;
                # the task still judges the physical drawer after release.
                for margin in (0.08, 0.06, 0.04, 0.02):
                    travel = max(0.0, cfg.open_success + margin - amount)
                    candidate = sapien.Pose(tcp.p + [0, -travel, 0], tcp.q)
                    if (common.screw_plans(planner, candidate, disable_lift_joint=True)
                            or common.screw_plans(planner, candidate)):
                        goal = candidate
                        break
                previous_refine_limit = planner.max_refine_steps
                try:
                    planner.max_refine_steps = min(previous_refine_limit, 8)
                    result = move("pull drawer with arm", goal, noisy=False,
                                  sync=False, contact=True, contact_stretch=1)
                finally:
                    planner.max_refine_steps = previous_refine_limit
            else:
                # Extend the hand along the physical slider. The open palm
                # pushes through contact; the base and finger command stay still.
                tcp = task.agent.tcp.pose[0].sp
                travel = max(0., amount) if direct_contact else amount + .04
                goal = sapien.Pose(tcp.p + [0, travel, 0], tcp.q)
                log("close drawer with arm", max_travel_m=travel)
                result = (planner._guard.last_step if amount <= cfg.closed_tol else
                          execute_straight(planner, straight_plan(planner, goal),
                            stop_when=lambda: float(array(task.drawer_open_amounts())[0, drawer]) <= cfg.closed_tol))
                if direct_contact and result != -1 and not stopped(result):
                    result = wait_for_state("physical bottom closure",
                        lambda: float(array(task.drawer_open_amounts())[0, drawer]) <= cfg.closed_tol,
                        6, result)
            if stopped(result):
                return result, False
            amount = float(array(task.drawer_open_amounts())[0, drawer])
            ok = amount >= cfg.open_success if opening else amount <= cfg.closed_tol
            log("drawer stroke ended", drawer=drawer, opening=opening, open_amount=amount,
                reached=bool(ok), tcp=task.agent.tcp.pose[0].sp.p.tolist())
            if not ok or stopped(result):
                return result, False
            if opening:
                result = open_gripper(t=6)
            else:
                # Keep observing the handle until release and withdrawal finish.
                # Fully open, then withdraw far enough that the canonical rest
                # transition clears the countertop (6 cm leaves a collision). The
                # next primitive folds the arm with all drawer obstacles restored.
                result = open_gripper(t=6)
                if stopped(result):
                    return result, False
                tcp = task.agent.tcp.pose[0].sp
                withdraw_m = 0.24
                clearance = sapien.Pose(tcp.p - withdraw_m * tcp.to_transformation_matrix()[:3, 2], tcp.q)
                # The diagonal exit includes lateral motion, so move the arm;
                # a forward-only base cannot execute that Cartesian translation.
                result = move("clear handle with arm", clearance,
                              noisy=False, sync=False, contact=True, contact_stretch=1)
                if stopped(result):
                    return result, False
                amount = float(array(task.drawer_open_amounts())[0, drawer])
                if amount > cfg.closed_tol or bool(array(task.sequence_violated).item()):
                    log("drawer reopened during withdrawal", open_amount=amount)
                    return result, False
        planner.planner.update_from_simulation()
        return result, not stopped(result)

    # Show the whole column before erasing the cue; gaze does not reveal the answer.
    result = gaze(lambda: array(task.handle_home)[0].mean(0))
    if stopped(result):
        return result
    home = task._robot_start_np[0]
    extension = float(array(task.drawer_open_amounts())[0, target])
    dock_offset = 0.10 if array(task.handle_home)[0, target, 2] < 0.5 else -0.10
    # Keep the sampled initial heading while the hand is in its initial pose.
    # The contact approach plans from this measured pose; floor travel follows
    # the fixed stow below, after the first drawer has been closed.
    initial_forward = task.agent.base_link.pose[0].sp.to_transformation_matrix()[:3, 0]
    initial_yaw = math.atan2(float(initial_forward[1]), float(initial_forward[0]))
    result = drive("approach column dock", [home[0], home[1] + dock_offset - extension],
                   yaw=initial_yaw, already_at_station=dock_offset < 0)
    if stopped(result):
        return result
    result, closed = drawer_stroke(target, False)
    if stopped(result) or not closed:
        return result
    result = stow(from_handle=True)
    if stopped(result):
        return result
    dock = array(task.apple_dock)[0]
    planner.track_target(lambda: task.apple.pose[0].sp.p)
    # Leave clearance to unfold above the counter before approaching the apple.
    result = drive("travel to apple station", [dock[0], dock[1] + 0.10], float(dock[2]))
    if stopped(result):
        return result
    result = gaze(lambda: task.apple.pose[0].sp.p)
    if stopped(result):
        return result
    apple_p = task.apple.pose[0].sp.p
    base = task.agent.base_link.pose[0].sp
    if abs(float(apple_p[0] - base.p[0])) > 0.04:
        result = drive("align with apple", [apple_p[0], base.p[1]])
        if stopped(result):
            return result
    result = open_gripper()
    if stopped(result):
        return result
    mesh = task.apple.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        raise RuntimeError("The apple has no collision mesh")
    centre = np.asarray(mesh.bounding_box_oriented.center_mass)
    # A nearly spherical apple has an unstable OBB axis. Use the counter-facing
    # approach and pinch across it, keeping the wrist above the countertop.
    # Pinch below the crown: a shallow grasp can report opposing contacts
    # while the fingertips still slide off the apple as soon as it is lifted.
    centre[2] = float(mesh.bounds[1, 2]) - 0.022
    grasp = task.agent.build_grasp_pose(np.array([0., 2**-.5, -2**-.5]),
                                       np.array([1., 0., 0.]), centre)
    reach = sapien.Pose(grasp.p + [0, -0.06, 0.06], grasp.q)
    reach = noise.pose("apple approach", reach)
    planner.set_grasp_branch(elbow=1, wrist=1)
    result, held = common.try_grasp(env, planner, task, task.apple, grasp, reach,
        resync_before_grasp=True, close_on_contact=True, n_init_qpos=REACH_N_INIT,
        approach_by_line=True, approach_max_knots=160, grasp_max_knots=80, grasp_knot_draws=1)
    planner._grasp_branch = {}
    if stopped(result) or not held:
        return result
    common.hold_object_in_planner(env, planner, task, task.apple, True, who=WHO)
    tcp = task.agent.tcp.pose[0].sp
    result = move("lift apple", sapien.Pose(tcp.p + [0, 0, 0.07], tcp.q),
                  axes=(False, False, True), contact=True, contact_stretch=1)
    if stopped(result):
        return result
    if not bool(array(task.agent.is_grasping(task.apple)).item()):
        log("apple lost during lift")
        return result
    planner.track_target(lambda: task.plate.pose[0].sp.p)
    # Restore the published small forward alignment before the lateral transfer.
    base = task.agent.base_link.pose[0].sp
    face = base.to_transformation_matrix()[:3, 0]
    yaw = math.atan2(float(face[1]), float(face[0]))
    result = drive("step toward plate", base.p + 0.08 * face, yaw)
    if stopped(result):
        return result
    tcp = task.agent.tcp.pose[0].sp
    apple = task.apple.pose[0].sp
    hover = sapien.Pose(task.plate.pose[0].sp.p + [0, 0, 0.10], apple.q) * (tcp.inv() * apple).inv()
    placement_preview = (sapien.Pose(task.plate.pose[0].sp.p + [0, 0, 0.035], apple.q)
                         * (tcp.inv() * apple).inv())
    hover = noise.pose("transfer apple over plate", hover)
    planner.planner.update_from_simulation()
    with transfer_phase(planner, "carry apple over plate"):
        log("transfer apple over plate", goal=hover.p.tolist())
        result = planner.approach_for_contact(
            hover, placement_preview, 160,
            contact_context=lambda: common.touchable(planner, "plate"))
    planner.planner.update_from_simulation()
    if stopped(result):
        return result
    if not bool(array(task.agent.is_grasping(task.apple)).item()):
        log("apple lost during transfer")
        return result
    # Finish the noisy transfer above the plate, then lower in a clean straight
    # stroke. Dropping from the transfer hover can bounce the apple off the plate
    # into the open hand. Its resting origin is about 2.7 cm above the plate's;
    # a 3.5 cm placement target leaves only a short drop after opening.
    tcp = task.agent.tcp.pose[0].sp
    apple = task.apple.pose[0].sp
    placement = (sapien.Pose(task.plate.pose[0].sp.p + [0, 0, 0.035], apple.q)
                 * (tcp.inv() * apple).inv())
    planner.planner.update_from_simulation()
    with common.touchable(planner, "plate"):
        result = move("lower apple onto plate", placement, noisy=False, sync=False,
                      contact=True, contact_stretch=1)
    if stopped(result):
        return result
    # Enforce settling before release. The prior six-step wait ignored failure
    # of its stop condition. Pose differences measure the actual held-object
    # motion rather than the contact solver's instantaneous velocity field.
    settle = ReleaseSettling(array(task.apple.pose.raw_pose)[0], task.control_timestep)
    settle_started = int(task.elapsed_steps.item())
    def settled_for_release():
        arm_velocity = array(task.agent.robot.get_qvel())[0, [5, 7, 8, 9, 10, 11, 12]]
        ready = settle.update(array(task.apple.pose.raw_pose)[0], arm_velocity)
        # Leave on a complete held-action pair, so opening needs no unmeasured
        # padding step after the stable window.
        return ready and int(task.elapsed_steps.item()) % 2 == 0
    result = close_gripper(t=20, stop_when=settled_for_release)
    log("settle before apple release", reached=settle.ready,
        waited_steps=int(task.elapsed_steps.item()) - settle_started,
        stable_steps=settle.stable_steps, linear_speed=settle.linear_speed,
        angular_speed=settle.angular_speed, arm_speed=settle.arm_speed,
        solver_linear_speed=float(np.linalg.norm(array(task.apple.linear_velocity)[0])))
    if stopped(result):
        return result
    if not settle.ready:
        return common.fail(env, WHO, "apple or arm did not settle before release")
    result = open_gripper(t=6)
    common.hold_object_in_planner(env, planner, task, task.apple, False, who=WHO)
    if stopped(result):
        return result
    tcp = task.agent.tcp.pose[0].sp
    planner.planner.update_from_simulation()
    # The former horizontal joint-line retract swept a finger through the
    # released apple (7900002/7900017). Exit vertically first, with a straight
    # finger-axis retreat as the only fallback. Keep the apple an obstacle.
    away = -tcp.to_transformation_matrix()[:3, 2]
    candidates = [("lift open hand from apple", np.array([0., 0., 0.10])),
                  ("withdraw open hand along fingers", 0.14 * away),
                  ("withdraw open hand along fingers", 0.18 * away)]
    result = -1
    for label, offset in candidates:
        clearance = sapien.Pose(tcp.p + offset, tcp.q)
        path = straight_plan(planner, clearance)
        if path is not None:
            log(label, goal=clearance.p.tolist(), knots=len(path["position"]))
            result = execute_straight(planner, path)
            break
    if stopped(result):
        return result
    result = wait_for_state("wait for apple to settle",
                            lambda: bool(array(task.apple_done).item()), 35, result)
    if stopped(result) or not bool(array(task.apple_done).item()):
        return result
    if bool(array(task.apple_retention_violated).item()):
        return common.fail(env, WHO, "apple left the plate after release")
    result = stow(from_apple=True)
    if stopped(result):
        return result
    # Approach the left-hand dock from the open aisle. A direct westward
    # arrival sweeps the folded hand into the left wall before the final turn.
    # This nearer corner keeps clearance without the old deep southern detour.
    home = task._robot_start_np[0]
    aisle = np.array([home[0] + 0.30, home[1] - 0.70])
    dock_offset = 0.10 if array(task.handle_home)[0, reopen, 2] < 0.5 else -0.10
    dock = np.array([home[0], home[1] + dock_offset])
    # Observe the whole column; head aiming does not encode the remembered drawer.
    planner.track_target(lambda: array(task.handle_home)[0].mean(0))
    aisle = noise.point("return via open aisle", [*aisle, 0.], (True, True, False))
    dock = noise.point("return to drawer column", [*dock, 0.], (True, True, False))
    yaw = 2 * math.atan2(float(home[6]), float(home[3]))
    from planners.oracle.smooth_base import drive_rounded_route
    result = drive_rounded_route(env, planner, task, aisle, dock, yaw)
    if stopped(result):
        return result
    result = gaze(lambda: array(task.handle_home)[0].mean(0))
    if stopped(result):
        return result
    result, opened = drawer_stroke(reopen, True)
    if stopped(result) or not opened:
        return result
    return wait_for_state("wait for final drawer hold",
                          lambda: bool(array(task.evaluate()["success"]).item()),
                          cfg.hold_steps + 10, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--blind", action="store_true")
    args = parser.parse_args()
    env = gym.make("MikasaSameDrawer-v0", scene_idx=0, sim_backend="cpu",
                   obs_mode="state", control_mode="pd_joint_pos",
                   sim_config={"control_freq":20, "sim_freq":100})
    try:
        result = solve(env, args.seed, blind=args.blind)
        print("RESULT", result if isinstance(result, int) else result[-1])
    finally:
        env.close()


if __name__ == "__main__":
    main()
