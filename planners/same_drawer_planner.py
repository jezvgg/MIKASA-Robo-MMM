"""Action-only SameDrawer oracle: close, apple transfer, return, remembered pull.

The drawer answer is read once at reset. Blind evaluation replaces only the final
choice with an independent uniform guess. Every motion is an IK/RRT waypoint or a
base velocity command; fixture joints and robot state are never written here.

The bottom entry is one scene-checked curve with no intermediate stop. Fixed
progress coefficients were fitted once to the successful control free-space
approach (drawer-final-001 native seed 8000000, steps 7..104). Torso and shoulder
lift are projected to monotone cubic progress; elbow is linear. Runtime inputs are
only measured current joints, actual-handle IK endpoint, and scene geometry — no
lookup of seed-specific endpoint positions.

DrawerPathPlanner reserves wrist travel during the first upper-drawer contact
without touching physical joint limits or measured roll history.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os

import gymnasium as gym
import mplib
from mplib.pymp.collision_detection import AllowedCollision, fcl
import numpy as np
import sapien

import my_scenes  # noqa: F401
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from my_scenes.same_drawer import DRAWER_ART_SUFFIX, DRAWER_FIXTURES
from planners.oracle import oracle_common as common
from planners.oracle import oracle_same_drawer as same_drawer
from planners.oracle.paths import (
    ReleaseSettling,
    RollPathPlanner,
    curved_rest,
    dense_samples,
    drive_rounded_route,
    execute_straight,
    straight_plan,
)
from robots.fetch.utils import unwrap_toward
from utils.mikasa.execution_noise import configure_execution_noise, transfer_phase
from utils.mikasa.seeding import seed_everything
from utils.mikasa.waypoint_noise import WaypointNoise


WHO = "same_drawer_planner"
REACH_N_INIT = 40
# Medoids of the two arm families among the 29 measured apple reaches of
# profile 17 (drawer-final-004). They seed IK only after random restarts found
# no bounded solution; branch, roll window, collision and path checks are
# unchanged. The same starts serve every scene: no seed-to-pose lookup.
APPLE_REACH_STARTS = (
    ("apple_reach_pan_negative", dict(
        shoulder_pan_joint=-0.900531, shoulder_lift_joint=-0.646882,
        upperarm_roll_joint=2.045675, elbow_flex_joint=1.568071,
        forearm_roll_joint=-1.490697, wrist_flex_joint=1.461369,
        wrist_roll_joint=0.879655)),
    ("apple_reach_pan_positive", dict(
        shoulder_pan_joint=0.582838, shoulder_lift_joint=-1.075451,
        upperarm_roll_joint=-1.255287, elbow_flex_joint=1.944571,
        forearm_roll_joint=1.441230, wrist_flex_joint=0.783066,
        wrist_roll_joint=-1.470573)),
)

# Standoff arm medoids of the successful closing approaches of each drawer in
# profile 17 (drawer-final-004): 7, 10 and 10 episodes. Offered only after every
# random-restart goal fails the contact-and-push lookahead of that approach.
DRAWER_CLOSE_STARTS = {
    1: (("close_drawer1_standoff", dict(
        shoulder_pan_joint=-1.552237, shoulder_lift_joint=0.182986,
        upperarm_roll_joint=0.933785, elbow_flex_joint=1.767808,
        forearm_roll_joint=-0.340111, wrist_flex_joint=0.332939,
        wrist_roll_joint=-1.778397)),),
    2: (("close_drawer2_standoff", dict(
        shoulder_pan_joint=-1.505170, shoulder_lift_joint=0.173107,
        upperarm_roll_joint=1.640604, elbow_flex_joint=1.706350,
        forearm_roll_joint=-0.906887, wrist_flex_joint=0.983043,
        wrist_roll_joint=-0.998324)),),
    3: (("close_drawer3_standoff", dict(
        shoulder_pan_joint=-1.377302, shoulder_lift_joint=0.911281,
        upperarm_roll_joint=2.005231, elbow_flex_joint=1.536953,
        forearm_roll_joint=-1.068235, wrist_flex_joint=1.170071,
        wrist_roll_joint=-1.310662)),),
}

# Bezier progress control values. q(s)=q0+(q1-q0)*B(0,c1,c2,1;s).
# Pan's necessary excursion provides clearance around the counter; it is reported,
# not described as monotone. The other overshoots are bounded by URDF screening.
PROGRESS = {
    'torso_lift_joint': (.24356265113691844, 1.0),
    'shoulder_pan_joint': (.7235411583118544, 3.2484893124565657),
    'shoulder_lift_joint': (.3711874490030574, 1.0),
    'upperarm_roll_joint': (.28753258941446025, .47784108658036134),
    'elbow_flex_joint': (1/3, 2/3),
    'forearm_roll_joint': (.4352533305624711, 1.5414777951332241),
    'wrist_flex_joint': (.4142209566172384, 1.3592145964309263),
    'wrist_roll_joint': (.38528795568215, 1.1441944997639484),
}


class DrawerPathPlanner(RollPathPlanner):
    """Keep the first push from spending the apple grasp's wrist-roll range.

    This narrows planning candidates and checks complete paths. Physical joint
    limits and measured roll history stay unchanged, including after release.
    """

    wrist_budget = 1.45

    def accepts(self, qpos, *, move_group=False):
        if not super().accepts(qpos, move_group=move_group):
            return False
        if not getattr(self._owner, "_drawer_initial_wrist_budget", False):
            return True
        index = self._owner._roll_indices[2]
        if move_group:
            index = list(self.move_group_joint_indices).index(index)
        return bool(np.all(np.abs(np.atleast_2d(qpos)[:, index]) <= self.wrist_budget))

    @contextmanager
    def _limits(self):
        with super()._limits():
            if getattr(self._owner, "_drawer_initial_wrist_budget", False):
                limits = self._planner.joint_limits.copy()
                index = self._owner._roll_indices[2]
                limits[index, 0] = max(limits[index, 0], -self.wrist_budget)
                limits[index, 1] = min(limits[index, 1], self.wrist_budget)
                self._planner.joint_limits = limits
            yield


@contextmanager
def hand_contact(planner, task, drawer, *, allowed=True):
    """Permit intentional hand contact only; keep the forearm and all fixtures."""
    with _kept_hand_entries(planner, drawer):
        acm,pairs=_hand_pairs(planner, drawer)
        for a,b in pairs:
            acm.set_entry(a,b,allowed)
        yield


def moving_drawer_push(planner, task, current, goal, drawer):
    """Preview the slider at its proposed position, without changing physics."""
    p = planner.planner
    selected = DRAWER_FIXTURES[drawer] + DRAWER_ART_SUFFIX
    # Re-adding the removed model drops its ACM entries; keep the caller's.
    with _kept_hand_entries(planner, drawer), common.contact_stroke(planner, [selected]):
        path = straight_plan(planner, goal, current=current)
    if path is None:
        return None
    world = p.planning_world
    model = next(world.get_articulation(n) for n in world.get_articulation_names() if selected in n)
    original = model.get_qpos().copy()
    pin = p.pinocchio_model
    ee = p.link_name_2_idx[p.move_group]
    pin.compute_forward_kinematics(p.fold_qpos(current))
    y0 = float(pin.get_link_pose(ee).p[1])
    try:
        # The drawer moves with the pushing hand, not as a frozen open obstacle.
        with hand_contact(planner, task, drawer):
            for knot in dense_samples(path['position']):
                q = current.copy(); q[p.move_group_joint_indices] = knot
                folded = p.fold_qpos(q)
                pin.compute_forward_kinematics(folded)
                travel = max(0.,float(pin.get_link_pose(ee).p[1])-y0)
                drawer_q = original.copy(); drawer_q[0] = min(0.,float(original[0])+travel)
                model.set_qpos(drawer_q, True)
                p.robot.set_qpos(folded, True)
                if world.is_state_colliding():
                    planner._straight_failures.append('moving drawer collision: '+str(
                        [(c.link_name1,c.link_name2) for c in world.check_collision()]))
                    return None
    finally:
        model.set_qpos(original, True)
    return path


def direct_bottom_contact(planner, task, grasp, push, *, drawer):
    """One continuous arm/torso curve to contact, with no standoff fallback."""
    p = planner.planner
    current = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
    torso = int(task.agent.robot.active_joints_map['torso_lift_joint'].active_index[0])
    reasons = []; planner._bottom_contact_diagnostics = reasons
    planner._bottom_path_refusals = []
    with hand_contact(planner, task, drawer):
        for height in (.01, .10, None):
            initial = p.fold_qpos(current).copy()
            if height is not None:
                initial[torso] = height
            status, found = p.IK(
                p._transform_goal_to_wrt_base(mplib.Pose(grasp.p, grasp.q)), initial,
                [True]*3 + [height is not None] + [False]*11, n_init_qpos=40)
            reasons.append(dict(stage='contact_ik', torso=height, status=status))
            if status != 'Success':
                continue
            goals = [p.unfold_qpos(unwrap_toward(q, initial, p.joint_limits),
                                  ref_yaw=float(current[2])) for q in np.atleast_2d(found)]
            goals.sort(key=lambda q: float(np.linalg.norm(q[3:13] - current[3:13])))
            for goal in goals[:8]:
                goal[:3] = current[:3]
                reasons.append(dict(stage='endpoint', qpos=goal.tolist()))
                stroke = moving_drawer_push(planner, task, goal, push, drawer)
                if stroke is None:
                    reasons.append(dict(stage='push', torso=height,
                                        reason=list(planner._straight_failures)))
                    continue
                # Every contact is inspected geometrically by the curve helper;
                # disable the broad endpoint-only hand allowance for its sweep.
                with hand_contact(planner, task, drawer, allowed=False):
                    path = continuous_lower_curve(planner, task, current, goal, grasp, drawer)
                if path is None:
                    planner._bottom_path_refusals.extend(planner._continuous_lower_curve_failures)
                    continue
                if not p.accepts(np.vstack([path['position'], stroke['position']]), move_group=True):
                    reasons.append(dict(stage='complete_path',reason='joint_window'))
                    continue
                return path
    return None


def continuous_lower_curve(planner, task, current, goal, grasp, drawer):
    """Return one checked, retimed curve or None; does not execute physics.

    Check the subsequent moving-drawer push before executing this returned path.
    Use the existing scene/noise/roll-window policy; this helper relaxes none of
    its limits. It permits only measured low hand contact with the selected
    moving drawer wall/handle. No global allowed-collision entry is changed.
    """
    p = planner.planner
    moves = list(p.move_group_joint_indices)
    idx = lambda name: int(task.agent.robot.active_joints_map[name].active_index[0])
    alpha = np.linspace(0., 1., 241)
    full = np.repeat(current[None], len(alpha), axis=0)
    for name, (c1, c2) in PROGRESS.items():
        i = idx(name)
        progress = (3*(1-alpha)**2*alpha*c1
                    + 3*(1-alpha)*alpha**2*c2 + alpha**3)
        full[:, i] = current[i] + progress*(goal[i]-current[i])

    selected = DRAWER_FIXTURES[drawer] + DRAWER_ART_SUFFIX
    model = next(p.planning_world.get_articulation(name)
                 for name in p.planning_world.get_articulation_names()
                 if selected in name)
    selected_links = set(model.get_user_link_names())
    inner = next(link for link in task._drawer_arts[drawer].get_links()
                 if link.name == 'inner_box')
    upper_edge = float(get_component_mesh(inner._objs[0]).bounds[1, 2])
    request = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    limits = np.asarray(p.joint_limits)
    failures = []
    planner._continuous_lower_curve_failures = failures

    def checked(poses):
        # RollPathPlanner.accepts checks the collection windows, not every URDF
        # bound. Explicitly check every folded joint against the unchanged URDF.
        if not p.accepts(poses):
            failures.append(dict(reason='collection_joint_window'))
            return False
        previous_distance = None
        for knot, q in enumerate(dense_samples(poses)):
            folded = p.fold_qpos(q)
            if (np.any(folded < limits[:, 0]-1e-7)
                    or np.any(folded > limits[:, 1]+1e-7)):
                failures.append(dict(reason='URDF_joint_limit', knot=knot))
                return False
            p.pinocchio_model.compute_forward_kinematics(folded)
            tcp = np.asarray(p.pinocchio_model.get_link_pose(
                p.link_name_2_idx[p.move_group]).p)
            distance = float(np.linalg.norm(tcp-grasp.p))
            p.robot.set_qpos(folded, True)
            for collision in p.planning_world.check_collision(request):
                pair = (collision.link_name1, collision.link_name2)
                contacts = collision.res.get_contacts()
                hand = any('ds_fetch' in name and 'gripper' in name for name in pair)
                wall = any(name in selected_links and
                           ('inner_box' in name or 'handle' in name) for name in pair)
                # Contact is beneath the top edge, approached toward the actual
                # handle, with bounded geometric overlap. This rejects landing
                # on the drawer from above while allowing its low front surface.
                allowed = (hand and wall and bool(contacts)
                    and all(float(c.pos[2]) < upper_edge-.04
                            and float(c.penetration_depth) < .025 for c in contacts)
                    and (previous_distance is None
                         or distance <= previous_distance+1e-5))
                if not allowed:
                    failures.append(dict(reason='collision', knot=knot, pair=pair,
                                         tcp=tcp.tolist()))
                    return False
            previous_distance = distance
        return True

    if not checked(full):
        return None
    old_velocity = p.joint_vel_limits.copy()
    old_acceleration = p.joint_acc_limits.copy()
    torso_column = moves.index(idx('torso_lift_joint'))
    try:
        speed = old_velocity.copy()
        acceleration = old_acceleration.copy()
        speed[torso_column] = min(speed[torso_column], .06)
        acceleration[torso_column] = min(acceleration[torso_column], .12)
        p.joint_vel_limits = speed
        p.joint_acc_limits = acceleration
        times, pos, vel, acc, duration = p.TOPP(full[:, moves], task.control_timestep)
    except RuntimeError as error:
        failures.append(dict(reason='TOPP', error=str(error)))
        return None
    finally:
        p.joint_vel_limits = old_velocity
        p.joint_acc_limits = old_acceleration
    timed = np.repeat(current[None], len(pos), axis=0)
    timed[:, moves] = pos
    if not checked(timed):
        return None
    return dict(status='Success', time=times, position=pos, velocity=vel,
                acceleration=acc, duration=duration,
                curve_family='lower_handle_cubic_v1', named_progress=PROGRESS,
                contact_torso_m=float(goal[idx('torso_lift_joint')]),
                no_intermediate_pose=True)


def _hand_pairs(planner, drawer):
    world=planner.planner.planning_world
    hand=[n for n in planner.planner.robot.get_user_link_names() if 'gripper' in n]
    selected=DRAWER_FIXTURES[drawer]+DRAWER_ART_SUFFIX
    links=[n for name in world.get_articulation_names() if selected in name
           for n in world.get_articulation(name).get_user_link_names()]
    return world.get_allowed_collision_matrix(),[(a,b) for a in hand for b in links]


@contextmanager
def _kept_hand_entries(planner, drawer):
    """Restore hand/drawer ACM entries that removing the drawer model erases."""
    acm,pairs=_hand_pairs(planner, drawer)
    previous=[(a,b,acm.get_entry(a,b)) for a,b in pairs]
    try:yield
    finally:
        for a,b,value in previous:
            if value is None:acm.remove_entry(a,b)
            else:acm.set_entry(a,b,value==AllowedCollision.ALWAYS)


@contextmanager
def reference_ik_starts(planner, task, starts):
    joints = task.agent.robot.active_joints_map
    planner._ik_reference_arms = [
        (label, {int(joints[name].active_index[0]): value for name, value in arm.items()})
        for label, arm in starts]
    planner._ik_reference_used = []
    try:
        yield planner._ik_reference_used
    finally:
        planner._ik_reference_arms = None


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def bar_poses(task, drawer, amount, *, opening=False):
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
    # The same centred handle grasp is used in both directions. Leave space
    # for the drawer front behind the pads, and enter along the finger axis.
    centre -= 0.01 * approaching
    grasp = task.agent.build_grasp_pose(approaching, closing, centre)
    # Enter and leave along the same finger axis. The closing dock is farther
    # back by the drawer extension, so an open drawer does not crowd the arm.
    reach = centre - 0.14 * approaching
    return grasp, sapien.Pose(reach, grasp.q)


@common.planning_budget(4.0)
def solve(env, seed=None, debug=False, vis=False, blind=False, *,
          waypoint_noise_seed=None, waypoint_noise_m=0.005,
          planner_factory=common.collection_planner_factory,
          execution_noise_seed=None, action_noise=0.003, noise_hold=10):
    env.reset(seed=seed)
    seed_everything(seed)
    task = env.unwrapped
    assert task.control_mode == "pd_joint_pos", task.control_mode
    cfg = task.cfg
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
    noise = WaypointNoise((0 if seed is None else seed) + 200003 if waypoint_noise_seed is None
                          else waypoint_noise_seed, waypoint_noise_m, log)
    target = int(array(task.target_drawer).item())
    reopen = int(np.random.default_rng((0 if seed is None else seed) + 900019).choice(cfg.drawer_choices)) if blind else target
    log("remember open drawer", target=target, reopen=reopen, blind=bool(blind))

    def stopped(result):
        return isinstance(result, (int, np.integer)) and result == -1 or common.stopped_by_horizon(planner)

    def open_amount(drawer):
        return float(array(task.drawer_open_amounts())[0, drawer])

    def dock_offset(drawer):
        # Lower drawers sit back from the column dock; upper ones come forward.
        return 0.10 if array(task.handle_home)[0, drawer, 2] < 0.5 else -0.10

    def wait_for_state(label, condition, max_steps, result):
        waited = 0
        while not condition() and waited < max_steps:
            result = planner.idle_steps(t=1)
            waited += 1
            if stopped(result):
                break
        log(label, waited_steps=waited, reached=bool(condition()))
        return result

    def move(label, pose, *, noisy=True, axes=(True, True, True), sync=True, contact=True,
             stop_on_touch=None, stretch=1, contact_stretch=None):
        if noisy:
            pose = noise.pose(label, pose, axes)
        log(label, goal=pose.p.tolist())
        if sync:
            planner.planner.update_from_simulation()
        # Contact must follow a Cartesian line, not a joint-space/RRT detour.
        # Freeze the lift first: a lowered torso otherwise jams the Jacobian
        # solver at its lower limit despite a feasible arm-only stroke.
        fixed_lift = False
        if contact:
            fixed_lift = same_drawer.screw_plans(planner, pose, disable_lift_joint=True)
            if not fixed_lift and not same_drawer.screw_plans(planner, pose):
                return common.fail(env, WHO, "no straight contact path", waypoint=label)
        return planner.static_manipulation(pose, disable_lift_joint=fixed_lift,
            n_init_qpos=REACH_N_INIT, max_knots=160, knot_draws=1, knot_refuse=True,
            by_line=not contact,
            stretch=(contact_stretch if contact and contact_stretch is not None else stretch),
            stop_on_touch=stop_on_touch)

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

    def open_gripper():
        aperture = float(array(task.agent.robot.get_qpos())[0, -2:].sum())
        if planner.gripper_state > 0 and aperture >= 0.095:
            return planner._guard.last_step
        return planner.open_gripper(t=6)

    def close_gripper(*args, **kwargs):
        result = planner.close_gripper(*args, **kwargs)
        planner._grasp_branch = {}
        return result

    @contextmanager
    def refine_steps_capped(enabled=True):
        # A pad-edge first contact otherwise keeps refining into the handle.
        limit = planner.max_refine_steps
        try:
            if enabled:
                planner.max_refine_steps = min(limit, 8)
            yield
        finally:
            planner.max_refine_steps = limit

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
        attempts = 4 if from_handle else 3
        for attempt in range(attempts):
            result = common.plan_joints(env, planner, task, targets,
                                       label="fold to canonical rest", who=WHO)
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
                              noisy=False, stretch=1)
            if stopped(result):
                return result
            planner.planner.update_from_simulation()

    def drawer_stroke(drawer, opening):
        amount = open_amount(drawer)
        grasp, reach = bar_poses(task, drawer, amount, opening=opening)
        bottom_pull = opening and grasp.p[2] < 0.25
        # Physically close the empty fingers for the free approach past adjacent
        # fronts, then open at the standoff before entering the selected handle.
        result = close_gripper(t=6) if bottom_pull else open_gripper()
        if stopped(result):
            return result, False
        planner.planner.update_from_simulation()
        # Keep the open drawer as a planning obstacle during the free approach;
        # otherwise a feasible arm path can sweep the bar before the grasp.
        low_handle = grasp.p[2] < 0.5
        middle_handle = 0.5 <= grasp.p[2] < 0.65
        # The bottom drawer of a closing stroke is entered by its contact curve,
        # not by a free standoff approach.
        direct_contact = not opening and drawer == 0
        # A non-bottom closing push follows contact from the chosen approach;
        # pick an approach from which that push also plans (same goal as below).
        push_preview = (None if opening or drawer == 0 else
                        sapien.Pose(grasp.p + [0, amount + .04, 0], grasp.q))
        close_starts = () if opening else DRAWER_CLOSE_STARTS.get(drawer, ())
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
        contact_context = (hand_contact(planner, task, drawer) if direct_contact else
                           common.contact_stroke(planner, [DRAWER_FIXTURES[drawer] + DRAWER_ART_SUFFIX]))
        with contact_context:
            if not direct_contact:
                # The upper closing handle keeps the bar_poses reach; the low and
                # middle handles and every opening approach enter from 0.25 m out.
                if opening or low_handle or middle_handle:
                    approach = grasp.to_transformation_matrix()[:3, 2]
                    reach = sapien.Pose(grasp.p - 0.25 * approach, grasp.q)
                reach = noise.pose("approach drawer", reach)
                # Check a reachable straight final stroke before leaving the compact
                # travel posture: a free-space approach alone can exhaust the reach,
                # and a low handle has to be lowered outside the worktop.
                if middle_handle or (not opening and not low_handle):
                    torso, n_init = (0.10 if middle_handle else 0.20), 120
                elif low_handle:
                    torso, n_init = (0.01 if grasp.p[2] < 0.25 else 0.10), REACH_N_INIT
                else:
                    torso, n_init = None, 80
                # Reserve enough wrist roll for the later apple grasp before
                # committing the arm. The low closing handle is not budgeted: it is
                # lowered outside the worktop and keeps its own roll range (the
                # measured closing approaches of profile 17 do).
                budgeted = middle_handle or (not opening and not low_handle)
                planner._drawer_initial_wrist_budget = budgeted and not opening
                log("approach drawer", goal=reach.p.tolist(), torso_height_m=torso)
                with reference_ik_starts(planner, task, close_starts) as reference_used:
                    # The contact stroke is the upper closing approach's precondition;
                    # the low handle plans its reach clean and takes contact later, in
                    # `grasp drawer handle`.
                    result = approach_for_contact(
                        reach, grasp, n_init, torso_height=torso,
                        contact_context=None if low_handle else lambda: common.contact_stroke(
                            planner, [DRAWER_FIXTURES[drawer] + DRAWER_ART_SUFFIX]),
                        follow_through=push_preview)
                    if (not opening and low_handle and torso == 0.10
                            and isinstance(result, (int, np.integer)) and result == -1):
                        # At 0.10 m some lower-middle handles need wrist_flex below the
                        # grasp-branch margin at contact. Nothing has moved yet; plan the
                        # same approach at the bottom-handle torso height.
                        torso = 0.01
                        log("lower torso for closing approach", torso_height_m=torso)
                        result = approach_for_contact(reach, grasp, n_init, torso_height=torso,
                                                      follow_through=push_preview)
                    # Unfold the upper arm before lowering through the countertop
                    # edge. The direct diagonal joint line from the travel posture
                    # can sweep the sink at the bottom handle even when both endpoint
                    # poses are reachable.
                    if (opening and low_handle and isinstance(result, (int, np.integer))
                            and result == -1):
                        result = common.plan_joints(
                            env, planner, task,
                            {"shoulder_lift_joint": 0.0, "elbow_flex_joint": 1.8,
                             "wrist_flex_joint": 1.7},
                            label="clear counter for low handle", who=WHO,
                            )
                        if stopped(result):
                            return result, False
                        result = approach_for_contact(reach, grasp, 120, torso_height=torso)
                if reference_used:
                    log("drawer approach IK from fixed reference start", starts=list(reference_used))
                if stopped(result):
                    return result, False
            if bottom_pull:
                result = open_gripper()
                if stopped(result):
                    return result, False
            grasp, _ = bar_poses(task, drawer, open_amount(drawer), opening=opening)
            # Execute the full approach: first contact can be the edge of a pad,
            # before the handle is between the fingers. Once at a closed drawer,
            # bound the settling hold instead of pressing into it for seconds.
            with refine_steps_capped(opening):
                if not direct_contact:
                    result = move("grasp drawer handle", grasp, noisy=False, sync=False,
                                  stretch=1 if opening else 2)
            if stopped(result):
                return result, False
            planner._drawer_initial_wrist_budget = False
            result = close_gripper(t=6) if opening else open_gripper()
            if stopped(result):
                return result, False
            aperture = float(array(task.agent.robot.get_qpos())[0, -2:].sum())
            touching = planner.gripper_touching(task._drawer_arts[drawer])
            if not opening and not touching:
                # Re-aim from the measured state if compliant contact was lost
                # during settling. Stop at hand contact and keep fingers open.
                result = move("establish open-hand push contact", grasp,
                              noisy=False, sync=False,
                              stop_on_touch=task._drawer_arts[drawer])
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
            amount = open_amount(drawer)
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
                    if (same_drawer.screw_plans(planner, candidate, disable_lift_joint=True)
                            or same_drawer.screw_plans(planner, candidate)):
                        goal = candidate
                        break
                with refine_steps_capped():
                    result = move("pull drawer with arm", goal, noisy=False, sync=False)
            else:
                # Extend the hand along the physical slider. The open palm
                # pushes through contact; the base and finger command stay still.
                tcp = task.agent.tcp.pose[0].sp
                travel = max(0., amount) if direct_contact else amount + .04
                goal = sapien.Pose(tcp.p + [0, travel, 0], tcp.q)
                log("close drawer with arm", max_travel_m=travel)
                result = planner._guard.last_step
                if amount > cfg.closed_tol:
                    if direct_contact:
                        planner.planner.update_from_simulation()
                        current = array(task.agent.robot.get_qpos())[0].astype(float)
                        stroke = moving_drawer_push(planner, task, current, goal, drawer)
                        log("actual bottom push checked", accepted=stroke is not None,
                            remaining_travel_m=travel)
                    else:
                        stroke = straight_plan(planner, goal)
                    result = execute_straight(planner, stroke,
                        stop_when=lambda: open_amount(drawer) <= cfg.closed_tol)
                if direct_contact and result != -1 and not stopped(result):
                    result = wait_for_state("physical bottom closure",
                        lambda: open_amount(drawer) <= cfg.closed_tol, 6, result)
            if stopped(result):
                return result, False
            amount = open_amount(drawer)
            ok = amount >= cfg.open_success if opening else amount <= cfg.closed_tol
            log("drawer stroke ended", drawer=drawer, opening=opening, open_amount=amount,
                reached=bool(ok), tcp=task.agent.tcp.pose[0].sp.p.tolist())
            if not ok or stopped(result):
                return result, False
            if opening:
                result = open_gripper()
            else:
                # Keep observing the handle until release and withdrawal finish.
                # Fully open, then withdraw far enough that the canonical rest
                # transition clears the countertop (6 cm leaves a collision). The
                # next primitive folds the arm with all drawer obstacles restored.
                result = open_gripper()
                if stopped(result):
                    return result, False
                tcp = task.agent.tcp.pose[0].sp
                withdraw_m = 0.24
                clearance = sapien.Pose(tcp.p - withdraw_m * tcp.to_transformation_matrix()[:3, 2], tcp.q)
                # The diagonal exit includes lateral motion, so move the arm;
                # a forward-only base cannot execute that Cartesian translation.
                result = move("clear handle with arm", clearance,
                              noisy=False, sync=direct_contact)
                if stopped(result):
                    return result, False
                amount = open_amount(drawer)
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
    extension = open_amount(target)
    offset = dock_offset(target)
    # Keep the sampled initial heading while the hand is in its initial pose.
    # The contact approach plans from this measured pose; floor travel follows
    # the fixed stow below, after the first drawer has been closed.
    initial_forward = task.agent.base_link.pose[0].sp.to_transformation_matrix()[:3, 0]
    initial_yaw = math.atan2(float(initial_forward[1]), float(initial_forward[0]))
    result = drive("approach column dock", [home[0], home[1] + offset - extension],
                   yaw=initial_yaw, already_at_station=offset < 0)
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
    # The apple dock already holds the hand within the 4 cm alignment band; a
    # violation is reported rather than corrected by an extra base drive.
    x_offset = float(apple_p[0] - base.p[0])
    if abs(x_offset) > 0.04:
        log("apple off the dock alignment band", x_offset_m=x_offset)
        result = drive("align with apple", [apple_p[0], base.p[1]])
        if stopped(result):
            return result
    result = open_gripper()
    if stopped(result):
        return result
    mesh = task.apple.get_first_collision_mesh(to_world_frame=True)
    assert mesh is not None, "The apple has no collision mesh"
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
    with reference_ik_starts(planner, task, APPLE_REACH_STARTS) as reference_used:
        result, held = same_drawer.try_grasp(env, planner, task, task.apple, grasp, reach,
            resync_before_grasp=True, close_on_contact=True, n_init_qpos=REACH_N_INIT,
            approach_by_line=True, approach_max_knots=160, grasp_max_knots=80,
            grasp_knot_draws=1)
    if reference_used:
        log("apple reach IK from fixed reference start", starts=list(reference_used))
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
        # The standoff the first restart set finds can put the held apple through
        # the plate or the counter eight knots into the straight line (HF seeds
        # 8250001/25/43/103/122). Nothing has executed yet, so trying a higher
        # hover and then a wider restart set costs only planning, and a seed whose
        # first attempt plans sees exactly the motion it saw before.
        for attempt, (raise_m, restarts) in enumerate(
                ((0.0, 160), (0.05, 160), (0.10, 160), (0.0, 500))):
            if attempt:
                log("transfer hover refused; retrying",
                    raise_m=raise_m, ik_restarts=restarts)
                planner.planner.update_from_simulation()
            result = approach_for_contact(
                sapien.Pose(hover.p + [0, 0, raise_m], hover.q), placement_preview, restarts,
                contact_context=lambda: common.touchable(planner, "plate"))
            if result != -1:
                break
        if result == -1:
            return common.fail(env, WHO, "no transfer path to the plate")
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
    result = open_gripper()
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
    dock = np.array([home[0], home[1] + dock_offset(reopen)])
    # Observe the whole column; head aiming does not encode the remembered drawer.
    planner.track_target(lambda: array(task.handle_home)[0].mean(0))
    aisle = noise.point("return via open aisle", [*aisle, 0.], (True, True, False))
    dock = noise.point("return to drawer column", [*dock, 0.], (True, True, False))
    yaw = 2 * math.atan2(float(home[6]), float(home[3]))
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
    parser.add_argument("--curobo", action="store_true", help="propose IK goals from cuRobo instead of the mplib default (robots/fetch/IK_BACKENDS.md; needs MIKASA_CUROBO_SOCKET or MIKASA_CUROBO_PY)")
    parser.add_argument("--assisted-grasp", action=argparse.BooleanOptionalAction, default=True, help="weld the picked object to the hand by OmniGibson's rule (utils/mikasa/assisted_grasp.py; changes the physics)")
    args = parser.parse_args()
    if args.curobo:
        os.environ["MIKASA_IK_BACKEND"] = "curobo"
    env = gym.make("MikasaSameDrawer-v0", scene_idx=0, sim_backend="cpu",
                   obs_mode="state", control_mode="pd_joint_pos",
                   sim_config={"control_freq":20, "sim_freq":100},
                   assisted_grasp=args.assisted_grasp)
    try:
        result = solve(env, args.seed, blind=args.blind)
        print("RESULT", result if isinstance(result, int) else result[-1])
        ag = env.unwrapped._ag
        if ag is not None:
            kinds = [k for _, k, _ in ag.events]
            print(f"assisted_grasp establish={kinds.count('establish')} "
                  f"release={kinds.count('release')} events={ag.events}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
