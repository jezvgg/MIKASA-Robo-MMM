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
from utils.mikasa.seeding import seed_everything
from utils.mikasa.waypoint_noise import WaypointNoise

WHO = "same_drawer_planner"
BAR_STANDOFF = 0.22
REACH_N_INIT = 40


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def bar_poses(task, drawer, amount, *, opening=False):
    home = array(task.handle_home)[0, drawer].astype(float)
    centre = home + [0, -float(amount), 0]
    angle = math.radians(20 if drawer == 3 else 40)
    approaching = np.array([0., math.cos(angle), -math.sin(angle)])
    closing = np.array([0., math.sin(angle), math.cos(angle)])
    if opening:
        # Leave room behind the finger pads for the drawer front, then enter
        # along the fingers' approach axis instead of sweeping across the bar.
        centre -= 0.01 * approaching
    grasp = task.agent.build_grasp_pose(approaching, closing, centre)
    reach = centre - 0.14 * approaching if opening else home + [0, -BAR_STANDOFF, 0.08]
    return grasp, sapien.Pose(reach, grasp.q)


def solve(env, seed=None, debug=False, vis=False, blind=False, *,
          waypoint_noise_seed=None, waypoint_noise_m=0.005,
          planner_factory=common.default_planner_factory):
    with common.planning_budget(4.0):
        return _solve(env, seed, debug, vis, blind, waypoint_noise_seed,
                      waypoint_noise_m, planner_factory)


def _solve(env, seed, debug, vis, blind, noise_seed, noise_m, planner_factory):
    env.reset(seed=seed)
    seed_everything(seed)
    task = env.unwrapped
    assert task.control_mode == "pd_joint_pos", task.control_mode
    cfg = task.cfg
    planner = planner_factory(env, debug, vis)
    log = lambda stage, **kw: common.say(env, WHO, stage, **kw)
    noise = WaypointNoise((0 if seed is None else seed) + 200003 if noise_seed is None
                          else noise_seed, noise_m, log)
    target = int(array(task.target_drawer).item())
    reopen = int(np.random.default_rng((0 if seed is None else seed) + 900019).choice(cfg.drawer_choices)) if blind else target
    log("remember open drawer", target=target, reopen=reopen, blind=bool(blind))

    def stopped(result):
        return isinstance(result, (int, np.integer)) and result == -1 or common.stopped_by_horizon(planner)

    def move(label, pose, *, noisy=True, axes=(True, True, True), sync=True, contact=False):
        if noisy:
            pose = noise.pose(label, pose, axes)
        log(label, goal=pose.p.tolist())
        if sync:
            planner.planner.update_from_simulation()
        fixed_lift = False
        if contact:
            # Contact must follow a Cartesian line, not a joint-space/RRT detour.
            # Freeze the lift first: a lowered torso otherwise jams the Jacobian
            # solver at its lower limit despite a feasible arm-only stroke.
            fixed_lift = common.screw_plans(planner, pose, disable_lift_joint=True)
            if not fixed_lift and not common.screw_plans(planner, pose):
                return common.fail(env, WHO, "no straight contact path", waypoint=label)
        return planner.static_manipulation(pose, disable_lift_joint=fixed_lift,
            n_init_qpos=REACH_N_INIT, max_knots=160, knot_draws=1, knot_refuse=True,
            by_line=not contact, stretch=2 if contact else 1)

    def drive(label, point, yaw=math.pi/2):
        p = noise.point(label, [float(point[0]), float(point[1]), 0.], (True, True, False))
        log(label, position=p.tolist())
        return planner.drive_base(target_pos=p, target_view_vec=[math.cos(yaw), math.sin(yaw), 0.],
                                  freeze_arm=True)

    def gaze(point, tilt):
        local = (task.agent.base_link.pose[0].sp.inv() * sapien.Pose(point)).p
        pan = float(np.clip(np.arctan2(local[1], local[0]), -0.6, 0.6))
        return planner.hold_head(pan=pan, tilt=tilt, t=15, ramp=10)

    def stow(ahead):
        result = planner.close_gripper()
        if stopped(result):
            return result
        tcp = task.agent.tcp.pose[0].sp
        p = (task.agent.base_link.pose[0].sp * sapien.Pose([ahead, 0, 0])).p.copy()
        p[2] = 1.10
        result = move("stow empty hand", sapien.Pose(p, tcp.q))
        if not (isinstance(result, (int, np.integer)) and result == -1):
            return result
        if common.stopped_by_horizon(planner):
            return result
        # The empty hand need not keep its previous grasp orientation. Make
        # room away from the counter before folding to a known travel posture.
        base = task.agent.base_link.pose[0].sp
        point = base.p - 0.25 * base.to_transformation_matrix()[:3, 0]
        point = noise.point("room to fold arm", point, (True, True, False))
        log("back off to fold arm", position=point.tolist())
        result = planner.move_base_forward(point, freeze_arm=True)
        if stopped(result):
            return result
        planner.planner.update_from_simulation()
        rest = np.asarray(task.agent.keyframes["rest"].qpos)
        joints = task.agent.robot.active_joints_map
        names = task.agent.controller.controllers["arm"].config.joint_names
        targets = {name: float(rest[int(joints[name].active_index[0])]) for name in names}
        targets.update(torso_lift_joint=0.20, wrist_flex_joint=1.7)
        return common.plan_joints(env, planner, task, targets, label="fold for travel",
                                  tries=1, who=WHO)

    def drawer_stroke(drawer, opening):
        amount = float(array(task.drawer_open_amounts())[0, drawer])
        grasp, reach = bar_poses(task, drawer, amount, opening=opening)
        result = planner.change_gripper_state(gripper_state=0.4, t=10) if opening else planner.close_gripper()
        if stopped(result):
            return result, False
        planner.planner.update_from_simulation()
        # Deliberate contact is allowed only with the selected drawer in the
        # planning model. All physical contacts remain enabled in the simulator.
        with common.contact_stroke(planner, [DRAWER_FIXTURES[drawer] + DRAWER_ART_SUFFIX]):
            result = move("approach drawer", reach, sync=False)
            if stopped(result):
                return result, False
            if opening:
                result = move("grasp drawer handle", grasp, noisy=False, sync=False, contact=True)
                if stopped(result):
                    return result, False
                result = planner.close_gripper(t=12)
                if stopped(result):
                    return result, False
                aperture = float(array(task.agent.robot.get_qpos())[0, -2:].sum())
                log("handle grip", aperture_m=aperture)
                if aperture <= common.FINGER_EMPTY_M:
                    return result, False
                end = grasp.p + [0., -(cfg.open_success + 0.08 - amount), 0.]
            else:
                # Descend just in front of the handle, then push along the slide.
                result = move("lower fist in front of handle",
                              sapien.Pose(grasp.p + [0, -0.035, 0], grasp.q),
                              noisy=False, sync=False)
                if stopped(result):
                    return result, False
                end = grasp.p + [0., amount + 0.01, 0.]
            if opening:
                # Roll straight back while keeping the grasp. This avoids an arm
                # joint limit turning a drawer pull into a curved hand trajectory.
                base = task.agent.base_link.pose[0].sp
                forward = base.to_transformation_matrix()[:3, 0]
                point = base.p - (cfg.open_success + 0.08 - amount) * forward
                log("pull drawer with base", position=point.tolist())
                result = planner.move_base_forward(point, freeze_arm=True)
                end = task.agent.tcp.pose[0].sp.p.copy()
            else:
                result = move("push drawer", sapien.Pose(end, grasp.q),
                              noisy=False, sync=False)
            if stopped(result):
                return result, False
            result = planner.idle_steps(t=10)
            amount = float(array(task.drawer_open_amounts())[0, drawer])
            ok = amount >= cfg.open_success if opening else amount <= cfg.closed_tol
            log("drawer stroke ended", drawer=drawer, opening=opening, open_amount=amount,
                reached=bool(ok), tcp=task.agent.tcp.pose[0].sp.p.tolist())
            if not ok or stopped(result):
                return result, False
            if opening:
                # The requested motion is complete. Release and let the drawer
                # settle; an unnecessary arm detour can hook the bar again.
                result = planner.open_gripper(t=12, ramp=8)
            else:
                back = end + [0., -BAR_STANDOFF, 0.]
                result = move("withdraw from drawer", sapien.Pose(back, grasp.q), sync=False)
        planner.planner.update_from_simulation()
        return result, not stopped(result)

    # Show the whole column before erasing the cue; gaze does not reveal the answer.
    result = gaze(array(task.handle_home)[0].mean(0), 0.65)
    if stopped(result):
        return result
    home = task._robot_start_np[0]
    result = drive("approach column dock", [home[0], home[1] - 0.10])
    if stopped(result):
        return result
    result, closed = drawer_stroke(target, False)
    if stopped(result) or not closed:
        return result
    result = stow(0.60)
    if stopped(result):
        return result
    dock = array(task.apple_dock)[0]
    result = drive("travel to apple station", [dock[0], dock[1] + 0.10], float(dock[2]))
    if stopped(result):
        return result
    result = gaze((task.apple.pose[0].sp.p + task.plate.pose[0].sp.p) / 2, 0.45)
    if stopped(result):
        return result
    apple_p = task.apple.pose[0].sp.p
    base = task.agent.base_link.pose[0].sp
    if abs(float(apple_p[0] - base.p[0])) > 0.04:
        result = drive("align with apple", [apple_p[0], base.p[1]])
        if stopped(result):
            return result
    result = planner.open_gripper()
    if stopped(result):
        return result
    mesh = task.apple.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        raise RuntimeError("The apple has no collision mesh")
    centre = np.asarray(mesh.bounding_box_oriented.center_mass)
    # A nearly spherical apple has an unstable OBB axis. Use the counter-facing
    # approach and pinch across it, keeping the wrist above the countertop.
    centre[2] = float(mesh.bounds[1, 2]) - 0.012
    grasp = task.agent.build_grasp_pose(np.array([0., 2**-.5, -2**-.5]),
                                       np.array([1., 0., 0.]), centre)
    reach = sapien.Pose(grasp.p + [0, -0.06, 0.06], grasp.q)
    reach = noise.pose("apple approach", reach)
    result, held = common.try_grasp(env, planner, task, task.apple, grasp, reach,
        resync_before_grasp=True, close_on_contact=True, n_init_qpos=REACH_N_INIT,
        approach_by_line=True, approach_max_knots=160, grasp_max_knots=80, grasp_knot_draws=1)
    if stopped(result) or not held:
        return result
    common.hold_object_in_planner(env, planner, task, task.apple, True, who=WHO)
    tcp = task.agent.tcp.pose[0].sp
    result = move("lift apple", sapien.Pose(tcp.p + [0, 0, 0.07], tcp.q), axes=(False, False, True))
    if stopped(result):
        return result
    if not bool(array(task.agent.is_grasping(task.apple)).item()):
        log("apple lost during lift")
        return result
    # A short forward adjustment avoids sweeping the extended hand through the
    # upper cabinets during a turn. Transfer sideways with the arm over the counter.
    base = task.agent.base_link.pose[0].sp
    yaw = 2 * math.atan2(float(base.q[3]), float(base.q[0]))
    face = np.array([math.cos(yaw), math.sin(yaw), 0.])
    result = drive("step toward plate", base.p + 0.08 * face, yaw)
    if stopped(result):
        return result
    tcp = task.agent.tcp.pose[0].sp
    apple = task.apple.pose[0].sp
    hover = sapien.Pose(task.plate.pose[0].sp.p + [0, 0, 0.10], apple.q) * (tcp.inv() * apple).inv()
    result = move("transfer apple over plate", hover)
    if stopped(result):
        return result
    if not bool(array(task.agent.is_grasping(task.apple)).item()):
        log("apple lost during transfer")
        return result
    tcp = task.agent.tcp.pose[0].sp
    apple = task.apple.pose[0].sp
    target_pose = sapien.Pose(task.plate.pose[0].sp.p + [0, 0, 0.055], apple.q)
    with common.touchable(planner, "plate"):
        result = move("lower apple over plate", target_pose * (tcp.inv() * apple).inv(),
                      axes=(True, True, False), sync=False)
    if stopped(result):
        return result
    result = planner.open_gripper(t=16, ramp=12)
    common.hold_object_in_planner(env, planner, task, task.apple, False, who=WHO)
    if stopped(result):
        return result
    tcp = task.agent.tcp.pose[0].sp
    result = move("withdraw from apple", sapien.Pose(tcp.p - 0.18 * face, tcp.q))
    if stopped(result):
        return result
    result = planner.idle_steps(t=35)
    if stopped(result) or not bool(array(task.apple_done).item()):
        return result
    result = stow(0.45)
    if stopped(result):
        return result
    result = drive("return via open floor", [1.0, -2.3])
    if stopped(result):
        return result
    home = task._robot_start_np[0]
    yaw = 2 * math.atan2(float(home[6]), float(home[3]))
    result = drive("return to drawer column", [home[0], home[1] - 0.10], yaw)
    if stopped(result):
        return result
    result = gaze(array(task.handle_home)[0].mean(0), 0.65)
    if stopped(result):
        return result
    result, opened = drawer_stroke(reopen, True)
    if stopped(result) or not opened:
        return result
    return planner.idle_steps(t=cfg.hold_steps + 10)


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
