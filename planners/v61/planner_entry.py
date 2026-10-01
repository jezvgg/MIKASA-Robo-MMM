"""planning(): the v6.1 whole-body takeitback-tray episode (approach, grasp, carry, place)."""
from __future__ import annotations

from typing import cast

import numpy as np
from mani_skill.agents.robots import Fetch
from my_scenes.my_robocasa_takeitback_tray import MyRoboCasaSceneTakeItBackTray
from robots.fetch.extand import FetchMotionPlanningSapienSolver

from . import manip
from .config import APPROACH_ATTEMPTS, RETRY_KEEP_ARM_DIST, RETRY_KEEP_ARM_YAW, SETTLE_AFTER_RELEASE, GRIPPER_OPEN, STAND_NOISE
from .gaze import base_xyyaw, install_gaze
from .servo import wrap
from .stands import find_stand
from .whole_body import drive_concurrent


def _noise(seed) -> np.ndarray:
    rng = np.random.default_rng(1_000_003 + int(seed if seed is not None else 0))
    return rng.uniform(-STAND_NOISE, STAND_NOISE, size=2)


def planning(env, seed, debug=False, vis=None, info=False) -> bool:
    """Run one episode; True when the cup ends on the tray (the env's own success verdict).

    Whatever the outcome, an `episode_trace` event carries the 10 Hz base trace and the servo log.
    """
    holder: dict = {}
    try:
        return _episode(env, seed, debug, vis, info, holder)
    finally:
        if "trace" in holder:
            env.log_event("episode_trace", "v61 base trace (10 Hz) and servo log", base_trace=[[round(float(v), 4) for v in p] for p in holder["trace"][::2]],
                          servo_log=holder["planner"].v61_servo_log)


def _episode(env, seed, debug, vis, info, holder) -> bool:
    unwenv: MyRoboCasaSceneTakeItBackTray = env.unwrapped
    env.reset(seed=seed, options={"reconfigure": True})
    agent: Fetch = cast(Fetch, unwenv.agent)
    planner = FetchMotionPlanningSapienSolver(env, base_pose=agent.robot.pose.sp, vis=bool(vis),
                                              print_env_info=info, debug=debug)
    planner.control_mode = "pd_joint_pos"
    env.track_object(unwenv.cup, "cup")
    env.track_object(unwenv.tray, "tray")
    env.track_object(agent.base_link, "robot_base")
    env.track_object(agent.tcp, "robot_tcp")
    gaze = {"target": unwenv.cup}
    trace = install_gaze(planner, agent, lambda: gaze["target"].pose.p[0].cpu().numpy())
    planner.v61_servo_log = []
    holder.update(trace=trace, planner=planner)
    planner.gripper_state = GRIPPER_OPEN

    failed: list = []
    ok = False
    arm_now = None          # arm command left by a failed grasp attempt; None when the arm is tucked
    last_pose = None
    for attempt in range(APPROACH_ATTEMPTS):
        planner.v61_attempt = attempt
        stand = (find_stand(planner, unwenv, agent, _noise(seed), exclude=tuple(failed)) if arm_now is None else
                 find_stand(planner, unwenv, agent, _noise(seed), exclude=tuple(failed), from_arm=arm_now))
        if stand is None:
            env.log_event("error", "v6: no stand pose with IK for pregrasp, grasp and place")
            return False
        env.log_event("waypoint", "v6 approach: base, torso and arm move together",
                      start=base_xyyaw(agent).tolist(), stand=stand.pose.tolist(), torso_pre=stand.torso_pre)
        failed.append(stand.pose[:2].copy())
        keep = (arm_now is not None and last_pose is not None and float(np.hypot(*(stand.pose[:2] - last_pose[:2]))) < RETRY_KEEP_ARM_DIST
                and abs(wrap(float(stand.pose[2] - last_pose[2]))) < RETRY_KEEP_ARM_YAW)
        last_pose = stand.pose
        if not drive_concurrent(planner, agent, stand.pose, stand.arm_pre, stand.torso_pre, keep_arm=keep):
            env.log_event("error", "v6: approach did not arrive", base=base_xyyaw(agent).tolist(), reason=getattr(planner, "v6_reason", ""))
            continue
        planner.idle_steps(t=4)
        planner.planner.update_from_simulation()
        env.log_event("waypoint_complete", "v6 approach done", base=base_xyyaw(agent).tolist(), steps=len(trace))
        ok = manip.grasp_cup(planner, unwenv, agent, stand.grasp_pose, stand.pregrasp_pose)
        env.log_event("waypoint_complete" if ok else "error", "v6 grasp", grasped=ok, reason=getattr(unwenv, "reason", ""))
        if ok:
            break
        arm_now = agent.controller.controllers["arm"].qpos[0].cpu().numpy().astype(np.float64)
    if not ok:
        return False
    gaze["target"] = unwenv.tray
    ok = manip.carry_over_tray(planner, env, unwenv, stand.place_pose)
    env.log_event("waypoint_complete" if ok else "error", "v6 carry over tray", ok=ok)
    if not ok:
        return False
    ok = manip.lower_and_release(planner, env, unwenv, agent)
    planner.idle_steps(t=SETTLE_AFTER_RELEASE)
    manip.settle_cup(planner, unwenv)            # the verdict needs a static cup
    success = bool(unwenv.evaluate()["success"].item())
    env.log_event("waypoint_complete" if ok else "error", "v6 release", released=ok, task_success=success,
                  base_trace=[[round(float(v), 4) for v in p] for p in trace[::2]])
    return ok and success
