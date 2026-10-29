"""DepthRecall root placement and shelf-stow stages."""
from __future__ import annotations

import os

import numpy as np
import sapien

from planners.oracle import oracle_common as common

REST_TORSO = 0.386
GRASP_TORSO_RUNGS = (0.0, 0.10, 0.20)
RISE_TORSO_RUNGS = (REST_TORSO, 0.34, 0.30)
STOW_WHO = "cabinet_stow_planner"
ENTRY_RISE = float(os.environ.get("MIKASA_STOW_ENTRY_RISE", "0.04"))
PRE_RISE_BACK = float(os.environ.get("MIKASA_STOW_PRE_RISE_BACK", "0.12"))
ENTRY_SETTLE_STEPS = int(os.environ.get("MIKASA_STOW_ENTRY_SETTLE", "6"))
PLACE_DROP = float(os.environ.get("MIKASA_STOW_PLACE_DROP", "0.008"))
RELEASE_LIFT_RUNGS = tuple(
    float(v) for v in os.environ.get("MIKASA_STOW_RELEASE_LIFT", "0.06,0.045,0.03").split(",")
)
TILT_WARN_DEG = 15.0
ENTRY_SAG_TOL = float(os.environ.get("MIKASA_STOW_ENTRY_SAG_TOL", "0.005"))
PLACE_PROBE_SLACK = (0.0, 0.0, -0.005)


def root_joints_for(task, x: float, y: float, yaw: float) -> np.ndarray:
    """Return root-joint values that place ``base_link`` at world ``(x, y, yaw)``."""
    from robots.fetch.curobo_client import stand_matrix

    q = np.asarray(task.agent.robot.get_qpos()[0].cpu().numpy(), dtype=np.float64)
    T_now = task.agent.base_link.pose[0].sp.to_transformation_matrix()
    J_now = stand_matrix(q[0], q[1], q[2])
    T_root = T_now @ np.linalg.inv(J_now)
    goal = stand_matrix(x, y, yaw)
    goal[2, 3] = T_now[2, 3]
    J = np.linalg.inv(T_root) @ goal
    th = float(np.arctan2(J[1, 0], J[0, 0]))
    th = q[2] + (th - q[2] + np.pi) % (2 * np.pi) - np.pi
    return np.array([J[0, 3], J[1, 3], th])


def place_pose_for(task, *, rise: float = 0.0, obj=None, target=None,
                   compensate_xy: bool = False):
    """Compute the TCP pose that places a held prop on a DepthRecall target."""
    obj = task.cup if obj is None else obj
    mesh = obj.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return None
    bottom = float(np.asarray(mesh.bounds)[0][2])
    target = (common._np(task.place_target).reshape(-1, 3)[0] if target is None
              else np.asarray(target, dtype=np.float64).reshape(3))
    tcp = task.agent.tcp.pose.sp
    drop = bottom - (float(target[2]) + PLACE_DROP)
    x, y = float(target[0]), float(target[1])
    if compensate_xy:
        off = common._np(obj.pose.p).reshape(-1)[:3] - np.asarray(tcp.p, dtype=np.float64).reshape(3)
        x -= float(off[0])
        y -= float(off[1])
    return sapien.Pose(p=[x, y, float(tcp.p[2]) - drop + float(rise)], q=tcp.q)


def _straight_move(env, planner, pose, *, stage: str, tries: int = 1, who: str = STOW_WHO):
    """`arm_move` restricted to the straight channels: the screw with the torso held."""
    return common.arm_move(
        env, planner, pose, who=who, stage=stage, tries=tries,
        disable_lift_joint=True, max_knots=1, knot_refuse=True,
    )


def rise_with_the_cup(env, planner, task, *, who: str = STOW_WHO):
    """The loaded torso rise: the rung ladder, then a retreat and the ladder again."""
    for attempt in ("as grasped", "after a retreat"):
        for torso in RISE_TORSO_RUNGS:
            result = common.plan_joints(
                env, planner, task, {"torso_lift_joint": torso},
                label=f"raise the torso to {torso}", who=who,
            )
            if result != -1 and common.stopped_by_horizon(planner):
                return result
            if result != -1:
                common.say(env, who, "torso up", torso=torso, how=attempt)
                planner.planner.update_from_simulation()
                return result
        if attempt != "as grasped":
            break
        tcp = task.agent.tcp.pose.sp
        back = sapien.Pose(p=[tcp.p[0], tcp.p[1] - PRE_RISE_BACK, tcp.p[2]], q=tcp.q)
        common.say(env, who, "the loaded rise refused; retreating first", back=PRE_RISE_BACK)
        moved = _straight_move(env, planner, back, stage="retreat before the rise",
                               tries=2, who=who)
        if moved != -1 and common.stopped_by_horizon(planner):
            return moved
        if moved == -1:
            return common.fail(env, who, "retreat before the rise")
        planner.planner.update_from_simulation()
    return common.fail(env, who, "raise the torso with the cup", rungs=list(RISE_TORSO_RUNGS))
