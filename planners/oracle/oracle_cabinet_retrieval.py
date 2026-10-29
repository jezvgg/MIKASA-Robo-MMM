"""CabinetRetrieval-only loaded door-arc stages."""
from __future__ import annotations

import numpy as np

from planners.oracle import oracle_common as common

FINGER_EMPTY_M = 0.008


def pull_hinge_arc(env, planner, task, articulation_name, joint_name, *, target_rad,
                   who: str = "oracle", v_handle: float = 0.05, max_steps: int = 400,
                   stall_window: int = 50, stall_eps: float = 0.01, anchor=None,
                   open_dir: int = 1):
    """Drive the gripped handle along its hinge arc until target, slip, or stall."""
    if int(open_dir) not in (1, -1):
        raise ValueError(f"pull hinge arc: open_dir must be +1 or -1, got {open_dir!r}")
    open_dir = int(open_dir)
    if anchor is None:
        try:
            anchor = common.hinge_anchor(task, articulation_name, joint_name)
        except (ValueError, KeyError, IndexError) as exc:
            return common.fail(env, who, f"pull hinge arc: anchor unreadable ({exc})")
    anchor_xy, sense = anchor

    fingers = common._np(task.agent.robot.get_qpos()).reshape(-1)[-2:]
    if float(fingers.sum()) <= FINGER_EMPTY_M:
        return common.fail(env, who, "pull hinge arc: nothing between the pads",
                           fingers=[round(float(f), 4) for f in fingers])

    try:
        articulation = task.scene.articulations[articulation_name]
        names = [joint.name for joint in articulation.get_active_joints()]
        index = names.index(joint_name)
    except (KeyError, ValueError) as exc:
        return common.fail(env, who, f"pull hinge arc: joint unresolvable ({exc})")

    def read_rad() -> float:
        return float(common._np(articulation.get_qpos()).reshape(-1)[index])

    start_rad = read_rad()
    if open_dir * float(target_rad) <= open_dir * start_rad:
        return common.fail(env, who, "pull hinge arc: target not ahead of the start",
                           target_rad=round(float(target_rad), 3),
                           start_rad=round(start_rad, 3), open_dir=open_dir)

    common.say(env, who, "pull hinge arc", target_rad=round(float(target_rad), 3),
               anchor=[round(float(v), 3) for v in anchor_xy], sense=sense,
               open_dir=open_dir, start_rad=round(start_rad, 3))

    history: list[float] = []
    aperture: list[float] = []
    reason = ["planner-stop"]

    def stop() -> bool:
        theta = read_rad()
        history.append(theta)
        if open_dir * theta >= open_dir * float(target_rad):
            reason[0] = "target"
            return True
        current_fingers = common._np(task.agent.robot.get_qpos()).reshape(-1)[-2:]
        aperture.append(float(current_fingers.sum()))
        if float(current_fingers.sum()) <= FINGER_EMPTY_M:
            reason[0] = "slipped"
            return True
        if (len(history) >= stall_window
                and open_dir * (history[-1] - history[-stall_window]) < stall_eps):
            reason[0] = "stalled"
            return True
        return False

    result = planner.follow_arc(anchor_xy, sense * open_dir, v_handle=v_handle,
                                max_steps=max_steps, stop_when=stop)
    if result == -1:
        return common.fail(env, who, "pull hinge arc: no step was taken")
    if common.stopped_by_horizon(planner):
        return result
    if reason[0] == "planner-stop" and len(history) >= max_steps:
        reason[0] = "max-steps"
    common.say(
        env, who, "pull hinge arc done", rad=round(read_rad(), 3), why=reason[0],
        **({"door_tail": [round(v, 3) for v in history[-30:]],
            "aperture_tail": [round(v, 4) for v in aperture[-30:]]}
           if reason[0] in ("slipped", "stalled") else {}),
    )
    return result
