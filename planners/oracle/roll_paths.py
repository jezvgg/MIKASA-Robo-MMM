"""Collection motion limits, independent of the robot's physical joint limits."""

from contextlib import contextmanager
from functools import partial

import numpy as np


class RollPathPlanner:
    """Reject winding paths before execution; never rewrite simulator states.

    Python IK/screw limits narrow candidate generation. OMPL retains its original
    sampling bounds, so every returned path also needs the explicit check below.
    The check includes intermediate knots and the measured episode history.
    """

    margin = 0.05
    methods = {"IK", "plan_qpos_line", "plan_screw", "plan_qpos", "plan_pose"}

    def __init__(self, planner, owner):
        object.__setattr__(self, "_planner", planner)
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        if name in self.methods:
            return partial(self._plan, name)
        return getattr(self._planner, name)

    def __setattr__(self, name, value):
        setattr(self._planner, name, value)

    def accepts(self, qpos, *, move_group=False):
        full = np.atleast_2d(qpos)
        for index, sign in getattr(self._owner, "_grasp_branch", {}).items():
            column = list(self.move_group_joint_indices).index(index) if move_group else index
            if not np.all(sign * full[:, column] >= self.margin):
                return False
        indices = self._owner._roll_indices
        if move_group:
            indices = [list(self.move_group_joint_indices).index(i) for i in indices]
        roll = np.atleast_2d(qpos)[:, indices]
        if not np.isfinite(roll).all():
            return False
        # A joint line can report an empty success when the goal is already
        # reached. It adds no motion; the caller decides whether a step is needed.
        low, high = self._owner._roll_low, self._owner._roll_high
        if len(roll):
            low = np.minimum(low, roll.min(axis=0))
            high = np.maximum(high, roll.max(axis=0))
        return bool(
            np.all(low >= -np.pi + self.margin)
            and np.all(high <= np.pi - self.margin)
            and np.all(high - low <= np.pi - self.margin)
        )

    def ends_on_goal_branch(self, qpos, *, move_group=False):
        """Only the endpoint must carry the owner's `_goal_branch` signs (D3).

        Unlike `_grasp_branch`, the path may pass through the other branch: the
        transit posture itself has a positive elbow.
        """
        full = np.atleast_2d(qpos)
        if not len(full):
            return True
        for index, sign in getattr(self._owner, "_goal_branch", {}).items():
            column = list(self.move_group_joint_indices).index(index) if move_group else index
            if sign * full[-1, column] < self.margin:
                return False
        return True

    def _screw_on_goal_branch(self, goal, current, *args, **kwargs):
        """Retry a screw that ended on the wrong branch with shoulder_lift held.

        This is the solver's own `joint limit` unjam variant, which the published
        door grasps already take whenever their screw jams at shoulder_lift.
        """
        if kwargs.get("masked_joints") is None:
            return None
        joints = self._owner.env_agent.robot.active_joints_map
        held = np.array(kwargs["masked_joints"], dtype=bool).copy()
        held[int(joints["shoulder_lift_joint"].active_index[0])] = False
        result = self._planner.plan_screw(goal, current, *args, **{**kwargs, "masked_joints": held})
        if result.get("status") != "Success":
            return None
        if not (self.accepts(result["position"], move_group=True)
                and self.ends_on_goal_branch(result["position"], move_group=True)):
            return None
        report = getattr(self._owner, "_report", None)
        if callable(report):
            report("goal_branch", held="shoulder_lift_joint", knots=int(len(result["position"])))
        return result

    @contextmanager
    def _limits(self):
        previous = self._planner.joint_limits
        limits = previous.copy()
        idx = self._owner._roll_indices
        limits[idx, 0] = np.maximum(
            -np.pi + self.margin, self._owner._roll_high - np.pi + self.margin
        )
        limits[idx, 1] = np.minimum(
            np.pi - self.margin, self._owner._roll_low + np.pi - self.margin
        )
        for index, sign in getattr(self._owner, "_grasp_branch", {}).items():
            if sign > 0:
                limits[index, 0] = max(limits[index, 0], self.margin)
            else:
                limits[index, 1] = min(limits[index, 1], -self.margin)
        self._planner.joint_limits = limits
        try:
            yield
        finally:
            self._planner.joint_limits = previous

    def _plan(self, method, goal, current, *args, **kwargs):
        failure = "collection joint limit: no bounded path"
        # Prevent inherited fix_joint_limits from silently moving the start of a
        # proposed path to a different roll angle than the simulator's actual one.
        if not self.accepts(current):
            return (failure, None) if method == "IK" else {"status": failure}
        if method == "plan_qpos_line" and not (self.accepts(goal) and self.ends_on_goal_branch(goal)):
            return {"status": failure}
        if method == "plan_qpos":
            goal = [q for q in goal if self.accepts(q) and self.ends_on_goal_branch(q)]
            if not goal:
                return {"status": failure}
        if method == "IK":
            # IK random restarts sample the full robot, although the returned
            # path moves only the arm/base group. Keep the head and fingers at
            # their real values, including in subsequent contact previews.
            supplied = args[0] if args else kwargs.get("mask")
            mask = np.zeros(len(current), dtype=bool)
            if supplied is not None and len(supplied):
                mask[:] = supplied
            movable = np.zeros(len(current), dtype=bool)
            movable[self.move_group_joint_indices] = True
            mask |= ~movable
            if args:
                args = (mask, *args[1:])
            else:
                kwargs["mask"] = mask
        elif method == "plan_qpos_line":
            # Native joint-line collision checks update only the move group.
            # A previous hypothetical screw preview can leave its head/fingers
            # in the planning model. Restore the whole MODEL from the measured
            # start; do not change the simulator or restore omitted obstacles.
            self._planner.robot.set_qpos(self._planner.fold_qpos(current), True)
        attempts = 3 if method in {"plan_pose", "plan_qpos"} else 1
        with self._limits():
            for _ in range(attempts):
                result = getattr(self._planner, method)(goal, current, *args, **kwargs)
                if method == "IK":
                    status, candidates = result
                    if status != "Success":
                        return result
                    valid = [q for q in np.atleast_2d(candidates)
                             if self.accepts(q) and self.ends_on_goal_branch(q)]
                    if not valid:
                        return failure, None
                    return status, valid[0] if kwargs.get("return_closest") else valid
                if result.get("status") != "Success":
                    return result
                if self.accepts(result["position"], move_group=True):
                    if self.ends_on_goal_branch(result["position"], move_group=True):
                        return result
                    if method == "plan_screw":
                        held = self._screw_on_goal_branch(goal, current, *args, **kwargs)
                        return held if held is not None else {
                            "status": "collection goal branch: wrong elbow/wrist signs at the end"}
        return {"status": failure}
