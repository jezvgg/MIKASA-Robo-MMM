"""The Fetch drive stack exactly as the recorded TakeItBack and Veggie episodes ran it.

This is the recorded behaviour, not a copy of the recorded file: `robots/fetch/extand.py`
carries the same class with newer hardening, and only the five pieces below differ from
commit `4119aa4` -- the collection run whose sha256 the datasets' provenance pins.
Inherited pieces whose difference is a new optional argument with a default that keeps the
old behaviour (`drive_straight(hold_pose=..., v_floor=..., v_end=...)`,
`follow_forward_path_w_refinement(body_hold=...)`, the noise attributes in `__init__`) are
deliberately not overridden.
"""

from __future__ import annotations

from robots.fetch.extand import *  # noqa: F401,F403  (the public names the tests use)
from robots.fetch.extand import (
    HeldBodies,
    PLANNING_TIME,
    RRT_BASE_MASK,
    RRT_RANGE,
    _MASK_UNSET,
    _report_rrt_pose,
    _rrt_base_defaults,
    mplib,
    new_contacts,
    np,
    sapien,
    sweep_yaw,
    to_sapien_pose,
    yaw_path,
)
from robots.fetch.extand import FetchMotionPlanningSapienSolver as _RecordedLineage


class FetchMotionPlanningSapienSolver(_RecordedLineage):
    """The solver the recorded episodes drove, byte for byte where it differs."""

    def move_to_pose_with_RRTConnect(
        self,
        pose: sapien.Pose,
        dry_run: bool = False,
        refine_steps: int = 0,
        mask=_MASK_UNSET,
        n_init_qpos=20,
        base_free: bool = False,
        disable_lift_joint: bool = False,
    ):
        """RRTConnect to `pose`, with the base masked out of the reach by default.

        See `RRT_BASE_MASK`: `base_free=True` is the old behaviour, and an explicit `mask=`
        still wins over both.
        """
        mask, fixed = _rrt_base_defaults(mask, base_free)
        if disable_lift_joint:
            mask = list(RRT_BASE_MASK if mask is None else mask)
            mask[3] = True
            if fixed is not None:
                fixed = sorted(set(fixed) | {3})
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = mplib.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_pose(
            pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            # use_point_cloud=self.use_point_cloud,
            wrt_world=True,
            verbose=True,
            planning_time=PLANNING_TIME,
            rrt_range=RRT_RANGE,
            simplify=True,
            mask=mask,
            n_init_qpos=n_init_qpos,
            fixed_joint_indices=fixed,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        _report_rrt_pose(self, result, mask)
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def _compose(self, arm_target, body_target, base_action):
        """One env action from ABSOLUTE arm and body targets, in the env's control mode.

        Every primitive of this class computes what it wants the joints AT — the next
        knot of a plan, the measured pose to hold, a head target — and hands it here.
        In `pd_joint_pos` that is the action. In `pd_joint_delta_pos` the action is
        `(target - measured) / step`, clipped to [-1, 1]: holding is a zero, a knot
        within one step of the arm is reached exactly, a farther one is approached at
        the controller's rate (the caller's refinement loop keeps re-issuing it). The
        measured pose is read here, at the step, never cached (the delta controller
        anchors to the pose of the step it is applied on). The absolute form is kept
        for the tape (`start_tape`), whatever the mode.

        Args:
            arm_target: (7,) absolute arm joint targets, in the arm controller's order.
            body_target: (3,) absolute [head_pan, head_tilt, torso_lift].
            base_action: (2,) normalized base velocities [forward, yaw].

        Returns:
            the (13,) action for `env.step`.

        Example:
            >>> a = planner._compose(arm_q, np.array([0.0, 0.0, torso]), np.zeros(2))  # doctest: +SKIP
        """
        arm_target = np.asarray(arm_target, dtype=np.float64).reshape(-1)
        body_target = np.asarray(body_target, dtype=np.float64).reshape(-1)
        base_action = np.asarray(base_action, dtype=np.float64).reshape(-1).copy()
        base_action[np.abs(base_action) < 1e-6] = 0.0
        self._last_abs = np.hstack([arm_target, self.gripper_state, body_target, base_action])
        return self._from_abs(self._last_abs)

    def _from_abs(self, vec):
        """The action for the env's control mode from the ABSOLUTE form
        `[arm targets, gripper, body targets, base velocities]` (a tape entry)."""
        vec = np.asarray(vec, dtype=np.float64).reshape(-1).copy()
        # The base slots are normalized velocities in every mode, and the planned base
        # speed can exceed the controller's 1 m/s (mplib's TOPP on the base joints:
        # 1.16 on CabinetSearch seed 2100). ManiSkill clips them on the way in; clipping
        # here makes the RECORDED action the executed one (the dataset check reads
        # `|a| <= 1`), nothing else changes.
        vec[-2:] = np.clip(vec[-2:], -1.0, 1.0)
        vec[-2:][np.abs(vec[-2:]) < 1e-6] = 0.0
        mode = self.control_mode
        if mode == "pd_joint_pos":
            return vec
        if mode == "pd_joint_delta_pos":
            arm_ctrl = self.env_agent.controller.controllers["arm"]
            body_ctrl = self.env_agent.controller.controllers["body"]
            n_arm = len(arm_ctrl.config.joint_names)
            n_body = len(body_ctrl.config.joint_names)
            arm_target, grip = vec[:n_arm], vec[n_arm:n_arm + 1]
            body_target, base = vec[n_arm + 1:n_arm + 1 + n_body], vec[n_arm + 1 + n_body:]
            arm_q = arm_ctrl.qpos[0].cpu().numpy().astype(np.float64)
            body_q = body_ctrl.qpos[0].cpu().numpy().astype(np.float64)
            arm = np.clip((arm_target - arm_q) / self._delta_step(arm_ctrl), -1.0, 1.0)
            body = np.clip((body_target - body_q) / self._delta_step(body_ctrl), -1.0, 1.0)
            return np.hstack([arm, grip, body, base])
        raise NotImplementedError(f"{type(self).__name__} composes actions for "
                                  f"{self.COMPOSE_MODES}, not {mode!r}")

    def _hold_targets(self, head_zero: bool = True):
        """The measured arm pose and body pose, as ABSOLUTE targets to hold; the head
        at zero when `head_zero` (the drives' convention since the fork: the head is
        parked while the base moves)."""
        arm = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy().astype(np.float64)
        body = self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy().astype(np.float64)
        if head_zero:
            body[0] = body[1] = 0.0
        return arm, body

    def _plan_base_yaw(self, angle: float) -> dict:
        """The yaw-only path for a turn of `angle`, or a `status` naming the obstacle.

        Timing from the planner's limits for the yaw joint (`joint_vel_limits[2]`,
        `joint_acc_limits[2]` — 0.9 rad/s, 0.9 rad/s² as constructed); rate stored in
        the base controller's normalized channel units. Collision: the current full
        qpos with the yaw substituted, the planning world's robot↔env and self checks
        at `sweep_samples(arc)` poses per candidate arc, with whatever is held drawn
        where physics has it — `HeldBodies` (`base_yaw.py`) both re-syncs the planning
        world from the simulator and undoes mplib 0.2.1's missing base pose; without
        the sync the residual turn below, planned after `follow_rotation` has already
        swung the base, would sweep the arc against a cup metres from the hand.

        Pairs already in contact at the start pose (the held cup grazing a counter, a
        self-touch the SRDF does not list) are subtracted: they are not this turn's
        doing, and left in they would refuse every turn the robot ever asks for.
        """
        planner = self.planner
        world = planner.planning_world
        art = world.get_planned_articulations()[0]
        # Sync first — `HeldBodies` calls `planner.update_from_simulation()` before it
        # reads a single pose (why: its docstring; attachments survive the sync). The
        # base pose is the planning articulation's own: the identity when the world
        # folds the robot's pose into the root joints (K53, `SapienPlanningWorldV2`),
        # in which case HeldBodies' frame correction is a no-op and only its sync
        # matters; mplib's base pose otherwise.
        held = HeldBodies(world, art.get_base_pose(), planner.update_from_simulation)
        qpos = self.robot.get_qpos().cpu().numpy()[0].astype(np.float64)
        yaw_index = list(planner.move_group_joint_indices).index(self.YAW_JOINT_INDEX)

        def pairs_at(offset: float) -> set:
            q = qpos.copy()
            q[self.YAW_JOINT_INDEX] += offset
            art.set_qpos(planner.fold_qpos(q), True)
            held.place()
            pairs = list(world.check_robot_collision()) + list(world.check_self_collision())
            return {f"{c.link_name1}<->{c.link_name2}" for c in pairs}

        def colliding_at(offset: float):
            return new_contacts(pairs_at(offset), baseline)

        try:
            baseline = pairs_at(0.0)  # the robot as it stands: not the turn's doing
            chosen, why = sweep_yaw(angle, colliding_at)
        finally:
            art.set_qpos(planner.fold_qpos(qpos), True)
            held.restore()
        if chosen is None:
            if baseline:
                why += f" (already touching before the turn, ignored: {', '.join(sorted(baseline)[:2])})"
            self._report("rotate_base_z", short_way=round(float(angle), 3), chosen=None, refused=why)
            return {"status": why}

        base_controller = self.env_agent.controller.controllers["base"]
        rate_scale = (
            float(base_controller.config.upper[1]) if base_controller.config.normalize_action else 1.0
        )
        result = yaw_path(
            qpos[planner.move_group_joint_indices],
            yaw_index,
            chosen,
            v_max=float(planner.joint_vel_limits[yaw_index]),
            a_max=float(planner.joint_acc_limits[yaw_index]),
            dt=self.base_env.control_timestep,
            rate_scale=rate_scale,
        )
        result["short_way"] = float(angle)
        self._report(
            "rotate_base_z",
            short_way=round(float(angle), 3),
            chosen=round(float(chosen), 3),
            knots=int(result["position"].shape[0]),
            dur=round(float(result["duration"]), 2),
            held=held.names,
        )
        return result
