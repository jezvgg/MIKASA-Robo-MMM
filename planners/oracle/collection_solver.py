"""Collection-specific action generation; the canonical robot stack is unchanged."""

import numpy as np

from robots.fetch.extand import FetchMotionPlanningSapienSolver


class CollectionMotionPlanner(FetchMotionPlanningSapienSolver):
    """Add C3 noise to executed path knots, with clean contact and refinement.

    The inherited path executor calls this planner's `_step` once per knot and
    then for refinement. The counter confines noise to the actual knot actions:
    it does not alter the planned waypoints or recorded actions after execution.
    Other planner motions and every gripper primitive remain outside this scope.
    """

    forward_navigation = False
    navigation_arrival_tolerance = 0.0

    def move_base_forward(self, new_base_pose, *args, **kwargs):
        """Keep the canonical floor path and bound its optional correction leg."""
        previous = getattr(self, "_navigation_leg", None)
        self._navigation_leg = [np.asarray(new_base_pose, dtype=float)[:2], 0]
        try:
            return super().move_base_forward(new_base_pose, *args, **kwargs)
        finally:
            self._navigation_leg = previous

    def follow_moving_forward(self, result, refine_steps=0):
        leg = getattr(self, "_navigation_leg", None)
        tolerance = self.navigation_arrival_tolerance
        if leg is None or tolerance <= 0:
            return super().follow_moving_forward(result, refine_steps)
        target, _ = leg
        leg[1] += 1
        count = len(result["position"])
        out = None
        # Run the canonical follower in complete 10 Hz action pairs. Arrival is
        # checked before another pair is executed, including the braking tail
        # of the first path, where TOPP can contain a small reverse segment.
        for start in range(0, count + refine_steps, 2):
            here = np.asarray(self.env_agent.base_link.pose.sp.p)[:2]
            if np.linalg.norm(target - here) <= tolerance:
                return self.idle_steps(t=2)
            indices = np.minimum(np.arange(start, min(start + 2, count + refine_steps)), count - 1)
            pair = dict(result, position=result["position"][indices],
                        velocity=result["velocity"][indices])
            out = super().follow_moving_forward(pair)
            if self.truncated:
                return out
        return out

    def drive_base(self, target_pos=None, target_view_vec=None, freeze_arm=False,
                   arrive_tol=None, reverse_ok=None):
        """Accept an opt-in arrival region before turning toward the waypoint."""
        if reverse_ok is None:
            reverse_ok = not self.forward_navigation
        if target_pos is not None and not self.truncated:
            here = np.asarray(self.env_agent.base_link.pose.sp.p)[:2]
            distance = np.linalg.norm(np.asarray(target_pos)[:2] - here)
            if self.navigation_arrival_tolerance > 0 and distance <= self.navigation_arrival_tolerance:
                if target_view_vec is None:
                    return self.idle_steps(t=2)
                # The final view still goes through the canonical collision-
                # checked turn. Avoid driving in a circle for a tiny correction.
                target_pos = None
        return super().drive_base(
            target_pos=target_pos, target_view_vec=target_view_vec,
            freeze_arm=freeze_arm, arrive_tol=arrive_tol, reverse_ok=reverse_ok,
        )

    supports_execution_noise = True
    execution_noise = None
    _path_remaining = 0
    _path_first = False

    def track_target(self, target):
        """Aim the standard cameras at the current search/manipulation target."""
        from planners.oracle.head_tracking import HeadTracker

        if getattr(self, "_head_tracker", None) is None:
            self._head_tracker = HeadTracker(
                self.env_agent, 2 * self.base_env.control_timestep
            )
        self._head_tracker.target = target

    def approach_for_contact(self, reach, grasp, n_init_qpos=40, *, torso_height=None,
                             contact_context=None, follow_through=None):
        """Choose an approach whose following straight contact stroke also plans.

        `follow_through` is an optional next straight stroke (a push after
        contact); it must also plan from the contact endpoint, in the same
        contact context, and the whole sequence must pass the roll window.
        """
        from contextlib import nullcontext
        from planners.oracle.straight_paths import straight_plan
        import mplib

        # The caller synchronizes before entering any contact-obstacle context.
        # Synchronizing here would try to restore the deliberately omitted drawer.
        p = self.planner
        cur = self.robot.get_qpos()[0].cpu().numpy().astype(float)
        folded = p.fold_qpos(cur)
        initial = folded.copy()
        if torso_height is not None:
            initial[3] = float(torso_height)
        target = mplib.Pose(reach.p, reach.q)
        status, goals = p.IK(
            p._transform_goal_to_wrt_base(target),
            initial,
            [True, True, True, torso_height is not None] + [False] * 11,
            n_init_qpos=n_init_qpos,
        )
        if status != "Success":
            return -1

        def contact_follows(goal):
            line = p.plan_qpos_line(
                goal,
                cur,
                time_step=self.base_env.control_timestep,
                ref_yaw=float(cur[2]),
            )
            if line.get("status") != "Success":
                return False
            for freeze_lift in (True,) if torso_height is not None else (True, False):
                # Keep the target as an obstacle on the free approach; only the
                # subsequent deliberate contact may omit it from planning.
                knots = [line["position"]]
                with nullcontext() if contact_context is None else contact_context():
                    contact = p.plan_screw(
                        mplib.Pose(grasp.p, grasp.q),
                        goal,
                        time_step=self.base_env.control_timestep,
                        masked_joints=~np.array(
                            [True, True, True, freeze_lift] + [False] * 11
                        ),
                        goal_tolerance=self.ARM_SCREW_GOAL_TOLERANCE,
                    )
                    if contact.get("status") == "Success":
                        knots.append(contact["position"])
                        if follow_through is not None:
                            end = np.array(goal, dtype=float)
                            end[p.move_group_joint_indices] = contact["position"][-1]
                            stroke = straight_plan(self, follow_through, current=end)
                            if stroke is None:
                                continue
                            knots.append(stroke["position"])
                if contact.get("status") == "Success" and p.accepts(
                    np.vstack(knots),
                    move_group=True,
                ):
                    return True
            return False

        result = self._line_to_ik_goals(
            target,
            goals,
            cur,
            folded,
            float(cur[2]),
            1,
            None,
            None,
            tag="contact_standoff",
            precheck=contact_follows,
        )
        if result is None and getattr(self, "_ik_reference_arms", None):
            # Every random-restart goal failed the next-stroke checks; offer the
            # caller's fixed reference starts to the same checks once.
            goals = p.reference_goals(
                p._transform_goal_to_wrt_base(target), initial,
                [True, True, True, torso_height is not None] + [False] * 11)
            if goals:
                result = self._line_to_ik_goals(
                    target, goals, cur, folded, float(cur[2]), 1, None, None,
                    tag="contact_standoff", precheck=contact_follows)
        return -1 if result is None else result

    def gripper_touching(self, actor, threshold=1e-6):
        """Use link identity for articulations; drawer link names are shared."""
        if not hasattr(actor, "get_links"):
            return super().gripper_touching(actor, threshold)
        hand = [
            self.env_agent.finger1_link,
            self.env_agent.finger2_link,
            self.env_agent.robot.links_map["gripper_link"],
        ]
        for link in actor.get_links():
            for finger in hand:
                impulse = self.base_env.scene.get_pairwise_contact_impulses(
                    finger, link
                )
                if float(np.linalg.norm(impulse[0].cpu().numpy())) > threshold:
                    return True
        return False

    def prefer_low_roll_ik(self):
        """Prefer low-roll IK goals and reject winding paths before execution."""
        from planners.oracle.roll_paths import RollPathPlanner

        names = ("upperarm_roll_joint", "forearm_roll_joint", "wrist_roll_joint")
        joints = self.env_agent.robot.active_joints_map
        self._roll_indices = [int(joints[name].active_index[0]) for name in names]
        roll = self.env_agent.robot.get_qpos()[0].cpu().numpy()[self._roll_indices]
        self._roll_low = roll.copy()
        self._roll_high = roll.copy()
        self.planner = RollPathPlanner(self.planner, self)

    def set_grasp_branch(self, *, elbow=1, wrist=1):
        """Constrain candidate and intermediate poses to one grasp branch."""
        if elbow not in (-1, 1) or wrist not in (-1, 1):
            raise ValueError("Grasp branch signs must be -1 or +1")
        joints = self.env_agent.robot.active_joints_map
        self._grasp_branch = {
            int(joints[name].active_index[0]): sign
            for name, sign in (("elbow_flex_joint", elbow), ("wrist_flex_joint", wrist))
        }


    def _line_to_ik_goals(
        self,
        target_tcp_pose,
        goal_qpos,
        cur,
        cur_f,
        ref_yaw,
        stretch,
        stretch_tail,
        stop_on_touch,
        tag,
        precheck=None,
    ):
        if not hasattr(self, "_roll_indices"):
            return super()._line_to_ik_goals(
                target_tcp_pose,
                goal_qpos,
                cur,
                cur_f,
                ref_yaw,
                stretch,
                stretch_tail,
                stop_on_touch,
                tag,
                precheck,
            )
        from robots.fetch.utils import unwrap_toward

        goals = [
            unwrap_toward(g, cur_f, self.planner.joint_limits)
            for g in np.atleast_2d(goal_qpos)
        ]
        current = np.asarray(cur_f)[self._roll_indices]

        def extent(goal):
            roll = goal[self._roll_indices]
            return np.maximum(self._roll_high, roll) - np.minimum(self._roll_low, roll)

        goals = [
            g
            for g in goals
            if np.all(np.abs(g[self._roll_indices]) <= np.pi - 0.02)
            and np.all(extent(g) <= np.pi - 0.02)
        ]
        goals.sort(
            key=lambda g: (
                float(np.max(extent(g))),
                float(np.sum(np.abs(g[self._roll_indices] - current))),
            )
        )
        # The inherited routine sorts its argument by all-joint distance. Passing
        # each candidate separately preserves the roll preference while retaining
        # its collision check and the caller's next-leg feasibility check.
        for goal in goals[:12]:
            result = super()._line_to_ik_goals(
                target_tcp_pose,
                [goal],
                cur,
                cur_f,
                ref_yaw,
                stretch,
                stretch_tail,
                stop_on_touch,
                tag,
                precheck,
            )
            if result is not None:
                return result
        return None

    def follow_forward_path_w_refinement(self, result, refine=False, stop_when=None):
        if hasattr(self, "_roll_indices"):
            if not self.planner.accepts(result["position"], move_group=True):
                raise RuntimeError(
                    "Attempt to execute a path outside collection roll limits"
                )
        if self._path_remaining:
            raise RuntimeError("Nested path execution is unsupported")
        if self.control_mode != "pd_joint_pos":
            raise ValueError(
                "CollectionMotionPlanner requires absolute pd_joint_pos targets"
            )
        self._path_remaining = len(result["position"])
        self._path_first = True
        self._path_start_arm = (
            self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy().copy()
        )
        try:
            out = super().follow_forward_path_w_refinement(result, refine, stop_when)
            # Complete the sampled target before leaving a transfer/contact path.
            # The next primitive starts on a policy tick and cannot inherit half
            # of a noisy transfer command as its first contact command.
            if (not self.truncated and int(self.base_env.elapsed_steps[0]) % 2
                    and getattr(self, "_source_action", None) is not None):
                out = self._step(self._source_action)
            return out
        finally:
            self._path_remaining = 0
            self._path_first = False

    def change_gripper_state(self, t=6, gripper_state=1, stop_when=None, ramp=0):
        """Binary gripper commands, with a switch on an even 20 Hz step.

        A one-step hold when needed keeps the previous command on the odd step.
        All commands, including that hold, pass through env.step and recording.
        Finger motion is still governed by the unchanged physical PD controller.
        """
        goal = float(gripper_state)
        if goal not in (-1.0, 1.0) or ramp:
            raise ValueError(
                "Collection gripper commands must be binary, without ramps"
            )
        if t < 1:
            raise ValueError("A gripper command must execute at least one control step")
        if self.execution_noise is not None and self.execution_noise.stage is not None:
            raise RuntimeError(
                "Gripper changes are forbidden in an execution-noise phase"
            )
        if self.truncated:
            return self._guard.last_step
        if (
            goal != float(self.gripper_state)
            and int(self.base_env.elapsed_steps[0]) % 2
        ):
            self.idle_steps(t=1)
            if self.truncated:
                return self._guard.last_step
        return super().change_gripper_state(
            t=t, gripper_state=goal, stop_when=stop_when, ramp=0
        )

    def _step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        if action.shape != (13,) or action[7] not in (-1.0, 1.0):
            raise ValueError(
                "Collection actions require 13 values and a binary gripper"
            )
        previous = getattr(self, "_last_gripper_command", None)
        if previous is not None and action[7] != previous:
            if int(self.base_env.elapsed_steps[0]) % 2:
                raise RuntimeError("Gripper switch bypassed the even-step primitive")
        self._last_gripper_command = float(action[7])
        step = int(self.base_env.elapsed_steps[0])
        previous_target = getattr(self, "_source_action", None)
        sample_tick = step % 2 == 0 or previous_target is None
        # Clean numerical roundoff while generating the command, before env.step.
        action[-2:] = np.clip(action[-2:], -1.0, 1.0)
        action[-2:][np.abs(action[-2:]) <= 1e-12] = 0.0
        if self._path_remaining:
            if self.execution_noise is not None and sample_tick:
                step = int(self.base_env.elapsed_steps[0].item())
                if self._path_first:
                    self.execution_noise.start_path(
                        action[:7], self._path_start_arm, step
                    )
                action[:7] = self.execution_noise.apply(action[:7], step, hold_steps=2)
            self._path_remaining -= 1
            if sample_tick:
                self._path_first = False
        tracker = getattr(self, "_head_tracker", None)
        if tracker is not None and sample_tick:
            action[8:10] = tracker.command()
        for index, value in getattr(self, "fixed_action_targets", {}).items():
            action[index] = value
        # Generate policy targets on the 10 Hz clock, while physics, observation
        # and recording still step at 20 Hz. Sampling here, before env.step and
        # recording, makes stride-two export preserve every executed command.
        # Replay uses those commands directly and has no copy of this oracle.
        if step % 2 and previous_target is not None:
            action = previous_target.copy()
        else:
            self._source_action = action.copy()
        # Tape entries describe the same absolute action as the physical step.
        self._last_abs = action.copy()
        result = super()._step(action)
        if hasattr(self, "_roll_indices"):
            roll = self.env_agent.robot.get_qpos()[0].cpu().numpy()[self._roll_indices]
            self._roll_low = np.minimum(self._roll_low, roll)
            self._roll_high = np.maximum(self._roll_high, roll)
        return result
