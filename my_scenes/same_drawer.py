"""Close a cued drawer, put an apple on a distant plate, reopen the same drawer.

Three upper drawers of one kitchen column form the answer space. Closing erases
its geometric cue: the task's detent resets each drawer within 5 mm of shut to
exactly zero. This prevents residual joint position from marking the answer.
The rule applies identically in collection, replay and policy evaluation.

The order is enforced by task latches. A settled apple placed on the plate before
closing the drawer invalidates the attempt. After closing, the cued drawer must
remain within the 5 mm closed tolerance until the apple is placed. Opening another
drawer beyond 2 cm also permanently invalidates the attempt. Final opening must reach 10 cm and stay settled
for 10 control steps. A uniform final guess selects the correct drawer with
probability 1/3 before manipulation errors; an overall blind success rate must be
measured, not inferred from that probability alone.

RGB/proprio/text policy observations omit task answers and stage latches. State
observations and info remain privileged diagnostics. The design erases the drawer
cue, but does not prove the absence of every possible active external-memory
strategy (for example, deliberately arranging the apple on its plate).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from utils.robocasa_utils import (
    parking_pose,
    counter_frame,
    dock_pose_for,
    load_objaverse_actor,
    require_get_fixture,
    restore_task_tensor,
)

#: The drawer column this task plays on, and the load-bearing reason it is ONE column:
#: the four drawers share a dock to the millimetre (`robocasa_utils.dedup_docks`), so
#: the robot's parking pose cannot encode which drawer is the answer.
DRAWER_FIXTURES = (
    "stack_1_main_group_1",
    "stack_1_main_group_2",
    "stack_1_main_group_3",
    "stack_1_main_group_4",
)

#: Each drawer articulation is `<fixture>_0` with a single prismatic `slidejoint`,
#: limits [-0.6, 0]: qpos 0 is closed, negative slides the box toward the robot.
#: "Open amount" throughout this file is the positive quantity `-qpos`.
DRAWER_ART_SUFFIX = "_0"

INSTRUCTIONS = (
    "Remember which drawer is open and close it. Keep it closed while putting the apple "
    "on the plate at the far counter, then return and open the same drawer. "
    "Leave every other drawer closed.",
)


@dataclass
class SameDrawerConfig:
    horizon: int = int(os.environ.get("MIKASA_SAME_DRAWER_HORIZON", "1600"))
    """80 seconds at 20 Hz. Development seeds 1000-1005 on master DSFetch
    exhausted the former 1200-step estimate during the return/final reach.
    The final reach, grasp, pull and hold need about 200 more steps; 1600 leaves
    modest execution slack without changing the physical success conditions."""

    n_drawers: int = 4
    "Drawers in the column (fixed by the kitchen); the obs one-hot stays this wide."

    drawer_choices: tuple = (1, 2, 3)
    """Three upper drawers; the bottom handle is below the qualified work envelope.
    All choices share a column and the same nominal robot dock."""

    init_open_range: tuple[float, float] = (0.10, 0.16)
    """Metres the target drawer stands open at reset, drawn per seed from this range.
    A range, not a constant, so the single scene quantity that varies at t=0 cannot be
    used as a fixed template."""

    closed_tol: float = 0.005
    """Open amounts at or below this count as closed, and the detent snaps them to
    exactly 0 while closing or stationary (see the module docstring)."""

    wrong_open_tol: float = 0.02
    """A non-target drawer opened past this, at any step, latches failure. Below it,
    brushing a front while closing the neighbour is not a crime."""

    open_success: float = 0.10
    "The target must be re-opened at least this far for success."

    hold_steps: int = 10
    """Consecutive steps the terminal state (target open, others closed, drawer
    static) must hold before success latches — the pull must end settled, not be a
    yank through the threshold."""

    apple_z_band: tuple[float, float] = (0.0, 0.12)
    "Apple origin height over the plate origin that counts as 'on the plate'."

    plate_radius: float = 0.10
    "Horizontal distance apple->plate that counts as on it."

    apple_along: float = 0.85
    """Metres along counter_main from its centre to the apple station — the sink-free
    right span (x ~ 2.35 on kitchen 102), SeasonDish's proven grasp neighbourhood; the
    drawer column sits ~1.8 m away, which is the mobility the delay rides on."""

    apple_dock_toward: float = 0.15
    """Metres toward the counter from the scene builder's nominal dock."""

    plate_from_centre: float = 0.125
    """Offset of the apple's spawn row toward the counter's front edge."""

    apple_from_plate: float = 0.28
    "Along-counter separation between the apple (front band) and the plate (deep row)."

    robot_jitter_xy: float = 0.02
    robot_jitter_yaw: float = 0.03

    spawn_jitter_xy: float = 0.02
    "Uniform +/- jitter on the plate position and the apple offset, per seed."

    def validate(self) -> None:
        """Every constraint the thresholds must satisfy, checked at construction.

        Example:
            >>> SameDrawerConfig().validate()
            >>> SameDrawerConfig(closed_tol=0.05).validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            AssertionError
        """
        lo, hi = self.init_open_range
        assert 0.0 < self.closed_tol < self.wrong_open_tol < self.open_success, (
            self.closed_tol, self.wrong_open_tol, self.open_success)
        assert self.wrong_open_tol < lo <= hi <= 0.55, self.init_open_range
        assert self.open_success <= lo, (self.open_success, lo)
        assert self.n_drawers == len(DRAWER_FIXTURES), self.n_drawers
        assert len(self.drawer_choices) >= 2, self.drawer_choices
        assert all(0 <= int(i) < self.n_drawers for i in self.drawer_choices)
        assert self.hold_steps > 0 and self.horizon > 10 * self.hold_steps
        assert 0.0 <= self.apple_z_band[0] < self.apple_z_band[1]
        assert self.plate_radius > 0 and self.apple_from_plate > 2 * self.plate_radius


def score_same_drawer(
    target_amt: torch.Tensor,
    others_max_amt: torch.Tensor,
    closed_done: torch.Tensor,
    apple_done: torch.Tensor,
    wrong_touched: torch.Tensor,
    held: torch.Tensor,
    *,
    open_success: float,
    closed_tol: float,
    hold_steps: int,
    sequence_violated: torch.Tensor | None = None,
) -> dict:
    """The episode verdict as a pure function of the latched quantities.

    No simulator, no `self`: the sequencing lives in the latches (`closed_done` gates
    `apple_done` at latch time, in `evaluate`), so scoring is a conjunction.

    Example:
        >>> t = lambda *v: torch.tensor(v)
        >>> out = score_same_drawer(t(0.12), t(0.0), t(True), t(True), t(False),
        ...                         t(10), open_success=0.10, closed_tol=0.005,
        ...                         hold_steps=10)
        >>> bool(out["success"][0]), bool(out["failed"][0])
        (True, False)
        >>> out = score_same_drawer(t(0.12), t(0.0), t(True), t(True), t(True),
        ...                         t(10), open_success=0.10, closed_tol=0.005,
        ...                         hold_steps=10)
        >>> bool(out["success"][0]), bool(out["failed"][0])   # wrong drawer touched
        (False, True)
        >>> out = score_same_drawer(t(0.12), t(0.0), t(True), t(False), t(False),
        ...                         t(10), open_success=0.10, closed_tol=0.005,
        ...                         hold_steps=10)
        >>> bool(out["success"][0])                           # apple never done
        False
    """
    reopened = target_amt >= open_success
    others_closed = others_max_amt <= closed_tol
    sequence_violated = torch.zeros_like(wrong_touched) if sequence_violated is None else sequence_violated
    clean = ~(wrong_touched | sequence_violated)
    success = closed_done & apple_done & reopened & others_closed & clean & (
        held >= hold_steps)
    return dict(
        success=success,
        failed=wrong_touched | sequence_violated,
        sequence_violated=sequence_violated,
        reopened=reopened,
        others_closed=others_closed,
        closed_done=closed_done,
        apple_done=apple_done,
        held=held,
    )


@register_env(
    "MikasaSameDrawer-v0",
    max_episode_steps=SameDrawerConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class SameDrawerTask(BaseEnv):
    """Close the open drawer, put the apple on the plate, open the same drawer.

    A blind final choice is uniform over the three eligible drawers. Physical
    success is measured separately from this choice probability.
    """

    SUPPORTED_ROBOTS = ["ds_fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = SameDrawerConfig()

    apple: Actor
    plate: Actor

    def __init__(self, *args, robot_uids="ds_fetch", scene_idx: int | None = None, **kwargs):
        self.cfg.validate()
        self.scene_idx = scene_idx
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        # Frames the drawer column (x=0.5) and the run toward the apple counter.
        # 512, not 2048 — RecordEpisode buffers every frame in host RAM.
        pose = sapien_utils.look_at(eye=[2.4, -2.8, 2.1], target=[1.6, -0.4, 0.6])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # ------------------------------------------------------------------ load --

    def _load_agent(self, options: dict):
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        """Geometry and initial poses only; which drawer is open is drawn per episode
        in `_initialize_episode`."""
        super()._load_scene(options)

        self.scene_builder = RoboCasaSceneBuilder(self)
        if self.scene_idx is None:
            self.scene_builder.build()
        else:
            self.scene_builder.build([self.scene_idx] * self.num_envs)

        self._fix_ds_fetch_collision_bits()

        # Per-env fixture lookups, called once and cached (a repeated substring
        # lookup draws from the shared episode RNG — see robocasa_utils).
        drawer_starts, apple_docks, plate_homes, apple_homes = [], [], [], []
        for i in range(self.num_envs):
            fixtures = self.scene_builder.scene_data[i]["fixtures"]
            # The drawer column. require_get_fixture raises with the available names
            # if this kitchen has no stack_1 — the task needs a 4-drawer column.
            for name in DRAWER_FIXTURES:
                require_get_fixture(self.scene_builder, fixtures, name, scene_idx=self.scene_idx)

            # The robot starts at the column's shared dock, facing the open drawer.
            dock_p, dock_yaw = dock_pose_for(self.scene_builder, fixtures, DRAWER_FIXTURES[1])
            drawer_starts.append(np.array(
                [dock_p[0], dock_p[1], 0.0,
                 np.cos(dock_yaw / 2), 0.0, 0.0, np.sin(dock_yaw / 2)],
                dtype=np.float32,
            ))

            # The apple station: the far counter. Plate at its centre + jitter (drawn
            # per episode); apple `apple_from_plate` along the counter from it.
            # Mid-kitchen, not the right counter: counter_right's free surface ends
            # 13 cm past the plate row with the room wall beyond, and every elbow-out
            # IK candidate strikes it — the whole station refused wholesale there
            # (probed 2026-08-28). counter_main's right span (x ~ 2.2) is SeasonDish's
            # own neighbourhood, where this exact grasp geometry is proven.
            counter = require_get_fixture(
                self.scene_builder, fixtures, "counter_main_main_group", scene_idx=self.scene_idx)
            top_c, along, across = counter_frame(counter)
            top = top_c + along * self.cfg.apple_along
            a_p, a_yaw = dock_pose_for(
                self.scene_builder, fixtures, "counter_main_main_group",
                offset=(self.cfg.apple_along, self.cfg.apple_dock_toward))
            apple_docks.append(np.array([a_p[0], a_p[1], a_yaw], dtype=np.float32))
            # The FRONT band, not the centreline: the counter's centre sits 1.07 m
            # from the dock — past the arm's measured envelope (the SeasonDish
            # reach-band lesson, re-learned here with `IK Failed` on every apple
            # grasp). `front` is derived from the dock, so the across-sign trap in
            # robocasa_utils cannot bite.
            front = np.asarray([a_p[0], a_p[1], top[2]], dtype=np.float64) - np.asarray(top, dtype=np.float64)
            front[2] = 0.0
            front /= max(np.linalg.norm(front), 1e-9)
            along_u = np.asarray(along, dtype=np.float64)
            # The APPLE owns the front band (grasp reach lives there); the PLATE sits
            # deeper and on the other side of the dock, out of the grasp corridor —
            # at the same depth and 22 cm apart the reach screws grazed the plate and
            # the wrist grazed the counter (measured). A release is a high hover, so
            # depth costs the plate nothing.
            apple_home = np.asarray(top, dtype=np.float64) + front * self.cfg.plate_from_centre
            # Keep the plate wholly on the worktop and behind the grasp corridor.
            plate_home = (np.asarray(top, dtype=np.float64) + front * 0.05
                          - along_u * self.cfg.apple_from_plate)
            plate_homes.append(plate_home)
            apple_homes.append(apple_home)

        self._robot_start_np = np.stack(drawer_starts)
        self._apple_dock_np = np.stack(apple_docks)
        self._plate_home_np = np.stack(plate_homes)
        self._apple_home_np = np.stack(apple_homes)

        # The plate is kinematic: it is scenery the apple must land on, re-posed per
        # episode; a dynamic plate would add a plate-drift failure mode the task is
        # not about. The apple is the manipulated object.
        self.plate = load_objaverse_actor(
            self, "plate", "plate", sapien.Pose(p=self._plate_home_np[0] + [0, 0, 0.02]),
            index=0, body_type="kinematic",
        )
        self.apple = load_objaverse_actor(
            self, "apple", "apple", sapien.Pose(p=self._apple_home_np[0] + [0, 0, 0.05]),
            index=0, dynamic=True,
        )
        # A grippy apple, the same way `build_colorful_cube` ships grippy stack cubes:
        # at the asset's default material the sphere ROLLS out from under descending
        # fingertips before the close (verified by state: closes at 2 mm pose error,
        # micro-lift raises nothing). Fruit is not frictionless; the material is task
        # physicality, not a solver knob.
        grippy = sapien.physx.PhysxMaterial(static_friction=2.0, dynamic_friction=1.5,
                                            restitution=0.0)
        for body in self.apple._bodies:
            for shape in body.get_collision_shapes():
                shape.set_physical_material(grippy)

    def _fix_ds_fetch_collision_bits(self):
        if self.robot_uids != "ds_fetch" or self.agent is None:
            return
        for link in (self.agent.l_wheel_link, self.agent.r_wheel_link):
            for bit in range(25, 31):
                link.set_collision_group_bit(group=2, bit_idx=bit, bit=1)
        self.agent.base_link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    # ---------------------------------------------------------------- buffers --

    def _after_reconfigure(self, options: dict):
        super()._after_reconfigure(options)
        n, dev = self.num_envs, self.device
        self._drawer_arts = [
            self.scene.articulations[f"{name}{DRAWER_ART_SUFFIX}"] for name in DRAWER_FIXTURES
        ]
        # The handle of each drawer, located from its own collision geometry with the
        # links POSED — here, after scene setup; in _load_scene they still sit at
        # their local frames and the scan returns local coordinates (found the hard
        # way: (0, -0.335, 0)). The moving link (`inner_box`) carries one long
        # cylinder — the bar (r~0.013 m, half-length~0.064), proud of the front. Its
        # world centre at closed is the home; it travels with the joint as
        # (x, y + qpos, z), qpos <= 0.
        handle_homes = []
        for i in range(n):
            per_drawer = []
            for art in self._drawer_arts:
                bar = None
                for link in art.links:
                    if link.name.split("/")[-1] != "inner_box":
                        continue
                    comp = link._objs[i]
                    raw = link.pose.raw_pose[i].cpu().numpy()
                    lp = sapien.Pose(p=raw[:3], q=raw[3:])
                    for sh in comp.get_collision_shapes():
                        if sh.__class__.__name__.endswith("Cylinder") and float(sh.half_length) > 0.05:
                            bar = np.asarray((lp * sh.get_local_pose()).p, dtype=np.float64)
                assert bar is not None, "no handle bar found on a drawer"
                per_drawer.append(bar)
            handle_homes.append(np.stack(per_drawer))
        self._handle_home_np = np.stack(handle_homes)  # (n, 4, 3), at closed
        self.target_drawer = torch.zeros(n, dtype=torch.int64, device=dev)
        self.init_open = torch.zeros(n, dtype=torch.float32, device=dev)
        self.closed_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.apple_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.wrong_drawer_touched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.sequence_violated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.held_count = torch.zeros(n, dtype=torch.int32, device=dev)
        self._last_eval_step = torch.full((n,), -1, dtype=torch.int32, device=dev)
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)  # noqa: E731
        self._robot_start = t(self._robot_start_np)
        self.apple_dock = t(self._apple_dock_np)
        self.plate_home = t(self._plate_home_np)
        self.apple_home = t(self._apple_home_np)
        self.handle_home = t(self._handle_home_np)

    # ------------------------------------------------------------- initialize --

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # The one bit the benchmark rests on comes from the batched episode RNG,
            # never torch.rand (writing-tasks.md): reproducible from the seed alone.
            rng = self._batched_episode_rng[env_idx]
            choices = torch.as_tensor(self.cfg.drawer_choices, dtype=torch.int64)
            pick = torch.as_tensor(rng.randint(0, len(self.cfg.drawer_choices)),
                                   dtype=torch.int64)
            self.target_drawer[env_idx] = choices[pick]
            lo, hi = self.cfg.init_open_range
            self.init_open[env_idx] = torch.as_tensor(
                lo + (hi - lo) * rng.rand(), dtype=torch.float32)
            jit = lambda: torch.as_tensor(  # noqa: E731
                np.stack([rng.uniform(-self.cfg.spawn_jitter_xy, self.cfg.spawn_jitter_xy),
                          rng.uniform(-self.cfg.spawn_jitter_xy, self.cfg.spawn_jitter_xy)],
                         axis=-1), dtype=torch.float32)

            # Open the target drawer; every other drawer shut. scene_builder
            # .initialize has just restored the fixtures, so write after it.
            for k, art in enumerate(self._drawer_arts):
                q = art.get_qpos().clone()
                is_target = (self.target_drawer[env_idx] == k)
                q[env_idx, 0] = torch.where(
                    is_target, -self.init_open[env_idx], torch.zeros(b))
                art.set_qpos(q)
                art.set_qvel(torch.zeros_like(q))

            # Plate and apple at their homes + per-seed jitter; velocities implied
            # zero for the kinematic plate, explicit for the apple.
            plate_p = self.plate_home[env_idx].clone()
            plate_p[:, :2] += jit()
            self.plate.pose = Pose.create_from_pq(p=plate_p + torch.tensor([0.0, 0.0, 0.01]))
            apple_p = self.apple_home[env_idx].clone()
            apple_p[:, :2] += jit()
            apple_p[:, 2] += 0.06
            self.apple.pose = Pose.create_from_pq(p=apple_p)
            self.apple.set_linear_velocity(torch.zeros(b, 3))
            self.apple.set_angular_velocity(torch.zeros(b, 3))

            start = torch.tensor(self._robot_start_np[env_idx.cpu().numpy()], device=self.device)
            start[:, :2] += torch.as_tensor(
                np.stack([rng.uniform(-self.cfg.robot_jitter_xy, self.cfg.robot_jitter_xy),
                          rng.uniform(-self.cfg.robot_jitter_xy, self.cfg.robot_jitter_xy)], axis=-1),
                dtype=torch.float32, device=self.device)
            yaw = 2 * torch.atan2(start[:, 6], start[:, 3])
            yaw += torch.as_tensor(rng.uniform(-self.cfg.robot_jitter_yaw, self.cfg.robot_jitter_yaw),
                                   dtype=torch.float32, device=self.device)
            start[:, 3] = torch.cos(yaw / 2)
            start[:, 6] = torch.sin(yaw / 2)
            self._robot_start[env_idx] = start
            self._restore_robot(env_idx)

            self.closed_done[env_idx] = False
            self.apple_done[env_idx] = False
            self.wrong_drawer_touched[env_idx] = False
            self.sequence_violated[env_idx] = False
            self.held_count[env_idx] = 0
            self._last_eval_step[env_idx] = -1

    def _restore_robot(self, env_idx: torch.Tensor):
        """Rest keyframe + the column dock, overwriting whatever dock the scene
        builder picked (same rationale and ordering as season_dish)."""
        if self.agent is None:
            return
        self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
        self.agent.robot.set_pose(Pose.create(self._robot_start[env_idx]))

    # --------------------------------------------------------------- evaluate --

    def drawer_open_amounts(self) -> torch.Tensor:
        """`(n_envs, n_drawers)` positive open amounts, read from the joints."""
        cols = [(-art.get_qpos()[:, 0]).clamp(min=0.0) for art in self._drawer_arts]
        return torch.stack(cols, dim=-1).to(torch.float32)

    def evaluate(self) -> dict:
        """Runs every step (and at t=0 inside reset). All mutation — the detent
        snap and every latch — sits behind the advance guard, so a second call in
        the same sim step is a no-op and out-of-band callers cannot move latches."""
        cfg = self.cfg
        amts = self.drawer_open_amounts()
        step = self.elapsed_steps.to(torch.int32)
        advance = step != self._last_eval_step

        if bool(advance.any()):
            # Detent: a drawer within closed_tol clicks to exactly zero. Kills the
            # sub-tolerance notebook; also what real drawer hardware does.
            for art in self._drawer_arts:
                q = art.get_qpos()
                v = art.get_qvel()
                # Negative velocity opens this slide. Snapping an opening drawer
                # each tick prevents any pull slower than closed_tol/control_dt.
                # Snap only while closing or stationary; intentional opening must
                # pass continuously through the first few millimetres.
                near = (advance & (q[:, 0] > -cfg.closed_tol) & (q[:, 0] < 0.0)
                        & (v[:, 0] >= -1e-4))
                if bool(near.any()):
                    q, v = q.clone(), v.clone()
                    q[near, 0] = 0.0
                    v[near, 0] = 0.0
                    art.set_qpos(q)
                    art.set_qvel(v)
            amts = self.drawer_open_amounts()

            idx = self.target_drawer.unsqueeze(-1)
            target_amt = torch.gather(amts, 1, idx).squeeze(-1)
            others = amts.scatter(1, idx, 0.0)
            others_max = others.max(dim=-1).values

            self.wrong_drawer_touched |= advance & (others_max > cfg.wrong_open_tol)
            closed_before = self.closed_done.clone()
            self.closed_done |= advance & (target_amt <= cfg.closed_tol)

            apple_p = self.apple.pose.p
            plate_p = self.plate.pose.p
            flat = torch.linalg.norm(apple_p[:, :2] - plate_p[:, :2], dim=-1)
            z = apple_p[:, 2] - plate_p[:, 2]
            on_plate = (flat <= cfg.plate_radius) & (z >= cfg.apple_z_band[0]) & (
                z <= cfg.apple_z_band[1])
            settled = self.apple.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            grasped = self.agent.is_grasping(self.apple) if self.agent is not None else (
                torch.zeros_like(on_plate))
            # Neither preplacing the apple nor reopening the cued drawer during
            # the interlude may bypass the memory interval. A later close cannot
            # retroactively credit either order. Use the same tolerance as closed.
            early_apple = ~closed_before & on_plate & settled & ~grasped
            early_reopen = closed_before & ~self.apple_done & (target_amt > cfg.closed_tol)
            self.sequence_violated |= advance & (early_apple | early_reopen)
            self.apple_done |= (
                advance & closed_before & on_plate & settled & ~grasped
                & (target_amt <= cfg.closed_tol) & ~self.sequence_violated)

            # The terminal hold: consecutive steps with the target open, the others
            # closed and the target drawer still. Reset on any break.
            drawer_still = torch.ones_like(self.closed_done)
            for k, art in enumerate(self._drawer_arts):
                v = art.get_qvel()[:, 0].abs()
                drawer_still &= v < 0.02
            terminal = (
                self.apple_done & (target_amt >= cfg.open_success)
                & (others_max <= cfg.closed_tol) & drawer_still)
            self.held_count = torch.where(
                advance & terminal, self.held_count + 1,
                torch.where(advance & ~terminal, torch.zeros_like(self.held_count),
                            self.held_count))

            self._last_eval_step = torch.where(advance, step, self._last_eval_step)

        idx = self.target_drawer.unsqueeze(-1)
        target_amt = torch.gather(amts, 1, idx).squeeze(-1)
        others_max = amts.scatter(1, idx, 0.0).max(dim=-1).values
        out = score_same_drawer(
            target_amt, others_max, self.closed_done, self.apple_done,
            self.wrong_drawer_touched, self.held_count,
            open_success=cfg.open_success, closed_tol=cfg.closed_tol,
            hold_steps=cfg.hold_steps, sequence_violated=self.sequence_violated,
        )
        out["target_open_amt"] = target_amt
        return out

    # -------------------------------------------------------------------- obs --

    def _get_obs_extra(self, info: dict):
        """The cue one-hot is the state-mode analogue of seeing the open drawer: it
        is emitted every step but zeroed once `closed_done` fires — masked by the
        latch, exactly as the pixels go dark when the drawer shuts. Geometry that a
        state agent legitimately owns sits behind the use_state fence. No key is
        named for the answer."""
        cue = torch.nn.functional.one_hot(
            self.target_drawer, num_classes=self.cfg.n_drawers).to(torch.float32)
        cue = torch.where(self.closed_done.unsqueeze(-1), torch.zeros_like(cue), cue)
        obs = {}
        if self.obs_mode_struct.use_state:
            obs.update(
                drawer_cue=cue,
                drawer_open_amounts=self.drawer_open_amounts(),
                apple_pose=self.apple.pose.raw_pose,
                plate_pos=self.plate.pose.p,
                handle_home=self.handle_home.reshape(self.num_envs, -1),
            )
        return obs

    def get_language_instruction(self, **kwargs):
        return [INSTRUCTIONS[0]] * self.num_envs

    # ------------------------------------------------------------- checkpoint --

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        state["_robot_start"] = self._robot_start.clone()
        state["target_drawer"] = self.target_drawer.clone()
        state["init_open"] = self.init_open.clone()
        state["closed_done"] = self.closed_done.clone()
        state["apple_done"] = self.apple_done.clone()
        state["wrong_drawer_touched"] = self.wrong_drawer_touched.clone()
        state["sequence_violated"] = self.sequence_violated.clone()
        state["held_count"] = self.held_count.clone()
        state["_last_eval_step"] = self._last_eval_step.clone()
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        """Tolerant of a state dict with no task keys (the BaseEnv.set_state round
        trip drops them) and of numpy values from recorded trajectories — both via
        `restore_task_tensor`, `super()` first so the flags land on a consistent
        scene (same rationale as season_dish)."""
        super().set_state_dict(state, env_idx)
        for key in ("_robot_start", "target_drawer", "init_open", "closed_done", "apple_done",
                    "wrong_drawer_touched", "sequence_violated", "held_count", "_last_eval_step"):
            if key in state:
                setattr(self, key, restore_task_tensor(
                    getattr(self, key), state[key], self.device))
