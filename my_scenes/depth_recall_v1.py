"""MikasaDepthRecall-v1: retrieve the deepest prop and restore its three blockers.

Four coloured props occupy a seeded front-to-back permutation on a cabinet shelf.
The counter carries four pads, one in each prop's colour, and the episode shuffles
which counter slot each colour's pad lies on. The props are cleared onto their own
pads, the one from the deepest position stays there, and the others go back to their
original places. Every restored prop must previously have stood on its own pad, and a
prop released on another colour's pad voids the episode (`wrong_pad`).

The answer is the association between prop identity and original depth. It is
available while the shelf is being emptied and excluded from policy observations
once the props are staged. The pads fix WHERE each prop is staged (by colour, drawn
per episode, independent of the row order), so the staging layout cannot encode the
order: what is on the counter says nothing about where each prop came from.
The 1/6 guessing reference is conditional on the three blockers and a uniformly
shuffled restore assignment, not a universal memoryless-policy bound.

The four counter slots are spaced wider than the original 12 cm pitch; the pitch and
its reach evidence are on `slot_x_offsets`. Collection uses the unmodified master
DSFetch; historical reachability results from a different robot are not qualification
of this configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Pose

from utils.mikasa.assisted_grasp import AssistedGrasp
from utils.robocasa_utils import parking_pose, require_get_fixture, restore_task_tensor

#: Everything `get_state_dict` adds beyond sim state. All of it is task MEMORY or a
#: latch; every entry is rank <= 2 (`flatten_state_dict` ends in `torch.hstack`).
TASK_STATE_KEYS = (
    "original_slots", "left_slot", "restored", "was_lifted", "staged",
    "wrong_assign", "target_returned", "two_in_slot", "succeeded",
    "arrangement_hold", "last_eval_step", "pad_slots", "wrong_pad",
)

INSTRUCTIONS = (
    "Take the props out of the cabinet onto the counter pads of their own colour, "
    "keep the one from the back, and put the others back where they stood.",
    "Clear the row of props onto the matching coloured pads on the counter, keep the "
    "one that stood at the back, and return every other prop to the shelf spot it "
    "came from.",
    "Empty the shelf row onto the counter, each prop on the pad of its colour, hold on "
    "to the prop from the deepest spot, and put the rest back exactly where each of "
    "them was.",
)

#: The staging pads' plate size. One pad per prop, in that prop's colour; the episode
#: draws which counter slot each pad lies on. Decoration with a job: the pads are where
#: the chore says to put each prop, and a viewer (or a policy reading pixels) cannot
#: see a coordinate. Kinematic and collision-free — `hide_visual` asserts no collision
#: shapes and the pose setter asserts a non-static body under GPU sim, which is the pair
#: of constraints every marker in this package is built to. 9 cm across: wider than a
#: prop (4.5 cm) by a finger's clearance a side, so the colour shows around a standing
#: prop from the head cameras.
PAD_HALF = (0.045, 0.045, 0.002)

#: Fixed identity palette — a prop cannot be recoloured per episode, so colour IS
#: identity and the episode's ticket permutes which SLOT each colour occupies. The pads
#: use the same colours: pad c belongs to prop c.
PROP_COLORS = (
    (1.0, 0.1, 0.1, 1.0),   # red
    (0.1, 0.8, 0.1, 1.0),   # green
    (0.15, 0.3, 1.0, 1.0),  # blue
    (1.0, 0.85, 0.0, 1.0),  # yellow
)


@dataclass
class DepthRecallV1Config:
    """Every threshold named once; a magic number inside evaluate() is how a failure
    becomes unattributable."""

    horizon: int = 5700
    """MEASURED 2026-09-10 by the K22 rule on the FOUR-prop row (seven transfers), on
    calibration seeds 11-22 and never the eval seeds 0-9: **12/12**, worst episode 3786
    steps, so 3786 x 1.5 = 5679 -> 5700. (5500 was this same measurement before the
    mid-height cross, when calibration was 11/12 at 3666.)

    The three-prop row measured 4300 the same way (worst successful 2816). Its history
    is worth keeping because the difference was only how much the oracle stopped doing:
    3566-3809 steps re-docking before every transfer, 2872-3141 with one dock but a fold
    and a duck around it, 2118-2816 once those were dropped."""

    # -- the cabinet, the row, the props (v0's measured territory) -------------
    cabinet_name: str = "cab_main_main_group"
    counter_name: str = "counter_main_main_group"
    door_hinge: str = "rightdoorhinge"
    door_open_rad: float = 1.75
    n_props: int = 4
    prop_half_xy: float = 0.0225
    prop_half_h: float = 0.06
    "4.5 x 4.5 x 12 cm — v0's prop, and the object `MikasaCabinetStowProp-v0` measured."
    shelf_top_z: float = 1.4200
    spawn_clearance: float = 0.004
    finger_reach: float = 0.030
    """How far the finger plates stand PAST the TCP along the approach axis, metres.

    Not a threshold the task checks — a fact about the gripper that `validate` uses to
    refuse a row it cannot take apart. Measured 2026-09-10 on this task's first run:
    at v0's 5.5 cm pitch the fingers are in contact with the NEXT prop while grasping
    one, mplib refuses the lift with `l_gripper_finger_link <-> prop_1` at the first
    knot, and physics carries the neighbour out of the cabinet with it — both props
    left the shelf together and landed 5 cm apart on the same counter slot."""

    row_slots_y: tuple = (-0.30, -0.23, -0.16, -0.09)
    """Front -> deep. The DEEP one holds the target.

    FOUR slots at the same 7 cm pitch (owner, 2026-09-10). The row grows FORWARD, toward
    the opening, and not deeper: deeper is where the placing limit is (-0.08, measured),
    while forward is the easy direction — the front slot at -0.30 is 10 cm inside the
    cabinet's own face at -0.40 and nearer the arm than any slot the three-prop row had.
    The alternative, keeping the row's span and shrinking the pitch to 6.3 cm, was
    rejected: it puts the finger clearance at 4.05 cm, inside the band between the 3.25
    that FAILED and the 4.75 that works, i.e. exactly where nothing is known.

    7 cm pitch, not v0's 5.5. v0 could live with the tighter row because its blank
    accepts a prop plowed aside and put back ("снос предметов предплечьем без хвата и
    возврат на место — допустим"); v1 cannot, because a neighbour dragged OUT of the
    cabinet is then staged on the counter as if it had been chosen, and the restore is
    aiming at a row that no longer holds what it thinks. The arithmetic `validate`
    enforces: pitch - prop_half_xy > finger_reach + 5 mm, i.e. > 5.75 cm at this
    gripper. 7 cm leaves 1.75 cm.

    Both ends stay inside W18's measured band: the front moves 3 cm nearer the opening
    (easier, and inside the retrieval family's own -0.28 spawn limit) and the deep slot
    does not move at all. The restore only ever fills the front and middle, which leaves
    15 and 8 cm against the 2026-09-10 placing limit."""
    row_margin_x: float = 0.13
    row_jitter_x: float = 0.04
    "Row x: jitter about the exposed half's centre -> x in [2.46, 2.54]."

    # -- the four counter slots ------------------------------------------------
    slot_x_offsets: tuple = (-0.54, -0.36, -0.18, 0.0)
    """Where the four staging slots sit along the counter, relative to the ROW's x.

    West of the row, evenly spaced by 18 cm (the owner, 2026-10-16: wider than the
    original 12 cm). West, because east of the row the open right door hangs over the
    counter's last 25 cm. 18 cm, because that is what the arm reaches from the ONE dock:
    cuRobo's scan of the planner's own poses (`scripts/dr_pad_reach.py`, every grasp
    lift, the row's +-4 cm jitter, self-collision, hits re-checked in float64) at the
    pads' y (`place_across`) reaches 64 cm west for the restore grasp and, with the
    extraction carrying the prop back to the dock line
    (`depth_recall_v1_planner.carry_out`), 60 cm for the place; 54 cm keeps 6 cm of that
    in hand. The old carry (back only until the prop is over the pads' line) placed no
    further than 44 cm. With pads 18 cm apart a prop is on one pad or none
    (`slot_radius`)."""
    place_across: float = -0.56
    """World y of the pads: the open-sky counter band, 3.5 cm nearer the robot than the
    retrieval family's -0.525. Measured 2026-10-16 by the same scan as `slot_x_offsets`:
    the cross to a pad 54 cm west plans at two of the three torso rungs from every row
    slot at -0.56, at one rung for the deepest slot at -0.525; the counter's front edge
    is at -0.65, the pad's at -0.605."""
    slot_radius: float = 0.08
    """How close to a pad's centre a prop must stand to count as ON that pad. Half the
    narrowest pitch at most, so a prop is on one pad or none."""

    # -- what counts as standing, and what voids the episode -------------------
    slot_xy_tol: float = 0.035
    "How far (xy) a prop may sit off a ROW slot and still count as in it."
    upright_tilt_deg: float = 15.0
    "Maximum lean from vertical for a standing prop, in degrees."
    upright_tol: float = 0.015
    """A standing prop's centre sits `prop_half_h` above its surface; a tipped 12 cm
    prop's centre drops ~3.5 cm, far outside this."""
    hold_steps: int = 15
    """Consecutive steps the finished arrangement must hold before success latches.

    Without it a single frame in which every predicate happens to be true wins the
    episode, and the verdict then survives whatever follows: measured 2026-09-10 by
    setting `succeeded` and displacing a prop — `restored` went to all-False and
    `success` stayed True. `settled` alone does not cover this, because it only forbids
    a prop passing THROUGH the right pose; it says nothing about one that arrives, is
    counted, and is knocked over afterwards.

    15 is `SeasonDishConfig.hold_steps`, ~0.75 s at the control rate, and the oracle
    clears it easily: it idles `SETTLE_STEPS` (30) after the last release."""

    settle_lin_speed: float = 0.05
    settle_ang_speed: float = 0.2
    leave_tol: float = 0.06
    """A movable counts as having LEFT its row slot once its xy runs this far from the
    slot point. Success requires all three blockers to have left and come back, so
    "reach past the row without touching it" cannot be credited as a restore."""

    def validate(self) -> None:
        assert self.horizon > 0
        assert self.n_props in (3, 4), (
            "v1 is floored for 3 (one target, two movable -> 0.5) or 4 (three movable, "
            "and the restore order is forced by depth, so 3x2x1 -> 1/6). More would need "
            "a fifth counter slot"
        )
        assert len(self.slot_x_offsets) >= self.n_props, (
            "every prop has to have a slot to stand on while the row is apart"
        )
        assert 0.9 <= self.door_open_rad <= 2.9
        assert 0 < self.prop_half_xy <= 0.028
        assert 0.04 <= self.prop_half_h <= 0.10
        assert len(self.row_slots_y) == self.n_props
        ys = list(self.row_slots_y)
        assert ys == sorted(ys), "row slots must run front (south) -> deep"
        for a, b in zip(ys, ys[1:]):
            assert b - a - self.prop_half_xy > self.finger_reach + 0.005, (
                f"row pitch {b - a:.3f} m leaves "
                f"{b - a - self.prop_half_xy:.3f} m between the TCP and the next prop's "
                f"face, and the fingers reach {self.finger_reach:.3f} past the TCP: "
                "grasping one prop would touch its neighbour and carry it along "
                "(measured 2026-09-10)"
            )
        assert -0.31 <= ys[0] and ys[-1] <= -0.035, (
            "row leaves the usable band: the front slot must stay inside the cabinet "
            "(its face is at -0.40) and the deep one inside the placing limit. W18 "
            "measured -0.20/-0.145/-0.09; -0.30 is 10 cm forward of that and nearer the "
            "arm, which is the easy direction, but it is an EXTRAPOLATION of W18 and the "
            "first sweep is what confirms it"
        )
        assert len(self.slot_x_offsets) == 4, "the owner's design is four slots"
        offs = list(self.slot_x_offsets)
        assert offs == sorted(offs), "slot offsets run west -> east"
        for a, b in zip(offs, offs[1:]):
            assert b - a >= 0.10, (
                "slots closer than 10 cm: a 4.5 cm prop plus the ~2.8 cm a side the "
                "pads need does not fit between them"
            )
            assert b - a >= 2 * self.slot_radius, (
                "pads closer than twice `slot_radius`: one prop could count as standing "
                "on two pads, its own and a wrong one"
            )
            assert b - a >= 2 * PAD_HALF[0] + 0.02, "the pads would touch"
        assert max(offs) <= 0.0, (
            "a slot east of the row is under the open right door, which hangs over the "
            "counter's last 25 cm"
        )
        assert min(offs) >= -0.54, (
            "the place's reach from the dock line runs out at 60 cm west of the row "
            "(cuRobo scan 2026-10-16, kinematics + self-collision, before the scene's "
            "own collisions): keep at least 6 cm of it"
        )
        assert self.hold_steps > 0
        assert self.hold_steps * 2 < self.horizon, (
            "the hold must fit after the work with room to spare"
        )
        assert 0 < self.slot_xy_tol < 2 * self.prop_half_xy
        assert 0 < self.upright_tilt_deg <= 45.0
        assert 0 < self.upright_tol < self.prop_half_h / 2
        assert self.slot_xy_tol < self.leave_tol <= 0.10
        assert self.slot_radius > 2 * self.prop_half_xy
        assert -0.65 < self.place_across < -0.448, "outside the open-sky counter band"

    # -- the robot start -------------------------------------------------------
    start_y: float = -1.90
    start_yaw_deg: float = 90.0
    start_jitter_x: float = 0.02
    start_jitter_y: float = 0.08
    start_jitter_yaw: float = 0.10


@register_env(
    "MikasaDepthRecall-v1",
    max_episode_steps=DepthRecallV1Config.horizon,
    asset_download_ids=["RoboCasa"],
)
class DepthRecallV1Task(BaseEnv):
    """Clear the row onto the counter, keep the deepest prop, put the rest back."""

    SUPPORTED_ROBOTS = ["ds_fetch", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = DepthRecallV1Config()

    def __init__(self, *args, robot_uids="ds_fetch", scene_idx: int | None = 0,
                 assisted_grasp: bool = False, **kwargs):
        """``assisted_grasp=True`` welds a prop to the hand by OmniGibson's rule
        (`utils/mikasa/assisted_grasp.py`). Off by default: it changes the physics, so
        demos and evaluations made with it are a different task and must say so."""
        self.cfg.validate()
        self.scene_idx = scene_idx
        self.assisted_grasp = bool(assisted_grasp)
        self._ag = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    @property
    def _default_sensor_configs(self):
        # ds_fetch carries its own head and wrist rigs; a task camera would double-render.
        return []

    @property
    def _default_human_render_camera_configs(self):
        # Two views, side by side in a recorded clip. From behind: the robot, the row and
        # the counter. From the west, along the counter: the four pads and the row in the
        # open cabinet, which the robot's own body hides from behind while it works.
        pose = sapien_utils.look_at(eye=[1.95, -2.80, 2.05], target=[2.22, -0.40, 1.20])
        side = sapien_utils.look_at(eye=[1.30, -1.70, 1.90], target=[2.35, -0.35, 1.20])
        return [CameraConfig("render_camera", pose, 512, 512, 1.2, 0.01, 100),
                CameraConfig("render_camera_side", side, 512, 512, 1.2, 0.01, 100)]

    # ------------------------------------------------------------------- load --

    def _load_agent(self, options: dict):
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        """Geometry and initial poses only — never `set_pose` (discarded by `_setup`)."""
        super()._load_scene(options)

        self.scene_builder = RoboCasaSceneBuilder(self)
        if self.scene_idx is None:
            self.scene_builder.build()
        else:
            self.scene_builder.build([self.scene_idx] * self.num_envs)
        self._fix_ds_fetch_collision_bits()

        self.counters = [
            require_get_fixture(
                self.scene_builder, self.scene_builder.scene_data[i]["fixtures"],
                self.cfg.counter_name, scene_idx=self.scene_idx,
            )
            for i in range(self.num_envs)
        ]
        self.cabinets = [
            require_get_fixture(
                self.scene_builder, self.scene_builder.scene_data[i]["fixtures"],
                self.cfg.cabinet_name, scene_idx=self.scene_idx,
            )
            for i in range(self.num_envs)
        ]

        # The door articulation, one per sub-scene, resolved once and strictly.
        matches = [(n, a) for n, a in self.scene.articulations.items()
                   if self.cfg.cabinet_name in n]
        assert len(matches) == self.num_envs, (
            f"expected one {self.cfg.cabinet_name!r} articulation per env, found "
            f"{[n for n, _ in matches]!r} for num_envs={self.num_envs}"
        )
        self._door = []
        for _name, art in sorted(matches):
            names = [j.name for j in art.get_active_joints()]
            assert self.cfg.door_hinge in names, (_name, names)
            self._door.append((art, names.index(self.cfg.door_hinge)))

        # Three props. Colour is identity and is FIXED per index; the episode ticket
        # permutes which row slot each index occupies, which is the answer.
        self.props = [
            actors.build_box(
                self.scene,
                half_sizes=[self.cfg.prop_half_xy, self.cfg.prop_half_xy,
                            self.cfg.prop_half_h],
                color=list(PROP_COLORS[i]),
                name=f"prop_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[2.50, -0.19, 1.60 + 0.15 * i]),
            )
            for i in range(self.cfg.n_props)
        ]

        # One flat pad per prop, in that prop's colour; the episode puts each on a slot.
        self.pads = [
            actors.build_box(
                self.scene,
                half_sizes=list(PAD_HALF),
                color=list(PROP_COLORS[c]),
                name=f"pad_{c}",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(p=[2.30 - 0.18 * c, -0.525, 0.93]),
            )
            for c in range(self.cfg.n_props)
        ]

        counter_tops = np.array(
            [float(np.asarray(c.pos)[2] + np.asarray(c.size)[2] / 2.0) for c in self.counters],
            dtype=np.float32,
        )
        self._counter_top_z = torch.as_tensor(counter_tops, device=self.device)
        half_lo, half_hi = self._exposed_half()
        self._row_centre_x = torch.full((self.num_envs,), float((half_lo + half_hi) / 2.0),
                                        device=self.device)
        # Filled per episode: the three row points and the four counter points.
        self._row_points = torch.zeros((self.num_envs, self.cfg.n_props, 3), device=self.device)
        self._slot_points = torch.zeros((self.num_envs, 4, 3), device=self.device)

    def _exposed_half(self) -> tuple[float, float]:
        """The x-span the open right door exposes, inset by the row margin."""
        cab = self.cabinets[0]
        pos = np.asarray(cab.pos, dtype=np.float64)
        size = np.asarray(cab.size, dtype=np.float64)
        return (float(pos[0]) + self.cfg.row_margin_x,
                float(pos[0] + size[0] / 2.0) - self.cfg.row_margin_x)

    def _fix_ds_fetch_collision_bits(self):
        """RoboCasa wheel/base exclusions; preserve arm, head and finger contacts."""
        if self.robot_uids != "ds_fetch" or self.agent is None:
            return
        for link in (self.agent.l_wheel_link, self.agent.r_wheel_link):
            for bit in range(25, 31):
                link.set_collision_group_bit(group=2, bit_idx=bit, bit=1)
        self.agent.base_link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    # ------------------------------------------------------------- initialize --

    def _after_reconfigure(self, options: dict):
        n, k, dev = self.num_envs, self.cfg.n_props, self.device
        # Allocated once at full width: get_state_dict is called before the first
        # _initialize_episode (sapien_env.py:332, and RecordEpisode on every reset).
        self.original_slots = torch.zeros((n, k), dtype=torch.long, device=dev)
        self.left_slot = torch.zeros((n, k), dtype=torch.bool, device=dev)
        self.restored = torch.zeros((n, k), dtype=torch.bool, device=dev)
        self.was_lifted = torch.zeros((n, k), dtype=torch.bool, device=dev)
        self.staged = torch.zeros((n, k), dtype=torch.bool, device=dev)
        # Which counter slot each colour's pad lies on, drawn per episode; and the latch
        # for a prop released on a pad that is not its own.
        self.pad_slots = torch.zeros((n, k), dtype=torch.long, device=dev)
        self.wrong_pad = torch.zeros(n, dtype=torch.bool, device=dev)
        self.wrong_assign = torch.zeros(n, dtype=torch.bool, device=dev)
        self.target_returned = torch.zeros(n, dtype=torch.bool, device=dev)
        self.two_in_slot = torch.zeros(n, dtype=torch.bool, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.arrangement_hold = torch.zeros(n, dtype=torch.int32, device=dev)
        # `evaluate()` runs at t=0 inside reset() and is called out of band by the
        # diagnose scripts; a counter that advanced on every call would be wrong twice.
        self.last_eval_step = torch.full((n,), -1, dtype=torch.int32, device=dev)
        # The props and the agent exist now; the drives are made once per build.
        self._ag = AssistedGrasp(self, self.props) if self.assisted_grasp else None
        return super()._after_reconfigure(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Release any weld BEFORE the props are posed: an enabled drive would pull the
        # freshly spawned prop back to the hand on the first physics step.
        if self._ag is not None:
            self._ag.reset()
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            for i in env_idx.tolist():
                art, j = self._door[int(i)]
                q = art.get_qpos()
                q = q.clone() if torch.is_tensor(q) else torch.as_tensor(q).clone()
                q.reshape(-1)[j] = self.cfg.door_open_rad
                art.set_qpos(q)

            # The ANSWER comes from the episode RNG, never torch: sapien_env.py:948-953
            # only seeds the torch generator when a seed was passed, so a torch draw is
            # not reproducible from a seed — fine for jitter, wrong for the one fact the
            # whole task rests on.
            # The pad draw comes AFTER the row permutation and the jitter, so a seed keeps
            # the row and the start it had before the pads were coloured.
            perms, jit, pads = [], [], []
            for i in env_idx.tolist():
                rng = self._batched_episode_rng[int(i)]
                perms.append(rng.permutation(self.cfg.n_props))
                jit.append(rng.uniform(-1.0, 1.0, size=3))
                pads.append(rng.permutation(4)[: self.cfg.n_props])
            slots = torch.as_tensor(np.stack(perms), dtype=torch.long)
            jx = torch.as_tensor(np.stack(jit), dtype=torch.float32)
            self.original_slots[env_idx] = slots
            pad_slots = torch.as_tensor(np.stack(pads), dtype=torch.long)
            self.pad_slots[env_idx] = pad_slots

            row_x = self._row_centre_x[env_idx] + jx[:, 0] * self.cfg.row_jitter_x

            row_pts = torch.zeros((b, self.cfg.n_props, 3))
            row_pts[:, :, 0] = row_x.unsqueeze(-1)
            row_pts[:, :, 1] = torch.as_tensor(self.cfg.row_slots_y).unsqueeze(0)
            row_pts[:, :, 2] = self.cfg.shelf_top_z
            self._row_points[env_idx] = row_pts

            slot_pts = torch.zeros((b, 4, 3))
            slot_pts[:, :, 0] = row_x.unsqueeze(-1) + torch.as_tensor(self.cfg.slot_x_offsets)
            slot_pts[:, :, 1] = self.cfg.place_across
            slot_pts[:, :, 2] = self._counter_top_z[env_idx].unsqueeze(-1)
            self._slot_points[env_idx] = slot_pts

            # Each pad on its drawn slot, a hair above the surface so it is drawn rather
            # than z-fighting with it.
            for c, pad in enumerate(self.pads):
                mp = slot_pts[torch.arange(b), pad_slots[:, c]].clone()
                mp[:, 2] = mp[:, 2] + PAD_HALF[2] + 0.001
                pad.set_pose(Pose.create_from_pq(p=mp))

            # Each prop onto its drawn row slot, by MESH BOTTOM: a box's origin sits
            # `prop_half_h` above its base, and spawning by origin buries it in the shelf.
            for i, prop in enumerate(self.props):
                p = torch.zeros((b, 3))
                own = row_pts[torch.arange(b), slots[:, i]]
                p[:, :2] = own[:, :2]
                p[:, 2] = self.cfg.shelf_top_z + self.cfg.spawn_clearance + self.cfg.prop_half_h
                prop.set_pose(Pose.create_from_pq(p=p, q=self._yaw_quat(b)))
                prop.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                prop.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            self._restore_robot(env_idx, row_x, jx)

            self.left_slot[env_idx] = False
            self.restored[env_idx] = False
            self.was_lifted[env_idx] = False
            self.staged[env_idx] = False
            self.wrong_pad[env_idx] = False
            self.wrong_assign[env_idx] = False
            self.target_returned[env_idx] = False
            self.two_in_slot[env_idx] = False
            self.succeeded[env_idx] = False
            self.arrangement_hold[env_idx] = 0
            self.last_eval_step[env_idx] = -1

    def _yaw_quat(self, b: int) -> torch.Tensor:
        """Yaw only. The upright test compares the body's +Z against world +Z, which is
        only a statement about "standing" if the spawn never rolls or pitches."""
        yaw = torch.zeros(b)
        return torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b),
                            torch.sin(yaw / 2)], dim=1)

    def _restore_robot(self, env_idx: torch.Tensor, row_x: torch.Tensor, jx: torch.Tensor):
        """Rest keyframe, then the task's own start on the open floor south of the row.

        `scene_builder.initialize` restores the robot only for uid "fetch", and then to
        a dock IT drew (`robot_poses`), not to the task's. No uid test here: this runs
        after it and simply overwrites, which is safe and idempotent.
        """
        keyframe = self.agent.keyframes["rest"]
        qpos = torch.as_tensor(np.asarray(keyframe.qpos, dtype=np.float32))
        self.agent.robot.set_qpos(qpos.unsqueeze(0).repeat(len(env_idx), 1))
        self.agent.robot.set_root_pose(sapien.Pose())
        q = self.agent.robot.get_qpos()
        q[env_idx, 0] = row_x + jx[:, 0] * self.cfg.start_jitter_x
        q[env_idx, 1] = self.cfg.start_y + jx[:, 1] * self.cfg.start_jitter_y
        q[env_idx, 2] = math.radians(self.cfg.start_yaw_deg) + jx[:, 2] * self.cfg.start_jitter_yaw
        self.agent.robot.set_qpos(q[env_idx])
        self.agent.robot.set_qvel(torch.zeros_like(q[env_idx]))

    def _after_simulation_step(self):
        # The rule runs on every physics step, as OmniGibson's does: contacts and rays
        # after the step, the weld switched before the next one.
        if self._ag is not None:
            self._ag.after_substep()

    # --------------------------------------------------------------- evaluate --

    def _prop_state(self):
        """Positions, tilts, grasp flags and settle flags for the four props."""
        pos = torch.stack([p.pose.p for p in self.props], dim=1)                # (N, k, 3)
        rot = torch.stack([p.pose.to_transformation_matrix()[:, :3, :3]
                           for p in self.props], dim=1)                          # (N, k, 3, 3)
        tilt = torch.arccos(rot[:, :, 2, 2].clamp(-1 + 1e-6, 1 - 1e-6))
        grasped = torch.stack([self.agent.is_grasping(p) for p in self.props], dim=1)
        settled = torch.stack(
            [p.is_static(lin_thresh=self.cfg.settle_lin_speed,
                         ang_thresh=self.cfg.settle_ang_speed) for p in self.props], dim=1)
        return pos, tilt, grasped, settled

    def evaluate(self) -> dict:
        """Runs every step, and once inside reset() at t=0 before any action.

        Everything returned is batched, and the intermediates are returned alongside
        `success` rather than only the verdict: they are threaded into `info`, and when
        a run fails they are the only way to say which of eight criteria missed.
        """
        cfg = self.cfg
        pos, tilt, grasped, settled = self._prop_state()
        n, k = self.num_envs, cfg.n_props
        upright = tilt <= math.radians(cfg.upright_tilt_deg)
        free = settled & ~grasped

        # Each prop's OWN row slot point, gathered by the episode's permutation.
        idx = self.original_slots.unsqueeze(-1).expand(-1, -1, 3)
        own_row = torch.gather(self._row_points, 1, idx)                         # (N, k, 3)
        is_target = self.original_slots == (k - 1)                               # deepest

        # Distances from every prop to every row slot and to every counter slot.
        d_row = torch.linalg.norm(
            pos[:, :, None, :2] - self._row_points[:, None, :, :2], dim=-1)      # (N, k, k)
        d_slot = torch.linalg.norm(
            pos[:, :, None, :2] - self._slot_points[:, None, :, :2], dim=-1)     # (N, k, 4)

        shelf_h = torch.abs(pos[:, :, 2] - (cfg.shelf_top_z + cfg.prop_half_h)) <= cfg.upright_tol
        counter_h = torch.abs(
            pos[:, :, 2] - (self._counter_top_z[:, None] + cfg.prop_half_h)) <= cfg.upright_tol
        on_shelf = shelf_h & upright
        on_counter = counter_h & upright

        # --- the latches, all sticky ----------------------------------------
        self.was_lifted = self.was_lifted | grasped
        d_own = torch.linalg.norm(pos[:, :, :2] - own_row[:, :, :2], dim=-1)
        self.left_slot = self.left_slot | (d_own > cfg.leave_tol)

        in_row = (d_row <= cfg.slot_xy_tol) & on_shelf.unsqueeze(-1) & free.unsqueeze(-1)
        own_hot = torch.nn.functional.one_hot(self.original_slots, k).bool()
        # A movable settled and released in a row slot that is not its own IS the
        # committed wrong answer; the episode cannot be won from there.
        wrong_now = (in_row & ~own_hot & ~is_target.unsqueeze(-1)).any(dim=(1, 2))
        self.wrong_assign = self.wrong_assign | wrong_now
        # The target belongs on the counter. Put back, it is a different chore — but
        # only AFTER it has left: it starts in the row, and a latch that reads the
        # starting arrangement as a violation fires at t=0 inside reset(), before the
        # agent has acted. Measured on the first build: `target_returned` was True on
        # every seed at step 0.
        self.target_returned = self.target_returned | (
            (in_row & is_target.unsqueeze(-1) & self.left_slot.unsqueeze(-1))
            .any(dim=(1, 2)))
        # Two props settled in one row slot: the row is not as it was, whatever else.
        self.two_in_slot = self.two_in_slot | (in_row.sum(dim=1) >= 2).any(dim=1)

        # --- the verdict -----------------------------------------------------
        # "Restored" means LEFT and came back, not "never moved": every prop stands in
        # its own slot at t=0, and a flag that is true there says nothing. The verdict
        # already gates on `all_left`; folding it in here keeps the per-prop flag in the
        # trace honest too, which is what a failed run is read from.
        # Leaving a shelf slot alone also admits an in-cabinet sideways nudge.
        # v1 requires every blocker to have stood, released, on the counter.
        # A prop stands ON a pad within `slot_radius` of its centre; the pads are at
        # least two radii apart, so on one pad or none. Its OWN pad is its colour's.
        on_pad = (on_counter & free).unsqueeze(-1) & (d_slot <= cfg.slot_radius)  # (N, k, 4)
        own_pad = torch.nn.functional.one_hot(self.pad_slots, 4).bool()           # (N, k, 4)
        on_slot = on_pad.any(dim=-1)
        on_own = (on_pad & own_pad).any(dim=-1)
        self.wrong_pad = self.wrong_pad | ((on_pad & ~own_pad).any(dim=-1)
                                           & self.was_lifted).any(dim=1)
        self.staged = self.staged | (on_own & self.was_lifted)
        restored = ((in_row & own_hot).any(dim=-1) & ~is_target
                    & self.left_slot & self.staged)
        self.restored = restored
        movable = ~is_target
        all_restored = ((restored | ~movable).all(dim=1))
        all_left = ((self.left_slot | ~movable).all(dim=1))
        target_ok = ((on_own & self.was_lifted) | ~is_target).all(dim=1)

        raw = (all_restored & all_left & target_ok & ~self.wrong_pad
               & ~self.wrong_assign & ~self.target_returned & ~self.two_in_slot)

        # The hold counter, idempotent within a step. It advances only when
        # `elapsed_steps` has actually moved, so a second `evaluate()` inside the same
        # step — reset()'s own call at t=0, a wrapper's, a debug print's — is a no-op.
        step = self.elapsed_steps.to(torch.int32)
        advance = step != self.last_eval_step
        self.arrangement_hold = torch.where(
            advance & raw,
            self.arrangement_hold + 1,
            torch.where(advance, torch.zeros_like(self.arrangement_hold),
                        self.arrangement_hold),
        )
        self.last_eval_step = torch.where(advance, step, self.last_eval_step)
        self.succeeded = self.succeeded | (self.arrangement_hold >= cfg.hold_steps)

        extra = {}
        ag = getattr(self, "_ag", None)
        if ag is not None:
            # Which prop is welded to the hand, -1 for none. Not an observation.
            extra["ag_held"] = torch.full((n,), ag.held_index(), dtype=torch.long,
                                          device=pos.device)
        return {
            **extra,
            # Keep historical completion separate from a valid final arrangement.
            "success": raw & (self.arrangement_hold >= cfg.hold_steps),
            "success_once": self.succeeded,
            "all_restored": all_restored,
            "all_left": all_left,
            "target_on_its_pad": target_ok,
            "target_on_a_slot": ((on_slot & self.was_lifted) | ~is_target).all(dim=1),
            "on_own_pad": on_own,
            "wrong_pad": self.wrong_pad,
            "restored": restored,
            "left_slot": self.left_slot,
            "was_lifted": self.was_lifted,
            "staged": self.staged,
            "wrong_assign": self.wrong_assign,
            "target_returned": self.target_returned,
            "two_in_slot": self.two_in_slot,
            "arrangement_ok": raw,
            "arrangement_hold": self.arrangement_hold,
            "grasped_any": grasped.any(dim=1),
            "tilt_max_deg": torch.rad2deg(tilt.max(dim=1).values),
            "prop_tilt_deg": torch.rad2deg(tilt),
            # Ground truth for the trajectory file and the oracle's sighted arm. `info`
            # is NOT an observation: only what _get_obs_extra copies out reaches a policy.
            "original_slots": self.original_slots,
        }

    # -------------------------------------------------------------------- obs --

    def get_language_instruction(self, **kwargs):
        """One phrasing: the text names the chore, never which prop came from where."""
        return [INSTRUCTIONS[0]] * self.num_envs

    def _get_obs_extra(self, info: dict) -> dict:
        """The memory fence.

        Three rules, each of which a memory task has been lost to before:

        1. `original_slots` — the answer — is NEVER emitted, and neither is any latch
           or which prop is the target.
        2. Prop poses are emitted in INDEX order, which is identity (colour) order, not
           in slot order. Slot order would put the front-origin prop at a fixed slice
           index and hand a state policy the answer for free.
        3. The slot geometry is emitted because it is honest scenery: where the four
           counter points and the four row points ARE says nothing about which prop
           belongs to which. What must be remembered is the association, and no key here
           carries it. The pads (`pad_points`, in prop order: where prop i is to be
           staged) are scenery too: drawn on the counter, independent of the row.
        """
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            base_pose=self.agent.robot.pose.raw_pose,
            grasped_any=info["grasped_any"],
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                prop_poses=torch.stack([p.pose.raw_pose for p in self.props],
                                       dim=1).reshape(self.num_envs, -1),
                row_points=self._row_points.reshape(self.num_envs, -1),
                slot_points=self._slot_points.reshape(self.num_envs, -1),
                pad_points=torch.gather(
                    self._slot_points, 1, self.pad_slots.unsqueeze(-1).expand(-1, -1, 3)
                ).reshape(self.num_envs, -1),
            )
        return obs

    # ------------------------------------------------------------------ state --

    def get_state_dict(self) -> dict:
        """The episode's answer and every latch have to survive a checkpoint.

        For a memory benchmark this is the state that matters most: without it
        `env.set_state(env.get_state())` drops the answer and a recorded success
        replays as a failure.
        """
        state = super().get_state_dict()
        for key in TASK_STATE_KEYS:
            state[key] = getattr(self, key).clone()
        state["row_points"] = self._row_points.reshape(self.num_envs, -1).clone()
        state["slot_points"] = self._slot_points.reshape(self.num_envs, -1).clone()
        if self._ag is not None:
            # Which prop is welded and where: without it a restored checkpoint drops a
            # held prop the moment physics resumes.
            state.update(self._ag.state_dict())
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        """Tolerates a dict with no task keys, because one is routine: `set_state(flat)`
        rebuilds a dict holding only "actors" and "articulations", so every task-added
        key is gone by the time it arrives here. Numpy-tolerant too — a replayed h5
        hands numpy back, and a torch-only restore is the defect all three earlier tasks
        shipped with."""
        super().set_state_dict(state, env_idx)
        for key in TASK_STATE_KEYS:
            if key in state:
                setattr(self, key, restore_task_tensor(getattr(self, key), state[key],
                                                       self.device))
        for key, attr in (("row_points", "_row_points"), ("slot_points", "_slot_points")):
            if key in state:
                flat = restore_task_tensor(
                    getattr(self, attr).reshape(self.num_envs, -1), state[key], self.device)
                setattr(self, attr, flat.reshape(getattr(self, attr).shape))
        if self._ag is not None:
            self._ag.load_state_dict(state)
