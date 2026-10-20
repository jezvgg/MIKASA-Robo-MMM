"""Assisted grasp: OmniGibson's attach rule, held by a PhysX drive. Off unless a task asks.

BEHAVIOR's simulator (OmniGibson, ``grasping_mode: assisted``) joins an object to the hand
when the closing fingers really hold it, and its own evaluator runs that way; the
BIMBO_EvoX port reproduced the rule tick for tick (``skills/assisted-grasp``). The rule,
as both implement it:

* **applying** — the gripper is commanded anywhere below fully open;
* **candidate** — among the objects touched by a finger in this physics step AND hit
  first by one of the four rays between the finger pads, the one nearest the TCP; it
  must then be touched by BOTH fingers. A closed hand pushing an object from outside
  touches it without a ray hit and never attaches — BIMBO's emulation without the ray
  test grabbed books on every push and dropped them on opening;
* **window** — the same candidate for ``window_s`` (0.3 s) of physics steps, then the
  joint is made at the CURRENT relative pose (no snap);
* **release** — the moment the gripper stops applying; a new grasp is refused for
  ``release_s`` (1/30 s).

Here the joint is a ``PhysxDriveComponent`` created once per object when the task is
built and toggled with ``enable``/``disable``: its x/y/z and twist limits at zero lock
the position and one axis of rotation, and a stiff slerp drive to the identity locks the
other two (a zero cone limit does NOT: with it alone a hand turned 76.6 deg and the
welded prop did not turn at all). Measured on mani_skill 3.0.0b22, CPU: a welded prop
follows the hand in position and orientation and falls when the drive is disabled
(``test_assisted_grasp.py``). CPU simulation only (one env): under GPU
simulation the drives would have to exist before ``gpu_init`` and toggling them there is
untested, so the constructor refuses.

The rule changes the physics of a task — a weld holds what friction would let slip — so
a task that enables it has to say so wherever its data or scores are reported.

`AssistedGraspTask` wires it into a task with one base class and one method.
"""

# Ported without behavioral changes from integration/four-tasks:
# jezvgg/MIKASA-Robo-MMM@5639c2b1b4981979bec1bc68c3170258944e283a
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AssistedGraspConfig:
    window_s: float = 0.3
    "Continuous closing on the same candidate before the weld; OmniGibson's value."
    release_s: float = 1.0 / 30.0
    "Refractory time after a release, before a new grasp may form; OmniGibson's value."
    min_force: float = 1e-3
    "Newtons. A finger counts as touching above this contact force (any real contact)."
    pad_offsets: tuple = (-0.01, 0.01)
    """Metres along the hand's approach axis: two points per finger pad, four rays from
    the first finger's points to the second's. BIMBO measured its pad points on native;
    here they are placed symmetrically about the finger origins, which on Fetch sit on
    the pads (ManiSkill's TCP is their midpoint)."""
    open_tol: float = 1e-4
    "Gripper target within this of fully open counts as open (not applying)."
    mass_limit: float = 10.0
    "Kilograms. Heavier bodies are never attached (OmniGibson's limit)."
    ray_skip: float = 0.002
    """Metres a ray advances past a robot shape it hit. It starts inside a pad: PhysX
    reports that pad at distance 0 until the ray leaves it, and on ds_fetch the finger
    origins sit ~1.5 cm outside the faces of a gripped 4.5 cm prop (measured: pads
    7.6 cm apart), so the first ~1.5 cm of every ray is finger."""
    max_ray_skips: int = 64
    "Upper bound on robot shapes skipped per ray (64 x 2 mm covers the 10 cm opening)."
    weld_stiffness: float = 1e6
    "N m/rad of the slerp drive that holds the welded orientation."
    weld_damping: float = 1e4
    "N m s/rad of the same drive."


class GraspWindow:
    """The rule's state machine for ONE environment, without any simulator.

    ``tick(applying, candidate)`` once per physics step; returns ``("establish", i)``,
    ``("release", i)`` or ``None``. ``held`` stays set through the refractory time after a
    release, which is what blocks an immediate regrasp; ``attached`` is True only while
    the weld itself is on.
    """

    def __init__(self, dt: float, window_s: float = 0.3, release_s: float = 1.0 / 30.0):
        assert dt > 0 and window_s > 0 and release_s >= 0
        self.dt, self.window_s, self.release_s = float(dt), float(window_s), float(release_s)
        self.reset()

    def reset(self) -> None:
        self.held: Optional[int] = None
        self.grasp_counter: Optional[int] = None
        self.release_counter: Optional[int] = None
        self._candidate: Optional[int] = None

    @property
    def attached(self) -> bool:
        return self.held is not None and self.release_counter is None

    def tick(self, applying: bool, candidate: Optional[int]):
        if self.held is not None:
            if self.release_counter is not None:
                self.release_counter += 1
                if self.release_counter * self.dt >= self.release_s - 1e-12:
                    self.held, self.release_counter = None, None
                return None
            if not applying:
                self.release_counter = 0
                return ("release", self.held)
            return None
        if not applying:
            self.grasp_counter = None
            return None
        if self.grasp_counter is None:
            if candidate is not None:
                self.grasp_counter = 0
                self._candidate = candidate
            return None
        if candidate is None or candidate != self._candidate:
            # Lost, or a different body now: the window starts over (OmniGibson resets on
            # loss; a switch counts as loss of the first).
            self.grasp_counter = None
            return None
        self.grasp_counter += 1
        if self.grasp_counter * self.dt >= self.window_s - 1e-12:
            self.grasp_counter = None
            self.held = candidate
            return ("establish", candidate)
        return None

    # A checkpoint holds three integers; -1 stands for None.
    def state(self) -> tuple:
        none = lambda v: -1 if v is None else int(v)  # noqa: E731
        return (none(self.held), none(self.grasp_counter), none(self.release_counter),
                none(self._candidate))

    def load(self, state: Sequence[int]) -> None:
        held, grasp, release, cand = (int(v) for v in state)
        some = lambda v: None if v < 0 else v  # noqa: E731
        self.held, self.grasp_counter, self.release_counter = some(held), some(grasp), some(release)
        self._candidate = some(cand)


def pick_candidate(touch: Sequence[int], ray_hit: Sequence[bool], dist: Sequence[float],
                   min_fingers: int = 2) -> Optional[int]:
    """Index of the body the rule would attach, or None.

    ``touch[i]`` is how many fingers touch body i, ``ray_hit[i]`` whether a pad ray hit it
    first, ``dist[i]`` its distance from the TCP. The NEAREST body touched by any finger
    and hit by a ray is chosen BEFORE the two-finger test — OmniGibson's order, so a nearer
    one-finger body blocks a farther two-finger one (BIMBO kept the quirk on purpose).
    """
    eligible = [i for i in range(len(touch)) if touch[i] > 0 and ray_hit[i]]
    if not eligible:
        return None
    i = min(eligible, key=lambda k: dist[k])
    return i if touch[i] >= min_fingers else None


#: Rotation range of a link's hold, rad: as free as PhysX allows (twist below 2 pi,
#: cone below pi).
LINK_TWIST = 3.1
LINK_CONE = 3.0


def _entity_id(obj) -> int:
    """Scene id of an actor's entity, or of the entity a link's component lives on."""
    o = obj._objs[0]
    return (o.entity if hasattr(o, "entity") else o).per_scene_id


def _is_link(obj) -> bool:
    return hasattr(obj._objs[0], "entity")


def _pose(p):
    import sapien

    return sapien.Pose(np.asarray(p, dtype=np.float64))


@dataclass
class AssistedGrasp:
    """The rule wired into one CPU ManiSkill environment for a fixed list of objects.

    An object is an actor (a free body) or an articulation link (a door, a drawer: what
    its handle is part of). A free body is welded rigidly at its current pose. A link is
    held at the grasp point with its rotation free, OmniGibson's spherical joint for a
    link under a revolute or prismatic joint: the handle turns in the hand as the door
    swings, as it would between real pads (a rigid weld locked a CabinetSearch door at
    0.26 rad of the 1.2 its pull needed, measured 2026-10-16). A link stays held by its own
    joint, so the mass limit, which is about what the hand can carry, applies to free
    bodies only.

    Build it in the task's ``_after_reconfigure`` (the agent and the objects exist),
    ``reset()`` it at the top of ``_initialize_episode``, call ``after_substep()`` from
    ``_after_simulation_step``. ``events`` keeps ``(step, kind, object name)`` for logs.
    """

    env: object
    objects: Sequence
    cfg: AssistedGraspConfig = field(default_factory=AssistedGraspConfig)

    def __post_init__(self):
        import sapien

        env = self.env
        if getattr(env, "gpu_sim_enabled", False) or env.num_envs != 1:
            raise NotImplementedError(
                "assisted grasp is implemented for CPU simulation with one env only")
        agent = env.agent
        self.hand = agent.robot.links_map["gripper_link"]
        self.fingers = (agent.finger1_link, agent.finger2_link)
        self.window = GraspWindow(1.0 / float(env.sim_freq), self.cfg.window_s, self.cfg.release_s)
        self._px = env.scene.sub_scenes[0].physx_system
        self._robot_ids = {link._objs[0].entity.per_scene_id for link in agent.robot.get_links()}
        self._obj_ids = {_entity_id(obj): i for i, obj in enumerate(self.objects)}
        self._masses = []
        for obj in self.objects:
            if _is_link(obj):
                self._masses.append(0.0)          # held by its joint; never refused for mass
                continue
            body = obj._objs[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            self._masses.append(float(body.mass) if body is not None else float("inf"))
        # One drive per object, hand -> object, made now and switched later.
        self._drives = []
        for obj in self.objects:
            drive = env.scene.create_drive(self.hand, sapien.Pose(), obj, sapien.Pose())
            component = drive._objs[0]
            for set_limit in (component.set_limit_x, component.set_limit_y, component.set_limit_z):
                set_limit(0, 0)
            if not _is_link(obj):
                component.set_limit_twist(0, 0)
                # The swing axes: a cone limit does not lock them on a drive, a stiff
                # slerp drive to the identity does.
                component.set_drive_property_slerp(self.cfg.weld_stiffness, self.cfg.weld_damping)
            else:
                # Free rotation for a link. NOT the default: a fresh drive limits twist to
                # +-90 deg and the cone to 90 deg (SAPIEN 3.0.3), and a CabinetSearch door
                # pulled ~100 deg open hit that stop and dragged the hand round with it
                # (measured 2026-10-16: the hand ended facing the leaf at -0.77 instead of
                # -0.16, and the retreat started in collision).
                component.set_limit_twist(-LINK_TWIST, LINK_TWIST)
                component.set_limit_cone(LINK_CONE, LINK_CONE)
            component.set_drive_target(sapien.Pose())
            component.disable()
            self._drives.append(component)
        self.rel_pose = np.zeros(7, dtype=np.float32)
        self.anchor = np.zeros(3, dtype=np.float32)   # grasp point in the hand frame (links)
        self.events: list = []

    # ------------------------------------------------------------------ state --

    def reset(self) -> None:
        for component in self._drives:
            component.disable()
        self.window.reset()
        self.rel_pose[:] = 0.0
        self.anchor[:] = 0.0
        self.events = []

    def held_index(self) -> int:
        return self.window.held if self.window.attached else -1

    def state_dict(self) -> dict:
        import torch

        return {
            "ag_window": torch.tensor([self.window.state()], dtype=torch.int32),
            "ag_rel_pose": torch.as_tensor(self.rel_pose).reshape(1, 7).clone(),
            "ag_anchor": torch.as_tensor(self.anchor).reshape(1, 3).clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        import sapien

        if "ag_window" not in state:
            # A state saved without the rule (e.g. `env.set_state` of actors and
            # articulations only): no weld survives it, or the old one would yank the
            # restored prop to the hand on the next step.
            for component in self._drives:
                component.disable()
            self.window.reset()
            self.rel_pose[:] = 0.0
            return
        window = np.asarray(state["ag_window"]).reshape(-1)[:4]
        self.window.load(window)
        self.rel_pose[:] = np.asarray(state["ag_rel_pose"], dtype=np.float32).reshape(-1)[:7]
        if "ag_anchor" in state:
            self.anchor[:] = np.asarray(state["ag_anchor"], dtype=np.float32).reshape(-1)[:3]
        for component in self._drives:
            component.disable()
        if self.window.attached:
            self._weld(self.window.held, sapien.Pose(self.rel_pose[:3], self.rel_pose[3:]))

    # -------------------------------------------------------------- the rule --

    def _applying(self) -> bool:
        ctrl = self.env.agent.controller.controllers["gripper"]
        target = getattr(ctrl, "_target_qpos", None)
        if target is None:
            return False
        upper = float(np.max(np.asarray(ctrl.config.upper)))
        return bool(float(target.min()) < upper - self.cfg.open_tol)

    def _touch_counts(self) -> list:
        scene = self.env.scene
        counts = []
        for obj in self.objects:
            n = 0
            for finger in self.fingers:
                force = scene.get_pairwise_contact_forces(finger, obj)
                n += int(float(force.norm(dim=-1).max()) > self.cfg.min_force)
            counts.append(n)
        return counts

    def _first_object_hit(self, a: np.ndarray, b: np.ndarray) -> Optional[int]:
        """Object index hit first along a -> b, passing through robot shapes."""
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            return None
        u = (seg / length).astype(np.float32)
        start, travelled = a.astype(np.float32), 0.0
        for _ in range(self.cfg.max_ray_skips):
            hit = self._px.raycast(start, u, length - travelled)
            if hit is None:
                return None
            entity_id = hit.component.entity.per_scene_id
            if entity_id in self._robot_ids:
                step = max(float(hit.distance), 0.0) + self.cfg.ray_skip
                start, travelled = (start + u * step).astype(np.float32), travelled + step
                if travelled >= length:
                    return None
                continue
            return self._obj_ids.get(entity_id)   # None for a shelf, a wall, ...
        return None

    def _ray_hits(self) -> list:
        hand = self.hand.pose.sp
        approach = hand.to_transformation_matrix()[:3, 0]
        pads = [np.asarray(f.pose.sp.p, dtype=np.float64) for f in self.fingers]
        hits = [False] * len(self.objects)
        for o1 in self.cfg.pad_offsets:
            for o2 in self.cfg.pad_offsets:
                i = self._first_object_hit(pads[0] + o1 * approach, pads[1] + o2 * approach)
                if i is not None:
                    hits[i] = True
        return hits

    def candidate(self) -> Optional[int]:
        touch = self._touch_counts()
        if not any(touch):
            return None
        tcp = (np.asarray(self.fingers[0].pose.sp.p) + np.asarray(self.fingers[1].pose.sp.p)) / 2
        dist = [float(np.linalg.norm(np.asarray(obj.pose.sp.p) - tcp)) for obj in self.objects]
        i = pick_candidate(touch, self._ray_hits(), dist)
        if i is not None and self._masses[i] > self.cfg.mass_limit:
            return None
        return i

    def after_substep(self) -> None:
        applying = self._applying()
        idle = self.window.held is None and applying
        event = self.window.tick(applying, self.candidate() if idle else None)
        if event is None:
            return
        kind, i = event
        if kind == "establish":
            self._attach_here(i)
        else:
            self._drives[i].disable()
        step = int(np.asarray(self.env.elapsed_steps).reshape(-1)[0])
        self.events.append((step, kind, self.objects[i].name))
        if os.environ.get("MIKASA_AG_LOG") == "1":
            print(f"[assisted_grasp] {kind} {self.objects[i].name} step={step}", flush=True)

    def attach(self, i: int) -> None:
        """Weld object ``i`` where it is now, bypassing the rule (tests, scripted setups).

        The rule still governs the release: with the gripper commanded open, the next
        physics step lets go."""
        for component in self._drives:
            component.disable()
        self.window.reset()
        self.window.held = int(i)
        self._attach_here(int(i))

    def _attach_here(self, i: int) -> None:
        hand = self.hand.pose.sp
        rel = hand.inv() * self.objects[i].pose.sp
        self.rel_pose[:3], self.rel_pose[3:] = rel.p, rel.q
        tcp = (np.asarray(self.fingers[0].pose.sp.p) + np.asarray(self.fingers[1].pose.sp.p)) / 2
        self.anchor[:] = (hand.inv() * _pose(tcp)).p
        self._weld(i, rel)

    def _weld(self, i: int, rel) -> None:
        """Free body: the object frame locked at ``rel`` in the hand. Link: the grasp point
        (``anchor`` in the hand frame) locked to the same material point of the link,
        rotation free."""
        import sapien

        component = self._drives[i]
        if _is_link(self.objects[i]):
            at = sapien.Pose(np.asarray(self.anchor, dtype=np.float64))
            component.set_pose_in_parent(at)
            component.set_pose_in_child(rel.inv() * at)
        else:
            component.set_pose_in_parent(rel)
            component.set_pose_in_child(sapien.Pose())
        component.enable()


class AssistedGraspTask:
    """Mixin for a task env: ``gym.make(..., assisted_grasp=True)`` welds by the rule.

    Put it before ``BaseEnv`` in the bases and define ``_assisted_grasp_objects()`` (the
    actors the robot picks up). Off by default, and then it adds nothing: no drives, no
    info key, no state key — a recorded dataset of the task is unchanged. When on:

    * the rule is built after every reconfigure (the agent and the objects exist);
    * every weld is released at the top of ``reset`` (an enabled drive would pull a
      freshly posed object back to the hand on the first physics step);
    * the rule runs after every physics step, as OmniGibson's does;
    * ``info["ag_held"]`` is the welded object's index or -1 (not an observation);
    * the weld survives ``get_state_dict``/``set_state_dict``.
    """

    def __init__(self, *args, assisted_grasp: bool = False, **kwargs):
        self.assisted_grasp = bool(assisted_grasp)
        self._ag = None
        super().__init__(*args, **kwargs)

    def _assisted_grasp_objects(self) -> Sequence:
        raise NotImplementedError("a task with assisted grasp lists its graspable actors")

    def _after_reconfigure(self, options: dict):
        out = super()._after_reconfigure(options)
        self._ag = AssistedGrasp(self, list(self._assisted_grasp_objects())) if self.assisted_grasp else None
        return out

    def reset(self, *args, **kwargs):
        if getattr(self, "_ag", None) is not None:
            self._ag.reset()
        return super().reset(*args, **kwargs)

    def _after_simulation_step(self):
        super()._after_simulation_step()
        if self._ag is not None:
            self._ag.after_substep()

    def get_info(self, *args, **kwargs):
        info = super().get_info(*args, **kwargs)
        if self._ag is not None:
            import torch

            info["ag_held"] = torch.full((self.num_envs,), self._ag.held_index(), dtype=torch.long,
                                         device=self.device)
        return info

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        if self._ag is not None:
            state.update(self._ag.state_dict())
        return state

    def set_state_dict(self, state: dict, env_idx=None):
        super().set_state_dict(state, env_idx)
        if self._ag is not None:
            self._ag.load_state_dict(state)
