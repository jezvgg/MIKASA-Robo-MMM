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
"""

from __future__ import annotations

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


@dataclass
class AssistedGrasp:
    """The rule wired into one CPU ManiSkill environment for a fixed list of objects.

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
        self._obj_ids = {obj._objs[0].per_scene_id: i for i, obj in enumerate(self.objects)}
        self._masses = []
        for obj in self.objects:
            body = obj._objs[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            self._masses.append(float(body.mass) if body is not None else float("inf"))
        # One drive per object, hand -> object, made now and switched later.
        self._drives = []
        for obj in self.objects:
            drive = env.scene.create_drive(self.hand, sapien.Pose(), obj, sapien.Pose())
            component = drive._objs[0]
            for set_limit in (component.set_limit_x, component.set_limit_y, component.set_limit_z):
                set_limit(0, 0)
            component.set_limit_twist(0, 0)
            # The swing axes: a cone limit does not lock them on a drive, a stiff slerp
            # drive to the identity does.
            component.set_drive_property_slerp(self.cfg.weld_stiffness, self.cfg.weld_damping)
            component.set_drive_target(sapien.Pose())
            component.disable()
            self._drives.append(component)
        self.rel_pose = np.zeros(7, dtype=np.float32)
        self.events: list = []

    # ------------------------------------------------------------------ state --

    def reset(self) -> None:
        for component in self._drives:
            component.disable()
        self.window.reset()
        self.rel_pose[:] = 0.0
        self.events = []

    def held_index(self) -> int:
        return self.window.held if self.window.attached else -1

    def state_dict(self) -> dict:
        import torch

        return {
            "ag_window": torch.tensor([self.window.state()], dtype=torch.int32),
            "ag_rel_pose": torch.as_tensor(self.rel_pose).reshape(1, 7).clone(),
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
        rel = self.hand.pose.sp.inv() * self.objects[i].pose.sp
        self.rel_pose[:3], self.rel_pose[3:] = rel.p, rel.q
        self._weld(i, rel)

    def _weld(self, i: int, rel) -> None:
        import sapien

        component = self._drives[i]
        component.set_pose_in_parent(rel)
        component.set_pose_in_child(sapien.Pose())
        component.enable()
