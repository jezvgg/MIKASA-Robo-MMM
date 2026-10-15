"""What the Depth Recall v1 planner believes about the props when it may not read them.

``--perception pointcloud`` swaps every prop pose the planner acts on for an estimate from
the robot's own cameras (`utils/mikasa/point_cloud.py`), the way BIMBO_EvoX's
observation-only planners work:

* **look, then act from where you stand.** The base cameras give a coarse fit of the row
  from the dock; the wrist camera re-fits the one prop at the stand the grasp is made
  from (b1k and BIMBO both re-measure at the action stand);
* **the hand-to-object transform is frozen at the grasp** and the held prop's pose is the
  hand's pose times it (b1k's carry rule) — no simulator read while carrying;
* **memory comes from perception**: which prop stood in which row slot is the slot nearest
  to where the prop was SEEN before it was taken, not the episode's answer.

* **the map is static, the row is measured**: row and counter slot points come from
  the scene's static geometry (the cabinet's row centre, the configured slot y's and
  offsets, shelf and counter heights — BIMBO's rules allowed an offline map). The
  episode shifts the whole row sideways (`row_jitter_x`, up to 4 cm); that shift is not
  in an image observation, so the first look from the dock measures it
  (`calibrate_row`) and every row and counter point after that carries the measured x.

* **the planning world is the belief's** (`bind_world`): mplib's collision world keeps
  the static scene (the map) and the robot, but every prop in it stands where perception
  put it — a prop not seen yet is parked out of reach — and a held prop rides on the
  hand at the transform frozen at the grasp, never re-read from the simulator. This is
  BIMBO_EvoX's fair rule: an offline map of static geometry plus the objects as seen;
* **holding is felt, not read**: `fingers_holding` is the finger encoders — both fingers
  stopped apart while the gripper is commanded closed — not contact forces.

What is still read from the simulator: the static scene map above (fixtures, their
footprints and heights, the props' known size). Truth also goes into the LOGS
(`pc.audit`, the row shift's error) and nowhere else.
"""

from __future__ import annotations

import numpy as np
import sapien

from utils.mikasa import point_cloud as pc

BASE_CAMS = ("left_base_camera_link", "right_base_camera_link")
HAND_CAM = "fetch_hand"


class _BeliefPose:
    """Just enough of a ManiSkill pose for the planner's reads (`.p`, `.sp`)."""

    def __init__(self, pose: sapien.Pose):
        self.sp = pose
        self.p = np.asarray(pose.p, dtype=np.float64).reshape(1, 3)


class PerceivedProp:
    """Stands in for a prop actor wherever the planner computes a pose from it."""

    def __init__(self, belief: "PropBelief", index: int, name: str):
        self._belief, self.index, self.name = belief, index, name

    @property
    def pose(self) -> _BeliefPose:
        return _BeliefPose(self._belief.pose(self.index))

    def get_first_collision_mesh(self, to_world_frame: bool = True):
        import trimesh

        half = self._belief.half
        T = self._belief.pose(self.index).to_transformation_matrix() if to_world_frame else np.eye(4)
        return trimesh.creation.box(extents=2 * np.asarray(half), transform=T)


class PropBelief:
    """Estimated prop poses, the wrist-camera re-fits, the carry transform and the memory."""

    def __init__(self, task, say=None, fit_cfg: pc.FitConfig = pc.FitConfig()):
        from my_scenes.depth_recall_v1 import PROP_COLORS

        self.task, self.cfg, self.fit_cfg = task, task.cfg, fit_cfg
        self.say = say or (lambda message, **data: None)
        self.colors = PROP_COLORS
        missing = [uid for uid in BASE_CAMS + (HAND_CAM,) if uid not in task._sensors]
        if missing:
            raise RuntimeError(f"pointcloud perception needs the cameras {missing}; the env "
                               "was built without them (a render backend is required)")
        self.half = (task.cfg.prop_half_xy, task.cfg.prop_half_xy, task.cfg.prop_half_h)
        self.model = pc.box_model(self.half, n=fit_cfg.model_points)
        self.cutter = pc.RobotCutter(task.agent.robot)
        self.est: dict = {}          # prop index -> dict(x, y, z, yaw, where, fit metrics)
        self.row_slot: dict = {}     # prop index -> row slot it was SEEN in before it moved
        self.held = None             # (prop index, hand->prop sapien.Pose)
        self.search_radius = 0.10    # metres around a stand point or the last estimate
        self.looks = []              # per look: where, cams, per-prop outcome + audit
        self.row_dx = None           # the row's measured sideways shift, once calibrated
        self.slot_pts = None         # counter slot points the planner uses (map + row_dx)

    # --------------------------------------------------------------- regions --

    def _region(self, where: str):
        """Footprint box and height band of a support, from the static scene map."""
        if where == "shelf":
            fx = self.task.cabinets[0]
            z0 = float(self.cfg.shelf_top_z)
        elif where == "counter":
            fx = self.task.counters[0]
            z0 = float(np.asarray(self.task._counter_top_z).reshape(-1)[0])
        else:
            raise ValueError(where)
        pos, size = np.asarray(fx.pos, dtype=np.float64), np.asarray(fx.size, dtype=np.float64)
        lo, hi = pos[:2] - size[:2] / 2, pos[:2] + size[:2] / 2
        return lo, hi, z0

    def map_points(self):
        """(row points (n, 3), counter slot points (4, 3)) from the STATIC map only.

        The row centre x is the cabinet's exposed half (fixed per scene); the episode's
        sideways shift is not in it, `calibrate_row` measures it.
        """
        cfg = self.cfg
        x0 = float(np.asarray(self.task._row_centre_x).reshape(-1)[0])
        zc = float(np.asarray(self.task._counter_top_z).reshape(-1)[0])
        row = np.array([[x0, y, float(cfg.shelf_top_z)] for y in cfg.row_slots_y])
        slots = np.array([[x0 + o, float(cfg.place_across), zc] for o in cfg.slot_x_offsets])
        return row, slots

    def calibrate_row(self, fits: dict, row_pts, slot_pts, max_shift: float = 0.06):
        """Shift the map's row and slot points by the row's measured sideways offset.

        The offset is the median over props of (estimated x - map x): accepted fits when
        there are any, else the coarse (cluster-x) estimates. Nothing seen: no shift.
        Returns the shifted (row_pts, slot_pts).
        """
        x0 = float(np.asarray(row_pts)[0, 0])
        good = [f["x"] for f in fits.values() if f.get("accepted")]
        coarse = [self.est[i]["x"] for i, f in fits.items() if "coarse_xy" in f and i in self.est]
        xs, source = (good, "fits") if good else (coarse, "coarse") if coarse else ([], "none")
        dx = float(np.clip(np.median(xs) - x0, -max_shift, max_shift)) if xs else 0.0
        self.row_dx = dx
        row, slots = np.array(row_pts, dtype=np.float64), np.array(slot_pts, dtype=np.float64)
        row[:, 0] += dx
        slots[:, 0] += dx
        self.slot_pts = slots
        # Truth, for the LOG only: how far the measured row x is from the episode's.
        true_x = float(np.asarray(self.task._row_points).reshape(-1, 3)[0, 0])
        self.say("row calibrated", dx=round(dx, 4), source=source, n=len(xs),
                 err_mm=round((x0 + dx - true_x) * 1000, 1))
        return row, slots

    # ------------------------------------------------------------------ look --

    def look(self, where: str, cams, only=None, anchors=None, min_seen: int = 5,
             trust_anchor_x: bool = True) -> dict:
        """Capture, cut, band, cluster by colour, fit; update the estimates that pass.

        `only` restricts the fit to these prop indices (a wrist re-fit of one prop).
        `anchors` are the support's slot points (the row or the counter slots, from the
        static map shifted by the measured row offset): a colour cluster of at least
        `min_seen` points is tied to the anchor nearest its centroid, ICP starts there,
        and when the fit is refused the anchor itself is kept as a COARSE estimate. Coarse
        is enough to choose and approach a prop; a grasp needs an accepted fit.
        Without anchors, ICP starts from the previous estimate or the centroid.
        ``trust_anchor_x=False`` (the shelf row, whose x the episode shifts): ICP starts
        at the cluster's x and the anchor's y, and the coarse estimate is that point —
        the cameras look along y, so the seen face's x is the prop's x.
        Returns {index: fit dict} for every prop tried; only a fit in THIS look is fresh.
        """
        cloud = pc.capture(self.task, cams)
        extra = []
        if self.held is not None:
            T = self.pose(self.held[0]).to_transformation_matrix()
            extra.append((T, self.half))
        keep = self.cutter.keep(cloud.points, extra)
        lo, hi, z0 = self._region(where)
        h = 2 * self.cfg.prop_half_h
        in_band = keep & pc.band(cloud.points, lo, hi, z0 + 0.01, z0 + h + 0.02)
        pts, rgb = cloud.points[in_band], cloud.rgb[in_band]
        labels = pc.hue_labels(rgb, self.colors)
        fits = {}
        idxs = range(len(self.colors)) if only is None else only
        for i in idxs:
            if self.held is not None and self.held[0] == i:
                continue
            mine = pts[labels == i]
            # Only near where a prop can stand: the support's stand points, or the last
            # estimate. The counter band also holds the brick backsplash, whose hue is
            # red enough to pass for prop_0 (measured: a 362 mm "fit" on the wall).
            if anchors is not None:
                centres = np.asarray(anchors, dtype=np.float64)[:, :2]
            elif i in self.est:
                centres = np.array([[self.est[i]["x"], self.est[i]["y"]]])
            else:
                centres = None
            if centres is not None and len(mine):
                d = np.linalg.norm(mine[:, None, :2] - centres[None], axis=2).min(axis=1)
                mine = mine[d <= self.search_radius]
            # 2 cm cells keep a sparse far view in one piece; the row pitch is 7 cm.
            groups = pc.voxel_groups(mine, cell=0.02)
            if not groups or len(groups[0]) < min_seen:
                fits[i] = dict(accepted=False, reasons=["not seen"], n=len(groups[0]) if groups else 0)
                continue
            seen = mine[groups[0]]
            z = z0 + self.cfg.prop_half_h
            init = None
            anchor = None
            if anchors is not None:
                a = np.asarray(anchors, dtype=np.float64)[:, :2]
                c = seen[:, :2].mean(axis=0)
                anchor = a[int(np.argmin(np.abs(a[:, 1] - c[1]) if not trust_anchor_x
                                         else np.linalg.norm(a - c, axis=1)))].copy()
                if not trust_anchor_x:
                    anchor[0] = c[0]
                init = anchor
            elif i in self.est:
                init = (self.est[i]["x"], self.est[i]["y"])
            fit = pc.fit_planar(seen, self.model, z, cloud.views, self.fit_cfg, init_xy=init)
            fits[i] = fit
            if fit["accepted"]:
                self.est[i] = dict(fit, where=where, coarse=False)
            elif anchor is not None:
                self.est[i] = dict(x=float(anchor[0]), y=float(anchor[1]), z=z, yaw=0.0,
                                   where=where, coarse=True, n=fit.get("n", len(seen)))
                fit["coarse_xy"] = [round(float(v), 3) for v in anchor]
        self._log(where, cams, fits)
        return fits

    def _log(self, where, cams, fits):
        out = {}
        for i, fit in fits.items():
            entry = dict(n=fit.get("n", 0), ok=bool(fit.get("accepted")))
            if "rms" in fit:
                entry.update(rms_mm=round(fit["rms"] * 1000, 1), inlier=round(fit["inlier"], 2),
                             through=round(fit["see_through"], 2))
                # Truth, for the LOG only: how far the estimate is from the prop.
                entry.update(pc.audit(fit, self.task.props[i].pose.sp))
            if not fit.get("accepted"):
                entry["why"] = "; ".join(fit.get("reasons", []))
            if "coarse_xy" in fit:
                entry["coarse_xy"] = fit["coarse_xy"]
            out[self.task.props[i].name] = entry
        self.looks.append(dict(where=where, cams=list(cams), props=out))
        self.say("perceive", where=where, cams=list(cams), props=out)

    # ----------------------------------------------------------------- reads --

    def has(self, i: int, fine: bool = False) -> bool:
        """Known at all, or (``fine``) known from an accepted fit — not a coarse cluster
        point and not dead reckoning from a release. A fit is not FRESH for that: a
        caller that needs one this moment checks what its own `look` returned."""
        if self.held is not None and self.held[0] == i:
            return True
        e = self.est.get(i)
        return e is not None and not (fine and (e.get("coarse") or e.get("from_carry")))

    def pose(self, i: int) -> sapien.Pose:
        if self.held is not None and self.held[0] == i:
            return self.task.agent.tcp.pose.sp * self.held[1]
        e = self.est[i]
        return sapien.Pose([e["x"], e["y"], e["z"]], _yaw_quat(e["yaw"]))

    def proxy(self, prop) -> PerceivedProp:
        return PerceivedProp(self, list(self.task.props).index(prop), prop.name)

    # ------------------------------------------------------ carry and memory --

    def attach(self, i: int) -> None:
        """Freeze hand->prop at the grasp, from the estimate the grasp was aimed at."""
        self.held = (i, self.task.agent.tcp.pose.sp.inv() * self.pose(i))

    def release(self, i: int, where: str) -> None:
        if self.held is not None and self.held[0] == i:
            p = self.pose(i)
            yaw = float(np.arctan2(2 * (p.q[0] * p.q[3] + p.q[1] * p.q[2]),
                                   1 - 2 * (p.q[2] ** 2 + p.q[3] ** 2)))
            self.est[i] = dict(x=float(p.p[0]), y=float(p.p[1]), z=float(p.p[2]), yaw=yaw,
                               where=where, from_carry=True)
            self.held = None

    # ------------------------------------------------------ planning world --

    #: Where an unseen prop waits in the planning world: far below the floor.
    PARKED = sapien.Pose([0.0, 0.0, -10.0])

    def bind_world(self, solver) -> None:
        """Make the solver's mplib planning world follow the belief, not the simulator.

        mplib re-reads every free actor's pose from the simulator on each sync; after
        each sync the props are put back where the belief has them (unseen: parked out of
        reach), and an attached (held) prop keeps the transform it was attached with
        instead of the simulator's.
        """
        from robots.fetch.utils import FCLObject, convert_object_name

        world = solver.planner.planning_world
        original = world.update_from_simulation
        names = [convert_object_name(p._objs[0]) for p in self.task.props]

        def sync(*, update_attached_object: bool = True):
            original(update_attached_object=False)
            for i, name in enumerate(names):
                if world.get_attached_object(name):
                    continue
                obj = world.get_object(name)
                if obj is None:          # taken out for a touchable stroke
                    continue
                pose = self.pose(i) if self.has(i) else self.PARKED
                world.add_object(FCLObject(name, pose, obj.shapes, obj.shape_poses))

        world.update_from_simulation = sync
        self.world_names = names
        sync()

    def link_to_prop(self, i: int):
        """gripper_link -> prop from the robot's own link pose and the belief (an mplib.Pose)."""
        import mplib

        link = next(l for l in self.task.agent.robot.get_links() if l.name.endswith("gripper_link"))
        rel = link.pose.sp.inv() * self.pose(i)
        return mplib.Pose(rel.p, rel.q)

    def fingers_holding(self, lo: float = 0.004, hi: float = 0.048) -> bool:
        """The gripper is commanded shut and both finger joints stopped apart (an object
        between them) — the command and the encoders, not contact forces. Shut on
        nothing the fingers reach 0; open they sit at 0.05; the rest pose's 0.015 is not
        a grasp because nothing commanded the close."""
        ctrl = self.task.agent.controller.controllers["gripper"]
        target = getattr(ctrl, "_target_qpos", None)
        if target is None or float(np.asarray(target.cpu() if hasattr(target, "cpu") else target).min()) \
                > float(np.min(np.asarray(ctrl.config.lower))) + 0.005:
            return False
        names = [j.name for j in self.task.agent.robot.get_active_joints()]
        q = np.asarray(self.task.agent.robot.get_qpos().cpu().numpy()).reshape(-1)
        gaps = [float(q[names.index(n)]) for n in ("r_gripper_finger_joint", "l_gripper_finger_joint")]
        return all(lo < g < hi for g in gaps)

    def remember_row_slot(self, i: int, row_pts) -> int:
        e = self.est[i]
        d = np.linalg.norm(np.asarray(row_pts)[:, :2] - [e["x"], e["y"]], axis=1)
        self.row_slot[i] = int(np.argmin(d))
        return self.row_slot[i]


def _yaw_quat(yaw: float):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
