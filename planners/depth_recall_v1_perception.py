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
  stopped apart while the gripper is commanded closed — not contact forces;
* **the pads are read by colour**: which counter slot holds which prop's pad is the
  majority hue of the flat band just over the counter at each slot (`identify_pads`);
* **close only on something**: before the fingers close, the prop's own points must lie
  in the volume between the finger pads (`between_pads`) — BIMBO_EvoX's assisted-grasp
  rule (rays cast between the pads must hit the object) made from the cameras.

What is still read from the simulator: the static scene map above (fixtures, their
footprints and heights) and the props' 3D models — their collision meshes in their own
frames, which is what BEHAVIOR's rules gave BIMBO_EvoX ("3D models of the objects").
Truth also goes into the LOGS (`pc.audit`, the row shift's error) and nowhere else.

The props are RoboCasa objects (`my_scenes.depth_recall_v1.PROPS`), so the model a fit
uses is each prop's own mesh (`pc.mesh_model`), fitted at its own rest height, over a
full turn of yaw; a prop is named by its measured `PROP_HUES` signature.
"""

from __future__ import annotations

import numpy as np
import sapien

from utils.mikasa import point_cloud as pc

BASE_CAMS = ("left_base_camera_link", "right_base_camera_link")
HAND_CAM = "fetch_hand"
FINGERS = ("l_gripper_finger_link", "r_gripper_finger_link")

#: Points of the prop's colour that must lie between the open finger pads before a
#: close. Seen by three cameras a 4.5 cm prop face between the pads gives hundreds.
PAD_MIN_POINTS = 15

#: How each prop's colour looks to the robot's three cameras, in identity order:
#: ketchup, pepper shaker body, salt shaker body, mustard. MEASURED 2026-10-16 on the
#: task's own renders (`left/right_base_camera_link`, `fetch_hand`; seeds 1000-1004, the
#: props standing in the row and on their pads; only pixels the segmentation gives to
#: the prop): circular-mean hue 0.0 / 120.0 / 224.5 / 55.3 deg, 99th percentile of the
#: deviation from it 0.0 / 0.0 / 1.4 / 1.3 deg, colourful pixels' saturation >= 0.57 and
#: value >= 0.54 (5th percentiles). The materials' own hues are 0 / 120 / 229 / 51: the
#: renderer moves blue and yellow by 4-5 deg, which is why these are measured. The pads
#: measured the same hues (0.0 / 120.0 / 224.5 / 55.5), so one palette reads both.
#: The band is 10 deg, seven times the widest measured spread and well short of the
#: nearest pair (red and yellow, 55 deg apart); the saturation and value floors are the
#: ones every colour test in this package used (0.35, 0.12). The shakers' tops (black,
#: white) are grey to it and belong to no prop, which costs a fit 15-40 % of a shaker's
#: points and nothing else.
PROP_HUES = (
    pc.HueSignature(0.0, 10.0),
    pc.HueSignature(120.0, 10.0),
    pc.HueSignature(224.5, 10.0),
    pc.HueSignature(55.3, 10.0),
)

#: A fit's yaw search for a general mesh: a full turn (a box needed only a quarter).
MESH_YAW_SPAN_DEG = 360.0
MESH_YAW_STEP_DEG = 30.0


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
        """The prop's own collision mesh (its 3D model), at the believed pose."""
        mesh = self._belief.meshes[self.index].copy()
        if to_world_frame:
            mesh.apply_transform(self._belief.pose(self.index).to_transformation_matrix())
        return mesh


class PropBelief:
    """Estimated prop poses, the wrist-camera re-fits, the carry transform and the memory."""

    def __init__(self, task, say=None, fit_cfg: pc.FitConfig = pc.FitConfig(
            yaw_span_deg=MESH_YAW_SPAN_DEG, yaw_step_deg=MESH_YAW_STEP_DEG)):
        from mani_skill.utils.geometry.trimesh_utils import get_component_meshes

        self.task, self.cfg, self.fit_cfg = task, task.cfg, fit_cfg
        self.say = say or (lambda message, **data: None)
        assert len(PROP_HUES) >= task.cfg.n_props, "one measured colour signature per prop"
        self.colors = PROP_HUES[: task.cfg.n_props]
        missing = [uid for uid in BASE_CAMS + (HAND_CAM,) if uid not in task._sensors]
        if missing:
            raise RuntimeError(f"pointcloud perception needs the cameras {missing}; the env "
                               "was built without them (a render backend is required)")
        # Each prop's 3D model, in its own frame: the merged collision mesh (grasp and
        # place poses read its bounds), its convex parts (the fit's surface samples), the
        # origin's height over the mesh bottom, and the mesh's bounds (the carried prop's
        # box for the robot cut).
        self.meshes, self.models, self.rest_lift, self.bounds, self.height = [], [], [], [], []
        for prop in task.props:
            mesh = prop.get_first_collision_mesh(to_world_frame=False)
            parts = get_component_meshes(prop._bodies[0])
            b = np.asarray(mesh.bounds, dtype=np.float64)
            self.meshes.append(mesh)
            self.models.append(pc.mesh_model(parts, n=fit_cfg.model_points))
            self.rest_lift.append(-float(b[0, 2]))
            self.bounds.append(b)
            self.height.append(float(b[1, 2] - b[0, 2]))
        self.cutter = pc.RobotCutter(task.agent.robot)
        self.est: dict = {}          # prop index -> dict(x, y, z, yaw, where, fit metrics)
        self.row_slot: dict = {}     # prop index -> row slot it was SEEN in before it moved
        self.held = None             # (prop index, hand->prop sapien.Pose)
        self.search_radius = 0.10    # metres around a stand point or the last estimate
        self.looks = []              # per look: where, cams, per-prop outcome + audit
        self.row_dx = None           # the row's measured sideways shift, once calibrated
        self.slot_pts = None         # counter slot points the planner uses (map + row_dx)
        self.pad_of = None           # prop index -> counter slot of its pad, once read
        self.checks = []             # per between-pads check: (prop index, points, ok)

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
            extra.append((T, self.bounds[self.held[0]][0], self.bounds[self.held[0]][1]))
        keep = self.cutter.keep(cloud.points, extra)
        lo, hi, z0 = self._region(where)
        h = max(self.height)
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
            z = z0 + self.rest_lift[i]          # the origin's height on this support
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
            fit = pc.fit_planar(seen, self.models[i], z, cloud.views, self.fit_cfg, init_xy=init)
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
                # Truth, for the LOG only: how far the estimate is from the prop. The
                # four props are round, so their yaw is not observable and its error
                # is logged over a full turn, meaning nothing.
                entry.update(pc.audit(fit, self.task.props[i].pose.sp,
                                      yaw_period_deg=self.fit_cfg.yaw_span_deg))
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

    # -------------------------------------------------------------- pads --

    def identify_pads(self, slot_pts, cams=BASE_CAMS, radius: float = 0.035,
                      min_points: int = 10):
        """{prop index: counter slot} from the pads' colours, or None if not all read.

        The flat band 3 mm under to 1 cm over the counter top, within `radius` of each
        slot point (less than the pad's half width, so only the pad's own top), voted by
        hue against the props' palette. All pads must be read, each colour once.
        """
        cloud = pc.capture(self.task, cams)
        lo, hi, z0 = self._region("counter")
        flat = pc.band(cloud.points, lo, hi, z0 - 0.003, z0 + 0.010)
        pts, rgb = cloud.points[flat], cloud.rgb[flat]
        slots = np.asarray(slot_pts, dtype=np.float64)
        near = np.linalg.norm(pts[:, None, :2] - slots[None, :, :2], axis=2) <= radius
        keep = near.any(axis=1)
        labels = pc.hue_labels(rgb[keep], self.colors)
        near = near[keep]
        found, votes = {}, {}
        for k in range(len(slots)):
            lab = labels[near[:, k] & (labels >= 0)]
            counts = np.bincount(lab, minlength=len(self.colors))
            votes[k] = counts.tolist()
            if counts.max() >= min_points:
                found[k] = int(counts.argmax())
        n = self.cfg.n_props
        ok = len(set(found.values())) == len(found) and set(found.values()) == set(range(n))
        pad_of = {c: k for k, c in found.items()} if ok else None
        # Truth, for the LOG only.
        truth = np.asarray(self.task.pad_slots.cpu() if hasattr(self.task.pad_slots, "cpu")
                           else self.task.pad_slots).reshape(-1, n)[0]
        self.say("pads", ok=ok, slots={k: self.task.props[c].name for k, c in found.items()},
                 votes=votes, right=None if pad_of is None else
                 all(int(truth[c]) == pad_of[c] for c in range(n)))
        self.pad_of = pad_of
        return pad_of

    # ----------------------------------------------------- grasp volume --

    def _closing_volume(self):
        """(lo, hi, closing axis, gripper_link 4x4) of the box between the open finger
        pads, in gripper_link's frame.

        From the fingers' own collision boxes at this moment: along the axis the two
        fingers are apart on, from one inner face to the other; along the other two,
        where both pads overlap.
        """
        from mani_skill.utils.geometry.trimesh_utils import get_component_meshes

        links = {l.name: l for l in self.task.agent.robot.get_links()}
        grip = next(l for n, l in links.items() if n.endswith("gripper_link") and "finger" not in n)
        G = grip.pose.sp.to_transformation_matrix()
        boxes = []
        for name in FINGERS:
            link = next(l for n, l in links.items() if n.endswith(name))
            v = np.concatenate([np.asarray(m.vertices) for m in get_component_meshes(link._objs[0])])
            T = np.linalg.inv(G) @ link.pose.sp.to_transformation_matrix()
            local = v @ T[:3, :3].T + T[:3, 3]
            boxes.append((local.min(axis=0), local.max(axis=0)))
        return closing_box(boxes[0], boxes[1]) + (G,)

    def between_pads(self, i: int, cams=(HAND_CAM,) + BASE_CAMS,
                     min_points: int = PAD_MIN_POINTS, inset: float = 0.003) -> dict:
        """Are prop ``i``'s own points between the open finger pads? Ask before a close.

        BIMBO_EvoX casts rays from one pad to the other and needs the object to cut them.
        No camera sees that band: the hand stands in front of it (measured 2026-10-16,
        seed 1000: 600 prop points within 20 cm of the hand, none in the pads' own
        band). The prop is a rigid upright body, so the test is the one the rays make,
        taken over its whole height: points of its colour between the pads' inner faces
        (closing axis) and within the pads' length (approach axis), at any height the
        prop has (the remaining axis, extended by the prop's height both ways). A close
        on air, or a hand that stopped short of the prop, counts none.
        """
        lo, hi, axis, G = self._closing_volume()
        # gripper_link's z is the approach (`build_grasp_pose`); of the other two, the one
        # that is not the closing axis runs along the prop's height.
        up = next((k for k in range(3) if k not in (axis, 2)), None)
        lo, hi = lo.copy(), hi.copy()
        H = self.height[i]
        if up is not None:
            lo[up], hi[up] = lo[up] - H, hi[up] + H
        cloud = pc.capture(self.task, cams)
        reach = np.linalg.norm(cloud.points - G[:3, 3], axis=1) <= 0.25
        mine = cloud.points[reach][pc.hue_labels(cloud.rgb[reach], self.colors) == i]
        n, centre = count_between(mine, G, lo, hi, inset)
        out = dict(ok=n >= min_points, n=n, near=int(len(mine)),
                   gap_mm=round(float(hi[axis] - lo[axis]) * 1000, 1))
        if n:
            out["off_mm"] = round(float(centre[axis] - (lo[axis] + hi[axis]) / 2) * 1000, 1)
        self.checks.append((i, n, out["ok"]))
        return out

    def remember_row_slot(self, i: int, row_pts) -> int:
        e = self.est[i]
        d = np.linalg.norm(np.asarray(row_pts)[:, :2] - [e["x"], e["y"]], axis=1)
        self.row_slot[i] = int(np.argmin(d))
        return self.row_slot[i]


def closing_box(a, b):
    """(lo, hi, axis) of the space between two finger boxes, each (lo, hi) in one frame.

    ``axis`` is the one the boxes' centres are furthest apart on; along it the space runs
    from the nearer box's inner face to the other's, along the other two it is where
    both boxes overlap.
    """
    (a_lo, a_hi), (b_lo, b_hi) = (np.asarray(a[0], float), np.asarray(a[1], float)), \
        (np.asarray(b[0], float), np.asarray(b[1], float))
    axis = int(np.argmax(np.abs((a_lo + a_hi) / 2 - (b_lo + b_hi) / 2)))
    lo, hi = np.maximum(a_lo, b_lo), np.minimum(a_hi, b_hi)
    first, second = ((a_lo, a_hi), (b_lo, b_hi)) if a_lo[axis] < b_lo[axis] else ((b_lo, b_hi), (a_lo, a_hi))
    lo[axis], hi[axis] = first[1][axis], second[0][axis]
    return lo, hi, axis


def count_between(points, G, lo, hi, inset: float = 0.0):
    """(how many world ``points`` lie in the box (lo, hi) of frame ``G`` (4x4), inset on
    every side, and their mean in that frame or None)."""
    local = (np.asarray(points, float).reshape(-1, 3) - G[:3, 3]) @ G[:3, :3]
    inside = np.all((local >= np.asarray(lo) + inset) & (local <= np.asarray(hi) - inset), axis=1)
    n = int(inside.sum())
    return n, (local[inside].mean(axis=0) if n else None)


def _yaw_quat(yaw: float):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
