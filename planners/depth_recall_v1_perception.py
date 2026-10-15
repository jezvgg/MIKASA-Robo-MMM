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

What is still read from the simulator, and named so it is not mistaken for honest:
the planning world's collision geometry (mplib mirrors the whole scene, props at their
true poses), `is_grasping` (contact forces), `hold_object_in_planner` (attaches the true
mesh for collision checks), and the static scene map (shelf height, cabinet and counter
footprints — BIMBO's rules allowed an offline map of static geometry).
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
        self.half = (task.cfg.prop_half_xy, task.cfg.prop_half_xy, task.cfg.prop_half_h)
        self.model = pc.box_model(self.half, n=fit_cfg.model_points)
        self.cutter = pc.RobotCutter(task.agent.robot)
        self.est: dict = {}          # prop index -> dict(x, y, z, yaw, where, fit metrics)
        self.row_slot: dict = {}     # prop index -> row slot it was SEEN in before it moved
        self.held = None             # (prop index, hand->prop sapien.Pose)
        self.search_radius = 0.10    # metres around a stand point or the last estimate
        self.looks = []              # per look: where, cams, per-prop outcome + audit

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

    # ------------------------------------------------------------------ look --

    def look(self, where: str, cams, only=None, anchors=None, min_seen: int = 5) -> dict:
        """Capture, cut, band, cluster by colour, fit; update the estimates that pass.

        `only` restricts the fit to these prop indices (a wrist re-fit of one prop).
        `anchors` are the support's known stand points (the row or the counter slots —
        static scenery, also in the policy's observation): a colour cluster of at least
        `min_seen` points is tied to the anchor nearest its centroid, ICP starts there,
        and when the fit is refused the anchor itself is kept as a COARSE estimate. Coarse
        is enough to choose and approach a prop; a grasp needs an accepted fit.
        Without anchors, ICP starts from the previous estimate or the centroid.
        Returns {index: fit dict} for every prop tried.
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
                anchor = a[int(np.argmin(np.linalg.norm(a - seen[:, :2].mean(axis=0), axis=1)))]
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
                fit["coarse_anchor"] = [round(float(v), 3) for v in anchor]
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
            if "coarse_anchor" in fit:
                entry["coarse_anchor"] = fit["coarse_anchor"]
            out[self.task.props[i].name] = entry
        self.looks.append(dict(where=where, cams=list(cams), props=out))
        self.say("perceive", where=where, cams=list(cams), props=out)

    # ----------------------------------------------------------------- reads --

    def has(self, i: int, fine: bool = False) -> bool:
        """Known at all, or (``fine``) known from an accepted fit, not just an anchor."""
        if self.held is not None and self.held[0] == i:
            return True
        return i in self.est and not (fine and self.est[i].get("coarse"))

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

    def remember_row_slot(self, i: int, row_pts) -> int:
        e = self.est[i]
        d = np.linalg.norm(np.asarray(row_pts)[:, :2] - [e["x"], e["y"]], axis=1)
        self.row_slot[i] = int(np.argmin(d))
        return self.row_slot[i]


def _yaw_quat(yaw: float):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
