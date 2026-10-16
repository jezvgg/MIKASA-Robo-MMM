"""Object poses from depth, the way BIMBO_EvoX's observation-only planners find them.

BEHAVIOR's rules gave those planners depth, camera calibration and 3D models of the
objects, and nothing else: no segmentation, no object poses. Their pipeline
(``skills/fair-skills``: ``search``, ``body``, ``fit``) is reproduced here for a CPU
ManiSkill task:

1. **Back-project** each camera's depth (millimetres, 0 = no hit) through its intrinsics
   into the world frame.
2. **Cut the robot out** — points inside any link's collision box (plus a margin), and
   the box of a carried object. Without it BIMBO's table detector took the robot's torso
   and the fax in its hand for a table.
3. **Keep a height band over a known support** (a shelf, a counter) inside its footprint.
   The support comes from the static scene, which BIMBO's rules allowed as an offline map.
4. **Cluster** the band on a voxel grid, and label each cluster by colour. BIMBO prefers
   size over colour; here colour IS the identity of a prop by the task's design, so it
   only names a cluster — geometry still comes from depth.
5. **Fit the object's model** (surface samples of its collision mesh) by trimmed planar
   ICP started from every yaw on a grid, and **accept** a fit only if enough of the
   cloud lies on the model (inliers within 8 mm >= 0.55), the residual is small
   (RMS <= 1.2 cm), the cloud is big enough, and the cameras did not see THROUGH the
   model (observed depth more than 4 cm beyond a model point means the model stands
   where the camera saw empty space). The thresholds are BIMBO's.

Nothing here reads a pose from the simulator; ``audit`` exists to compare an estimate
with the truth for logs, and the planner must never act on what it returns.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FitConfig:
    yaw_step_deg: float = 15.0
    yaw_span_deg: float = 90.0
    "A box is symmetric under a quarter turn; a general model would use 360."
    iters: int = 12
    trim: float = 0.03
    "Metres. Matches farther than this are ignored in an ICP step (trimmed ICP)."
    inlier_tol: float = 0.008
    min_inlier: float = 0.55
    max_rms: float = 0.012
    min_points: int = 20
    see_through_tol: float = 0.04
    max_see_through: float = 0.3
    """Share of visible model points the cameras saw past; above it the fit is refused.
    Measured on Depth Recall v1 (three cameras, one of them at 10 cm): good fits read
    0.04-0.19 (one 0.19 was 0.6 mm off), the one fit that slid 43 mm read 0.45."""
    model_points: int = 1500


@dataclass
class Cloud:
    points: np.ndarray            # (N, 3) world, float64
    rgb: np.ndarray               # (N, 3) uint8
    views: list = field(default_factory=list)   # dict(depth (H,W) metres, K 3x3, world2cam 4x4)


# ------------------------------------------------------------------ capture --

def capture(task, uids: Sequence[str], max_range: float = 4.0) -> Cloud:
    """Render the named cameras now and back-project their depth into the world."""
    task.scene.update_render()
    pts, cols, views = [], [], []
    for uid in uids:
        cam = task._sensors[uid]
        cam.capture()
        obs = cam.get_obs(rgb=True, depth=True, position=False, segmentation=False)
        depth = obs["depth"][0, ..., 0].cpu().numpy().astype(np.float64) / 1000.0
        rgb = obs["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
        params = cam.get_params()
        K = params["intrinsic_cv"][0].cpu().numpy().astype(np.float64)
        ext = params["extrinsic_cv"][0].cpu().numpy().astype(np.float64)   # world -> camera (OpenCV)
        world2cam = np.vstack([ext[:3, :4], [0, 0, 0, 1]])
        views.append(dict(depth=depth, K=K, world2cam=world2cam, uid=uid))
        v, u = np.nonzero((depth > 0) & (depth < max_range))
        d = depth[v, u]
        cam_pts = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * d,
                            (v + 0.5 - K[1, 2]) / K[1, 1] * d, d], axis=1)
        cam2world = np.linalg.inv(world2cam)
        pts.append(cam_pts @ cam2world[:3, :3].T + cam2world[:3, 3])
        cols.append(rgb[v, u])
    if not pts:
        return Cloud(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), views)
    return Cloud(np.concatenate(pts), np.concatenate(cols), views)


# ----------------------------------------------------------------- self cut --

class RobotCutter:
    """Removes points inside the robot's link collision boxes (local frame, + margin)."""

    def __init__(self, robot, margin: float = 0.015):
        self.margin = float(margin)
        self.links = []
        from mani_skill.utils.geometry.trimesh_utils import get_component_meshes

        for link in robot.get_links():
            # Link-frame collision geometry (works on mani_skill 3.0.0b22 and the fork).
            meshes = get_component_meshes(link._objs[0])
            verts = [np.asarray(m.vertices) for m in meshes if len(m.vertices)]
            if not verts:
                continue
            allv = np.concatenate(verts)
            self.links.append((link, allv.min(axis=0) - self.margin, allv.max(axis=0) + self.margin))

    def keep(self, points: np.ndarray, extra_boxes: Sequence = ()) -> np.ndarray:
        """Mask of points outside every link box and every (pose 4x4, half sizes) box."""
        keep = np.ones(len(points), dtype=bool)
        boxes = [(link.pose.sp.to_transformation_matrix(), lo, hi) for link, lo, hi in self.links]
        boxes += [(T, -np.asarray(h) - self.margin, np.asarray(h) + self.margin) for T, h in extra_boxes]
        for T, lo, hi in boxes:
            idx = np.nonzero(keep)[0]
            if len(idx) == 0:
                break
            local = (points[idx] - T[:3, 3]) @ T[:3, :3]
            inside = np.all((local >= lo) & (local <= hi), axis=1)
            keep[idx[inside]] = False
        return keep


# ------------------------------------------------------ band, colour, groups --

def band(points: np.ndarray, xy_lo, xy_hi, z_lo: float, z_hi: float) -> np.ndarray:
    xy_lo, xy_hi = np.asarray(xy_lo), np.asarray(xy_hi)
    return (np.all((points[:, :2] >= xy_lo) & (points[:, :2] <= xy_hi), axis=1)
            & (points[:, 2] >= z_lo) & (points[:, 2] <= z_hi))


def hue_labels(rgb: np.ndarray, palette: Sequence, min_sat: float = 0.35, min_val: float = 0.12,
               max_hue_gap: float = 0.07) -> np.ndarray:
    """Index of the palette colour nearest in hue, or -1 (grey, dark, or far from all)."""
    ref = [colorsys.rgb_to_hsv(*c[:3])[0] for c in palette]
    out = np.full(len(rgb), -1, dtype=np.int64)
    if len(rgb) == 0:
        return out
    hsv = np.array([colorsys.rgb_to_hsv(*(c / 255.0)) for c in rgb])
    ok = (hsv[:, 1] >= min_sat) & (hsv[:, 2] >= min_val)
    gaps = np.stack([np.minimum(np.abs(hsv[:, 0] - h), 1 - np.abs(hsv[:, 0] - h)) for h in ref], axis=1)
    best = gaps.argmin(axis=1)
    ok &= gaps[np.arange(len(rgb)), best] <= max_hue_gap
    out[ok] = best[ok]
    return out


def voxel_groups(points: np.ndarray, cell: float = 0.01) -> list:
    """Connected components of occupied voxels (26-neighbourhood), largest first."""
    if len(points) == 0:
        return []
    keys = np.floor(points / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    index = {tuple(k): i for i, k in enumerate(uniq)}
    parent = list(range(len(uniq)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) > (0, 0, 0)]
    for i, k in enumerate(uniq):
        for d in offsets:
            j = index.get((k[0] + d[0], k[1] + d[1], k[2] + d[2]))
            if j is not None:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    roots = np.array([find(i) for i in range(len(uniq))])[inv]
    groups = [np.nonzero(roots == r)[0] for r in np.unique(roots)]
    return sorted(groups, key=len, reverse=True)


# ---------------------------------------------------------------- the model --

def box_model(half_sizes, n: int = 1500, seed: int = 0) -> np.ndarray:
    """Area-weighted samples of a box surface, in the box frame (BIMBO: 1500 points)."""
    hx, hy, hz = (float(v) for v in half_sizes)
    faces = [(0, hx, (1, 2), hy * hz), (0, -hx, (1, 2), hy * hz),
             (1, hy, (0, 2), hx * hz), (1, -hy, (0, 2), hx * hz),
             (2, hz, (0, 1), hx * hy), (2, -hz, (0, 1), hx * hy)]
    area = np.array([f[3] for f in faces])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, area / area.sum())
    half = np.array([hx, hy, hz])
    out = []
    for (axis, value, free, _), c in zip(faces, counts):
        p = np.zeros((c, 3))
        p[:, axis] = value
        for f in free:
            p[:, f] = rng.uniform(-half[f], half[f], size=c)
        out.append(p)
    return np.concatenate(out)


def _transform(model: np.ndarray, x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.empty_like(model)
    out[:, 0] = c * model[:, 0] - s * model[:, 1] + x
    out[:, 1] = s * model[:, 0] + c * model[:, 1] + y
    out[:, 2] = model[:, 2] + z
    return out


def see_through(model_world: np.ndarray, views: Sequence[dict], tol: float) -> float:
    """Share of model points that some camera looked past (observed depth > point + tol)."""
    seen = np.zeros(len(model_world), dtype=bool)
    visible = np.zeros(len(model_world), dtype=bool)
    for v in views:
        T, K, depth = v["world2cam"], v["K"], v["depth"]
        cam = model_world @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        front = z > 0.05
        u = np.floor(cam[:, 0] / np.where(front, z, 1) * K[0, 0] + K[0, 2]).astype(np.int64)
        r = np.floor(cam[:, 1] / np.where(front, z, 1) * K[1, 1] + K[1, 2]).astype(np.int64)
        h, w = depth.shape
        inside = front & (u >= 0) & (u < w) & (r >= 0) & (r < h)
        obs = np.zeros(len(model_world))
        obs[inside] = depth[r[inside], u[inside]]
        obs[inside & (obs <= 0)] = np.inf          # no hit: the camera saw nothing there
        visible |= inside
        seen |= inside & (obs > z + tol)
    return float(seen.sum() / max(visible.sum(), 1))


def fit_planar(cloud: np.ndarray, model: np.ndarray, z: float, views: Sequence[dict] = (),
               cfg: FitConfig = FitConfig(), init_xy=None) -> dict:
    """Trimmed planar ICP of ``model`` (object frame) onto ``cloud`` from every yaw.

    x, y and yaw are free; the height is given (the object stands on a known support).
    ``init_xy`` starts every hypothesis there instead of at the cloud's centroid — a
    one-face view's centroid sits half an object off the centre, and from there ICP can
    slide the model along the face (measured: 43 mm off, rejected by the see-through test).
    Returns the best hypothesis with its metrics and ``accepted``/``reasons``.
    """
    from scipy.spatial import cKDTree

    result = dict(accepted=False, reasons=["too few points"], n=len(cloud))
    if len(cloud) < cfg.min_points:
        return result
    centre = cloud[:, :2].mean(axis=0) if init_xy is None else np.asarray(init_xy, dtype=np.float64)[:2]
    yaws = np.deg2rad(np.arange(0.0, cfg.yaw_span_deg, cfg.yaw_step_deg))
    best = None
    for yaw0 in yaws:
        x, y, yaw = float(centre[0]), float(centre[1]), float(yaw0)
        for _ in range(cfg.iters):
            placed = _transform(model, x, y, z, yaw)
            dist, idx = cKDTree(placed).query(cloud)
            use = dist <= cfg.trim
            if use.sum() < 3:
                break
            a = placed[idx[use], :2]                  # model points (current pose)
            b = cloud[use, :2]                        # their matches in the cloud
            ca, cb = a.mean(axis=0), b.mean(axis=0)
            H = (a - ca).T @ (b - cb)
            dtheta = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
            c, s = np.cos(dtheta), np.sin(dtheta)
            R = np.array([[c, -s], [s, c]])
            new_xy = R @ (np.array([x, y]) - ca) + cb
            x, y, yaw = float(new_xy[0]), float(new_xy[1]), yaw + float(dtheta)
        placed = _transform(model, x, y, z, yaw)
        dist, _ = cKDTree(placed).query(cloud)
        inlier = float((dist <= cfg.inlier_tol).mean())
        near = dist[dist <= cfg.trim]
        rms = float(np.sqrt((near ** 2).mean())) if len(near) else float("inf")
        through = see_through(placed, views, cfg.see_through_tol) if views else 0.0
        cost = rms + 0.02 * (1.0 - inlier) + 0.02 * through
        if best is None or cost < best["cost"]:
            best = dict(x=x, y=y, z=z, yaw=float(np.mod(yaw, np.deg2rad(cfg.yaw_span_deg))),
                        rms=rms, inlier=inlier, see_through=through, cost=cost, n=len(cloud))
    reasons = []
    if best["inlier"] < cfg.min_inlier:
        reasons.append(f"inlier {best['inlier']:.2f} < {cfg.min_inlier}")
    if best["rms"] > cfg.max_rms:
        reasons.append(f"rms {best['rms'] * 100:.1f} cm > {cfg.max_rms * 100:.1f}")
    if best["see_through"] > cfg.max_see_through:
        reasons.append(f"see-through {best['see_through']:.2f} > {cfg.max_see_through}")
    best.update(accepted=not reasons, reasons=reasons)
    return best


def audit(estimate: dict, true_pose, yaw_period_deg: float = 90.0) -> dict:
    """Estimate vs simulator truth — for LOGS ONLY; a planner must not act on it."""
    p = np.asarray(true_pose.p, dtype=np.float64)
    q = np.asarray(true_pose.q, dtype=np.float64)   # wxyz
    yaw_true = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    period = np.deg2rad(yaw_period_deg)
    dyaw = (estimate["yaw"] - yaw_true + period / 2) % period - period / 2
    return dict(err_xy_mm=round(float(np.linalg.norm([estimate["x"] - p[0], estimate["y"] - p[1]])) * 1000, 1),
                err_yaw_deg=round(float(np.rad2deg(abs(dyaw))), 1))
