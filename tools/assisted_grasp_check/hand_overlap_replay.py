"""Replay recorded episodes of an assisted-grasp dataset and measure how far the hand goes into the
graspable objects it cannot collide with.

utils/mikasa/assisted_grasp.py builds one SAPIEN drive hand -> object for every graspable object when
the env is made, and a drive switches collision between its two bodies off for the whole episode
(drive_collision_test.py). So the physics never stops `gripper_link` (the hand, which carries the hand
camera) from entering those objects. This script measures it geometrically from the collision shapes:
- "panel" shapes (thin and wide: a door leaf, a drawer front): for the hand's vertices that lie over
  the panel, the distance the hand would have to move along the panel normal to clear it (0 when it
  is beside the panel). At or above the panel thickness the hand straddles the panel: gone through.
  (Distance to the nearest face would saturate at half the thickness.)
- "other" shapes (handle bars, props): depth of the hand's vertices inside the shape's convex hull.

Replay: the dataset's own profile env kwargs, reset with the episode seed, each 10 Hz action held for
two 20 Hz control steps (the protocol our datasets replay bit-exactly with); the replayed 12D state is
compared with the recorded one every frame.
usage: hand_overlap_replay.py <lerobot dataset dir> <out.json> <episode> [<episode> ...]"""
import glob, json, sys
import numpy as np, gymnasium as gym, pyarrow.parquet as pq
import sapien
from scipy.spatial import ConvexHull
import mani_skill  # noqa: F401
import my_scenes  # noqa: F401

root, out_path, episodes = sys.argv[1], sys.argv[2], [int(e) for e in sys.argv[3:]]
prof = json.load(open(f"{root}/meta/source_rlds_metadata.json"))["profile"]
meta = pq.read_table(sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")),
                     columns=["episode_index", "episode_seed"]).to_pandas().set_index("episode_index")
data = pq.read_table(sorted(glob.glob(f"{root}/data/*/*.parquet")),
                     columns=["episode_index", "frame_index", "action", "observation.state"]).to_pandas()


def shape_vertices(s):
    if isinstance(s, sapien.physx.PhysxCollisionShapeBox):
        return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * np.asarray(s.half_size)
    if isinstance(s, sapien.physx.PhysxCollisionShapeConvexMesh):
        return np.asarray(s.vertices) * np.asarray(s.scale)
    return None


def name_of(obj):
    o = obj._objs[0]
    return o.name if isinstance(o, sapien.Entity) else o.entity.name


def to_world(comp, s, v):
    T = (comp.entity_pose * s.local_pose).to_transformation_matrix()
    return v @ T[:3, :3].T + T[:3, 3]


rows = []
for ep in episodes:
    env = gym.make(prof["env_id"], **{**prof["env_kwargs"], "obs_mode": "state"})
    base = env.unwrapped
    env.reset(seed=int(meta.loc[ep, "episode_seed"]))
    ag = base._ag
    hand = ag.hand._objs[0]
    hand_shapes = [(s, v) for s in hand.collision_shapes if (v := shape_vertices(s)) is not None]
    objects = []
    for obj in ag.objects:
        comp = obj._objs[0]
        comp = comp if isinstance(comp, sapien.physx.PhysxRigidBaseComponent) else \
            comp.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        for s in comp.collision_shapes:
            v = shape_vertices(s)
            if v is None or len(v) < 4:
                continue
            ext = np.sort(np.ptp(v, axis=0))
            kind = "panel" if ext[0] < 0.04 and ext[2] > 0.20 else "other"
            objects.append((comp.entity.name, kind, comp, s, ConvexHull(v).equations, v.min(0), v.max(0)))
    rec = data[data.episode_index == ep].sort_values("frame_index")
    A, S = np.stack(rec["action"].to_numpy()), np.stack(rec["observation.state"].to_numpy())
    per_frame, state_err = [], []
    for k, a in enumerate(A):
        for _ in range(2):
            env.step(a)
        q = np.asarray(base.agent.robot.get_qpos().cpu()).reshape(-1)[3:]
        if k + 1 < len(S):
            state_err.append(float(np.abs(q - S[k + 1]).max()))
        hv = np.concatenate([to_world(hand, s, v) for s, v in hand_shapes])
        best = {"panel": (0.0, ""), "other": (0.0, "")}
        for name, kind, comp, s, eq, lo, hi in objects:
            T = (comp.entity_pose * s.local_pose).to_transformation_matrix()
            p = (hv - T[:3, 3]) @ T[:3, :3]                       # hand vertices in the shape frame
            if kind == "panel":
                n = int(np.argmin(hi - lo))                        # the thin axis
                plane = [i for i in range(3) if i != n]
                over = np.all((p[:, plane] > lo[plane]) & (p[:, plane] < hi[plane]), axis=1)
                if not over.any():
                    continue
                pn = p[over, n]
                if pn.max() <= lo[n] or pn.min() >= hi[n]:
                    continue                                       # the hand is beside the panel
                # The distance the hand would have to move along the normal to clear the panel:
                # at least the panel thickness when the hand straddles it (gone through).
                depth = float(min(pn.max() - lo[n], hi[n] - pn.min()))
            else:
                signed = p @ eq[:, :3].T + eq[:, 3]                # <= 0 inside each face
                inside = np.all(signed < 0, axis=1)
                depth = float((-signed[inside]).min(axis=1).max()) if inside.any() else 0.0
            if depth > best[kind][0]:
                best[kind] = (depth, name)
        held = ag.held_index()
        per_frame.append({"frame": k, "panel_mm": best["panel"][0] * 1e3, "panel_obj": best["panel"][1],
                          "other_mm": best["other"][0] * 1e3, "other_obj": best["other"][1],
                          "held": None if held is None or held < 0 else name_of(ag.objects[held])})
    env.close()
    pm = np.array([f["panel_mm"] for f in per_frame]); om = np.array([f["other_mm"] for f in per_frame])
    row = {"episode": ep, "frames": len(per_frame), "max_state_err": max(state_err, default=0.0),
           "panel_max_mm": float(pm.max()), "panel_frames_gt5mm": int((pm > 5).sum()),
           "panel_frames_gt10mm": int((pm > 10).sum()), "panel_frames_gt20mm": int((pm > 20).sum()),
           "panel_frames_gt50mm": int((pm > 50).sum()),
           "panel_frames_through": int(sum(f["panel_mm"] >= 20 for f in per_frame)),
           "panel_thickness_mm": sorted({round(float(np.min(hi - lo)) * 1e3, 1) for _, kd, _, _, _, lo, hi in objects
                                         if kd == "panel"}),
           "other_max_mm": float(om.max()), "other_frames_gt10mm": int((om > 10).sum()),
           "other_gt10mm_while_not_held": int(sum(f["other_mm"] > 10 and f["held"] is None for f in per_frame)),
           "objects": sorted({f[k] for f in per_frame for k in ("panel_obj", "other_obj") if f[k]}),
           "per_frame": per_frame}
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "per_frame"}), flush=True)
json.dump(rows, open(out_path, "w"))
