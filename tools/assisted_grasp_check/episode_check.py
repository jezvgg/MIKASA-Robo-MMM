"""Run one collection episode (scripts/collection_episode.py, same arguments) and measure, after
every control step, how far the hand (`gripper_link`) goes into the graspable objects' panels (door
leaves, drawer fronts: the distance it would have to move along the panel normal to clear it, as in
hand_overlap_replay.py) and how hard it presses them (contact force hand <-> object).

    PYTHONPATH=. python tools/assisted_grasp_check/episode_check.py cabinet_search 7880000 --assisted-grasp
"""
import atexit, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import sapien
import mani_skill.envs.sapien_env as sapien_env
from scipy.spatial import ConvexHull  # noqa: F401  (same environment as hand_overlap_replay)

STATS = {"steps": 0, "depth_mm": [], "press_n": []}
_cache = {}


def _shapes(comp):
    out = []
    for s in comp.collision_shapes:
        if isinstance(s, sapien.physx.PhysxCollisionShapeBox):
            v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * np.asarray(s.half_size)
        elif isinstance(s, sapien.physx.PhysxCollisionShapeConvexMesh):
            v = np.asarray(s.vertices) * np.asarray(s.scale)
        else:
            continue
        out.append((s, v))
    return out


def _setup(env):
    ag = env._ag
    hand = ag.hand._objs[0]
    panels = []
    for obj in ag.objects:
        o = obj._objs[0]
        if isinstance(o, sapien.Entity):
            continue                                  # free bodies: not panels
        for s, v in _shapes(o):
            ext = np.sort(np.ptp(v, axis=0))
            if ext[0] < 0.04 and ext[2] > 0.20:
                panels.append((o, s, v.min(0), v.max(0)))
    return {"hand": hand, "hand_shapes": _shapes(hand), "panels": panels, "objects": list(ag.objects),
            "hand_link": ag.hand}


def _measure(env):
    c = _cache.get(id(env)) or _cache.setdefault(id(env), _setup(env))
    hv = []
    for s, v in c["hand_shapes"]:
        T = (c["hand"].entity_pose * s.local_pose).to_transformation_matrix()
        hv.append(v @ T[:3, :3].T + T[:3, 3])
    hv = np.concatenate(hv)
    depth = 0.0
    for comp, s, lo, hi in c["panels"]:
        T = (comp.entity_pose * s.local_pose).to_transformation_matrix()
        p = (hv - T[:3, 3]) @ T[:3, :3]
        n = int(np.argmin(hi - lo)); plane = [i for i in range(3) if i != n]
        over = np.all((p[:, plane] > lo[plane]) & (p[:, plane] < hi[plane]), axis=1)
        if over.any():
            pn = p[over, n]
            if pn.max() > lo[n] and pn.min() < hi[n]:
                depth = max(depth, float(min(pn.max() - lo[n], hi[n] - pn.min())))
    press = 0.0
    for obj in c["objects"]:
        f = env.scene.get_pairwise_contact_forces(c["hand_link"], obj)
        f = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
        press = max(press, float(np.linalg.norm(f.reshape(-1, 3), axis=1).max()))
    STATS["steps"] += 1
    STATS["depth_mm"].append(depth * 1e3)
    STATS["press_n"].append(press)


_orig_step = sapien_env.BaseEnv.step


def _step(self, action):
    out = _orig_step(self, action)
    if getattr(self, "_ag", None) is not None:
        _measure(self)
    return out


sapien_env.BaseEnv.step = _step


@atexit.register
def _report():
    d, f = np.array(STATS["depth_mm"] or [0.0]), np.array(STATS["press_n"] or [0.0])
    print(f"hand_check steps={STATS['steps']} panel_depth_max_mm={d.max():.1f} steps_depth_gt5mm={int((d > 5).sum())} "
          f"steps_depth_gt20mm={int((d > 20).sum())} press_max_n={f.max():.1f} steps_press_gt1n={int((f > 1).sum())} "
          f"steps_press_gt20n={int((f > 20).sum())}", flush=True)


if os.environ.get("AGCHECK_RENDER_MODE"):
    # With --video-dir: record the robot's own cameras (render_mode "sensors") instead of the
    # third-person render camera that collection_episode.py asks for.
    import gymnasium
    _make = gymnasium.make

    def _make_sensors(*args, **kwargs):
        if kwargs.get("render_mode"):
            kwargs["render_mode"] = os.environ["AGCHECK_RENDER_MODE"]
        return _make(*args, **kwargs)

    gymnasium.make = _make_sensors

if __name__ == "__main__":
    import collection_episode
    sys.argv = ["collection_episode.py"] + sys.argv[1:]
    collection_episode.main()
