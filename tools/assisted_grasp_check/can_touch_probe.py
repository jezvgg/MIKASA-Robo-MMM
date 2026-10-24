"""Run one collection episode (scripts/collection_episode.py, same arguments) and record every
contact between a robot link and CabinetSearch's can (`env.cube`) before the push starts: the
step, the planner phase it happened in (the last `[event] ... phase <name>` line), the link and
the contact force, and the can's displacement and tilt from its spawn; and the first step the
can has moved (> 1 mm or > 1 deg) with its phase, since a strike inside one control step can
leave no contact at the step's end (7880257: first contact read with the can already 4.6 cm
off). The question it answers: which motion strikes the can on a seed the can-early gate
rejects.

It needs the GPU like any episode: with `render_backend="none"` SAPIEN 3.0.3 segfaults building
the robot's visual meshes (`urdf_loader._build_link`, 2026-10-23). The CPU-only view of the same
question is can_clearance_fk.py on a saved trajectory.

    PYTHONPATH=. python tools/assisted_grasp_check/can_touch_probe.py \
        cabinet_search 7880257 --assisted-grasp --trajectory-out /path/p.npz
"""
import atexit, builtins, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import mani_skill.envs.sapien_env as sapien_env

PHASE = {"name": "", "push": None}
TOUCH = []          # (step, phase, link, force N, can xy m, can tilt deg)
SPAWN = {}
MOVED = []          # the first (step, phase, can xy m, can tilt deg) past 1 mm or 1 deg
_print = builtins.print


def _watch(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("[event] ") and " phase " in args[0]:
        PHASE["name"] = args[0].split(" phase ", 1)[1].strip()
        if PHASE["push"] is None and PHASE["name"] == "push start":
            PHASE["push"] = int(args[0].split("step=")[1].split()[0])
    return _print(*args, **kwargs)


builtins.print = _watch


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _can(env):
    pose = env.cube.pose
    p, (w, x, y, z) = _np(pose.p).reshape(-1)[:3], _np(pose.q).reshape(-1)[:4]
    return p, np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


_orig_step = sapien_env.BaseEnv.step


def _step(self, action):
    out = _orig_step(self, action)
    cube = getattr(self, "cube", None)
    if cube is None or PHASE["push"] is not None:
        return out
    p, up = _can(self)
    if not SPAWN:
        SPAWN.update(p=p, up=up)
    xy = float(np.linalg.norm(p[:2] - SPAWN["p"][:2]))
    tilt = float(np.degrees(np.arccos(np.clip(up @ SPAWN["up"], -1, 1))))
    step = int(_np(self.elapsed_steps).reshape(-1)[0])
    if not MOVED and (xy > 1e-3 or tilt > 1.0):
        MOVED.append((step, PHASE["name"], round(xy, 4), round(tilt, 1)))
    for link in self.agent.robot.links:
        f = float(np.linalg.norm(_np(self.scene.get_pairwise_contact_forces(link, cube)).reshape(-1, 3), axis=1).max())
        if f > 0.1:
            TOUCH.append((step, PHASE["name"], link.name, round(f, 2), round(xy, 4), round(tilt, 1)))
    return out


sapien_env.BaseEnv.step = _step


@atexit.register
def _report():
    phases = {}
    for step, phase, link, f, xy, tilt in TOUCH:
        d = phases.setdefault(phase, {"first_step": step, "links": set(), "max_n": 0.0, "steps": set()})
        d["links"].add(link); d["max_n"] = max(d["max_n"], f); d["steps"].add(step)
    print("can_touch push_start=%s contacts=%d first=%s moved=%s" % (
        PHASE["push"], len(TOUCH), json.dumps(TOUCH[0]) if TOUCH else None,
        json.dumps(MOVED[0]) if MOVED else None), flush=True)
    for phase, d in phases.items():
        print("can_touch phase=%r first_step=%d steps=%d max_n=%.1f links=%s" % (
            phase, d["first_step"], len(d["steps"]), d["max_n"], sorted(d["links"])), flush=True)
    if TOUCH:
        last = TOUCH[-1]
        print("can_touch last=%s" % json.dumps(last), flush=True)


if __name__ == "__main__":
    import collection_episode
    sys.argv = ["collection_episode.py"] + sys.argv[1:]
    collection_episode.main()
