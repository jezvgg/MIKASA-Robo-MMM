"""Diagnostic driver: runs utils.test_planner for one seed and dumps a per-env-step record.

    PYTHONPATH=<code> python v61c_diag.py <out_dir> <test_planner args...>

The record (npz + json, <out_dir>/diag_seed<S>.*) holds, per env step: the commanded action, the measured
active-joint qpos, the TCP position, the base pose and the robot-vs-world contacts with a non-zero impulse.
The hook is installed from here (monkeypatch of install_gaze), so the planner snapshot stays clean.
"""
import atexit
import logging
import json
import sys
import traceback
from pathlib import Path

import numpy as np

import mani_skill  # noqa: F401  (must precede my_scenes)
import my_scenes  # noqa: F401
from planners.v61 import planner_entry
from utils import test_planner

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
logging.getLogger("planners.v61").setLevel(logging.DEBUG)
OUT = Path(sys.argv[1])
ARGS = sys.argv[2:]
SEED = int(ARGS[ARGS.index("--start-seed") + 1])
REC: dict = {"action": [], "qpos": [], "tcp": [], "base": [], "cup": [], "tray": [], "cup_q": [], "flags": [], "cup_vw": [], "tray_q": [], "contacts": []}
META: dict = {}
FLAGS = ("strict_success", "was_lifted", "held_over_tray", "released", "on_plate", "at_rest")


def _contacts(agent) -> list:
    mine = {l.entity.name for l in agent.robot._objs[0].links}
    out = set()
    for c in agent.robot.scene.px.get_contacts():
        n0, n1 = c.bodies[0].entity.name, c.bodies[1].entity.name
        if (n0 in mine) == (n1 in mine):
            continue
        if sum(float(np.linalg.norm(p.impulse)) for p in c.points) < 1e-4:
            continue
        a, b = (n0, n1) if n0 in mine else (n1, n0)
        out.add(f"{a}~{b}")
    return sorted(out)


def _patched(planner, agent, target_pos):
    trace = _orig(planner, agent, target_pos)
    raw = planner._guard.step
    u = planner.base_env.unwrapped
    try:
        META["counter_pos"] = [float(v) for v in np.asarray(u.counter_pos).reshape(-1)]
        META["counter_size"] = [float(v) for v in np.asarray(u.counter_size).reshape(-1)]
        META["cup_half"] = [float(v) for v in np.asarray(u.cup_half).reshape(-1)]
        META["cup_pos0"] = [float(v) for v in u.cup.pose.p[0].cpu().numpy()]
        META["cup_bottom"] = float(np.asarray(u._cup_bottom().cpu() if hasattr(u._cup_bottom(), "cpu") else u._cup_bottom()).reshape(-1)[0])
        META["tray_half"] = [float(v) for v in np.asarray(u.tray_half).reshape(-1)]
        META["tray_pos0"] = [float(v) for v in u.tray.pose.p[0].cpu().numpy()]
    except Exception:
        traceback.print_exc(file=sys.stdout)
    META["joints"] = [j.get_name() for j in agent.robot.get_active_joints()]

    def step(action, tape_entry=None):
        out = raw(action, tape_entry=tape_entry)
        REC["action"].append(np.asarray(action, dtype=np.float64).copy())
        REC["qpos"].append(agent.robot.get_qpos()[0].cpu().numpy().astype(np.float64))
        REC["tcp"].append(agent.tcp.pose.p[0].cpu().numpy().astype(np.float64))
        REC["base"].append(trace[-1].copy() if trace else np.zeros(3))
        REC["cup"].append(u.cup.pose.p[0].cpu().numpy().astype(np.float64))
        REC["tray"].append(u.tray.pose.p[0].cpu().numpy().astype(np.float64))
        REC["cup_q"].append(u.cup.pose.q[0].cpu().numpy().astype(np.float64))
        REC["tray_q"].append(u.tray.pose.q[0].cpu().numpy().astype(np.float64))
        info = out[4] if isinstance(out, tuple) and len(out) >= 5 else {}
        REC["flags"].append([float(np.asarray(info[k]).reshape(-1)[0]) if k in info else -1.0 for k in FLAGS])
        REC["cup_vw"].append([float(u.cup.linear_velocity.norm()), float(u.cup.angular_velocity.norm())])
        REC["contacts"].append(_contacts(agent))
        return out

    planner._guard.step = step
    return trace


def _dump() -> None:
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"diag_seed{SEED}.npz", action=np.array(REC["action"]), qpos=np.array(REC["qpos"]),
                            tcp=np.array(REC["tcp"]), cup=np.array(REC["cup"]), tray=np.array(REC["tray"]), base=np.array(REC["base"]),
                            cup_q=np.array(REC["cup_q"]), flags=np.array(REC["flags"]), cup_vw=np.array(REC["cup_vw"]), tray_q=np.array(REC["tray_q"]))
        (OUT / f"diag_seed{SEED}.json").write_text(json.dumps({"meta": META, "contacts": REC["contacts"]}))
    except Exception:
        traceback.print_exc(file=sys.stdout)


_orig = planner_entry.install_gaze
planner_entry.install_gaze = _patched
atexit.register(_dump)
sys.exit(test_planner.main(ARGS))
