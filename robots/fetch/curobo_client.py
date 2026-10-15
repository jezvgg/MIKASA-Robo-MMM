"""The simulator side of the cuRobo IK backend (``MIKASA_IK_BACKEND=curobo``).

Starts `curobo_server.py` once per process under the python named by
``MIKASA_CUROBO_PY`` (cuRobo's own venv; the simulator's venv never imports cuRobo) and
asks it for IK proposals. `ik_backend.ik` refines each proposal with mplib's CLIK and
keeps it only if mplib's FCL check passes, so a proposal is a start, never an answer.

Frames: mplib's goal is in the planning model's frame; the server's variants are rooted
at base_link (torso free) or torso_lift_link (torso held, i.e. masked by the caller), so
the goal is re-expressed in that link's frame from mplib's own FK at the start qpos.

``MIKASA_CUROBO_SEEDS`` (default 32) is how many proposals one call asks for; failures
of the server are raised, never silently replaced by mplib.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np

PY_ENV, SEEDS_ENV, CACHE_ENV = "MIKASA_CUROBO_PY", "MIKASA_CUROBO_SEEDS", "MIKASA_CUROBO_CACHE"
SERVER_ENV = "MIKASA_CUROBO_SERVER"
PREFIX = "CUROBO "
TIMEOUT_S = float(os.environ.get("MIKASA_CUROBO_TIMEOUT_S", "900"))   # cold start builds the model
ARM = ("shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint",
       "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint")
_SERVER = {}
STATS = dict(calls=0, proposals=0, server_ms=0.0, wall_ms=0.0, start_s=None)


class _Server:
    def __init__(self, python: str):
        here = Path(__file__).resolve().parent
        script = os.environ.get(SERVER_ENV) or str(here / "curobo_server.py")    # a stub in tests
        cmd = [python, "-u", script, "--urdf", str(here / "fetch.urdf")]
        if os.environ.get(CACHE_ENV):
            cmd += ["--cache", os.environ[CACHE_ENV]]
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}
        env["PYTHONNOUSERSITE"] = "1"
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     env=env, start_new_session=True)
        self.lines: Queue = Queue()
        self.next_id = 1
        threading.Thread(target=self._pump, daemon=True).start()
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"curobo server did not start: {ready}")
        STATS["start_s"] = round(time.perf_counter() - t0, 1)
        atexit.register(self.close)

    def _pump(self):
        for line in self.proc.stdout:
            if line.startswith(PREFIX):
                self.lines.put(line[len(PREFIX):])
        self.lines.put("")

    def _read(self) -> dict:
        try:
            line = self.lines.get(timeout=TIMEOUT_S)
        except Empty:
            raise RuntimeError(f"curobo server: no reply in {TIMEOUT_S:.0f} s") from None
        if not line:
            raise RuntimeError(f"curobo server exited (rc={self.proc.poll()})")
        return json.loads(line)

    def call(self, op: str, **args) -> dict:
        rid, self.next_id = self.next_id, self.next_id + 1
        self.proc.stdin.write(json.dumps(dict(id=rid, op=op, args=args)) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        if "error" in reply:
            raise RuntimeError(f"curobo server: {reply['error']}")
        return reply

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def server() -> _Server:
    if "s" not in _SERVER:
        python = os.environ.get(PY_ENV)
        if not python:
            raise RuntimeError(f"MIKASA_IK_BACKEND=curobo needs {PY_ENV}: the python of a venv with cuRobo")
        _SERVER["s"] = _Server(python)
        atexit.register(_report)
    return _SERVER["s"]


def _report():
    if STATS["calls"]:
        print(f"[curobo_client] calls={STATS['calls']} proposals={STATS['proposals']} "
              f"server_ms={STATS['server_ms']:.0f} wall_ms={STATS['wall_ms']:.0f} start_s={STATS['start_s']}",
              file=sys.stderr, flush=True)


def _matrix(pose) -> np.ndarray:
    """mplib.Pose (p, q wxyz) -> 4x4."""
    w, x, y, z = (float(v) for v in np.asarray(pose.q).reshape(-1))
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = np.asarray(pose.p, dtype=np.float64).reshape(-1)
    return T


def _index(names, suffix: str) -> int:
    """Index of the one name equal to ``suffix`` or ending with ``_suffix`` (mplib prefixes names)."""
    hits = [i for i, n in enumerate(names) if n == suffix or n.endswith("_" + suffix)]
    if len(hits) != 1:
        raise KeyError(f"{suffix!r}: {len(hits)} matches among {list(names)}")
    return hits[0]


def propose(planner, goal_pose, start_qpos, mask) -> list:
    """Full-length qpos starts from cuRobo, best first, for `ik_backend.solve_from_starts`."""
    joint_names = list(planner.user_joint_names)
    mask = np.zeros(len(start_qpos), dtype=bool) if mask is None or len(mask) == 0 else np.asarray(mask, bool)
    torso = _index(joint_names, "torso_lift_joint")
    variant, frame = ("arm", "torso_lift_link") if mask[torso] else ("torso_arm", "base_link")
    links = list(planner.link_name_2_idx)
    planner.pinocchio_model.compute_forward_kinematics(np.asarray(start_qpos, dtype=np.float64))
    T_frame = _matrix(planner.pinocchio_model.get_link_pose(planner.link_name_2_idx[links[_index(links, frame)]]))
    goal = np.linalg.inv(T_frame) @ _matrix(goal_pose)
    s = server()
    names = _SERVER.setdefault(("names", variant), s.call("names", variant=variant)["names"])
    cols = [_index(joint_names, n) for n in names]
    t0 = time.perf_counter()
    k = int(os.environ.get(SEEDS_ENV, "32"))
    r = s.call("propose", variant=variant, goal=goal.tolist(), seed=[float(start_qpos[c]) for c in cols], k=k)
    STATS["calls"] += 1
    STATS["server_ms"] += r["ms"]
    STATS["wall_ms"] += (time.perf_counter() - t0) * 1e3
    q, ok = np.asarray(r["q"], dtype=np.float64), np.asarray(r["ok"], dtype=bool)
    order = list(np.flatnonzero(ok)) or list(range(min(4, len(q))))    # nothing solved: still try the best few
    starts = []
    for i in order:
        full = np.asarray(start_qpos, dtype=np.float64).copy()
        full[cols] = q[i]
        starts.append(full)
    STATS["proposals"] += len(starts)
    return starts
