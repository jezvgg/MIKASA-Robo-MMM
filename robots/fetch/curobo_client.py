"""The simulator side of the cuRobo IK backend (``MIKASA_IK_BACKEND=curobo``).

Starts `curobo_server.py` once per process under the python named by
``MIKASA_CUROBO_PY`` (cuRobo's own venv; the simulator's venv never imports cuRobo) and
asks it for IK proposals. `ik_backend.ik` refines each proposal with mplib's CLIK and
keeps it only if mplib's FCL check passes, so a proposal is a start, never an answer.

Frames: mplib's goal is in the planning model's frame; the server's variants are rooted
at base_link (torso free) or torso_lift_link (torso held, i.e. masked by the caller), so
the goal is re-expressed in that link's frame from mplib's own FK at the start qpos.

``MIKASA_CUROBO_SOCKET`` names a shared server started once with ``--socket`` (see
`curobo_server.py`): the client connects to it instead of starting its own, so the
start-up is not paid per episode. ``MIKASA_CUROBO_SEEDS`` (default 32) is how many
proposals one call asks for; failures of the server are raised, never silently replaced
by mplib.

Two of BIMBO_EvoX's guards (skills/gpu-ik/gpu_ik.py, tasks/62_halve_an_egg/.../ik.py):

* ``MIKASA_CUROBO_MAX_JUMP`` (rad, default 0 = off): a proposal whose largest joint
  change from the current joints exceeds it is on another branch and is dropped; the
  rest go nearest first. The current joints themselves are always tried first
  (`ik_backend.ik`). OFF by default because our planners ask for postures far from the
  current one by design (rest -> reach): at 1.2 (BIMBO's servo value) SeasonDish went
  from 28/30 to 1/30 and Depth Recall from 100/100 to 0/200 (measured 2026-10-16).
* ``scan(..., verify=True)``: every hit is re-checked in float64 against the URDF
  (`urdf_fk`), pose within ``VERIFY_TOL`` and joints within their limits; a claim that
  does not hold is dropped and counted in ``STATS["verify_dropped"]``.
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

from robots.fetch.urdf_fk import pose_error

PY_ENV, SEEDS_ENV, CACHE_ENV = "MIKASA_CUROBO_PY", "MIKASA_CUROBO_SEEDS", "MIKASA_CUROBO_CACHE"
SERVER_ENV, SOCKET_ENV = "MIKASA_CUROBO_SERVER", "MIKASA_CUROBO_SOCKET"
MAX_JUMP_ENV = "MIKASA_CUROBO_MAX_JUMP"
#: BIMBO_EvoX's tolerance for its float64 re-check: metres, radians.
VERIFY_TOL = (0.005, 0.05)
PREFIX = "CUROBO "
TIMEOUT_S = float(os.environ.get("MIKASA_CUROBO_TIMEOUT_S", "900"))   # cold start builds the model
ARM = ("shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint",
       "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint")
_SERVER = {}
STATS = dict(calls=0, proposals=0, server_ms=0.0, wall_ms=0.0, start_s=None,
             jump_dropped=0, verify_checked=0, verify_dropped=0)
_CHAINS = {}


class _Server:
    def __init__(self, python: str | None, socket_path: str | None = None):
        t0 = time.perf_counter()
        self.lines: Queue = Queue()
        self.next_id = 1
        self.proc = self.sock = None
        if socket_path:
            import socket

            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(socket_path)
            self.reader, self.writer = self.sock.makefile("r"), self.sock.makefile("w")
        else:
            self._spawn(python)
            self.reader, self.writer = self.proc.stdout, self.proc.stdin
        threading.Thread(target=self._pump, daemon=True).start()
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"curobo server did not start: {ready}")
        STATS["start_s"] = round(time.perf_counter() - t0, 1)
        atexit.register(self.close)

    def _spawn(self, python: str):
        here = Path(__file__).resolve().parent
        script = os.environ.get(SERVER_ENV) or str(here / "curobo_server.py")    # a stub in tests
        cmd = [python, "-u", script, "--urdf", str(here / "fetch.urdf")]
        if os.environ.get(CACHE_ENV):
            cmd += ["--cache", os.environ[CACHE_ENV]]
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}
        env["PYTHONNOUSERSITE"] = "1"
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     env=env, start_new_session=True)

    def _pump(self):
        for line in self.reader:
            if line.startswith(PREFIX):
                self.lines.put(line[len(PREFIX):])
        self.lines.put("")

    def _read(self) -> dict:
        try:
            line = self.lines.get(timeout=TIMEOUT_S)
        except Empty:
            raise RuntimeError(f"curobo server: no reply in {TIMEOUT_S:.0f} s") from None
        if not line:
            raise RuntimeError("curobo server closed the connection" if self.sock else
                               f"curobo server exited (rc={self.proc.poll()})")
        return json.loads(line)

    def call(self, op: str, **args) -> dict:
        rid, self.next_id = self.next_id, self.next_id + 1
        self.writer.write(json.dumps(dict(id=rid, op=op, args=args)) + "\n")
        self.writer.flush()
        reply = self._read()
        if "error" in reply:
            raise RuntimeError(f"curobo server: {reply['error']}")
        return reply

    def close(self):
        if self.sock is not None:          # a shared server stays up for the next client
            self.sock.close()
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def server() -> _Server:
    if "s" not in _SERVER:
        python, sock = os.environ.get(PY_ENV), os.environ.get(SOCKET_ENV)
        if not python and not sock:
            raise RuntimeError(f"MIKASA_IK_BACKEND=curobo needs {SOCKET_ENV} (a shared server) or "
                               f"{PY_ENV} (the python of a venv with cuRobo)")
        _SERVER["s"] = _Server(python, sock)
        atexit.register(_report)
    return _SERVER["s"]


def _report():
    if STATS["calls"]:
        print(f"[curobo_client] calls={STATS['calls']} proposals={STATS['proposals']} "
              f"jump_dropped={STATS['jump_dropped']} server_ms={STATS['server_ms']:.0f} "
              f"wall_ms={STATS['wall_ms']:.0f} start_s={STATS['start_s']}",
              file=sys.stderr, flush=True)


def max_jump() -> float:
    v = float(os.environ.get(MAX_JUMP_ENV, "0"))
    if v < 0:
        raise ValueError(f"{MAX_JUMP_ENV}={v}: expected >= 0 (0 = off)")
    return v


def near_first(q: np.ndarray, ok: np.ndarray, seed: np.ndarray, jump: float, periodic=None) -> tuple:
    """(rows to try, how many solved rows the jump dropped).

    Solved rows within ``jump`` of ``seed`` (largest single-joint change), nearest first by
    L2; with none solved, the first four rows as plain starts, under the same jump rule.
    ``jump`` 0 keeps every row. ``periodic`` marks the columns whose range exceeds a turn
    (the roll joints): their change is taken modulo 2 pi, because the planner unwraps a
    goal toward the current joints before it moves (`unwrap_toward`).
    """
    delta = np.asarray(q, dtype=np.float64) - seed
    if periodic is not None and np.any(periodic):
        per = np.asarray(periodic, dtype=bool)
        delta[:, per] = (delta[:, per] + np.pi) % (2 * np.pi) - np.pi
    d = np.linalg.norm(delta, axis=1)
    far = np.abs(delta).max(axis=1) > jump if jump > 0 else np.zeros(len(q), bool)
    solved = np.flatnonzero(ok)
    keep = [int(i) for i in solved[np.argsort(d[solved], kind="stable")] if not far[i]]
    if len(solved):
        return keep, int(len(solved) - len(keep))
    return [i for i in range(min(4, len(q))) if not far[i]], 0


def variant_base(variant: str) -> str:
    return "torso_lift_link" if variant == "arm" else "base_link"


def chain(base: str = "base_link"):
    """The URDF chain from ``base`` to gripper_link, parsed once."""
    if base not in _CHAINS:
        from robots.fetch.urdf_fk import Chain

        _CHAINS[base] = Chain(Path(__file__).resolve().parent / "fetch.urdf", base, "gripper_link")
    return _CHAINS[base]


def verify(goals: np.ndarray, q: np.ndarray, ok: np.ndarray, names, tol=VERIFY_TOL,
           base: str = "base_link") -> np.ndarray:
    """``ok`` with every claim dropped that float64 FK of the URDF does not confirm.

    ``goals`` (..., 4, 4) gripper_link poses in ``base``'s frame, ``q`` (..., len(names))
    the solutions, ``ok`` the solver's flags; returns a new mask of the same shape.
    """
    ok = np.asarray(ok, dtype=bool).copy()
    hits = np.flatnonzero(ok.reshape(-1))
    if len(hits) == 0:
        return ok
    c = chain(base)
    Q = np.asarray(q, dtype=np.float64).reshape(-1, len(names))[hits]
    G = np.asarray(goals, dtype=np.float64).reshape(-1, 4, 4)[hits]
    pos, rot = pose_error(c.fk_many(Q, names), G)
    lo, hi = c.limits(names)
    inside = np.all((Q >= lo - 1e-6) & (Q <= hi + 1e-6), axis=1)
    good = (pos < tol[0]) & (rot < tol[1]) & inside
    flat = ok.reshape(-1)
    flat[hits[~good]] = False
    STATS["verify_checked"] += int(len(hits))
    STATS["verify_dropped"] += int((~good).sum())
    return flat.reshape(ok.shape)


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
    seed = np.asarray([float(start_qpos[c]) for c in cols], dtype=np.float64)
    lo, hi = chain(variant_base(variant)).limits(names)
    order, dropped = near_first(q, ok, seed, max_jump(), periodic=(hi - lo) > 2 * np.pi)
    STATS["jump_dropped"] += dropped
    starts = []
    for i in order:
        full = np.asarray(start_qpos, dtype=np.float64).copy()
        full[cols] = q[i]
        starts.append(full)
    STATS["proposals"] += len(starts)
    return starts


def stand_matrix(x: float, y: float, yaw: float) -> np.ndarray:
    """base_link on the floor at (x, y) facing ``yaw`` (the root joints' pose), as a 4x4."""
    T = np.eye(4)
    c, s_ = np.cos(yaw), np.sin(yaw)
    T[:2, :2] = [[c, -s_], [s_, c]]
    T[:2, 3] = (x, y)
    return T


def scan(goals_world, stands, *, coll: bool = False, verify_hits: bool = True, tol=VERIFY_TOL):
    """Which (stand, goal) pairs the arm reaches with the torso free: a ``(len(stands),
    len(goals))`` boolean mask from ONE batched cuRobo call (plus its solutions).

    ``goals_world`` are gripper_link poses (N, 4, 4) in the world; ``stands`` (M, 3) are
    base poses (x, y, yaw). Kinematics only (and self-collision with ``coll``): the scene is
    the caller's to check. BIMBO_EvoX's rule applies: use the mask to choose WHERE to stand
    or WHAT to reach for; never move to a scan solution, it may be on any branch — solve
    the motion from the current joints. ``verify_hits`` (BIMBO's ``verify=True``) keeps
    a hit only if float64 FK of the URDF confirms it (`verify`).
    """
    G = np.asarray(goals_world, dtype=np.float64).reshape(-1, 4, 4)
    S = np.asarray(stands, dtype=np.float64).reshape(-1, 3)
    rel = np.stack([np.linalg.inv(stand_matrix(*st)) @ G for st in S])          # (M, N, 4, 4)
    t0 = time.perf_counter()
    r = server().call("scan", variant="torso_arm", goals=rel.reshape(-1, 16).tolist(), coll=bool(coll))
    STATS["calls"] += 1
    STATS["server_ms"] += r["ms"]
    STATS["wall_ms"] += (time.perf_counter() - t0) * 1e3
    ok = np.asarray(r["ok"], dtype=bool).reshape(len(S), len(G))
    q = np.asarray(r["q"], dtype=np.float64).reshape(len(S), len(G), -1)
    if verify_hits:
        ok = verify(rel, q, ok, r["names"], tol)
    return ok, q
