"""cuRobo IK proposals for ds_fetch, served over a pipe. Runs under cuRobo's OWN venv.

    <curobo venv>/bin/python curobo_server.py --urdf robots/fetch/fetch.urdf [--cache DIR]

Started by `curobo_client.py`; never imported by the simulator's venv (cuRobo pins its own
torch, warp and numpy, and switches TF32 on globally). Requests and replies are JSON
lines; replies carry the prefix ``CUROBO `` so cuRobo's own prints cannot be mistaken
for one.

Robot model: built once from the repository's URDF with cuRobo's RobotBuilder
(collision spheres fitted to the link meshes, the SRDF's disabled pairs added to the
self-collision ignore matrix) and cached under ``--cache`` by the URDF's hash. Two
planning variants, as the collection planner masks its IK:

* ``torso_arm``: base_link -> gripper_link, torso + 7 arm joints;
* ``arm``: torso_lift_link -> gripper_link, the 7 arm joints (torso held). The base body
  is not in this chain, so arm-versus-base contacts are left to the caller's FCL check.

Head and fingers are locked. One solver per (variant, self-collision) with batch 1:
the planner asks one goal at a time.

Ops: ``names`` {variant} -> joint names; ``propose`` {variant, goal (4x4 in the variant's
base frame), seed (joint values), k} -> up to k solutions seeded with ``seed``, cuRobo's
best first, with their success flags; ``warmup`` {variants}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PREFIX = "CUROBO "
LOCK = {"head_pan_joint": 0.0, "head_tilt_joint": 0.0,
        "r_gripper_finger_joint": 0.05, "l_gripper_finger_joint": 0.05}
ROOTS = ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint")
#: Links outside a variant base's subtree have no chain from it and are dropped.
OUTSIDE = {"torso_lift_link": {"base_link", "r_wheel_link", "l_wheel_link", "estop_link",
                               "laser_link", "torso_fixed_link"}}
VARIANTS = {"torso_arm": ("base_link", ROOTS), "arm": ("torso_lift_link", ROOTS + ("torso_lift_joint",))}
NUM_SEEDS = 32


def build_configs(urdf: Path, out: Path) -> dict:
    """fetch_<variant>.yml for both variants, built from ``urdf`` if not cached."""
    import numpy as np
    import torch
    import yaml

    paths = {v: out / f"fetch_{v}.yml" for v in VARIANTS}
    if all(p.exists() for p in paths.values()):
        return paths
    from curobo.robot_builder import RobotBuilder

    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    b = RobotBuilder(str(urdf), str(urdf.parent), tool_frames=["gripper_link"])
    b.fit_collision_spheres(sphere_density=1.0)
    b.compute_collision_matrix()
    srdf = urdf.with_suffix(".srdf")
    if srdf.exists():
        names = set(b.collision_link_names)
        for d in ET.parse(srdf).getroot().findall("disable_collisions"):
            a, c = d.get("link1"), d.get("link2")
            if a in names and c in names:
                b.add_collision_ignore(a, [c])
    full = out / "fetch_full.yml"
    b.save(b.build(), str(full))
    for variant, (base, drop) in VARIANTS.items():
        v = yaml.safe_load(full.read_text())
        k = v["kinematics"]
        k["base_link"], k["lock_joints"] = base, dict(LOCK)
        cs = k["cspace"]
        names = list(cs["joint_names"])
        keep = [i for i, j in enumerate(names) if j not in drop]
        for key, val in list(cs.items()):
            if isinstance(val, list) and len(val) == len(names):
                cs[key] = [val[i] for i in keep]
        gone = OUTSIDE.get(base, set())
        if gone:
            k["collision_link_names"] = [n for n in k["collision_link_names"] if n not in gone]
            k["mesh_link_names"] = [n for n in k.get("mesh_link_names") or [] if n not in gone]
            for key in ("collision_spheres", "self_collision_buffer"):
                if isinstance(k.get(key), dict):
                    k[key] = {n: x for n, x in k[key].items() if n not in gone}
            if isinstance(k.get("self_collision_ignore"), dict):
                k["self_collision_ignore"] = {n: [m for m in x if m not in gone]
                                              for n, x in k["self_collision_ignore"].items() if n not in gone}
        paths[variant].write_text(yaml.safe_dump(v, sort_keys=False))
    return paths


def no_grad_graph_replays() -> None:
    """Replay cuRobo's CUDA graphs without autograd.

    At 78fd485 every IK call copies a fresh grad-requiring seed tensor into a graph input
    with autograd on, and each copy keeps the previous one alive: BIMBO_EvoX measured
    1.43 MiB of GPU memory per call, never freed. A replay needs no autograd; the
    recording call is left as it is.
    """
    import torch
    from curobo._src.util import cuda_graph_util as cgu

    original = cgu.GraphExecutor.__call__
    if getattr(original, "_mikasa_no_grad", False):
        return

    def call(self, *inputs, **kwargs):
        if getattr(self, "_graph", None) is not None:
            with torch.no_grad():
                return original(self, *inputs, **kwargs)
        return original(self, *inputs, **kwargs)

    call._mikasa_no_grad = True
    cgu.GraphExecutor.__call__ = call


class Solvers:
    def __init__(self, paths: dict):
        self.paths, self.ik = paths, {}

    def get(self, variant: str, coll: bool = True):
        key = (variant, coll)
        if key not in self.ik:
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

            self.ik[key] = InverseKinematics(InverseKinematicsCfg.create(
                robot=str(self.paths[variant]), num_seeds=NUM_SEEDS, max_batch_size=1,
                self_collision_check=coll, load_collision_spheres=coll,
                position_tolerance=0.002, orientation_tolerance=0.02))
        return self.ik[key]

    def propose(self, variant: str, goal, seed, k: int, coll: bool = True) -> dict:
        import numpy as np
        import torch
        from curobo.types import GoalToolPose, Pose

        ik = self.get(variant, coll)
        T = np.asarray(goal, dtype=np.float64).reshape(4, 4)
        quat = _wxyz(T[:3, :3])
        pose = Pose(position=torch.tensor(T[None, :3, 3], dtype=torch.float32, device="cuda"),
                    quaternion=torch.tensor(quat[None], dtype=torch.float32, device="cuda"))
        dof = len(ik.joint_names)
        seed_t = torch.tensor(np.asarray(seed, dtype=np.float32).reshape(1, 1, dof), device="cuda")
        ik.reset_seed()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = ik.solve_pose(GoalToolPose.from_poses({ik.tool_frames[0]: pose}, num_goalset=1),
                          seed_config=seed_t, return_seeds=k)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        ok = r.success.reshape(-1).cpu().numpy().astype(bool)
        sol = r.solution.reshape(-1, dof).double().cpu().numpy()[: len(ok)]
        return dict(names=list(ik.joint_names), q=sol.tolist(), ok=ok.tolist(), ms=ms)


def _wxyz(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), numerically safe branch."""
    import numpy as np

    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "mikasa-curobo")
    args = ap.parse_args()
    urdf = args.urdf.resolve()
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()[:16]
    solvers = Solvers(build_configs(urdf, args.cache / digest))
    no_grad_graph_replays()
    out = sys.stdout
    out.write(PREFIX + json.dumps({"id": 0, "ready": True, "cache": str(args.cache / digest)}) + "\n")
    out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        try:
            op, a = req["op"], req.get("args", {})
            if op == "names":
                reply = dict(names=list(solvers.get(a["variant"], a.get("coll", True)).joint_names))
            elif op == "propose":
                reply = solvers.propose(a["variant"], a["goal"], a["seed"], int(a.get("k", NUM_SEEDS)),
                                        a.get("coll", True))
            elif op == "warmup":
                for v in a.get("variants", list(VARIANTS)):
                    ik = solvers.get(v, a.get("coll", True))
                    import numpy as np
                    solvers.propose(v, np.eye(4).tolist(), [0.0] * len(ik.joint_names), 4, a.get("coll", True))
                reply = dict(ok=True)
            else:
                raise ValueError(f"unknown op {op!r}")
            reply["id"] = req.get("id")
        except Exception as exc:  # the client decides; the server stays up
            reply = dict(id=req.get("id"), error=f"{type(exc).__name__}: {exc}")
        out.write(PREFIX + json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
