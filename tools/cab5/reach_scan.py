"""Where a fist can press cab_1's leaf from inside: an mplib IK scan over stances and fist poses.

    PYTHONPATH=. python tools/cab5/reach_scan.py [--seed 7991002]

No episode runs. The env is reset, cab_1's leaf is SET to each angle of the push (crack, mid,
open), the base and torso are SET to each candidate stance, the planning world is refreshed, and
mplib's IK (base fixed, torso fixed, the scene in the world, as `static_manipulation` asks it) is
queried for the fist at the leaf's inner face. A stance-and-orientation counts when the press
point is reachable at EVERY angle of the push (the base may only turn in place between them).
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

PROFILES = Path(__file__).resolve().parents[2] / "utils" / "collection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7991002)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--side", choices=("inner", "outer"), default="inner",
                    help="inner: the opening press; outer: the closing push on the outer face")
    args = ap.parse_args()

    import gymnasium as gym
    import mplib
    import sapien

    import my_scenes  # noqa: F401
    from planners import cabinet_retrieval_planner as crp
    from planners import cabinet_search_planner as csp
    from planners import cabinet_door_fist as fd
    from planners.oracle import door_swing as ds
    from planners.oracle import oracle_common as common
    from planners.oracle.stance_scan import root_joints_for
    from utils.collection.profile import profile_env_kwargs

    profile = json.loads((PROFILES / "cabinet_search_profile.json").read_text())
    env = gym.make("MikasaCabinetSearch5-v0", **profile_env_kwargs(profile, obs_mode="state"))
    env.reset(seed=args.seed)
    task = env.unwrapped
    planner = common.collection_planner_factory(env, False, False)
    comp = task.cfg.compartments[0]
    door = csp.door_spec_for(comp, closed_rad=task.cfg.theta_closed)
    art = task.scene.articulations[crp._art_key(task, door)]
    names = [j.name for j in art.get_active_joints()]
    hinge_i = names.index(door.hinge)
    robot = task.agent.robot
    jm = robot.active_joints_map
    torso_i = int(jm["torso_lift_joint"].active_index[0])
    p = planner.planner
    mask = [True, True, True, True] + [False] * (len(robot.get_qpos()[0]) - 4)

    def set_door(rad):
        q = art.get_qpos().clone()
        q[0, hinge_i] = door.open_dir * rad
        art.set_qpos(q)

    fingers = [int(jm[n].active_index[0]) for n in jm if "finger" in n]

    def set_base(x, y, yaw, torso):
        q = robot.get_qpos().clone()
        q[0, :3] = torch_tensor(root_joints_for(task, x, y, yaw))
        q[0, torso_i] = torso
        for f in fingers:                    # a fist: the fingers shut
            q[0, f] = 0.0
        robot.set_qpos(q)

    import torch

    def torch_tensor(a):
        return torch.as_tensor(np.asarray(a), dtype=q_dtype)

    q_dtype = robot.get_qpos().dtype

    def ik_ok(T) -> bool:
        cur = robot.get_qpos()[0].cpu().numpy().astype(np.float64)
        cur_f = p.fold_qpos(cur)
        pose = sapien.Pose(p=T[:3, 3], q=fd._quat(T))
        goal_b = p._transform_goal_to_wrt_base(mplib.Pose(p=pose.p, q=pose.q))
        try:
            status, sol = p.IK(goal_b, cur_f, mask, n_init_qpos=args.n_init)
        except Exception:
            return False
        return status == "Success" and sol is not None and len(np.atleast_2d(sol)) > 0

    leaf = fd.Leaf(task, door)

    def pose_for(rad, along, off, z, yaw_off, pitch, closing):
        """Fist at the press point; approach = -u turned yaw_off toward the room, pitched up;
        fingers closing across the leaf ("across") or as near vertical as the approach allows."""
        sign = 1.0 if args.side == "inner" else -1.0
        xy = leaf.point(rad, along, sign * off)
        u, nin = leaf.u(rad), leaf.inner(rad)
        a2 = ds.rot2(sign * leaf.s * yaw_off) @ (-u)
        a = np.array([math.cos(pitch) * a2[0], math.cos(pitch) * a2[1], math.sin(pitch)])
        c = np.array([nin[0], nin[1], 0.0]) if closing == "across" else np.array([0.0, 0.0, 1.0])
        if abs(c @ a) > 0.95:
            c = np.array([nin[0], nin[1], 0.0])
        c = c - (c @ a) * a
        c = c / np.linalg.norm(c)
        return fd._T(task.agent.build_grasp_pose(a, c, np.array([xy[0], xy[1], z])))

    if args.side == "inner":
        rads = (fd.CRACK_RAD, 0.9, fd.OPEN_RAD)
        stances = list(itertools.product((0.85, 0.95, 1.05, 1.15), (-1.25, -1.10, -0.95)))
    else:
        rads = (fd.OPEN_RAD, 0.9, 0.5, 0.1)
        stances = list(itertools.product((0.55, 0.70, 0.85, 1.0), (-1.45, -1.30, -1.15)))
    torsos = (0.20, 0.386)
    yaw_offs = (0.0, 0.35, 0.7, 1.05)
    pitches = (0.0, math.radians(35), math.radians(70), math.radians(90))
    zs = (1.50,)
    alongs = (0.08,)
    closings = ("across", "vertical")
    offsets = (0.06, 0.09)
    rows = []
    for (x, y), torso in itertools.product(stances, torsos):
        for yaw_off, pitch, z, along, closing, off in itertools.product(yaw_offs, pitches, zs, alongs,
                                                                         closings, offsets):
            ok = []
            for rad in rads:
                set_door(rad)
                press = leaf.point(rad, along, off if args.side == "inner" else -off)
                yaw = math.atan2(press[1] - y, press[0] - x)
                set_base(x, y, yaw, torso)
                planner.planner.update_from_simulation()
                ok.append(ik_ok(pose_for(rad, along, off, z, yaw_off, pitch, closing)))
            rows.append(dict(x=x, y=y, torso=torso, yaw_off=yaw_off, pitch_deg=round(math.degrees(pitch)),
                             z=z, along=along, closing=closing, off=off, ok=ok, all=all(ok)))
        good = [r for r in rows if r["x"] == x and r["y"] == y and r["torso"] == torso and r["all"]]
        print(f"stance ({x}, {y}) torso {torso}: {len(good)} poses reachable at every angle "
              f"{[(r['yaw_off'], r['pitch_deg'], r['closing'], r['off']) for r in good][:8]}", flush=True)
    by = {}
    for r in rows:
        by.setdefault((r["yaw_off"], r["pitch_deg"], r["closing"], r["off"]), []).append(r["ok"])
    print("per (yaw_off rad, pitch deg, closing, offset): reachable fraction at crack / 0.9 / open")
    for k, v in sorted(by.items()):
        a = np.array(v, dtype=float).mean(axis=0)
        print(f"  {k}: {np.round(a, 2).tolist()}")
    Path(f"reach_scan_{args.side}.json").write_text(json.dumps(rows))
    env.close()


if __name__ == "__main__":
    main()
