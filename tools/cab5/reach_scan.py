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
    ap.add_argument("--open-rad", type=float, default=None,
                    help="--into: the leaf's open angle to scan (default OPEN_RAD)")
    ap.add_argument("--rads", default=None, help="comma list of leaf angles (overrides the mode's)")
    ap.add_argument("--along", action="store_true",
                    help="inner face: a horizontal fist pointing along the leaf toward the hinge (into the "
                         "gap), as from the rest pose with the arm straightened; pushes with its side")
    ap.add_argument("--yaws", default=None, help="comma list of yaw offsets (rad)")
    ap.add_argument("--stances", default=None, help="x1:x2:dx,y1:y2:dy grid")
    ap.add_argument("--into", action="store_true",
                    help="the family's closing-push fist: pointing INTO the face (horizontal-ish), "
                         "fingers closing vertically, the TCP 3 cm off the face")
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

    roll_idx = [int(jm[n].active_index[0]) for n in ("upperarm_roll_joint", "forearm_roll_joint",
                                                       "wrist_roll_joint")]
    rest_roll = robot.get_qpos()[0].cpu().numpy().astype(np.float64)[roll_idx]
    from robots.fetch.utils import unwrap_toward

    def ik_need(T):
        """None if no IK; else the smallest, over solutions, of the largest roll excursion from
        the reset posture's rolls (what the episode's roll window would have to grow by)."""
        cur = robot.get_qpos()[0].cpu().numpy().astype(np.float64)
        cur_f = p.fold_qpos(cur)
        pose = sapien.Pose(p=T[:3, 3], q=fd._quat(T))
        goal_b = p._transform_goal_to_wrt_base(mplib.Pose(p=pose.p, q=pose.q))
        try:
            status, sol = p.IK(goal_b, cur_f, mask, n_init_qpos=args.n_init)
        except Exception:
            return None
        if status != "Success" or sol is None or len(np.atleast_2d(sol)) == 0:
            return None
        best = np.inf
        for g in np.atleast_2d(sol):
            g = unwrap_toward(g, cur_f, p.joint_limits)
            best = min(best, float(np.max(np.abs(np.asarray(g)[roll_idx] - rest_roll))))
        return best

    def ik_ok(T) -> bool:
        return ik_need(T) is not None

    leaf = fd.Leaf(task, door)

    def pose_for(rad, along, off, z, yaw_off, pitch, closing):
        """Fist at the press point; approach = -u turned yaw_off toward the room, pitched up;
        fingers closing across the leaf ("across") or as near vertical as the approach allows."""
        sign = 1.0 if args.side == "inner" else -1.0
        xy = leaf.point(rad, along, sign * off)
        u, nin = leaf.u(rad), leaf.inner(rad)
        a2 = ds.rot2(sign * leaf.s * yaw_off) @ (-u)
        a = np.array([math.cos(pitch) * a2[0], math.cos(pitch) * a2[1], math.sin(pitch)])
        c = {"across": np.array([nin[0], nin[1], 0.0]), "vertical": np.array([0.0, 0.0, 1.0]),
             "diag": np.array([nin[0], nin[1], 1.0]) / math.sqrt(2.0)}[closing]
        if abs(c @ a) > 0.95:
            c = np.array([nin[0], nin[1], 0.0])
        c = c - (c @ a) * a
        c = c / np.linalg.norm(c)
        return fd._T(task.agent.build_grasp_pose(a, c, np.array([xy[0], xy[1], z])))

    def into_pose(rad, r_hinge, yaw_off, pitch, stand=0.06):
        """The fist pointing into the side's face (turned yaw_off about z, pitched), r_hinge out
        along the leaf, 3 cm off the face, fingers closing as near vertical as allowed."""
        sign = 1.0 if args.side == "inner" else -1.0
        xy = leaf.point(rad, leaf.L - r_hinge, sign * stand)
        n_side = sign * leaf.inner(rad)
        a2 = ds.rot2(yaw_off) @ (-n_side)
        a = np.array([math.cos(pitch) * a2[0], math.cos(pitch) * a2[1], math.sin(pitch)])
        c = np.array([0.0, 0.0, 1.0]) - math.sin(pitch) * a
        c = c / np.linalg.norm(c)
        return fd._T(task.agent.build_grasp_pose(a, c, np.array([xy[0], xy[1], 1.59])))

    def grid(spec):
        xs, ys = spec.split(",")
        a = [float(v) for v in xs.split(":")]
        b = [float(v) for v in ys.split(":")]
        return list(itertools.product(np.arange(a[0], a[1] + 1e-6, a[2]).round(3),
                                      np.arange(b[0], b[1] + 1e-6, b[2]).round(3)))

    if args.along:
        rads = tuple(float(v) for v in args.rads.split(",")) if args.rads else (0.9, 1.2, 1.55)
        stances = grid(args.stances) if args.stances else list(itertools.product(
            (0.85, 0.95, 1.05, 1.15), (-1.35, -1.2, -1.05, -0.95)))
        yaws = tuple(float(v) for v in args.yaws.split(",")) if args.yaws else (-0.3, 0.0, 0.3)
        rows = []
        for (x, y), torso in itertools.product(stances, (0.20, 0.386)):
            for yaw_off, closing, stand, along, z in itertools.product(yaws, ("vertical", "across"), (0.09, 0.12),
                                                                     (0.08, 0.15), (1.55,)):
                ok, needs = [], []
                for rad in rads:
                    set_door(rad)
                    T = pose_for(rad, along, stand, z, yaw_off, 0.0, closing)
                    yaw = math.atan2(T[1, 3] - y, T[0, 3] - x)
                    set_base(x, y, yaw, torso)
                    planner.planner.update_from_simulation()
                    need = ik_need(T)
                    ok.append(need is not None)
                    needs.append(need)
                rows.append(dict(x=float(x), y=float(y), torso=torso, yaw_off=yaw_off, closing=closing,
                                 stand=stand, along=along, ok=ok, all=all(ok),
                                 roll_need=(max(needs) if all(ok) else None)))
            good = [r for r in rows if r["x"] == x and r["y"] == y and r["torso"] == torso and r["all"]]
            print(f"stance ({x}, {y}) torso {torso}: {len(good)} along-the-leaf poses reachable at every angle",
                  flush=True)
        by = {}
        for r in rows:
            by.setdefault((r["yaw_off"], r["closing"], r["stand"], r["along"]), []).append(r)
        print(f"rads {rads}; per (yaw_off, closing, stand, along): reachable fraction per angle | stances at all | min roll")
        for k, v in sorted(by.items()):
            a = np.array([r["ok"] for r in v], dtype=float).mean(axis=0)
            n = [r["roll_need"] for r in v if r["roll_need"] is not None]
            print(f"  {k}: {np.round(a, 2).tolist()} | {len(n)} | {round(min(n), 2) if n else '-'}")
        best = sorted((r for r in rows if r["roll_need"] is not None), key=lambda r: r["roll_need"])[:10]
        print("lowest roll:")
        for r in best:
            print("  ", r)
        env.close()
        return

    if args.into:
        top = fd.OPEN_RAD if args.open_rad is None else args.open_rad
        if args.side == "inner":
            rads = (0.9, 0.5 * (0.9 + top), top)
            stances = list(itertools.product((0.95, 1.05, 1.15, 1.25), (-0.95, -1.05, -1.20, -1.35)))
        else:
            rads = (top, 0.9, 0.5, 0.1)
            stances = list(itertools.product((0.45, 0.60, 0.75, 0.90), (-1.15, -1.30, -1.45)))
        if args.rads:
            rads = tuple(float(v) for v in args.rads.split(","))
        if args.stances:
            stances = grid(args.stances)
        rows = []
        for (x, y), torso in itertools.product(stances, (0.20, 0.386)):
            yaws = (tuple(float(v) for v in args.yaws.split(",")) if args.yaws else
                    ((-0.9, -0.6, -0.3) if args.side == "inner" else (0.3, 0.6, 0.9)))
            for yaw_off, pitch, r_h, stand in itertools.product(yaws, (0.35, 0.6), (0.30, 0.38), (0.09,)):
                ok, needs = [], []
                for rad in rads:
                    set_door(rad)
                    T = into_pose(rad, r_h, yaw_off, pitch, stand)
                    yaw = math.atan2(T[1, 3] - y, T[0, 3] - x)
                    set_base(x, y, yaw, torso)
                    planner.planner.update_from_simulation()
                    need = ik_need(T)
                    ok.append(need is not None)
                    needs.append(need)
                rows.append(dict(x=x, y=y, torso=torso, yaw_off=yaw_off, pitch=pitch, r=r_h, stand=stand, ok=ok,
                                 all=all(ok), roll_need=(max(needs) if all(ok) else None)))
            good = [r for r in rows if r["x"] == x and r["y"] == y and r["torso"] == torso and r["all"]]
            print(f"stance ({x}, {y}) torso {torso}: {len(good)} into-face poses reachable at every angle "
                  f"{[(r['yaw_off'], r['pitch'], r['r'], round(r['roll_need'], 2)) for r in good][:6]}", flush=True)
        by = {}
        for r in rows:
            by.setdefault((r["yaw_off"], r["pitch"], r["r"], r["stand"]), []).append(r)
        print("per (yaw_off, pitch, r): reachable fraction per angle | stances at all | min roll need")
        for k, v in sorted(by.items()):
            a = np.array([r["ok"] for r in v], dtype=float).mean(axis=0)
            n = [r["roll_need"] for r in v if r["roll_need"] is not None]
            print(f"  {k}: {np.round(a, 2).tolist()} | {len(n)} | {round(min(n), 2) if n else '-'}")
        best = sorted((r for r in rows if r["roll_need"] is not None), key=lambda r: r["roll_need"])[:10]
        print("lowest roll:")
        for r in best:
            print("  ", {k: r[k] for k in ("x", "y", "torso", "yaw_off", "pitch", "r", "stand", "roll_need")})
        Path(f"reach_into_{args.side}.json").write_text(json.dumps(rows))
        env.close()
        return

    if args.side == "inner":
        rads = (fd.CRACK_RAD, 0.9, fd.OPEN_RAD)
        stances = list(itertools.product((0.85, 0.95, 1.05, 1.15), (-1.25, -1.10, -0.95)))
    else:
        rads = (fd.OPEN_RAD, 0.9, 0.5, 0.1)
        stances = list(itertools.product((0.55, 0.70, 0.85, 1.0), (-1.45, -1.30, -1.15)))
    torsos = (0.20, 0.386)
    yaw_offs = (-0.35, 0.0, 0.35, 0.7)
    pitches = (0.0, math.radians(35), math.radians(70), math.radians(90))
    zs = (1.50,)
    alongs = (0.08,)
    offsets = (0.09,)
    closings = ("across", "vertical", "diag")
    rows = []
    for (x, y), torso in itertools.product(stances, torsos):
        for yaw_off, pitch, z, along, closing, off in itertools.product(yaw_offs, pitches, zs, alongs,
                                                                         closings, offsets):
            ok, needs = [], []
            for rad in rads:
                set_door(rad)
                press = leaf.point(rad, along, off if args.side == "inner" else -off)
                yaw = math.atan2(press[1] - y, press[0] - x)
                set_base(x, y, yaw, torso)
                planner.planner.update_from_simulation()
                need = ik_need(pose_for(rad, along, off, z, yaw_off, pitch, closing))
                ok.append(need is not None)
                needs.append(need)
            rows.append(dict(x=x, y=y, torso=torso, yaw_off=yaw_off, pitch_deg=round(math.degrees(pitch)),
                             z=z, along=along, closing=closing, off=off, ok=ok, all=all(ok),
                             roll_need=(max(needs) if all(ok) else None)))
        good = [r for r in rows if r["x"] == x and r["y"] == y and r["torso"] == torso and r["all"]]
        print(f"stance ({x}, {y}) torso {torso}: {len(good)} poses reachable at every angle "
              f"{[(r['yaw_off'], r['pitch_deg'], r['closing'], r['off']) for r in good][:8]}", flush=True)
    by = {}
    for r in rows:
        by.setdefault((r["yaw_off"], r["pitch_deg"], r["closing"], r["off"]), []).append(r["ok"])
    need = {}
    for r in rows:
        if r["roll_need"] is not None:
            need.setdefault((r["yaw_off"], r["pitch_deg"], r["closing"], r["off"]), []).append(r["roll_need"])
    print("per (yaw_off rad, pitch deg, closing, offset): reachable fraction at each angle | "
          "stances reachable at all | median and min roll excursion (rad) over those stances")
    for k, v in sorted(by.items()):
        a = np.array(v, dtype=float).mean(axis=0)
        n = need.get(k, [])
        print(f"  {k}: {np.round(a, 2).tolist()} | {len(n)} | "
              f"{(round(float(np.median(n)), 2), round(float(np.min(n)), 2)) if n else '-'}")
    best = sorted((r for r in rows if r["roll_need"] is not None), key=lambda r: r["roll_need"])[:12]
    print("lowest roll excursion:")
    for r in best:
        print("  ", {k: r[k] for k in ("x", "y", "torso", "yaw_off", "pitch_deg", "closing", "roll_need")})
    Path(f"reach_scan_{args.side}.json").write_text(json.dumps(rows))
    env.close()


if __name__ == "__main__":
    main()
