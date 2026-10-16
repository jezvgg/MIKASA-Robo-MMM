"""Which counter pad positions Depth Recall v1's arm reaches from its ONE dock, by cuRobo's scan.

    MIKASA_CUROBO_PY=<curobo venv python> PYTHONPATH=. python scripts/dr_pad_reach.py OUT.json

The poses are the planner's own (`depth_recall_v1_planner`), built with the env's
`build_grasp_pose` and its scene numbers:

* **restore grasp**: the base on the dock line (row x, ``READY_DOCK_Y``, facing +y), the
  side grasp of a prop standing on the pad, every ``GRASP_LIFTS`` x ``SIDE_GRASP_DEPTHS``
  rung, and its pre-grasp 12 cm back along the approach;
* **extraction place**: the base where the carry-out leaves it (``WORK_DOCK_Y`` minus the
  carry, which depends on the row slot the prop came from; ``CARRY=slot`` is the old rule,
  back until the prop is over the pads' line, ``CARRY=dock`` the planner's
  `carry_out`), the hand over the pad at the place height and at the three cross heights.

A pad offset counts as reachable by the planner's own rule, from every stand (the row's x
jitters by +-``row_jitter_x``) and for a prop from any row slot: some grasp lift whose
grasp and pre-grasp both reach, the place height, and some cross height. Every pose's
own flag is saved, so a stricter rule can be read off the file. Kinematics and self-collision only, hits
re-checked in float64 (`curobo_client.verify`): the scene is the planner's to check, so
this is an upper bound that the planner's own sweep then confirms or refutes.
"""
import json
import os
import sys
import time

import numpy as np

CARRY = os.environ.get("CARRY", "dock")


def main(out):
    import gymnasium as gym

    import mani_skill  # noqa: F401
    import my_scenes  # noqa: F401
    from planners import cabinet_retrieval_planner as retr
    from planners import depth_recall_v1_planner as dr
    from robots.fetch import curobo_client

    env = gym.make("MikasaDepthRecall-v1", num_envs=1, robot_uids="ds_fetch",
                   control_mode="pd_joint_pos", obs_mode="state", sim_backend="cpu")
    env.reset(seed=0)
    task = env.unwrapped
    cfg = task.cfg
    x0 = float(np.asarray(task._row_centre_x).reshape(-1)[0])
    z_top = float(np.asarray(task._counter_top_z).reshape(-1)[0])
    counter, cab = task.counters[0], task.cabinets[0]
    facts = dict(row_centre_x=x0, counter_top_z=z_top,
                 counter_pos=np.asarray(counter.pos).tolist(), counter_size=np.asarray(counter.size).tolist(),
                 cabinet_pos=np.asarray(cab.pos).tolist(), cabinet_size=np.asarray(cab.size).tolist())
    print("scene", json.dumps(facts), flush=True)

    R = task.agent.build_grasp_pose(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                                    np.zeros(3)).to_transformation_matrix()[:3, :3]
    back = np.eye(4)
    back[2, 3] = -0.12          # `side_grasp_pose`: pre = grasp * Pose([0, 0, -0.12])

    def pose(p):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    offsets = np.round(np.arange(-0.80, 0.4001, 0.02), 3)
    ys = (-0.60, -0.56, -0.525, -0.49, -0.46)
    jitter = (-cfg.row_jitter_x, 0.0, cfg.row_jitter_x)
    mid = z_top + cfg.prop_half_h
    shelf_tcp = cfg.shelf_top_z + cfg.prop_half_h + dr.GRASP_LIFTS[0] + dr.UNSHELVE_DZ
    cross_z = tuple(shelf_tcp - (retr.READY_TORSO - t) for t in (retr.READY_TORSO, 0.30, 0.20))

    face_y = float(np.asarray(cab.pos)[1] - np.asarray(cab.size)[1] / 2)

    def carry(prop_y, y):
        """How far the extraction backs the base off with the prop: CARRY=slot (until the
        prop is over the pad's line) or CARRY=dock (`depth_recall_v1_planner.carry_out`)."""
        if CARRY == "slot":
            return prop_y - y
        return dr.carry_out(retr.WORK_DOCK_Y, prop_y - dr.UNSHELVE_BACK, face_y, cfg.prop_half_xy)

    report = dict(facts=facts, carry=CARRY, offsets=offsets.tolist(), ys=list(ys), jitter=list(jitter),
                  lifts=list(dr.GRASP_LIFTS), depths=list(retr.SIDE_GRASP_DEPTHS),
                  row_slots_y=list(cfg.row_slots_y), cross_z=list(cross_z), maps={})
    for coll in (False, True):
        for y in ys:
            t0 = time.perf_counter()
            # restore: [jitter, offset, lift, depth, (grasp, pre)]
            goals = []
            for dx in offsets:
                for lift in dr.GRASP_LIFTS:
                    for depth in retr.SIDE_GRASP_DEPTHS:
                        g = pose([dx, y + depth, mid + lift])
                        goals += [g, g @ back]
            G = np.stack(goals)
            restore = []
            for j in jitter:
                Gw = G.copy()
                Gw[:, 0, 3] += x0 + j
                ok, _ = curobo_client.scan(Gw, [(x0 + j, retr.READY_DOCK_Y, np.pi / 2)], coll=coll)
                restore.append(ok[0].reshape(len(offsets), len(dr.GRASP_LIFTS), len(retr.SIDE_GRASP_DEPTHS), 2))
            # extraction: [row slot, jitter, offset, (place, cross at each torso rung)]
            place = []
            for prop_y in cfg.row_slots_y:
                base_y = retr.WORK_DOCK_Y - carry(prop_y, y)
                G = np.stack([pose([dx, y, z]) for dx in offsets
                              for z in (mid + dr.GRASP_LIFTS[0] + 0.005,) + tuple(cross_z)])
                per = []
                for j in jitter:
                    Gw = G.copy()
                    Gw[:, 0, 3] += x0 + j
                    ok, _ = curobo_client.scan(Gw, [(x0 + j, base_y, np.pi / 2)], coll=coll)
                    per.append(ok[0].reshape(len(offsets), 1 + len(cross_z)))
                place.append(per)
            restore, place = np.asarray(restore), np.asarray(place)
            # The planner's rule: restore needs SOME lift whose first-depth grasp and its
            # pre-grasp both reach; the extraction needs the place height and SOME cross
            # height. Every jitter, and (extraction) every row slot the prop can come from.
            r_ok = (restore[..., 0, :].all(axis=-1)).any(axis=-1).all(axis=0)
            p_ok = (place[..., 0] & place[..., 1:].any(axis=-1)).all(axis=(0, 1))
            key = f"y{y}_{'selfcoll' if coll else 'kinematic'}"
            report["maps"][key] = dict(restore=restore.astype(int).tolist(), place=place.astype(int).tolist(),
                                       restore_ok=r_ok.astype(int).tolist(), place_ok=p_ok.astype(int).tolist(),
                                       wall_s=round(time.perf_counter() - t0, 2))
            both = offsets[r_ok & p_ok]
            print(key, "reachable offsets:", f"{both.min():.2f}..{both.max():.2f}" if len(both) else "none",
                  "| restore", offsets[r_ok].min() if r_ok.any() else None, offsets[r_ok].max() if r_ok.any() else None,
                  "| place", offsets[p_ok].min() if p_ok.any() else None, offsets[p_ok].max() if p_ok.any() else None,
                  flush=True)
    report["stats"] = {k: v for k, v in curobo_client.STATS.items()}
    print("stats", report["stats"], flush=True)
    with open(out, "w") as f:
        json.dump(report, f)
    env.close()


if __name__ == "__main__":
    main(sys.argv[1])
