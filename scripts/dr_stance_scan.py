"""How robust Depth Recall v1's ONE dock is to base error, and where along the row it should be.

    MIKASA_CUROBO_SOCKET=<server socket> PYTHONPATH=. python scripts/dr_stance_scan.py OUT.json

Item 2 of plan curious-wishing-newell (planners/oracle/stance_scan.py). The planner docks
once at (row x, ``READY_DOCK_Y``, facing +y) and every transfer is rigid relative to the
row's x, so a kinematic stance optimum is one offset ``dx`` from the row, the same in every
episode — scanned once here, not per episode. For each ``dx`` in -0.20..+0.10 every
perturbation of the dock (along and across the heading, and turned) is scanned with the
goals held where the episode puts them, and the dock counts as robust under a
perturbation when EVERY transfer of every row slot and every pad keeps a reachable
alternative, with the torso held at the planner's own rungs (the `arm` variant):

* extraction grasp, row slot i (base driven in to ``WORK_DOCK_Y``): some shelf torso rung
  and grasp lift with the side grasp and its pre-grasp;
* extraction cross over pad j, carried from row slot i (base backed out by `carry_out`):
  the hand at the carry height less some drop at the shelf torso, or at a lower torso rung;
* extraction place on pad j: some counter torso rung;
* restore grasp from pad j (base at the dock): some (torso, lift, depth) with the grasp and
  its pre-grasp;
* restore row place and entry, row slot i (base driven in): some shelf torso rung.

Unlike ``scripts/dr_pad_reach.py`` the perturbation moves the STAND with the goals fixed (a
kinematic scan is blind to moving both), and the crosses are today's hand drops.
Heights are the planner's nominal ones (grasp at the prop's centre plus a lift; places a
lift plus 5 mm); kinematics only, every hit re-checked in float64, the scene is not seen.
"""
import json
import sys
import time

import numpy as np

DX = np.round(np.arange(-0.20, 0.1001, 0.02), 3)
#: (along, lateral, yaw) perturbation grids, the first the plan's +-3 cm / +-3 deg.
PERTURBATIONS = {
    "3cm_3deg": ((-0.03, 0.0, 0.03), (-0.03, 0.0, 0.03), (-0.05, 0.0, 0.05)),
    "5cm_5deg": ((-0.05, 0.0, 0.05), (-0.05, 0.0, 0.05), (-0.087, 0.0, 0.087)),
}


class Bank:
    """Goals grouped by torso height, and the requirements over them."""

    def __init__(self):
        self.goals: dict = {}
        self.need: list = []
        self.names: list = []

    def add(self, torso, T) -> int:
        rows = self.goals.setdefault(torso, [])
        rows.append(np.asarray(T, dtype=np.float64))
        return len(rows) - 1

    def require(self, name, alternatives):
        self.names.append(name)
        self.need.append(alternatives)

    def arrays(self):
        return {t: np.stack(v) for t, v in self.goals.items()}


def main(out):
    import gymnasium as gym

    import mani_skill  # noqa: F401
    import my_scenes  # noqa: F401
    from planners import cabinet_retrieval_planner as retr
    from planners import depth_recall_v1_planner as dr
    from planners.oracle import cabinet_place as sp
    from planners.oracle import stance_scan
    from robots.fetch import curobo_client

    env = gym.make("MikasaDepthRecall-v1", num_envs=1, robot_uids="ds_fetch",
                   control_mode="pd_joint_pos", obs_mode="state", sim_backend="cpu")
    env.reset(seed=1000)
    task = env.unwrapped
    cfg = task.cfg
    row = np.asarray(task._row_points.cpu()).reshape(-1, 3)
    pads = np.asarray(task._slot_points.cpu()).reshape(-1, 3)
    row_x = float(row[0][0])
    z_top = float(np.asarray(task._counter_top_z.cpu()).reshape(-1)[0])
    counter_mid = z_top + cfg.prop_half_h
    shelf_mid = cfg.shelf_top_z + cfg.prop_half_h
    cab = task.cabinets[0]
    face_y = float(np.asarray(cab.pos)[1] - np.asarray(cab.size)[1] / 2)

    R = task.agent.build_grasp_pose(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                                    np.zeros(3)).to_transformation_matrix()[:3, :3]
    back = np.eye(4)
    back[2, 3] = -0.12

    def pose(p, base_y):
        """The world pose, moved into the frame where the base stands on the dock line."""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        T[1, 3] -= base_y - retr.READY_DOCK_Y
        return T

    bank = Bank()
    lift0 = dr.GRASP_LIFTS[0]
    carry_z = shelf_mid + lift0 + dr.UNSHELVE_DZ
    for i, (rx, ry, _) in enumerate(row):
        alts = []
        for t in dr.SHELF_TORSO_RUNGS:
            for lift in dr.GRASP_LIFTS:
                g = pose([rx, ry + retr.SIDE_GRASP_DEPTHS[0], shelf_mid + lift], retr.WORK_DOCK_Y)
                alts.append((t, [bank.add(t, g), bank.add(t, g @ back)]))
        bank.require(f"extraction grasp row{i}", alts)
        alts = []
        for t in dr.SHELF_TORSO_RUNGS:
            place = pose([rx, ry, shelf_mid + lift0 + 0.005], retr.WORK_DOCK_Y)
            entry = pose([rx, ry, shelf_mid + lift0 + 0.005 + sp.ENTRY_RISE], retr.WORK_DOCK_Y)
            alts.append((t, [bank.add(t, place), bank.add(t, entry)]))
        bank.require(f"restore entry row{i}", alts)
        base_y = retr.WORK_DOCK_Y - dr.carry_out(retr.WORK_DOCK_Y, ry - dr.UNSHELVE_BACK, face_y,
                                                 cfg.prop_half_xy)
        for j, (px, py, _) in enumerate(pads):
            alts = [(dr.SHELF_TORSO_RUNGS[0], [bank.add(dr.SHELF_TORSO_RUNGS[0],
                                                        pose([px, py, carry_z - drop], base_y))])
                    for drop in (0.0, 0.10, 0.20)]
            alts += [(t, [bank.add(t, pose([px, py, carry_z], base_y))]) for t in (0.30, 0.20)]
            bank.require(f"extraction cross row{i} pad{j}", alts)
            alts = [(t, [bank.add(t, pose([px, py, counter_mid + lift0 + 0.005], base_y))])
                    for t in dr.COUNTER_TORSO_RUNGS]
            bank.require(f"extraction place row{i} pad{j}", alts)
    for j, (px, py, _) in enumerate(pads):
        alts = []
        for t in dr.COUNTER_GRASP_TORSO_RUNGS:
            for lift in dr.GRASP_LIFTS:
                for depth in retr.SIDE_GRASP_DEPTHS:
                    g = pose([px, py + depth, counter_mid + lift], retr.READY_DOCK_Y)
                    alts.append((t, [bank.add(t, g), bank.add(t, g @ back)]))
        bank.require(f"restore grasp pad{j}", alts)
    goals = bank.arrays()
    print("scene", dict(row_x=round(row_x, 3), pads=np.round(pads, 3).tolist(), row=np.round(row, 3).tolist(),
                        counter_top=round(z_top, 3), torsos={str(k): len(v) for k, v in goals.items()},
                        requirements=len(bank.need)), flush=True)

    report = dict(dx=DX.tolist(), requirements=bank.names, results={})
    for label, (along, lateral, yaw) in PERTURBATIONS.items():
        t0 = time.perf_counter()
        rows = []
        for dx in DX:
            stand = (row_x + float(dx), retr.READY_DOCK_Y, np.pi / 2)
            score, cov = stance_scan.robust_coverage(goals, [stand], bank.need, along=along,
                                                     lateral=lateral, yaw=yaw)
            # Which requirement fails, per perturbation: rerun the coverage per requirement.
            perturbed = stance_scan.jitter(stand, along, lateral, yaw)
            masks = {t: curobo_client.scan(G, perturbed, torso=t)[0] for t, G in goals.items()}
            weak = {bank.names[k]: round(float(stance_scan.coverage(masks, [req]).mean()), 3)
                    for k, req in enumerate(bank.need)}
            weak = {k: v for k, v in weak.items() if v < 1.0}
            rows.append(dict(dx=float(dx), robust=round(float(score[0]), 3), weak=weak))
            print(label, f"dx={dx:+.2f} robust={score[0]:.3f}", "weak:", weak, flush=True)
        report["results"][label] = dict(rows=rows, wall_s=round(time.perf_counter() - t0, 1))
    report["stats"] = dict(curobo_client.STATS)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    env.close()


if __name__ == "__main__":
    main(sys.argv[1])
