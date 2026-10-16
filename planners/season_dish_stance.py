"""The SeasonDish station stance by a batched reach scan (``MIKASA_STANCE_SCAN=1``).

The planner parks at the station dock backed off ``station_backoff_m`` and grasps from
wherever the drive stops: up to 0.10 m short along the drive (`navigation_arrival_tolerance`)
and up to 0.25 rad off in heading. This chooses the stance instead
(planners/oracle/stance_scan.py): every candidate is scanned under that arrival error
for "some grasp request has its contact, standoff, via-above point and lift reachable",
ranked by robust coverage and by distance from the shipped stance, and the best few are
checked against the scene the way item 1 asks the arm — an IK chain from the contact back
to the standoff (planners/oracle/ik_chain.py), with the base where that stance puts it.
The first that passes is driven to; with none, the shipped stance stands.

Only the drive target changes. Everything after it (the grasp geometry from the actual
base, the candidates, the chain or the screw) is computed from where the base actually is.
"""
from __future__ import annotations

import math
import time

import numpy as np
import sapien
from transforms3d.quaternions import mat2quat

from planners import season_dish_candidates as cand
from planners.oracle import ik_chain, stance_scan
from planners.oracle.wrist_pour import rotation

#: Candidate stances around the noised station dock: slide along the counter, back-off
#: from it (the profile's allowed set), heading offset.
SLIDE_M = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
BACKOFF_M = (0.0, 0.05, 0.10, 0.15)
YAW_RAD = (0.0, -0.15, 0.15)
#: The arrival error every stance must survive: the drive stops up to 0.10 m short, a
#: little to either side, and the final turn leaves up to 0.25 rad (0.1 covers most).
ALONG_ERR = (0.0, -0.05, -0.10)
LATERAL_ERR = (-0.03, 0.0, 0.03)
YAW_ERR = (-0.1, 0.0, 0.1)
#: The lift goal: straight up from the contact (grasp_chain lifts 0.15 m or to clear the
#: taller condiment, whichever is higher).
LIFT_M = 0.15


def specifications(task):
    """The grasp requests `season_dish_candidates.candidates` will make, for this profile."""
    parameters = getattr(task, "motion_parameters", {})
    specs = cand.DIVERSE_GEOMETRIES if parameters.get("grasp_selection", "legacy") == "diverse" else cand.GEOMETRIES
    if parameters.get("grasp_heights") in ("low", "mixed"):
        specs = cand.LOW_GEOMETRIES if parameters["grasp_heights"] == "low" else cand.MIXED_GEOMETRIES
    return specs


def request_poses(grasp: sapien.Pose, reach: sapien.Pose, spec):
    """(contact, standoff) of one request, exactly as `candidates` builds them."""
    yaw, dz = spec[0], spec[1]
    rot = rotation([0, 0, 1], np.deg2rad(yaw))
    orientation = mat2quat(rot @ grasp.to_transformation_matrix()[:3, :3])
    g = sapien.Pose(grasp.p + [0, 0, dz], orientation)
    return g, sapien.Pose(g.p + rot @ (reach.p - grasp.p), orientation)


def grasp_from(task, obb, mesh, stand, grasp_info, below_top: float):
    """The grasp and reach poses the planner would build with the base at `stand`.

    The approach is the horizontal stand->object ray (K59), so the reach is on the stand's
    side and `oracle_common.grasp_geometry`'s far-side flip never applies.
    """
    ee = np.asarray(obb.center_mass, dtype=np.float64) - np.array([stand[0], stand[1], 0.0])
    ee[2] = 0.0
    ee /= np.linalg.norm(ee)
    closing = np.cross(ee, [0.0, 0.0, 1.0])
    closing /= np.linalg.norm(closing)
    info = grasp_info(obb, ee, closing)
    grasp = task.agent.build_grasp_pose(info["approaching"], info["closing"], info["center"])
    reach = grasp * sapien.Pose([0, 0, -0.1])
    z = max(float(obb.center_mass[2]), float(mesh.bounds[1][2]) - below_top)
    up = np.array([0.0, 0.0, z - float(grasp.p[2])])
    return (sapien.Pose(p=np.asarray(grasp.p) + up, q=grasp.q),
            sapien.Pose(p=np.asarray(reach.p) + up, q=reach.q))


def stance_goals(task, specs, grasp, reach, above_m: float):
    """``({torso: (N, 4, 4)}, need)``: some request with all of its poses reachable."""
    goals: dict = {}
    alternatives = []
    for spec in specs:
        torso = spec[2]
        g, r = request_poses(grasp, reach, spec)
        poses = [g, r, sapien.Pose(np.asarray(g.p) + [0, 0, LIFT_M], g.q)]
        if above_m > 0:
            poses.append(sapien.Pose(np.asarray(r.p) + [0, 0, above_m], r.q))
        rows = goals.setdefault(torso, [])
        alternatives.append((torso, list(range(len(rows), len(rows) + len(poses)))))
        rows.extend(p.to_transformation_matrix() for p in poses)
    return {t: np.stack(v) for t, v in goals.items()}, [alternatives]


def scene_check(planner, task, stand, specs, grasp, reach) -> dict:
    """Item 1's chain at a stance the base is not at: some request's contact solved with the
    base there (mplib, scene and all) and walked back to its standoff on one branch."""
    p = planner.planner
    current = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
    what_if = current.copy()
    what_if[:3] = stance_scan.root_joints_for(task, *stand)
    folded = p.fold_qpos(what_if)
    names = [int(task.agent.robot.active_joints_map[n].active_index[0]) for n in cand.NAMES]
    wrist = int(task.agent.robot.active_joints_map["wrist_flex_joint"].active_index[0])
    family = getattr(task, "motion_parameters", {}).get("grasp_family", "rolled")
    arm = ik_chain.arm_columns(task)

    def accept(q):
        return p.accepts(q) and not (family == "unrolled" and q[wrist] > -.05)

    for request, spec in enumerate(specs):
        yaw, dz, torso, template = spec
        g, r = request_poses(grasp, reach, spec)
        initial = folded.copy()
        seed = cand.TEMPLATES[template]
        initial[names] = cand.unrolled(seed) if family == "unrolled" else seed
        if torso is not None:
            initial[3] = torso
        mask = [True] * 3 + [torso is not None] + [False] * 11
        report = ik_chain.ChainReport()
        chains = ik_chain.chain_back(ik_chain.mplib_solver(planner, mask, accept=accept), g, r, initial,
                                     cols=arm, n_hard=16, keep=2, report=report)
        if chains:
            return dict(ok=True, request=request, chain=report.as_dict())
    return dict(ok=False)


def choose_station_stance(env, planner, task, target, dock_xyz, face, grasp_info, *, say,
                          backoff: float, below_top: float):
    """``(dock_xyz, face)`` to drive to, or None to keep the shipped stance.

    `dock_xyz` is the noised station dock before the back-off, `face` its facing.
    """
    stance_scan.require_server()
    t0 = time.perf_counter()
    mesh = target.get_first_collision_mesh(to_world_frame=True)
    obb = mesh.bounding_box_oriented
    specs = specifications(task)
    above_m = float(getattr(task, "motion_parameters", {}).get("approach_via_above_m", 0.0))
    yaw0 = math.atan2(float(face[1]), float(face[0]))
    side = np.array([-face[1], face[0]], dtype=np.float64)
    stands, costs, cases = [], [], []
    for slide in SLIDE_M:
        for back in BACKOFF_M:
            for dyaw in YAW_RAD:
                xy = np.asarray(dock_xyz[:2], dtype=np.float64) - back * np.asarray(face[:2]) + slide * side
                stands.append((float(xy[0]), float(xy[1]), yaw0 + dyaw))
                costs.append(abs(slide) + abs(back - backoff) + 0.5 * abs(dyaw))
                cases.append((slide, back, dyaw))
    scores, geometry = [], []
    for stand in stands:
        grasp, reach = grasp_from(task, obb, mesh, stand, grasp_info, below_top)
        goals, need = stance_goals(task, specs, grasp, reach, above_m)
        score, _ = stance_scan.robust_coverage(goals, [stand], need, along=ALONG_ERR,
                                               lateral=LATERAL_ERR, yaw=YAW_ERR)
        scores.append(float(score[0]))
        geometry.append((grasp, reach))
    t_scan = time.perf_counter() - t0
    order = stance_scan.rank(scores, costs)
    shipped = int(np.argmin(costs))
    verdicts = {}

    def verify(i):
        verdicts[i] = scene_check(planner, task, stands[i], specs, *geometry[i])
        return verdicts[i]["ok"]

    chosen, tried = stance_scan.choose(order, scores, verify)
    planner.planner.update_from_simulation()  # the what-if IK left the model at a stance
    say(env, "stance scan", seconds=round(time.perf_counter() - t0, 2), scan_seconds=round(t_scan, 2),
        stances=len(stands), shipped_robust=round(scores[shipped], 3),
        top=[dict(slide=cases[i][0], backoff=cases[i][1], dyaw=cases[i][2], robust=round(scores[i], 3))
             for i in order[:3]],
        tried=[dict(slide=cases[i][0], backoff=cases[i][1], dyaw=cases[i][2], ok=v,
                    **{k: x for k, x in verdicts[i].items() if k != "ok"}) for i, v in tried],
        chosen=None if chosen is None else dict(slide=cases[chosen][0], backoff=cases[chosen][1],
                                                dyaw=cases[chosen][2], robust=round(scores[chosen], 3)))
    if chosen is None:
        return None
    x, y, yaw = stands[chosen]
    return np.array([x, y, 0.0]), np.array([math.cos(yaw), math.sin(yaw), 0.0])
