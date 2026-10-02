"""Bounded, scene-checked IK endpoints; no seed-dependent solution lookup."""
import time
import mplib
import numpy as np
import sapien
from transforms3d.quaternions import mat2quat
from robots.fetch.utils import unwrap_toward
from planners.oracle.wrist_pour import rotation
from planners.oracle import oracle_common as common

TEMPLATES = ((.5, 2.1, 1.8, -2.3, .75, .1),
             (-.6, .9, 1.8, -3.0, 1.05, 2.4))
NAMES = ('shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint',
         'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint')
GEOMETRIES = ((0.,0.,.20,1),(30.,0.,.20,0),(30.,0.,.10,0),
              (30.,.01,None,0),(0.,0.,.20,0),(0.,0.,None,0),
              (-30.,0.,.20,0),(0.,0.,None,1))
DIVERSE_GEOMETRIES = ((0.,0.,.20,1),(30.,0.,.10,0),
                      (0.,0.,None,1),(0.,0.,.20,0))
# Saved-state full-continuation probes favoured .10/.15 over .25/.30 m.
# This remains four scene-specific IK requests, not a seed-to-solution table.
LOW_GEOMETRIES = ((0.,0.,.20,1),(0.,0.,.10,1),
                  (0.,0.,None,1),(0.,0.,.15,1))
MIXED_GEOMETRIES = ((0.,0.,.20,1),(0.,0.,.15,1),
                    (30.,0.,.10,0),(0.,0.,None,1))
# Four medoids from successful archived grasp configurations, in NAMES order.
# They initialize IK only; every endpoint and continuation is checked anew.
ARCHIVE_WARM_STARTS = (
    (-.662554536,.825080844,1.834440277,-3.016126218,1.061119352,2.454279152),
    (-.521575145,1.079342946,1.905642317,-3.020458467,1.158130825,2.210852100),
    (-.207222610,1.399782844,1.956271905,-3.088443710,1.253635717,1.815211959),
    (-.394138262,1.227037228,1.921497867,-3.053495105,1.177937660,2.045369398))


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def unrolled(template):
    """The same hand pose with the forearm rolled by pi and the wrist flexed the other way.

    For forearm_roll / wrist_flex / wrist_roll, (r1, f, r2) and (r1+pi, -f, r2+pi) put the
    hand in the identical pose; the shoulder and elbow are unchanged.
    """
    sl, ua, el, fa, wf, wr = template
    return (sl, ua, el, _wrap(fa + np.pi), -wf, _wrap(wr + np.pi))


def joint_margin(planner, task, q):
    """Nearest physical or episode roll limit for the arm, in radians."""
    names = task.agent.controller.controllers['arm'].config.joint_names
    ix = [int(task.agent.robot.active_joints_map[n].active_index[0]) for n in names]
    limits = task.agent.robot.get_qlimits()[0].cpu().numpy().copy()
    for j, lo, hi in zip(planner._roll_indices, planner._roll_low, planner._roll_high):
        limits[j] = [max(-np.pi+.05, hi-np.pi+.05),
                     min(np.pi-.05, lo+np.pi-.05)]
    q = np.asarray(q)
    return float(np.minimum(q[ix]-limits[ix,0], limits[ix,1]-q[ix]).min())


def distinct_goals(goals, indices, threshold=.12):
    kept = []
    for q in goals:
        if all(np.linalg.norm(q[indices]-g[indices]) >= threshold for g in kept):
            kept.append(q)
    return kept


def journal(planner, record, stage, reason=None, **details):
    budget=getattr(planner,'search_budget',None)
    row = dict(record, candidate_stage=stage, reason=reason,
               planning_seconds=budget.used if budget is not None else None, **details)
    planner._candidate_journal.append(row)
    common.say(planner.env, 'season_dish_planner', 'grasp candidate', **row)


def failure_kind(statuses, collision=None):
    text = ' '.join(map(str, statuses)).lower()
    if collision or 'collision' in text:
        return 'collision'
    if 'limit' in text or 'bounded' in text:
        return 'joint_limit'
    if 'jump' in text or 'discontinu' in text:
        return 'configuration_jump'
    if 'grasp' in text or 'hold' in text:
        return 'lost_grasp'
    return 'unreachable'


def candidates(planner, task, grasp, reach, reasons, *, n_init=32):
    p = planner.planner
    current = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
    folded = p.fold_qpos(current)
    indices = [int(task.agent.robot.active_joints_map[n].active_index[0]) for n in NAMES]
    parameters = getattr(task, 'motion_parameters', {})
    policy = parameters.get('grasp_selection', 'legacy')
    if policy not in ('legacy', 'margin', 'diverse'):
        raise ValueError(f'Unknown grasp selection: {policy}')
    specifications = DIVERSE_GEOMETRIES if policy == 'diverse' else GEOMETRIES
    if parameters.get('grasp_heights') in ('low','mixed'):
        if policy != 'diverse':
            raise ValueError('Low-height comparison requires the four-query policy')
        specifications=LOW_GEOMETRIES if parameters['grasp_heights']=='low' else MIXED_GEOMETRIES
    family = parameters.get('grasp_family', 'rolled')
    if family not in ('rolled', 'unrolled'):
        raise ValueError(f'Unknown grasp family: {family}')
    wrist = int(task.agent.robot.active_joints_map['wrist_flex_joint'].active_index[0])
    warm_starts=bool(parameters.get('grasp_archive_warm_starts',False))
    warm_queries=parameters.get('grasp_archive_queries',
        {str(i):i%4 for i in range(len(specifications))}) if warm_starts else {}
    result = []
    planner._candidate_journal = []
    for request, (yaw, dz, torso, template) in enumerate(specifications):
        if 'diagnostic_grasp_torso_m' in parameters:
            torso = parameters['diagnostic_grasp_torso_m']
        start = time.perf_counter()
        rot = rotation([0,0,1], np.deg2rad(yaw))
        orientation = mat2quat(rot @ grasp.to_transformation_matrix()[:3,:3])
        g = sapien.Pose(grasp.p+[0,0,dz], orientation)
        reach_pose = sapien.Pose(g.p+rot@(reach.p-grasp.p), orientation)
        initial = folded.copy()
        archive_index=warm_queries.get(str(request))
        seed_template = ARCHIVE_WARM_STARTS[archive_index] if archive_index is not None else TEMPLATES[template]
        initial[indices] = unrolled(seed_template) if family == 'unrolled' else seed_template
        if torso is not None:
            initial[3] = torso
        status, found = p.IK(p._transform_goal_to_wrt_base(mplib.Pose(reach_pose.p,reach_pose.q)),
                             initial, [True]*3+[torso is not None]+[False]*11,
                             n_init_qpos=min(32,n_init))
        reasons['ik '+str((yaw,dz,torso,template))+' '+status] += 1
        record = dict(request=request, yaw_deg=yaw, lift_offset_m=dz,
                      requested_torso_m=torso, template=template, policy=policy,
                      template_source='archive_medoid' if archive_index is not None else 'original',
                      archive_template=archive_index,
                      ik_seconds=time.perf_counter()-start)
        if status != 'Success':
            journal(planner,record,'ik',failure_kind([status]),status=status)
            continue
        goals = []
        for raw in np.atleast_2d(found):
            wrapped = unwrap_toward(raw,initial,p.joint_limits)
            # Unwrapping toward an artificial IK seed can cross the episode's
            # roll window even though mplib's original representative is valid.
            q = wrapped if p.accepts(wrapped) else np.asarray(raw).copy()
            if family == 'unrolled' and q[wrist] > -.05:
                continue
            if p.accepts(q):
                goals.append(q)
        if not goals:
            journal(planner,record,'ik','joint_limit_after_unwrap')
            continue
        goals.sort(key=lambda q: float(np.linalg.norm(q[3:13]-initial[3:13])))
        ranks = {id(q): i for i,q in enumerate(goals)}
        if policy != 'legacy':
            # Limit clearance is primary; distance breaks ties without changing
            # the selected elbow/wrist branch. Continuations are checked below.
            goals.sort(key=lambda q: (-round(joint_margin(planner,task,q),2),
                                     float(np.linalg.norm(q[3:13]-initial[3:13]))))
        goals = distinct_goals(goals, indices)
        for q in goals[:2 if policy == 'diverse' else 1]:
            for idx in p._root_cols():
                q[idx] = current[idx]
            detail = dict(record, ik_rank_by_template=ranks[id(q)],
                          torso_m=float(q[3]), joint_margin_rad=joint_margin(planner,task,q),
                          qpos=q.tolist(), candidate=len(result))
            if any(np.linalg.norm(q[3:13]-old[0][3:13]) < .12 for old in result):
                journal(planner,detail,'deduplicate','near_identical_endpoint')
                continue
            journal(planner,detail,'ik_selected')
            result.append((q,g,reach_pose,detail))
    assert len(result) <= 8
    reasons['bounded candidates'] = len(result)
    return result
