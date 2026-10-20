import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import gymnasium as gym,h5py,numpy as np
import my_scenes
from planners.oracle import oracle_common as common
root=Path(sys.argv[1]);A=Path(__file__).parent;run=json.loads((root/'run.json').read_text());p=run['signature']['profile'];env=gym.make(p['env_id'],obs_mode='state',**p['env_kwargs']);rows=[]
def sample(g,i):return {k:sample(v,i) if isinstance(v,h5py.Group) else v[i] for k,v in g.items()}
try:
 for seed in run['seeds']:
  d=root/'oracle'/str(seed);rp=d/'result.json'
  if not rp.exists():continue
  ev=[json.loads(x) for x in (d/'events.jsonl').read_text().splitlines()];ev=[e for e in ev if e['event']=='phase'];intervals=[]
  rules={'duck for the drive: joint line':['drive home'],'fold to the drive posture: joint line':['in the drive posture'], 'tuck after finding the can: joint line':['nudge the cube'],'drive home':['parked at home'],'drive to the handle dock':['grasp the handle'],'drive to the touch dock':['ready posture for the touch: joint line']}
  if 'SameDrawer' in run['scene']:
   for event in ev:
    if event.get('waypoint')=='return via open aisle':event['message']='return route start'
   rules={'fold to canonical rest: joint line':['travel to apple station'], 'fold to canonical rest: curved joint path':['return route start'], 'travel to apple station':['align with apple'], 'return route start':['smooth return arrived']}
  for k,e in enumerate(ev):
   if e['message'] in rules:
    end=next((x['step'] for x in ev[k+1:] if x['message'] in rules[e['message']]),None)
    if end is not None:intervals.append((e['step'],end,e['message']))
  env.reset(seed=seed);t=env.unwrapped;planner=common.collection_planner_factory(env,False,False);world=planner.planner;hits=[];frames=0;endposes=[]
  with h5py.File(next(d.glob('*trajectory.h5'))) as f:
   states=f['traj_0/env_states']
   for start,end,label in intervals:
    for i in range(start,end+1):
     t.set_state_dict(sample(states,i));t._elapsed_steps[:]=i;world.update_from_simulation();frames+=1
     pairs=[(c.link_name1,c.link_name2) for c in world.planning_world.check_collision() if 'ds_fetch' in c.link_name1 or 'ds_fetch' in c.link_name2]
     if pairs:hits.append(dict(step=i,stage=label,pairs=pairs))
    if 'joint line' in label or 'curved joint path' in label:
     q=t.agent.robot.get_qpos()[0].cpu().numpy();names=['shoulder_pan_joint','shoulder_lift_joint','upperarm_roll_joint','elbow_flex_joint','forearm_roll_joint','wrist_flex_joint','wrist_roll_joint'];idx=[int(t.agent.robot.active_joints_map[n].active_index[0]) for n in names];ref=np.array([-.370,-1.032,.695,.955,-.1,2.077,0]);endposes.append(dict(step=end,stage=label,max_arm_error_rad=float(max(abs(q[idx]-ref))),wrist_flex_rad=float(q[idx[-2]]),torso_m=float(q[3])))
  rows.append(dict(seed=seed,frames=frames,intervals=intervals,collision_frames=len(hits),hits=hits,rest_endpoints=endposes));(A/(root.name+'-transport-clearance.json')).write_text(json.dumps(dict(scope='All robot collision pairs at every 20Hz recorded state during complete rest transitions and free transit, excluding deliberate door contact phases.',episodes=rows),indent=2)+'\n');print('RESULT',seed,frames,len(hits),flush=True)
finally:env.close()
