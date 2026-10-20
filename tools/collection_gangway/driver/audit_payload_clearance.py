"""Read-only reconstruction of source transfer states; no new oracle episodes."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import gymnasium as gym,h5py,numpy as np
import my_scenes
from planners.oracle import oracle_common as common
root=Path(sys.argv[1]);A=Path(__file__).parent;run=json.loads((root/'run.json').read_text());p=run['signature']['profile'];out=A/(root.name+'-payload-clearance.json');report=json.loads(out.read_text()) if out.exists() else dict(scope='Every recorded20Hz source state from confirmed grasp through final pour. Initial payload/support contact is separately recorded until separation. Robot and held payload vs scene fixtures; only gripper/wrist payload contact allowed. This is not a claim about all100Hz internal physics substeps.',run=str(root),episodes=[]);done={x['seed'] for x in report['episodes']};env=gym.make(p['env_id'],obs_mode='state',**p['env_kwargs'])
def sample(g,i):return {k:sample(v,i) if isinstance(v,h5py.Group) else v[i] for k,v in g.items()}
try:
 for seed in run['seeds']:
  d=root/'oracle'/str(seed);rp=d/'result.json'
  if seed in done or not rp.exists() or not json.loads(rp.read_text()).get('success'):continue
  ev=[json.loads(x) for x in (d/'events.jsonl').read_text().splitlines()];start=next(x['step'] for x in ev if x['message']=='grasped');env.reset(seed=seed);t=env.unwrapped;target=t.shaker if bool(t.target_is_shaker.item()) else t.condiment_bottle;target_name='shaker' if target is t.shaker else 'condiment_bottle';planner=common.collection_planner_factory(env,False,False);world=planner.planner;hits=[];support_contacts=[];support_separated=False
  with h5py.File(d/'trajectory.h5') as f:
   g=f['traj_0'];states=g['env_states'];end=len(g['actions'])
   for i in range(start,end+1):
    t.set_state_dict(sample(states,i));t._elapsed_steps[:]=i;world.update_from_simulation();pairs=[]
    support_now=[]
    for c in world.planning_world.check_collision():
     pair=[c.link_name1,c.link_name2]
     if not any('ds_fetch' in n or target_name in n for n in pair):continue
     if any(target_name in n for n in pair) and any('ds_fetch' in n and ('gripper' in n or 'wrist' in n) for n in pair):continue
     if any(target_name in n for n in pair) and any('counter_main' in n for n in pair) and not support_separated:
      support_now.append(pair);continue
     pairs.append(pair)
    if support_now:support_contacts.append(dict(step=i,pairs=support_now))
    else:support_separated=True
    if pairs:hits.append(dict(step=i,pairs=pairs))
  row=dict(seed=seed,start=start,end=end,frames=end-start+1,collision_frames=len(hits),hits=hits,initial_support_contacts=support_contacts);report['episodes'].append(row);out.write_text(json.dumps(report,indent=2)+'\n');print('RESULT',seed,row['frames'],len(hits),flush=True)
finally:env.close()
