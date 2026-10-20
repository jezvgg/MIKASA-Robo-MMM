"""Read-only model collision audit of every recorded control state, not replay."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import gymnasium as gym,h5py,numpy as np
import my_scenes
from planners.oracle import oracle_common as common
root=Path(sys.argv[1]);a=Path(__file__).parent;profile=json.loads((root/'run.json').read_text())['signature']['profile']
env=gym.make(profile['env_id'],obs_mode='state',**profile['env_kwargs']);report={'run':str(root),'frames_hz':20,'scope':'Robot vs upper cabinet geometry including handles, every recorded source state','episodes':[]}
def sample(g,i):return {k:sample(v,i) if isinstance(v,h5py.Group) else v[i] for k,v in g.items()}
try:
 for result_path in sorted((root/'oracle').glob('*/result.json')):
  seed=int(result_path.parent.name);source=next(result_path.parent.glob('*trajectory.h5'),None)
  if source is None:continue
  try:
   with h5py.File(source) as probe:assert 'traj_0/env_states' in probe and 'traj_0/actions' in probe
  except (OSError,KeyError,AssertionError) as error:
   report['episodes'].append(dict(seed=seed,frames=0,upper_collision_frames=0,hits=[],diagnostic_unavailable=type(error).__name__+': '+str(error)));continue
  env.reset(seed=seed);t=env.unwrapped;planner=common.collection_planner_factory(env,False,False);p=planner.planner;hits=[]
  with h5py.File(source) as h:
   states=h['traj_0/env_states'];count=len(h['traj_0/actions'])+1
   for i in range(count):
    t.set_state_dict(sample(states,i));t._elapsed_steps[:]=i;p.update_from_simulation()
    pairs=[]
    for c in p.planning_world.check_collision():
     n1,n2=c.link_name1,c.link_name2
     if ('ds_fetch' in n1 and any(x in n2 for x in ['cab_','microwave'])) or ('ds_fetch' in n2 and any(x in n1 for x in ['cab_','microwave'])):pairs.append([n1,n2])
    if pairs:hits.append(dict(step=i,pairs=pairs))
  row=dict(seed=seed,frames=count,upper_collision_frames=len(hits),hits=hits);report['episodes'].append(row);print('RESULT',seed,count,len(hits),flush=True);(a/(root.name+'-upper-clearance.json')).write_text(json.dumps(report,indent=2)+'\n')
finally:env.close()
