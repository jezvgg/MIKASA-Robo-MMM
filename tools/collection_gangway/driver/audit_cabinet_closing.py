"""Reconstruct every recorded closing state; retain all robot contact pairs."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import h5py,numpy as np
from utils.collection.profile import make_env
from planners.oracle import oracle_common as common
root=Path(sys.argv[1]);run=json.loads((root/'run.json').read_text());out=Path(__file__).with_name(root.name+'-closing-clearance.json')
report=dict(run=str(root),scope='All source 20 Hz states during panel push through released closure; model contacts, not 100 Hz force measurements',episodes=[])
if out.exists():report=json.loads(out.read_text())
done={x['seed'] for x in report['episodes']}
def sample(g,i):return {k:sample(v,i) if isinstance(v,h5py.Group) else v[i] for k,v in g.items()}
env=make_env(run)
try:
 for seed in run['seeds']:
  if seed in done:continue
  d=root/'oracle'/str(seed);rp=d/'result.json'
  if not rp.exists():continue
  ev=[json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()]
  starts=[e['step'] for e in ev if e['message']=='push the door closed']
  ends=[e['step'] for e in ev if e['message']=='physical compartment closed after release']
  path=d/('trajectory.h5' if (d/'trajectory.h5').exists() else 'failed-trajectory.h5')
  if not path.exists():continue
  env.reset(seed=seed);t=env.unwrapped;p=common.collection_planner_factory(env,False,False).planner
  row=dict(seed=seed,intervals=[],contacts=[],unexpected_contacts=[])
  with h5py.File(path) as f:
   g=f['traj_0'];n=len(g['actions']);states=g['env_states']
   for start in starts:
    end=next((e for e in ends if e>start),n);row['intervals'].append([start,end])
    for i in range(start,end+1):
     t.set_state_dict(sample(states,i));t._elapsed_steps[:]=i;p.update_from_simulation()
     for c in p.planning_world.check_collision():
      pair=[c.link_name1,c.link_name2]
      if not any('ds_fetch' in s for s in pair):continue
      expected=any('ds_fetch' in s and any(k in s for k in ['gripper','wrist']) for s in pair) and any('cab_' in s and 'door' in s for s in pair)
      item=dict(step=i,pair=pair,expected_hand_door=expected)
      row['contacts'].append(item)
      if not expected:row['unexpected_contacts'].append(item)
  report['episodes'].append(row);out.write_text(json.dumps(report,indent=2)+'\n');print(seed,len(row['contacts']),len(row['unexpected_contacts']),flush=True)
finally:env.close()
