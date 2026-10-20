"""Phase-aligned RGB/segmentation and grasp camera geometry, never human E5."""
import bisect,json,sys,os
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path.cwd()))
import gymnasium as gym,h5py,numpy as np
from PIL import Image,ImageDraw
import my_scenes
root=Path(sys.argv[1]);profile=json.loads((root/'run.json').read_text())['signature']['profile'];kind=profile['env_id'];out=Path(__file__).parent/(root.name+'-views'+os.environ.get('RESTORATION_VIEW_SHARD',''));out.mkdir(exist_ok=True)
report={'run':str(root),'scope':'Every 10 Hz physical state; current phase targets and real grasp camera geometry. Not human E5.','episodes':[],'grasps':[]}
if (out/'result.json').exists():report=json.loads((out/'result.json').read_text())
done={e['seed'] for e in report['episodes']};seeds=[int(x) for x in sys.argv[2:]] or sorted(int(p.parent.name) for p in (root/'validated').glob('*/result.json') if json.loads(p.read_text()).get('success'))
def cache_group(g):return {k:cache_group(v) if isinstance(v,h5py.Group) else v[:] for k,v in g.items()}
def sample(g,i):return {k:sample(v,i) if isinstance(v,dict) else v[i].copy() for k,v in g.items()}
def ids(obj):return [int(link.per_scene_id.item()) for link in (obj.get_links() if hasattr(obj,'get_links') else [obj])]
env=gym.make(kind,obs_mode='rgb+segmentation',**profile['env_kwargs'])
try:
 for seed in seeds:
  if seed in done:continue
  env.reset(seed=seed);t=env.unwrapped;events=[json.loads(x) for x in (root/'oracle'/str(seed)/'events.jsonl').read_text().splitlines()]
  stamp=lambda name:next(e['step'] for e in events if e['message']==name)
  with h5py.File(root/'validated'/str(seed)/'trajectory.h5') as f:
   g=f['traj_0'];a=g['actions'][:];q=g['qpos'][:];state_cache=cache_group(g['env_states']);closes=np.flatnonzero(np.diff(a[:,7])<0)+1;trans=[];grasp_indices={}
   def grasp_window(label,end,obj):
    previous=closes[closes<end]
    if not len(previous):return
    close=int(previous[-1])
    for i in sorted(set([*range(max(0,close-10),close+1,2),end])):grasp_indices[i]=(label,close,ids(obj))
   if 'Season' in kind:
    obj=t.shaker if bool(t.target_is_shaker.item()) else t.condiment_bottle
    picture=t.fridge_pictures.actors[int(not bool(t.target_is_shaker.item()))]
    trans=[(0,'cue',ids(picture)),(100,'travel_to_condiment',ids(obj)),(stamp('grasp the target'),'grasp',ids(obj)),(stamp('grasped'),'lift_and_fold',ids(obj)),(stamp('bowl route selected'),'travel_to_bowl',ids(t.bowl)),(stamp('hover over bowl'),'prepare_and_pour',ids(t.bowl))]
    grasp_window('condiment',stamp('grasped'),obj)
   elif 'CabinetSearch' in kind:
    from planners import cabinet_search_planner as search,cabinet_retrieval_planner as retrieval
    trans=[(0,'home_start',ids(t.home_marker))];rounds=[e for e in events if e['message']=='round']
    for k,rnd in enumerate(rounds):
     begin=rnd['step'];end=rounds[k+1]['step'] if k+1<len(rounds) else len(a)+1
     door=search.door_spec_for(t.cfg.compartments[rnd['pick']],closed_rad=t.cfg.theta_closed);art=t.scene.articulations[retrieval._art_key(t,door)];link=next(j for j in art.get_active_joints() if j.name==door.hinge).child_link
     trans.append((begin,f'door_{k}_{rnd["pick"]}',ids(link)))
     for event in events:
      if begin<=event['step']<end and event['message']=='pull hinge arc':grasp_window(f'door_{k}',event['step'],link)
     found=[e for e in events if begin<=e['step']<end and e['message']=='FOUND, not yet touched']
     trans.extend((e['step'],'can',ids(t.cube)) for e in found)
     trans.extend((e['step'],f'home_after_{k}',ids(t.home_marker)) for e in events if begin<=e['step']<end and e['message']=='drive home')
   else:
    drawer=t._drawer_arts[int(t.target_drawer.item())];folds=[e['step'] for e in events if e['message']=='fold to canonical rest: joint line' or e['message']=='fold to canonical rest: curved joint path']
    lift=stamp('lift apple');plate=stamp('step toward plate');return_step=next(e['step'] for e in events if e.get('message')=='waypoint noise' and e.get('waypoint')=='return via open aisle');attached=stamp('object attached to gripper_link in the planning world');grip=stamp('handle grip')
    column=sum((ids(x) for x in t._drawer_arts),[])
    trans=[(0,'initial_drawer',column),(stamp('travel to apple station'),'apple',ids(t.apple)),(plate,'plate',ids(t.plate)),(return_step,'drawer_column',column)]
    grasp_window('apple',attached,t.apple);grasp_window('drawer',grip,drawer)
   trans.sort(key=lambda e:e[0]);stamps=[x[0] for x in trans];rows=[];byphase=defaultdict(list)
   for i in sorted(set(range(0,len(a),2))|set(grasp_indices)):
    t.set_state_dict(sample(state_cache,i));t._elapsed_steps[:]=i;obs=t.get_obs();_,phase,target_ids=trans[bisect.bisect_right(stamps,i)-1]
    if i%2==0 and i<len(a):
     pixels={n:int(np.count_nonzero(np.isin(s['segmentation'][0].cpu().numpy(),target_ids))) for n,s in obs['sensor_data'].items()};row=dict(step=i,phase=phase,pixels=pixels,visible=max(pixels.values())>0);rows.append(row);byphase[phase].append(row)
    if i in grasp_indices:
     label,close,object_ids=grasp_indices[i];hand=t.agent.robot.links_map['gripper_link'].pose[0].sp;cfg=next(c for c in t.agent._sensor_configs if c.uid=='fetch_hand');cam=hand*cfg.pose[0].sp;pixels={n:int(np.count_nonzero(np.isin(s['segmentation'][0].cpu().numpy(),object_ids))) for n,s in obs['sensor_data'].items()};row=dict(seed=seed,target=label,step=i,close_step=close,signs=np.sign(q[i,[9,11]]).astype(int).tolist(),camera_above_hand_m=float(cam.p[2]-hand.p[2]),gripper_frame_x_pitch_deg=float(np.degrees(np.arcsin(np.clip(hand.to_transformation_matrix()[2,0],-1,1)))),tcp_approach_pitch_deg=float(np.degrees(np.arcsin(np.clip(t.agent.tcp.pose[0].sp.to_transformation_matrix()[2,2],-1,1)))),image_up_world_z=float(cam.to_transformation_matrix()[2,2]),pixels=pixels);report['grasps'].append(row)
     if i==close:
      sheet=Image.new('RGB',(768,282),'white')
      for k,n in enumerate(['left_base_camera_link','fetch_hand','right_base_camera_link']):
       sheet.paste(Image.fromarray(obs['sensor_data'][n]['rgb'][0].cpu().numpy()).resize((256,256)),(256*k,26));ImageDraw.Draw(sheet).text((256*k+4,4),f'{seed} {label} {n}',fill='black')
      sheet.save(out/f'{seed}-{label}-grasp.jpg')
   summary={k:dict(frames=len(v),visible_fraction=float(np.mean([r['visible'] for r in v])),missing_steps=[r['step'] for r in v if not r['visible']]) for k,v in byphase.items()};r=dict(seed=seed,visible_fraction=float(np.mean([r['visible'] for r in rows])),summary=summary,transitions=[x[:2] for x in trans],frames=rows);report['episodes'].append(r);(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',seed,r['visible_fraction'],{k:v['visible_fraction'] for k,v in summary.items()},flush=True)
finally:env.close();(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
