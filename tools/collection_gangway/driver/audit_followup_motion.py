"""Audit recorded commands; keep all attempts, including failures, in the denominator."""
import json,sys,collections
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import h5py,numpy as np
from utils.collection.checklist import motion_metrics,pause_check
root=Path(sys.argv[1]);run=json.loads((root/'run.json').read_text());prof=run['signature']['profile'];task=prof['env_id'];rows=[]
for seed in run['seeds']:
 d=root/'oracle'/str(seed);p=d/'result.json'
 if not p.exists():continue
 outcome=json.loads(p.read_text());row={'seed':seed,'status':outcome['status'],'source_success':outcome.get('success'),'phases':{}}
 ev=[json.loads(s) for s in (d/'events.jsonl').read_text().splitlines()] if (d/'events.jsonl').exists() else []
 for phase in ['oracle','native','validated','rgb']:
  rp=root/phase/str(seed)/'result.json'
  if rp.exists():row['phases'][phase]=json.loads(rp.read_text()).get('success')
 h=d/('trajectory.h5' if (d/'trajectory.h5').exists() else 'failed-trajectory.h5')
 if h.exists():
  try:
   with h5py.File(h) as probe:
    assert 'traj_0/actions' in probe and len(probe['traj_0/actions'])>0 and 'traj_0/env_states/articulations/ds_fetch' in probe
  except (OSError,KeyError,AssertionError) as error:
   row['diagnostic_unavailable']=type(error).__name__+': '+str(error);rows.append(row);continue
  with h5py.File(h) as f:
   g=f['traj_0'];a=g['actions'][:];q=g['env_states/articulations/ds_fetch'][:,13:28]
  meta=json.loads(h.with_suffix('.json').read_text());names=meta.get('mikasa_data',{}).get('state_joint_names')
  if names is None:names=['root_x_axis_joint','root_y_axis_joint','root_z_rotation_joint','torso_lift_joint','head_pan_joint','shoulder_pan_joint','head_tilt_joint','shoulder_lift_joint','upperarm_roll_joint','elbow_flex_joint','forearm_roll_joint','wrist_flex_joint','wrist_roll_joint','r_gripper_finger_joint','l_gripper_finger_joint']
  m=motion_metrics(a[::2],q[::2],names,fps=10);row['motion10']=m;row['D6']=pause_check(a[::2],prof);row['action_pairs_equal']=bool(np.array_equal(a[:2*(len(a)//2):2],a[1:2*(len(a)//2):2]));row['final_partial_pair']=bool(len(a)%2);row['odd_gripper_switches']=[int(i) for i in np.flatnonzero(np.diff(a[:,7])!=0)+1 if i%2]
  row['D2_pass']=bool(max(m['roll_span'])<=np.pi and min(m['roll_min'])>=-np.pi and max(m['roll_max'])<=np.pi)
  row['D4_pass']=bool(m['max_continuous_turn_rad']<=np.pi)
  row['D5_direction_pass']=bool(m['base_direction_changes']<=2)
  if 'Season' in task:
   row['cue_wait_full100']=bool(len(a)>=100 and not np.any(a[:100,11:13]))
   pours=[e for e in ev if e['message']=='pour' and e.get('motion')=='wrist_roll_only']
   if pours:
    start=pours[-1]['step'];seg=a[start:];cols=[0,1,2,3,4,5,10];row['pour']={'start':start,'other_joint_command_span':np.ptp(seg[:,cols],axis=0).tolist(),'base_stopped':bool(not np.any(seg[:,11:13])),'wrist_command_span':float(np.ptp(seg[:,6])),'target_degrees':pours[-1]['target_degrees']};row['pour']['wrist_only']=max(row['pour']['other_joint_command_span'])<1e-7 and row['pour']['base_stopped']
   selected=[e for e in ev if e['message']=='connected grasp/lift/carry selected']
   folds=[e for e in ev if e['message']=='compact upright condiment carry']
   drive=[e for e in ev if e['message']=='drive to bowl dock']
   if selected:
    finish=next((e['step'] for e in ev if e['message']=='pour'),len(q)-1)
    seg=q[selected[-1]['step']:finish+1]
    row['grasp_lift_carry']={'elbow_range': [float(seg[:,9].min()),float(seg[:,9].max())], 'no_elbow_flip': bool(np.all(seg[:,9]>0)), 'selected_chain_count':len(selected), 'carry_event_count':len(folds)}
   route=next((e for e in ev if e['message']=='bowl route selected'),None)
   row['bowl_route']=route.get('route') if route else None
   if route:
    stop=next((e['step'] for e in ev if e['message']=='pour'),len(a))
    row['transfer_base_commands_zero']=bool(not np.any(a[route['step']:stop,11:13]))
    target=next(e['target_is_shaker'] for e in ev if e['message']=='target chosen')
    from transforms3d.quaternions import quat2mat
    row['upright_transfer']={}
    for phase in ['oracle','native','validated']:
     hd=root/phase/str(seed);files=list(hd.glob('*trajectory.h5'))
     if not files:continue
     with h5py.File(files[0]) as hf:
      actors=hf['traj_0/env_states/actors'];key=next(k for k in actors if ('shaker' if target else 'condiment_bottle') in k);states=actors[key][:]
      end=min(stop,len(states)-1);start=min(next((e['step'] for e in ev if e['message']=='grasped'),route['step']),end)
      tilt=np.rad2deg(np.arccos(np.clip([quat2mat(v[3:7])[2,2] for v in states[start:end+1]],-1,1)))
     row['upright_transfer'][phase]={'start':start,'end':end,'max_tilt_deg':float(max(tilt)),'end_tilt_deg':float(tilt[-1]),'strictly_under_15_deg':bool(max(tilt)<15)}
  elif 'CabinetSearch' in task:
   openings=[e for e in ev if e['message']=='door open'];empty_rest=[]
   for begin in openings:
    following=[e for e in ev if e['step']>=begin['step'] and e['event']=='phase']
    empty=next((e for e in following if e['message'] in ['empty','FOUND, not yet touched']),None)
    if empty and empty['message']=='empty':
     closed=next((e for e in following if e['message']=='door closed'),None)
     end=closed['step'] if closed else len(a)
     empty_rest.extend(e for e in following if e['step']<end and ('fold to' in e['message'] or 'stow hand' in e['message']))
   row['empty_inspection_rest_events']=empty_rest
   row['home_drives']=[e for e in ev if e['message']=='drive home'];row['forbidden_route_events']=[e for e in ev if 'south waypoint' in e['message'] or 'fold after the can' in e['message']]
   row['finish_at_can']=bool(outcome.get('success') and not row['forbidden_route_events'] and outcome.get('task_metrics',{}).get('nudged'))
  else:
   pushes=[e for e in ev if e['message']=='close drawer with arm'];ends=[e for e in ev if e['message']=='drawer stroke ended'];row['push_count']=len(pushes);row['push_fingers_open']=bool(pushes) and all(np.all(a[e['step']:next((x['step'] for x in ends if x['step']>=e['step']),len(a)),7]==1) for e in pushes)
   row['push_base_stopped']=bool(pushes) and all(not np.any(a[e['step']:next((x['step'] for x in ends if x['step']>=e['step']),len(a)),11:13]) for e in pushes)
   pulls=[e for e in ev if e['message']=='pull drawer with arm'];row['pull_base_stopped']=all(not np.any(a[e['step']:next((x['step'] for x in ends if x['step']>=e['step']),len(a)),11:13]) for e in pulls if any(x['step']>=e['step'] for x in ends));row['grips']=[e for e in ev if e['message']=='handle grip'];row['pre_release_settle']=[e for e in ev if e['message']=='settle before apple release']
   row['release_clearance']=[e for e in ev if e['message'] in ['lift open hand from apple','withdraw open hand along fingers','extend finger-axis release clearance']]
   with h5py.File(h) as f:
    states=f['traj_0/env_states'];ap=states['actors/apple'][:];pl=states['actors/plate'][:]
    monitored=states['apple_placement_started'][:].astype(bool)
    on=(np.linalg.norm(ap[:,:2]-pl[:,:2],axis=1)<=prof['task']['plate_radius']) & (ap[:,2]-pl[:,2]>=prof['task']['apple_z_band'][0]) & (ap[:,2]-pl[:,2]<=prof['task']['apple_z_band'][1])
    row['apple_retention']={'started':bool(monitored.any()),'first_release_step':int(np.flatnonzero(monitored)[0]) if monitored.any() else None, 'off_plate_steps':np.flatnonzero(monitored & ~on).tolist(), 'latch_violated':bool(np.any(states['apple_retention_violated'][:])),'final_on_plate':bool(on[-1])}
   starts=[e for e in ev if e['message']=='smooth return through open aisle'];travel_ends=[e for e in ev if e['message']=='smooth travel finished']
   if starts and travel_ends:
    start=starts[-1]['step'];end=travel_ends[-1]['step'];v=a[start:end:2,11];w=a[start:end:2,12]*3.14
    moving=np.flatnonzero(v>1e-5);interior=v[moving[0]:moving[-1]+1] if len(moving) else v
    row['smooth_return']={'start':start,'end':end,'duration_s':(end-start)/20,'moving_ticks':int(len(moving)),'intermediate_stops':int(np.sum(interior<=1e-5)),'linear_acceleration_max':float(np.max(abs(np.diff(v)))/.1) if len(v)>1 else 0,'yaw_acceleration_max':float(np.max(abs(np.diff(w)))/.1) if len(w)>1 else 0,'max_v':float(max(v)),'max_w':float(max(abs(w))),'configuration':starts[-1]}
   if row['pre_release_settle']:
    with h5py.File(h) as f:
     states=f['traj_0/env_states'];apple=states['actors/apple'][:] if 'apple' in states['actors'] else states['actors/'+next(k for k in states['actors'] if 'apple' in k)][:];joint=states['articulations/ds_fetch'][:]
    checks=[]
    for event in row['pre_release_settle']:
     i=event['step'];xyz=apple[i-4:i+1,:3];quat=apple[i-4:i+1,3:7];quat=quat/np.linalg.norm(quat,axis=1,keepdims=True);v=np.linalg.norm(np.diff(xyz,axis=0),axis=1)*20;w=2*np.arccos(np.clip(np.abs(np.sum(quat[1:]*quat[:-1],axis=1)),0,1))*20;arm=np.max(np.abs(joint[i-3:i+1,28:43][:,[5,7,8,9,10,11,12]]),axis=1);checks.append(dict(step=i,linear_speed=v.tolist(),angular_speed=w.tolist(),arm_speed=arm.tolist(),pass_four=bool(np.all(v<.01)&np.all(w<.5)&np.all(arm<.05))))
    row['measured_release_checks']=checks
 rows.append(row)
summary={'attempted':len(rows),'assigned':len(run['seeds']),'source_statuses':dict(collections.Counter(r['status'] for r in rows)),'source_successes':sum(r['source_success'] is True for r in rows),'native_successes':sum(r['phases'].get('native') is True for r in rows),'held_successes':sum(r['phases'].get('validated') is True for r in rows),'rgb_successes':sum(r['phases'].get('rgb') is True for r in rows)}
report={'run':str(root),'profile_version':prof['profile_version'],'signature_sha256':run['signature']['code_sha256'],'complete':len(rows)==len(run['seeds']),'summary':summary,'episodes':rows};out=Path(__file__).parent/(root.name+'-motion-audit.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary));print(out)
