"""Read-only measurements on retained source H5; no dataset qualification by inference."""
import json,sys
from pathlib import Path
import h5py,numpy as np
A=Path(__file__).parent;R=Path(sys.argv[1]);run=json.loads((R/'run.json').read_text());rows=[];initial=[];allactions=[];allq=[];unavailable=[]
for seed in run['seeds']:
 p=R/'oracle'/str(seed)/'result.json'
 if not p.exists():raise RuntimeError(f'incomplete pool: {seed}')
 result=json.loads(p.read_text());h=p.parent/('trajectory.h5' if result.get('success') else 'failed-trajectory.h5')
 if not h.exists():
  unavailable.append(dict(seed=seed,status=result['status'],reason='No completed H5'))
  continue
 try:
  with h5py.File(h) as probe: probe['traj_0/actions'].shape
 except (OSError,KeyError) as exc:
  if result.get('success'):raise
  unavailable.append(dict(seed=seed,status=result['status'],reason=str(exc)))
  continue
 with h5py.File(h) as f:
  g=f['traj_0'];a=g['actions'][:];q=g['qpos'][:];n=len(a);initial.append(q[0]);allactions.append(a);allq.append(q)
  row=dict(seed=seed,success=result.get('success'),status=result['status'],actions13=bool(a.shape==(n,13)),qpos15=bool(q.shape==(n+1,15)),proprio12_exact=bool(np.array_equal(g['proprio'][:],q[:,3:])),global3_exact=bool(np.array_equal(g['global_state'][:],q[:,:3])),timestamp20hz=bool(np.allclose(g['timestamp'][:],np.arange(n+1)/20,atol=1e-8,rtol=0)),binary_gripper=bool(set(np.unique(a[:,7]))<= {-1,1}),base_bounded=bool(np.abs(a[:,11:13]).max()<=1),no_tiny_base=bool(not np.any((np.abs(a[:,11:13])>0)&(np.abs(a[:,11:13])<1e-6))),steps=n,final_success=result.get('final_success'),horizon=run['signature']['profile']['task']['horizon'])
  base=a[:,11:13]; norm=np.max(np.abs(base),axis=1)
  row['small_base_components_1e6']=int(np.sum((np.abs(base)>0)&(np.abs(base)<1e-6)))
  row['nonzero_near_stop_frames']=np.where((norm>0)&(norm<1e-6))[0].tolist()
  row['terminal_base_exact_zero']=bool(np.all(base[-1]==0))
  row['below_horizon']=n<row['horizon'];rows.append(row)
initial=np.asarray(initial);a=np.concatenate(allactions);q=np.concatenate(allq)
report=dict(run=str(R),source_hz=20,recordings=len(rows),unavailable_failed_recordings=unavailable,initial_qpos_std=initial.std(0).tolist(),pooled_action_std=a.std(0).tolist(),pooled_proprio_std=q[:,3:].std(0).tolist(),episodes=rows,scope='Source recordings only. LeRobot G1-G5 and stored 10Hz dataset F3 require a fresh export; not claimed here. Unreadable timeout H5 files remain failed attempts in the run denominator.')
(A/(R.name+'-structure.json')).write_text(json.dumps(report,indent=2)+'\n');print(R.name,len(rows),{k:all(r[k] for r in rows) for k in ['actions13','qpos15','proprio12_exact','global3_exact','timestamp20hz','binary_gripper','base_bounded','terminal_base_exact_zero']})
