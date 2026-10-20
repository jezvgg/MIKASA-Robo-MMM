import json,os,subprocess,sys,fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
A=Path(__file__).parent;root=Path(sys.argv[1]);repo=Path(sys.argv[2]);n=int(sys.argv[3]);tag=sys.argv[4]
script=sys.argv[5] if len(sys.argv)>5 else 'audit_restored_views.py'  # Depth Recall: audit_restored_views_dr.py
lock=(A/f'{root.name}-views-supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
seeds=sorted(int(p.parent.name) for p in (root/'validated').glob('*/result.json') if json.loads(p.read_text()).get('success'))
shards={k:[s for s in seeds if s%n==k] for k in range(n)}
def run(k):
 env=dict(os.environ,RESTORATION_VIEW_SHARD=f'-{tag}{k}')
 with (A/f'{root.name}-views-{tag}{k}.log').open('a') as log:
  result=subprocess.run([sys.executable,str(A/script),str(root),*[str(x) for x in shards[k]]],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 return k,result.returncode
with ThreadPoolExecutor(max_workers=n) as pool:
 for row in pool.map(run,range(n)):print(row,flush=True)
report={'run':str(root),'scope':'All 10 Hz states, all three cameras; not human E5','episodes':[],'grasps':[],'inputs':[]}
for k in range(n):
 p=A/f'{root.name}-views-{tag}{k}'/'result.json';d=json.loads(p.read_text());report['inputs'].append(str(p));report['episodes'].extend(d['episodes']);report['grasps'].extend(d['grasps'])
report['episodes'].sort(key=lambda x:x['seed']);report['seeds']=seeds
assert sorted(r['seed'] for r in report['episodes'])==seeds
(A/f'{root.name}-views-combined.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',len(seeds),flush=True)
