"""Prospective drawer seed banks, all-attempt SR, and accepted-data quotas."""
from __future__ import annotations
import argparse
from pathlib import Path

from .contract import read_json, write_json


def interleave(strata):
    return [seeds[i] for i in range(max(map(len,strata.values()),default=0))
            for _,seeds in sorted(strata.items()) if i<len(seeds)]


def build_manifest(output, start_seed, per_drawer, excluded=()):
    """Classify resets before any oracle attempts; do not select by success."""
    from .profile import load_profile, runtime_signature, make_env
    if output.exists() or start_seed<0 or per_drawer<1:
        raise ValueError('Use a new manifest, nonnegative seed and positive quota')
    signature=runtime_signature(load_profile('same_drawer'))
    env=make_env({'signature':signature})
    strata={i:[] for i in range(4)};excluded=set(excluded);seed=start_seed
    try:
        while min(map(len,strata.values()))<per_drawer:
            if seed-start_seed>max(1000,per_drawer*40):
                raise RuntimeError('Seed classification budget exhausted')
            if seed not in excluded:
                env.reset(seed=seed)
                target=int(env.unwrapped.target_drawer.item())
                if len(strata[target])<per_drawer:strata[target].append(seed)
            seed+=1
    finally:
        env.close()
    manifest=dict(version=1,task='same_drawer',selection='prospective_reset_target_only',
        start_seed=start_seed,examined_through=seed-1,per_drawer=per_drawer,
        excluded_seeds=sorted(excluded),seeds=interleave(strata),
        seed_to_drawer={str(s):i for i,ss in strata.items() for s in ss},
        task_config=signature['task_config'],profile_version=signature['profile']['profile_version'])
    write_json(output,manifest)
    return manifest


def report(root, manifest, accepted_seeds=()):
    """Accepted means external quality gates passed, not just physical success."""
    accepted=set(map(int,accepted_seeds));pool=set(manifest['seeds'])
    if not accepted<=pool:raise ValueError('Accepted seed outside prospective pool')
    rows={i:dict(attempts=0,successes=0,held_replays=0,accepted=0,failures=[]) for i in range(4)}
    for seed in manifest['seeds']:
        target=int(manifest['seed_to_drawer'][str(seed)]);row=rows[target]
        folder=root/'oracle'/str(seed);result=folder/'result.json'
        if not result.exists():
            if seed in accepted:raise ValueError('Unattempted seed cannot be accepted')
            continue
        import json
        event_file=folder/'events.jsonl'
        events=[json.loads(line) for line in event_file.read_text().splitlines()] if event_file.exists() else []
        actual=next((e['target'] for e in events if e.get('message')=='remember open drawer'),None)
        if actual is not None and actual!=target:raise ValueError(f'Drawer manifest mismatch for seed {seed}')
        outcome=read_json(result);row['attempts']+=1;row['successes']+=int(outcome.get('success',False))
        if outcome.get('success') and actual is None:raise ValueError('Successful episode lacks drawer provenance')
        replay=root/'validated'/str(seed)/'result.json'
        held=replay.exists() and read_json(replay).get('success',False)
        row['held_replays']+=int(held)
        if seed in accepted:
            if not outcome.get('success') or not held:raise ValueError('Accepted seed failed physical gates')
            row['accepted']+=1
        if not outcome.get('success'):
            stages=[e.get('message') for e in events if e.get('event')=='phase' and 'waypoint noise' not in e.get('message','')]
            row['failures'].append(dict(seed=seed,status=outcome['status'],last_stage=stages[-1] if stages else None))
    for row in rows.values():
        row['success_rate']=row['successes']/row['attempts'] if row['attempts'] else None
        row['quota_remaining']=max(0,250-row['accepted'])
    return dict(version=1,drawer_numbering='0 bottom, 3 top',drawers=rows,
                total_attempts=sum(r['attempts'] for r in rows.values()),
                total_accepted=sum(r['accepted'] for r in rows.values()))


def next_seeds(manifest, attempted, accepted, count=4, quota=250):
    """Resume without retrying failures or changing chronological seed selection."""
    pool=set(manifest['seeds']);attempted=set(attempted);accepted=set(accepted)
    if not accepted<=attempted<=pool:raise ValueError('Invalid attempt/acceptance membership')
    accepted_counts={i:0 for i in range(4)}
    for seed in accepted:accepted_counts[int(manifest['seed_to_drawer'][str(seed)])]+=1
    selected=[];pending={i:0 for i in range(4)}
    for seed in manifest['seeds']:
        target=int(manifest['seed_to_drawer'][str(seed)])
        if seed not in attempted and accepted_counts[target]+pending[target]<quota:
            selected.append(seed);pending[target]+=1
            if len(selected)==count:break
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--output',type=Path,required=True)
    plan.add_argument('--start-seed',type=int,required=True);plan.add_argument('--per-drawer',type=int,default=10)
    plan.add_argument('--exclude',type=Path,action='append',default=[])
    stats=sub.add_parser('report');stats.add_argument('--manifest',type=Path,required=True)
    stats.add_argument('--run',type=Path,required=True);stats.add_argument('--accepted',type=Path)
    stats.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='plan':
        excluded=[]
        for path in args.exclude:
            data=read_json(path);excluded.extend(data if isinstance(data,list) else data['seeds'])
        build_manifest(args.output,args.start_seed,args.per_drawer,excluded)
    else:
        accepted=read_json(args.accepted) if args.accepted else []
        write_json(args.output,report(args.run,read_json(args.manifest),accepted))


if __name__=='__main__':main()
