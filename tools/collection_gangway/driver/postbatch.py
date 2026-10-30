"""Post-batch gate for the 2026-09-28 collection: audits -> acceptance -> compact export -> retention -> ledger.

usage: postbatch.py <task> <batch>   (task in season_dish, cabinet_search, same_drawer)
Runs from anywhere; audits and exports run inside the task worktree. Never edits
recorded actions or deletes anything except verified RGB H5 via retention prune.
"""
import json, os, subprocess, sys, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

A = Path(__file__).parent
C = Path(os.path.expanduser(os.environ.get('SD_RUNS', '~/sd-collect/runs-ag')))  # lab H100 server; profile 21 (assisted grasp) by default
TOK = os.path.expanduser('~/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model')
SIM = os.environ['SIM']  # ~/sd-collect/env.sh
EXP = os.environ['EXP']  # ~/sd-collect/venv-export
REPOS = dict(same_drawer=os.path.expanduser('~/sd-collect/repo'),
             cabinet_search5=os.path.expanduser('~/sd-collect/repo-cab5c7'),  # five compartments, profile 7, feat/cabinet-fifth-door d65cd4c (unpushed)
             season_dish=os.path.expanduser('~/sd-collect/repo-sdish'),  # profile 26, collection/seasondish-ag 5dbf5a8
             cabinet_search=os.path.expanduser('~/sd-collect/repo-cab19'),  # profile 19 (d674243, unpushed); p18 repo-cab18 never collected;, check/assisted-grasp-collisions 3eef80c + render lanes 4e3ba84; p16 was repo-cab 4562981
             depth_recall=os.path.expanduser('~/sd-collect/repo-dr6'))  # profile 6, feat/depth-recall-mustard ddfbccf
AUDITS = dict(
    season_dish=['audit_followup_motion.py', 'audit_pilot_structure.py', 'audit_payload_clearance.py', 'audit_upper_clearance.py'],
    same_drawer=['audit_followup_motion.py', 'audit_pilot_structure.py', 'audit_transport_clearance.py', 'audit_upper_clearance.py'],
    cabinet_search=['audit_followup_motion.py', 'audit_pilot_structure.py', 'audit_transport_clearance.py', 'audit_cabinet_closing.py',
                    'audit_can_early.py'],  # can_early: profile 16 gate, ACCEPTANCE_RULES v4
    depth_recall=['audit_followup_motion_dr.py'])  # the neighbour's DR audit (task-guarded SameDrawer branch), v5
AUDITS['cabinet_search5'] = AUDITS['cabinet_search']  # five compartments: CabinetSearch's audits and rules
#: Tasks whose source .h5 go once the batch is exported and every step passed (owner 2026-10-19 «3) удалить»,
#: Depth Recall); json, logs and the ledger stay. Manifest: deletions/<batch>-h5.json.
H5_AFTER_EXPORT = ()  # 10-19: sources go to HF first (tools/hf_sources_loop.py on the Mac), then the loop deletes
ENV = dict(os.environ, PYTHONPATH='.', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           PYTHONUNBUFFERED='1')  # MS_ASSET_DIR, VK_ICD_FILENAMES, CUDA_VISIBLE_DEVICES come from ~/sd-collect/env.sh


def sh(args, log, cwd):
    with open(log, 'w') as f:
        return subprocess.run(args, cwd=cwd, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


def status(root, phase, seed):
    p = root / phase / str(seed) / 'result.json'
    return json.loads(p.read_text()).get('status') if p.exists() else None


def by_seed(path, key='episodes'):
    return {e['seed']: e for e in json.loads(Path(path).read_text())[key]} if Path(path).exists() else {}


def main(task, batch):
    import fcntl
    (A / 'locks').mkdir(exist_ok=True)
    lock = open(A / 'locks' / f'{task}-{batch}.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f'gate already running for {task}-{batch}', flush=True)
        return
    root = C / f'{task}-{batch}'
    repo = REPOS[task]
    run = json.loads((root / 'run.json').read_text())
    seeds = run['seeds']
    missing = [s for s in seeds if status(root, 'oracle', s) is None]
    if missing:
        raise SystemExit(f'batch incomplete: {missing[:5]}')
    ready = [s for s in seeds if status(root, 'rgb', s) == 'success']
    # 1. audits (source success episodes), views recorded not gated
    with ThreadPoolExecutor(max_workers=5) as pool:
        codes = dict(pool.map(lambda a: (a, sh([SIM, str(A / a), str(root)], A / f'{root.name}-{a[:-3]}.log', repo)),
                              AUDITS[task]))
    views_script = ['audit_restored_views_dr.py'] if task == 'depth_recall' else []
    codes['views'] = sh([SIM, str(A / 'run_view_shards.py'), str(root), repo, '2', 'b', *views_script],
                        A / f'{root.name}-views.log', repo)
    motion = by_seed(A / f'{root.name}-motion-audit.json')
    transport = by_seed(A / f'{root.name}-transport-clearance.json')
    upper = by_seed(A / f'{root.name}-upper-clearance.json')
    payload = by_seed(A / f'{root.name}-payload-clearance.json')
    closing = by_seed(A / f'{root.name}-closing-clearance.json')
    can = by_seed(A / f'{root.name}-can-early.json')
    views = by_seed(A / f'{root.name}-views-combined.json')
    # 2. acceptance
    rows, accepted = [], []
    for s in seeds:
        src = status(root, 'oracle', s)
        reasons = []
        if src != 'success':
            rows.append(dict(seed=s, source=src, accepted=False, reasons=['not a source success']))
            continue
        for ph in ('native', 'validated', 'rgb'):
            if status(root, ph, s) != 'success':
                reasons.append(f'{ph} not success')
        m = motion.get(s)
        if m is None:
            reasons.append('no motion audit')
        else:
            gated = ('D2_pass', 'D4_pass', 'action_pairs_equal') + (
                () if task in ('cabinet_search', 'cabinet_search5', 'depth_recall') else ('D5_direction_pass',))  # DR D5: owner 2026-10-18
            for k in gated:
                if not m.get(k):
                    reasons.append(k + ' failed')
            if m.get('odd_gripper_switches'):
                reasons.append('odd gripper switch')
            if task in ('season_dish', 'depth_recall'):
                if not (m.get('D6') or {}).get('passed'):
                    reasons.append('D6 identical actions')
            if task == 'season_dish':
                ut = m.get('upright_transfer') or {}
                if not all(ut.get(ph, {}).get('strictly_under_15_deg') for ph in ('oracle', 'native', 'validated')):
                    reasons.append('transfer tilt >= 15 deg')
                if not (m.get('pour') or {}).get('wrist_only'):
                    reasons.append('pour not wrist-only')
            if task == 'same_drawer':
                r = m.get('apple_retention') or {}
                if not r.get('final_on_plate') or r.get('latch_violated'):
                    reasons.append('apple retention')
        if task in ('same_drawer', 'cabinet_search', 'cabinet_search5'):
            t = transport.get(s)
            if t is None or t['collision_frames']:
                reasons.append('transport clearance' if t else 'no transport audit')
        if task in ('season_dish', 'same_drawer'):
            u = upper.get(s)
            if u is None or u['upper_collision_frames']:
                reasons.append('upper clearance' if u else 'no upper audit')
        if task == 'season_dish':
            p = payload.get(s)
            if p is None or p['collision_frames']:
                reasons.append('payload clearance' if p else 'no payload audit')
        if task in ('cabinet_search', 'cabinet_search5'):
            c = closing.get(s)
            if c is None or c['unexpected_contacts']:
                reasons.append('closing unexpected contact' if c else 'no closing audit')
            k = can.get(s)
            if k is None or not k['passed']:
                reasons.append('can moved before the push' if k else 'no can audit')
        row = dict(seed=s, source=src, accepted=not reasons, reasons=reasons,
                   D5=(m or {}).get('D5_direction_pass'), D6=(m or {}).get('D6', {}).get('passed'),
                   E1=(views.get(s) or {}).get('visible_fraction'))
        rows.append(row)
        if not reasons:
            accepted.append(s)
    # 3. export accepted (compact), 4. retention
    ledger = A / 'ledger' / f'{root.name}.json'
    previous = json.loads(ledger.read_text()) if ledger.exists() else {}
    exported_before = previous.get('exported_seeds', previous.get('accepted_seeds', []) if previous.get('export') else [])
    exports_before = previous.get('exports', [previous['export']] if previous.get('export') else [])
    to_export = [s for s in accepted if s not in exported_before]
    export = None
    if to_export:
        part = '' if not exports_before else f'-part{len(exports_before) + 1}'
        name = f'{root.name}{part}-lerobot-test-{len(to_export)}ep'
        export = C / 'exports' / name
        export.parent.mkdir(exist_ok=True)
        if not export.exists():
            codes['export'] = sh([EXP, '-m', 'utils.collection.export_lerobot', '--input', str(root), '--output', str(export),
                                  '--repo-id', f'mikasa-local/{name}', '--tokenizer', TOK, '--seeds', *map(str, to_export),
                                  '--video-profile', 'compact'], A / f'{root.name}-export.log', repo)
        if codes.get('export', 0) == 0:
            codes['retention_prepare'] = sh([EXP, '-m', 'utils.collection.retention', 'prepare', str(export)],
                                            A / f'{root.name}-retention-prepare.log', repo)
            if codes['retention_prepare'] == 0:
                codes['retention_prune'] = sh([EXP, '-m', 'utils.collection.retention', 'prune', str(export)],
                                              A / f'{root.name}-retention-prune.log', repo)
    source_success = sum(r['source'] == 'success' for r in rows)
    summary = dict(task=task, batch=batch, run=str(root), signature=run['signature']['code_sha256'],
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), attempts=len(seeds),
                   source_successes=source_success, sr=source_success / len(seeds), rgb_ready=len(ready),
                   accepted=len(accepted), accepted_seeds=accepted,
                   exported_seeds=sorted(set(exported_before) | (set(to_export) if export and codes.get('export', 0) == 0 else set())),
                   exports=exports_before + ([str(export)] if export and codes.get('export', 0) == 0 else []),
                   rules_version=json.loads((A / 'ACCEPTANCE_RULES.json').read_text())['version'],
                   step_codes=codes, rows=rows)
    (A / 'ledger').mkdir(exist_ok=True)
    ledger.write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'accepted_seeds', 'exported_seeds')}), flush=True)
    rejected = [r for r in rows if r['source'] == 'success' and not r['accepted']]
    for r in rejected:
        print('REJECTED', r['seed'], r['reasons'], flush=True)
    if task in H5_AFTER_EXPORT and not any(codes.values()) and summary['exports']:
        gone = []
        for f in sorted(root.rglob('*.h5')):
            gone.append(dict(path=str(f.relative_to(C)), bytes=f.stat().st_size))
            f.unlink()
        (A / 'deletions').mkdir(exist_ok=True)
        (A / 'deletions' / f'{root.name}-h5.json').write_text(json.dumps(dict(
            rule='owner 2026-10-19: source h5 deleted after export', files=gone), indent=1) + '\n')
        print('deleted', len(gone), 'source h5', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
