"""Resumable per-task collection loop for the 2026-09-28 collection.

usage: collection_driver.py <task> <batch_size> <jobs> <target_accepted> <budget_gb> [prefix] [first_seed]
prefix names a profile generation (e.g. p23-) so its batches never mix with older ones.

Each batch is declared (manifest written) before it runs; every attempt counts
toward SR. The next batch's campaign overlaps the previous batch's postbatch
gate. When the collection folder exceeds the disk budget the driver writes a
PAUSED status and waits; it never deletes anything itself.
"""
import json, os, subprocess, sys, time, datetime
from pathlib import Path

A = Path(__file__).parent
C = Path(os.path.expanduser(os.environ.get('SD_RUNS', '~/sd-collect/runs-ag')))  # lab H100 server; profile 21 (assisted grasp) by default
TOK = os.path.expanduser('~/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model')
SIM = os.environ['SIM']  # ~/sd-collect/env.sh
REPOS = dict(same_drawer=os.path.expanduser('~/sd-collect/repo'),
             cabinet_search5=os.path.expanduser('~/sd-collect/repo-cab5c7'),  # five compartments, profile 7, feat/cabinet-fifth-door d65cd4c (unpushed)
             season_dish=os.path.expanduser('~/sd-collect/repo-sdish'),  # profile 26, collection/seasondish-ag 5dbf5a8
             cabinet_search=os.path.expanduser('~/sd-collect/repo-cab19'),  # profile 19 (d674243, unpushed); p18 repo-cab18 never collected; (rug, hand collides), check/assisted-grasp-collisions 3eef80c + render lanes 4e3ba84; p16 was repo-cab 4562981
             depth_recall=os.path.expanduser('~/sd-collect/repo-dr6'))  # profile 6, feat/depth-recall-mustard ddfbccf
RANGES = dict(season_dish=(7780000, 7789999),  # profile 26; 7760000-7769999 was the pod's p25
               cabinet_search5=(7920000, 7929999),  # five compartments: profile 1 from 7920000, 2 from 7920200, 3 from 7920300 (launch_cab5.sh first seed)
               cabinet_search=(7890000, 7899999),  # profile 18; 7880000-7889999 was profile 16, 7860000-7869999 the pod's p14/p15
               depth_recall=(7970000, 7979999))  # profile 5 (spices, pad map)
BANK = A / 'bank.json'  # 500 per reset drawer from 9100000 (utils.collection.drawer_balance plan)
QUOTA = 262  # accepted per drawer: 250 for the dataset plus room for D6 rejects at selection
ENV = dict(os.environ, PYTHONPATH='.', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           PYTHONUNBUFFERED='1')  # MS_ASSET_DIR, VK_ICD_FILENAMES, CUDA_VISIBLE_DEVICES come from ~/sd-collect/env.sh


def log(task, **kw):
    kw['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(A / f'driver-{task}.jsonl', 'a') as f:
        f.write(json.dumps(kw) + '\n')
    print(json.dumps(kw), flush=True)


def size_gb(path):
    out = subprocess.run(['du', '-s', '--block-size=1M', str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) / 1024 if out else 0.


def state(task, prefix=''):
    attempted, accepted, batches = [], [], []
    for run in sorted(C.glob(f'{task}-{prefix}train-*')):
        if not (run / 'run.json').exists():
            continue
        batches.append(run.name)
        attempted += json.loads((run / 'run.json').read_text())['seeds']
        ledger = A / 'ledger' / f'{run.name}.json'
        if ledger.exists():
            accepted += json.loads(ledger.read_text())['accepted_seeds']
    return attempted, accepted, batches


def next_seeds(task, attempted, accepted, count, first_seed=None):
    if task == 'same_drawer':
        sys.path.insert(0, REPOS[task])
        from utils.collection.drawer_balance import next_seeds as drawer_next
        return drawer_next(json.loads(BANK.read_text()), attempted, accepted, count=count, quota=QUOTA)
    first, last = RANGES[task]
    first = first_seed or first
    start = max(attempted) + 1 if attempted else first
    return list(range(start, min(start + count, last + 1)))


def campaign(task, batch, seeds, jobs):
    manifest = A / 'batches' / f'{task}-{batch}.json'
    if not manifest.exists():
        manifest.write_text(json.dumps(dict(seeds=seeds, task=task, batch=batch, purpose='train',
            declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rule='SEED_PLAN.json training rule; all attempts count toward H2'), indent=1))
    seeds = json.loads(manifest.read_text())['seeds']
    if task == 'same_drawer':
        how = ['--seed-manifest', str(manifest)]
    else:
        assert seeds == list(range(seeds[0], seeds[0] + len(seeds)))
        how = ['--start-seed', str(seeds[0]), '--num-seeds', str(len(seeds))]
    args = [SIM, '-m', 'utils.collection.campaign', '--profile', task, '--tokenizer', TOK,
            '--output', str(C / f'{task}-{batch}'), *how, '--purpose', 'train', '--jobs', str(jobs),
            '--through', 'rgb', '--skip-native-rgb', '--timeout', '1800']
    with open(A / f'{task}-{batch}.log', 'a') as f:
        return subprocess.run(args, cwd=REPOS[task], env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


MAX_BATCHES = int(os.environ.get('MAX_BATCHES', '0'))  # 0 = no limit
MAX_PENDING_GATES = 2  # 2026-09-30 22:xx: two exports (~10 GiB temporary PNG each) fit the budget


def gate_running(name):
    import fcntl
    path = A / 'locks' / f'{name}.lock'
    if not path.exists():
        return False
    with open(path, "a") as f:  # NFS: an exclusive flock needs a writable descriptor (EBADF on read-only)
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
    return False


def main(task, batch_size, jobs, target, budget, prefix='', first_seed=None):
    pending = []
    while True:
        attempted, accepted, batches = state(task, prefix)
        if len(accepted) >= target:
            log(task, event='target reached', accepted=len(accepted), attempts=len(attempted))
            break
        used = size_gb(C)
        if used > budget:
            log(task, event='PAUSED disk budget', collection_gb=round(used, 2), budget_gb=budget)
            time.sleep(900)
            continue
        # resume an unfinished batch before declaring a new one
        unfinished = [b for b in batches if not (A / 'ledger' / f'{b}.json').exists()
                      and b not in [p[0] for p in pending] and not gate_running(b)]
        if unfinished:
            batch = unfinished[0].split(f'{task}-', 1)[1]
            seeds = None
        else:
            n = len(batches) + 1
            if MAX_BATCHES and len(batches) >= MAX_BATCHES:  # a pilot of a fixed number of batches
                log(task, event='max batches', batches=len(batches), accepted=len(accepted))
                break
            batch = f'{prefix}train-{n:03d}'
            seeds = next_seeds(task, attempted, accepted, batch_size, first_seed)
            if not seeds:
                log(task, event='seed range exhausted', accepted=len(accepted))
                break
        log(task, event='campaign start', batch=batch, seeds=len(seeds) if seeds else 'resume')
        code = campaign(task, batch, seeds, jobs)
        log(task, event='campaign end', batch=batch, exit=code)
        while len(pending) >= MAX_PENDING_GATES:
            name, proc = pending.pop(0)
            proc.wait()
            log(task, event='postbatch end', batch=name, exit=proc.returncode)
        f = open(A / f'{task}-{batch}-postbatch.log', 'w')
        pending.append((f'{task}-{batch}', subprocess.Popen([SIM, str(A / 'postbatch.py'), task, batch],
                                                            stdout=f, stderr=subprocess.STDOUT)))
    for name, proc in pending:
        proc.wait()
        log(task, event='postbatch end', batch=name, exit=proc.returncode)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]),
         sys.argv[6] if len(sys.argv) > 6 else '', int(sys.argv[7]) if len(sys.argv) > 7 else None)
