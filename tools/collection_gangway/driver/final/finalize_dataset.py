"""Finish a merged exact-1000 dataset for upload: validation seeds inside the dataset
(H4 check), and a README dataset card built from the dataset's own metadata.

usage (any python3): finalize_dataset.py DATASET VALIDATION_JSON TASK_TITLE
"""
import hashlib, json, sys
from pathlib import Path

root, validation, title = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
info = json.loads((root / 'meta/info.json').read_text())
meta = json.loads((root / 'source_h5_metadata.json').read_text())
rlds = json.loads((root / 'meta/source_rlds_metadata.json').read_text())
val = json.loads(validation.read_text())

# H4: 100 planner-successful validation seeds, disjoint from every episode and every attempt.
seeds = [int(s) for s in val['seeds']]
selected = {int(c['seed']): c for c in val['selected']}
episode_seeds = {int(e['scene_seed']) for e in meta['episodes']}
attempted = {int(o['scene_seed']) for o in meta['source_episode_outcomes']}
checks = dict(
    count_100=len(seeds) == 100 and len(set(seeds)) == 100,
    all_planner_success=all(selected[s]['planner_success'] for s in seeds),
    disjoint_from_episodes=not (set(seeds) & episode_seeds),
    disjoint_from_all_attempts=not (set(seeds) & attempted),
    same_profile=val['signature']['profile']['profile_version'] == meta['source_run']['signature']['profile']['profile_version'],
)
if not all(checks.values()):
    raise SystemExit(f'H4 failed: {checks}')
(root / 'validation_seeds.json').write_text(json.dumps(dict(
    task=title,
    profile_version=val['signature']['profile']['profile_version'],
    rule=val['selection'] + '; ' + val['selection_requires'],
    seeds=seeds,
    h4_checks=checks,
    source_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),
), indent=1) + '\n')

profile = meta['source_run']['signature']['profile']
sel = meta.get('episode_selection', {})
attempts = len(meta['source_episode_outcomes'])
successes = sum(1 for o in meta['source_episode_outcomes'] if o.get('success'))
instructions = sorted({e['instruction'] for e in meta['episodes']})
cams = {k.split('.')[-1]: v['shape'] for k, v in info['features'].items() if k.startswith('observation.images.')}
card = f"""---
license: mit
task_categories:
- robotics
tags:
- LeRobot
- MIKASA
- memory
- mobile-manipulation
---

# {title} — {info['total_episodes']} successful episodes (LeRobot v3.0)

Oracle (motion-planner) demonstrations of the MIKASA memory task `{profile['env_id']}` on the
DSFetch mobile manipulator (`{profile['robot_reference']['uid']}`, {profile['robot_reference']['repository']}
@ {profile['robot_reference']['commit'][:7]}), ManiSkill 3, CPU PhysX, collection profile {profile['profile_version']}.
{(chr(10) + '**Assisted grasp is ON** (`assisted_grasp: true` in the collection env kwargs): OmniGibson’s rule welds the '
'grasped object to the hand (rigid weld for free objects, spherical hold for handles) while at least two fingers '
'touch it with the object between the pads and the gripper is closing, and releases it when the gripper opens. '
'Evaluate policies in the same environment.' + chr(10)) if profile['env_kwargs'].get('assisted_grasp') else ''}

- Episodes: {info['total_episodes']} (frames: {info['total_frames']}), {info['fps']} Hz, recorded at 20 Hz control
  and kept every second step (actions held for two control steps and physically replayed at 10 Hz).
- `action` (13D, `pd_joint_pos`): 7 arm joints, gripper, head pan, head tilt, torso, base forward speed, base yaw rate.
- `observation.state` (12D): joint positions `qpos[3:]`; `global_state` (3D base pose) is debug-only, not a policy input.
- Cameras (RGB only): {', '.join(f'{k} {v[0]}x{v[1]}' for k, v in cams.items())}.
- Normalization statistics include q01/q10/q50/q90/q99 (pi0 / pi0.5 quantile normalization), globally and per episode.
- Language instruction{'s' if len(instructions) > 1 else ''}: {len(instructions)} (each under 100 PaliGemma tokens), listed below.

## Selection and success rate

Every attempted scene seed is kept in `meta/source_rlds_metadata.json` / `source_h5_metadata.json`
(seeds, lengths, durations, rewards, success, failed seeds). Planner success rate over all
{attempts} attempted seeds: {successes}/{attempts} = {successes / attempts:.1%}.
Episode selection: {sel.get('rule', 'all accepted episodes')} ({sel.get('selected', info['total_episodes'])} kept;
{len(sel.get('accepted_not_selected_seeds', []))} further accepted episodes were not included).

## Validation

`validation_seeds.json`: 100 scene seeds the planner solves, none used for collection or development
(evaluate a policy on exactly these seeds).

## Instructions

""" + '\n'.join(f'- {t}' for t in instructions) + '\n'
(root / 'README.md').write_text(card)
print(root.name, 'H4', checks, '| README written,', len(instructions), 'instructions,', f'SR {successes}/{attempts}')
