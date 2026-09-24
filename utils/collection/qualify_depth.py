"""Finite reset, visual-evidence and hidden-answer checks for Depth Recall v1.

Counterfactual state edits are diagnostics only, never demonstrations.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .client import as_numpy, policy_observation
from .contract import CAMERAS, read_json, write_json
from .pipeline import assert_signature, read_state
from .profile import make_env


def capture(base):
    return {k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in policy_observation(base.get_obs(),
                base.get_language_instruction()[0]).items()}


def snapshot(base):
    return dict(qpos=as_numpy(base.agent.robot.get_qpos())[0].tolist(),
        props=[as_numpy(p.pose.raw_pose)[0].tolist() for p in base.props],
        original_slots=as_numpy(base.original_slots)[0].tolist())


def qualify(run_path, output, source_seed=300, start_seed=300, count=32):
    import json
    import torch
    from mani_skill.utils.structs import Pose
    from utils.mikasa.seeding import seed_everything
    from my_scenes.depth_recall_v1 import TASK_STATE_KEYS

    output.mkdir(parents=True, exist_ok=False)
    run = read_json(run_path / 'run.json')
    assert run['scene'] == 'MikasaDepthRecall-v1'
    assert_signature(run)
    directory = run_path / 'oracle' / str(source_seed)
    assert read_json(directory / 'result.json')['status'] == 'success'
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    env = make_env(run, rgb=True)
    base = env.unwrapped
    resets, visible, permutations = [], [], []
    try:
        for seed in range(start_seed, start_seed+count):
            seed_everything(seed)
            env.reset(seed=seed)
            resets.append(dict(seed=seed, **snapshot(base)))
            for key in ('staged', 'left_slot', 'restored', 'was_lifted', 'succeeded',
                        'wrong_assign', 'target_returned', 'two_in_slot', 'arrangement_hold'):
                assert not as_numpy(getattr(base,key)).any(), key
        seed_everything(start_seed)
        env.reset(seed=start_seed)
        repeat = snapshot(base)
        repeat_equal = all(np.array_equal(repeat[k],resets[0][k]) for k in repeat)
        seed_everything(source_seed)
        env.reset(seed=source_seed)
        with h5py.File(directory/'trajectory.h5') as h5:
            trajectory = h5['traj_0']
            # Each newly exposed object must be observable before its first grasp.
            strokes = [e for e in events if e['message']=='shelf: grasp stroke']
            seen = set()
            for event in strokes:
                if event['prop'] in seen: continue
                seen.add(event['prop'])
                step = event['step']
                base.set_state_dict(read_state(trajectory['env_states'],step))
                base._elapsed_steps[:] = step
                present = capture(base)
                prop = next(p for p in base.props if p.name==event['prop'])
                raw = prop.pose.raw_pose.clone()
                hidden_pose = raw.clone(); hidden_pose[:,2] += 100
                prop.set_pose(Pose.create(hidden_pose))
                absent = capture(base)
                prop.set_pose(Pose.create(raw))
                pixels = {}
                for camera in CAMERAS:
                    key='observation.images.'+camera
                    pixels[camera]=int(np.any(np.abs(present[key].astype(np.int16)-
                        absent[key].astype(np.int16))>2,axis=-1).sum())
                    Image.fromarray(present[key]).save(output/f'{event["prop"]}-{step}-{camera}.png')
                visible.append(dict(prop=event['prop'],step=step,changed_pixels=pixels,
                    visible=max(pixels.values())>=20))
            cleared = next(e for e in events if e['message']=='row cleared')['step']+12
            state=read_state(trajectory['env_states'],cleared)
            base.set_state_dict(state);base._elapsed_steps[:]=cleared
            reference=capture(base)
            # Same rendered physical state, every possible remembered association.
            for values in itertools.permutations(range(base.cfg.n_props)):
                base.set_state_dict(state);base._elapsed_steps[:]=cleared
                base.original_slots[:]=torch.tensor(values,device=base.device)
                altered=capture(base)
                equal=all(np.array_equal(reference[k],altered[k]) for k in reference)
                permutations.append(dict(original_slots=list(values),policy_inputs_identical=equal))
            base.set_state_dict(state)
            roundtrip=base.get_state_dict()
            for key in TASK_STATE_KEYS:
                np.testing.assert_array_equal(as_numpy(roundtrip[key]),state[key])
            for camera in CAMERAS:
                Image.fromarray(reference['observation.images.'+camera]).save(output/f'staged-{camera}.png')
        report=dict(source_sha256=run['signature']['code_sha256'],source_seed=source_seed,
            resets=resets,repeat_equal=repeat_equal,
            unique_robot_starts=len({tuple(r['qpos'][:3]) for r in resets}),
            unique_row_positions=len({r['props'][0][0] for r in resets}),
            unique_assignments=len({tuple(r['original_slots']) for r in resets}),
            revealed_objects=visible,hidden_answer_counterfactuals=permutations,
            task_state_roundtrip=True,
            note='At least 20 changed pixels in one native camera before each extraction. '
                 'Finite observation check, not proof that a learned policy uses internal memory. '
                 'An unrestricted policy may encode order in its chosen staging layout.')
        passed=(repeat_equal and len(visible)==base.cfg.n_props and all(c['visible'] for c in visible)
            and all(c['policy_inputs_identical'] for c in permutations)
            and report['unique_robot_starts']==count and report['unique_row_positions']==count)
        report['status']='success' if passed else 'failed'
        write_json(output/'result.json',report)
        print({k:report[k] for k in ('status','repeat_equal','unique_robot_starts','unique_row_positions','unique_assignments')})
        assert passed, 'Depth Recall qualification failed; inspect result.json and frames'
    finally:
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-seed',type=int,default=300)
    parser.add_argument('--start-seed',type=int,default=300)
    parser.add_argument('--count',type=int,default=32)
    args=parser.parse_args()
    qualify(args.run.resolve(),args.output.resolve(),args.source_seed,args.start_seed,args.count)
