"""VLA readiness: load an exported dataset in LeRobot 0.4.3 and run samples through the
pi0.5 pre-processor built from the dataset's own stats (no model weights needed).

usage (LeRobot venv): vla_ready.py OUT.json DATASET_DIR [DATASET_DIR ...]
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (AddBatchDimensionProcessorStep, NormalizerProcessorStep, PolicyProcessorPipeline,
                               RenameObservationsProcessorStep, UnnormalizerProcessorStep)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
import sentencepiece as spm

# pi0.5's own steps up to the prompt; the prompt is then counted with the local
# PaliGemma sentencepiece model (TokenizerProcessorStep needs transformers + a gated download).
SP = spm.SentencePieceProcessor(model_file='/home/jovyan/shares/SR008.nfs2/users/echerepanov/interns/dnurtaev/mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model')

report = {}
for root in map(Path, sys.argv[2:]):
    info = json.loads((root / 'meta/info.json').read_text())
    ds = LeRobotDataset(repo_id=f'local/{root.name}', root=root)
    feats = dataset_to_policy_features(ds.meta.features)
    cfg = PI05Config(device='cpu')
    cfg.input_features = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    cfg.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    pre = PolicyProcessorPipeline(steps=[
        RenameObservationsProcessorStep(rename_map={}), AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(features={**cfg.input_features, **cfg.output_features},
                                norm_map=cfg.normalization_mapping, stats=ds.meta.stats),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=cfg.max_state_dim)])
    post = PolicyProcessorPipeline(steps=[UnnormalizerProcessorStep(features=cfg.output_features,
                                   norm_map=cfg.normalization_mapping, stats=ds.meta.stats)],
                                   to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    idx = sorted({0, len(ds) // 3, 2 * len(ds) // 3, len(ds) - 1})
    rows = []
    for i in idx:
        s = ds[i]
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in s.items()}
        out = pre(batch)
        prompt = out['task'][0]
        tokens = len(SP.encode(prompt, add_bos=True))
        state = out.get('observation.state')
        act = out.get('action')
        back = post(act.clone())
        roundtrip = float((back - batch['action']).abs().max())
        finite = all(torch.isfinite(v).all().item() for v in out.values() if torch.is_tensor(v) and v.is_floating_point())
        rows.append(dict(index=i, task=s.get('task'), finite=finite, prompt=prompt, prompt_tokens=tokens,
                         token_budget=cfg.tokenizer_max_length, action_roundtrip_max_abs=roundtrip,
                         state_shape=list(state.shape), action_shape=list(act.shape),
                         state_norm_range=[float(state.min()), float(state.max())],
                         action_norm_range=[float(act.min()), float(act.max())],
                         images={k: list(v.shape) for k, v in out.items() if 'image' in k and torch.is_tensor(v)}))
    stats = ds.meta.stats
    degenerate = {k: [int(j) for j in np.nonzero(np.asarray(stats[k]['q99']) - np.asarray(stats[k]['q01']) <= 1e-6)[0]]
                  for k in ('observation.state', 'action')}
    report[root.name] = dict(episodes=ds.meta.total_episodes, frames=len(ds), fps=info['fps'],
                             codebase_version=info.get('codebase_version'),
                             input_features={k: [v.type.name, list(v.shape)] for k, v in cfg.input_features.items()},
                             output_features={k: [v.type.name, list(v.shape)] for k, v in cfg.output_features.items()},
                             quantile_degenerate_dims=degenerate, samples=rows,
                             ready=all(r['finite'] and r['prompt_tokens'] <= r['token_budget'] and r['action_roundtrip_max_abs'] < 1e-4 for r in rows) and not any(degenerate.values()))
    print(root.name, 'ready', report[root.name]['ready'], 'degenerate', degenerate,
          'prompt tokens', sorted({r['prompt_tokens'] for r in rows}), 'roundtrip', max(r['action_roundtrip_max_abs'] for r in rows),
          'state', rows[0]['state_shape'], 'action', rows[0]['action_shape'], 'norm ranges', rows[0]['state_norm_range'], rows[0]['action_norm_range'],
          'images', rows[0]['images'], flush=True)
Path(sys.argv[1]).write_text(json.dumps(report, indent=1) + '\n')
