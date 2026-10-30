# Third-party training code

Git submodules, each pinned to one commit. The training recipes are the authors'; what we change
and why is in `vla/TRAINING_DEVIATIONS.md`. Our patches live under `vla/*/patches/` and are applied
by the setup scripts; the submodules stay at their pins.

| Submodule | Pin | What we use it for |
|---|---|---|
| `memer` (memer-policy/memer) | `fb6dbd0` | MemER high level: `scripts/generate_sft_data.py` (LeRobot v3 + `meta/subtasks.parquet` → Qwen SFT data; `vla/memer/patches/memer-0001` keeps early keyframes), offline rollout eval, `MemERDeploymentPolicy` |
| `Qwen3-VL` (QwenLM/Qwen3-VL) | `9658872` | `qwen-vl-finetune`, the trainer `memer/TRAINING_QWEN3VL.md` names: Qwen3-VL-4B-Instruct, vision and merger frozen, LLM tuned, 1500 steps, effective batch 128, LR 6e-5 cosine, warmup 0.05 |
| `MemoryVLA` (shihao1895/MemoryVLA, branch `mikasa-robo`) | `082ce25` | The authors' MIKASA-Robo recipe, `script/train/mikasa/train_mikasa*.sh`: OpenVLA-7B prismatic init, DiT-L, LR 2e-5, 20k steps, global batch 256, RLDS data |
| `openpi` (Physical-Intelligence/openpi) | `215abfb` | MemER low level (pi0.5 with the current subtask as the prompt), plus `vla/pi05_first_frame/openpi.patch` |

Clone with `git submodule update --init`. MemoryVLA's Llama-2 comes from a local copy of
NousResearch/Llama-2-7b-hf whose weight shards and tokenizer match meta-llama's sha256
(the copy's extra `"pad_token_id": 0` removed from `config.json`).
