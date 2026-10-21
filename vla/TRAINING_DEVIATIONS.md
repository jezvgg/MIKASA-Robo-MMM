# Training: what follows the papers and what we changed

Checked 2026-10-21 against the papers and the authors' code (pins in `third_party/README.md`):
MemER (Sridhar et al., arXiv 2510.20328 v1, §3–4, App. A Tables 4–5, App. E; code `memer`
fb6dbd0 + `TRAINING_QWEN3VL.md`) and MemoryVLA (Shi et al., arXiv 2508.19236, §4 "Implementation
Details", §4.4 Mikasa-Robo; code `MemoryVLA` 082ce25, `script/train/mikasa/train_mikasa*.sh`).

## Same as the authors

| | MemER high level | MemER low level | MemoryVLA |
|---|---|---|---|
| Base model | Qwen3-VL-4B-Instruct (code recipe, owner's choice) | pi0.5-DROID | OpenVLA-7B prismatic (step 295000), DINOv2 + SigLIP, DiT-L |
| Trained / frozen | LLM trained; vision + merger frozen | everything trained | everything trained (full fine-tune) |
| Steps, global batch | 1500, 128 | 18,000, 128 | 20,000, 256 |
| Optimizer | AdamW, LR 6e-5, cosine, warmup 0.05, wd 0, clip 1, bf16, grad ckpt | AdamW b1 0.9 b2 0.95, wd 0 (openpi 1e-10), clip 1, cosine, warmup 0.033 | LR 2e-5 (their script), FSDP |
| Other | max_pixels 115200, min_pixels 50176, model_max_length 8192; export: frame_subsample 5, recent 8, keyframes 8, prediction_horizon 2 | chunk of 15 actions; prompt = the frame's subtask | repeated diffusion 4, future window 3 (chunk 4), image_aug, group dataloader, shuffle buffer 64k, mem_length 16, no proprio |

A batch the hardware cannot hold is reached by accumulation, never shrunk: MemoryVLA 4 GPUs x 32 x
accumulation 2 = 256 (theirs 8 x 32); MemER high level per-device x accumulation x GPUs = 128.

## Filled where the paper says nothing

- MemER low level, peak LR and EMA: Table 5 gives neither; openpi's defaults (2.5e-5 -> 2.5e-6, EMA
  0.99), as their `pi05_droid_finetune` leaves them.
- MemER: the code release has no model merging (the paper's 0.2 pre + 0.8 fine-tuned); the code
  recipe is followed, so no merging.

## Changed, and why

| Change | Where | Why |
|---|---|---|
| Three cameras (left base, fetch_hand, right base) | all | owner's decision 2026-10-21. Papers: MemER wrist + exterior; MemoryVLA one third-person view (its MIKASA script: + optional wrist). MemoryVLA uses the authors' own three-view code path (`--load_bi_wrist`), which no script of theirs runs |
| Stacked views 192x192 (planned; 256x256 tested) | MemER high | Qwen resizes the three-view stack to 192x576 under their max_pixels anyway; theirs: 2 x 320x180 |
| 13D Fetch actions (arm 7 joint targets, gripper, torso/head 3, base forward/yaw velocity) | all | our robot. Theirs: 7D EEF (MemoryVLA), 8D DROID joint velocity (pi0.5-DROID). MemoryVLA gets a FETCH_JOINT encoding (patch 0001: joint/body absolute, base velocities relative, all but the gripper normalized) and `--action_dim 13` |
| Our own norm stats | MemER low, MemoryVLA | DROID's/MIKASA's statistics describe other action spaces |
| Subtask boundaries from the oracle's recorded events; subtask names and keyframe rules written by us | MemER | theirs: operator key presses and their own per-task rules (App. E); every task needs its own. Approved list: ~/Downloads/memer_subtasks_2026-10-20.txt |
| Data: 6 tasks x 1000 oracle episodes, no interventions | all | our benchmark. MemER: 3 tasks x 50 demos (+10-15 interventions for the low level); MemoryVLA on MIKASA-Robo: 5 tasks x 250 demos at 128x128 |
| Data formats: RLDS built from our LeRobot; MemER export from LeRobot | MemoryVLA, MemER high | format only; their loaders read them |
| Llama-2 from the NousResearch copy | MemoryVLA | weights sha256-identical to meta-llama's |
| Software pins (tensorflow-metadata 1.14, wandb 0.17.9, setuptools), rendezvous on 127.0.0.1 | MemoryVLA, MemER high | their unpinned dependencies no longer import together; lab container |

## Not settled: training length on our data

The step counts were set for the authors' datasets. Ours have 4,844,230 frames (6000 episodes,
~80 s each), so the same steps see each frame far fewer times:

| | samples (steps x batch) | our epochs | authors' epochs (estimate) |
|---|---|---|---|
| MemER high (code) | 1500 x 128 = 192k | 0.04 | ~6 (50 dusting episodes, ~31k frames) |
| MemER low (Table 5) | 18,000 x 128 = 2.3M | 0.48 | ~10-20 (3 tasks x ~62 episodes) |
| MemoryVLA (MIKASA) | 20,000 x 256 = 5.1M | 1.06 | ~50 (5 tasks x 250 short episodes) |

Matching the authors' epochs would take 10-150x their steps. The owner decides between the
literal steps, more steps, or fewer episodes per task.

## Evaluation protocol (not training, for later)

MemoryVLA reports the best validation step (validation every 1k steps); its checkpoints are 32 GB
each, so they are evaluated and pruned as the run goes. Inference: DDIM 10 steps, CFG 1.5. MemER:
high level ~1 Hz, low level executes 8 of 15 actions; their deploy wrapper uses merge_distance 5.
jezv's tasks reset with `reconfigure=True`.
