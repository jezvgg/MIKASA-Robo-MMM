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
| Steps, global batch | 1500, 128 | 18,000, 128 | 20,000, 256 (ours 128, see below) |
| Optimizer | AdamW, LR 6e-5, cosine, warmup 0.05, wd 0, clip 1, bf16, grad ckpt | AdamW b1 0.9 b2 0.95, wd 0 (openpi 1e-10), clip 1, cosine, warmup 0.033 | LR 2e-5 (their script), FSDP |
| Other | max_pixels 115200, min_pixels 50176, model_max_length 8192; export: frame_subsample 5, recent 8, keyframes 8, prediction_horizon 2 | chunk of 15 actions; prompt = the frame's subtask | repeated diffusion 4, future window 3 (chunk 4), image_aug, group dataloader, shuffle buffer 64k, mem_length 16, no proprio |

Batch on our 4 x H100, the owner's rule (2026-10-21): no gradient accumulation of our own; what does
not fit, gets a smaller batch. The authors' per-GPU batch is kept, so the global batch follows the
GPU count:

| | authors | ours (4 x H100) |
|---|---|---|
| MemoryVLA | 8 GPUs x 32 = 256, no accumulation | 4 x 32 = **128** |
| MemER high level | 2 B200 x 8 x accumulation 8 = 128 (their recipe accumulates) | 4 x 4 x their accumulation 8 = 128 (per-device 4: an H100 has 80 GB, a B200 180 GB) |
| MemER low level | 128 | 128 (openpi, no accumulation; fits 4 x H100) |

## Filled where the paper says nothing

- MemER low level, peak LR and EMA: Table 5 gives neither; openpi's defaults (2.5e-5 -> 2.5e-6, EMA
  0.99), as their `pi05_droid_finetune` leaves them.
- MemER: the code release has no model merging (the paper's 0.2 pre + 0.8 fine-tuned); the code
  recipe is followed, so no merging.

## Changed, and why

| Change | Where | Why |
|---|---|---|
| Three cameras (left base, fetch_hand, right base) | all | owner's decision 2026-10-21. Papers: MemER wrist + exterior; MemoryVLA one third-person view (its MIKASA script: + optional wrist). MemoryVLA uses the authors' own three-view code path (`--load_bi_wrist`), which no script of theirs runs |
| Three views stacked into one 192x576 image (planned; 256x768 tested) | MemER high | Qwen would shrink a larger stack to 192x576 under their max_pixels anyway (108 tokens; theirs 110 for 2 x 320x180). Their stack_frame_images pads the 128-wide hand image to 256 with black before scaling, so the hand view is 96x115 beside a black block and each base view 192x230; training, their eval and their deploy build it the same way |
| 13D Fetch actions (arm 7 joint targets, gripper, torso/head 3, base forward/yaw velocity) | all | our robot. Theirs: 7D EEF (MemoryVLA), 8D DROID joint velocity (pi0.5-DROID). MemoryVLA gets a FETCH_JOINT encoding (patch 0001: joint/body absolute, base velocities relative, all but the gripper normalized) and `--action_dim 13`; its loader otherwise accepts only EEF_POS/EEF_R6 (EEF deltas), so absolute joints have no precedent there (open question, `artifacts/memoryvla-questions-for-supervisor.md`). MemER low: arm, head and torso become deltas from the current joint position for the whole 15-step chunk (openpi `DeltaActions` with `DELTA_ACTION_MASK`; `AbsoluteActions` adds the state back at inference), openpi's own convention for absolute joint data (ALOHA, DROID joint-position); gripper and base velocities stay as recorded, and pi0.5 also gets the 12D state |
| Our own norm stats | MemER low, MemoryVLA | DROID's/MIKASA's statistics describe other action spaces |
| Subtask boundaries from the oracle's recorded events; subtask names and keyframe rules written by us | MemER | theirs: operator key presses and their own per-task rules (App. E); every task needs its own. Approved list: ~/Downloads/memer_subtasks_2026-10-20.txt |
| A keyframe before a sample's first subsampled frame moves to that frame instead of being dropped (`vla/memer/patches/memer-0001-early-keyframes.patch`, one line in their `generate_sft_data.py`) | MemER high | owner 2026-10-22; without it SameDrawer's cue reaches 20% of the samples that need it. See "Early keyframes" below |
| Data: 6 tasks x 1000 oracle episodes, no interventions | all | our benchmark. MemER: 3 tasks x 50 demos (+10-15 interventions for the low level); MemoryVLA on MIKASA-Robo: 5 tasks x 250 demos at 128x128 |
| Data formats: RLDS built from our LeRobot; MemER export from LeRobot | MemoryVLA, MemER high | format only; their loaders read them |
| Llama-2 from the NousResearch copy | MemoryVLA | weights sha256-identical to meta-llama's |
| Software pins (tensorflow-metadata 1.14, wandb 0.17.9, setuptools), rendezvous on 127.0.0.1 | MemoryVLA, MemER high | their unpinned dependencies no longer import together; lab container |

## Early keyframes in the MemER high-level export (owner's decision, 2026-10-22)

**What the authors' code does.** `scripts/generate_sft_data.py` makes one SFT sample per timestep
t. The sample sees the frames t, t-5, t-10, ... (frame_subsample 5), a grid that starts at frame
t mod 5. Its memory holds the rule keyframes aligned to that grid by `align_frames_with_subsampling`
(`memer_eval/utils.py`), which keeps a keyframe k only if k >= t mod 5. So a keyframe on frames 0-3
reaches only the samples whose grid starts at or before it: a keyframe on frame 0 reaches one
sample in five.

**What it did to our data.** Our keyframe rules are of the authors' kind (first or last frame of a
named subtask; approved list above). In two tasks the cue that must be remembered is visible only
at the very start of the episode:
- SameDrawer: the open drawer. The rule is the first frame of "close the X drawer" = frame 0.
  The drawer was in memory for 20% of the "open the X drawer" samples (365 of 1823 in the
  20-episode debug export; 72,072 of 90,117 missing over all 1000 episodes,
  `artifacts/data-audit/memer-high.md`). All 160 keyframe targets fell on samples with
  t mod 5 = 0. The 400-step debug model never selected a keyframe; its 0.886 on held-out
  episode 990 came from naming the most frequent drawer.
- Veggie: the picture on the fridge. The rule is the last frame of "look at the picture on the
  fridge" = frame 2, so the picture was missing from 40% of the samples that need it.

The authors never meet this: their rules (paper App. E, `configs/tasks_from_paper`) take the last
frame of a subtask or nothing, and none of their cues sits in the first frames of an episode.

**The change.** In `build_qwen_example` a keyframe k < t mod 5 moves to frame t mod 5, the
sample's first frame, instead of being dropped. One line, applied by `vla/memer/setup_memer.sh`.
`align_frames_with_subsampling`, the authors' eval and their deploy code are untouched, and
keyframes on frame 4 or later are not affected. In the first 0.4 s the robot has barely moved: the
exported images of frames 0 and 4 show the same open drawer (checked on SameDrawer episodes 0, 7
and 13). At deployment the model picks keyframes from its own recent window, which at the first
call contains frame 0, so the cue is available there anyway; the change gives training the same.

**Effect** (20-episode debug exports, before -> after):

| Task | samples that need the cue with it in memory or in the recent window | keyframe targets |
|---|---|---|
| SameDrawer ("open the X drawer") | 365 / 1823 (20%) -> 1823 / 1823 (100%) | 160 -> 800 of 11,248 |
| Veggie ("pick up the X", "put the X on the plate") | 11,191 / 18,701 (60%) -> 18,661 / 18,701 (99.8%; the rest are frames 0-1, before the keyframe) | 440 -> 760 of 18,701 |

DepthRecall and TakeItBack change too: their first keyframe (first frame of "take the ... to the
... pad", "pick up the cup") is frame 0, which now reaches all of their samples instead of one in
five (22,008 of 27,520 and 8,626 of 10,788 samples changed). SeasonDish and CabinetSearch exports
are byte-identical.

**Alternatives we did not take.**
- Split "close the X drawer" at the first contact so the keyframe lands later: it changes the
  approved subtask list, the contact time is not in the dataset, and Veggie has no such fix (the
  picture is in full view only in the first ~6 frames).
- Export only samples with t mod 5 = 0: it fixes the coverage but drops 80% of the authors'
  samples per episode, a larger deviation.
- Leave it: the high level then cannot learn to remember the drawer in SameDrawer.

## Not settled: training length on our data

The step counts were set for the authors' datasets. Ours have 4,844,230 frames (6000 episodes,
~80 s each), so the same steps see each frame far fewer times:

| | samples (steps x batch) | our epochs | authors' epochs (estimate) |
|---|---|---|---|
| MemER high (code) | 1500 x 128 = 192k | 0.04 | ~6 (50 dusting episodes, ~31k frames) |
| MemER low (Table 5) | 18,000 x 128 = 2.3M | 0.48 | ~10-20 (3 tasks x ~62 episodes) |
| MemoryVLA (MIKASA) | 20,000 x 256 = 5.1M (ours 128: 2.6M) | 1.06 (ours 0.53) | ~50 (5 tasks x 250 short episodes) |

Matching the authors' epochs would take 10-150x their steps. The owner decides between the
literal steps, more steps, or fewer episodes per task.

## Evaluation protocol (not training, for later)

MemoryVLA reports the best validation step (validation every 1k steps); its checkpoints are 32 GB
each, so they are evaluated and pruned as the run goes. Inference: DDIM 10 steps, CFG 1.5. MemER:
high level ~1 Hz, low level executes 8 of 15 actions; their deploy wrapper uses merge_distance 5.
jezv's tasks reset with `reconfigure=True`.
