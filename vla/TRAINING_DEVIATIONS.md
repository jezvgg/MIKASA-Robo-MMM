# Training: what follows the papers and what we changed

Checked 2026-10-21 against the papers and the authors' code (pins in `third_party/README.md`):
MemER (Sridhar et al., arXiv 2510.20328 v1, §3–4, App. A Tables 4–5, App. E; code `memer`
fb6dbd0 + `TRAINING_QWEN3VL.md`) and MemoryVLA (Shi et al., arXiv 2508.19236, §4 "Implementation
Details", §4.4 Mikasa-Robo; code `MemoryVLA` 082ce25, `script/train/mikasa/train_mikasa*.sh`).

## Same as the authors

| | MemER high level | MemER low level | MemoryVLA |
|---|---|---|---|
| Base model | Qwen3-VL-4B-Instruct (code recipe, owner's choice) | pi0.5 base, RoboMME's recipe (the paper's pi0.5-DROID is a DROID-robot checkpoint; below) | OpenVLA-7B prismatic (step 295000), DINOv2 + SigLIP, DiT-L |
| Trained / frozen | LLM trained; vision + merger frozen | SigLIP frozen, the rest trained (RoboMME; Table 5 trains everything) | everything trained (full fine-tune) |
| Steps, global batch | 1500, 128 | batch 128; steps: RoboMME's 80,000, not Table 5's 18,000 (below) | RoboMME's 160,000 (MIKASA script: 20,000), 256 (ours 128, see below) |
| Optimizer | AdamW, LR 6e-5, cosine, warmup 0.05, wd 0, clip 1, bf16, grad ckpt | AdamW b1 0.9 b2 0.95, wd 0 (openpi 1e-10), clip 1; RoboMME's 5e-5 constant after 10,000 warmup steps, EMA 0.999 | LR 2e-5 (their script), FSDP |
| Other | max_pixels 115200, min_pixels 50176, model_max_length 8192; export: frame_subsample 5, recent 8, keyframes 8, prediction_horizon 2 | chunk of 20, 16 executed (RoboMME's; paper 15 / 8); no state in the prompt (RoboMME); prompt = the frame's subtask | repeated diffusion 4, future window 15 (chunk 16, 8 executed: RoboMME's; their MIKASA script 3, chunk 4), image_aug, no proprio; their MIKASA group loader (16 random frames of an episode per group, mem_length 16) for now; their long-episode stream + mem_length 256 is `LOADER=stream` (not settled, below) |

Batch on our 4 x H100, the owner's rule (2026-10-21): no gradient accumulation of our own; what does
not fit, gets a smaller batch. The authors' per-GPU batch is kept, so the global batch follows the
GPU count:

| | authors | ours (4 x H100) |
|---|---|---|
| MemoryVLA | 8 GPUs x 32 = 256, no accumulation | 4 x 32 = **128** |
| MemER high level | 2 B200 x 8 x accumulation 8 = 128 (their recipe accumulates) | 4 x 4 x their accumulation 8 = 128 (per-device 4: an H100 has 80 GB, a B200 180 GB) |
| MemER low level | 128 | 128 (openpi, no accumulation; fits 4 x H100) |

## Filled where the paper says nothing

- MemER low level, LR and EMA: Table 5 gives neither. RoboMME's recipe for pi0.5 base is taken (5e-5
  constant after 10,000 warmup steps, EMA 0.999; openpi's `pi05_libero` schedule), not openpi's
  defaults (cosine 2.5e-5 -> 2.5e-6, EMA 0.99), which go with the paper's pi0.5-DROID start: the
  start is pi0.5 base, so the recipe made for it is taken whole.
- MemER: the code release has no model merging (the paper's 0.2 pre + 0.8 fine-tuned); the code
  recipe is followed, so no merging.
- MemER high level, frame rate: the paper feeds the recent frames at 2 Hz (section 4), the code
  takes every 5th frame, and the authors' released data is 15 Hz, so their code runs at 3 Hz. Our
  data is 10 Hz, so the same stride gives the paper's 2 Hz (window 3.5 s; subtask target 1.0 s ahead).
- MemER high level, chat roles: qwen-vl-finetune renders the export's `system` turn as an assistant
  turn (`qwenvl/data/data_processor.py:160`), so the system prompt is in the loss, while the
  authors' eval defaults to the `system` role (`memer_eval/contract.py:63`). Training is left as
  theirs; evaluation should compare both roles (`--system-role assistant`).

## Changed, and why

| Change | Where | Why |
|---|---|---|
| Three cameras (left base, fetch_hand, right base) | all | owner's decision 2026-10-21. Papers: MemER wrist + exterior; MemoryVLA one third-person view (its MIKASA script: + optional wrist). MemoryVLA uses the authors' own three-view code path (`--load_bi_wrist`), which no script of theirs runs |
| Three views stacked into one 192x576 image (planned; 256x768 tested) | MemER high | Qwen would shrink a larger stack to 192x576 under their max_pixels anyway (108 tokens; theirs 110 for 2 x 320x180). Their stack_frame_images pads the 128-wide hand image to 256 with black before scaling, so the hand view is 96x115 beside a black block and each base view 192x230; training, their eval and their deploy build it the same way |
| 13D Fetch actions (arm 7 joint targets, gripper, torso/head 3, base forward/yaw velocity) | all | our robot. Theirs: 7D EEF (MemoryVLA), 8D DROID joint velocity (pi0.5-DROID). MemoryVLA gets a FETCH_JOINT encoding (patch 0001) and `--action_dim 13`: its loader otherwise accepts only EEF_POS/EEF_R6 (EEF deltas), so, on the supervisor's answer (2026-10-22, option (b) of `artifacts/memoryvla-questions-for-supervisor.md`), arm, head and torso are recomputed as deltas from the same frame's recorded joints (`fetch_dataset_transform`; the measured joints are added back at inference) and, like the base velocities, are relative; the gripper stays absolute; all but the gripper are normalized. MemER low: arm, head and torso become deltas from the current joint position for the whole 15-step chunk (openpi `DeltaActions` with `DELTA_ACTION_MASK`; `AbsoluteActions` adds the state back at inference), openpi's own convention for absolute joint data (ALOHA, DROID joint-position); gripper and base velocities stay as recorded, and pi0.5 also gets the 12D state |
| Our own norm stats | MemER low, MemoryVLA | DROID's/MIKASA's statistics describe other action spaces |
| MemER low level on RoboMME's recipe for pi0.5 base (start from pi0.5 base, SigLIP frozen, 5e-5 constant after 10,000 warmup steps, EMA 0.999, 80,000 steps, chunk 20 with 16 executed, no state) instead of pi0.5-DROID, openpi's defaults, 15 / 8 and the state as text | MemER low | pi0.5-DROID is fine-tuned for a Franka with 8D joint velocities and two cameras (third slot masked); Fetch is a mobile manipulator with 13D actions and three cameras. RoboMME, not a DROID robot either, trains this recipe from pi0.5 base (`mme_vla_suite`); one recipe per model, deviations only where the robot, the data or the GPUs force them. Own norm stats per chunk length (`assets/memer-lowlevel-h20`) |
| Normalization bounds: each task's min/max instead of its 1st/99th percentile for every normalized channel (base velocities and arm/head/torso deltas; `vla/memoryvla/set_norm_bounds.py` edits q01/q99 in the authors' cached `dataset_statistics_*.json`; their code is unchanged) | MemoryVLA | their BOUNDS_Q99 clips each task to its percentiles; on our data that erases DepthRecall's turn (q01 = q99 = 0) and cuts TakeItBack's drives from 0.90 to 0.155 m/s; a 20k-step one-GPU check with their bounds indeed never turned and never drove faster than 0.17. Supervisor 2026-10-22: "do not clip the real important actions". Min/max clips nothing; it is what GR00T N1.5 uses for a mobile base and OpenVLA-OFT for joints and base (ALOHA; R1Pro in BEHAVIOR). The action contract's fixed -1/+1 for the base (the BEHAVIOR baselines' way) was tried first: at 10k debug steps it turned the model's residual noise into large base commands on still frames, so the narrower per-task scale was chosen (2026-10-23). Survey: `artifacts/base-normalization.md` |
| Until 2026-10-31, now `LOADER=stream` only (the default is their MIKASA group loader; not settled, below). Memory settings for long episodes: `--dataloader_type stream --mem_length 256` from the authors' `script/train/real_world/train_real.sh` in their MIKASA recipe (OpenVLA init and the 4-step window kept) | MemoryVLA | supervisor 2026-10-22 ("take the authors' settings for long episodes"). The paper (App. G, Tab. 9-10) ties memory length to episode length and uses 256 for its long real tasks (median 236, up to 902 frames after filtering); ours run 452-1373 frames (median per task), MIKASA's 60-90 |
| With `LOADER=stream` only (group keeps their 64,000 groups of 16 frames, ~0.9 MB each of ours, ~57 GB per rank, est.): shuffle buffer of 500 episodes per rank instead of their 64,000 (`vla/memoryvla/train.sh`, `SHUFFLE`) | MemoryVLA | the stream dataloader shuffles whole episodes (their `make_interleaved_episodic_dataset` leaves out the frame flatten, and images are decoded after the shuffle), so the number counts encoded episodes, ~50 MB each of ours (RLDS: 6.0 GB for 120 episodes). Their 64,000 comes from the group-dataloader MIKASA script and would mean ~3 TB per rank: on 2026-10-27 the four ranks filled ~640 GB in 90 s and the pod was killed at its 728 GiB limit. Their stream script (`train_real.sh`) uses 32,000, with the comment to lower it if memory is limited; 500 is ~25 GB per rank and covers 8% of our 6000 episodes on each rank |
| Dataloader workers per GPU: 2 instead of their 8 (`vla/memer/train_high.sh`, `WORKERS`; owner's choice 2026-10-29) | MemER high | each worker of their Qwen loader turns its copy-on-write view of the parsed `train.json` into a private copy within ~10 min (Python's reference counts write to every page): ~21 GiB per worker for four tasks, plateaus 260 / 345 / 429 GiB at 2 / 3 / 4 workers (`vla/memer/probe_loader_mem.sh`); their 8 killed the pod on 2026-10-27. The sampler runs in the main process, so the samples and their order do not depend on it, only the loading speed does. Six tasks of the every-frame export need fewer: the fit 7 + 4·(1 + W)·D GiB with D = 29.6–32.7 GiB per copy gives 362–399 GiB at 2 workers, 481–530 at 3, 599–661 at 4 (the guard stops at 582; the longer new SeasonDish adds ~3%). The price of 2: a step of 6.2 s instead of 5.4 s at 4 (probe of 2026-10-27, four tasks) |
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

## Training length on our data (settled 2026-10-31)

The step counts were set for the authors' datasets. Ours have 5,141,099 frames (6000 episodes,
~86 s each; 4,844,230 before CabinetSearch became five compartments), so the same steps see each
frame far fewer times:

| | samples (steps x batch) | our epochs | authors' epochs (estimate) |
|---|---|---|---|
| MemER high (code) | 1500 x 128 = 192k | 0.04 | ~6 (50 dusting episodes, ~31k frames) |
| MemER low (RoboMME's steps) | 80,000 x 128 = 10.2M (Table 5's 18,000: 2.3M) | 1.99 (0.45) | ~10-20 (3 tasks x ~62 episodes) |
| MemoryVLA (RoboMME's steps) | 160,000 x 128 = 20.5M (MIKASA's 20,000 x 256: 5.1M) | 3.98 (1.00) | ~50 (5 tasks x 250 short episodes) |

Matching the authors' epochs would take 10-150x their steps. MemER low level and MemoryVLA take
RoboMME's steps instead of the authors'. MemER low level: RoboMME's 80,000 steps (`mme_vla_suite`,
batch 64 there, so twice their samples here); ~1.5 s/step on 4 x H100 with chunk 20 (est.: 1.50 with
chunk 15, 1.68 with 50), ~33 h. MemoryVLA: RoboMME's 160,000 steps (their script, 2 x 32 = 64, so
twice their samples here); the LR stays constant 2e-5 with no warmup (the diffusion VLA type
inherits `lr_scheduler_type = "constant"`), so only `train.sh`'s <max steps> changes. Their code
keeps every 32 GB checkpoint: RoboMME's save interval 10,000 means 512 GB, over our 400 GB, so
checkpoints are evaluated and deleted as the run goes. Not timed on 4 GPUs yet. MemER high level:
the code's 1500 steps to start; whether to go on is decided on oracle episodes recorded on the
validation seeds.

## MemER low level: RoboMME's recipe, no state (settled 2026-10-31)

RoboMME's recipe is taken whole (above). The batch stays 128 (4 x H100 x 32) against RoboMME's 64 (4
A40 x 16): an H100 has 80 GB, an A40 48 GB, so it holds twice their per-device batch. No state in
the prompt, as RoboMME (Table 6: "Proprioception False", no reason given; their actions are absolute
joints, so it is not about deltas): pi0.5 has no state input of its own, only the text
(`discrete_state_input`), and the deltas still take the state outside the model. Beyond the one-
source rule, with smooth oracle demonstrations a state-fed policy can continue the motion from the
numbers instead of looking at the cameras. The chunk of 20 with 16 executed is 1.6 s at our 10 Hz
without a new subtask from the high level, against the authors' 0.53 s.

## MemoryVLA's chunk: 16 with 8 executed, as RoboMME (settled 2026-10-31)

`train.sh` trains a chunk of 16 (`CHUNK`, i.e. `--future_action_window_size 15`, which is also
`train.py`'s default and the paper's "Action chunking size 16", App. C.1, Table 8), and RoboMME
executes 8 of them per call (`script/eval/robomme/server.sh`). Their MIKASA script uses 3 (chunk 4);
the debug runs so far used 4. At our 10 Hz a chunk of 16 is 1.6 s, 8 executed 0.8 s. Past an
episode's end the authors' chunking fills the chunk with neutral actions (zero deltas and base
velocities, the gripper repeated; `chunk_act_obs` in `vla/datasets/rlds/traj_transforms.py`), and
`action_masks` reaches the model but not the loss, so the last 15 frames of every episode also train
on these 'hold still' tails, as the authors' own chunk does.

Memory cadence, found 2026-10-28: memory is written once per `predict_action` call at inference
(`vla/memory_vla.py:828-836`), i.e. every 8 frames here. The group loader trains on 16 random frames
of an episode in time order (`vla/datasets/rlds/dataset.py:716-736`; the bank is cleared per group),
about one entry per T/16 frames: ~56 for a 900-frame episode, so inference writes about 7x more
often than training (RoboMME, ~481-step episodes: ~4x). Stream would write every frame in training,
8x more often than inference. muVLA (arXiv 2606.12497, sections 4-5) reports a mismatch of tens of
times breaking memory (LIBERO-Long 95.8 -> 5.4); ours is a few times.

## MemoryVLA's loader: group, as the authors (settled 2026-10-31)

`train.sh` defaults to the authors' MIKASA group loader since 2026-10-31 (`LOADER=group`: group 16,
mem_length 16, shuffle 64,000 groups), as RoboMME trains MemoryVLA; `LOADER=stream` gives the stream
+ mem_length 256 of their `train_real.sh`, chosen on 2026-10-22 for our long episodes when the chunk
was 4. With 8 executed per call neither loader matches inference's write cadence (above); group is
chosen by the one-source rule. It does not train short-term memory (the last seconds), it has not
been run on our data, and its buffer is an estimate.

The one precedent with long episodes is RoboMME, and it is a poor result. RoboMME trained MemoryVLA
with group (their LIBERO settings: chunk 16, memory 16; `script/train/robomme/train.sh`) and
executed 8 of 16 per call (`script/eval/robomme/server.sh`). Their episodes average 481 steps; the
longest tasks are VideoPlaceOrder (1134), VideoPlaceButton (974) and VideoRepick (687) (Table 1).
MemoryVLA scores "below 10%" overall (App. C.4); the mean of its 16 tasks in Table 15 is 7.1% (our
count), under pi0.5 without memory (17.93). Long tasks: VideoRepick 0.0 (pi0.5 0.4),
VideoPlaceButton 13.3 (31.1), VideoPlaceOrder 4.0 (25.8); short ones are low too (PatternLock, 208
steps: 12.0; VideoUnmask, 217: 16.0), so the failure does not follow episode length. The paper gives
no reason. Our guess is the backbone: OpenVLA-OFT with their memory also stays near 20%, "less
effective than the pi0.5 backbone in our setting" (C.4), and MemoryVLA is built on the same
Prismatic/OpenVLA. So there is no evidence that group works on long episodes, and none that it is
what fails.

## Not settled: MemoryVLA's memory time scale

An entry's time is the frame index in training and the call count at inference (`cur_timestep +=
1`), so executing k actions per call compresses time k times: 4x in the authors' MIKASA eval, 8x in
RoboMME's and here. The authors' code is kept as is; a deploy adapter could add k - 1 to
`cur_timestep` after each call without editing it, which would be a deviation from them.

## Evaluation protocol (not training, for later)

MemoryVLA reports the best validation step (validation every 1k steps); its checkpoints are 32 GB
each, so they are evaluated and pruned as the run goes. Inference: DDIM 10 steps, CFG 1.5. MemER:
high level ~1 Hz, low level executes 8 of 15 actions; their deploy wrapper uses merge_distance 5.
jezv's tasks reset with `reconfigure=True`.
