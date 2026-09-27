# Season Dish collection

This profile prepares `MikasaSeasonDish-v0` in kitchen 0 for the shared
[BIBLE](../../BIBLE.md) and [41-item checklist](../../DATASET_CHECKLIST.md). It uses the unchanged DSFetch
from upstream `master` at `4c5c8b3de941565ee2d57b153a10254fe64c2849`.
The branch currently depends on `feat/cabinet-search-collection`, which supplies
the shared H5/replay/LeRobot implementation. No replacement robot is introduced.

## Task and information

Two short salt/pepper models and a bowl are randomly positioned on the counter.
The robot starts by the refrigerator. A picture identifies the requested
condiment for 100 control steps (5 seconds); the robot waits until it disappears
before driving. Condiment sides are randomized independently of the answer.
Language does not disclose which condiment is requested.

The condiment must be held within 0.10 m horizontally of the bowl, 0.05–0.30 m
above it and tilted at least 155 degrees for 15 consecutive control steps, after
the cue and delay. The unchanged horizon is 1100 steps (55 seconds). The other
condiment must not be grasped or displaced more than 0.10 m. This is a pose proxy
for seasoning; no fluid/substance simulation is implied.

Profile v7 retains a horizontal side grasp with the wrist camera above the hand.
There is no empty-arm fold before driving or additional arrival preparation pose.
After grasp/lift the arm folds to the earlier fixed compact condiment carry pose.
The original canonical keyframe itself is unchanged. The hover is
prepared before pouring. Pouring changes only `wrist_roll`: the other arm/torso
commands stay constant and both base commands remain zero. The complete wrist arc
is checked for collisions, available roll range and a reachable 165-degree target
while the object is held. An infeasible arc is a failure, without an arm-motion
fallback or relaxed success threshold. The head follows the current stage.

The owner explicitly approved a D6 exception only for the initial 100 control
steps of picture observation. All frames remain in the recordings and exports.
The checklist reports raw pause metrics and a separate post-cue check; a later
pause is not covered by this exception.

## Motion and noise

The existing expert uses geometric grasps, IK/screw paths, collision-checked
joint lines and bounded monotone joint curves. Grasp selection prechecks the
whole approach, horizontal contact, vertical lift and fixed carry continuation.
The same positive elbow branch is kept through those stages. There is no RRT
lift, top-down rescue or simultaneous base/arm grasp fallback. RL is not used. The scene adds RoboCasa's kitchen
exclusions only to wheels/base; arm and fingers retain physical kitchen contacts.
DSFetch class, URDF/SRDF, controllers, cameras and meshes are unchanged.

An independent RNG (`scene_seed + 200003`) adds uniform positional noise of up
to 5 mm per enabled world axis to these goals:

| Goal | Perturbed coordinates |
|---|---|
| Initial station dock after the cue | x, y |
| Free approach before grasp | x, y, z |
| Connected vertical lift | Chosen from collision-checked heights; no independent noise |
| Base stop at the bowl dock | x, y |
| Feasible hover and hover correction | x, y, z |

Contact grasps and orientations remain geometry-derived. Each goal is perturbed
before the existing collision/IK checks. A candidate's feasibility probe and
execution reuse the same sampled pose. `events.jsonl` records label, sample index,
original goal, offset and resulting goal. Counts include proposed alternatives;
an unexecuted candidate is not an extra movement. The number of draws varies
with retries and recovery stages, rather than being fixed per seed.

## Run

Use a simulator environment with the versions in `season_dish_profile.json`,
RoboCasa assets (`MS_ASSET_DIR`), and a working NVIDIA Vulkan ICD for RGB
(`VK_ICD_FILENAMES` if discovery needs it). Headless RGB requires no desktop.
For the preflight PaliGemma check install `sentencepiece==0.2.1` in that environment
with `uv pip install --python /path/to/sim-env/bin/python sentencepiece==0.2.1`.
Run commands through `uv run --no-project --python /path/to/sim-env/bin/python` so
the legacy editable engine in the repository is not selected accidentally.

```bash
python -m utils.collection.campaign --profile season_dish \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --output /path/to/development --start-seed 1100 --num-seeds 100 \
  --purpose development --jobs 4 --through validated
python -m utils.collection.campaign --profile season_dish \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --output /path/to/train --start-seed 10000 --num-seeds 24 \
  --purpose train --jobs 4 --through rgb --skip-native-rgb
```

The full instruction is tokenized before collection and again at export, with
BOS and a trailing newline, without truncation. Run metadata records the tokenizer
hash and counts. Resume with the same pool/profile/purpose; a changed implementation
requires a new run directory. All attempt outcomes remain in `attempts.json`.

State-only H5 is recorded at 20 Hz. Native action replay and a fresh 10 Hz physical
replay must succeed before rendering training RGB. The 10 Hz variant holds the first
absolute action of each pair for two 20 Hz steps; an odd final step is padded.
Success is measured again, not inferred from the original demonstration.
`--skip-native-rgb` omits only the redundant native-action RGB copy, after that
path has been qualified on development data. The final RGB H5 remains at 20 Hz.

Export with the separate environment from `requirements-lerobot.txt`:

```bash
python -m utils.collection.export_lerobot --input /path/to/train \
  --output /path/to/lerobot --repo-id mikasa-local/season-dish \
  --tokenizer /path/to/paligemma_tokenizer.model
```

The writer/readback use actual LeRobot v3 at 10 Hz. Actions are 13D absolute
arm/head/torso targets plus normalized gripper and base velocity channels.
Exactly three native RGB streams are kept: two 256x256 head cameras (FOV 1.5)
and the 128x128 wrist camera (FOV 2.0). No depth or extra scene camera enters VLA.
Proprio is exactly `qpos[3:]` (12D); `qpos[:3]` is separate debug-only `global_state`.
The articulation root pose is also retained in raw H5: it varies with the initial
dock, so initial qpos alone is not the world position of the robot in this scene.
The policy client forwards only the three RGBs, proprio and task text.

`source_h5_metadata.json` preserves source/episode mapping, seeds, durations,
rewards, final `success`, `success_once`, all attempted seeds, robot/runtime/source
versions, camera and action semantics. An error without a complete recording has
unknown summary values (`null`), not fabricated successful/failed physical flags.

For validation collect disjoint candidate pools with `--purpose validation`.
Then `python -m utils.collection.validation --runs ... --exclude-runs ... --output ...`
freezes the first 100 planner-success seeds in ascending order. Exclude every
assigned train/development seed, including unsuccessful attempts and diagnostic
seeds supplied through `--exclude-seeds`. Ten-Hz replay results are reported
separately and do not replace planner-success validation seeds. Never replace a
fixed seed because the evaluated policy failed it.

## Repeat the qualification

Use disjoint development seeds for qualification, and record them among the
validation exclusions. These commands use the simulator environment above:

```bash
python -m pytest utils/collection/test_contract.py -q
python -m utils.collection.qualify_season --run /path/to/development \
  --output /path/to/cue-check --start-seed 300 --count 32
python -m utils.collection.qualify_head --profile season_dish \
  --output /path/to/head-check.json
python -m utils.collection.audit check --root /path/to/train \
  --output /path/to/train-contract-check.json
python -m utils.collection.audit collection --root /path/to/train \
  --output /path/to/train-statistics.json
```

The cue check compares both possible answers at the same physical state. It
checks marker visibility during the cue and identical RGB/proprio/text after
removing it. This is a finite check of the available observations, not evidence
that a learned policy uses memory. The H5 check covers every completed recording,
including physical failures, and reproduces logged noise draws from their seed.

## Distant-start qualification (profile v2)

Measured on kitchen 0, CPU physics, source SHA256
`d40c57a61ff5eb90f7f0231079037bb6c0a892596db1f8917de8fe8f0ac63a02`:

| Fixed pool | Source planner | Native 20 Hz | Paired 10 Hz | RGB |
|---|---|---|---|---|
| Development, 60000–60099 | 98/100 | Not run | Not run | Separate video |
| Training, 11000–11007 | 8/8 | 8/8 | 8/8 | 8 native + 8 paired |
| Validation candidates, 20000–20127 | 127/128 | Not run | Not run | Not required for selection |

The two pilot failures are a pour miss (60004) and horizon exhaustion (60064).
Every pilot attempt physically travelled 0.693–0.799 m after cue step 40;
grasping started at steps 98–104, a 2.9–3.2 s interval after the cue vanished.
This is measured base motion in H5, not just the requested dock distance.

All 64 answer counterfactuals over 32 starts passed: each head camera showed
31–41 changed yellow pixels during the cue; RGB/proprio/text were identical after
hiding it. The check now requires at least 20 changed yellow pixels in *each*
head camera. All eight accepted training RGB episodes also passed that visibility
criterion and kept the marker hidden after step 40. The wrist is not required to
see the initial cue. Head tilt changes from 0.20 at the distant start to 0.45 at
the manipulation dock; both gazes target the station midpoint, not the answer.

LeRobot v3 contains 8 episodes / 2,183 frames at 10 Hz. Readback checked every
numeric sample and 120 decoded RGB samples. All 39 native H5 arrays matched
exactly in all eight episodes. All 268 H5 recordings (100 pilot, 128 validation,
40 training phases), including failures, passed the action/state/metadata/noise
audit. Seventeen contract tests passed. A recorded-action policy exercised the
actual RGB/12D interface successfully for 235 requests / 470 control steps with
no clipped channels; this is not a learned-policy result. The 40 DSFetch model
files remain byte-identical to the pinned master reference.

The new [100-seed v2 validation list](validation_seeds/season_dish_v2.json) is
selected by source-planner success, excludes 270 assigned training/development/
diagnostic seeds, and records the candidate outcomes and provenance hashes.
Only candidate 20026 failed; the last selected seed is 20100. Validation native
and 10 Hz replays were not run, and are not selection conditions. The v1 list
below is retained as a historical qualification of the earlier start geometry.

## Historical near-station qualification (profile v1)

Measured on kitchen 0 with CPU physics, DSFetch from master `4c5c8b3`, and
simulation source SHA256
`769942c0a689f4c12d5231fe189ee1f92d2dff04215d8c4c9ddee410f911f5a1`:

| Fixed pool | Source planner | Native 20 Hz replay | Paired-action 10 Hz replay | Training RGB |
|---|---|---|---|---|
| Development pilot, seeds 1100–1199 | 100/100 | 100/100 | 97/100 | Qualification only |
| Training, seeds 10000–10023 | 24/24 | 24/24 | 23/24 | 23 episodes |

The 23 accepted training episodes export to genuine LeRobot v3: 5,795 frames
at 10 Hz. Every numeric sample and episode summary was compared with H5;
345 decoded RGB samples were checked (MAE 0.347–4.292 on the 0–255 scale).
The cue is visible in all 23 source and decoded training episodes and is hidden
in every saved state after the cue period. One training replay ended with
14 valid hold steps instead of 15 and was correctly excluded.

Seventeen regression tests passed. All 95 training and 302 pilot H5 recordings,
including unsuccessful replays, passed metadata/action/state/noise audits.
Native replay reproduced all saved state arrays exactly in the 24 training cases.
32 reset cases produced 32 distinct robot starts and object layouts; all 64
counterfactual cue cases were visible and had identical policy inputs after
hiding the cue. A recorded-action policy exercised the actual 12D/RGB client
for 224 requests successfully. The full instruction takes 37 PaliGemma tokens.
All 40 robot files were compared byte-for-byte with the master reference.

The 120 validation candidates (20000–20119) produced 118 planner successes,
118 successful native replays and 114 successful 10 Hz replays. The immutable
[100-seed validation list](validation_seeds/season_dish_v1.json) excludes all
162 assigned training, development and diagnostic seeds. Selection uses source
planner success only: all four 10 Hz replay failures in the selected set remain
in the list. The manifest records the pinned implementation and pool outcomes.

These measurements qualify collection on this task configuration. No learned
VLA performance or coverage of all kitchens is claimed.

## Restoration provenance

Profile v5 follows the owner's selected changes 1, 2, 4, 8, 9 and common 43.
The movement reference is published commit `dbb95ed`. Old profile v1/v2/v3
recordings and their qualification below retain their original provenance.
New source/native/held-action/RGB runs and human E5 review qualify v5 separately.

The first restoration pilot (v4, commit `ab956b5`, fixed seeds 7700000–7700039)
produced 7/40 source successes; all seven passed native and held-action replay.
It does not meet the planner success-rate requirement. Its records are retained.
Version 5 reserves canonical rest values inside the planning roll limits, selects
a hover only after checking its full wrist-only pour, and approaches left-wall
docks diagonally from the open aisle with the same rest arm. This avoids driving
a west-facing forearm into the wall without adding a different transport pose.
These changes require their own fixed-pool qualification and E5 review.

## Restored profile v5 pilot (2026-09-26)

Fixed pool `season-restore-pilot-002`: 30/40 source successes, all 30
native replays, held-action replays and three-camera RGB recordings pass.
No failed seed was replaced. Roll-range and individual-turn checks pass for all
successful episodes. New E5 packages contain ten complete three-camera episodes
and storyboards for the remaining successes; human acceptance is pending.
D6 is not fully qualified: 7 successful episodes exceed the pause
fraction limit; no additional exception is assumed. The authorized initial
five-second cue exception is evaluated separately; post-cue failures remain failures.
Every successful pour has constant commands for all non-wrist arm joints/torso
and zero base commands. Segmentation visibility exceeds 95% overall in every
episode, but brief manipulation occlusions remain for E1 review.

## Follow-up motion guards (2026-09-27)

The compact carry targets are torso .38 m; shoulder pan -.37, shoulder lift -.8,
upper-arm roll 1.7, elbow flex 2.1, forearm roll -.6, wrist flex .5 and wrist roll
-.4 rad. They are assigned by joint name and reserved in the roll budget before
grasp selection. The published planner normally drove from its lifted pose and
used a Cartesian carry only as recovery; it did not define one fixed carry target.

The entire timed approach/lift/carry path is checked between knots against all
fixtures. At lift start only, an explicitly named payload/counter contact may
remain from the object's support; every subsequent sample must be collision free.
Robot/upper-handle contacts never receive that allowance. Failed continuations
are rejected before grasp; the physical lift is checked again with the actual
attachment. Pouring still changes only wrist_roll, with a 165-degree goal and
155-degree checker threshold. The gaze remains on the condiment until the lift
and carry fold finish, then switches to the bowl.

The 100-step cue pause is the sole approved D6 exception. Source 20 Hz recordings,
native replay and true held-action 10 Hz replay are unchanged. Earlier rollout
success rates do not qualify profile v7; full checklist and human review remain
separate from task success.

## Follow-up profile v7 fixed-pool result

The development pool `season-final-001`, seeds 7700000–7700039, contains 40
attempts: 39 successes and one planning refusal (7700036). All 39 successes
passed native 20 Hz replay and physical 10 Hz replay with each action held for
two control steps. All 39 have wrist-roll-only pouring, stationary base and
constant other arm/torso commands, a positive elbow through grasp/lift/carry,
and the complete initial 100-step cue observation. At grasp the hand camera is
above the hand; measured horizontal approach pitch is within 0.287 degrees.

A three-camera segmentation audit of every 10 Hz state gives minimum per-episode
target visibility 97.08%. Roll/turn limits and even gripper switching pass.
D6 still fails after the authorized cue exception on 7700017, 7700029, 7700035;
the raw and post-cue metrics are retained. No further waiver is assumed.

For the reported elbow inversion on 7700000, 118 triptychs at 0.1-second spacing
were inspected. Grasp-to-drive elbow inversion disappears; arm joint travel
falls from 30.032 to 16.870 radians and duration from 312 to 224 control steps.
Ten complete episodes retain all three original-resolution 20 Hz camera streams;
29 remaining successes have storyboards. Owner E5 approval is pending.
This is a development pilot, not a qualified 1000-episode dataset.

## Profile v8: upright compact carry, reachable-bowl shortcut, salad

The compact shoulder/elbow endpoint stays fixed. Wrist flex and roll are solved
from the held object's measured attachment throughout the fold, with a planned
5-degree target and a hard physical 10-degree transfer limit. Every executed
transfer step also checks that the object remains grasped. The final wrist-only
pour retains 165 degrees target / 155 degrees success and constant other arm,
torso and stopped-base targets.

Before grasping, the connected path must admit an upright lift followed by a
complete fixed-base pour or a checked upright carry. After the physical lift,
the fixed-base pour is tried first; a reachable bowl skips folding and driving.
Failed previews execute no actions. The bowl contains deterministic visual-only
lettuce, tomato wedges and cucumber slices attached to the same bowl actor;
these add no physical actors or state fields and consume no episode RNG.

Two development smoke attempts exercised both routes successfully. These are
not the final profile-v8 pilot; fresh 40 attempts and both replays are required.

## Profiles v9–v11: complete lookahead and loaded dock clearance

The grasp preview includes roll extrema of the proposed approach, grasp and lift
before reserving the complete wrist-only pour. A carry continuation may establish
grasp feasibility; the actual post-lift decision still prioritizes direct pouring.
Expensive curved approaches are planned only after continuation feasibility.

Before any loaded base motion, check every turn and straight segment against the
complete robot/payload geometry and check a full pour at the proposed arrival.
At the left wall, try the original dock followed by deterministic offsets of
0.15, 0.25, 0.35 and 0.45 m to the right. Execute only a checked route, then replan
from the physical arrival. The gaze switches to the bowl after the lift.
Earlier profile-v8/v9 development attempts retain their original code binding;
they do not qualify v11. No extra D6 or collision exception is introduced.
