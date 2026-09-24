# Season Dish collection

This profile prepares `MikasaSeasonDish-v0` in kitchen 0 for the shared
[16-point data contract](../DATA_CONTRACT.md). It uses the unchanged DSFetch
from upstream `master` at `4c5c8b3de941565ee2d57b153a10254fe64c2849`.
The branch currently depends on `feat/cabinet-search-collection`, which supplies
the shared H5/replay/LeRobot implementation. No replacement robot is introduced.

## Task and information

Two condiments and a bowl are randomly positioned on the usable counter.
A yellow marker indicates the requested condiment for 40 control steps (2 s),
then disappears. The two condiments swap sides independently of the answer.
The demonstrator looks at their midpoint during the cue, independent of the
answer. The language instruction also does not name the answer.

The requested condiment must be grasped, within 0.10 m horizontally of the
bowl, 0.05–0.30 m above it, and tilted at least 155 degrees for 15 consecutive
control steps. Success is allowed after the 40-step cue and 40-step delay.
The other condiment must not be grasped or displaced more than 0.10 m from
its initial position. These are the existing task predicates; seasoning is
a pose proxy, without substance/fluid simulation. The horizon is 1100 steps
(55 s). The manipulation docks are derived from the object placements. The robot starts
0.75 ± 0.05 m behind the station dock, facing the condiments. After the cue disappears
it drives to that dock, then begins grasping. Distance jitter uses the seeded
placement RNG after the object draw. Initial poses vary across seeds; they are
not statistically independent of the station. The cue marker radius is 4.5 cm so
it remains visible from the distant start; the robot cameras are unchanged.

## Motion and noise

The existing expert uses geometric grasps, IK/screw paths, collision-checked
joint lines and RRT recovery. RL is not used. The scene adds RoboCasa's kitchen
exclusions only to wheels/base; arm and fingers retain physical kitchen contacts.
DSFetch class, URDF/SRDF, controllers, cameras and meshes are unchanged.

An independent RNG (`scene_seed + 200003`) adds uniform positional noise of up
to 5 mm per enabled world axis to these goals:

| Goal | Perturbed coordinates |
|---|---|
| Initial station dock after the cue | x, y |
| Free approach before grasp | x, y, z |
| Lift above the neighbouring object | z |
| Base stop at the bowl dock | x, y |
| Hover, optional pre-hover and hover correction | x, y, z |
| Pour candidates above the bowl | x, y, z |

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
