# Depth Recall v1 collection

`MikasaDepthRecall-v1` uses four coloured props in a cabinet row. The expert
removes them onto four shuffled counter slots, keeps the deepest prop, and
returns the three blockers to their original shelf positions. This profile
implements the shared [16-point data contract](../DATA_CONTRACT.md) for kitchen 0
with the unmodified master DSFetch (`4c5c8b3de941565ee2d57b153a10254fe64c2849`).

The branch starts at `50b9ed02`, which supplies the Cabinet Search / Season Dish
collection infrastructure not yet merged into master. It does not include the
later Same Drawer or distant-start Season Dish task changes. After those shared
prerequisites merge, rebase the task changes onto master before merging.

## Information and completion

The remembered answer associates each prop's identity with its original depth.
The initial permutation is seeded; the front prop hides deeper props, which
become visible as the row is cleared. A separate oracle RNG (`seed + 10007`)
shuffles counter destinations. After staging, the policy receives no original
slot labels, target identity or task counters: only three RGBs, 12D proprio and
the task instruction. Answers and stage metrics remain in debug state / info.

Random expert staging does not prevent an unrestricted policy from encoding
order in its own placement choices. The 1/6 guessing reference conditions on
the three blockers and randomly assigns them to their three original slots;
it is not a universal bound for policies without internal memory. No learned
memory-performance claim follows from expert feasibility or data qualification.

Every blocker must have left its shelf position and previously stood upright,
settled and released on a counter slot after a grasp. A sideways shelf nudge
cannot count as staging. Wrong settled assignments, returning the deepest prop,
or two props in one row slot latch an error. Final completion requires all three
blockers in their own slots and the target standing on a counter slot for 15
control steps. `success` describes the current final arrangement;
`success_once` separately remembers whether completion happened earlier.

The scene restores the private v1 staging predicate, checkpoints it, resets
velocities, and applies RoboCasa kitchen exemptions only to wheels and base.
Arm and fingers retain physical contacts with the kitchen. The robot class,
URDF/SRDF, controllers, cameras and meshes are unchanged.

## Geometry and motion

The four props measure 4.5 × 4.5 × 12 cm. The shelf row has 7 cm pitch; the counter
slots retain 12 cm pitch across a 36 cm span west of the row. Their 7.6 cm visual
pads are collision-free markings on the existing physical countertop, not
separate platforms. Increasing a marker would not add support surface or grasp
clearance. The initial 16-case development sweep staged all four props in every
episode; its failures occurred on the return into the cabinet, not from a lack
of counter space. No countertop geometry was enlarged on that evidence.

The expert uses geometric grasps, collision-checked IK / joint lines and screw
paths, with bounded RRT recovery. It docks once and performs seven transfers.
Loaded base travel uses `drive_straight(..., hold_pose=True)`: arm, torso and head
targets are held fixed during the drive. The opt-in parameter in the shared
motion solver restores a feature from the original private implementation.
Previously, reusing each newly measured joint position accumulated gravity sag;
a development prop dropped 5.6 cm and caught the shelf edge. This changes action
generation, not the robot model or controller gains. Other callers retain the
existing compliant default. Physical joint angles are never rewritten to remove
roll winding; all motion must be represented by recorded actions.

An independent RNG (`seed + 200003`) adds uniform noise of up to 5 mm per enabled
world axis before planning:

| Waypoint | Perturbed axes |
|---|---|
| Initial base dock | x, y |
| Free shelf approach | x, z |
| High transfer over the counter | x, y, z |
| Free counter approach | x, z |
| Loaded shelf entry | x, z |

Contact grasp and final release goals remain geometry-derived. A perturbed goal
is sampled once and reused for its collision/IK probe and execution. Logs retain
sample index, axes, original pose, offset and proposed goal. Rejected candidates
also appear in the log; draw counts are not counts of executed movements.
The head initially faces the row, then the counter midpoint before restoration,
independently of the remembered prop assignment. It is not forced to zero on eval.

## Collect, replay and export

Use the simulator versions pinned in `depth_recall_profile.json`, RoboCasa assets
in `MS_ASSET_DIR`, and working NVIDIA Vulkan rendering (`VK_ICD_FILENAMES` when
needed). RGB works without a desktop. Use `uv run --no-project --python` with the
qualified simulator environment rather than an old editable ManiSkill checkout.
Pass the PaliGemma SentencePiece model; every full instruction is checked with
BOS and a trailing newline, without truncation.

```bash
python -m utils.collection.campaign --profile depth_recall \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --output /path/to/train --start-seed 11000 --num-seeds 8 \
  --purpose train --jobs 2 --through rgb --timeout 1200
python -m utils.collection.campaign --profile depth_recall \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --output /path/to/validation --start-seed 200000 --num-seeds 128 \
  --purpose validation --jobs 8 --through oracle --timeout 1200
```

Source H5 is state-only at 20 Hz. Successful source attempts must pass native
20 Hz action replay and a fresh replay holding the first action of each pair
for two control steps. Only qualified successes receive training RGB. Native
RGB and paired-action RGB are separate stages; the latter supplies LeRobot at
10 Hz. Changing metadata fps alone is not resampling. Failed attempts remain in
metadata and diagnostic H5 and never enter the training dataset.

Use the separate environment described in `requirements-lerobot.txt`:

```bash
python -m utils.collection.export_lerobot --input /path/to/train \
  --output /path/to/lerobot --repo-id mikasa-local/depth-recall-v1 \
  --tokenizer /path/to/paligemma_tokenizer.model
```

Actions are 13D: seven absolute arm targets, gripper, two absolute head targets,
absolute torso target and two normalized base velocity channels. RGB comprises
two native 256×256 / FOV 1.5 head cameras and one 128×128 / FOV 2.0 wrist camera.
There is no depth stream. `observation.state` is exactly `qpos[3:]` (12D);
`global_state = qpos[:3]` (3D) remains debug-only and is excluded by the policy
client. Metadata records all seeds and attempts, durations, rewards, final and
ever success, source H5 mapping, robot/runtime/asset versions, instruction tokens,
noise seeds, camera specifications and action/state semantics. The exporter uses
actual LeRobot v3 and checks readback against H5, including decoded RGB samples.

Freeze 100 unique planner-success validation seeds with
`python -m utils.collection.validation --runs ... --exclude-runs ... --output ...`.
Exclude all assigned training and development attempts, including failures and
diagnostic seeds passed via `--exclude-seeds`. Selection uses source success,
not 10 Hz replay or future policy performance. Never replace a frozen seed due
to an evaluated policy's failure.

## Repeat qualification

```bash
python -m pytest utils/collection/test_contract.py utils/collection/test_depth_recall.py -q
python -m utils.collection.qualify_depth --run /path/to/development \
  --source-seed 300 --output /path/to/observation-check --start-seed 300 --count 32
python -m utils.collection.qualify_head --profile depth_recall --output /path/to/head.json
python -m utils.collection.audit check --root /path/to/train --output /path/to/h5-check.json
```

The observation check verifies seeded resets, visibility of each newly exposed
prop before extraction, identical policy inputs for all 24 hidden associations
at a fixed staged state, and task-state roundtrip. These diagnostic counterfactuals
are never recorded as demonstrations. Behavioral tests reject shelf-only nudges,
wrong assignments, target returns, unstable staging and a knocked-over final
arrangement; repeated evaluation cannot advance the physical hold timer.

## Qualified implementation

Qualified on source SHA256
`c9b04da3b7197d4e3bdbd70311d4df44603db5ee08eb0d3a4ddba58169695aa2`:

- Development seeds 1000–1099: **99/100** source successes. The first 16 were
  development cases used to assess the loaded-drive fix (7/16 before, 16/16
  after); the remaining 84 ran the frozen candidate. This is not a learned-policy
  evaluation or an independent pre-registered estimate.
- All 100 cases staged all four props. The pool covered 100 different base
  starts, all 24 original assignments and all 24 counter staging assignments.
  Successful episodes took 179.05–196.6 s (median 183.15 s). Each episode logged
  13–15 perturbed proposed waypoints; rejected proposals are included.
- Eight training candidates passed source, native 20 Hz replay, paired-action
  replay, native RGB and paired RGB. All 48 native arrays matched the source
  exactly in every episode. Genuine LeRobot v3 contains **8 episodes / 14,660
  frames at 10 Hz**, with every numeric row and 120 decoded RGB samples checked.
- **100 fixed validation seeds** (200000–200099) were selected from 127/128
  source successes, excluding 441 historical/development/training/diagnostic
  seeds. See [the portable list and qualification hashes](validation_seeds/depth_recall_v1.json).
  Selection uses only source planner success; validation native/paired replays
  were not run. Do not replace this list after policy evaluation.
- 31 contract/behavior tests and 280 H5 recordings passed checks. The robot's 40
  model files match master byte-for-byte; all 70,204 pinned kitchen asset files
  match the reference content hash.
- 32 reset cases, four exposed-prop visibility checks, 24 hidden-answer
  counterfactuals and task-state roundtrip passed. Full instructions use
  29/32/34 PaliGemma tokens, including BOS and trailing newline.
- A recorded-action policy client completed seed 11000: 1,850 policy requests,
  3,700 control steps, no clipped targets, actual RGB/12D inputs. This checks
  the policy interface; it is not a trained VLA result.

Remaining planner failure: seeds 1092 and 200125 tipped the first blocker during
its counter re-grasp, then had no collision-free approach for the next blocker.
Those failures and their diagnostic trajectories remain in the attempt logs;
they were not discarded from the reported SR. Staging area was sufficient in all
100 development cases, so this release keeps the countertop and slot spacing.
Historical 197/200 results using a modified robot are not current qualification.
