# Depth Recall v1 collection

`MikasaDepthRecall-v1` uses four props in a cabinet row — RoboCasa spices and sauces
since 2026-10-16 (ketchup, pepper and salt shakers, mustard), each its own colour. The
expert removes each onto the counter pad of its own colour (the episode shuffles which
counter slot each colour's pad lies on), keeps the deepest prop, and returns the
three blockers to their original shelf positions.

> **Props changed 2026-10-16 (branch `feat/depth-recall-v1-realistic-props`).** The four
> coloured 4.5 × 4.5 × 12 cm boxes are RoboCasa objects; see
> [RoboCasa props](#robocasa-props-spices-and-sauces-2026-10-16) (the last section). Every
> measurement in the other sections was made with the boxes.

> **Task changed 2026-10-16 (branch `feat/depth-recall-v1-colored-pads`).** The pads are
> coloured per prop and 18 cm apart (were all yellow, 12 cm, any free slot). Everything
> under "Qualified implementation" below was measured on the previous version and does
> not qualify this one; the validation seed list and source hash are void for it. See
> [Coloured pads](#coloured-pads-2026-10-16). This profile
implements the shared [16-point data contract](../DATA_CONTRACT.md) for kitchen 0
with the unmodified master DSFetch (`4c5c8b3de941565ee2d57b153a10254fe64c2849`).

The branch starts at `50b9ed02`, which supplies the Cabinet Search / Season Dish
collection infrastructure not yet merged into master. It does not include the
later Same Drawer or distant-start Season Dish task changes. After those shared
prerequisites merge, rebase the task changes onto master before merging.

## Information and completion

The remembered answer associates each prop's identity with its original depth.
The initial permutation is seeded; the front prop hides deeper props, which
become visible as the row is cleared. Counter destinations are the pads of the
props' own colours, drawn by the episode RNG after the row permutation and the
start jitter (so a seed keeps its row and start); they are drawn independently of
the row order, so the staged layout carries no information about it. After staging, the policy receives no original
slot labels, target identity or task counters: only three RGBs, 12D proprio and
the task instruction. Answers and stage metrics remain in debug state / info.

Since 2026-10-16 the pad of each colour is fixed by the episode, so a policy can
no longer encode the row order in where it stages the props. The 1/6 guessing reference conditions on
the three blockers and randomly assigns them to their three original slots;
it is not a universal bound for policies without internal memory. No learned
memory-performance claim follows from expert feasibility or data qualification.

Every blocker must have left its shelf position and previously stood upright,
settled and released on its OWN pad after a grasp. A sideways shelf nudge
cannot count as staging. A prop released on another colour's pad (`wrong_pad`),
wrong settled assignments, returning the deepest prop, or two props in one row
slot latch an error. Final completion requires all three blockers in their own
slots and the target standing on its own pad for 15 control steps. `success` describes the current final arrangement;
`success_once` separately remembers whether completion happened earlier.

The scene restores the private v1 staging predicate, checkpoints it, resets
velocities, and applies RoboCasa kitchen exemptions only to wheels and base.
Arm and fingers retain physical contacts with the kitchen. The robot class,
URDF/SRDF, controllers, cameras and meshes are unchanged.

## Geometry and motion

The four props measured 4.5 × 4.5 × 12 cm until 2026-10-16; they are now RoboCasa
objects 4.2-5.2 cm across and 9.45-16.5 cm tall
([RoboCasa props](#robocasa-props-spices-and-sauces-2026-10-16)). The shelf row has 7 cm pitch; the counter
slots have 18 cm pitch across a 54 cm span west of the row, at y = -0.56 (see
[Coloured pads](#coloured-pads-2026-10-16)). Their 9 cm pads, one per prop colour, are
collision-free markings on the existing physical countertop, not separate platforms. Increasing a marker would not add support surface or grasp
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

## Coloured pads (2026-10-16)

The owner's change: wider spacing between the counter staging points, and a pad of each
prop's own colour generated per episode (they were four identical yellow plates and the
oracle staged on any free one).

**Task.** Pad c has prop c's colour; the episode RNG draws which of the four counter slots
each pad lies on, after the row permutation and the start jitter. A prop counts as staged
only on its own pad; one released on another colour's pad latches `wrong_pad`; the target
must finish on its own pad (`target_on_its_pad`). State observations gain `pad_points`
(where each prop's pad is, prop order). The clip has a second view along the counter.

**Spacing, and the measurement that set it.** 18 cm pitch (was 12), 54 cm west of the row,
at y = -0.56 (was -0.525). The arm works from ONE dock, so the spacing is a reach
question. `scripts/dr_pad_reach.py` asks cuRobo's batched scan for every pose the planner
makes there — the restore grasp at each grasp lift and its pre-grasp, the extraction's
place and its cross heights — from every stand (row x +-4 cm), with self-collision, hits
re-checked in float64, under the planner's own rule (some lift, the place height, some
cross height):

| carry-out rule | restore grasp reaches | place reaches |
|---|---|---|
| back until the prop is over the pads' line (old) | 62 cm west | **44 cm** |
| back to the dock line (`carry_out`, new) | 64 cm (y -0.56) | **60 cm** (y -0.56), 56 cm (y -0.525) |

So the extraction now backs the base to the dock line before crossing, and the pads moved
3.5 cm nearer the robot, where the cross to 54 cm plans at two torso heights from every
row slot. The scan is kinematics plus self-collision only: the scene is the planner's to
check, and the sweeps below are what confirm it.

**Planner changes it took.** Both crosses (extraction and restore) used to lower the
TORSO to come down, which takes the hand down WITH the shoulder and leaves its reach
unchanged: on seed 1000 every torso rung was `IK Failed` for the 54 cm pad, and the
restore then approached "from where we are", tipped the prop and grasped it lying. The
crosses now first lower the HAND under the shoulder (extraction: 0, 10, 20 cm below the
carry height; restore: 25-35 cm over the counter, under the closed left door and over the
standing props); the torso rungs remain as fallbacks.

**100 development seeds (1000-1099).** The first sweep (42ce525) gave 97/100 by state and
showed two defects, both fixed (`68c5f27`):

- after a counter placement the hand rises 6 cm, 7 mm above the released prop's top. The
  raise right after a release now leaves the released prop out of the check: it lifts the
  hand straight up, away from it;
- the restore's cross ended ~9 cm past the pads' line after a row placement, and the
  grasp posture then passed over the prop, pushing it 7-12 cm or tipping it (4 state
  episodes). The cross now ends 15 cm in front of the prop.

With the fixes (H100, tree `68c5f27`): state **100/100** (Wilson 95 % 0.963-1), steps
p50 3712 / p90 3774 / max 3969, 0 knocked over, 0 close misses.

Every final arrangement had the target on its own pad (100/100) and no prop ever stood
on another colour's pad. The longest episode, 3969 steps, is inside the 5700-step horizon.

Same tree, more seeds:

| run | success | notes |
|---|---|---|
| fresh seeds 3000-3199, state | **200/200** (0.981-1) | steps p50 3707, max 3931 |
| seeds 1000-1099, state + assisted grasp | **100/100** | welds 7/7 in every episode |
| seeds 1000-1099, state, cuRobo IK (shared server, no jump filter) | **100/100** | 6302 cuRobo calls over 200 episodes |
| seeds 1000-1099, state, **blind** restore | 18/100 (Wilson 0.12-0.27) | the 1/6 guessing reference is 0.167: the pads carry no row order |

Against the previous version on the same 100 seeds the episodes are 1.4 % longer in
steps (median 3712 vs 3662 by state), all of it the longer crosses to the far pad; the
wall clock is ~30 % longer, which is planning, not motion: ~4.8 refused rungs per
episode (the far pad's first extraction rung always, the restore's hand-height rungs for
near pads) before the one that plans.

**`db43115`: rungs ordered by distance.** Far pads (more than 45 cm west of the base)
start both crosses with the hand-under-the-shoulder rungs, nearer ones with the torso
rungs. Seeds 1000-1052 by state **53/53** (the sweep was stopped there, on the owner's
word, to free the machine); refused IK attempts per episode **0.00** (4.7 on the same
seeds before); steps p50 3734 (3716 before: the near pads' torso rung adds a short ride).

## Assisted grasp (experimental, off by default)

`MikasaDepthRecall-v1` accepts `assisted_grasp=True`. The prop is then welded to the
hand by OmniGibson's assisted-grasp rule, the one BEHAVIOR's own evaluator runs and
BIMBO_EvoX reproduced (`utils/mikasa/assisted_grasp.py`):

- the gripper is commanded anywhere below fully open;
- both fingers touch the prop and at least one of four rays between the finger pads
  hits it first (a closed hand pushing from outside touches without a ray hit and never
  attaches); the nearest such prop is the candidate;
- the same candidate for 0.3 s of physics steps — then a fixed PhysX drive joins hand
  and prop at their current relative pose;
- the weld goes the moment the gripper is commanded fully open; a new one is refused
  for 1/30 s.

How close this is to OmniGibson's own code, point by point, and what differs:
`utils/mikasa/ASSISTED_GRASP_FIDELITY.md`.

The rule runs in `_after_simulation_step`, i.e. on every 100 Hz physics step. The
drives are made once per build and only enabled or disabled; the weld state is part of
`get_state_dict`. CPU simulation, one env only.

**It changes the task.** Friction lets a prop slip; the weld does not. Demos recorded
with it and policies evaluated on them belong to a different variant, and both sides
must run with the same setting. Nothing above under "Qualified implementation" was
measured with it; the default (`False`) keeps that behaviour.

Enable it with `gym.make("MikasaDepthRecall-v1", ..., assisted_grasp=True)`, the
planner's `--assisted-grasp`, or `"assisted_grasp": true` in a profile's `env_kwargs`.
On a Mac, run under amd64 emulation (mplib has no macOS wheels):

```
utils/docker/amd64.sh python -m pytest -q utils/mikasa/test_assisted_grasp.py
utils/docker/amd64.sh python planners/depth_recall_v1_planner.py --seed 1000 --assisted-grasp
```

Measured 2026-10-15 on an H100 server (GPU rendering, CPU physics), development seeds
1000–1099, the same planner with and without the flag:

| | success (95 % interval) | control steps (median) | welds made / released per episode | grasp misses |
|---|---|---|---|---|
| `assisted_grasp=False` | 100/100 (96–100 %) | 3662 | — | 0 |
| `assisted_grasp=True` | 100/100 (96–100 %) | 3662 | 7 / 7 on every seed | 0 |

Seven welds per episode are the planner's seven grasps (four extractions, three
restores); each forms 0.3 s after the fingers close and goes when they open. On this
task, where friction already holds the 4.5 cm props, the weld neither helps nor hurts
the oracle.

## RoboCasa props: spices and sauces (2026-10-16)

The owner's change: the four coloured boxes become RoboCasa kitchen objects, spices and
sauces. Identity is still the colour, so each prop needs its own hue, and its pad has it.

| prop | pad | RoboCasa object (category / instance) | collision mesh at RoboCasa's scale | mass | drawn |
|---|---|---|---|---|---|
| `prop_0` ketchup | red | `ketchup` / `ketchup_7`, squeeze bottle | 4.20 × 4.33 × 16.50 cm | 19.3 g | its own red |
| `prop_1` pepper | green | `shaker` / `shaker_0` | 5.17 × 5.09 × 9.45 cm | 15.8 g | black body green; black top kept |
| `prop_2` salt | blue | `shaker` / `shaker_1` | 5.17 × 5.09 × 9.45 cm | 15.8 g | white body blue; white top kept |
| `prop_3` mustard | yellow | `ketchup` / `ketchup_8`, squeeze bottle | 4.28 × 4.30 × 15.00 cm | 17.4 g | flat orange-red drawn yellow |

**Why three are recoloured.** Every objaverse shaker, condiment, ketchup, jam, spray,
liquor and bottled-drink instance was loaded at its registry scale, rendered on this
kitchen's counter and segmented (`probe_props`, 2026-10-16). The ones that fit the row
(footprint half-width <= 2.8 cm) are red sauces (`ketchup_2/7/8/9/10/11`,
`condiment_0`, hues 0-8°) and achromatic shakers (black, white, cream glass, clear glass;
saturation <= 0.27). The spices and sauces with other hues are too wide (peach dressing
7.4 cm, BBQ sauce 7.9 cm, cooking spray 10.8 cm deep) or are short jam jars. So the
ketchup keeps its red and the other three keep their geometry and get a flat body colour
(`load_objaverse_actor(recolor=...)`; SAPIEN takes the albedo from a texture when there is
one, so the recoloured part's texture is dropped). The pads keep the four hues the
box version measured. **This is the owner's call:** recolouring is the only way found to
keep salt and pepper shakers with hue identity; the alternatives are other objects or an
identity that is not hue.

**Scene.** `PROPS` records each object and its measured footprint and height;
`DepthRecallV1Config.props` replaces `prop_half_xy` / `prop_half_h`, `validate()` checks
the row against the widest prop (7 cm pitch − 2.62 cm > finger reach + 5 mm) and every
prop's height against the side grasp's band (>= 9.2 cm, below), and `_load_scene`
re-measures the loaded meshes and refuses to build if an asset has drifted by more than
a millimetre. A RoboCasa object's origin is its mesh centre, not its base, so the spawn
and the "standing on the shelf / counter" checks use each prop's own rest lift (origin
over mesh bottom: 8.25 / 4.75 / 4.75 / 7.5 cm). The task logic (row memory, own-colour
pads, latches, success) is unchanged.

**PhysX contact offset 5 mm** (`SCENE_CONTACT_OFFSET`, ManiSkill's default 2 cm). PhysX
makes contacts between two shapes within the sum of their offsets and its solver acted on
them before anything touched: traced step by step on seed 1004, a 17 g mustard bottle
was launched out of the row while a finger and the shaker being lowered were 1.6-2.1 cm
from it and no contact of either with it was reported. The 243 g boxes never showed it.

**Planner** (`depth_recall_v1_planner.py`), each change from a traced failure:

- `grasp_lifts`: the grasp-lift ladder is clipped into the band where a side grasp is
  sound, TCP 7.9 cm or more over the support (the wrist hangs 6.7 cm under the hand) and
  the finger pads (+-1.32 cm) not above the prop's top. The 12 cm box and the bottles keep
  the ladder unchanged; the 9.45 cm shakers get 3.18 and 3.4 cm (their 2 cm rung left the
  wrist no room over the support).
- `settle_base` after both drives into the cabinet: the base coasts 1.1-1.2 cm after
  `drive_straight`, so a stroke planned at the drive's end overshot by that much (grasp
  stroke 1.7 cm past its goal; row placements 1.2-1.4 cm deep).
- `LOADED_DRIVE_COAST`: the loaded drive-in asks 1.3 cm less (measured overshoot
  1.15-1.44 cm); a finger otherwise reached the next prop in the row.
- After a release the torso only ever rises: after a row placement it is already at its
  0.386 stop, the empty line to it was booked as a refusal and the ladder LOWERED the open
  hand 4.6 cm onto the prop just released (a 15.8 g shaker slid 4 cm out of its slot).
- The carry-out uses the prop's own half-width.

**Tests** (H100, `run_tests.sh`): 68 passed — the evaluation tests with per-prop
heights (and two new: a prop judged at another prop's height, a lying shaker), validate
refusing a prop too wide or too short, the lift ladder, the contact offset.

**Measured** (H100, GPU rendering, CPU physics, tree `9c5ab71`, `sweep100.sh`, seeds
1000-1009, mplib's random IK restarts as in every sweep above):

| success | steps min / median / max | notes |
|---|---|---|
| **10/10** | 3595 / 3629 / 3700 | final tilt <= 0.2° in every episode |

A prop leans in the fingers right after the grasp, median 0.7°, worst 6.1°: flat pads
hold a round prop on a line (a tapered shaker on a point), not on a face as they held the
boxes. Ten seeds are a small sample, and with random IK restarts
a seed is not a repeatable outcome: before the contact offset was fixed, the same code
and seed failed in one run and passed in the next. During development (seeds 1000-1009,
40 episodes per configuration, same planner): contact offset 2 cm 36/40, 5 mm 39/40.
Not yet measured with the RoboCasa props: the 100-seed and validation sweeps, assisted
grasp, cuRobo IK, the blind restore.
