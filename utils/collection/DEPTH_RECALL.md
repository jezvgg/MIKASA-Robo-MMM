# Depth Recall v1 collection

`MikasaDepthRecall-v1` uses four coloured props in a cabinet row. The expert
removes each onto the counter pad of its own colour (the episode shuffles which
counter slot each colour's pad lies on), keeps the deepest prop, and returns the
three blockers to their original shelf positions.

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

The four props measure 4.5 × 4.5 × 12 cm. The shelf row has 7 cm pitch; the counter
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

**Perception mode** reads the pads' colours from the head cameras after the row is
calibrated (majority hue of the flat band over the counter within 3.5 cm of each slot;
all four colours must be read once), and closes the gripper only when the prop's own
points lie between the finger pads over its height (`PropBelief.between_pads`, BIMBO's
rays-between-pads rule; the band at the pads' own height is hidden from every camera by
the hand itself, measured on seed 1000).

**First runs** (H100 tree 42ce525 under a full machine): seeds 1000-1003 in both modes
**8/8**, every prop on its own pad, max tilt 0.1 deg, 3582-3815 steps; pads read right
4/4; all 28 between-pads checks passed with 218-622 points.

**100 development seeds (1000-1099).** The first sweep (42ce525) gave 97/100 by state and
79/100 by camera, and showed two defects, both fixed (`68c5f27`):

- after a counter placement the hand rises 6 cm, 7 mm above the released prop's top; a
  camera estimate is off by more, so the torso raise was refused at knot 0 (13 camera
  episodes). The raise right after a release now leaves the released prop out of the
  check: it lifts the hand straight up, away from it;
- the restore's cross ended ~9 cm past the pads' line after a row placement, and the
  grasp posture then passed over the prop, pushing it 7-12 cm or tipping it (4 state
  episodes, and most of the camera grasp-fit refusals). The cross now ends 15 cm in
  front of the prop.

With the fixes (H100, tree `68c5f27`):

| mode | success | steps p50 / p90 / max | notes |
|---|---|---|---|
| state | **100/100** (Wilson 95 % 0.963-1) | 3712 / 3774 / 3969 | 0 knocked over, 0 close misses |
| pointcloud | **100/100** (0.963-1) | 3806 / 3956 / 4221 | pads read right 100/100; between-pads checks 700/700 passed (195-624 points); wrist fits 700/700, error p95 2.3 mm |

Every final arrangement had the target on its own pad (200/200) and no prop ever stood
on another colour's pad. The between-pads check never had to refuse a close in these
200 episodes: it is a guard against closing on air, not a source of the success rate.
The longest episode, 4221 steps, is inside the 5700-step horizon.

Same tree, more seeds and modes:

| run | success | notes |
|---|---|---|
| fresh seeds 3000-3199, state | **200/200** (0.981-1) | steps p50 3707, max 3931 |
| fresh seeds 3000-3199, pointcloud | **199/200** (0.972-0.999) | pads read right 199/199; between-pads 1393/1393; the miss is seed 3011: the measured row shift (10.4 mm) was just over `REDOCK_MIN_M`, and `drive_base` turned the base for a 1 cm sideways re-dock, which the open right door refused (forearm) |
| seeds 1000-1099, state + assisted grasp | **100/100** | welds 7/7 in every episode |
| seeds 1000-1099, pointcloud + assisted grasp | **100/100** | welds 7/7; wrist fits 700/700 |

Against the previous version on the same 100 seeds the episodes are 1.4 % longer in
steps (median 3712 vs 3662 by state), all of it the longer crosses to the far pad; the
wall clock is ~30 % longer, which is planning, not motion: ~4.8 refused rungs per
episode (the far pad's first extraction rung always, the restore's hand-height rungs for
near pads) before the one that plans.

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

## Point-cloud perception (experimental, off by default)

`planners/depth_recall_v1_planner.py --perception pointcloud` (or
`solve(..., perception="pointcloud")`) makes every prop pose the planner acts on an
estimate from the robot's own cameras, the way BIMBO_EvoX's observation-only planners
work (`utils/mikasa/point_cloud.py`, `planners/depth_recall_v1_perception.py`):

1. Depth of the two base cameras and the wrist camera is back-projected into the world;
   points inside the robot's link collision boxes (and the carried prop) are cut out.
2. Row and counter slot points come from the static map (the cabinet's row centre, the
   configured slot y's and offsets, shelf and counter heights). The episode shifts the
   whole row sideways by up to 4 cm (`row_jitter_x`), which no image observation states:
   the first look from the dock measures that shift (median x of the accepted fits),
   the base re-docks on the measured row when it is more than 1 cm off, and every row
   and counter point after it carries the measured x.
3. A height band over the support within 10 cm of a slot point is clustered on a 2 cm
   voxel grid; a cluster is named by hue — colour is the prop's identity in this task by
   design.
4. A 1500-point model of the prop is fitted by trimmed planar ICP from a 15° yaw grid,
   started at the cluster's x and the slot's y on the shelf (the cameras look along y, so
   the seen face's x is the prop's), at the slot point on the counter (where the planner
   itself put the prop), and accepted only with inliers (8 mm) >= 0.55, RMS <= 1.2 cm and
   no more than 30 % of the model "seen through" by a camera. A cluster too small to fit
   keeps that start point as a coarse estimate.
5. From the dock the base cameras pick the front-most prop and remember the row slot it
   was seen in; that memory, not the episode's answer, drives the restore. A prop seen
   behind an unseen front one stops the episode rather than driving into the front one.
6. At the stand each grasp is made from, the wrist camera re-fits the prop (every rung of
   the grasp ladder), and the grasp goes ahead only if THAT look accepts a fit — an older
   estimate is exactly what a missed close may have made wrong. After the grasp the
   prop's pose is the hand's pose times the transform frozen at the grasp.

7. The planning world is the belief's, BIMBO_EvoX's fair rule (an offline map of the
   static scene plus the objects as seen): mplib keeps the fixtures and the robot, but
   after every sync each prop stands where perception put it (one not seen yet is
   parked out of reach), and a held prop is attached at the transform frozen at the
   grasp, never re-read from the simulator.
8. Holding is felt: the gripper commanded shut and both finger encoders stopped apart,
   not contact forces.

Still read from the simulator in this mode: the static scene map (fixtures, their
footprints and heights, the props' known size). Every perception step logs its fit and,
for the log only, its error against the true pose; the row shift logs its error against
the episode's row x.

Measured 2026-10-15 on an H100 server (GPU rendering, CPU physics), development seeds
1000–1099, the same planner in three modes:

| prop poses from | success (95 % interval) | control steps (median) |
|---|---|---|
| simulator (`state`) | 100/100 (96–100 %) | 3662 |
| cameras (`pointcloud`) | 100/100 (96–100 %) | 3766 |
| cameras + `--assisted-grasp` | 100/100 (96–100 %) | 3769 |

The row's sideways shift measured at the dock, against the episode's: error median
0.4 mm, 95th percentile 4.2 mm, worst 10.6 mm, for shifts up to 4 cm. Wrist-camera
re-fits before a grasp: 700 of 700 accepted in both camera modes; position error against
the true pose median 1.0 mm, 95th percentile 2.3 mm, worst 3.6 mm; yaw error median 1.0°,
worst 7.0°. The camera modes run ~100 steps longer: the re-dock and the extra looks.

The joint lines the planner executes are collision-checked every 0.01 rad
(`READY_LINE_STEP`). At `plan_qpos_line`'s default 0.1 rad, up to ~8 cm of hand travel on
an outstretched arm, two camera-mode episodes (seeds 1070, 1092) ran a line that put a
finger through the counter prop it was reaching for; the prop fell over and the wrist
camera then refused, rightly, to fit an upright prop to a lying one (traced by replaying
both seeds with every accepted line re-checked at 0.01 rad). With the dense check all
four modes pass 100/100.
