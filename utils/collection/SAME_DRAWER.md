# Same Drawer collection

`MikasaSameDrawer-v0`, profile `same_drawer`, uses the three upper drawers of
one four-drawer column in RoboCasa kitchen 0. The robot closes the initially open
drawer, puts an apple on a plate at the distant counter, returns, and opens that
same drawer by at least 10 cm. It must stay closed (within 5 mm) until the apple
is placed; preplacing the apple or reopening early permanently invalidates the
attempt. Opening a different drawer beyond 2 cm also invalidates it. A uniform final guess has a 1/3 chance of choosing correctly before
physical manipulation errors are considered.

The initial open drawer is the visual cue. Closing it removes that cue; drawers
within 5 mm of closed snap to zero while closing or stationary. The detent
does not reset an opening stroke. This rule belongs
to the environment and runs identically during collection, replay and evaluation.
The planner never writes robot or fixture poses. Task answers and completion
latches are available in diagnostic state/metadata, never the RGB/proprio policy
input. Text contains the chore, not the drawer number.

## Implementation and data contract

The task uses the unmodified DSFetch from upstream master `4c5c8b3`, the
`pd_joint_pos` controller and 13D actions. Robot source/URDF/SRDF/meshes are checked
against `robots/fetch/reference.json` before recording. The scene applies RoboCasa's
wheel/base collision exclusions; arm and finger collisions remain enabled.

Object positions, the initially open drawer, its opening distance, and the robot's
initial XY/yaw vary with the scene seed. The recorded state retains the articulation
root pose: `qpos[:3]` alone is not the world base pose when this root varies.
Training receives only `qpos[3:]` (12D). `qpos[:3]` is stored separately as the 3D
debug global state. Action channels 8 and 9 carry the head targets unchanged.

The oracle grasps the centre of the handle for both drawer strokes. The free
approach to the initially open drawer keeps that drawer in the collision model;
only the deliberate contact stroke permits touching it. Closing moves the base
at 6 cm/s and stops when the measured drawer position reaches the existing closed
tolerance. It releases to a nominal 64 mm finger aperture, then clears the handle
by 6 cm along the measured finger direction before backing away. The short exit
uses base translation and the torso lift with arm/yaw held; at the lift limit it
tries the same straight exit with the arm. A refused exit or an early reopening
fails the attempt. Physical collisions remain enabled throughout.

The apple is pinched 22 mm below its mesh top and lifted along a straight Cartesian
path at half the normal path speed. This gives the pads purchase below the crown;
a shallow contact alone can report a grasp that slips as soon as the apple lifts.
All these changes use existing motion-planning and velocity-control primitives.

Small positional waypoint noise is sampled independently from a separate seed,
using uniform +/-5 mm on enabled world axes, before collision-checked planning.
Contact handle positions and the apple grasp remain exact. If the ordinary
travel pose refuses, the planner makes one short retreat and plans a named joint
posture for the empty arm; it never rewrites physical joint positions. Each proposed noisy
goal, including refused plans, is logged with its original position and offset.

Source H5 is state-only at 20 Hz. Training accepts only successful source,
native-action and paired-action replay episodes before RGB rendering. The 10 Hz
policy holds each selected absolute action for two 20 Hz control steps. RGB uses
only the robot's native three streams: two 256x256/FOV 1.5 cameras and one
128x128/FOV 2.0 wrist camera. Export uses genuine LeRobot v3 and verifies the result
by reading it back. Every attempt, including failure, remains in metadata.

This task branch is based on `feat/season-dish-collection` (`50b9ed02`) because
its shared collection infrastructure has not yet merged into master. Review the
SameDrawer changes relative to that branch, then rebase after the shared work lands.

## Commands

Use the pinned simulator environment and existing RoboCasa assets. `MS_ASSET_DIR`
points to the parent of `data/scene_datasets`; headless rendering needs a working
NVIDIA Vulkan ICD. Run through `uv run --no-project --python /path/to/sim/python`.
Full instructions are checked with the supplied PaliGemma tokenizer, without
truncation, before collection and again during export.

```bash
python -m utils.collection.campaign --profile same_drawer \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --output /path/to/development --start-seed 1000 --num-seeds 100 \
  --purpose development --jobs 4 --through validated
python -m utils.collection.campaign --profile same_drawer \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --output /path/to/train --start-seed 10000 --num-seeds 16 \
  --purpose train --jobs 4 --through rgb --skip-native-rgb
python -m utils.collection.export_lerobot --input /path/to/train \
  --output /path/to/lerobot --repo-id mikasa-local/same-drawer \
  --tokenizer /path/to/paligemma_tokenizer.model
```

The export command uses the separate environment in `requirements-lerobot.txt`.
Use a new run directory after any implementation change; resume never silently
combines different versions. Qualify native RGB before omitting its redundant copy.

Validation needs exactly 100 unique source-planner-success seeds, excluding every
assigned training and development seed, including failures and diagnostic seeds.
Collect candidates with `--purpose validation`, then freeze them using
`python -m utils.collection.validation --runs ... --exclude-runs ... --output ...`.
Ten-Hz replay results remain a separate measurement and do not replace fixed seeds
when a tested policy or resampled replay fails.

## Qualification

```bash
python -m pytest utils/collection/test_contract.py -q
python -m utils.collection.qualify_drawer --run /path/to/development \
  --output /path/to/cue-check --start-seed 300 --count 32
python -m utils.collection.qualify_head --profile same_drawer --output /path/to/head.json
python -m utils.collection.audit check --root /path/to/train --output /path/to/h5-audit.json
```

Measured on kitchen 0, CPU physics, DSFetch from master `4c5c8b3`. The revised
planner has runtime SHA256
`cfee6b3ddb898ad5fdd9cf2c8844d8174cf70034aea429a1f26c30ad5aa4ec10`.
The baseline is commit `8024c0e`, runtime
`eeabb7707cca5d51027984badf8f32747c73feea99ca939f15d9826d1ca97f6c`.
Only `planners/same_drawer_planner.py` differs in the runtime signature; the robot,
engine, task configuration, success rules and 1600-step horizon are identical.

| Fixed pool, all attempts | Baseline source success | Revised source success |
|---|---|---|
| Development, seeds 2400–2499 | 54/100 | **73/100** |
| Fresh comparison, seeds 40000–40099 | 61/100 | **72/100** |

Development seeds were used during implementation. The final planner was frozen
before evaluating the fresh comparison pool; that pool is separate from the fixed
validation list. All candidates, including failures, contribute to these rates.
The revised rates exceed the desired >60% collection target on these two pools;
95% Wilson intervals are 63.6–80.7% and 62.5–79.9%. The fresh paired comparison has
26 gains and 15 losses (two-sided exact McNemar p=0.117): the observed increase
still has substantial sampling uncertainty, and is not a guarantee for other pools.

Premature reopening during withdrawal fell from 12 to 0 development cases and
from 7 to 0 fresh cases. Apple losses during lifting fell from 14 to 0 and from
14 to 2. The remaining development failures were 15 drawer-approach refusals,
4 plate-step refusals, 4 horizon expirations, 2 travel refusals, 1 straight-contact
refusal and 1 missed handle grasp. The fresh pool had 10 drawer-approach refusals,
8 plate-step refusals, 5 horizon expirations, 2 lift losses, 1 transfer loss,
1 travel refusal and 1 apple-withdrawal refusal. These are planner limitations;
they do not establish that a failed scene seed is physically unsolvable.

Motion is not uniformly shorter. On the 46 fresh seeds successful in both
versions, median cumulative wrist rotation decreased from 538 to 416 degrees;
median base rotation decreased from 337 to 332 degrees. Median base travel grew
from 5.12 to 5.46 m, and duration from 71.65 to 74.95 s. Travel and stow motions
remain opportunities for improvement. These measurements concern the shared
successful subset, not all assigned seeds.

The full collection check used training seeds 10000–10007:

| Source | Native 20 Hz replay | Paired-action 10 Hz replay | RGB | LeRobot v3 |
|---|---|---|---|---|
| 6/8 | 6/6 | 6/6 | 6 episodes | **6 episodes / 4,488 frames at 10 Hz** |

Both unsuccessful source attempts remain in metadata. Native replay matched all
38 saved datasets exactly in every successful episode. Export readback checked
all numerical samples and 90 decoded RGB samples (MAE 1.414–3.864 on the 0–255
scale). All 226 new H5 recordings and the 100 fresh baseline recordings passed
format, provenance, episode-summary and waypoint-noise audits. Six accepted
training episodes are a collection qualification sample, not a production dataset
or an estimate that resampling always preserves success.

Twenty-three regression tests passed on the revised code. All 40 robot files
match the master reference. A recorded-action policy successfully exercised the
actual 12D/RGB client for 735 requests / 1470 control steps, with no clipped
actions. This checks the interface; it is not a learned-policy result.

The unchanged scene's initial qualification included 32 reset cases and 96
counterfactual cue/hidden-answer input comparisons. The full instruction takes
40 PaliGemma tokens. The completed original validation pools (20000–20199)
produced 109 source-planner successes under `8024c0e`. The immutable
[100-seed list](validation_seeds/same_drawer_v1.json) still selects the first 100
successful scene seeds in ascending order, excluding all 360 originally assigned
training, development and diagnostic seeds. The new comparison pools are also
disjoint from that list. Its membership and original provenance are unchanged;
it was not reselected against the revised planner. Original validation
qualification is source-only: native and 10 Hz replays were not conditions for
selecting or replacing seeds. The original manifest pins the implementation,
robot, assets and planner/noise settings and records candidate outcomes.

The qualification scope is kitchen 0 with CPU physics; it does not establish
other-kitchen or GPU-batching performance. The cue check compares answer variants
at the same physical state. It does not rule out a policy using its own joint
posture or object arrangements as external memory; interpreting memory use still
requires history/privileged-answer policy comparisons.
