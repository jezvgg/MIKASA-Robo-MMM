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

Measured on kitchen 0, CPU physics, DSFetch from master `4c5c8b3`, simulation
source SHA256 `eeabb7707cca5d51027984badf8f32747c73feea99ca939f15d9826d1ca97f6c`:

| Fixed pool | Source planner | Native 20 Hz replay | Paired-action 10 Hz replay | RGB |
|---|---|---|---|---|
| Development pilot, seeds 2400–2499 | 54/100 | Not run | Not run | Not run |
| Training, seeds 10000–10015 | 14/16 | 14/14 | 10/14 | 10 episodes |
| Control, seeds 1000–1005 | 5/6 | 5/5 | 3/5 | Seed 1001 at both rates |

The pilot's expert SR is **54%**, below the desired >60% collection target
(95% Wilson interval 44.3–63.4%). The target is aspirational; no success threshold
or physical rule was relaxed to improve the reported rate. Of 46 failures,
14 lost the apple during lifting, 12 reopened the drawer during withdrawal,
and 10 failed to plan a drawer approach; the remaining cases involved other
path/contact or placement failures. Improving the apple grip and disengaging the
hand after closing are known planner limitations.

Twenty-three regression tests passed, including attempts to preplace the apple
or reopen the cue drawer before the interlude. 32 reset cases produced distinct
robot/object starts; all 96 counterfactual cue comparisons passed visibility and
hidden-answer input checks. The full instruction takes 40 PaliGemma tokens.
All 40 robot files equal the master reference. A recorded-action policy exercised
the actual 12D/RGB client successfully for 754 requests / 1508 control steps.
This checks the interface; it is not a learned-policy result.

The 10 accepted episodes export to genuine LeRobot v3: **7,356 frames at 10 Hz**.
Every numeric sample and episode summary was compared with H5; 150 decoded RGB
samples passed (MAE 1.393–4.051 on the 0–255 scale). Native replay matched every
saved array exactly in all 14 successful training episodes. All 372 recordings
from the control, pilot, training and qualification pools passed H5 audits,
including unsuccessful attempts and replays.

The completed validation pools (20000–20199) produced 109 source-planner successes.
The immutable [100-seed list](validation_seeds/same_drawer_v1.json) selects the
first 100 successful scene seeds in ascending order, excluding all 360 assigned
training, development and diagnostic seeds. Validation qualification is source-only:
native and 10 Hz replays were not run and are not conditions for selecting or
replacing these seeds. The manifest pins the implementation, robot, assets and
planner/noise settings and records candidate outcomes.
The qualification scope is kitchen 0 with CPU physics; it does not establish
other-kitchen or GPU-batching performance. The cue check compares answer variants
at the same physical state. It does not rule out a policy using its own joint
posture or object arrangements as external memory; interpreting memory use still
requires history/privileged-answer policy comparisons.
