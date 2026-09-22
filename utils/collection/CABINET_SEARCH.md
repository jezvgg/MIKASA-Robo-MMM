# CabinetSearch collection profile

`cabinet_search_profile.json` specifies the four-compartment NUDGE task.
The terminal is a horizontal cube displacement of at least 0.02 m with the
correct compartment open and no rule failure. Merely opening the door is not
success. The alternative SEEN terminal is not this dataset.

Before opening the first compartment, visit the floor mark. Close an empty
compartment and return to the mark before checking a different one. Do not
search a compartment again. The cabinet containing the found cube need not
be closed after success. Door-grasp retries before leaving are distinguished
from a new search decision by the task's existing rules.

The episode limit is 7100 control steps at 20 Hz (355 seconds). Two physical
cabinets contain four separated searchable compartments. The red cube has
side length 0.10 m. Its compartment and within-compartment placement, the
robot start, and instruction choice are randomized by the scene seed.
Non-search kitchen-joint movement is logged; `foreign_drift_fails=False` is
the existing explicit rule, not an unreported new failure condition.

The robot is the existing DSFetch from `jezvgg/MIKASA-Robo-MMM` at commit
`4c5c8b3de941565ee2d57b153a10254fe64c2849`. Its source, joint limits,
controllers, URDF/SRDF and meshes are preserved byte-for-byte; see
`robots/fetch/reference.json`. The collection profile is version 2.
The scene adds RoboCasa's usual wheel exclusions (bits 25-30)
and base exclusion (bit 31), with no kitchen exclusions on the arm,
head or fingers. Solver/head/noise fixes are separate from the robot model.
Recordings under the former RoboBenchMart robot profile or former blanket
collision exclusions retain their original provenance. They require new
physical qualification before being used as evidence for this profile.
The engine/runtime versions are also pinned by the profile.

The dataset records 20 Hz, validates replay and the 10 Hz action-hold variant,
then exports LeRobot v3 with three native RGB cameras, 13D actions, 12D
proprioception, language, and separate debug-only 3D global base state.
All candidate attempts and conversion losses remain in source metadata.
Each episode stores `success` for its final control step and `success_once`
for any successful control step. Both are computed from the full H5 history;
final success still controls training selection. Failed attempts retain both
flags. Worker failures without a complete recording report unknown values as
`null`. Export readback compares the summary fields against the source H5.
No production dataset or completed qualification is implied by this profile.

## Current verification

On the unchanged `master` DSFetch, 13 contract tests pass, including distinct
final-success and ever-success cases. Development seed 10001 passed state-only
collection, native 20 Hz replay, paired-action 10 Hz replay, both RGB passes,
and LeRobot v3 export/readback (467 frames). This is a pipeline check, not a
success-rate estimate. A new fixed-pool pilot and training/100-seed validation
release on this robot remain to be collected. No learned VLA was evaluated.

The robot originates from the existing upstream source; the RoboBenchMart MIT
notice is retained in `robots/fetch/LICENSE.RoboBenchMart`. Machine-readable
source and asset hashes live in `robots/fetch/reference.json`.
The shared [data contract](../DATA_CONTRACT.md) defines the 16 requirements.

## Run the pipeline

Use a simulation environment with the exact packages from the profile. `uv run
--no-project --python /path/to/sim-env/bin/python` avoids resolving the repository's
legacy editable ManiSkill source. Install RoboCasa assets and set `MS_ASSET_DIR`
for that installation. A working NVIDIA Vulkan ICD is required for offscreen RGB;
set `VK_ICD_FILENAMES` to its JSON when driver discovery needs it. No desktop or
interactive viewer is required. State-only collection does not render RGB frames.

```bash
python -m utils.collection.campaign --output /path/to/development \
  --start-seed 1000 --num-seeds 100 --purpose development \
  --jobs 4 --through validated
python -m utils.collection.campaign --output /path/to/development \
  --start-seed 1000 --num-seeds 100 --purpose development \
  --jobs 4 --through rgb
```

The second command resumes completed phases. Failed results are retained and are
not silently retried. Worker locks prevent a second observer from restarting a
live job. A changed implementation requires a new run directory. The saved run
contains the full task configuration, source hashes, engine hash, robot reference,
noise settings, seeds and dependency versions. Failure H5s are named
`failed-trajectory.h5`; they are diagnostic artifacts and cannot enter export.

The phase order is `oracle` → `native` → `validated` → `native_rgb` → `rgb`.
The first two execute all 20 Hz actions. `validated` executes the first target of
each pair twice in a fresh environment, including an extra hold for an odd final
step. Both RGB phases render complete saved states from their corresponding
successful execution. Restoring states for rendering is not a physical replay.

H5 includes actions, rewards, success/failure flags, complete task state,
`timestamp` (T+1 control timestamps), `qpos` (15D), `proprio` (12D), and
`global_state` (3D, debug only). The 13D action uses absolute arm/head positions
in radians, absolute torso height in metres, normalized gripper opening, and
normalized base velocity commands. The reference base scales them to ±1 m/s and
±3.14 rad/s; these scales are also recorded in each trajectory's metadata.

Export with the separate environment from `requirements-lerobot.txt` and a local
PaliGemma SentencePiece model. No upload to the Hub occurs:

```bash
python -m utils.collection.export_lerobot --input /path/to/run \
  --output /path/to/dataset --tokenizer /path/to/paligemma_tokenizer.model
```

The exporter uses LeRobot 0.4.3's v3 writer and reader, preserves native camera
sizes, and checks every numerical sample plus representative decoded RGB frames.
MP4 is H.264 at CRF 18; decoding returns RGB. `source_h5_metadata.json` retains
all candidate attempts, seeds, lengths, rewards, successes, versions, and sampling
rules. `global_state` is a separate debug feature; the policy client explicitly
passes only `observation.state`, the three images, and the instruction.

Development, training, and validation use disjoint scene-seed pools. After
collection, `utils.collection.validation` freezes 100 unique planner-success
validation seeds in a separate JSON. All assigned training/development seeds,
including failed attempts, are excluded. A 10 Hz replay failure is reported and
does not silently replace a planner-success validation seed. There is no inherited
70/15/15 split, and no final test split is implied.


### Storage for a production collection

After qualifying native RGB on development episodes, a production campaign may use
`--skip-native-rgb`. This retains the original 20 Hz state-only H5 and both physical
replay H5 files. It renders the verified paired-action trajectory into a 20 Hz RGB H5
and exports every second observation/action at 10 Hz to LeRobot. Thus RGB observations
match the actions that passed replay. The additional native-action RGB copy is omitted;
the final RGB H5 and LeRobot videos remain available. The choice is recorded in run.json
and cannot change when resuming the same campaign. Native physics replay is always run.
