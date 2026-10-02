# Physical closing revision (profile 12)

The environment initializes closed hinges at reset and never changes their positions
or velocities afterward. Observations and the checker use actual angles, including
small drift on uninspected leaves. Empty compartments count as closed only after all
leaves remain within 0.01 rad, still and at least 0.25 m from the TCP for ten control steps.

The oracle pushes at 0.20 m/s, slowing to 0.05 m/s for the last 0.10 rad. The target is
zero; it never commands a negative opening angle. After the panel push it withdraws
0.30 m so the release check can run. At most two handle corrections are allowed.
There is no rest between empty inspection and closing; the robot returns directly
to the sampled mark, and a successful can push ends the episode at the can.

Development motion variants are explicit in each run and recording's
`motion_parameters`; old recordings retain their original source snapshots and
profiles. The 40-scene comparison and both physical replays must pass before release.

# Existing collection contract

`cabinet_search_profile.json` specifies the four-compartment NUDGE task.
The terminal is a horizontal cola-can displacement of at least 0.02 m with the
correct compartment open and no rule failure. Merely opening the door is not
success. The alternative SEEN terminal is not this dataset.

Before opening the first compartment, visit the floor mark. Close an empty
compartment and return to the mark before checking a different one. Return directly to its actual sampled position,
without a south waypoint or parking offset. Finish at the can immediately after
a successful nudge; no final return is required. Do not
search a compartment again. The cabinet containing the found can need not
be closed after success. Door-grasp retries before leaving are distinguished
from a new search decision by the task's existing rules.

The episode limit is 7100 control steps at 20 Hz (355 seconds). Two physical
cabinets contain four separated searchable compartments. A short cola-can mesh replaces the former red cube. Its compartment and within-compartment placement, the
robot start, and instruction choice are randomized by the scene seed.
Non-search kitchen-joint movement is logged; `foreign_drift_fails=False` is
the existing explicit rule, not an unreported new failure condition.

The robot is the existing DSFetch from `jezvgg/MIKASA-Robo-MMM` at commit
`4c5c8b3de941565ee2d57b153a10254fe64c2849`. Its source, joint limits,
controllers, URDF/SRDF and meshes are preserved byte-for-byte; see
`robots/fetch/reference.json`. Current profile version is specified in the JSON; earlier qualification below is historical.
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

## Historical pipeline verification

On the unchanged `master` DSFetch, 13 contract tests pass, including distinct
final-success and ever-success cases. Development seed 10001 passed state-only
collection, native 20 Hz replay, paired-action 10 Hz replay, both RGB passes,
and LeRobot v3 export/readback (467 frames). This is a pipeline check, not a
success-rate estimate. A new fixed-pool pilot and training/100-seed validation
release on this robot remain to be collected. No learned VLA was evaluated.

The robot originates from the existing upstream source; the RoboBenchMart MIT
notice is retained in `robots/fetch/LICENSE.RoboBenchMart`. Machine-readable
source and asset hashes live in `robots/fetch/reference.json`.
The current [BIBLE](../../BIBLE.md) and [41-item checklist](../../DATASET_CHECKLIST.md)
are authoritative; older 16-item checks remain historical.

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
MP4 is H.264; reference/compact settings and measured pixel-error fallback are
recorded in export metadata. Decoding returns RGB. `source_h5_metadata.json` retains
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

## Motion restoration, profile v4 (2026-09-26)

The published `317211ed` door-opening, inspection and closing sequence is restored.
The travel arm follows the robot rest keyframe, with torso 0.20 m and wrist flex
1.7 rad. No custom elbow fold, additional handle preparation pose or can detour
is used. Head tracking follows the current stage continuously. All actions remain
recorded physical commands; simulator joint coordinates are never rewritten.

Success requires a revealed target compartment, >=0.02 m horizontal can
displacement, and no search-rule violation. Discovery alone, vertical settling,
an insufficient nudge, repeated search, or a missed required home visit cannot
produce success. The checker does not require a final return.

The restored door sequence contains reverse base motion. Raw D5 metrics must be
reported; no D5 exception has been approved for this profile. Fresh fixed-pool
collection and human E5 review are required; previous pilots do not qualify v4.

Profile v5 restores the published reverse-capable approach to the nearby closing
dock inside door work. The v4 forward-only override caused turns exceeding pi
in 3/37 successful pilot episodes. Paired probes at those exact recorded states
reduced total rotation on the closing-dock leg from 5.52–5.57 rad to 1.06–1.42 rad.
This does not add a waypoint or parking offset. Free-floor/home navigation remains
forward, and the direct home leg can still turn more overall than the published
south-waypoint route. The D5 conflict of published door work remains explicit;
no owner exception is assumed. The v4 pilot and its review remain historical.

## Restored profile v5 pilot (2026-09-26)

Fixed pool `cabinet-restore-pilot-002`: 37/40 source successes; all three prior
long closing-dock turns are removed. D2 roll range, D4 individual turns, even
gripper switching and termination at the can pass in all 37 successes.
D6 pause metrics fail in one of the 37 successful episodes.
The episode still has the published door-work reversals: D5 is not qualified.
Centered home parking also occludes the mark, so E1 fails; neither a parking
offset nor an owner exception has been silently introduced. The new E5 review
and unresolved checklist items must be addressed before mass collection.

## Profile v7 follow-up (2026-09-27)

Empty compartments are inspected with the arm left out. Closing starts from the
current posture, without an intermediate rest fold or a stowing fallback. The
short checked hand retreat remains. After the door closes, the published lowered
rest and direct home return resume. When the can is found, a checked arm tuck
clears the neighboring open leaf before the can approach; the episode ends at
the successful can nudge without subsequent travel.

On the same four target compartments, all 20 calibration trials passed: opening
.20/.25/.30 m/s with closing .05 m/s, then .30 opening with .075/.10 closing.
Profile v7 initially selected opening .30 m/s and closing .10 m/s based on
physical success; the additional pause audit below supersedes that choice. The full fixed
40-attempt pilot, both physical replays, current-phase camera visibility, raw
D5/D6 and new human E5 review are evaluated separately. No D5 waiver is assumed.
Canonical robot, controller, cameras, action format and recording clock are
unchanged. Old profile reports and recordings remain historical.

## Profile v8 speed correction (2026-09-27)

Opening remains .30 m/s; closing returns to .05 m/s. Faster closing succeeded
physically but failed D6 in 21/40 v7 episodes. In the four-compartment calibration,
.075 closing failed one episode and .10 failed two; .05 passed all four at each
opening speed. Speed selection therefore includes pause quality, not only task
success. A fresh, unchanged 40-seed pool and both physical replays are required
for v8. V7 recordings retain their original code signature and remain historical.

## Profile v9 final speed selection (2026-09-27)

Opening returns to .20 m/s; closing remains .05 m/s. The same-scene 40-attempt
baseline at these speeds succeeds 40/40 and fails D6 on 7800013 and 7800017.
Opening .30 m/s adds pause failures, including 7800019 and 7800024. The
intermediate .25 m/s also newly fails 7800019, so neither acceleration meets
the requirement of no regressions. Faster contact motion is not retained.
The empty-inspection rest removal remains and shortens the workflow without
faster door contact. V7 and v8 records remain historical candidates. A fresh
profile-v9 fixed pool uses the same 40 seeds and both physical replays.

## Final profile v9 fixed-pool result

The development pool `cabinet-final-003`, seeds 7800000–7800039, succeeds
40/40; every success passes native 20 Hz replay and physical held-action 10 Hz
replay. It covers all four target compartments. Every empty inspection goes
to closing without rest, and all episodes finish at the successfully nudged can.
The inclusive 0.02 m nudge boundary is tested for all four compartments. Sixteen
additional controlled-state checks on the actual environment confirm that
discovery, vertical settling and a 0.019 m push fail; a 0.025 m horizontal push
succeeds without a home return. These state edits are isolated test setup, never
demonstrations. Repeated inspection and skipped-home counterexamples remain.

All 40 final command traces match the .20/.05 speed calibration at the recorded
float32 precision, and the robot states match exactly. D6 fails on 7800013 and
7800017. The .25/.05 candidate newly fails 7800019; .30/.05 fails four episodes,
and .30/.10 fails 21. No accelerated contact speed is retained.

The all-state three-camera audit finds minimum per-episode visibility 88.38%;
centered parking occludes the home mark, so E1 remains unqualified. All 111 door
grasps share the published negative-elbow/positive-wrist branch and keep the
hand camera above the gripper. Roll limits, individual turns and gripper timing
pass. Published door work still reverses in 40/40 episodes, with more than two
base direction changes in 36/40: D5 is not qualified and no waiver is assumed.

Ten complete three-camera episodes and 30 storyboards are ready for owner E5.
All 20 Hz frames of the ten full episodes are retained in original-resolution
camera videos; every decoded frame was compared with its clean render. The
combined viewer is 10 Hz and removes no pauses. This is a development pilot,
not a qualified mass dataset or a new full LeRobot export.

## Profile v10: canonical rest arm, lowered torso

All seven arm joints use the named Fetch rest keyframe, including wrist flex
2.077 rad. Only the torso is lowered to 0.20 m. Door handling and terminal
nudge rules are unchanged. New fixed-pool and replay results are required;
profile-v9 results are not evidence for this posture.

## Profile v10 fixed-pool verification

The unchanged 40-seed pool 7800000–7800039 passed source, native 20 Hz and true
held-action 10 Hz replay: 40/40 at each stage. All four compartments occur.
Across 151 complete rest transitions, final arm error from canonical Fetch rest
is below 0.01 rad (wrist flex 2.0716–2.0790 rad around the 2.077 target).
49,079 recorded control-state checks over complete rest transitions and free
transit found no robot/model collisions. Deliberate gripper/door contact phases
are reported separately, not treated as free-transit collisions.

All 40 successful episodes pass the pause check. Minimum per-episode target
visibility is 89.54%, below E1; the inherited door motions still violate D5.
These remain open requirements, not owner-approved exceptions. Ten complete
three-camera reviews plus 30 storyboards are prepared; owner E5 is pending.
This development pilot does not authorize mass collection or dataset publication.

D3 is also open: 105 of 111 door grasps share the dominant elbow/wrist sign
combination (94.59%, below the 95% criterion). This is measured at the actual
finger-closing frames; canonical transport rest does not by itself constrain
the IK branch used for a door grasp.

## Profile v12: physical closure without environment assistance

The environment no longer writes hinge positions or velocities after reset and
observations retain actual angles, including small drift. Every leaf must remain
within 0.01 rad for ten released control steps. The panel push targets zero:
0.20 m/s through the main stroke, 0.05 m/s over the final 0.10 rad, followed by
30 cm hand withdrawal. At most two handle corrections are permitted.

The fixed 40-scene comparison passed 40/40 with the selected speed versus 37/40
at 0.05 m/s under the same physical-only checker. All successful records passed
native 20 Hz and true held-action 10 Hz replay. All recorded closing states were
checked for unexpected robot/furniture contacts; none were found. Full rest and
transit clearance, camera audits, ten complete three-camera episodes and the
remaining storyboards accompany this new profile. Earlier profiles retain their
original code bindings. Owner E5 and the remaining dataset requirements still
gate mass collection; this is a development result.

## Profile v14 checklist fixes (2026-09-28)

D3: in profile 13 the door grasp closed on the published negative-elbow/positive-wrist
branch at 146 of 154 grasps (94.8 %). All eight exceptions were first-door pre-grasp
screws that did not jam at shoulder_lift and kept a positive elbow. The pre-grasp and
grasp stroke now require that branch at the plan endpoint only (the transit posture
itself has a positive elbow). A screw ending on the other branch is re-planned with
shoulder_lift held, which is the solver's own unjam variant that the other 95 % of
grasps already take; otherwise the plan is refused. The robot, its limits and the
door motions are unchanged.

D5: the free-floor home drive stops within 0.03 m of the sampled mark centre (the
home disk is 0.20 m) instead of reversing 1–2 cm at the end of the TOPP braking tail.
Door, closing and touch docks keep full precision; the published reverse inside door
work is unchanged and remains an open owner decision. E1 is unchanged: centered
parking still covers the mark (no parking offset or south waypoint is restored).

Paired development pool 7800000–7800039 against profile 13 (`cabinet-final-001`):
source 40/40 and held 40/40 in both. Door-grasp branch 111/111 (was 105/111). Median
free-navigation direction changes 6 → 4 and reverse frames 191 → 136; D6 maximum
identical-pair fraction 4.73 % → 3.08 %, longest pause 15 → 6 frames. The eight
profile-13 wrong-branch seeds (six of them in this pool, plus 7860000 and 7860003)
all succeed on the published branch with both physical replays.
