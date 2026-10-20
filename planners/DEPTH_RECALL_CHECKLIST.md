# Depth Recall v1 against DATASET_CHECKLIST

Collection profile `depth_recall` version 4 (`utils/collection/depth_recall_profile.json`).
It sets `env_kwargs.assisted_grasp: true`, the owner's call of 2026-10-18: every task is
collected, trained and evaluated with assisted grasp. All numbers below were measured on
the H100 server on 2026-10-18. Each run went through `utils.collection.campaign`, the pod's
collection audits and, for the RGB runs, `export_lerobot` and `checklist.py`.

## What changed to get here

| Item | Before | Change |
|---|---|---|
| Profile | version 1, stale `slot_x_offsets` / `place_across`; `campaign` refused it | current task geometry; collection planner fields |
| B3, B4, D7, F3 | ramped gripper, near-zero base commands, no 10 Hz pairs | the collection planner by default (`oracle_common.collection_planner_factory`); a binary release |
| C3 | no execution noise | noise on the extraction carry across to the counter column |
| C1 | bit-identical start joints | `utils.initial_pose.randomize_initial_joints` at reset, with the other tasks' ranges |
| D2 | 13/19 | `prefer_low_roll_ik()` (episode roll window) |
| D3 | 91.7 % | 97–98 % after the roll window; no episode-wide grasp branch (it refused the first dock) |
| D6 | 41 % repeated 10 Hz actions, longest run 34 frames | see below |

The D6 changes, in the order they mattered:
1. **The base coasted ~1 cm after every straight drive.** The path follower's refinement
   waits for base, torso and arm to be within 1 cm of the path's end, but it never commands
   the base, so it held one action until its "stuck" exit, 27 frames each time.
   `drive_and_settle` now waits until the base is below 5 mm/s, and the drives end at
   0.01 m/s.
2. **Drives at a constant 0.10 m/s** recorded one action for the whole drive. They now
   follow a sine speed taper (`drive_straight(v_floor=, v_end=)`), peaking at the old speed.
3. **Torso lines.** They now ask 0.4 cm a step instead of 2 cm, as one line instead of 8 cm
   chunks. A torso already at its rung counts as there; a one-knot line used to count as a
   refusal.
4. **Gripper holds and the final wait.** The release is held 6 steps instead of 18, the
   close 10 instead of 12. The closing stillness ends when the task reports success.

## Measured on profile 4

Seeds 7990400–7990439:
- **Successes:** 40/40.
- **Native and held 10 Hz replays:** 40/40.
- **Motion:** D2, D4, D6, action pairs and no odd gripper switch, 40/40 each.
  - D6: identical-action fraction 4.4–4.8 %, longest run 5 frames.
- **D3:** one sign combination at 97.6 % of grasps.
- **The 10-episode RGB run (7990400–7990409):** 10/10 accepted.
  - Export readback: 10 episodes, 17311 frames.
  - E1, target visible: 1.0 in every phase of every episode.
  - `checklist.py`: 17 PASS, 12 PARTIAL, 11 UNKNOWN. The one FAIL is D5 (below).
  - C1: initial joint std 0.003–0.028 on arm, torso and head.

Over profiles 3 and 4 that is 90 fresh seeds with no failure. Tests: 319 passed, 4 skipped.

## Acceptance gates for collection

- **Hard, per episode:**
  - source success;
  - native, held 10 Hz and RGB replays succeed;
  - D2, D4, action pairs, no odd gripper switch, D6 (`pause_check`, no task exception);
  - export readback.
- **Recorded, not gated:**
  - **D5.** Owner decision 2026-10-18, also in the profile as `quality_exceptions.D5`.
    Every transfer drives the base into the cabinet and back out, 13–15 direction changes
    an episode, as for CabinetSearch.
  - **D3.** A task-level fraction.
  - **E1.**
- **Not yet done:** E5 (the owner's look at complete episodes), validation100 (H4) and the
  collection itself (H3).
