# Assisted grasp: the hand passes through cabinet doors (2026-10-23)

## Cause

`utils/mikasa/assisted_grasp.py` builds one SAPIEN drive from the hand (`gripper_link`) to every
graspable object when the environment is made (line 226), and switches it on only while holding.
A SAPIEN drive turns collision between its two bodies off, whatever its stiffness and limits, and
SAPIEN 3.0.3 has no call to turn it back on. `drive_collision_test.py`: a 10 cm box dropped on a
slab rests on it (172 contact steps); with a force-free drive between them it falls through (0
contacts). So the hand never collides with any graspable object, held or not, for the whole episode.
The fingers are other links and still collide.

## Measured on the collected datasets

`hand_overlap_replay.py` replays each episode with the dataset's own profile (every replay matches
the recorded 12D state exactly, max error 0.0) and measures the hand's collision vertices against the
objects' collision shapes. Five episodes per task (0, 200, 400, 600, 800):

| Task (assisted grasp objects) | Hand inside an object |
|---|---|
| SameDrawer (apple, drawer boxes) | never |
| SeasonDish (shaker, bottle) | never |
| DepthRecall (props) | never |
| CabinetSearch (door leaves, can) | every episode: the hand crosses the 1-2 cm door leaf |

CabinetSearch, frames where the hand straddles a leaf (it would have to move >= 20 mm along the leaf
normal to clear it, more than the leaf is thick):

| Episode | Frames | Through the leaf | Deepest |
|---|---|---|---|
| 0 | 1307 | 95 (7.3%) | 57 mm |
| 200 | 445 | 33 (7.4%) | 37 mm |
| 400 | 399 | 11 (2.8%) | 36 mm |
| 600 | 880 | 62 (7.0%) | 56 mm |
| 800 | 437 | 33 (7.6%) | 37 mm |

Episode 0 against its subtask labels: 184-216, 646-677 and 1087-1100 are the last frames of the three
"open" subtasks (BIMBO keeps the hand's world orientation while the leaf turns into it, the handle
held); 280-287 and 744-751 are inside the "close" subtasks, hand empty (the fist push; the deepest
frame, 57 mm, is 283). The can is never entered. The hand camera then sees the cabinet from inside the
leaf: `artifacts/assisted-grasp-check/cabinetsearch_ep0_hand_through_door.png` (frames 200, 212, 280,
283, 286, 1092, 1098; hand, left base, right base).

Not measured: the visual meshes (the dataset videos show the visual leaf, which may be thicker than its
collision box), and jezv's TakeItBack and Veggie, which use the same assisted grasp code in this tree
but were collected with his copy of it.

## The fix and what it changes (2026-10-23, commit b98cee0)

`utils/mikasa/assisted_grasp.py` now starts every object's drive at one shapeless, gravity-free proxy
that one more drive locks to the hand while something is held; the hand is in no drive. The test
(`drive_collision_test.py`): a direct drive, enabled OR disabled, lets the box fall through the slab;
the proxy chain rests it on the slab with the same 172 contact steps as no drive at all.
`MIKASA_AG_LEGACY_DRIVE=1` (or `AssistedGraspConfig.hand_collides=False`) rebuilds the old drives.

**Legacy mode reproduces the collected data:** episode 0 of each task replays with state error 0.0 and
succeeds; CabinetSearch shows the same 95 through-the-leaf frames (57 mm) as before the change.

**The recorded actions under the fixed physics** (5 episodes per task, open loop):

| Task | Replay succeeds | Largest 12D state difference from the dataset | Hand inside an object |
|---|---|---|---|
| SeasonDish | 5/5 | 0.6 mrad | never |
| DepthRecall | 5/5 | 5.8 mrad | never |
| SameDrawer | 5/5 | 1-7 mrad; episode 400: 31 mrad (from frame 250) | never |
| CabinetSearch | 1/5 | 45-386 mrad, from frames 148-185 (the end of the first opening) | never |

So SeasonDish, DepthRecall and SameDrawer stay valid: the same actions do the same thing within a few
mrad and still succeed; only their bit-exact replay now needs `MIKASA_AG_LEGACY_DRIVE=1`.
CabinetSearch's recorded actions no longer work once the hand collides with the leaf: re-collect it.

**The CabinetSearch planner itself under the fixed physics** (`scripts/collection_episode.py
cabinet_search <seed> --assisted-grasp`, the seeds of dataset episodes 0, 200, 400, 600): 4/4
succeed. All seven openings end the same way: `swing done why=stalled` at 1.684-1.713 rad (target
1.75; the hand's own reading says 1.77-1.80), 157-159 controls. The leaf stops against the hand and
the swing keeps commanding for its 60-control stall window (STALL_STEPS) before it gives up, i.e. the
hand presses the leaf for ~3 s per opening. After the release the leaves rest at 1.625-1.666 rad. The
closing pushes (seeds 7880000 and 7880793 close two and one leaves) all close.

## What to change in BIMBO (planners/cabinet_door_bimbo.py)

The cause of the contact: `HAND_TURN = 0` keeps the hand's world orientation while the leaf turns
~100 deg about the hinge (full BIMBO turns the hand with the leaf, out of a Fetch's reach), and the
weld lets the bar rotate freely in the hand, so near the end the leaf face meets the hand body.
1. End the swing at the contact, not after a 60-control stall: stop when the hand touches the leaf
   (contact force hand <-> leaf from the scene) or when the door has not advanced for a few controls,
   and target the angle the hand allows (~1.68 rad) instead of 1.75.
2. Keep the palm off the leaf: let the hand's yaw follow part of the leaf's turn (`HAND_TURN` 0.2-0.3,
   inside the roll window), so the leaf reaches 1.75 without touching. Needs a reach check: the shipped
   reason for 0 is that the full turn (1.0) is out of reach.
3. Re-measure the later stages from the new end state (they were measured from the arc pull's end
   stance, LEVER_END): push from here, tuck, touch-dock drive. The four runs above passed them.
Then validate on 20-40 seeds (success, motion audit, clearance gates, `hand_overlap_replay.py` = 0 on
the new episodes, a look at the hand camera) before re-collecting 1000.

## BIMBO swing fix, validated (2026-10-23, commits 457051f, 5c21a3c, 2512dd2)

The swing now ends when the hand presses the leaf (> 1 N) and the leaf moved < 0.005 rad over the
last 8 controls (`LEAF_CONTACT_N`, `LEAF_STOP_STEPS`, `LEAF_STOP_EPS`). A first version stopped at the
first touch: the face touches the hand from ~1.0 rad and slides along it to ~1.7, so the leaves were
left at ~1.0 rad and 4/4 rounds were lost.

`episode_check.py` (the collection episode, measured every control), fixed assisted grasp:

| Seeds (dataset episodes) | Success | Swings | Swing end | Hand in a leaf, max | Hand on leaf, max force |
|---|---|---|---|---|---|
| 5: eps 0, 200, 400, 600, 800 | 5/5 | 8, all `hand-on-leaf`, 106-108 controls | 1.683-1.709 rad | 0.9 mm | 193-215 N |
| 10: eps 50, 100, 150, 250, 300, 350, 450, 500, 650, 900 | 9/10 | 19: 17 `hand-on-leaf`, 2 `target` | 1.686-1.723 rad | 1.0 mm | 185-216 N |

Before the fix the hand went 36-57 mm into the leaf; before the swing fix every swing pressed the
leaf for its 60-control stall window (157-159 controls). The one miss (seed 7880608, episode 450,
a success in the dataset) is in the close, not the opening: the fist pushed the leaf shut (rad 0.0),
the retreat off the panel found no plan, the closing correction could not re-grasp the handle
(`finger <-> hingerightdoor` in the screw), and the round was booked "the door did not close" -
the failure class of the whole assisted-grasp collection (~20 % of its seeds).

Open: in the second half of every swing the leaf slides along the hand with up to ~200 N (85-100
controls above 20 N per swing). The task succeeds, but no real arm presses like that; a partial
`HAND_TURN` (the hand's yaw following part of the leaf's turn) is the way to remove it.

**Correction to the counts above: one of the 14 "successes" knocks the can over.** Seed 7880257
(episode 200) ends `success` in both runs with the swing fix, but the can is never pushed: while
the arm is readied for the touch (`raise the torso` / `level the hand for the touch`, joint lines,
steps ~780-805) the hand strikes the can and lays it on its side, then every pre-touch plan is
refused and the planner books `MISSED: the cube was not nudged`; the env still reports success. The
collection's can-early gate (cube xy <= 0.01 m and tilt <= 20 deg until the push starts) would
reject it, so the acceptable count is 13/15 (the other 13 push the can 6-8 cm as designed). Before
the fix the same seed pushed the can: the swing now ends earlier, the stance after it differs, and
the readying path that followed reaches the can, which the hand now cannot pass through. To fix
in the planner: check the readying joint lines against the can (or raise and level the hand before
the hand is inside the compartment).

## HAND_TURN 0.25 and the closing push (2026-10-23, all with the fixed assisted grasp)

`MIKASA_BD_HAND_TURN=0.25` (the hand's yaw follows a quarter of the leaf's turn), seeds of episodes
0, 200, 400, 600, 800:

| | HAND_TURN 0 (2512dd2) | HAND_TURN 0.25 |
|---|---|---|
| success | 5/5 (one knocks the can: 4 acceptable) | 3/5, all acceptable |
| swing end | 1.68-1.71 rad, all `hand-on-leaf` | 1.73-1.75 rad, 3 of 5 on the target |
| hand on leaf, max | 193-215 N | 67-112 N |
| controls above 20 N per run | 94-279 | 14-60 |

Both misses (seeds 7880000, 7880793; left leaves) are in the close: the fist pushes the leaf shut
(0.0 rad), the arm cannot withdraw (`joint limit at index [7]`, no IK), and whatever withdrawal
follows pulls the shut leaf open again (0.38-0.71 rad), then the correction fails. No weld forms (no
assisted-grasp event between the push and the withdrawal). Tried and reverted, none fixed a miss:
an RRT retreat with the cabinet out of the planning world (swept the hand into the leaf, 0.44-0.50),
a 0.20 m and a 0.35 m straight base back-off with the arm held (0.38-0.71), withdrawing with the fist
and opening when away (0.44, 0.71), and `MIKASA_CLOSE_R_PUSH=0.25` (no effect: the push from here does
not use that radius). The same miss class appeared once with HAND_TURN 0 (seed 7880608). The planner
here is back to the validated 2512dd2 closing code.

## For the re-collection (owner 2026-10-23: no HAND_TURN, new HF repo)

`utils/collection/cabinet_search_profile.json` is profile 17: the profile-16 settings on this code
(proxy-held assisted grasp, the swing that ends on the stall, HAND_TURN 0). The branch sits on
`train/vla` (main code); the policy branch can merge it afterwards. Profile-16 data (the published
`cabinetsearch-ag-1000ep`) replays bit-exactly only with `MIKASA_AG_LEGACY_DRIVE=1`; so do SameDrawer,
SeasonDish and DepthRecall, whose recorded actions also succeed under the fix (15/15, <= 31 mrad).
