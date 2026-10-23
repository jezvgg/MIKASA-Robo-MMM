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
