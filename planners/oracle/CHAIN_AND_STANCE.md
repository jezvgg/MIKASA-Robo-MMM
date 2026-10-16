# IK chain from the hardest pose, and stance choice by scan

These are two tricks taken from BIMBO_EvoX and b1k_rl_guys. They apply to SeasonDish and
Depth Recall v1. Both are opt-in: the pinned profiles are unchanged.

| Flag | Environment | What it does |
|---|---|---|
| `--ik-chain` | `MIKASA_IK_CHAIN=1` | IK chain from the hardest pose ([`ik_chain.py`](ik_chain.py)) |
| `--stance-scan` | `MIKASA_STANCE_SCAN=1` | Stance choice by a reach scan ([`stance_scan.py`](stance_scan.py)). Needs the cuRobo server. |

Both flags exist on `scripts/collection_episode.py`. On the Depth Recall planner's command
line only `--ik-chain` exists. `MIKASA_DOCK_ERR=sx,sy,syaw` is a measurement hook in Depth
Recall's `dock_at`: it moves the dock target by an error the planner is never told about.

All numbers below are measured on gangway with 10 seeds, paired. They are the only
numbers in this file.

## 1. IK chain from the hardest pose

### What the planners did before

The planners solve the *easy* pose first and only then try the stroke to the contact. The
easy pose is a standoff or a pre-grasp, solved from a template or from the current
posture.

On SeasonDish this caused contact refusals:
- The standoff branch was picked without ever asking IK about the contact.
- The contact screw then put `wrist_flex_link` into the counter in its last centimetre.
- Example: seed 1005 lost 3 of its 7 candidates this way.

### What the chain does

1. Solve the contact pose first (mplib, collision-checked).
2. Keep at most 6 distinct branches (0.25 rad apart).
3. Walk each branch back to the easy pose in 2 cm steps. Each step is seeded with the
   previous solution, so CLIK's first start keeps the branch.
4. A step that moves a joint more than 0.6 rad breaks the chain.

This is BIMBO task 40's `chain()` and b1k's `plan_descent`.

Jumps are measured on unwrapped angles with no extra wrapping. A roll that stays far
after `unwrap_toward` is a real turn of the joint.

### SeasonDish

- **Candidates** (`season_dish_candidates.candidates`): every request first solves its
  contact and walks back to its standoff. Requests with no chain are asked the old way,
  after the chained ones.
- **Contact leg** (`season_dish_paths.chain_contact_path`): follows the chain's knots,
  with TOPP timing and the dense `path_clear` check.
- **Why the contact leg changed:** a screw to the same contact pose may drift along the
  7-DoF null space and leave the branch. On seed 1005 the contact IK was collision-free,
  yet the screw to the very same pose hit the counter.
- **At execution** the leg is re-planned the same way from the physical state.

### Depth Recall

The chain only re-ranks `ready_by_line`'s IK goals:
- **Counter grasp:** the grasp is solved before its pre-grasp, per torso, lift and depth
  rung.
  - A rung with no chain is skipped.
  - A torso with no chain is not even lowered to.
  - If that whole ladder finds nothing, the plain ladder runs as before.
- **Shelf entry:** the row place is solved at the work stance and walked up to the entry.
  The ready goals are then ranked nearest to the chain's entry end.

### Results

SeasonDish, seeds 1000–1009, `scripts/collection_episode.py` (the pinned collection profile):

| | baseline (mplib, 10-16) | `--ik-chain` (`79831ec`) |
|---|---|---|
| success | **6/10** | **10/10** |
| seeds lost | 1002 (tilt 44.8° in the lift), 1005 and 1006 (contact screw into the counter, no continuation), 1008 (closed on nothing) | none |
| accepted candidate | — | from the contact chain in 10/10 |
| steps, mean over the 6 seeds both solve | 894 | 886 |
| planning seconds, mean over the same 6 | 33.6 | 35.0 |

The baseline is identical in two independent runs (`cu3_c229f22`, `ag3_88283f3`). Extra IK
calls shift mplib's random starts, so a single seed can change for that reason alone.
This does not account for the result: in all 10 seeds the accepted grasp is a chained one.

Depth Recall, seeds 1000–1009, state mode:

| | base | `--ik-chain` |
|---|---|---|
| nominal dock | 10/10, 0 refused ready postures | 10/10, 0 refused, steps +1 % (3773 vs 3735) |
| unseen dock error 3 cm / 3 cm / 0.05 rad | 10/10, 0 refused | 10/10, 0 refused |

Depth Recall had no refusals left to remove. On these seeds the chain changes the order of
the IK goals and nothing else.
