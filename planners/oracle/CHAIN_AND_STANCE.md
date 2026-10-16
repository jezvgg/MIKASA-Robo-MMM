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

**SeasonDish**, seeds 1000–1009, run with `scripts/collection_episode.py` (the pinned
collection profile).
- Each seed was run under five mplib streams; `MIKASA_PLANNER_RNG_OFFSET=k` moves only
  mplib's random IK starts and RRT samples, the scene is the same.
- Extra IK calls shift that stream for every later plan, so a single stream mixes the
  change with the draw.

| stream | base | `--ik-chain` |
|---|---|---|
| 0 (`cu3_c229f22` / `79831ec`) | 6/10 | 10/10 |
| 1 | 4/10 | 8/10 |
| 2 | 3/10 | 6/10 |
| 3 | 4/10 | 7/10 |
| 4 | 4/10 | 6/10 |
| **all** | **21/50** | **37/50** |
| grasp reached and held to the lift | 34/50 | 46/50 |

- The chain loses none of the 50 (seed, stream) pairs that the base wins.
- Per seed, base → chain, by stream 0–4:
  - 1002: 00000 → 10000
  - 1005: 00000 → 11000
  - 1006: 00000 → 11111
  - 1007: 10010 → 11010
  - 1008: 00000 → 11111
  - 1001: 11001 → 11111
  - 1000, 1003, 1004 and 1009 are equal.
- Mean steps of successful episodes stay within the stream-to-stream spread (stream 0:
  894 base, 892 chain).
- What the chain does not fix comes **after** the grasp:
  - the loaded route to the bowl (1000: forearm into the left wall on the loaded drive,
    the same on streams 1–4;
    1005, 1007: "no checked loaded route with a complete bowl pour");
  - the carry fold (1007: "upright compact carry path refused");
  - seed 1002's grasp, which closes on nothing on four streams of five.

  Those belong to items 3, 4 and 7 of the plan.

**Depth Recall**, seeds 1000–1009, state mode. `MIKASA_DOCK_ERR` is the unseen dock error.

| dock error | base | `--ik-chain` | `--ik-chain --stance-scan` (dx −0.04) |
|---|---|---|---|
| none | 10/10, 0 refused, 3735 steps | 10/10, 0 refused, 3773 steps | 10/10, 0 refused, 3742 steps |
| 3 cm, 3 cm, 0.05 rad | 10/10, 0 refused | 10/10, 0 refused | — |
| 5 cm, 5 cm, 0.087 rad | 10/10, 1 refused | 10/10, 1 refused | 10/10, 0 refused |
| 8 cm, 8 cm, 0.14 rad | — | 9/10, 2 refused | 10/10, 0 refused |

- "Refused" counts ready postures that `ready_by_line` could not plan (`no IK solution` /
  `no line`).
- At 8 cm, seed 1004 (drawn error +7.2 cm east, −7.7 cm, 4.5°): the shipped dock's drive
  itself was refused. The −0.04 dock parked 10.7 cm off and the episode succeeded.
  That is one seed.

## 2. Stance choice by a robust scan

[`stance_scan.py`](stance_scan.py) works in two tiers:

1. **Scan.** For every torso setting, one cuRobo scan covers every candidate stance and
   every perturbation of it, with the goals held fixed.
   - A perturbation counts when every requirement keeps a reachable alternative.
   - Requirements are ANDed; alternatives are ORed; the goals inside one alternative are
     ANDed.
   - Stances are ranked by `cost + 1.5 * (1 - robust)`; those under 0.6 are held in
     reserve.
2. **Scene check.** At most three are then checked against the scene with item 1's
   chain, with the base where that stance puts it.

Other details:
- `curobo_client.scan(torso=t)` scans with the torso held: variant `arm`, goals in
  `torso_lift_link`.
- Every hit is re-checked in float64 (`verify`).

### SeasonDish

[`season_dish_stance.py`](../season_dish_stance.py) builds the candidates and checks:

| | values |
|---|---|
| slide along the counter | −0.15…+0.15 m |
| back-off | 0 / 0.05 / 0.10 / 0.15 m |
| heading offset | 0, ±0.15 rad |
| arrival error grid | along 0 / −0.05 / −0.10 m, across ±0.03 m, heading ±0.1 rad |

The requirement is that some grasp request has its contact, standoff, via-above point and
lift reachable.

On all 10 seeds the shipped stance (back-off 0.10) scored robust = 1.0, so it was kept:
- The scan took 12–37 s (92 s while it shared the server with the Depth Recall scan).
- It changed nothing but the mplib stream (6/10 on that stream).
- This repeats the earlier finding (stance slide 60/75 vs 59/75): what refuses here is
  the scene and the branch, not the reach, and item 1 is what fixes the branch.

### Depth Recall

Everything is rigid relative to the row's x, so the dock offset is ranked once, offline,
by [`scripts/dr_stance_scan.py`](../../scripts/dr_stance_scan.py) (`dx` from the row's x):
- 44 requirements over every row slot and pad;
- the torso is held at the planner's own rungs;
- 916,898 hits were verified and 0 dropped.

| dx (m) | −0.20 | −0.16 | −0.10 | −0.06 | −0.04 | −0.02 | 0.00 (shipped) | +0.02 | +0.04 | +0.10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ±3 cm, ±3 cm, ±0.05 rad | 0.778 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.926 | 0.889 | 0.222 |
| ±5 cm, ±5 cm, ±0.087 rad | 0.667 | 0.741 | 0.889 | 0.963 | **1.000** | 0.963 | 0.889 | 0.889 | 0.667 | 0.333 |

What limits each side of the table:
- **West:** the entry to the deepest row slot.
- **East:** the cross to the far pad, 54 cm west.

The shipped dock sits at the east edge of the 3 cm range. Only −0.04 survives at 5 cm.

`STANCE_DX = (-0.04, -0.06, -0.02, 0.0)` is tried in order, and the first offset the scene
allows wins (`choose_dock_dx`). The scene check is two chains with the base at that dock:
- the front row prop's shelf grasp, walked back to its pre-grasp from the work stance;
- the far pad's place, walked up by the release lift from the dock line.

The first version walked up to the cross 20 cm above the pad with the torso at the counter
rung. That is out of reach, so every offset was refused. Measured on seed 1000: both
branches broke on IK; 10 cm held.

The scene check picked −0.04 in all 30 state-mode episodes of the stance arm (no error,
5 cm and 8 cm). The pointcloud mode is measured separately.
