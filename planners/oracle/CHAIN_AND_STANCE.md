# IK chain from the hardest pose, and stance choice by scan

These are two tricks taken from BIMBO_EvoX and b1k_rl_guys. They apply to SeasonDish and
Depth Recall v1. **Defaults since 2026-10-17 (the owner's call on the measured gain):**
- The IK chain is on (`MIKASA_IK_CHAIN=0` or `--no-ik-chain` turns it off).
- Depth Recall's scanned dock offset is on (`MIKASA_STANCE_SCAN=0` / `--no-stance-scan`).
- SeasonDish's stance scan stays off: it never moved the stance.

| Flag | Environment | What it does |
|---|---|---|
| `--[no-]ik-chain` | `MIKASA_IK_CHAIN` (default 1) | IK chain from the hardest pose ([`ik_chain.py`](ik_chain.py)) |
| `--[no-]stance-scan` | `MIKASA_STANCE_SCAN` (Depth Recall default 1, SeasonDish 0) | Stance choice by a reach scan ([`stance_scan.py`](stance_scan.py)). SeasonDish's needs the cuRobo server. |
| — | `MIKASA_GRASP_ROUTE_CHECK` (default 1, with the chain) | SeasonDish: prefer grasps whose whole transfer previews (below) |

Both switches exist on `scripts/collection_episode.py` and on the Depth Recall planner's
command line. `MIKASA_DOCK_ERR=sx,sy,syaw` is a measurement hook in Depth
Recall's `dock_at`: it moves the dock target by an error the planner is never told about.

All numbers below are measured on the H100 server with 10 seeds, paired. They are the only
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

**SeasonDish on fresh seeds.** Seeds 1010–1029 were never used while the chain was built.
Streams 0–3:

| | base | `--ik-chain` |
|---|---|---|
| stream 0 | 15/20 | 17/20 |
| stream 1 | 15/20 | 16/20 |
| stream 2 | 17/20 | 18/20 |
| stream 3 | 15/20 | 17/20 |
| **all** | **62/80** | **68/80** |
| grasp reached | 70/80 | 77/80 |

Per seed, base → chain over streams 0–3:

| | seeds |
|---|---|
| chain wins (11 pairs) | 1013 0010→1111, 1014 1110→1111, 1018 0111→1111, 1022 0000→0001, 1025 0011→1111, 1026 1000→1111 |
| chain loses (5 pairs), all after the grasp | 1020 1111→1010 (carry path refused); 1021 1111→0010 (no loaded route with a complete pour, or no upright hover) |
| both arms lose | 1024 on every stream (grasp refused) |

Seed 1021's chained grasp branch passes the grasp-time carry check (`carry_or_direct`), yet
the loaded route after the lift fails. A chain that also ranked by the full route would need
the `complete` rule's previews at grasp time.

**All 130 pairs** (dev seeds × 5 streams, plus fresh seeds × 4 streams): base 83/130,
chain 105/130. The chain wins 27 pairs and loses 5.

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
- 20 more seeds at 8 cm (1010–1029): the shipped dock (`--ik-chain`) scored 13/20, the
  scanned −0.04 dock (`--ik-chain --stance-scan`) 20/20.
  - **Over 30 seeds: 22/30 vs 30/30.**
  - Every shipped-dock loss is "dock at the row" with the draw 5.8–7.6 cm east. The dock
    drive's screw puts `forearm_roll_link` into the cabinet's right door
    (`hingerightdoor`, seed 1013).
  - That limit is the scene, which the kinematic scan cannot see. The scan's own east
    limit is the reach to the far pad. Both say the shipped dock sits too far east, and
    four centimetres west clears both.

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

The scene check picked −0.04 in all 40 episodes of the stance arm: 30 in state mode (no
error, 5 cm, 8 cm) and 10 in pointcloud mode.

In pointcloud mode (seeds 1000–1009, no injected error, re-dock at the measured row on 8 of
10 seeds) the shipped planner and `--ik-chain --stance-scan` both scored 10/10 with 0
refused postures, at 3843 vs 3837 mean steps.

## 3. Seed 1021: the grasp must leave room for the pour

With the chain, SeasonDish seed 1021 lost on 3 of 4 streams; the base won it on all 4. The
chained grasp is a few millimetres off the screw's and on the same branch.
- The difference shows only after the lift: every hover's 165° wrist pour was refused by
  the episode's roll window, in both directions (`roll_window+`, `roll_window-` in the now
  logged reasons).
- The grasp-time check under `carry_or_direct` asks only for a carry endpoint, and that
  endpoint was fine.

**Fix (`season_dish_paths.ROUTE_CHECK_MAX`).** With the chain on, the first three grasps
that pass the carry rule also preview the whole transfer:
- the `complete` rule's loaded drive, hover and wrist pour, with the roll history of the
  grasp, lift and fold — or a direct pour from the station;
- a grasp that fails the preview is kept as the fallback, used only when nothing passes.

On the pairs the chain had lost:

| seed, stream | result |
|---|---|
| 1021 r1 | success, grasp request 1 |
| 1021 r3 | success, grasp request 2 |
| 1005 r3 | success |
| 1007 r4 | success |
| 1021 r0 | still lost: the preview passes, the physical route after the lift does not |

**Full measurement** (`b698cd2`): the same 130 (seed, stream) pairs, the chain with the
route check, which is the new default.

| | base | chain (`79831ec`/`e9c789f`) | chain + route check |
|---|---|---|---|
| dev seeds 1000–1009 × 5 streams | 21/50 | 37/50 | 42/50 |
| fresh seeds 1010–1029 × 4 streams | 62/80 | 68/80 | 69/80 |
| **all 130** | **83** | **105** | **111** |

- **Against the chain alone:** 9 pairs won, 3 lost.
  - Won: 1000 r4, 1002 r4, 1005 r2–r4, 1007 r4, 1021 r1/r3, 1022 r2.
  - Lost: 1000 r0, 1020 r2, 1026 r2.
- **Against the base:** 33 pairs won, 5 lost.
  - 1000 r0: the loaded drive puts the forearm into the left wall.
  - 1020 r1–r3: "upright compact carry path refused".
  - 1021 r0.
- **Seed 1021:** 3 of 4 streams now (base 4/4, chain alone 1/4).
- **Planning time:** median 33.5 s, p90 64 s, max 117 s of the 120 s budget; no episode
  exhausted it.
- **Steps:** successful episodes average 784, longest 1060 of 1100.
- **Tests:** 291 passed; the one failure is the known torch pin on that server.

**Tried and withdrawn: refining fast wrist steps in the upright fold** (`3802c14`).
- What it did: split a >0.2 rad wrist step into 8 sub-steps and kept it if continuous.
- Result: 102/130 against 105/130 for the chain alone (1 pair won, 4 lost).
- Why it lost: the sub-stepped segments are slow, and one episode ran out of its
  1100 steps.

## 4. Collection trial: the contact leg moved the base (fixed in `827cec6`)

The numbers above count successes only. The first trial through the collection pipeline
(`utils.collection.campaign --profile season_dish`, then the 2026-09-28 acceptance gates,
export and `utils.collection.checklist`) found what they could not.
- Setup: 20 fresh development seeds 7790000–7790019, one stream.
- **Chain as of `02eac4a`:** 18/20 successes, but 16 of the 18 rejected on D5.
  - D5 allows at most 2 base direction changes per episode.
  - These had 4–6 each, every one inside the chained contact leg.
- **Cause.** The knots keep the base of the grasp-planning step. The settle and the
  descent move the base by a fraction of a millimetre before the contact leg starts.
  - TOPP turned that offset into a creeping base correction (every command under 0.0014)
    whose sign flipped.
  - The straight contact path never moves the base.
- **Fix.** `chain_contact_path` keeps the root joints where the leg starts, with zero
  velocity.

Same 20 seeds, same gates:

| | successes | accepted | D5 failures |
|---|---|---|---|
| before the chain (`d29f363` + the torch pin `02eac4a`) | 15/20 | 15 | 0 |
| chain, `02eac4a` | 18/20 | 2 | 16 |
| chain, `827cec6` | 19/20 | 19 | 0 |

`827cec6` also passes:
- the payload and upper clearance audits (0 hit frames), transfer tilt under 15° (max 5.7°)
  and the wrist-only pour;
- D2, D4, D6 and D7, one grasp sign combination (19/19), E1 visibility ≥ 0.984;
- export readback of 19 episodes (7375 frames).

Tests: 319 passed, 4 skipped.

**60 more fresh seeds** (7790020–7790079, one stream, through the held 10 Hz replay with
the motion, clearance and view gates, no RGB):

| | successes | accepted |
|---|---|---|
| before the chain | 52/60 | 52 |
| chain, `827cec6` | 47/60 | 47 |

- No episode was rejected in either arm, so the base fix holds.
- The success gap is the other way round from the first 20 seeds:
  - the chain won 5 seeds and lost 10;
  - 6 of the 10 losses came after the grasp ("no checked loaded route with a complete
    bowl pour" 5, "no upright hover with a complete wrist-only pour" 1);
  - 2 were the grasp continuation;
  - 2 left no failure message.
- One stream cannot separate the chain from mplib's RNG on a single seed (section 1).

Over all fresh seeds measured so far:

| fresh seeds | before the chain | chain + route check |
|---|---|---|
| 1010–1029 × 4 streams | 62/80 | 69/80 |
| 1030–1049 × 4 streams | 65/80 | 66/80 |
| 7790000–7790079 × 1 stream (with the base fix) | 67/80 | 66/80 |
| **total** | **194/240** | **201/240** |

On fresh seeds, then, the chain is close to neutral: +7 of 240. The large gain (21/50 →
42/50) is on development seeds 1000–1009, the seeds where the contact refusals were
diagnosed.
