# Same Drawer follow-up, profile v7

Reference: published `eb20cc64`; preserved user selections 25–29, 31–33, 36–39
and common 43. Four drawers, including the bottom drawer, are sampled.
The actual initial heading is retained. Handle approaches remain diagonal from
its right, with the selected lowered torso/hand orientation for low handles and
a reserved wrist range for the later apple grasp.

The first closing stroke keeps the fingers open and requires measured hand contact.
The hand advances along the drawer slider with the base stopped. It stops on
measured drawer closure; it does not force the elbow to a fully straight limit.
Reopening closes the fingers, requires opposing contact forces on a link of the
selected drawer, and pulls with the arm while the base stays stopped. An aperture
alone is insufficient evidence of a grip. Handle-contact thresholds remain unchanged. The top initial handle approach uses
a 0.20 m torso target; the grasp orientation and diagonal are preserved. All
neighboring fixtures and upper cabinet handles remain in the planning world.

After initial closing, the hand withdraws along its approach axis before the arm
moves to the robot's canonical rest keyframe (joint names, including torso).
The selected withdrawal is extended only as needed to clear the countertop along
the rest trajectory: 24 cm initially, up to three extra 4 cm straight segments.
When a straight interpolation intersects furniture, the planner checks monotone
joint curves to the same rest endpoint, including every timed sample. No joint
reverses to an intermediate preset pose. The aisle corner is 20 cm farther from
the counter to clear the left wall with the restored arm envelope. It does not
introduce a special travel pose or move the base backwards.

Apple grasp geometry follows the published planner, with positive elbow/wrist
flex branches selected consistently. After release, item 38 now clears the apple
with a straight 10 cm upward stroke at unchanged hand orientation. If unavailable,
it checks 14/18 cm withdrawals along the negative finger axis. Every interpolated
path is collision checked, including neighboring fixtures. No joint-space
substitute is used for that clearance. The same rest endpoint follows; a blocked
fold can extend finger-axis clearance by two bounded 8 cm increments.

The checker latches a violation if a released, placed apple leaves the plate or
is regrasped. Returning it later does not repair the episode. Final success
requires the apple still on the plate, stationary and released, as well as the
correct reopened drawer. Retention flags are serialized for physical replay. The apple is lowered vertically to 3.5 cm above the plate and both hand
and apple must settle before release. Travel uses the selected aisle route.
The head follows the current stage: look at the plate after lifting the apple;
look at the entire drawer column only after folding for the return.

The DSFetch model, keyframe, controller and cameras are unchanged. Actions remain
13D and proprioception 12D, recorded at 20 Hz and physically replayed at held 10 Hz.
New profile v7 attempts and their native/held-action/RGB checks are kept separately.
Historical profile recordings retain their original hashes and qualification.
A fixed 40-attempt pilot and new human E5 review are required before mass collection.

Earlier setup notes and measured qualification remain in the unchanged
[pre-v4 documentation](history/SAME_DRAWER_BEFORE_V4.md). Those results do not
qualify the restored movements.

Profile v5 enforces pre-release settling: four consecutive 20 Hz intervals with
measured apple displacement speed < 0.01 m/s, rotation speed < 0.5 rad/s, and
maximum arm joint velocity < 0.05 rad/s. Wait at most 20 control steps; a timeout
is an explicit refusal with the gripper closed. The window ends on an even step.
The former v4 six-step callback did not enforce its stop condition and its 20/40
pilot is historical. Physical replay showed contact solver instantaneous apple
velocity disagreeing with actual pose motion; v5 records both but uses physical
pose differences for this pre-release check. Profile v7 additionally enforces
continuous apple retention through the episode end.

## Historical profile v5 pilot (2026-09-26)

Retrospective correction: seeds 7900002 and 7900017 knocked the apple off after
placement. The old latched checker reported success anyway. The 20 reported
successes below therefore include two false positives; only 18 retained the apple.
The recordings and historical reports have not been rewritten.

Fixed pool `drawer-restore-pilot-002`: 20/40 source successes, all 20
native replays, held-action replays and three-camera RGB recordings pass.
No failed seed was replaced. Roll-range and individual-turn checks pass for all
successful episodes. New E5 packages contain ten complete three-camera episodes
and storyboards for the remaining successes; human acceptance is pending.
D6 is not fully qualified: 3 successful episodes exceed the pause
fraction limit; no additional exception is assumed. H2 fails: the 50% source
success rate is below 60%. All 20 successful releases pass the measured four-interval
settling audit, initial pushes keep the fingers open and reopening keeps the base
stopped with a confirmed handle grasp. No mass-collection qualification is claimed.

The fixed pilot attempted the top drawer eight times without a complete success.
A separate, previously known diagnostic seed 512 passes profile v5 with both
physical replays and RGB; it supplements the E5 package and never replaces a
fixed-pool failure. Apple grasp signs also fail D3 consistency: (+,-)8, (+,+)7,
(-,+)5 of 20 successful episodes. The restored published grasp is preserved;
no D3 exception or full qualification is implied.

## Profile v7 follow-up (2026-09-27)

Fixed development pool 7900000–7900039: 32/40 source successes and 32/32
successes in each physical replay (native 20 Hz and held 10 Hz). All 32 keep
the apple on the plate, push with open fingers and a stopped base, and reopen
with a stopped base. Failed attempts remain in the denominator. D6 pause
qualification and the new human E5 review remain required; no new exception is
assumed. This pilot does not authorize mass collection.

## Follow-up profile v7 fixed-pool result

The development pool `drawer-final-001`, seeds 7900000–7900039, contains 40
attempts: 32 successes, seven planning refusals and one missed placement.
Failed seeds: 7900001, 7900004, 7900022, 7900025, 7900029, 7900035, 7900036,
7900037. Every success passed native 20 Hz and physical held-action 10 Hz replay.
Every successful first push keeps fingers open and base stopped; reopening uses
a confirmed grip and a stopped base. All 32 apples remain on the plate after
placement until episode end. Four consecutive physical settling checks pass
before release. D6 still fails on 7900000, 7900018, 7900021; no waiver is assumed.

All three cameras were audited at every 10 Hz state: minimum per-episode target
visibility is 99.24%; apple and reopening grasps consistently use positive
elbow/wrist branches with the camera above the hand. A full-fixture collision
audit over 39,735 recorded 20 Hz states in all 40 attempts found no robot
intersection with upper cabinets or microwave meshes, including cabinet
handles. This does not claim inspection of every 100 Hz physics substep.

The reported apple-strike seeds 7900002 and 7900017 were inspected at 0.1-second
spacing across release, clearance and stow (48 and 49 triptychs). Final apple
distance from plate center is 3.01 and 2.99 mm, versus 29.5 and 44.5 cm in the
previous recordings. Six of eight prior upper-drawer failures now succeed; the
remaining two fail at apple grasp or return, not initial upper-handle approach.
Ten complete three-camera episodes and 22 storyboards are ready for owner E5.
Mass collection and fresh full LeRobot export remain gated on qualification.

## Profile v8: continuous aisle return and drawer balance

The arm retains canonical named rest. The return uses a quintic rounded corner
through the existing aisle, with forward/yaw commands on the 10 Hz action clock,
bounded acceleration and full-robot collision checks. The intermediate corner
is not a stopping goal. No stop-go fallback is used when the route is refused.
Apple withdrawal and irreversible retention checking remain active.

Use `python -m utils.collection.drawer_balance plan --output seeds.json
--start-seed N --per-drawer 10` to preselect 40 attempts solely by reset drawer.
Pass `--seed-manifest seeds.json` to campaign instead of a contiguous range.
The all-attempt report groups failures by drawer and last stage. For training,
accepted quality-gated data has a quota of 250 per drawer (0 bottom, 3 top);
`next_seeds` resumes chronological banks without retrying failures or filling
quotas with successes that failed replay/quality. Equal successful quotas do
not imply equal SR. Mass collection still requires the existing quality gates
and owner E5 acceptance; this profile alone does not open that gate.

## Profile v8 balanced pilot

A prospective reset-only manifest selected exactly 10 attempts for each drawer,
using seeds from 8000000–8000045. Source successes, from top to bottom, are
6/10, 8/10, 5/10 and 5/10: 24/40 overall. All 24 successes passed both native
20 Hz and physically held-action 10 Hz replay. Every refusal remains in SR.
The old planner on exactly the same scenes also scores 24/40, distributed
6/10, 7/10, 6/10 and 5/10. The new route succeeds on 8000028 and loses 8000016
at the final handle grasp; failures before the return remain manipulation issues.

For 23 paired successes, stops between moving return segments fall from 42 to
zero. Mean return duration, including final alignment, is 16.030 s versus
15.874 s previously: smoother travel is verified; faster overall return is not.
All executed rounded sections meet the 0.35 m/s² and 1.5 rad/s² command limits.
No upper-cabinet/handle model intersections occur in 35,283 recorded 20 Hz
states across all 40 attempts. The eight prior failures were repeated separately;
7900035 now succeeds in both replays, and seven still fail.

All 24 successes pass D6, and minimum target visibility is 99.25%. The E5 package
contains three full bottom-drawer, three third-drawer, two second-drawer and two
top-drawer episodes, plus 14 storyboards. Owner review is pending.

A separate prospective bank fixes 750 candidates per drawer, classified from
9000000–9003100 with prior run/validation seeds excluded. It is a seed bank,
not collected training data. Training quotas remain 250 externally accepted
records per drawer; zero are accepted from this pilot pending the quality gate
and owner review. Rejections do not get silently retried or removed from SR.

Including the required adversarial repeats, this exact code snapshot has
25 successes in 48 physical oracle attempts (52.08%). Top-to-bottom totals are
7/12, 8/10, 5/13 and 5/13. Both denominators are retained: the balanced pilot is
60%, while H2 is not closed over all attempts of this snapshot. No failed seed
or targeted repeat is removed to authorize collection.
