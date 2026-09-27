# Continuous bottom-drawer contact (profile 14)

The first closure of the bottom drawer goes from the measured rest configuration
to freshly solved handle contact in one continuous arm/torso curve. The former
25 cm standoff is not executed. Shoulder lift, elbow and torso progress are
monotone; shoulder pan retains the excursion needed to clear the counter. There
is no fallback to the old intermediate pose. Fingers remain open and the wheel
base stays still during contact approach and push.

`same_drawer_curve.PROGRESS` contains named, normalized cubic progress
coefficients fitted once to a successful control approach. Runtime endpoints
come from the current joints and current scene IK, never from a seed lookup.
Both geometric and retimed paths must satisfy the original URDF limits,
collection roll windows and full collision model. Intentional contact is limited
to the gripper against the selected drawer's low front/handle; contact on the
upper edge is rejected. A moving-drawer model checks the subsequent push before
the approach starts. This modifies only planning geometry, not simulator state.

The bottom push ends at the physical closed position without the previous 4 cm
overtravel. The existing 5 mm checker tolerance is unchanged. Canonical rest,
confirmed reopening grasp, apple release clearance (item 38), irreversible plate
retention and the smooth return route remain in force.

The fixed paired pilot has 24/40 successes in both control and candidate, with
all 24 candidate successes passing native20 and true held10 replay. Successes
from bottom to top are 5/10, 5/10, 8/10 and 6/10. Eight of ten bottom approaches
complete their first closure. In those eight, elbow reversal is removed and
there are no interior approach stops. Closure takes 0.75 s longer on average
(individual changes range from -0.3 to +1.2 s). Later apple failures remain open.

A 15 mm shallower contact endpoint failed all ten trials and was rejected.
The old `experiment_selection` profile field records the previous distance
experiment; the current bottom-closure choice is specified by
`motion_parameters.closing_approach` and `bottom_contact_path`.

Private evidence is in `bounded-continuation-2026-09-27`: paired metrics, all
attempts including failed prototypes, source snapshots, collision audits and
three-camera review packages. This pilot is not full checklist qualification or
owner E5 acceptance. Mass collection remains gated on those requirements.

One recorded post-release state (seed 8000011, step 669) has a geometric
finger/apple overlap during the unchanged fold. An exact action replay matched
every joint state and measured zero apple/finger impulse at every 100 Hz physics
substep after release; apple retention remained valid. The geometric audit flag
is retained rather than relabelled as a collision-free recording. Full local
collection tests: 138 passed, 6 skipped.
