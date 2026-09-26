# Same Drawer restoration, profile v4

Reference: published `eb20cc64`; preserved user selections 25–29, 31–33, 36–37,
39 and common 43. Four drawers, including the bottom drawer, are sampled.
The actual initial heading is retained. Handle approaches remain diagonal from
its right, with the selected lowered torso/hand orientation for low handles and
a reserved wrist range for the later apple grasp.

The first closing stroke keeps the fingers open and requires measured hand contact.
Reopening closes the fingers, requires opposing contact forces on a link of the
selected drawer, and pulls with the arm while the base stays stopped. An aperture
alone is insufficient evidence of a grip. Collision and success thresholds remain
unchanged.

After initial closing, the hand withdraws along its approach axis before the arm
moves to the robot's canonical rest keyframe (joint names, including torso).
The selected withdrawal is extended only as needed to clear the countertop along
the rest trajectory: 24 cm initially, up to three extra 4 cm straight segments.
When a straight interpolation intersects furniture, the planner checks monotone
joint curves to the same rest endpoint, including every timed sample. No joint
reverses to an intermediate preset pose. The aisle corner is 20 cm farther from
the counter to clear the left wall with the restored arm envelope. It does not
introduce a special travel pose or move the base backwards.

Apple grasp geometry and the 18 cm withdrawal after release follow the published
planner. The apple is lowered vertically to 3.5 cm above the plate and both hand
and apple must settle before release. Travel uses the selected aisle route.
The head follows the current stage: look at the plate after lifting the apple;
look at the entire drawer column only after folding for the return.

The DSFetch model, keyframe, controller and cameras are unchanged. Actions remain
13D and proprioception 12D, recorded at 20 Hz and physically replayed at held 10 Hz.
New profile v4 attempts and their native/held-action/RGB checks are kept separately.
Historical profile recordings retain their original hashes and qualification.
A fixed 40-attempt pilot and new human E5 review are required before mass collection.

Earlier setup notes and measured qualification remain in the unchanged
[pre-v4 documentation](history/SAME_DRAWER_BEFORE_V4.md). Those results do not
qualify the restored movements.
