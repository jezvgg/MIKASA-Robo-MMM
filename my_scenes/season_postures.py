"""Task-level optional posture; the Fetch keyframe and model remain unchanged.

Fitted once, offline, to the owner's elbow-down/horizontal-gripper photograph.
The held-object wrist is subsequently corrected from the measured attachment.
"""
REFERENCE_COMPACT = {'torso_lift_joint': 0.3, 'shoulder_pan_joint': -0.233678, 'shoulder_lift_joint': 1.338583, 'upperarm_roll_joint': 2.997733, 'elbow_flex_joint': 2.2, 'forearm_roll_joint': -3.0, 'wrist_flex_joint': 0.842681, 'wrist_roll_joint': -0.059605}
