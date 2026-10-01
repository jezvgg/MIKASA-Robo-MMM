"""Named constants of the v6.1 whole-body takeitback-tray planner (metres, radians, seconds, 20 Hz steps)."""
from __future__ import annotations

import numpy as np

# -- geometry of the scene (probe_geometry, seeds 5001..5268) --------------------------------------
BASE_RADIUS = 0.35                # robot footprint radius used against the counter front
COUNTER_CLEARANCE = 0.12          # extra gap between the footprint and the counter front
STAND_Y_STEPS = (0.08, 0.13, 0.03, 0.18, 0.23, 0.28)  # extra distance from the counter tried (IK scan, seed 5001)
STAND_X_OFFSETS = (0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6)  # shift of the stand x from the cup-tray midpoint
STAND_NOISE = 0.04                # per-seed jitter of the stand position (BIBLE item 5)
STAND_HEADING = np.pi / 2         # facing the counter (+y)
STAND_HEADING_TILT = np.deg2rad(35.0)  # the final heading may differ from +y by this much (arm pan covers it)
FLOOR_MARGIN = 0.15               # keep the stand pose this far inside the floor bounds

# -- base servo ---------------------------------------------------------------------------------
BASE_MAX_SPEED = 0.45             # m/s along the heading (controller limit is 1 m/s)
BASE_MAX_YAW_RATE = 0.9           # rad/s
BASE_MAX_SPEED_NORM = 1.0         # PDBaseForwardVel normalisation: 1.0 == 1 m/s
BASE_MAX_YAW_NORM = 3.14          # PDBaseForwardVel normalisation: 1.0 == 3.14 rad/s
K_RHO, K_ALPHA, K_BETA = 1.0, 2.5, -0.4   # pose-servo gains (K_RHO > 0, K_BETA < 0, K_ALPHA > K_RHO)
TURN_FIRST_ALPHA = np.deg2rad(100.0)      # goal further off the heading than this: turn while creeping
NEAR_LATCH_RHO = 0.035            # m: within this of the stand the position is latched, the base only trims the yaw
NEAR_LATCH_PAR = 0.02             # m: ... or when the stand is abreast (along-track error) within this and
NEAR_LATCH_PAR_RHO = 0.05         # m: ... no further than this sideways
YAW_TRIM_MIN_RATE = 0.25          # rad/s: floor of the in-place yaw trim
ARRIVE_YAW_TOL = np.deg2rad(3.0)
CREEP_SPEED = 0.05                # m/s forward while turning towards a distant goal behind the base
CREEP_MIN_DISTANCE = 0.6          # nearer than this the base turns in place instead
MAX_APPROACH_STEPS = 700

# -- concurrent arm / torso schedule ------------------------------------------------------------
READY_ARM_POSTURE = np.array([0.0, 1.31, 0.0, -2.09, 0.0, 0.79, 0.0])   # planfix v4 compact posture
READY_RAMP_STEPS = 40            # steps to reach the compact posture, base already driving
ARM_START_PROGRESS = 0.15         # base path progress at which arm and torso start to move
ARM_END_PROGRESS = 0.90           # ... and at which they have arrived
ARM_MIN_TAIL_STEPS = 10

# -- manipulation (as planfix v4) -----------------------------------------------------------------
PREGRASP_GAP = 0.12
GRIPPER_OPEN = 1
GRIPPER_CLOSED = -1
GRIP_SETTLE_STEPS = 10
RELEASE_SETTLE_STEPS = 8
SETTLE_STEPS = 4
PLACE_HOVER = 0.08                # cup bottom this far above the tray top when the arm arrives over it
TRAY_DROP_GAP = 0.01
PLACE_REACH_LIFT = 0.02           # extra lift of the place pose used only in the stand IK check (grasp pose touches the cup)
IK_SEEDS = 60
GAZE_MAX_STEP = 0.06
HEAD_PAN_LIMITS = (-1.57, 1.57)
HEAD_TILT_LIMITS = (-0.76, 1.45)
REVERSE_MAX_DISTANCE = 0.45         # m: nearer than this a goal behind the base is reached in reverse
REVERSE_SPEED = 0.2               # m/s
STAND_EXCLUDE_RADIUS = 0.15        # m: a retried approach skips stands this close to a failed one
STALL_WINDOW = 30                 # steps
STALL_MIN_MOVE = 0.03             # m: less than this over the window while driving means blocked
APPROACH_ATTEMPTS = 3
SETTLE_AFTER_RELEASE = 10         # env steps the cup may settle before the success verdict
STAND_COUNTER_END_MARGIN = -10.0    # m: stands keep this far inside the counter's x extent
STAND_DX_WEIGHT = 0.0             # cost per metre away from the cup-tray midpoint
ARM_YAW_FULL = 0.35             # rad: the arm may unfold fully when the heading error is below this
ARM_YAW_BAND = 0.35               # rad: and not at all this much beyond it
PREDICT_DT = 0.05              # s: step of the kinematic rollout used to score stands
PREDICT_MAX_STEPS = 700
MONOTONE_SAMPLES = 120          # samples of the monotone reference path
MONOTONE_BEARING_SLACK = 8.0      # deg: the goal bearing may lie this far outside the start..goal heading cone
MONOTONE_LOOKAHEAD = 4            # path samples ahead used as the heading reference
MONOTONE_FEEDBACK = 1.5           # 1/s: heading feedback onto the reference
MONOTONE_MIN_SPEED = 0.05         # m/s
MONOTONE_REPLAN_EVERY = 6         # steps between re-plans of the monotone path from the current pose
PREDICT_ROT_WEIGHT = 1.0          # stand cost per radian turned by the predicted drive
PREDICT_REVERSAL_COST = 20.0      # ... per predicted reversal (change of turning direction)
PREDICT_BACKWARD_WEIGHT = 5.0     # ... per metre driven backwards
PREDICT_PATH_WEIGHT = 0.5         # ... per metre of path
PREDICT_FAIL_COST = 50.0          # ... when the predicted drive does not arrive
STAND_HEADING_STEPS = (-1.0, -0.5, 0.0, 0.5, 1.0)   # candidate final headings, fractions of STAND_HEADING_TILT around +y
MONOTONE_MAX_FAILED_PLANS = 3     # consecutive failed re-plans after which the polar law takes over
MONOTONE_MIN_CHORD = 0.3          # a path whose chord / length ratio is below this (heading sweeps too much) is rejected
NEAR_HOLD_RHO = 0.15              # m: nearer than this to the stand a path that went stale is kept (and latched at its end) instead of the polar law
CLOSE_IN_MAX_SPEED = 0.15           # m/s: along-track speed limit of the close-in mode (no path, stand within NEAR_HOLD_RHO)
