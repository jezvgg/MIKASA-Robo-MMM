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
LOWER_RAMP_STEPS = 40             # env steps of the torso ramp that lowers the held cup onto the tray (fast enough that no 10 Hz frame of it reads as a stall)
LOWER_TRIM_STEPS = 40             # extra steps trimming the torso on the measured cup height
LOWER_TRIM_GAIN = 0.7             # torso correction per metre of cup-height error
LOWER_TOL = 0.004                 # m: cup height error above which the lowering goes on
CUP_REST_V = 0.015                # m/s: stricter than the checker's 0.02, so the verdict is not on the edge
CUP_REST_W = 0.08                 # rad/s: stricter than the checker's 0.10
CUP_REST_STEPS = 8                # consecutive calm steps before the verdict
CUP_SETTLE_MAX = 80               # env steps the cup may take to settle after release
SETTLE_AFTER_RELEASE = 10         # env steps the cup may settle before the success verdict
STAND_COUNTER_END_MARGIN = -10.0    # m: stands keep this far inside the counter's x extent
STAND_DX_WEIGHT = 0.0             # cost per metre away from the cup-tray midpoint
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

# -- smooth arm motion (v6.1b) ---------------------------------------------------------------------
ARM_MAX_STEP = 0.05              # rad per env step (1.0 rad/s): v5/v4 peak at 0.09-0.11 rad per 10 Hz frame
ARM_MAX_ACCEL = 0.004             # rad per env step^2: reaches full speed in ~13 steps

# -- arm-aware stand choice and calmer retries (v6.1c) ---------------------------------------------
ARM_ROLL_WEIGHT = 3.0             # roll joints (upperarm, forearm, wrist) cost this much per radian, other joints 1
ARM_NON_ROLL_WEIGHT = 1.0
ARM_ROLL_ABS_WEIGHT = 0.3         # small pull of the rolls towards zero (neutral wrist)
PRE_CANDIDATES = 3                # IK branches of the pregrasp tried per stand
PAN_FREE = 0.5                    # rad of shoulder pan at the grasp that costs nothing
PAN_WEIGHT = 2.0                  # stand cost per rad of pan beyond it
PLACE_REACH_SOFT = 0.95           # m: stand-to-tray distance beyond which a stand is penalised
PLACE_REACH_WEIGHT = 6.0          # stand cost per metre beyond it
ARM_COST_WEIGHT = 0.05            # stand cost per unit of arm-chain cost (rotating the base 1 rad costs 1.0)
STAND_TOPK = 10                   # feasible stands compared per search
PLACE_EXTRA_HEIGHTS = (0.0, 0.04, 0.08)   # m above the place pose tried in turn when the planner refuses it
CARRY_LIFT = 0.06                 # m: straight lift of the held cup before it moves over the tray
CLOSE_IN_LATCH_RHO = 0.08         # m: once close in without a path the position latches this near the stand
GRASP_PROBE_GAP = 0.08            # m: the grasp reach is checked by IK at the grasp pose backed off this far along the approach
ARM_TAIL_SLACK = 60               # env steps beyond the unfold time the base may wait, arrived, for the arm to reach the pregrasp configuration
UNFOLD_PEAK_STEP = 0.04           # rad per env step: peak joint speed of the unfold (the limiter's cap is ARM_MAX_STEP)
UNFOLD_SAMPLES = 24               # samples of the unfold line checked for collisions at a candidate stand
UNFOLD_MARGIN = 0.04              # m: the line is also checked with the base this much further ahead (arrival error)
LOWER_MARGIN = 0.01               # m: the arm at the place pose must also be collision-free this much below the lowered cup height
STRAIGHT_MAX_DX = 0.9               # m: the straight-ahead stand family is kept within this of the cup-tray midpoint
BEARING_TILT = np.deg2rad(50.0)        # the heading that faces the stand from the start is also a candidate, within this of +y
TORSO_SOFT = 0.28                  # m: torso height at the pregrasp beyond which a stand is penalised (the carry needs headroom below the 0.386 limit)
TORSO_WEIGHT = 30.0               # stand cost per metre beyond it

# -- v6.1d ---------------------------------------------------------------------------------------------
ROLL_HARD = 3.0                   # rad: a stand whose pregrasp, grasp or place needs a roll joint beyond this is skipped (a roll past +-pi is a wrap)
ROLL_SOFT = 2.4                   # rad: roll magnitude beyond which a stand is penalised
ROLL_WEIGHT = 5.0                 # stand cost per radian beyond it
RETREAT_UP = 0.06                # m: torso rise that lifts the open hand off the released cup
RETREAT_STEPS = 40                # env steps of that rise
PLACE_MARGIN = 0.02               # m: a preferred stand also reaches the place pose this much further from the base (the carry IK with the cup attached is tighter than the bare-hand one)
MARGIN_WEIGHT = 6.0               # stand cost added when the place pose has no reach margin

# -- v6.1e ---------------------------------------------------------------------------------------------
ROUTE_STEP_M = 0.06               # m: spacing of the poses of the predicted base route that are checked for collisions
ROUTE_STEP_YAW = 0.10             # rad: ... or the yaw change between them
ARRIVAL_ENVELOPE = ((-0.10, 0.0), (-0.05, 0.04), (-0.05, -0.04))   # (along, sideways) m of the base frame: where the base really stops relative to the stand (d5: median 0.03 behind, worst 0.10), place IK is also needed there
RELEASE_RAMP_STEPS = 10           # env steps over which the gripper command goes from closed to open at the release
CARRY_WAYPOINTS = (0.4, 0.75)    # fractions of the way from the lifted cup to the place pose at which the carry stops first
