# IK backends for the Fetch planner (opt-in)

Every IK request of the Fetch stack goes through `SapienPlannerV2.IK`
(`robots/fetch/utils.py`). With no environment variable set it calls mplib's
`Planner.IK` unchanged, so collection and its recorded datasets are not affected.
Two opt-in changes live in `robots/fetch/ik_backend.py`.

## What mplib does by default

`Planner.IK(goal, start, mask, n_init_qpos=N)` runs CLIK (pinocchio) from the start
qpos, then from `N - 1` configurations drawn from mplib's **global** generator. Every
start runs to the end; a converged solution is kept if it wraps into the joint limits,
the planning world (robot + the whole scene, FCL) reports no collision, the pose is
within `threshold` (position + quaternion error 1e-3) and it differs from the kept ones
by at least 0.1 rad. The status string is the same for "did not converge", "out of
limits" and "collides" (`IK Failed! Cannot find valid solution.`).

Two consequences:

* the set of solutions, and so the branch a nearest-first caller picks, depends on
  how many random draws happened before the call; a refused call shifts every later
  plan (jezvgg/ManiSkill#6);
* in a kitchen scene each start pays an FCL check against the scene meshes, so one
  call costs seconds, not the ~15 ms it costs in free space.

## `MIKASA_IK_STARTS=fixed`

The same loop and the same checks; only the starts change. The first is the current
qpos; the rest are a pure function of a hash of the request (goal to 1 mm / 0.01,
start qpos to 1e-3, mask) and of how often that request was asked before on this
planner: half placed around the current posture (5, 10 and 20 % of each joint's range),
half over the whole range. So

* the same situation always gets the same answer, whatever happened earlier in the
  episode, and mplib's generator is never drawn from;
* different requests get different starts, as mplib's draws would;
* a retry of the same request (the planners' ladders, which counted on randomness)
  gets the next block instead of the refused answer again.

Both details were measured: one table for every request made SeasonDish episodes 4x
longer (every retry repeated the refusal), and one table for all first requests still
biased the answers (median 132 s against 48 s). b1k_rl_guys' controllers and BIMBO_EvoX's
`ArmServo` also start from the current joints and fixed postures.

## `MIKASA_IK_BACKEND=curobo`

cuRobo proposes up to `MIKASA_CUROBO_SEEDS` (32) solutions seeded with the current
joints; each is refined by mplib's CLIK and kept only if mplib's checks above pass, so
the scene check is still FCL's. cuRobo itself sees only the robot (fitted spheres,
self-collision), not the scene.

* `curobo_server.py` runs under cuRobo's own venv, named by `MIKASA_CUROBO_PY`; the
  simulator's venv never imports cuRobo (it pins its own torch and warp, and switches
  TF32 on globally). `curobo_client.py` starts it once per process and talks JSON over
  a pipe.
* The server builds the ds_fetch model from `robots/fetch/fetch.urdf` with cuRobo's
  RobotBuilder (spheres fitted to the link meshes, the SRDF's disabled pairs added) and
  caches it under `~/.cache/mikasa-curobo/<urdf hash>/` (`MIKASA_CUROBO_CACHE`).
* Two variants, as the planner masks its IK: torso free (`base_link` -> `gripper_link`,
  8 joints) and torso held (`torso_lift_link` -> `gripper_link`, 7 joints; the base body
  is not in that chain, so arm-versus-base contacts are left to FCL).
* Graph replays run under `torch.no_grad()`: at cuRobo `78fd485` each call otherwise
  leaks 1.43 MiB of GPU memory (measured by BIMBO_EvoX).

Install (tested on an H100, driver 570, Python 3.11):

```
git clone https://github.com/NVlabs/curobo && git -C curobo checkout 78fd485
python -m venv curobo-venv && . curobo-venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install "./curobo[cu12-torch]" warp-lang==1.12.1   # warp >= 1.13 needs driver >= 580
```

Never `pip install nvidia-curobo` (an unrelated package on PyPI).

## Use

```
MIKASA_IK_STARTS=fixed python planners/depth_recall_v1_planner.py --seed 1000
MIKASA_CUROBO_PY=/path/to/curobo-venv/bin/python \
  python planners/depth_recall_v1_planner.py --seed 1000 --ik-backend curobo
```

For many episodes start one shared server and point every planner process at it; the
start-up is then paid once (16 s on the H100) instead of per episode:

```
/path/to/curobo-venv/bin/python robots/fetch/curobo_server.py \
  --urdf robots/fetch/fetch.urdf --socket ~/.cache/mikasa-curobo/shared.sock &
MIKASA_IK_BACKEND=curobo MIKASA_CUROBO_SOCKET=~/.cache/mikasa-curobo/shared.sock \
  python planners/depth_recall_v1_planner.py --seed 1000
```

The server answers one request at a time and stays up when a client leaves.

Tests: `utils/mikasa/test_ik_backend.py` (default equals mplib on a solved and an
unsolved request; fixed starts repeat under any generator state and leave it alone;
the cuRobo client's frames, joint order and seeding against a stub server).

## Measured (2026-10-15, H100 server, simulator poses)

Every IK call timed by a probe around `IK` (`ik_probe.py`, kept with the experiment,
not in the repository); the modes of one table ran interleaved under the same load.

**Branch stability**, Depth Recall, 6 seeds: each call re-solved once with mplib's
generator elsewhere; the nearest-to-start solution changed (> 0.1 rad) on 14 of 102
calls with random starts and on 0 of 102 with fixed starts.

**Depth Recall over 100 seeds**, simulator poses: the arm configuration at every
successful grasp, grouped into branches (same branch if no joint differs by more than
1 rad, angles modulo 2*pi) per grasp kind:

| IK starts | episodes | grasps in the dominant branch of their kind | shelf grasps | first counter grasp |
|---|---|---|---|---|
| mplib, random | 100/100 | 92.3 % | one branch | 61 % in the largest of 5 |
| fixed | 96/100 | 92.8 % | one branch | 66 % in the largest of 7 |

On this task the branch variety between episodes comes from the geometry (which counter
slot, which prop), not from the generator, and fixed starts lost four episodes, all at
the entry posture into the shelf, where their set held no solution whose line and screw
plan. Keep them off for Depth Recall.

**SeasonDish** (the planning-bound task), 30 seeds per mode:

| IK | episodes | IK calls per episode | IK calls succeeding | episode, median | episode, mean | RRT runs |
|---|---|---|---|---|---|---|
| mplib, random starts | 29/30 | 10.8 | 48 % | 48 s | 71 s | 103 |
| fixed starts | 30/30 | 7.9 | 48 % | 52 s | 52 s | 74 |
| cuRobo proposals | 29/30 | 9.9 | 82 % | 77 s | — | 199 |

(The cuRobo row is from a separate interleaved run with the same random-start numbers.)
cuRobo answers 82 % of the requests mplib answers 48 % of, but on this planner that does
not shorten episodes: more of its solutions lead to joint lines that collide, and the
planner falls back to RRT more often (199 runs against 103). The planners' cascades were
tuned to what mplib returns; IK success alone is not path quality.

**Depth Recall**, 30 seeds per mode, IK calls never fail on this task (17 per episode):

| IK | one call, median | one call, 90th pct | IK per episode |
|---|---|---|---|
| mplib, random starts | 2.2 s | 3.0 s | 37 s |
| cuRobo proposals, server per process | 0.59 s | 0.84 s | 79 s |

With the shared server, Depth Recall, 20 seeds per mode, interleaved, lighter load:

| IK | IK per episode | one call, median | server time per call | episode, median |
|---|---|---|---|---|
| mplib, random starts | 5.6 s | 323 ms | — | 55 s |
| cuRobo, server per process | 11.1 s | 80 ms | 148 ms | 62 s |
| cuRobo, one shared server | 1.35 s | 75 ms | 4.8 ms | 50 s |

In a kitchen scene a call costs hundreds of milliseconds to seconds because every start
pays FCL against the scene meshes; cuRobo's proposals need fewer starts. Its episodes are slower when the server
starts in every planner process (imports and one CUDA graph per variant: ~2.2 s each
alone, tens of seconds under load); alone, one proposal call takes 3.4 ms (torso held) to
17 ms (torso free).

## Batched reachability scan (`curobo_client.scan`)

BIMBO_EvoX's other use of cuRobo: one batched call answers which of many (stand, gripper
pose) pairs the arm reaches with the torso free — a boolean mask. Take the mask to choose
where to stand or what to reach for, never the joint solution of a pair (it may be on any
branch; solve the motion from the current joints). Kinematics only by default, optionally
with self-collision; the scene is the caller's to check. The server pads every call to
1024 goals.

`scripts/curobo_reach_map.py` uses it for an offline reach map from one stand (grid
x 0.2-1.2 m ahead, y ±0.7 m, 2.5 cm steps; 9348 goals per pass), torso free, with
self-collision:

| height | side grasp (approach level from the base) | top grasp (approach down) |
|---|---|---|
| 0.60 m | 9 % of the grid | 73 % |
| 0.92 m (counter top) | 48 % | 74 % |
| 1.20 m | 50 % | 79 % |
| 1.48 m (Depth Recall shelf) | 50 % | 70 % |

A side grasp reaches about 0.8 m ahead of base_link, a top grasp about 1.1 m; at 0.6 m a
side grasp is mostly blocked by the robot's own body. Four passes (37 392 goals) took
68 s wall under a full machine load, 4-32 s each.

Not wired into a planner's stand choice: SeasonDish grasps from where the episode puts
the robot, and its old stand-slide stage bought nothing in a 75-seed A/B (60/75 with,
59/75 without) — its refusals are collisions with the stove and the counter, which a
kinematic scan does not see.
