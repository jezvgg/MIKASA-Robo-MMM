"""One episode of a collection task exactly as the collection runs it, minus the h5.

    PYTHONPATH=. python scripts/collection_episode.py season_dish 20000 [--assisted-grasp]
    PYTHONPATH=. python scripts/collection_episode.py season_dish 7762000 --actions actions.npy

The env comes from the task's pinned profile (`utils/collection/<task>_profile.json`: its
env kwargs, motion parameters, waypoint and execution noise), the planner is called by
`utils/collection/pipeline.run_oracle` — the same call `collect_one` makes — under the same
robot-state guard. The planners' own command lines build a different env (no motion
parameters, no noise), so their outcomes are not the collection's.

``--actions FILE.npy`` replays a recorded episode instead of planning: the file holds an
episode's 10 Hz absolute targets (the LeRobot ``action`` column of our published datasets,
frame order), every target is held for two 20 Hz control steps by
`utils.collection.client.execute_actions`, and with the weld rule off the recorded outcome
must come back.

``--assisted-grasp`` adds ``assisted_grasp=True`` to the env kwargs; with
``MIKASA_AG_LOG=1`` every weld and release is printed when it happens, between the
planner's own lines. ``--video-dir DIR`` records the render camera to DIR (RecordEpisode,
``render_mode="rgb_array"``); the actions are the same, only frames are kept.
``--trajectory-out`` saves every control step's robot qpos and action to an ``.npz``.
"""
import argparse
import json
from pathlib import Path

PROFILES = Path(__file__).resolve().parents[1] / "utils" / "collection"


def main():
    from utils.collection.profile import TASKS

    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=tuple(TASKS))
    ap.add_argument("seed", type=int)
    ap.add_argument("--actions", type=Path, default=None,
                    help="replay this .npy of 10 Hz action targets instead of planning")
    ap.add_argument("--assisted-grasp", action=argparse.BooleanOptionalAction, default=None,
                    help="override the profile's assisted_grasp (a profile without the key is the friction grasp)")
    ap.add_argument("--ik-backend", choices=("mplib", "curobo"), default=None, help="IK from mplib alone, or proposals from cuRobo (robots/fetch/IK_BACKENDS.md; needs MIKASA_CUROBO_SOCKET or MIKASA_CUROBO_PY)")
    ap.add_argument("--ik-chain", action=argparse.BooleanOptionalAction, default=None,
                    help="solve the contact before its standoff (planners/oracle/ik_chain.py); "
                         "off unless --ik-chain or MIKASA_IK_CHAIN=1")
    ap.add_argument("--video-dir", default=None, help="record the render camera's clip here")
    ap.add_argument("--trajectory-out", default=None,
                    help="save every control step's robot qpos and action to this .npz (for comparing arms)")
    args = ap.parse_args()
    import os

    if args.ik_backend is not None:
        os.environ["MIKASA_IK_BACKEND"] = args.ik_backend
    if args.ik_chain is not None:
        os.environ["MIKASA_IK_CHAIN"] = "1" if args.ik_chain else "0"

    import my_scenes  # noqa: F401
    from utils.collection.client import execute_actions
    from utils.collection.contract import ACTION_REPEAT
    from utils.collection.pipeline import run_oracle
    from utils.collection.profile import experiment_env, profile_env_kwargs
    from utils.mikasa.seeding import seed_everything

    profile = json.loads((PROFILES / f"{args.task}_profile.json").read_text())
    kwargs = {}
    if args.assisted_grasp is not None:
        kwargs["assisted_grasp"] = args.assisted_grasp
    if args.video_dir:
        kwargs["render_mode"] = "rgb_array"
    seed = args.seed
    seed_everything(seed)
    env = experiment_env(profile, **kwargs)
    base = env.unwrapped
    assisted_grasp = profile_env_kwargs(profile, **kwargs)["assisted_grasp"]
    if args.video_dir:
        from mani_skill.utils.wrappers.record import RecordEpisode

        env = RecordEpisode(env, args.video_dir, save_trajectory=False, save_video=True,
                            video_fps=20, max_steps_per_video=None)
    trace = {"qpos": [], "action": []}
    if args.trajectory_out:
        import numpy as np

        unwrapped_step = base.step

        def traced_step(action):
            out = unwrapped_step(action)
            trace["qpos"].append(np.asarray(base.agent.robot.get_qpos().cpu()).reshape(-1).copy())
            trace["action"].append(np.asarray(action.cpu() if hasattr(action, "cpu") else action,
                                              dtype=np.float32).reshape(-1).copy())
            return out

        base.step = traced_step
    env.log_event = lambda event, message, **extra: print(
        f"[event] step={int(base.elapsed_steps.reshape(-1)[0])} {event} {message}", flush=True)
    try:
        if args.actions is not None:
            import numpy as np

            actions = np.load(args.actions).astype(np.float32)
            seed_everything(seed)
            env.reset(seed=seed)
            result = execute_actions(env, actions, repeat=ACTION_REPEAT)
            info = result["info"]
            flag = lambda k: bool(np.asarray(info[k].cpu() if hasattr(info[k], "cpu") else info[k]).reshape(-1)[0]) \
                if k in info else None
            success = flag("success")
            print(f"replay task={args.task} seed={seed} assisted_grasp={assisted_grasp} "
                  f"status={result['status']} success={success} success_once={flag('success_once')} "
                  f"steps={result['control_steps']} of={len(actions) * ACTION_REPEAT}", flush=True)
            verdict = result["status"]
        else:
            verdict = run_oracle(env, profile, seed)
            info = base.get_info()
            success = verdict == "success" and bool(info["success"].reshape(-1)[0])
            print(f"collection_episode task={args.task} seed={seed} assisted_grasp={assisted_grasp} "
                  f"ik_backend={args.ik_backend or 'mplib'} ik_chain={os.environ.get('MIKASA_IK_CHAIN', '0')} "
                  f"verdict={verdict} success={success} steps={int(base.elapsed_steps.reshape(-1)[0])}",
                  flush=True)
            if (args.ik_backend or "mplib") == "curobo":
                from robots.fetch import curobo_client

                print(f"curobo calls={curobo_client.STATS['calls']} "
                      f"server_ms={curobo_client.STATS['server_ms']:.0f}", flush=True)
        ag = base._ag
        if ag is not None:
            kinds = [k for _, k, _ in ag.events]
            print(f"assisted_grasp establish={kinds.count('establish')} release={kinds.count('release')} "
                  f"events={ag.events}", flush=True)
        steps = int(base.elapsed_steps.reshape(-1)[0])
        if args.trajectory_out:
            import numpy as np

            np.savez_compressed(args.trajectory_out, qpos=np.asarray(trace["qpos"]),
                                action=np.asarray(trace["action"]), success=success, seed=seed,
                                ik_chain=os.environ.get("MIKASA_IK_CHAIN", "0"),
                                joints=np.array([j.name for j in base.agent.robot.get_active_joints()]))
            print(f"trajectory {args.trajectory_out} steps={steps}", flush=True)
        if args.video_dir:
            arm = "chain" if os.environ.get("MIKASA_IK_CHAIN", "0") == "1" else "base"
            name = f"{args.task}_seed{seed}_{arm}{'_ag' if assisted_grasp else ''}_{verdict}"
            env.flush_video(name=name)
            print(f"video {args.video_dir}/{name}.mp4", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
