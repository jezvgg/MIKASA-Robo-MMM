"""One episode of a collection task exactly as the collection source phase runs it, minus the h5.

    PYTHONPATH=. python scripts/collection_episode.py season_dish 20000 [--assisted-grasp]

The env comes from the task's pinned profile (`utils/collection/<task>_profile.json`: its
env kwargs, motion parameters, waypoint and execution noise), the planner is called the way
`utils/collection/pipeline.collect_one` calls it, under the same robot-state guard. The
planners' own command lines build a different env (no motion parameters, no noise), so
their outcomes are not the collection's. ``--assisted-grasp`` adds
``assisted_grasp=True`` to the env kwargs; with ``MIKASA_AG_LOG=1`` every weld and release
is printed when it happens, between the planner's own lines. ``--video-dir DIR`` records the
render camera to DIR (RecordEpisode, ``render_mode="rgb_array"``); the planner's actions are
the same, only frames are kept.
"""
import argparse
import json
from pathlib import Path

PROFILES = Path(__file__).resolve().parents[1] / "utils" / "collection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=("season_dish", "same_drawer", "cabinet_search"))
    ap.add_argument("seed", type=int)
    ap.add_argument("--assisted-grasp", action="store_true")
    ap.add_argument("--ik-backend", choices=("mplib", "curobo"), default=None, help="IK from mplib alone, or proposals from cuRobo (robots/fetch/IK_BACKENDS.md; needs MIKASA_CUROBO_SOCKET or MIKASA_CUROBO_PY)")
    ap.add_argument("--ik-chain", action=argparse.BooleanOptionalAction, default=None,
                    help="solve the contact before its standoff (planners/oracle/ik_chain.py); "
                         "on unless --no-ik-chain or MIKASA_IK_CHAIN=0")
    ap.add_argument("--stance-scan", action=argparse.BooleanOptionalAction, default=None,
                    help="choose the stance by a reach scan (planners/oracle/stance_scan.py; SeasonDish's "
                         "needs MIKASA_CUROBO_SOCKET or MIKASA_CUROBO_PY); the task's default otherwise")
    ap.add_argument("--video-dir", default=None, help="record the render camera's clip here")
    ap.add_argument("--trajectory-out", default=None,
                    help="save every control step's robot qpos and action to this .npz (for comparing arms)")
    args = ap.parse_args()
    import os

    if args.ik_backend is not None:
        os.environ["MIKASA_IK_BACKEND"] = args.ik_backend
    for flag, var in ((args.ik_chain, "MIKASA_IK_CHAIN"), (args.stance_scan, "MIKASA_STANCE_SCAN")):
        if flag is not None:
            os.environ[var] = "1" if flag else "0"

    import gymnasium as gym

    import my_scenes  # noqa: F401
    from utils.collection.pipeline import action_only_oracle, classify_collection_result
    from utils.mikasa.seeding import seed_everything
    from utils.test_planner import load_planner

    profile = json.loads((PROFILES / f"{args.task}_profile.json").read_text())
    kwargs = dict(profile["env_kwargs"], obs_mode="state")
    if args.assisted_grasp:
        kwargs["assisted_grasp"] = True
    if args.video_dir:
        kwargs["render_mode"] = "rgb_array"
    seed = args.seed
    seed_everything(seed)
    env = gym.make(profile["env_id"], **kwargs)
    base = env.unwrapped
    if args.video_dir:
        from mani_skill.utils.wrappers.record import RecordEpisode

        env = RecordEpisode(env, args.video_dir, save_trajectory=False, save_video=True,
                            video_fps=20, max_steps_per_video=None)
    base.motion_parameters = dict(profile["planner"].get("motion_parameters", {}))
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
    planner_cfg = profile["planner"]
    execution_kwargs = {}
    if "action_noise" in planner_cfg:
        execution_kwargs = dict(action_noise=planner_cfg["action_noise"], noise_hold=planner_cfg["noise_hold"],
                                execution_noise_seed=seed + planner_cfg["execution_noise_seed_offset"])
    try:
        seed_everything(seed)
        with action_only_oracle(env):
            verdict = classify_collection_result(load_planner(planner_cfg["module"])(
                env, seed, waypoint_noise_seed=seed + planner_cfg["noise_seed_offset"],
                waypoint_noise_m=planner_cfg["waypoint_noise_m"], **execution_kwargs))
        info = base.get_info()
        success = verdict == "success" and bool(info["success"].reshape(-1)[0])
        print(f"collection_episode task={args.task} seed={seed} assisted_grasp={args.assisted_grasp} "
              f"ik_backend={args.ik_backend or 'mplib'} ik_chain={os.environ.get('MIKASA_IK_CHAIN', '1')} stance_scan={args.stance_scan} "
              f"verdict={verdict} success={success} steps={int(base.elapsed_steps.reshape(-1)[0])}", flush=True)
        ag = base._ag
        if ag is not None:
            kinds = [k for _, k, _ in ag.events]
            print(f"assisted_grasp establish={kinds.count('establish')} release={kinds.count('release')} "
                  f"events={ag.events}", flush=True)
        if (args.ik_backend or "mplib") == "curobo":
            from robots.fetch import curobo_client

            print(f"curobo calls={curobo_client.STATS['calls']} server_ms={curobo_client.STATS['server_ms']:.0f}",
                  flush=True)
        if args.trajectory_out:
            import numpy as np

            np.savez_compressed(args.trajectory_out, qpos=np.asarray(trace["qpos"]),
                                action=np.asarray(trace["action"]), success=success, seed=seed,
                                ik_chain=os.environ.get("MIKASA_IK_CHAIN", "1"),
                                joints=np.array([j.name for j in base.agent.robot.get_active_joints()]))
            print(f"trajectory {args.trajectory_out} steps={len(trace['qpos'])}", flush=True)
        if args.video_dir:
            arm = "chain" if os.environ.get("MIKASA_IK_CHAIN", "1") == "1" else "base"
            name = f"{args.task}_seed{seed}_{arm}{'_ag' if args.assisted_grasp else ''}_{verdict}"
            env.flush_video(name=name)
            print(f"video {args.video_dir}/{name}.mp4", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
