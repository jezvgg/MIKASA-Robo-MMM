"""One MikasaCabinetSearch5-v0 episode (cab_1 opened by the fist, EXPERIMENTAL), the collection's way.

    PYTHONPATH=. python tools/cab5/run_cab5.py 7890000 [--first 0] [--video-dir DIR] [--trajectory-out F]

`scripts/collection_episode.py` for the five-compartment env: the CabinetSearch profile's env
kwargs, motion parameters and noise, the planner under the collection's action-only guard. With
`--first K` the search route starts at compartment K (0 = cab_1) and keeps the seed's order for
the rest, so a probe reaches the new leaf in the first round; without it the route is the seed's.
"""
import argparse
import json
from pathlib import Path

PROFILES = Path(__file__).resolve().parents[2] / "utils" / "collection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--first", type=int, default=None, help="open this compartment first (0 = cab_1)")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--trajectory-out", default=None)
    ap.add_argument("--n4", action="store_true", help="the four-compartment task (control)")
    ap.add_argument("--cab1-cam", action="store_true",
                    help="record from a camera close on cab_1 (south-east of it, above the counter)")
    args = ap.parse_args()

    import gymnasium as gym
    import numpy as np

    import my_scenes  # noqa: F401
    from planners import cabinet_search_planner as csp
    from utils.collection.pipeline import action_only_oracle, classify_collection_result
    from utils.collection.profile import profile_env_kwargs
    from utils.mikasa.seeding import seed_everything
    from utils.test_planner import load_planner

    profile = json.loads((PROFILES / "cabinet_search_profile.json").read_text())
    kwargs = profile_env_kwargs(profile, obs_mode="state")
    if args.video_dir:
        kwargs["render_mode"] = "rgb_array"
        if args.cab1_cam:
            from mani_skill.utils import sapien_utils

            kwargs["human_render_camera_configs"] = dict(
                pose=sapien_utils.look_at(eye=[1.75, -2.05, 2.25], target=[0.55, -0.70, 1.45]),
                width=720, height=540, fov=1.1)
    if args.first is not None:
        plan = csp.search_plan

        def first_then_seed(seed, n):
            rng, order = plan(seed, n)
            rest = [int(v) for v in order if int(v) != args.first]
            return rng, np.array([args.first] + rest)

        csp.search_plan = first_then_seed
    seed = args.seed
    seed_everything(seed)
    env = gym.make("MikasaCabinetSearch-v0" if args.n4 else "MikasaCabinetSearch5-v0", **kwargs)
    base = env.unwrapped
    if args.video_dir:
        from mani_skill.utils.wrappers.record import RecordEpisode

        env = RecordEpisode(env, args.video_dir, save_trajectory=False, save_video=True,
                            video_fps=20, max_steps_per_video=None)
    base.motion_parameters = dict(profile["planner"].get("motion_parameters", {}))
    trace = {"qpos": [], "action": []}
    if args.trajectory_out:
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
    cfg = profile["planner"]
    try:
        seed_everything(seed)
        with action_only_oracle(env):
            verdict = classify_collection_result(load_planner(cfg["module"])(
                env, seed, waypoint_noise_seed=seed + cfg["noise_seed_offset"],
                waypoint_noise_m=cfg["waypoint_noise_m"], action_noise=cfg["action_noise"],
                noise_hold=cfg["noise_hold"], execution_noise_seed=seed + cfg["execution_noise_seed_offset"]))
        info = base.get_info()
        cube = int(np.asarray(base.cube_cab.cpu()).reshape(-1)[0])
        success = verdict == "success" and bool(info["success"].reshape(-1)[0])
        print(f"cab5_episode seed={seed} first={args.first} cube_compartment={cube} verdict={verdict} "
              f"success={success} steps={int(base.elapsed_steps.reshape(-1)[0])}", flush=True)
        if args.trajectory_out:
            np.savez_compressed(args.trajectory_out, qpos=np.asarray(trace["qpos"]),
                                action=np.asarray(trace["action"]), success=success, seed=seed,
                                joints=np.array([j.name for j in base.agent.robot.get_active_joints()]))
        if args.video_dir:
            env.flush_video(name=f"cab5_seed{seed}_first{args.first}_{verdict}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
