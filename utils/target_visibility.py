"""Count how many pixels of the cup and the tray each camera sees, frame by frame.

Replays a state-only planner trajectory (like `utils.replay_rgb`) with segmentation
rendering at the dataset stride and writes, per kept frame and per camera, the number
of visible cup and tray pixels. With `--events` (the planner's events.jsonl) every
frame is also tagged with the waypoint it belongs to, so a lost target can be traced
to the motion that lost it.

Usage:
    python -m utils.target_visibility <run_dir> --out <json> --stride 2 \
        [--events <events.jsonl>]
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np

import mani_skill  # noqa: F401  (the cluster fork imports my_scenes from inside mani_skill)
import my_scenes  # noqa: F401
from mani_skill.trajectory import utils as trajectory_utils

# A target counts as seen when at least this many of its pixels are visible.
MIN_PIXELS = 20


def waypoint_at(events: list[dict], step: int) -> str:
    """The last waypoint announced at or before `step`."""
    name = "start"
    for e in events:
        if e["step"] > step:
            break
        if e.get("event") == "waypoint":
            name = e["message"].split(":")[0]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--robot", default="ds_fetch")
    parser.add_argument("--events", type=Path)
    args = parser.parse_args()

    meta = json.loads((args.run_dir / "trajectory.json").read_text())
    episode = meta["episodes"][0]
    kwargs = dict(meta["env_info"]["env_kwargs"])
    src_robot = kwargs.get("robot_uids")
    kwargs.update(num_envs=1, obs_mode="rgb+segmentation", render_mode="rgb_array",
                  robot_uids=args.robot)
    env = gym.make(meta["env_info"]["env_id"], **kwargs)
    base = env.unwrapped
    env.reset(seed=episode.get("episode_seed"), options={"reconfigure": True})
    ids = {"cup": int(base.cup.per_scene_id[0]), "tray": int(base.tray.per_scene_id[0])}

    with h5py.File(args.run_dir / "trajectory.h5", "r") as f:
        states = trajectory_utils.dict_to_list_of_dicts(f["traj_0"]["env_states"])
    if src_robot and src_robot != args.robot:
        for s in states:
            art = s.get("articulations", {})
            if src_robot in art:
                art[args.robot] = art.pop(src_robot)
    events = []
    if args.events and args.events.exists():
        events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]

    frames = []
    for t in range(0, len(states), args.stride):
        base.set_state_dict(states[t])
        obs = base.get_obs()
        row = {"step": t, "waypoint": waypoint_at(events, t)}
        for cam, data in obs["sensor_data"].items():
            seg = data["segmentation"][0, ..., 0].cpu().numpy()
            for name, i in ids.items():
                row[f"{cam}/{name}"] = int((seg == i).sum())
        frames.append(row)
    env.close()

    cams = [k.split("/")[0] for k in frames[0] if k.endswith("/cup")]
    grasp = next((r["step"] for r in frames if r["waypoint"] in ("10", "11")), None)

    def seen(r, cam, name):
        return r[f"{cam}/{name}"] >= MIN_PIXELS

    summary = {}
    for cam in cams:
        summary[cam] = {
            "cup_seen_frac": round(float(np.mean([seen(r, cam, "cup") for r in frames])), 3),
            "tray_seen_frac": round(float(np.mean([seen(r, cam, "tray") for r in frames])), 3),
        }
    # What the task needs in view: the cup until it is grasped, then the tray.
    def target(r):
        return "cup" if grasp is None or r["step"] < grasp else "tray"

    any_cam = [any(seen(r, c, target(r)) for c in cams) for r in frames]
    summary["target_seen_any_cam_frac"] = round(float(np.mean(any_cam)), 3)
    summary["target_lost_frames"] = int(len(any_cam) - sum(any_cam))
    lost_by_wp: dict[str, int] = {}
    for r, ok in zip(frames, any_cam):
        if not ok:
            lost_by_wp[r["waypoint"]] = lost_by_wp.get(r["waypoint"], 0) + 1
    summary["target_lost_by_waypoint"] = lost_by_wp
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "frames": frames}, indent=1))
    print("VISIBILITY", args.run_dir, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
