"""Check SeasonDish reset diversity, cue visibility and the policy boundary.

Answer swaps are diagnostic counterfactuals, never recorded demonstrations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .client import as_numpy, policy_observation
from .contract import CAMERAS, read_json, write_json
from .profile import make_env
from .pipeline import assert_signature


def capture(base):
    obs = policy_observation(base.get_obs(), base.get_language_instruction()[0])
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}


def image_changes(first, second):
    return {name: int(np.any(np.abs(first[f"observation.images.{name}"].astype(np.int16)
                   - second[f"observation.images.{name}"].astype(np.int16)) > 2, axis=-1).sum())
            for name in CAMERAS}


def qualify(run_path, output, start_seed=300, count=32):
    import torch
    from mani_skill.utils.structs import Pose
    from utils.mikasa.seeding import seed_everything

    output.mkdir(parents=True, exist_ok=False)
    run = read_json(run_path / "run.json")
    if run["scene"] != "MikasaSeasonDish-v0":
        raise ValueError("SeasonDish run required")
    assert_signature(run)
    env = make_env(run, rgb=True)
    base = env.unwrapped
    snapshots, cases = [], []
    try:
        for seed in range(start_seed, start_seed + count):
            seed_everything(seed)
            env.reset(seed=seed)
            snapshot = dict(seed=seed,
                robot_pose=as_numpy(base.agent.robot.pose.raw_pose)[0].tolist(),
                robot_qpos=as_numpy(base.agent.robot.get_qpos())[0].tolist(),
                bowl=as_numpy(base.bowl.pose.raw_pose)[0].tolist(),
                shaker=as_numpy(base.shaker.pose.raw_pose)[0].tolist(),
                bottle=as_numpy(base.condiment_bottle.pose.raw_pose)[0].tolist(),
                target_is_shaker=bool(base.target_is_shaker.item()),
                station_left_is_shaker=bool(base.station_left_is_shaker.item()),
                placement_fell_back=bool(base._placement_fell_back.item()))
            snapshots.append(snapshot)
            if base.pour_hold.item() != 0:
                raise AssertionError("Reset retained a previous pour hold")
            # Same answer-independent station gaze used by the demonstrator.
            from planners.season_dish_planner import cue_head_target
            pan, tilt = cue_head_target(base)
            arm = as_numpy(base.agent.controller.controllers["arm"].qpos)[0].copy()
            body = as_numpy(base.agent.controller.controllers["body"].qpos)[0].copy()
            initial_head = body[:2].copy()
            for step in range(12):
                body[:2] = initial_head + min(1., (step+1)/10) * (np.array([pan, tilt])-initial_head)
                env.step(np.r_[arm, 1., body, 0., 0.])
            # idle_steps holds measured head/body targets after the gaze ramp.
            body = as_numpy(base.agent.controller.controllers["body"].qpos)[0].copy()
            arm = as_numpy(base.agent.controller.controllers["arm"].qpos)[0].copy()
            for step in range(8):
                env.step(np.r_[arm, 1., body, 0., 0.])
            visible, hidden = [], []
            for answer in (False, True):
                base.target_is_shaker[:] = answer
                target = base.shaker if answer else base.condiment_bottle
                other = base.condiment_bottle if answer else base.shaker
                initial_target = snapshot["shaker" if answer else "bottle"][:3]
                initial_other = snapshot["bottle" if answer else "shaker"][:3]
                base._marker_home[:] = torch.tensor(initial_target, device=base.device) + torch.tensor([0., 0., base.cfg.marker_height], device=base.device)
                base._distractor_home[:] = torch.tensor(initial_other, device=base.device)
                base._elapsed_steps[:] = 20
                visible.append(capture(base))
                base._elapsed_steps[:] = base.cfg.cue_steps + 1
                hidden.append(capture(base))
            per_answer = []
            for answer, present, absent in zip((False, True), visible, hidden):
                changes = image_changes(present, absent)
                yellow = {}
                for name in CAMERAS:
                    key = f"observation.images.{name}"
                    frame = present[key].astype(float)
                    delta = np.any(np.abs(frame - absent[key].astype(float)) > 2, axis=-1)
                    color = (frame[..., 0] > 1.4 * frame[..., 2] + 15) & (frame[..., 1] > 1.4 * frame[..., 2] + 15)
                    yellow[name] = int((color & delta).sum())
                    if seed == start_seed:
                        Image.fromarray(present[key]).save(output / f"seed{seed}_answer{int(answer)}_cue_{name}.png")
                        Image.fromarray(absent[key]).save(output / f"seed{seed}_answer{int(answer)}_hidden_{name}.png")
                per_answer.append(dict(target_is_shaker=answer, changed_pixels=changes,
                    changed_yellow_pixels=yellow,
                    visible=min(yellow[c] for c in CAMERAS if c != "fetch_hand") >= 20))
            equal = all(np.array_equal(hidden[0][k], hidden[1][k]) for k in hidden[0])
            cases.append(dict(seed=seed, cue_cases=per_answer,
                hidden_answer_inputs_identical=equal,
                hidden_image_changes=image_changes(hidden[0], hidden[1]),
                head_qpos=as_numpy(base.agent.controller.controllers["body"].qpos)[0,:2].tolist()))
        # Repeat a seed after dirty answers/counters, including the episode buffers.
        seed_everything(start_seed)
        env.reset(seed=start_seed)
        repeat = dict(robot_pose=as_numpy(base.agent.robot.pose.raw_pose)[0].tolist(),
            robot_qpos=as_numpy(base.agent.robot.get_qpos())[0].tolist(),
            bowl=as_numpy(base.bowl.pose.raw_pose)[0].tolist(),
            shaker=as_numpy(base.shaker.pose.raw_pose)[0].tolist(),
            bottle=as_numpy(base.condiment_bottle.pose.raw_pose)[0].tolist())
        repeat_equal = all(np.array_equal(v, snapshots[0][k]) for k,v in repeat.items())
        report = dict(source_sha256=run["signature"]["code_sha256"], seeds=snapshots,
            cases=cases, repeat_equal=repeat_equal,
            unique_robot_starts=len({tuple(s["robot_pose"]) for s in snapshots}),
            unique_object_poses={k: len({tuple(s[k]) for s in snapshots}) for k in ("bowl", "shaker", "bottle")},
            placement_fallbacks=sum(s["placement_fell_back"] for s in snapshots),
            target_counts={str(b):sum(s["target_is_shaker"] == b for s in snapshots) for b in (False,True)},
            visibility_passed=all(c["visible"] for case in cases for c in case["cue_cases"]),
            hidden_passed=all(c["hidden_answer_inputs_identical"] for c in cases),
            visibility_criterion="At least 20 changed yellow pixels in each native head camera",
            note="CPU physics, scene 0; finite reset/counterfactual check, not a statistical-independence claim.")
        passed = (repeat_equal and report["visibility_passed"] and report["hidden_passed"]
                  and report["unique_robot_starts"] == count
                  and all(n == count for n in report["unique_object_poses"].values()))
        report["status"] = "success" if passed else "failed"
        write_json(output / "result.json", report)
        print({k:report[k] for k in ("status","visibility_passed","hidden_passed","repeat_equal","unique_robot_starts")})
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=300)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    qualify(args.run.resolve(), args.output.resolve(), args.start_seed, args.count)
