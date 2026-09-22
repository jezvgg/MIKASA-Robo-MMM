"""Counterfactual RGB/proprio checks at actual CabinetSearch look/home poses.

Direct state writes here construct diagnostic comparisons, never demonstrations.
The collector separately rejects physical robot writes outside reset.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .client import policy_observation
from .contract import CAMERAS, read_json, write_json
from .pipeline import read_state
from .profile import make_env


def capture(base):
    result = policy_observation(base.get_obs(), base.get_language_instruction()[0])
    return {key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in result.items()}


def changed(first, second):
    result = {}
    for name in CAMERAS:
        key = f"observation.images.{name}"
        delta = np.abs(first[key].astype(np.int16) - second[key].astype(np.int16))
        result[name] = int(np.any(delta > 2, axis=-1).sum())
    return result


def save_images(obs, directory, label):
    for name in CAMERAS:
        Image.fromarray(obs[f"observation.images.{name}"]).save(directory / f"{label}_{name}.png")


def qualify(roots, output):
    import torch
    from mani_skill.utils.structs import Pose

    output.mkdir(parents=True, exist_ok=False)
    runs = [read_json(root / "run.json") for root in roots]
    signature = runs[0]["signature"]
    if any(run["signature"] != signature for run in runs):
        raise ValueError("Visibility proof may not mix different implementations")
    locations, home = {}, None
    for root, run in zip(roots, runs):
        for seed in run["seeds"]:
            directory = root / "oracle" / str(seed)
            if not (directory / "result.json").exists():
                continue
            if read_json(directory / "result.json")["status"] != "success":
                continue
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            current = None
            for i, event in enumerate(events):
                if event["message"] == "round":
                    current = int(event["pick"])
                if event["message"] == "look with the head":
                    # The immediately following event is the verdict after head dwell.
                    locations.setdefault(current, (directory / "trajectory.h5", events[i+1]["step"], seed))
                if event["message"] == "home credited" and event.get("n_decisions", 0) > 0:
                    home = home or (directory / "trajectory.h5", event["step"], seed)
    if home is None:
        raise ValueError("Need an executed return-home state after an empty cabinet")
    env = make_env(runs[0], rgb=True)
    base = env.unwrapped
    env.reset(seed=home[2])
    report = dict(source_sha256=signature["code_sha256"], locations={}, hidden={}, randomization={})

    def restore(location):
        path, t, _ = location
        with h5py.File(path, "r") as h5:
            state = read_state(h5["traj_0/env_states"], t)
        base.set_state_dict(state)
        base._elapsed_steps[:] = t
        return state

    def park(position):
        base.cube.set_pose(Pose.create_from_pq(p=np.asarray(position, dtype=np.float32)))
        base.cube.set_linear_velocity(torch.zeros((1, 3)))
        base.cube.set_angular_velocity(torch.zeros((1, 3)))

    try:
        restore(home)
        first = capture(base)
        repeat = capture(base)
        report["hidden"]["repeat_control_changed_pixels"] = changed(first, repeat)
        save_images(first, output, "closed_home")
        before_proprio = first["observation.state"].copy()
        history = base.opened_count.clone()
        base.opened_count[:] = torch.roll(history, 1, dims=1)
        base.last_opened[:] = (base.last_opened + 1) % 4
        other_history = capture(base)
        report["hidden"]["other_history_changed_pixels"] = changed(first, other_history)
        report["hidden"]["history_proprio_equal"] = bool(np.array_equal(before_proprio, other_history["observation.state"]))
        report["hidden"]["cube_locations"] = []
        for cab in range(4):
            restore(home)
            x = float(base._spawn_centre_x[0, cab])
            park([x, base.cfg.spawn_depth, base.cfg.shelf_top_z + base.cfg.cube_half])
            base.cube_cab[:] = cab
            candidate = capture(base)
            report["hidden"]["cube_locations"].append(dict(compartment=cab,
                changed_pixels=changed(first, candidate),
                proprio_equal=bool(np.array_equal(before_proprio, candidate["observation.state"]))))
        for cab, location in sorted(locations.items()):
            cases = []
            # Centre and all four corners of the permitted object jitter band.
            offsets = [(0., 0.)] + [(x, y) for x in (-0.06, 0.06) for y in (-0.02, 0.02)]
            for index, (dx, dy) in enumerate(offsets):
                restore(location)
                park([0., 0., -5.])
                absent = capture(base)
                x = float(base._spawn_centre_x[0, cab]) + dx
                position = [x, base.cfg.spawn_depth + dy, base.cfg.shelf_top_z + base.cfg.cube_half]
                park(position)
                base.cube_cab[:] = cab
                present = capture(base)
                counts = changed(absent, present)
                red_pixels = {}
                for name in CAMERAS:
                    key = f"observation.images.{name}"
                    image = present[key].astype(float)
                    delta = np.any(np.abs(image - absent[key].astype(float)) > 2, axis=-1)
                    red = (image[..., 0] > 1.5 * image[..., 1] + 20) & (image[..., 0] > 1.5 * image[..., 2] + 20)
                    red_pixels[name] = int((delta & red).sum())
                cases.append(dict(offset_m=[dx, dy], changed_pixels=counts,
                    changed_red_pixels=red_pixels,
                    head_camera_visible=max(red_pixels[name] for name in CAMERAS if name != "fetch_hand") >= 8))
                if index == 0:
                    save_images(present, output, f"open_compartment_{cab}")
                    save_images(absent, output, f"empty_compartment_{cab}")
            report["locations"][str(cab)] = dict(source=str(location[0]), step=location[1], seed=location[2], cases=cases)
        # Exercise real reset, including after deliberately dirty task buffers.
        snapshots = []
        for seed in range(300, 332):
            env.reset(seed=seed)
            snapshots.append(dict(seed=seed, compartment=int(base.cube_cab.item()),
                instruction_index=int(base.instruction_idx.item()),
                base_qpos=base.agent.robot.get_qpos()[0, :3].cpu().numpy().tolist(),
                cube_xyz=base.cube.pose.p[0].cpu().numpy().tolist()))
            if bool(base.opened_count.any()) or bool(base.succeeded.any()):
                raise AssertionError("Reset retained a previous search history")
        env.reset(seed=300)
        repeat_qpos = base.agent.robot.get_qpos()[0, :3].cpu().numpy()
        np.testing.assert_array_equal(repeat_qpos, np.array(snapshots[0]["base_qpos"]))
        report["randomization"] = dict(seeds=snapshots, repeat_seed_300_equal=True,
            compartment_counts=np.bincount([s["compartment"] for s in snapshots], minlength=4).tolist(),
            unique_starts=len({tuple(s["base_qpos"]) for s in snapshots}),
            unique_instructions=len({s["instruction_index"] for s in snapshots}),
            inference="Independent RNG draws are verified in source; this finite reset audit checks implementation, not statistical independence.")
        hidden = report["hidden"]
        report["hidden_passed"] = (not any(hidden["repeat_control_changed_pixels"].values())
            and not any(hidden["other_history_changed_pixels"].values()) and hidden["history_proprio_equal"]
            and all(not any(v["changed_pixels"].values()) and v["proprio_equal"] for v in hidden["cube_locations"]))
        report["visibility_passed"] = len(locations) == 4 and all(case["head_camera_visible"]
            for value in report["locations"].values() for case in value["cases"])
        report["status"] = "success" if report["hidden_passed"] and report["visibility_passed"] else "incomplete_or_failed"
        write_json(output / "result.json", report)
        print(json.dumps({key: report[key] for key in ("status", "hidden_passed", "visibility_passed")}))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify([path.resolve() for path in args.runs], args.output.resolve())


if __name__ == "__main__":
    main()
