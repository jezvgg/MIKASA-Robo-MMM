"""Physical reset, cue and detent checks for SameDrawer.

Counterfactual answer/state edits below are diagnostics, never demonstrations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .client import as_numpy, policy_observation
from .contract import CAMERAS, read_json, write_json
from .pipeline import assert_signature
from .profile import make_env


def capture(base):
    return policy_observation(base.get_obs(), base.get_language_instruction()[0])


def changed(a, b):
    return {camera: int(np.any(np.abs(a[f"observation.images.{camera}"].astype(np.int16)
                - b[f"observation.images.{camera}"].astype(np.int16)) > 2, axis=-1).sum())
            for camera in CAMERAS}


def qualify(run_path, output, start_seed=300, count=32):
    import torch
    import sapien
    from utils.mikasa.seeding import seed_everything

    output.mkdir(parents=True, exist_ok=False)
    run = read_json(run_path / "run.json")
    if run["scene"] != "MikasaSameDrawer-v0":
        raise ValueError("A SameDrawer run is required")
    assert_signature(run)
    env = make_env(run, rgb=True)
    base = env.unwrapped
    snapshots, cases = [], []

    def snapshot(seed):
        return dict(seed=seed, robot_pose=as_numpy(base.agent.robot.pose.raw_pose)[0].tolist(),
            robot_qpos=as_numpy(base.agent.robot.get_qpos())[0].tolist(),
            apple=as_numpy(base.apple.pose.raw_pose)[0].tolist(),
            plate=as_numpy(base.plate.pose.raw_pose)[0].tolist(),
            target_drawer=int(base.target_drawer.item()), init_open=float(base.init_open.item()))

    def set_drawers(answer, amount):
        base.target_drawer[:] = answer
        for index, art in enumerate(base._drawer_arts):
            q = torch.zeros_like(art.get_qpos())
            q[:, 0] = -amount if index == answer else 0.
            art.set_qpos(q)
            art.set_qvel(torch.zeros_like(q))
        base.closed_done[:] = amount == 0.
        base.apple_done[:] = False
        base.wrong_drawer_touched[:] = False
        base.sequence_violated[:] = False
        base.held_count[:] = 0
        base._last_eval_step[:] = -1

    try:
        for seed in range(start_seed, start_seed + count):
            seed_everything(seed)
            env.reset(seed=seed)
            snapshots.append(snapshot(seed))
            if (base.closed_done.any() or base.apple_done.any()
                    or base.wrong_drawer_touched.any() or base.sequence_violated.any()):
                raise AssertionError("Reset retained task latches")
            arm = as_numpy(base.agent.controller.controllers["arm"].qpos)[0].copy()
            body = as_numpy(base.agent.controller.controllers["body"].qpos)[0].copy()
            initial_head = body[:2].copy()
            local = (base.agent.base_link.pose[0].sp.inv()
                     * sapien.Pose(as_numpy(base.handle_home)[0].mean(0))).p
            pan = float(np.clip(np.arctan2(local[1], local[0]), -.6, .6))
            for step in range(15):
                body[:2] = initial_head + min(1., (step+1)/10) * (np.array([pan, .65])-initial_head)
                env.step(np.r_[arm, 1., body, 0., 0.])
            visible, hidden, visibility = [], [], []
            for answer in base.cfg.drawer_choices:
                set_drawers(answer, base.cfg.init_open_range[0])
                seen = capture(base)
                set_drawers(answer, 0.)
                unseen = capture(base)
                visible.append(seen)
                hidden.append(unseen)
                delta = changed(seen, unseen)
                visibility.append(dict(answer=answer, changed_pixels=delta,
                    visible=max(delta[c] for c in CAMERAS if c != "fetch_hand") >= 20))
                if seed == start_seed:
                    for camera in CAMERAS:
                        Image.fromarray(seen[f"observation.images.{camera}"]).save(output / f"answer{answer}_{camera}.png")
                        if answer == base.cfg.drawer_choices[0]:
                            Image.fromarray(unseen[f"observation.images.{camera}"]).save(output / f"closed_{camera}.png")
            identical = all(all(np.array_equal(hidden[0][key], other[key]) for key in hidden[0])
                            for other in hidden[1:])
            # The detent removes tiny residual openings used as external memory.
            set_drawers(base.cfg.drawer_choices[0], base.cfg.closed_tol / 2)
            base.get_info()
            detent_zero = bool((base.drawer_open_amounts() == 0).all())
            counts_before = base.held_count.clone()
            for _ in range(3):
                base.get_info()
            evaluate_idempotent = bool(torch.equal(counts_before, base.held_count))
            cases.append(dict(seed=seed, cue_cases=visibility,
                              hidden_answer_inputs_identical=identical,
                              detent_zero=detent_zero, evaluate_idempotent=evaluate_idempotent))
        seed_everything(start_seed)
        env.reset(seed=start_seed)
        repeat_equal = snapshot(start_seed) == snapshots[0]
        report = dict(source_sha256=run["signature"]["code_sha256"], seeds=snapshots, cases=cases,
            repeat_equal=repeat_equal,
            unique_robot_starts=len({tuple(s["robot_pose"]) for s in snapshots}),
            unique_object_poses={key: len({tuple(s[key]) for s in snapshots}) for key in ("apple", "plate")},
            target_counts={str(answer):sum(s["target_drawer"] == answer for s in snapshots)
                           for answer in base.cfg.drawer_choices},
            note="CPU, kitchen 0; answer swaps are diagnostics. This does not measure learned policy memory.")
        passed = (repeat_equal and report["unique_robot_starts"] == count
                  and all(n == count for n in report["unique_object_poses"].values())
                  and all(c["hidden_answer_inputs_identical"] and c["detent_zero"] and c["evaluate_idempotent"]
                          and all(v["visible"] for v in c["cue_cases"]) for c in cases))
        report["status"] = "success" if passed else "failed"
        write_json(output / "report.json", report)
        print({k:report[k] for k in ("status", "repeat_equal", "unique_robot_starts", "target_counts")})
        if not passed:
            raise AssertionError("SameDrawer qualification failed; see report.json")
        return report
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=300)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    qualify(args.run, args.output, args.start_seed, args.count)
