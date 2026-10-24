"""Open-loop check of a MemER low-level checkpoint, per task, on held-out and on training episodes.

    $OPENPI_DIR/.venv/bin/python vla/pi05_first_frame/tests/check_memer_action_error.py CONFIG CHECKPOINT \
        [FRAMES_PER_TASK]

Runs the policy exactly as the server does (same transforms, same checkpoint loading) on random
frames, with the frame's subtask as the prompt, and compares the predicted action chunk with the
recorded one. Held-out = the episodes of each task the config does not train on (all of them if it
trains on every episode, then the check says so). As a scale it prints the error of "hold still"
(joint targets = current joints, base velocity 0): a model that learned the motions is well below 1.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from openpi.policies import policy_config

from mikasa_pi05 import transforms as sd
from mikasa_pi05.configs import get_config, memer_data_root
from mikasa_pi05.dataset import SameDrawerDataset

GROUPS = {"arm (rad)": list(range(7)), "head+torso": [8, 9, 10], "base velocity": [11, 12]}


def errors(policy, dataset, frames: int, rng) -> dict[str, float]:
    model_err, hold_err, gripper_agree = [], [], []
    for index in rng.choice(len(dataset), size=min(frames, len(dataset)), replace=False):
        sample = dataset[int(index)]
        truth = sample.pop("actions")
        predicted = np.asarray(policy.infer(sample)["actions"])[: len(truth)]
        state = np.asarray(sample["observation.state"])[list(sd.STATE_TO_ACTION_ORDER)]
        hold = np.zeros_like(truth)
        hold[:, :11] = state[:11]
        hold[:, 7] = truth[0, 7]  # the gripper has no "hold" equivalent; not scored here
        model_err.append(np.abs(predicted - truth))
        hold_err.append(np.abs(hold - truth))
        gripper_agree.append(np.mean(np.sign(predicted[:, 7]) == np.sign(truth[:, 7])))
    model_err, hold_err = np.stack(model_err), np.stack(hold_err)
    out = {f"{group} model": float(model_err[..., dims].mean()) for group, dims in GROUPS.items()}
    out |= {f"{group} hold": float(hold_err[..., dims].mean()) for group, dims in GROUPS.items()}
    out["gripper agreement"] = float(np.mean(gripper_agree))
    out["arm ratio"] = out["arm (rad) model"] / max(out["arm (rad) hold"], 1e-9)
    return out


def main(name: str, checkpoint: str, frames: int = 50) -> None:
    config = get_config(name)
    data = config.data
    root = pathlib.Path(data.data_root) if data.data_root else memer_data_root()
    policy = policy_config.create_trained_policy(config, checkpoint)
    rng = np.random.default_rng(0)
    horizon = config.model.action_horizon
    print(f"{frames} frames per task and split; chunk of {horizon}; mean |error| (model / hold still)")
    for task in data.tasks:
        every = SameDrawerDataset(root / task, horizon, first_frame=False, prompt="subtask", load_images=False).episodes
        trained = every if data.episodes is None else sorted(set(data.episodes) & set(every))
        held = sorted(set(every) - set(trained)) or trained
        tag = "held-out" if held is not trained else "training (nothing held out)"
        splits = {tag: held, "training": trained[-len(held):]}
        for split, episodes in splits.items():
            dataset = SameDrawerDataset(root / task, horizon, first_frame=False, prompt="subtask", episodes=episodes)
            e = errors(policy, dataset, frames, rng)
            print(f"{task:14s} {split:9s} eps {episodes[0]}-{episodes[-1]}: "
                  f"arm {e['arm (rad) model']:.4f}/{e['arm (rad) hold']:.4f} (ratio {e['arm ratio']:.2f})  "
                  f"head+torso {e['head+torso model']:.4f}/{e['head+torso hold']:.4f}  "
                  f"base {e['base velocity model']:.4f}/{e['base velocity hold']:.4f}  "
                  f"gripper {e['gripper agreement']:.3f}", flush=True)
    sample = SameDrawerDataset(root / data.tasks[0], horizon, first_frame=False, prompt="subtask",
                               load_images=False)[100]
    print(f"example prompt ({data.tasks[0]}, frame 100): {sample['prompt']!r}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
