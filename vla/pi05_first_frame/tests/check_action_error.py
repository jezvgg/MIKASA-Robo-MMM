"""Open-loop check: how closely a trained checkpoint predicts the recorded action chunks.

    $OPENPI_DIR/.venv/bin/python vla/pi05_first_frame/tests/check_action_error.py CONFIG CHECKPOINT [FRAMES]

Runs the policy exactly as the server does (same transforms, same checkpoint loading) on
random frames of the episodes the config trains on, and compares the 10 predicted targets
with the recorded ones. As a scale it prints the error of "hold still" (joint targets =
current joints, base velocity 0). After an overfit run the model must be far below that;
if it is not, training or the data pipeline is broken, whatever the closed-loop result.
"""

from __future__ import annotations

import sys

import numpy as np
from openpi.policies import policy_config

from mikasa_pi05 import transforms as sd
from mikasa_pi05.configs import get_config

GROUPS = {"arm (rad)": list(range(7)), "head+torso": [8, 9, 10], "base velocity": [11, 12]}


def main(name: str, checkpoint: str, frames: int = 200) -> None:
    config = get_config(name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_config.dataset_factory(data_config, config.model.action_horizon, config.model)
    policy = policy_config.create_trained_policy(config, checkpoint)
    rng = np.random.default_rng(0)
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
    print(f"{len(model_err)} frames from episodes {dataset.episodes[:10]}{'...' if len(dataset.episodes) > 10 else ''}")
    print(f"{'mean |error| over the 10-step chunk':38s} {'model':>10s} {'hold still':>12s}")
    for group, dims in GROUPS.items():
        print(f"  {group:36s} {model_err[..., dims].mean():10.4f} {hold_err[..., dims].mean():12.4f}")
    print(f"  {'gripper open/close agreement':36s} {np.mean(gripper_agree):10.3f}")
    ratio = model_err[..., GROUPS["arm (rad)"]].mean() / max(hold_err[..., GROUPS["arm (rad)"]].mean(), 1e-9)
    print(f"arm error relative to holding still: {ratio:.2f} (well below 1 = the model learned the motions)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
