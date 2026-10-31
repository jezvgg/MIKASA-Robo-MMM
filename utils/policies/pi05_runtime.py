"""Run π0.5 for one frame, on real weights, from this repository.

π0.5 is not vendored: openpi lives in its own checkout and its own virtualenv
(`vla/pi05_first_frame/scripts/setup_server.sh` builds `$OPENPI_DIR`), because its
pinned dependencies (`jax`, `flax`, `openpi-client`) are incompatible with the
simulation stack in `.venv-sim`. Point this module at that interpreter:

    ~/policy-integration-20261019/.venv-robomme/bin/python -m utils.policies.pi05_runtime

Two things the runtime has to do that a plain `create_trained_policy` does not:

1. `openpi` imports `lerobot.common.datasets.lerobot_dataset`, the pre-0.4 layout.
   Installed lerobot is 0.4.0, where the module moved to
   `lerobot.datasets.lerobot_dataset`; only the training dataloader uses it, so the
   module is aliased before openpi is imported.
2. A checkpoint directory must carry `assets/<asset-id>/norm_stats.json`. The
   published `pi05_base` has none (DROID statistics normally ship inside the
   fine-tuned checkpoints), and `create_trained_policy` stops with
   `Norm stats file not found`. `prepare_checkpoint` builds a directory that links
   the real parameters and writes neutral statistics (mean 0, std 1, q01 -1, q99 1),
   which is what a one-frame forward pass needs: the check is that the weights load
   and the action block is finite, not that its scale matches a task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

OPENPI_SRC = Path(
    os.environ.get(
        "OPENPI_SRC",
        Path.home() / "policy-integration-20261019" / "third_party" / "openpi" / "src",
    )
)
BASE_CHECKPOINT = Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_base"
ASSET_ID = "droid"
STATE_DIM, GRIPPER_DIM, IMAGE_SIDE = 7, 1, 224


def alias_lerobot() -> None:
    """Make `lerobot.common.datasets.lerobot_dataset` importable on lerobot 0.4."""
    try:
        import lerobot.common.datasets.lerobot_dataset  # noqa: F401
    except ModuleNotFoundError:
        import lerobot.datasets.lerobot_dataset as real

        for name in ("lerobot.common", "lerobot.common.datasets"):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        alias = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
        for attribute in dir(real):
            if not attribute.startswith("_"):
                setattr(alias, attribute, getattr(real, attribute))
        sys.modules["lerobot.common.datasets.lerobot_dataset"] = alias


def prepare_checkpoint(source: Path, target: Path, *, asset_id: str = ASSET_ID) -> Path:
    """Link `source`'s parameters into `target` and give it neutral norm statistics."""
    target.mkdir(parents=True, exist_ok=True)
    params = target / "params"
    if params.is_symlink() or params.exists():
        if params.is_symlink() and params.resolve() == (source / "params").resolve():
            pass
        elif not params.exists():
            params.symlink_to(source / "params")
    else:
        params.symlink_to(source / "params")
    for name in ("assets", "config.json"):
        if not (target / name).exists() and (source / name).exists():
            (target / name).symlink_to(source / name)
    stats_path = target / "assets" / asset_id / "norm_stats.json"
    if not stats_path.exists():
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        def neutral(dim: int) -> dict:
            return {
                "mean": [0.0] * dim,
                "std": [1.0] * dim,
                "q01": [-1.0] * dim,
                "q99": [1.0] * dim,
            }

        action_dim = 32
        stats_path.write_text(
            json.dumps({"norm_stats": {"state": neutral(STATE_DIM + GRIPPER_DIM),
                                       "actions": neutral(action_dim)}})
        )
    return target


def observation(seed: int = 0) -> dict:
    """One valid DROID-shaped observation (keys from openpi/policies/droid_policy.py)."""
    image = (np.random.default_rng(seed).random((IMAGE_SIDE, IMAGE_SIDE, 3)) * 255).astype(np.uint8)
    return {
        "observation/joint_position": np.zeros(STATE_DIM, dtype=np.float32),
        "observation/gripper_position": np.zeros(GRIPPER_DIM, dtype=np.float32),
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": image,
        "prompt": "close the drawer",
    }


def load_policy(config_name: str = "pi05_droid", checkpoint: Path | str = BASE_CHECKPOINT,
                workdir: Path | str = "/tmp/pi05_runtime"):
    """Return an openpi policy ready for `.infer`."""
    sys.path.insert(0, str(OPENPI_SRC))
    alias_lerobot()
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    prepared = prepare_checkpoint(Path(checkpoint), Path(workdir))
    return policy_config.create_trained_policy(training_config.get_config(config_name), str(prepared))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(BASE_CHECKPOINT))
    parser.add_argument("--config", default="pi05_droid")
    parser.add_argument("--workdir", default="/tmp/pi05_runtime")
    args = parser.parse_args(argv)

    policy = load_policy(args.config, args.checkpoint, args.workdir)
    print(f"weights loaded: {args.config} from {args.checkpoint}", flush=True)
    actions = np.asarray(policy.infer(observation())["actions"])
    finite = bool(np.isfinite(actions).all())
    print(f"actions: shape={actions.shape} dtype={actions.dtype} finite={finite} "
          f"range=[{actions.min():.3f}, {actions.max():.3f}]")
    print("pi05 forward ok" if finite else "pi05 forward produced non-finite actions")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
