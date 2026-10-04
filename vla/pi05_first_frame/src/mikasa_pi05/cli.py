"""Command line for the SameDrawer first-frame baseline (runs in the openpi venv).

    python -m mikasa_pi05 norm-stats <config>
    python -m mikasa_pi05 train <config> --exp-name NAME [--full-checkpoints | --params-only-checkpoints]
                                [openpi TrainConfig overrides]
    python -m mikasa_pi05 serve <config> --checkpoint DIR [--port 8000]

`train` hands the config to openpi's own scripts/train.py; `serve` is openpi's
serve_policy with this package's configs.

Params-only checkpoints (default for the dummy, overfit and 1x H100 configs) keep the
inference weights in bfloat16 plus the norm stats: ~6.6 GB instead of ~37 GB for the full
model with its optimizer state. They serve and evaluate like full ones but cannot --resume.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import logging
import os
import pathlib
import sys
import time

import numpy as np


def _openpi_script(name: str):
    import openpi

    path = pathlib.Path(openpi.__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"openpi_scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_stats(argv: list[str]) -> None:
    from openpi.shared import normalize

    from mikasa_pi05.configs import get_config

    parser = argparse.ArgumentParser(prog="mikasa_pi05 norm-stats")
    parser.add_argument("config")
    args = parser.parse_args(argv)
    config = get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)
    # Always over the whole dataset, even for configs that train on a subset of episodes.
    dataset = data_config.dataset_factory(
        data_config, config.model.action_horizon, config.model, load_images=False, episodes=None)
    steps = [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
    states, actions = [], []
    start = time.time()
    for index in range(len(dataset)):
        item = dataset[index]
        for step in steps:
            item = step(item)
        states.append(item["state"])
        actions.append(item["actions"])
    values = {"state": np.stack(states), "actions": np.concatenate(actions)}
    stats = {}
    for key, value in values.items():
        value = value.reshape(-1, value.shape[-1]).astype(np.float64)
        stats[key] = normalize.NormStats(
            mean=value.mean(0),
            std=value.std(0),
            q01=np.quantile(value, 0.01, axis=0),
            q99=np.quantile(value, 0.99, axis=0),
        )
    out = pathlib.Path(config.data.assets.assets_dir or config.assets_dirs) / data_config.asset_id
    normalize.save(out, stats)
    print(f"{len(dataset)} frames in {time.time() - start:.0f} s -> {out}/norm_stats.json")
    for key, value in stats.items():
        narrow = np.flatnonzero(value.q99 - value.q01 < 1e-6)
        print(f"  {key}: dims {value.mean.shape[0]}, q01==q99 at {narrow.tolist()}")


def _wandb_logged_in() -> bool:
    netrc = pathlib.Path.home() / ".netrc"
    return bool(os.environ.get("WANDB_API_KEY")) or (netrc.exists() and "api.wandb.ai" in netrc.read_text())


def _save_params_only() -> None:
    """Make openpi's checkpoints hold only bf16 inference params and the norm stats."""
    import jax
    import ml_dtypes
    from openpi.shared import normalize as _normalize
    from openpi.training import checkpoints

    def to_host_bf16(x):
        array = np.asarray(jax.device_get(x))  # on the host: no extra GPU memory at save time
        return array.astype(ml_dtypes.bfloat16) if np.issubdtype(array.dtype, np.floating) else array

    def save_state(checkpoint_manager, state, data_loader, step):
        def save_assets(directory):
            data_config = data_loader.data_config()
            if data_config.norm_stats is not None and data_config.asset_id is not None:
                _normalize.save(directory / data_config.asset_id, data_config.norm_stats)

        params = state.ema_params if state.ema_params is not None else state.params
        checkpoint_manager.save(step, {"assets": save_assets, "params": {"params": jax.tree.map(to_host_bf16, params)}})

    checkpoints.save_state = save_state


def train(argv: list[str]) -> None:
    import tyro

    from mikasa_pi05.configs import CONFIGS, PARAMS_ONLY_CHECKPOINTS

    argv = list(argv)
    params_only = None
    for flag, value in (("--params-only-checkpoints", True), ("--full-checkpoints", False)):
        if flag in argv:
            argv.remove(flag)
            params_only = value
    config = tyro.extras.overridable_config_cli({k: (k, v) for k, v in CONFIGS.items()}, args=argv)
    if params_only is None:
        params_only = config.name in PARAMS_ONLY_CHECKPOINTS
    if params_only:
        if config.resume:
            raise SystemExit("--resume needs full checkpoints (--full-checkpoints)")
        _save_params_only()
    if _wandb_logged_in() and not any("wandb-enabled" in a for a in argv):
        config = dataclasses.replace(config, wandb_enabled=True)
    logging.info("checkpoints: %s; wandb: %s", "params only (bf16)" if params_only else "full",
                 "on" if config.wandb_enabled else "off")
    _openpi_script("train").main(config)


def serve(argv: list[str]) -> None:
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    from mikasa_pi05.configs import get_config

    parser = argparse.ArgumentParser(prog="mikasa_pi05 serve")
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True, help="checkpoints/<config>/<exp>/<step>")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    config = get_config(args.config)
    policy = policy_config.create_trained_policy(config, args.checkpoint)
    logging.info("Serving %s from %s on port %d", args.config, args.checkpoint, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    )
    server.serve_forever()


COMMANDS = {"norm-stats": norm_stats, "train": train, "serve": serve}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python -m mikasa_pi05 {{{','.join(COMMANDS)}}} ...")
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
