"""Command line for the SameDrawer first-frame baseline (runs in the openpi venv).

    python -m mikasa_pi05 norm-stats <config>
    python -m mikasa_pi05 train <config> --exp-name NAME [--full-checkpoints | --params-only-checkpoints]
                                [--no-snapshots] [openpi TrainConfig overrides]
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


SNAPSHOT_KEEP = 3  # newest progress snapshots kept on disk (~6.6 GB each)


def _install_save_state(config, *, params_only: bool, snapshots: bool, full_every: int | None) -> None:
    """Decide what openpi's training loop writes at each save step (every config.save_interval).

    params_only  the checkpoint holds bf16 inference params and the norm stats (~6.6 GB instead of
                 ~45 GB with optimizer state); serves and evaluates the same, cannot --resume.
    snapshots    also write such a params-only snapshot to $PI05_WORK/snapshots/<config>/<exp>/<step>
                 at every save step, for the progress SR (scripts/sr_monitor.sh). The newest
                 SNAPSHOT_KEEP stay on disk. With snapshots, openpi's own checkpoint is written only
                 every `full_every` steps and at the last step (params-only mode: last step only).
    """
    import shutil

    import jax
    import ml_dtypes
    import orbax.checkpoint as ocp
    from openpi.shared import normalize as _normalize
    from openpi.training import checkpoints

    from mikasa_pi05.configs import WORK

    original = checkpoints.save_state
    last_step = config.num_train_steps - 1
    snapshot_root = WORK / "snapshots" / config.name / config.exp_name

    def to_host_bf16(x):
        array = np.asarray(jax.device_get(x))  # on the host: no extra GPU memory at save time
        return array.astype(ml_dtypes.bfloat16) if np.issubdtype(array.dtype, np.floating) else array

    def inference_params(state):
        params = state.ema_params if state.ema_params is not None else state.params
        return jax.tree.map(to_host_bf16, params)

    def save_norm_stats(directory, data_loader):
        data_config = data_loader.data_config()
        if data_config.norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(pathlib.Path(directory) / data_config.asset_id, data_config.norm_stats)

    def write_snapshot(state, data_loader, step):
        snapshot_root.mkdir(parents=True, exist_ok=True)
        partial = snapshot_root / f".{step}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir()
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(partial / "params", {"params": inference_params(state)})
        save_norm_stats(partial / "assets", data_loader)
        final = snapshot_root / str(step)
        shutil.rmtree(final, ignore_errors=True)
        partial.rename(final)  # the monitor only ever sees complete snapshots
        steps = sorted(int(d.name) for d in snapshot_root.iterdir() if d.name.isdigit())
        for old in steps[:-SNAPSHOT_KEEP]:
            shutil.rmtree(snapshot_root / str(old), ignore_errors=True)
        logging.info("snapshot %s", final)

    def save_state(checkpoint_manager, state, data_loader, step):
        last = step == last_step
        if snapshots:
            write_snapshot(state, data_loader, step)
        if params_only:
            if snapshots and not last:
                return
            checkpoint_manager.save(step, {
                "assets": lambda directory: save_norm_stats(directory, data_loader),
                "params": {"params": inference_params(state)},
            })
            return
        if snapshots and full_every and step % full_every and not last:
            return
        original(checkpoint_manager, state, data_loader, step)

    checkpoints.save_state = save_state


def train(argv: list[str]) -> None:
    import tyro

    from mikasa_pi05.configs import CONFIGS, FULL_CHECKPOINT_EVERY, PARAMS_ONLY_CHECKPOINTS

    argv = list(argv)
    params_only = None
    for flag, value in (("--params-only-checkpoints", True), ("--full-checkpoints", False)):
        if flag in argv:
            argv.remove(flag)
            params_only = value
    snapshots = "--no-snapshots" not in argv
    if not snapshots:
        argv.remove("--no-snapshots")
    config = tyro.extras.overridable_config_cli({k: (k, v) for k, v in CONFIGS.items()}, args=argv)
    if params_only is None:
        params_only = config.name in PARAMS_ONLY_CHECKPOINTS
    if params_only and config.resume:
        raise SystemExit("--resume needs full checkpoints (--full-checkpoints)")
    full_every = FULL_CHECKPOINT_EVERY.get(config.name)
    snapshots = snapshots and full_every is not None
    _install_save_state(config, params_only=params_only, snapshots=snapshots, full_every=full_every)
    if _wandb_logged_in() and not any("wandb-enabled" in a for a in argv):
        config = dataclasses.replace(config, wandb_enabled=True)
    logging.info(
        "checkpoints: %s; progress snapshots: %s; wandb: %s",
        "params only (bf16)" if params_only else f"full every {full_every or config.save_interval} steps",
        f"every {config.save_interval} steps" if snapshots else "off",
        "on" if config.wandb_enabled else "off",
    )
    _openpi_script("train").main(config)


def _warm_up(policy) -> None:
    """Compile before accepting connections.

    The server runs inference inside its event loop; a first request that compiles for
    20-40 s leaves the client's keepalive pings unanswered and the client drops the
    connection. A request of the real shapes compiles once here instead.
    """
    meta = policy.metadata
    cameras = meta["mikasa_data"]["cameras"]
    observation = {f"observation.images.{name}": np.zeros(shape, np.uint8) for name, shape in cameras.items()}
    observation[meta["first_frame"]["key"]] = np.zeros(cameras[meta["first_frame"]["camera"]], np.uint8)
    observation["observation.state"] = np.zeros(meta["mikasa_data"]["state_dim"], np.float32)
    observation["prompt"] = "warm-up"
    start = time.time()
    policy.infer(observation)
    logging.info("Compiled the policy in %.0f s", time.time() - start)


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
    _warm_up(policy)
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
