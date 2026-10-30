"""Check data, compute normalization stats, or fine-tune OpenPI Pi0.5."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import numpy as np

from utils.openpi_compat import install_lerobot_v3_compat
from utils.openpi_fetch import REPO_ROOT, make_train_config

OPENPI_ROOT = REPO_ROOT / "third_party/openpi"


def load_openpi_script(name: str):
    return runpy.run_path(str(OPENPI_ROOT / "scripts" / f"{name}.py"))


def check_data(config) -> None:
    api = load_openpi_script("compute_norm_stats")
    data_config = config.data.create(config.assets_dirs, config.model)
    loader, _ = api["create_torch_dataloader"](
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        num_workers=0,
        max_frames=config.batch_size * 2,
    )
    batch = next(iter(loader))
    state = np.asarray(batch["state"])
    actions = np.asarray(batch["actions"])
    images = batch["image"]
    expected_images = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    if state.shape[-1] != 12:
        raise ValueError(f"expected 12D state, got {state.shape}")
    if actions.shape[-2:] != (config.model.action_horizon, 13):
        raise ValueError(
            f"expected action chunk (*, {config.model.action_horizon}, 13), got {actions.shape}"
        )
    if set(images) != expected_images:
        raise ValueError(
            f"expected images {sorted(expected_images)}, got {sorted(images)}"
        )
    for key, image in images.items():
        shape = np.asarray(image).shape
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"expected batched HWC RGB images for {key}, got {shape}")
    image_shapes = {key: np.asarray(value).shape for key, value in images.items()}
    print(
        f"data OK: state={state.shape}, actions={actions.shape}, images={image_shapes}"
    )


def compute_norm_stats(config, max_frames: int | None) -> Path:
    api = load_openpi_script("compute_norm_stats")
    data_config = config.data.create(config.assets_dirs, config.model)
    loader, num_batches = api["create_torch_dataloader"](
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        config.num_workers,
        max_frames,
    )
    if num_batches < 1:
        raise ValueError("dataset is smaller than one batch; reduce --batch-size")

    from openpi.shared import normalize

    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in loader:
        for key, stats in running.items():
            stats.update(np.asarray(batch[key]))
    stats = {key: value.get_statistics() for key, value in running.items()}
    output = config.assets_dirs / data_config.repo_id
    normalize.save(output, stats)
    print(f"saved normalization stats: {output / 'norm_stats.json'}")
    return output / "norm_stats.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("check", "stats", "train", "all"), required=True
    )
    parser.add_argument(
        "--repo-id", required=True, help="Hugging Face LeRobot dataset repo_id"
    )
    parser.add_argument("--exp-name", default="fetch")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument(
        "--max-frames", type=int, help="optional cap for stats smoke checks"
    )
    parser.add_argument(
        "--lora", action="store_true", help="use OpenPI's LoRA model variants"
    )
    args = parser.parse_args()
    for name in ("steps", "batch_size", "action_horizon"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_frames is not None and args.max_frames < args.batch_size:
        parser.error("--max-frames must be at least --batch-size")
    return args


def main() -> None:
    args = parse_args()
    install_lerobot_v3_compat()
    config = make_train_config(
        args.repo_id,
        args.exp_name,
        action_horizon=args.action_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.steps,
        lora=args.lora,
    )

    if args.mode in ("check", "all"):
        check_data(config)
    if args.mode in ("stats", "all"):
        compute_norm_stats(config, args.max_frames)
    if args.mode in ("train", "all"):
        stats_path = config.assets_dirs / args.repo_id / "norm_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"missing {stats_path}; run with --mode stats first"
            )
        train_main = load_openpi_script("train")["main"]
        train_main(config)


if __name__ == "__main__":
    main()
