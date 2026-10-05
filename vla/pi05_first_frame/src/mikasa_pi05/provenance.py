"""What a training run records next to its checkpoints, for reproduction and the paper.

<checkpoint dir>/run_meta.json         written once the checkpoint directory exists: the full
                                       TrainConfig, code commits (this repository, openpi and the
                                       applied patch), dataset revision and checksums, norm stats,
                                       base weights, packages, GPUs, derived numbers (epochs, ...).
                                       A resumed run adds run_meta.resume-<time>.json.
<checkpoint dir>/metrics.jsonl         one line per logged step: step, wall time, loss, grad_norm,
                                       param_norm (the values openpi logs), with or without wandb.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import pathlib
import platform
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[4]  # MIKASA-Robo-MMM checkout
PACKAGES = ("jax", "jaxlib", "flax", "orbax-checkpoint", "optax", "numpy", "openpi", "mikasa-pi05", "av", "pyarrow")


def jsonable(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {"__type__": type(value).__name__}
        out.update({f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)})
        return out
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, pathlib.PurePath):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def sha256(path) -> str | None:
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(path, *args) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _gpus() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def run_meta(config, *, argv: list[str], checkpoints: dict) -> dict:
    import jax
    import openpi

    from mikasa_pi05 import configs

    openpi_dir = pathlib.Path(openpi.__file__).resolve().parents[2]
    data_dir = pathlib.Path(config.data.dataset_dir or configs.dataset_dir())
    info = json.loads((data_dir / "meta/info.json").read_text()) if (data_dir / "meta/info.json").is_file() else {}
    norm_stats = pathlib.Path(config.data.assets.assets_dir) / config.data.repo_id / "norm_stats.json"
    devices = jax.devices()
    frames = info.get("total_frames")
    episodes = list(config.data.episodes) if config.data.episodes else None
    patch = REPO / "vla/pi05_first_frame/openpi.patch"
    return {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, "-m", "mikasa_pi05", "train", *argv],
        "config_name": config.name,
        "exp_name": config.exp_name,
        "checkpoint_dir": str(config.checkpoint_dir),
        "checkpoints": checkpoints,
        "code": {
            "repo": str(REPO),
            "repo_commit": git(REPO, "rev-parse", "HEAD"),
            "repo_branch": git(REPO, "rev-parse", "--abbrev-ref", "HEAD"),
            "repo_dirty_files": (git(REPO, "status", "--porcelain") or "").splitlines(),
            "openpi_dir": str(openpi_dir),
            "openpi_commit": git(openpi_dir, "rev-parse", "HEAD"),
            "openpi_patch_sha256": sha256(patch),
            "openpi_working_tree_diff_sha256": hashlib.sha256((git(openpi_dir, "diff") or "").encode()).hexdigest(),
        },
        "data": {
            "dataset_dir": str(data_dir),
            "hf_repo": configs.HF_REPO_ID,
            "hf_revision": configs.HF_REVISION,
            "info_json_sha256": sha256(data_dir / "meta/info.json"),
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": frames,
            "train_episodes": episodes or "all",
            "norm_stats": str(norm_stats),
            "norm_stats_sha256": sha256(norm_stats),
            "validation_seeds_sha256": sha256(data_dir / "validation_seeds.json"),
        },
        "weights": {
            "base": configs.BASE_WEIGHTS,
            "manifest_sha256": sha256(REPO / "vla/pi05_first_frame/weights.sha256"),
        },
        "derived": {
            "devices": len(devices),
            "device_kind": devices[0].device_kind if devices else None,
            "global_batch": config.batch_size,
            "per_device_batch": config.batch_size // max(len(devices), 1),
            "samples_seen": config.batch_size * config.num_train_steps,
            "epochs": round(config.batch_size * config.num_train_steps / frames, 2) if frames and not episodes else None,
            "image_keys": list(config.model.image_keys),
            "first_frame_camera": config.data.first_frame_camera,
            "action_horizon": config.model.action_horizon,
            "max_token_len": config.model.max_token_len,
            "discrete_state_input": config.model.discrete_state_input,
            "frozen": repr(config.freeze_filter),
        },
        "train_config": jsonable(config),
        "packages": {name: _version(name) for name in PACKAGES},
        "host": {
            "node": platform.node(),
            "python": platform.python_version(),
            "gpus": _gpus(),
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION", "PI05_WORK")},
        },
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def install(config, *, argv: list[str], checkpoints: dict) -> None:
    """Write run_meta.json once openpi has created the checkpoint directory, and log metrics."""
    import wandb
    from openpi.training import checkpoints as _checkpoints

    ckpt_dir = pathlib.Path(config.checkpoint_dir)
    original_init_dir = _checkpoints.initialize_checkpoint_dir

    def initialize_checkpoint_dir(*args, **kwargs):
        result = original_init_dir(*args, **kwargs)
        meta = run_meta(config, argv=argv, checkpoints=checkpoints)
        target = ckpt_dir / "run_meta.json"
        if target.exists():  # resumed: keep the original, add this start
            target = ckpt_dir / f"run_meta.resume-{time.strftime('%Y%m%d-%H%M%S')}.json"
        target.write_text(json.dumps(meta, indent=2) + "\n")
        logging.info("run metadata: %s", target)
        return result

    _checkpoints.initialize_checkpoint_dir = initialize_checkpoint_dir

    metrics = ckpt_dir / "metrics.jsonl"
    original_init = wandb.init

    def init(*args, **kwargs):
        run = original_init(*args, **kwargs)
        run_log = wandb.log  # wandb.init rebinds the module-level log to the run

        def log(data, step=None, **log_kwargs):
            scalars = {}
            for key, value in dict(data).items():
                try:
                    scalars[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if scalars:
                with metrics.open("a") as handle:
                    handle.write(json.dumps({"step": step, "time": round(time.time(), 2), **scalars}) + "\n")
            return run_log(data, step=step, **log_kwargs)

        wandb.log = log
        return run

    wandb.init = init
