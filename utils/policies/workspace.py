"""Contain policy-job outputs and enforce a conservative workspace budget.

400 GB means decimal bytes. Existing input/model symlinks are read-only sources,
not cache destinations. A 50-GB headroom is retained for bounded model writes;
this is a process watchdog, NOT an administrative filesystem quota.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

LIMIT_BYTES = 400_000_000_000
STOP_BYTES = 350_000_000_000


def usage_bytes(root: Path) -> int:
    """Conservatively count logical/allocated bytes, without following links."""
    size = 0
    for directory, directories, files in os.walk(root, followlinks=False):
        size += Path(directory).lstat().st_blocks * 512
        for name in files + [name for name in directories if (Path(directory) / name).is_symlink()]:
            try:
                stat = (Path(directory) / name).lstat()
            except FileNotFoundError:  # concurrent atomic checkpoint finalization
                continue
            size += max(stat.st_size, stat.st_blocks * 512)
    return size


def contained(root: Path, destination: Path) -> Path:
    path = destination.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"write destination outside workspace: {destination}")
    return path


def environment(root: Path) -> dict:
    root = root.resolve()
    env = os.environ.copy()
    destinations = {
        "HOME": root / ".home", "TMPDIR": root / ".tmp",
        "TMP": root / ".tmp", "TEMP": root / ".tmp",
        "XDG_CACHE_HOME": root / ".cache", "XDG_CONFIG_HOME": root / ".config",
        "XDG_DATA_HOME": root / ".local/share", "XDG_STATE_HOME": root / ".local/state",
        "HF_HOME": root / ".cache/huggingface",
        "HF_HUB_CACHE": root / ".cache/huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": root / ".cache/huggingface/hub",
        "HF_DATASETS_CACHE": root / ".cache/huggingface/datasets",
        "TRANSFORMERS_CACHE": root / ".cache/huggingface/transformers",
        "TORCH_HOME": root / ".cache/torch", "TORCH_EXTENSIONS_DIR": root / ".cache/torch_extensions",
        "TRITON_CACHE_DIR": root / ".cache/triton", "CUDA_CACHE_PATH": root / ".cache/cuda",
        "UV_CACHE_DIR": root / ".cache/uv", "UV_PYTHON_INSTALL_DIR": root / ".local/uv-python",
        "OPENPI_DATA_HOME": root / ".cache/openpi", "JAX_COMPILATION_CACHE_DIR": root / ".cache/jax",
        "WANDB_DIR": root / "logs/wandb", "WANDB_CACHE_DIR": root / ".cache/wandb",
        "WANDB_CONFIG_DIR": root / ".config/wandb", "WANDB_DATA_DIR": root / ".cache/wandb-data",
        "PYTHONPYCACHEPREFIX": root / ".cache/pycache",
        "MPLCONFIGDIR": root / ".config/matplotlib", "NUMBA_CACHE_DIR": root / ".cache/numba",
    }
    for variable, path in destinations.items():
        contained(root, path).mkdir(parents=True, exist_ok=True)
        env[variable] = str(path)
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["POLICY_WORKSPACE"] = str(root)
    env["POLICY_WORKSPACE_LIMIT_BYTES"] = str(LIMIT_BYTES)
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    return env


def require_space(root: Path, reserve: int = 20_000_000_000) -> int:
    used = usage_bytes(root)
    if used + reserve >= STOP_BYTES:
        raise RuntimeError(f"workspace budget refused: used={used}, reserve={reserve}, stop={STOP_BYTES}")
    return used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--reserve-gb", type=float, default=30)
    parser.add_argument("--check-every", type=int, default=30,
                        help="watchdog seconds; 50-GB headroom covers bounded checkpoint writes")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    reserve = int(args.reserve_gb * 1_000_000_000)
    if not 0 <= reserve < STOP_BYTES or args.check_every < 1:
        parser.error("reserve must be nonnegative/below 350 GB; check-every must be positive")
    used = require_space(root, reserve)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print(json.dumps({"workspace": str(root), "used_bytes": used,
                          "hard_user_limit_bytes": LIMIT_BYTES,
                          "watchdog_stop_bytes": STOP_BYTES}))
        return
    env = environment(root)
    print(json.dumps({"event": "workspace_job_start", "used_bytes": used,
                      "reserve_bytes": reserve, "command": command}), flush=True)
    process = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
    def stop(signum, frame):
        if process.poll() is None:
            os.killpg(process.pid, signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while process.poll() is None:
        used = usage_bytes(root)
        if used >= STOP_BYTES:
            print(f"WORKSPACE BUDGET STOP: {used} >= {STOP_BYTES}", file=sys.stderr, flush=True)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            raise SystemExit(73)
        time.sleep(args.check_every)
    print(json.dumps({"event": "workspace_job_end", "returncode": process.returncode,
                      "used_bytes": usage_bytes(root)}), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
