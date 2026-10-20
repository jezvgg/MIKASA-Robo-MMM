"""Load pi05_base into the 4-image config and run inference on real dataset frames (openpi venv).

    $OPENPI_DIR/.venv/bin/python vla/pi05_first_frame/tests/check_base_inference.py [config]

Builds a throwaway checkpoint directory ($PI05_WORK/checkpoints/pi05_base_as_samedrawer/0)
that links the base params and this dataset's norm stats, then serves it through openpi's
own create_trained_policy, exactly as `python -m mikasa_pi05 serve` does. The actions are
meaningless (no fine-tuning); this checks weight loading, the 4-image prefix and memory.
"""

from __future__ import annotations

import pathlib
import sys
import time

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.shared import download

from mikasa_pi05.configs import BASE_WEIGHTS, NORM_STATS_DIR, WORK, get_config


def main(name: str = "pi05_sd_ff_4xh100") -> None:
    config = get_config(name)
    data_config = config.data.create(config.assets_dirs, config.model)
    params = download.maybe_download(BASE_WEIGHTS)
    ckpt = WORK / "checkpoints" / "pi05_base_as_samedrawer" / "0"
    (ckpt / "assets").mkdir(parents=True, exist_ok=True)
    for link, target in ((ckpt / "params", params), (ckpt / "assets" / data_config.asset_id.split("/")[0],
                                                     NORM_STATS_DIR / data_config.asset_id.split("/")[0])):
        if not link.exists():
            link.symlink_to(target)
    start = time.time()
    policy = policy_config.create_trained_policy(config, ckpt)
    print(f"loaded in {time.time() - start:.0f} s; metadata keys {sorted(policy.metadata)}")

    dataset = data_config.dataset_factory(data_config, config.model.action_horizon, config.model)
    episode = dataset.episodes[0]
    start_row, _ = dataset.episode_range[episode]
    for i, frame in enumerate((0, 200)):
        sample = dataset[start_row + frame]
        sample.pop("actions")
        start = time.time()
        out = policy.infer(sample)
        took = time.time() - start
        actions = np.asarray(out["actions"])
        print(f"frame {frame}: actions {actions.shape}, finite {np.isfinite(actions).all()}, "
              f"infer {took:.2f} s{' (includes compile)' if i == 0 else ''}")
    stats = jax.devices()[0].memory_stats() or {}
    print(f"peak GPU memory {stats.get('peak_bytes_in_use', 0) / 2**30:.1f} GiB "
          f"of {stats.get('bytes_limit', 0) / 2**30:.1f} GiB")


if __name__ == "__main__":
    main(*sys.argv[1:])
