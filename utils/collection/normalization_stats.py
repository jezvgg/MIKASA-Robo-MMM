"""Keep complete episode statistics, including normalization quantiles (G3/G4)."""

from pathlib import Path

import numpy as np

from .contract import read_json

QUANTILES = ("q01", "q10", "q50", "q90", "q99")


def read_episode_metadata(root):
    """Read original parquet columns; LeRobot's runtime view omits all stats/."""
    import pandas as pd

    paths = sorted((Path(root) / "meta/episodes").rglob("*.parquet"))
    if not paths:
        raise ValueError("Missing episode metadata parquet")
    return pd.concat(
        [pd.read_parquet(path) for path in paths], ignore_index=True
    ).sort_values("episode_index", ignore_index=True)


def verify_normalization(root):
    """Fail before expensive video reads if usable G3/G4 statistics are absent."""
    root = Path(root)
    info, stats = read_json(root / "meta/info.json"), read_json(
        root / "meta/stats.json"
    )
    episodes = read_episode_metadata(root)
    if len(episodes) != info["total_episodes"] or not len(episodes):
        raise ValueError("G4: episode statistics count disagrees with info.json")
    for feature in ("action", "observation.state"):
        shape = tuple(info["features"][feature]["shape"])
        values = stats.get(feature, {})
        for name in (*QUANTILES, "mean", "std", "min", "max"):
            array = np.asarray(values.get(name), dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"G3: missing or malformed {feature}/{name}")
        count = np.asarray(values.get("count"), dtype=np.float64)
        if count.size != 1 or not np.isfinite(count).all() or count.item() <= 0:
            raise ValueError(f"G3: invalid count for {feature}")
        for name in QUANTILES:
            column = f"stats/{feature}/{name}"
            if column not in episodes:
                raise ValueError(f"G4: missing {column}")
            for value in episodes[column]:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"G4: missing or malformed values in {column}")
    return dict(episodes=len(episodes), global_features=2, episode_quantile_columns=10)
