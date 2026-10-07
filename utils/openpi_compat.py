"""Bridge OpenPI's LeRobot v2 import path to the v3 LeRobot package."""

from __future__ import annotations

import importlib
import sys
import types


def install_lerobot_v3_compat() -> None:
    """Expose LeRobot v3 classes under the import path used by pinned OpenPI."""
    from lerobot.datasets.lerobot_dataset import (
        CODEBASE_VERSION,
        LeRobotDataset,
        LeRobotDatasetMetadata as V3Metadata,
    )

    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(
            f"OpenPI adapter requires LeRobot v3.0, got {CODEBASE_VERSION}"
        )

    class MetadataCompat:
        def __init__(self, repo_id: str):
            metadata = V3Metadata(repo_id)
            if metadata.tasks is None:
                raise ValueError(f"LeRobot dataset {repo_id!r} has no task metadata")
            self.tasks = {
                int(row["task_index"]): str(task)
                for task, row in metadata.tasks.iterrows()
            }
            self.fps = int(metadata.fps)

    common = sys.modules.setdefault(
        "lerobot.common", types.ModuleType("lerobot.common")
    )
    common.__path__ = []
    datasets = sys.modules.setdefault(
        "lerobot.common.datasets", types.ModuleType("lerobot.common.datasets")
    )
    datasets.__path__ = []

    compat = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    compat.LeRobotDataset = LeRobotDataset
    compat.LeRobotDatasetMetadata = MetadataCompat
    sys.modules[compat.__name__] = compat
    setattr(importlib.import_module("lerobot"), "common", common)
    setattr(common, "datasets", datasets)
    setattr(datasets, "lerobot_dataset", compat)
