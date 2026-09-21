"""Seeded episode metadata and action-only trajectory replay.

This is the project-owned equivalent of RoboMME's episode builder.  It does not
replace ManiSkill's ``RoboCasaSceneBuilder``: it recreates one environment from
an immutable episode specification and replays actions without applying saved
simulator states.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np

import my_scenes  # noqa: F401  (registers project environments)
from utils.mikasa.seeding import seed_everything


SCHEMA_VERSION = 1


def _jsonable(value: Any):
    """Convert common simulator/config values to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything needed to recreate one benchmark episode."""

    episode_id: int
    env_id: str
    env_kwargs: dict[str, Any]
    reset_kwargs: dict[str, Any]
    seed: int
    planner: str
    control_mode: str | None = None
    schema_version: int = SCHEMA_VERSION
    source_verdict: str | None = None
    trajectory_id: str | None = None

    def __post_init__(self):
        reset_seed = self.reset_kwargs.get("seed")
        if reset_seed is not None and int(reset_seed) != int(self.seed):
            raise ValueError(
                f"EpisodeSpec seed={self.seed} disagrees with reset_kwargs seed={reset_seed}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe representation."""
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeSpec":
        """Load a spec and reject missing identity fields early."""
        required = ("episode_id", "env_id", "env_kwargs", "reset_kwargs", "seed", "planner")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"EpisodeSpec missing required fields: {', '.join(missing)}")
        return cls(
            episode_id=int(value["episode_id"]),
            env_id=str(value["env_id"]),
            env_kwargs=copy.deepcopy(value["env_kwargs"]),
            reset_kwargs=copy.deepcopy(value["reset_kwargs"]),
            seed=int(value["seed"]),
            planner=str(value["planner"]),
            control_mode=value.get("control_mode"),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
            source_verdict=value.get("source_verdict"),
            trajectory_id=value.get("trajectory_id"),
        )

    def with_updates(self, **updates: Any) -> "EpisodeSpec":
        data = self.to_dict()
        data.update(updates)
        return type(self).from_dict(data)


@dataclass(frozen=True)
class ReplayResult:
    """Normalized result of action replay."""

    verdict: str
    steps: int
    success: bool
    terminated: bool
    truncated: bool
    trajectory_id: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _scalar(value: Any):
    if isinstance(value, (list, tuple, np.ndarray)) and np.size(value) == 1:
        value = np.asarray(value).reshape(-1)[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    return value


def _bool(value: Any) -> bool:
    return bool(_scalar(value))


def verdict_from_transition(info: dict, terminated: Any, truncated: Any) -> str:
    """Classify final replay transition without importing the planner runner."""
    if "success" in info and _bool(info["success"]):
        return "success"
    if _bool(truncated):
        return "truncated"
    return "missed"


class SeededEpisodeBuilder:
    """Create fresh environments and replay actions on their seeded initial state."""

    def __init__(self, spec: EpisodeSpec):
        self.spec = spec

    def make_env(self):
        return gym.make(self.spec.env_id, **copy.deepcopy(self.spec.env_kwargs))

    def reset_env(self, env):
        reset_kwargs = copy.deepcopy(self.spec.reset_kwargs)
        if reset_kwargs.get("seed") is None:
            reset_kwargs["seed"] = self.spec.seed
        seed_everything(self.spec.seed)
        return env.reset(**reset_kwargs)

    def replay(self, trajectory_h5: str | Path, trajectory_id: str | None = None) -> ReplayResult:
        """Replay actions from one HDF5 group; never call ``set_state_dict``."""
        trajectory_id = trajectory_id or self.spec.trajectory_id or "traj_0"
        h5_path = Path(trajectory_h5)
        with h5py.File(h5_path, "r") as source:
            if trajectory_id not in source:
                raise KeyError(f"{trajectory_id!r} not found in {h5_path}")
            actions = source[trajectory_id]["actions"][:]

        env = self.make_env()
        try:
            self.reset_env(env)
            last_info: dict[str, Any] = {}
            last_terminated = False
            last_truncated = False
            for action in actions:
                _, _, last_terminated, last_truncated, last_info = env.step(action)
            verdict = verdict_from_transition(
                last_info, last_terminated, last_truncated
            )
            return ReplayResult(
                verdict=verdict,
                steps=len(actions),
                success=verdict == "success",
                terminated=_bool(last_terminated),
                truncated=_bool(last_truncated),
                trajectory_id=trajectory_id,
            )
        except Exception as exc:
            return ReplayResult(
                verdict="errored",
                steps=len(actions),
                success=False,
                terminated=False,
                truncated=False,
                trajectory_id=trajectory_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            env.close()


def load_episode_specs(path: str | Path) -> list[EpisodeSpec]:
    """Load a manifest containing either a list or ``{"episodes": [...]}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("episodes", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("episode manifest must be a list or contain an 'episodes' list")
    return [EpisodeSpec.from_dict(row) for row in rows]


def save_episode_manifest(path: str | Path, specs: list[EpisodeSpec]) -> None:
    Path(path).write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "episodes": [s.to_dict() for s in specs]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
