"""Shared helpers for physical-backend and per-episode trajectory provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PHYSICAL_SIM_BACKENDS = frozenset({"physx_cpu", "physx_cuda"})
SIM_BACKEND_ALIASES = {"cpu": "physx_cpu", "gpu": "physx_cuda"}


def resolve_sim_backend(env: Any) -> str:
    """Return the resolved physical backend (not the requested ``auto`` alias)."""
    base_env = getattr(env, "unwrapped", env)
    backend = getattr(base_env, "backend", None)
    sim_backend = getattr(backend, "sim_backend", None)
    if hasattr(sim_backend, "value"):
        sim_backend = sim_backend.value
    if sim_backend not in PHYSICAL_SIM_BACKENDS:
        raise ValueError(
            "could not resolve a physical sim_backend from "
            "env.unwrapped.backend.sim_backend; got "
            f"{sim_backend!r}"
        )
    return str(sim_backend)


def _normalize_seed(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if value is None or isinstance(value, bool):
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if seed != value:
        return None
    return seed


def _reset_seed(episode: dict[str, Any]) -> int | None:
    reset_kwargs = episode.get("reset_kwargs")
    if not isinstance(reset_kwargs, dict):
        return None
    return _normalize_seed(reset_kwargs.get("seed"))


def patch_trajectory_metadata(
    metadata_path: str | Path,
    sim_backend: str,
    *,
    episode_seed: int | None = None,
    requested_sim_backend: str | None = None,
) -> None:
    """Annotate a RecordEpisode JSON with its resolved backend and actual seeds.

    When ``episode_seed`` is supplied, it applies to the last newly flushed
    trajectory (the sweep recorder flushes once per episode). Otherwise seeds are
    recovered from each episode's recorded ``reset_kwargs``.
    """
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"trajectory metadata not found: {path}")
    if sim_backend not in PHYSICAL_SIM_BACKENDS:
        raise ValueError(f"unsupported physical sim_backend: {sim_backend!r}")
    requested_physical_backend = SIM_BACKEND_ALIASES.get(
        requested_sim_backend, requested_sim_backend
    )
    if requested_physical_backend not in (None, "auto", sim_backend):
        raise ValueError(
            f"requested sim_backend={requested_sim_backend!r} resolved to "
            f"{sim_backend!r}"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    existing_backend = data.get("sim_backend")
    if existing_backend is not None and existing_backend != sim_backend:
        raise ValueError(
            f"trajectory already records sim_backend={existing_backend!r}, "
            f"cannot change it to {sim_backend!r}: {path}"
        )
    data["sim_backend"] = sim_backend
    if requested_sim_backend is not None:
        data["requested_sim_backend"] = requested_sim_backend

    # RecordEpisode keeps constructor arguments under env_info. Make replay use
    # the resolved backend rather than a potentially machine-dependent "auto".
    env_info = data.get("env_info")
    if isinstance(env_info, dict):
        env_kwargs = env_info.setdefault("env_kwargs", {})
        if isinstance(env_kwargs, dict):
            env_kwargs["sim_backend"] = sim_backend
    env_kwargs = data.get("env_kwargs")
    if isinstance(env_kwargs, dict):
        env_kwargs["sim_backend"] = sim_backend

    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{path} has no episodes list")
    target_index = len(episodes) - 1 if episode_seed is not None else None
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            continue
        reset_seed = _reset_seed(episode)
        if index == target_index and episode_seed is not None:
            seed = _normalize_seed(episode_seed)
            if seed is None:
                raise ValueError(f"{path} received an invalid expected episode_seed")
            if reset_seed is not None and reset_seed != seed:
                raise ValueError(
                    f"{path} last reset seed {reset_seed} does not match "
                    f"expected episode_seed {seed}"
                )
            reset_kwargs = episode.setdefault("reset_kwargs", {})
            if isinstance(reset_kwargs, dict):
                reset_kwargs["seed"] = seed
        else:
            seed = reset_seed
            if seed is None:
                seed = _normalize_seed(episode.get("source_seed"))
        if seed is None:
            raise ValueError(f"{path} episode {index} has no verifiable reset seed")
        episode["episode_seed"] = seed
        episode["source_seed"] = seed
        episode["sim_backend"] = sim_backend

    validate_trajectory_metadata(data, path)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_trajectory_metadata(metadata: dict[str, Any], source: str | Path) -> str:
    """Require backend provenance and one unique, recorded seed per trajectory."""
    backend = metadata.get("sim_backend")
    if backend not in PHYSICAL_SIM_BACKENDS:
        raise ValueError(
            f"{source} must record resolved sim_backend as one of "
            f"{sorted(PHYSICAL_SIM_BACKENDS)}, got {backend!r}"
        )
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{source} must contain a non-empty episodes list")

    seen: set[int] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"{source} episode {index} metadata is not an object")
        seed = _normalize_seed(episode.get("episode_seed"))
        if seed is None:
            raise ValueError(f"{source} episode {index} has no valid episode_seed")
        if seed in seen:
            raise ValueError(f"{source} has duplicate episode_seed {seed}")
        seen.add(seed)

        reset_seed = _reset_seed(episode)
        source_seed = _normalize_seed(episode.get("source_seed"))
        if reset_seed is not None and reset_seed != seed:
            raise ValueError(
                f"{source} episode {index} episode_seed={seed} does not match "
                f"reset_kwargs.seed={reset_seed}"
            )
        if source_seed is not None and source_seed != seed:
            raise ValueError(
                f"{source} episode {index} episode_seed={seed} does not match "
                f"source_seed={source_seed}"
            )
        if reset_seed is None and source_seed is None:
            raise ValueError(
                f"{source} episode {index} has no recorded reset_kwargs.seed "
                "or source_seed to verify episode_seed"
            )
        episode_backend = episode.get("sim_backend", backend)
        if episode_backend != backend:
            raise ValueError(
                f"{source} episode {index} sim_backend={episode_backend!r} "
                f"does not match run sim_backend={backend!r}"
            )
    return backend


def validate_rlds_metadata(metadata: dict[str, Any], source: str | Path) -> str:
    """Validate the required per-episode and parallel-array G5 fields."""
    backend = validate_trajectory_metadata(metadata, source)
    episodes = metadata["episodes"]
    required_episode_fields = (
        "episode_seed",
        "episode_length",
        "duration_s",
        "success",
        "success_once",
        "reward_sum",
        "terminated",
        "truncated",
        "sim_backend",
    )
    for index, episode in enumerate(episodes):
        missing = [key for key in required_episode_fields if episode.get(key) is None]
        if missing:
            raise ValueError(
                f"{source} episode {index} is missing required G5 fields: {missing}"
            )

    expected_arrays = {
        "episode_seeds": [episode["episode_seed"] for episode in episodes],
        "episode_lengths": [episode["episode_length"] for episode in episodes],
        "episode_durations_s": [episode["duration_s"] for episode in episodes],
        "success_once": [episode["success_once"] for episode in episodes],
        "reward_sums": [episode["reward_sum"] for episode in episodes],
    }
    for key, expected in expected_arrays.items():
        values = metadata.get(key)
        if not isinstance(values, list) or len(values) != len(episodes):
            raise ValueError(
                f"{source} must contain {key} with one value per episode"
            )
        if values != expected:
            raise ValueError(
                f"{source} {key} does not match its per-episode metadata"
            )
    if any(episode["sim_backend"] != backend for episode in episodes):
        raise ValueError(f"{source} episode backends do not match {backend}")
    return backend
